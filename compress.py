"""Compress a real Hy4-flash matrix into bit-packed low-rank factors. Requires NumPy."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from urllib.request import Request, urlopen

import numpy as np

REVISION = "1e3e4005da28b6ef7f53c1298b09c8e00bfc659e"
BASE = f"https://huggingface.co/tencent/Hy4-flash/resolve/{REVISION}/"
HEADER = struct.Struct("<4sIIII")  # magic, output rows, input columns, rank, per-row scales


def read_range(shard, start, size):
    end = start + size - 1
    request = Request(f"{BASE}{shard}?range={start}-{end}",
                      headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=120) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
                f"bytes {start}-{end}/"):
            raise ValueError("Server did not return the requested byte range")
        data = response.read(size + 1)
    if len(data) != size:
        raise ValueError("Incomplete shard range")
    return data


def load_matrix(name, expert, row, rows):
    with urlopen(BASE + "model.safetensors.index.json", timeout=60) as response:
        shard = json.load(response)["weight_map"][name]
    size, = struct.unpack("<Q", read_range(shard, 0, 8))
    info = json.loads(read_range(shard, 8, size))[name]
    shape = info["shape"]
    if info["dtype"] != "BF16" or len(shape) not in (2, 3):
        raise ValueError("Expected a BF16 matrix or stack of expert matrices")
    height, width = shape[-2:]
    if len(shape) == 3 and (expert is None or not 0 <= expert < shape[0]):
        raise ValueError("Specify a valid --expert for this stacked tensor")
    if len(shape) == 2 and expert is not None:
        raise ValueError("This tensor has no expert axis")
    rows = height - row if rows is None else rows
    if row < 0 or rows <= 0 or row + rows > height:
        raise ValueError("Requested rows are outside the matrix")
    if rows * width > 32_000_000:
        raise ValueError("Use --rows to select a bounded block for the in-memory SVD")
    offset = ((expert or 0) * height + row) * width * 2
    start = 8 + size + info["data_offsets"][0] + offset
    data = read_range(shard, start, rows * width * 2)
    bits = np.frombuffer(data, "<u2").astype(np.uint32) << 16
    return bits.view(np.float32).reshape(rows, width), hashlib.sha256(data).hexdigest()


def packed_size(m, n, rank, per_row):
    return (HEADER.size + (m * rank + 7) // 8 + (rank * n + 7) // 8
            + 2 * (per_row * m + rank + n))


def refine_signs(target, basis, scales, signs):
    """Minimize ||target - (basis @ signs) * scales|| one binary row at a time."""
    cross = (basis.T @ target) * scales
    gram = basis.T @ basis
    np.fill_diagonal(gram, 0)
    scale_sq = scales**2
    for k in range(len(signs)):
        score = cross[k] - (gram[k] @ signs) * scale_sq
        signs[k] = np.where(score == 0, signs[k], np.sign(score))


def fit(weights, bpw, per_row, sign_sweeps=0):
    if sign_sweeps < 0:
        raise ValueError("Sign sweep count must be nonnegative")
    m, n = weights.shape
    budget = math.floor(bpw * weights.size / 8)
    rank = max((k for k in range(1, min(m, n) + 1)
                if packed_size(m, n, k, per_row) <= budget), default=0)
    if rank == 0:
        raise ValueError("Bit budget is too small for rank one and the selected scales")
    u, s, vt = np.linalg.svd(weights, full_matrices=False)
    ceiling = np.sum(s[:rank] ** 2, dtype=np.float64) / np.sum(s ** 2, dtype=np.float64)
    left = u[:, :rank]
    right = vt[:rank]
    # Approximate singular vectors by signs; fit the diagonal scales by least squares.
    a = np.where(left >= 0, np.float32(1), np.float32(-1))
    b = np.where(right >= 0, np.float32(1), np.float32(-1))
    row = np.ones(m, np.float32)
    col = np.ones(n, np.float32)
    for step in range(5 + sign_sweeps):
        if step >= 5:
            refine_signs(weights, a * row[:, None] * latent, col, b)
            refine_signs(weights.T, b.T * col[:, None] * latent, row, a.T)
        x = row[:, None] * a
        y = b * col
        gram = (x.T @ x) * (y @ y.T)
        rhs = np.sum((weights @ y.T) * x, axis=0)
        latent = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        base = (a * latent) @ b
        if per_row:
            scaled = base * col
            row = np.sum(weights * scaled, axis=1) / np.maximum((scaled**2).sum(axis=1), 1e-30)
        scaled = base * row[:, None]
        col = np.sum(weights * scaled, axis=0) / np.maximum((scaled**2).sum(axis=0), 1e-30)
    parts = [HEADER.pack(b"HYB1", m, n, rank, per_row), np.packbits(a > 0).tobytes(),
             np.packbits(b > 0).tobytes()]
    for scale in ([row] if per_row else []) + [latent, col]:
        parts.append(scale.astype("<f2").tobytes())
    return b"".join(parts), rank, ceiling


def unpack(data):
    magic, m, n, rank, per_row = HEADER.unpack_from(data)
    if magic != b"HYB1" or len(data) != packed_size(m, n, rank, per_row):
        raise ValueError("Invalid packed matrix")
    offset = HEADER.size

    def signs(shape):
        nonlocal offset
        count = math.prod(shape)
        raw = np.frombuffer(data, "u1", (count + 7) // 8, offset)
        offset += raw.nbytes
        return (2 * np.unpackbits(raw, count=count).astype(np.float32) - 1).reshape(shape)

    a, b = signs((m, rank)), signs((rank, n))
    scales = np.frombuffer(data, "<f2", offset=offset).astype(np.float32)
    if per_row:
        a *= scales[:m, None]
        scales = scales[m:]
    return a * scales[:rank], b * scales[rank:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor")
    parser.add_argument("output", type=Path)
    parser.add_argument("--expert", type=int)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--bpw", type=float, default=0.1)
    parser.add_argument("--sign-sweeps", type=int, default=0,
                        help="Refit both binary factors after scale initialization")
    parser.add_argument("--no-row-scale", action="store_true",
                        help="Omit per-row FP16 scales for narrow embedding tables")
    args = parser.parse_args()
    weights, digest = load_matrix(args.tensor, args.expert, args.row, args.rows)
    data, rank, ceiling = fit(weights, args.bpw, not args.no_row_scale, args.sign_sweeps)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as tmp:
        tmp.write(data)
    try:
        a, b = unpack(Path(tmp.name).read_bytes())
        relative_error = np.linalg.norm(weights - a @ b) / np.linalg.norm(weights)
        if not np.isfinite(relative_error):
            raise ValueError("Non-finite reconstruction")
        os.replace(tmp.name, args.output)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    print(json.dumps(dict(revision=REVISION, tensor=args.tensor, expert=args.expert,
                          first_row=args.row, shape=weights.shape, source_sha256=digest,
                          rank=rank, bytes=len(data), bpw=8*len(data)/weights.size,
                          relative_frobenius_error=float(relative_error),
                          optimal_rank_energy_fraction=float(ceiling)), indent=2))


if __name__ == "__main__":
    main()
