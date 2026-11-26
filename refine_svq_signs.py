"""Refine binary factors inside SVQ1 while holding its scale codebooks fixed."""
import argparse
from pathlib import Path

import numpy as np

import compress
import verify


def unpack_scales_and_signs(data):
    magic, rows, cols, rank, per_row = compress.HEADER.unpack_from(data)
    if magic != b"SVQ1" or per_row != 1:
        raise ValueError("Expected an SVQ1 per-row-scale payload")
    offset = compress.HEADER.size
    left_n = (rows * rank + 7) // 8
    right_n = (rank * cols + 7) // 8
    left_bits = np.frombuffer(data, dtype=np.uint8, count=left_n, offset=offset)
    offset += left_n
    right_bits = np.frombuffer(data, dtype=np.uint8, count=right_n, offset=offset)
    offset += right_n
    left = np.where(np.unpackbits(left_bits, count=rows * rank).reshape(rows, rank), 1.0, -1.0)
    right = np.where(np.unpackbits(right_bits, count=rank * cols).reshape(rank, cols), 1.0, -1.0)
    scales = []
    for count in (rows, rank, cols):
        centroids = np.frombuffer(data, dtype="<f2", count=16, offset=offset).astype(np.float32)
        offset += 32
        index_bytes = (count + 1) // 2
        packed = np.frombuffer(data, dtype=np.uint8, count=index_bytes, offset=offset)
        labels = np.empty(index_bytes * 2, dtype=np.uint8)
        labels[::2] = packed >> 4
        labels[1::2] = packed & 15
        scales.append(centroids[labels[:count]])
        offset += index_bytes
    if offset != len(data):
        raise ValueError("SVQ1 payload has trailing bytes")
    return (rows, cols, rank), left.astype(np.float32), right.astype(np.float32), *scales


def refine_one(source, destination, projection, references, sweeps):
    data = Path(source).read_bytes()
    (rows, cols, rank), left, right, row, latent, col = unpack_scales_and_signs(data)
    weights = verify.reference(projection, references).astype(np.float32)
    if weights.shape != (rows, cols):
        raise ValueError(f"Unexpected reference shape for {projection}")
    for _ in range(sweeps):
        compress.refine_signs(weights, left * row[:, None] * latent, col, right)
        compress.refine_signs(weights.T, right.T * col[:, None] * latent, row, left.T)
    header = compress.HEADER.pack(b"SVQ1", rows, cols, rank, 1)
    left_bytes = np.packbits(left > 0).tobytes()
    right_bytes = np.packbits(right > 0).tobytes()
    scales_offset = (compress.HEADER.size + (rows * rank + 7) // 8
                     + (rank * cols + 7) // 8)
    encoded = header + left_bytes + right_bytes + data[scales_offset:]
    Path(destination).write_bytes(encoded)
    return len(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate_up_in", type=Path)
    parser.add_argument("down_in", type=Path)
    parser.add_argument("gate_up_out", type=Path)
    parser.add_argument("down_out", type=Path)
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--references", type=Path)
    args = parser.parse_args()
    if args.sweeps < 0:
        parser.error("--sweeps must be nonnegative")
    sizes = [
        refine_one(args.gate_up_in, args.gate_up_out, "gate-up", args.references, args.sweeps),
        refine_one(args.down_in, args.down_out, "down", args.references, args.sweeps),
    ]
    result = verify.evaluate([args.gate_up_out, args.down_out], args.references)
    print({"input_bytes": sum(Path(p).stat().st_size for p in (args.gate_up_in, args.down_in)),
           "output_bytes": sum(sizes), **result})


if __name__ == "__main__":
    main()
