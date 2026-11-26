"""Verify the byte budget and combined relative weight error for the selected expert."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compress import load_matrix, unpack
from scale_vq import decode_factors

HERE = Path(__file__).resolve().parent
CAP = 158632
TENSORS = {
    "gate-up": ("gate_up_proj", (3072, 1536), "05659c95b224b6902b9ec61305b4790ce5b8f5c723aa0a52932ba4af58ef3d21"),
    "down": ("down_proj", (1536, 1536), "f887b094ca7d153d86d2ac04da0aba1a3ce03cfbefd7bac944eda93ea97644fa"),
}


def reference(name, directory):
    suffix, shape, digest = TENSORS[name]
    if directory is None:
        w, actual = load_matrix(f"model.layers.1.mlp.experts.{suffix}", 0, 0, None)
    else:
        w = np.load(directory / f"hy4-{suffix}.npy", allow_pickle=False)
        if w.shape != shape or w.dtype != np.float32:
            raise ValueError("Invalid reference")
        bits = w.view(np.uint32)
        if np.any(bits & 65535):
            raise ValueError("Reference differs from BF16")
        actual = hashlib.sha256((bits >> 16).astype("<u2").tobytes()).hexdigest()
    if actual != digest or w.shape != shape:
        raise ValueError("Wrong source expert")
    return w.astype(np.float64)


def decode(raw):
    magic = raw[:4]
    if magic == b"HYB1":
        return unpack(raw)
    if magic == b"SVQ1":
        return decode_factors(raw)
    raise ValueError(f"Unknown packed format {magic!r}")


def evaluate(paths, refs=None):
    sizes = [p.stat().st_size for p in paths]
    if sum(sizes) > CAP:
        raise ValueError(f"Payload exceeds {CAP} bytes")
    errors, norms, parts = [], [], {}
    for (name, (_, shape, _)), path in zip(TENSORS.items(), paths, strict=True):
        raw = path.read_bytes()
        left, right = decode(raw)
        if (left.shape[0], right.shape[1]) != shape:
            raise ValueError("Wrong matrix dimensions")
        w = reference(name, refs)
        residual = w - (left @ right)
        error = float(np.sum(residual**2))
        norm = float(np.sum(w**2))
        if not np.isfinite(error):
            raise ValueError("Non-finite error")
        errors.append(error)
        norms.append(norm)
        parts[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                           format=raw[:4].decode("ascii"), file=path.name,
                           relative_error=(error/norm)**0.5)
    return dict(bytes=sum(sizes), score=(sum(errors)/sum(norms))**0.5, projections=parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--candidate", nargs=2, type=Path, metavar=("GATE_UP", "DOWN"),
                        help="Score these packed projections instead of the recorded champion")
    args = parser.parse_args()
    if args.candidate:
        print(json.dumps(evaluate(args.candidate, args.references), indent=2))
        return
    folder = HERE / "champion"
    manifest = json.loads((folder / "result.json").read_text())
    paths = [folder / manifest.get("projections", {}).get(n, {}).get("file", f"{n}.hyb")
             for n in TENSORS]
    measured = evaluate(paths, args.references)
    if abs(measured["score"] - manifest["score"]) > 1e-6 or any(
            measured["projections"][n]["sha256"] != manifest["projections"][n]["sha256"]
            for n in TENSORS):
        raise ValueError("Champion differs from its recorded result")
    print(json.dumps(measured, indent=2))


if __name__ == "__main__":
    main()
