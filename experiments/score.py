"""Score decoded layer 1 expert 0 matrices, supplied as gate/up and down NPY files."""
import argparse
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).parent / "reference"


def score(gate_up, down):
    error_sq, ref_sq = 0.0, 0.0
    for name, decoded in (("gate_up_proj", gate_up), ("down_proj", down)):
        ref = np.load(REFERENCE / f"hy4-{name}.npy", allow_pickle=False).astype(np.float64)
        decoded = np.asarray(decoded)
        if ref.shape != decoded.shape or not np.isfinite(decoded).all():
            raise ValueError(f"Invalid reconstruction for {name}")
        residual = ref - decoded
        err, norm = np.vdot(residual, residual).item(), np.vdot(ref, ref).item()
        error_sq += err
        ref_sq += norm
        print(name, (err / norm) ** 0.5)
    combined = (error_sq / ref_sq) ** 0.5
    print("combined", combined)
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate_up")
    parser.add_argument("down")
    args = parser.parse_args()
    score(np.load(args.gate_up, allow_pickle=False), np.load(args.down, allow_pickle=False))
