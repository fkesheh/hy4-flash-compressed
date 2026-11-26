"""Reproduce the ITQ + learned scale-codebook SVQ1 candidate."""
import argparse
from pathlib import Path

import compress
from itq import fit_itq
from scale_vq import encode_hyb1
from verify import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection", choices=["gate-up", "down"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sweeps", type=int, required=True)
    parser.add_argument("--references", type=Path)
    args = parser.parse_args()

    weights = reference(args.projection, args.references).astype("<f4")
    budget = compress.packed_size(*weights.shape, args.rank, 1)
    hyb, rank = fit_itq(weights, budget, args.sweeps, args.seed)
    if rank != args.rank:
        raise ValueError(f"Requested rank {args.rank}, fitter selected {rank}")
    args.output.write_bytes(encode_hyb1(hyb))


if __name__ == "__main__":
    main()
