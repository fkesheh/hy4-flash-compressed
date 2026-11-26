#!/usr/bin/env python3
"""Create a deterministic synthetic NPZ with the teacher-QAT fit/tune/test schema."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import distill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=901)
    args = parser.parse_args()
    x, y, xtest, ytest, w, wtest, _ = distill.synthetic_data(args.seed)
    arrays = {"x_train": x[:64], "y_train": y[:64], "route_weight_train": w[:64],
              "x_tune": x[64:], "y_tune": y[64:], "route_weight_tune": w[64:],
              "x_val": xtest, "y_val": ytest, "route_weight_val": wtest}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = {"kind": "synthetic-explicit-document-fixture", "split_unit": "document",
        "dataset": str(args.output), "dataset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "training_documents": ["fit-a", "fit-b"], "tuning_documents": ["tune-a"],
        "heldout_documents": ["test-a", "test-b"],
        "training_document_rows": [{"doc_id": "fit-a", "routed_rows": 32}, {"doc_id": "fit-b", "routed_rows": 32}],
        "tuning_document_rows": [{"doc_id": "tune-a", "routed_rows": 32}],
        "heldout_document_rows": [{"doc_id": "test-a", "routed_rows": 24}, {"doc_id": "test-b", "routed_rows": 24}],
        "training_routed_rows": len(arrays["x_train"]), "tuning_routed_rows": len(arrays["x_tune"]),
        "heldout_routed_rows": len(arrays["x_val"])}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
