# Narrow layer-1 expert-0 student distillation

This experiment trains a narrower SwiGLU student while preserving the routed
expert's 1,536-wide input and output. Teacher targets are standalone expert-0
outputs before router weighting. When available, router weights are included in
the fit objective and weighted error is reported separately. Results belong to
the heldout behavioral scoreboard; raw weight reconstruction remains separate.

The student is packed as two HYB1 matrices and evaluated after reading the
serialized factors back. `student.json` is part of the model-specific byte
budget; the pair of HYB1 files plus this manifest must not exceed 158,632 bytes.
Evaluation result JSON is a separate experiment report, not model data.
Behavioral results are kept in [behavioral-scoreboard.json](behavioral-scoreboard.json),
separate from raw-weight scores.

## Single configuration

Use the shared aggregate produced by `experiments/teacher-qat/prepare_routed_data.py`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/narrow-expert-distill/distill.py \
  experiments/teacher-qat/routed_data.npz \
  --hidden 384 --epochs 10 --batch-size 32 --lr 1e-3 \
  --out experiments/narrow-expert-distill/real-width384
```

The run resolves the current published/champion format through its result
manifest, including SVQ1 scale-codebook factors, and scores it on the same
validation arrays when present. For an official final test comparison, use the
same-shape teacher-QAT pair is included when available. The search command
below splits calibration documents into fit and tuning
validation groups, then opens the capture's heldout documents only after
selecting both policy winners.

## Random forest search vs uniform random search

Both policies reuse six shared random initial trials, then receive four
additional proposals each. Training examples are scaled inversely with hidden
width so `examples × hidden width` is held near a common budget;
the configured epoch count is the reference at width 256. Candidate features
are SwiGLU intermediate width, HYB1 row-scale versus
global-scale quantization, learning rate and sign-refinement sweeps. The
lightweight NumPy random forest proposes candidates by minimizing predicted
routed output error minus half a standard deviation of tree disagreement.
Uniform random search is the equal-trial comparator. Repeated configs are
reused from disk rather than retrained.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/narrow-expert-distill/search.py \
  experiments/teacher-qat/routed_data.npz \
  --initial-trials 6 --additional-trials 4 --epochs 5 --batch-size 32 \
  --out experiments/narrow-expert-distill/real-search
```

The current aggregate's explicit `x_train` and `x_tune` arrays are consumed as
provided, with their sidecar document IDs checked for overlap. A legacy
aggregate without tune arrays is split by document from `training_document_rows`.
The `heldout` document set remains untouched until both policy winners are
fixed, then scores each winner and the current published baseline. If the
teacher-QAT packed control exists at
`experiments/teacher-qat/qat-routed`, it is scored on the same final test rows too.
The result records both per-policy trial counts, selected configs, tuning
errors, final test errors, byte totals and ratios versus available controls.

## Synthetic-only smoke

The smoke fixtures preserve 1,536-wide I/O but use generated targets, are
explicitly labeled synthetic, and must not be presented as behavioral results.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/narrow-expert-distill/distill.py \
  --synthetic --hidden 64 --epochs 2 --batch-size 16 \
  --out experiments/narrow-expert-distill/synthetic-smoke

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/narrow-expert-distill/search.py \
  --synthetic --initial-trials 2 --additional-trials 1 --epochs 1 \
  --out experiments/narrow-expert-distill/synthetic-search-smoke-v4

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/narrow-expert-distill/search.py \
  --synthetic-documents --initial-trials 2 --additional-trials 1 --epochs 1 \
  --out experiments/narrow-expert-distill/synthetic-document-search-smoke

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/narrow-expert-distill/make_synthetic_contract.py \
  /tmp/narrow-explicit-split-fixture.npz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/narrow-expert-distill/search.py \
  /tmp/narrow-explicit-split-fixture.npz --initial-trials 2 --additional-trials 1 \
  --epochs 1 --batch-size 32 --out experiments/narrow-expert-distill/synthetic-explicit-search-smoke
```

The single-student smoke completed with intermediate width 64, rank 64 factors,
and 33,431 model-specific bytes including metadata. Its heldout relative output
MSE was 1.29743; current `published/champion` SVQ1 scored 1.00730 on the same
synthetic targets (student/baseline ratio 1.2880). The packed leader is commit
`c190ee1`, ranks 182/129, and its separate raw weight score is 0.8983728.
These are separate weight and behavioral metrics. This is a pipeline check
only. The bounded search smoke ran two shared starts and one additional
proposal per policy (four unique model training runs total). Random selected
width 320 and scored routed output MSE 1.12477 on synthetic test rows
(1.11640× the SVQ1 packed baseline); RF-guided selected width 128 and scored
1.09169 (1.08358× baseline). The teacher-QAT pair is unavailable because only
its small synthetic fixture exists. The published leader's 0.89837 raw weight
score is a separate metric. None of these synthetic results are behavioral
evidence.

The document-split pipeline smoke placed `cal-0` in tuning validation,
`cal-1`/`cal-2` in training, and `test-a`/`test-b` in the final test group.
Its output is separately labeled synthetic and does not stand in for captured
documents. The explicit-aggregate smoke uses the same `x_train/x_tune/x_val`
keys and document-ID sidecar as teacher-QAT; it places `fit-a/fit-b` in
training, `tune-a` in tuning, and `test-a/test-b` in final test.
