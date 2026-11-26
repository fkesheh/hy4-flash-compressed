# Ternary SwiGLU student: layer 1 expert 0

This experiment is a BitNet b1.58-inspired, quantization-aware student for
Hy4-flash layer 1 expert 0. It keeps the original 1,536-wide input and output,
uses SwiGLU intermediate width 128 by default, and trains ternary
`{-1, 0, +1}` weights with learned positive per-output-row scales. The forward
path is ternary at every minibatch. The straight-through estimator passes the
weight gradient to latent weights and differentiates the loss through each
learned scale. Gate is capped at 10, up is clamped to `[-10, 10]`, and the
activation is `SiLU(gate) * up`.

The self-contained T58X format packs four 2-bit weight codes per byte and
stores each row scale as FP16. Headers, scales, both packed matrices, and
`student.json` are all included in the model-specific byte count. The cap is
158,632 bytes. Result JSON is an experiment report and is excluded from the
model payload count.

## Synthetic smoke and reproduction

Real teacher captures were absent when this run began, so the initial
distillation uses labeled synthetic teacher outputs. This is a pipeline smoke,
not a routed-document behavioral result. CPU BLAS is restricted to one thread.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/bitnet-expert/run.py \
  --synthetic --hidden 128 --epochs 10 --batch-size 16 --lr 0.001 \
  --seed 53 --out experiments/bitnet-expert/synthetic-smoke-10ep
```

This completed with 96 synthetic training rows and 48 synthetic heldout rows.
Decoded packed student relative expert-output MSE was **1.89622** and
router-weighted relative output MSE was **2.11286**. Model bytes, including the
manifest, were **151,799 / 158,632**. Same heldout rows and teacher targets:

| Packed model | Bytes | Relative output MSE | Router-weighted relative MSE |
| --- | ---: | ---: | ---: |
| Ternary student, hidden 128 | 151,799 | 1.89622 | 2.11286 |
| Local verified SVQ1 champion | 158,596 | 1.00746 | 1.00840 |
| Existing narrow student, hidden 64 | 33,431 | 1.29743 | 1.35047 |

These rows are synthetic and cannot establish behavioral performance. The
ternary implementation is currently behind both controls on this smoke.

The initial attempt with system `python` failed because NumPy is not installed
there; using the already prepared `/tmp/teacher-qat-venv` succeeded. Re-run
settings and measurements are stored in `synthetic-smoke-10ep/result.json`.

## Routed data

Use the shared `experiments/teacher-qat/routed_data.npz` aggregate when the
routed-data worker finishes capture and aggregation. It must preserve
document-disjoint fit and final heldout groups; use only calibration documents
for fitting. Expected arrays are `x_train`, `y_train`, `x_val`, `y_val`, with
optional `route_weight_train` and `route_weight_val`; both input and output
widths must be 1,536. The adjacent `.metadata.json` provenance is retained in
the student manifest/result. Run with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments/bitnet-expert/run.py \
  experiments/teacher-qat/routed_data.npz --hidden 128 --epochs 10 \
  --batch-size 16 --lr 0.001 --seed 53 \
  --out experiments/bitnet-expert/routed-width128
```

When real captures land, compare the decoded packed output against the latest
verified SVQ1 and narrow student on identical heldout routed rows. Keep those
document-level behavioral results separate from synthetic diagnostics and raw
weight reconstruction metrics.
