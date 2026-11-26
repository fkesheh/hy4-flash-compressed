# Dual-SVID layer 1 expert 0 ablation

Implements the paper's rank-one absolute-factor SVD initialization: signs are
taken from balanced truncated SVD factors, and leading rank-one SVDs of each
factor's magnitudes initialize row, column and latent scales. Two-path runs
initialize path 2 from the residual and refine both paths against the joint
target. The separate proxy ablation uses sign in the forward pass and compares
the STE derivative with the paper's `100 * (1 - tanh(100x)^2)` SmoothSign
derivative.

## Best raw-reference result

| Configuration | Ranks | Packed bytes | Combined relative Frobenius error |
| --- | ---: | ---: | ---: |
| One path, 160 coordinate rounds | 167 / 121 | 158,632 | **0.9072653203** |
| Two residual paths, 160 rounds | 75+75 / 53+53 | 158,416 | 0.9142664277 |
| Two paths, both initialized from W (control) | 75+75 / 53+53 | 158,416 | 0.9162131518 |

The two-path rank step sizes leave 216 bytes unused under the 158,632-byte
limit. Their residual and control configurations have identical packed sizes.
The one-path result improves the 0.90944985 baseline by about 0.00218.

## SmoothSign / STE proxy-gradient results

Each row compares the same initialization, rank and byte count; 12 NumPy Adam
steps were used for this small matrix-level optimization.

| Paths | Initialization | Bytes | STE | SmoothSign |
| ---: | --- | ---: | ---: | ---: |
| 1 | W | 158,632 | 0.9374078993 | 0.9369344659 |
| 2 | Residual | 158,416 | 0.9400457452 | 0.9396952760 |
| 2 | Both from W (control) | 158,416 | 0.9478997871 | 0.9477471593 |

SmoothSign slightly beats STE in each proxy-gradient pair, while both proxy
optimizers trail alternating sign/scale coordinate refinement.

## Reproduction and scope

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/hy4-compress-env/bin/python experiments/dual-svid/run.py --sweeps 160
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/hy4-compress-env/bin/python experiments/dual-svid/proxy_ablation.py
```

The first command reproduces the raw-reference score and writes self-contained
HYB1 factors under this directory. `attempts.jsonl` records all measurements,
failures and superseded runs. CPU scoring used the shared `experiments/score.py`
references. No real routed teacher captures were present on this bridge, so no
held-out behavioral score was measured. No RTX 3060 inference check was run.
