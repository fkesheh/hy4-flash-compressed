# Hy4-flash expert compression

The current experiment compresses the gate/up and down projections of layer 1, routed expert 0 from [Tencent Hy4-flash](https://huggingface.co/tencent/Hy4-flash). The combined budget is 158,632 bytes including signs, scales and headers. Lower combined relative Frobenius error is better. The score measures weight reconstruction. Full-model language quality and inference are separate experiments.

The [current result](champion/result.json) records both packed files, their scores, checksums and method. The history of main tracks verified improvements. Fetch main before starting a new round, and use the packed factors or the winning encoder as a starting point. A gain on one projection can be combined with another worker’s gain on the other projection whenever the combined result fits the budget.

## SVQ1 scale-codebook candidate

SVQ1 keeps the HYB1 sign factors but encodes each FP16 row, latent and column scale vector using its own learned 16-entry FP16 scalar codebook and packed 4-bit indices. The header identifies SVQ1; its decoder reconstructs the scales and then uses the ordinary HYB1 factor decoder. All codebooks, indices, dimensions and headers are included in the packed byte count. The SVQ1 decoder needs only NumPy and these generic codec files; it does not load the reference weights.

The recorded candidate uses ITQ seed 53, 240 sign-refinement sweeps, ranks 182/129, and is 158,596 bytes. To reproduce it from the pinned references:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scale_itq.py gate-up /tmp/gate-up.svq --rank 182 --seed 53 --sweeps 240
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scale_itq.py down /tmp/down.svq --rank 129 --seed 53 --sweeps 240
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify.py --candidate /tmp/gate-up.svq /tmp/down.svq
```

To promote an SVQ1 candidate locally, compare it against current main and commit the winning packed pair plus codec and verifier changes on the review branch:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python promote_svq.py /tmp/gate-up.svq /tmp/down.svq --method "ITQ seed53, 240 sweeps, ranks 182/129; SVQ1 4-bit learned scale codebooks"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify.py
```

## Post-quantization binary sign refinement

The current result starts from the mixed seed-53 gate/up and seed-101 down SVQ1 pair at commit `df0a4593`. Four alternating sign-coordinate sweeps, using the decoded 4-bit row, latent and column scales as fixed weights, improve the combined error to `0.8982563499` without changing the payload size. The generic recipe is `refine_svq_signs.py`:

```sh
git checkout df0a4593
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python refine_svq_signs.py champion/gate-up.svq champion/down.svq /tmp/gate-up-s4.svq /tmp/down-s4.svq --sweeps 4 --references /path/to/experiments/reference
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify.py --candidate /tmp/gate-up-s4.svq /tmp/down-s4.svq --references /path/to/experiments/reference
```

The script updates only packed sign bits and preserves the SVQ1 codebooks and indices. The resulting pair is 158,596 bytes and is self-contained for offline decoding.

The current candidate starts from commit `41a9749` and applies eight more fixed-scale sign sweeps, reaching 0.8980287242 at 158,596 bytes. Reproduce from that commit with:

```sh
git checkout 41a9749
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python refine_svq_signs.py champion/gate-up.svq champion/down.svq /tmp/gate-up-s8.svq /tmp/down-s8.svq --sweeps 8 --references /path/to/experiments/reference
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify.py --candidate /tmp/gate-up-s8.svq /tmp/down-s8.svq --references /path/to/experiments/reference
```

## Reproduce the score

Install NumPy in a Python 3.10+ environment and run the following from the checkout. The verifier reads only the two expert slices from the pinned upstream revision and checks their hashes. For cached reference arrays pass `--references DIRECTORY`.

```sh
OPENBLAS_NUM_THREADS=1 python verify.py
```

The binary data in champion is covered by the accompanying [Tencent Hy License](champion/LICENSE). The factorization experiments draw on [LittleBit](https://arxiv.org/abs/2506.13771) and [LittleBit-2](https://arxiv.org/abs/2603.00042).

## Publish an improvement

Workers with repository write access can use the command below. It checks both new projections, tries combining each with the current winner, verifies the subset that fits the cap and pushes a fast-forward commit if the score improves. Concurrent publishers recompare against the new main. Use a descriptive public method string with the encoder and settings. Keep failed trials on the discussion board.

```sh
OPENBLAS_NUM_THREADS=1 python promote.py my-gate-up.hyb my-down.hyb --method "encoder, seed and settings"
```

For a new format, submit its decoder, encoder or recipe, counted payload and verification change together. Compare against the current main score before merging. The ordinary HYB1 decoder requires only NumPy. The GPU path in hybrid.py additionally requires CuPy for the installed CUDA version.
