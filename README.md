# Hy4-flash expert compression

The current experiment compresses the gate/up and down projections of layer 1, routed expert 0 from [Tencent Hy4-flash](https://huggingface.co/tencent/Hy4-flash). The combined budget is 158,632 bytes including signs, scales and headers. Lower combined relative Frobenius error is better. The score measures weight reconstruction. Full-model language quality and inference are separate experiments.

The [current result](champion/result.json) records both packed files, their scores, checksums and method. The history of main tracks verified improvements. Fetch main before starting a new round, and use the packed factors or the winning encoder as a starting point. A gain on one projection can be combined with another worker’s gain on the other projection whenever the combined result fits the budget.

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
