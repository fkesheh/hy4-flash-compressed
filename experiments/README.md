# Alternate expert compression trials

Both trials use layer 1 routed expert 0. Scores are combined relative weight error.

| Trial | Packed bytes | Error |
|---|---:|---:|
| Dual-SVID, one path | 158,632 | 0.9072653203 |
| SVQ3, three-bit scale indices | 158,462 | 0.8986520352 |
| SVQ1 baseline at 41a9749 | 158,596 | 0.8981078939 |

Install NumPy and run these commands from the repository root. The first command fetches and verifies the pinned reference slices and caches them locally.

```sh
python - <<'PY'
from pathlib import Path
import numpy as np
from verify import TENSORS, reference
folder = Path("experiments/reference")
folder.mkdir(exist_ok=True)
for name, (suffix, _, _) in TENSORS.items():
    np.save(folder / f"hy4-{suffix}.npy", reference(name, None).astype(np.float32))
PY
OPENBLAS_NUM_THREADS=1 python experiments/dual-svid/run.py --sweeps 160
OPENBLAS_NUM_THREADS=1 python experiments/dual-svid/proxy_ablation.py
OPENBLAS_NUM_THREADS=1 python experiments/product-vq/scale_alloc3.py --rg 183 --rd 130 --sweeps 240 --gate-seed 53 --down-seed 101
```
