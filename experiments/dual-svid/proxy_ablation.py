"""Small NumPy QAT proxy-gradient ablation: STE versus paper SmoothSign."""
import json
from pathlib import Path

import numpy as np

import run


def optimize(target, p, smooth, steps=12, lr=1e-3):
    """Adam on latent factors/scales with sign forward and STE/SmoothSign backward."""
    params = [p[k].astype(np.float64).copy() for k in ("u", "v", "h", "g", "ell")]
    moments = [np.zeros_like(x) for x in params]
    variances = [np.zeros_like(x) for x in params]
    for step in range(1, steps + 1):
        u, v, h, g, ell = params
        su, sv = np.where(u >= 0, 1., -1.), np.where(v >= 0, 1., -1.)
        left = su * h[:, None] * ell[None, :]
        right = sv * g[:, None]
        error = left @ right.T - target
        gu = (error @ right) * (h[:, None] * ell[None, :])
        gv = (error.T @ left) * g[:, None]
        if smooth:
            gu *= 100. * (1. - np.tanh(100. * u) ** 2)
            gv *= 100. * (1. - np.tanh(100. * v) ** 2)
        gh = np.sum(error * ((su * ell[None, :]) @ right.T), axis=1)
        gg = np.sum(error * (left @ sv.T), axis=0)
        gl = np.sum((h[:, None] * su) * (error @ right), axis=0)
        grads = [gu, gv, gh, gg, gl]
        for i, grad in enumerate(grads):
            moments[i] = .9 * moments[i] + .1 * grad
            variances[i] = .999 * variances[i] + .001 * grad * grad
            mhat = moments[i] / (1. - .9**step)
            vhat = variances[i] / (1. - .999**step)
            params[i] -= lr * mhat / (np.sqrt(vhat) + 1e-8)
    u, v, h, g, ell = params
    p.update(a=np.where(u >= 0, 1., -1.).astype(np.float32),
             b=np.where(v.T >= 0, 1., -1.).astype(np.float32),
             h=h.astype(np.float32), g=g.astype(np.float32),
             ell=ell.astype(np.float32), u=u.astype(np.float32), v=v.astype(np.float32))
    return p


def main():
    base = Path(__file__).resolve().parents[2]
    matrices = [np.load(base / "experiments/reference/hy4-gate_up_proj.npy"),
                np.load(base / "experiments/reference/hy4-down_proj.npy")]
    out = Path(__file__).resolve().parent
    # Residual pairs use the same total-byte allocations as run.py.
    for smooth in (False, True):
        for np_paths, residual in ((1, False), (2, True), (2, False)):
            ranks = [167, 121] if np_paths == 1 else [75, 53]
            reconstructions, blobs = [], []
            for w, rank in zip(matrices, ranks):
                paths = []
                for _ in range(np_paths):
                    target = w.copy()
                    if residual and paths:
                        target -= sum(run.output(x) for x in paths)
                    paths.append(optimize(target, run.dual_svid_init(target, rank), smooth))
                # Jointly adjust each path to the original target minus its peers.
                for i in range(np_paths):
                    other = sum(run.output(x) for j, x in enumerate(paths) if j != i)
                    paths[i] = optimize(w - other, paths[i], smooth)
                for i, path in enumerate(paths):
                    blob = run.pack(path)
                    name = f"proxy-{'smooth' if smooth else 'ste'}-{np_paths}path-{int(residual)}res-{len(reconstructions)}-{i}.hyb"
                    (out / name).write_bytes(blob)
                    blobs.append(blob)
                reconstructions.append(sum(run.output(x) for x in paths))
            score = run.score(reconstructions[0], reconstructions[1])
            total = sum(map(len, blobs))
            tag = f"proxy-{'smooth' if smooth else 'ste'}-{np_paths}path-res{int(residual)}"
            rec = {"tag": tag, "hypothesis": "Paper SmoothSign's tanh(100x) proxy gradient may improve sign-factor fitting over the STE under the same packed byte budget.",
                   "packed_bytes": total, "score": score, "status": "measured",
                   "command": "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/hy4-compress-env/bin/python experiments/dual-svid/proxy_ablation.py",
                   "artifact": str(out)}
            with (out / "attempts.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
