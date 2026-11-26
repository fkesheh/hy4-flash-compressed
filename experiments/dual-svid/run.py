"""CPU Dual-SVID initialization and residual-path byte-budget ablation.

Uses only NumPy.  All BLAS work is intended to run with one thread.
"""
import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import compress
from experiments.score import score

OUT = Path(__file__).resolve().parent
LIMIT = 158632


def dual_svid_init(w, rank):
    """Paper Dual-SVID: truncated-SVD signs and rank-1 magnitude scales."""
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    root = np.sqrt(s[:rank])
    up = u[:, :rank] * root
    vp = vt[:rank].T * root
    # Split each leading singular value symmetrically between row and latent vectors.
    uu, eu, euh = np.linalg.svd(np.abs(up), full_matrices=False)
    vv, ev, evh = np.linalg.svd(np.abs(vp), full_matrices=False)
    h = eu[0] ** 0.5 * uu[:, 0]
    lu = eu[0] ** 0.5 * euh[0]
    g = ev[0] ** 0.5 * vv[:, 0]
    lv = ev[0] ** 0.5 * evh[0]
    ell = lu * lv
    a = np.where(up >= 0, 1., -1.).astype(np.float32)
    b = np.where(vp.T >= 0, 1., -1.).astype(np.float32)
    return {"a": a, "b": b, "h": h.astype(np.float32),
            "g": g.astype(np.float32), "ell": ell.astype(np.float32),
            "u": up.astype(np.float32), "v": vp.astype(np.float32)}


def output(p):
    return (p["h"][:, None] * p["a"] * p["ell"][None, :]) @ (p["b"] * p["g"][None, :])


def fit_single_path(target, p, sweeps=2):
    """Refine signs using coordinate minimization while fitting rank scales."""
    a, b, h, g, ell = p["a"], p["b"], p["h"], p["g"], p["ell"]
    for _ in range(sweeps):
        # Refine each side against the current opposite side and current scales.
        compress.refine_signs(target, a * h[:, None] * ell[None, :], g, b)
        compress.refine_signs(target.T, b.T * g[:, None] * ell[None, :], h, a.T)
        base = (a * ell[None, :]) @ b
        # Conditional least-squares update for row then column scales.
        bg = b * g[None, :]
        q = (a * ell[None, :]) @ bg
        h = np.sum(target * q, axis=1) / np.maximum(np.sum(q*q, axis=1), 1e-30)
        ah = a * h[:, None] * ell[None, :]
        q = ah @ b
        g = np.sum(target * q, axis=0) / np.maximum(np.sum(q*q, axis=0), 1e-30)
        # Joint least-squares update for all latent scales.
        left, right = h[:, None] * a, b * g[None, :]
        gram = (left.T @ left) * (right @ right.T)
        rhs = np.sum((left.T @ target) * right, axis=1)
        ell = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    p.update(a=a, b=b, h=h, g=g, ell=ell)
    return p


def size(m, n, r):
    return compress.packed_size(m, n, r, 1)


def pack(p):
    a, b, h, g, ell = p["a"], p["b"], p["h"], p["g"], p["ell"]
    m, r = a.shape
    n = b.shape[1]
    parts = [compress.HEADER.pack(b"HYB1", m, n, r, 1),
             np.packbits(a.reshape(-1) > 0).tobytes(),
             np.packbits(b.reshape(-1) > 0).tobytes()]
    for x in (h, ell, g):
        parts.append(x.astype("<f2").tobytes())
    return b"".join(parts)


def decode(path):
    a, b = compress.unpack(path.read_bytes())
    return a @ b


def best_ranks(shapes, paths):
    """Maximize the sum of ranks, keeping per-path ranks equal, under LIMIT."""
    if paths == 1:
        m, n = shapes[0]
        return [max(r for r in range(1, min(m, n)+1) if size(m,n,r) <= LIMIT)]
    (m, n) = shapes[0]
    return [max(r for r in range(1, min(m,n)+1) if paths*size(m,n,r) <= LIMIT)]*paths


def run_case(tag, matrices, ranks, residual=True, sweeps=2):
    paths = []
    for i, (w, rank) in enumerate(zip(matrices, ranks)):
        target = w.copy()
        if residual and i:
            target -= sum(output(p) for p in paths)
        p = dual_svid_init(target, rank)
        p = fit_single_path(target, p, sweeps=sweeps)
        paths.append(p)
    # Joint refinement: fit each path to W minus the other path's current output.
    for i in range(len(paths)):
        other = sum(output(p) for j,p in enumerate(paths) if j != i)
        paths[i] = fit_single_path(matrices[0] - other, paths[i], sweeps=sweeps)
    blobs = [pack(p) for p in paths]
    stem = OUT / tag
    for i, blob in enumerate(blobs):
        (OUT / f"{tag}-path{i}.hyb").write_bytes(blob)
    reconstructed = sum(decode(OUT / f"{tag}-path{i}.hyb") for i in range(len(paths)))
    return blobs, reconstructed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=2)
    args = ap.parse_args()
    matrices = [np.load(ROOT/"experiments/reference/hy4-gate_up_proj.npy"),
                np.load(ROOT/"experiments/reference/hy4-down_proj.npy")]
    m, n = matrices[0].shape
    # Jointly allocate rank across both tensors and paths, spending as much of LIMIT
    # as possible while maximizing total latent dimensions.
    def allocate(numpaths):
        targets = [LIMIT * w.size / sum(x.size for x in matrices) for w in matrices]
        cand0 = [(r, numpaths*size(*matrices[0].shape, r))
                 for r in range(1, min(matrices[0].shape)+1)
                 if numpaths*size(*matrices[0].shape, r) <= LIMIT]
        cand1 = [(r, numpaths*size(*matrices[1].shape, r))
                 for r in range(1, min(matrices[1].shape)+1)
                 if numpaths*size(*matrices[1].shape, r) <= LIMIT]
        feasible = [(-abs(b0-targets[0])-abs(b1-targets[1]), r0+r1,
                     -(b0+b1), r0, r1)
                    for r0,b0 in cand0 for r1,b1 in cand1
                    if b0+b1 <= LIMIT]
        _, _, _, r0, r1 = max(feasible)
        return [r0, r1]
    cases = []
    for tag, numpaths, residual in (("one-path",1,False), ("two-residual",2,True),
                                    ("two-no-residual",2,False)):
        ranks = allocate(numpaths)
        pair_blobs=[]; decoded=[]
        for idx, w in enumerate(matrices):
            blobs, recon = run_case(f"{tag}-{idx}",[w]*numpaths,[ranks[idx]]*numpaths,residual=residual,sweeps=args.sweeps)
            pair_blobs.extend(blobs); decoded.append(recon)
        total=sum(map(len,pair_blobs))
        s=score(decoded[0],decoded[1])
        rec={"tag":tag,"hypothesis":"Rank-one absolute-SVD scales provide a good SVID initialization; residual initialization should improve the secondary path, while the no-residual case is the same-byte control.","npaths":numpaths,"residual_init":residual,
             "ranks":ranks,"packed_bytes":total,"score":s,
             "status":"measured",
             "command":f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/hy4-compress-env/bin/python experiments/dual-svid/run.py --sweeps {args.sweeps}",
             "artifact":str(OUT)}
        cases.append(rec)
        with (OUT/"attempts.jsonl").open("a") as f: f.write(json.dumps(rec)+"\n")
        print(json.dumps(rec))


if __name__ == "__main__":
    main()
