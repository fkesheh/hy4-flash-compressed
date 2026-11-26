#!/usr/bin/env python3
"""BitNet b1.58-inspired ternary SwiGLU QAT/distillation for Hy4 layer 1 expert 0.

Packed matrices use two-bit symbols (zero, +1, -1), FP16 learned row scales,
and an explicit header. The artifact count includes both headers and student.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WIDTH = 1536
CAP = 158632
HEADER = struct.Struct("<4sHHII")  # magic, version, rows, cols, scale count


def swiglu(x, wg, wd, cache=False):
    gu = x @ wg.T
    rg, ru = np.split(gu, 2, axis=-1)
    g = np.minimum(rg, 10.0)
    u = np.clip(ru, -10.0, 10.0)
    sig = 1.0 / (1.0 + np.exp(-np.clip(g, -80, 80)))
    silu = g * sig
    h = silu * u
    y = h @ wd.T
    return (y, (rg, ru, g, u, sig, silu, h)) if cache else y


def rel_mse(a, b):
    return float(np.square(a - b).sum(dtype=np.float64) / max(np.square(b).sum(dtype=np.float64), 1e-20))


def routed_mse(a, b, w):
    return None if w is None else rel_mse(a * w[:, None], b * w[:, None])


def data_synthetic(seed=53):
    rng = np.random.default_rng(seed)
    ntr, nv, th = 96, 48, 48
    wg = rng.normal(0, 1 / math.sqrt(WIDTH), (2 * th, WIDTH)).astype(np.float32)
    wd = rng.normal(0, 1 / math.sqrt(th), (WIDTH, th)).astype(np.float32)
    xtr = rng.normal(size=(ntr, WIDTH)).astype(np.float32)
    xv = rng.normal(size=(nv, WIDTH)).astype(np.float32)
    ytr, yv = swiglu(xtr, wg, wd), swiglu(xv, wg, wd)
    return xtr, ytr, xv, yv, rng.uniform(.1, 1, ntr).astype(np.float32), rng.uniform(.1, 1, nv).astype(np.float32), {
        "kind": "synthetic-smoke-only", "seed": seed, "split_unit": "synthetic-example",
        "training_document_count": 0, "heldout_document_count": 0}


def load_dataset(path):
    with np.load(path, allow_pickle=False) as d:
        xtr, ytr, xv, yv = [np.asarray(d[k], np.float32) for k in ("x_train", "y_train", "x_val", "y_val")]
        wt = np.asarray(d["route_weight_train"], np.float32) if "route_weight_train" in d else None
        wv = np.asarray(d["route_weight_val"], np.float32) if "route_weight_val" in d else None
    if any(a.ndim != 2 or a.shape[1] != WIDTH for a in (xtr, ytr, xv, yv)) or len(xtr) != len(ytr) or len(xv) != len(yv):
        raise ValueError("expected aligned rank-two 1536-wide expert inputs/outputs")
    if wt is not None and (len(wt) != len(xtr) or wv is None or len(wv) != len(xv)):
        raise ValueError("route weights must align with both train and heldout rows")
    meta = path.with_suffix(".metadata.json")
    provenance = json.loads(meta.read_text()) if meta.exists() else {"kind": "routed-capture", "dataset": str(path)}
    return xtr, ytr, xv, yv, wt, wv, provenance


def quantize(w, log_scale):
    scale = np.exp(log_scale).astype(np.float32)
    code = np.where(w >= (0.5 * scale[:, None]), 1.0,
                    np.where(w <= (-0.5 * scale[:, None]), -1.0, 0.0)).astype(np.float32)
    return code * scale[:, None], code, scale


def train(xtr, ytr, xv, yv, wt, wv, hidden, epochs, batch, lr, seed):
    rng = np.random.default_rng(seed)
    params = [rng.normal(0, math.sqrt(2 / WIDTH), (2 * hidden, WIDTH)).astype(np.float32),
              rng.normal(0, math.sqrt(1 / hidden), (WIDTH, hidden)).astype(np.float32)]
    # One learned positive scale per output row (per hidden feature for down).
    logs = [np.log(np.maximum(np.mean(np.abs(p), axis=1) / .85, 1e-4)).astype(np.float32) for p in params]
    all_params = params + logs
    ms = [np.zeros_like(p) for p in all_params]
    vs = [np.zeros_like(p) for p in all_params]
    hist, step = [], 0
    for ep in range(epochs):
        order = rng.permutation(len(xtr))
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            xb, yb = xtr[idx], ytr[idx]
            qg, cg, sg = quantize(params[0], logs[0])
            qd, cd, sd = quantize(params[1], logs[1])
            pred, (rg, ru, g, u, sig, silu, h) = swiglu(xb, qg, qd, True)
            weight = wt[idx, None] if wt is not None else 1.0
            dy = (pred - yb) * (weight * weight) * (2.0 / pred.size)
            grad_d = dy.T @ h
            dh = dy @ qd
            dg = dh * u * (sig * (1 + g * (1 - sig))) * (rg <= 10)
            du = dh * silu * ((ru >= -10) & (ru <= 10))
            grad_g = np.concatenate((dg, du), axis=-1).T @ xb
            # STE through ternary projection for latent weights; scale derivative
            # treats ternary codes as fixed, then applies d(log scale)/dscale.
            grad_logs = [np.sum(grad_g * cg, axis=1) * sg,
                         np.sum(grad_d * cd, axis=1) * sd]
            step += 1
            for p, grad, m, v in zip(all_params, (grad_g, grad_d, *grad_logs), ms, vs):
                m *= .9; m += .1 * grad
                v *= .999; v += .001 * grad * grad
                p -= lr * (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
            logs[0][:] = np.clip(logs[0], -12, 5)
            logs[1][:] = np.clip(logs[1], -12, 5)
        qg, _, _ = quantize(params[0], logs[0]); qd, _, _ = quantize(params[1], logs[1])
        yp = swiglu(xv, qg, qd)
        hist.append({"epoch": ep + 1, "validation_relative_output_mse": rel_mse(yp, yv),
                     "validation_relative_routed_output_mse": routed_mse(yp, yv, wv)})
    return params, logs, hist


def pack_matrix(path, latent, logs):
    q, code, scale = quantize(latent, logs)
    # 00=zero, 01=+1, 10=-1; little-endian symbols within each byte.
    symbols = np.where(code > 0, 1, np.where(code < 0, 2, 0)).astype(np.uint8).ravel()
    symbols = np.pad(symbols, (0, (-len(symbols)) % 4))
    packed = (symbols[0::4] | (symbols[1::4] << 2) | (symbols[2::4] << 4) | (symbols[3::4] << 6)).tobytes()
    scales = scale.astype("<f2").tobytes()
    rows, cols = latent.shape
    blob = HEADER.pack(b"T58X", 1, rows, cols, rows) + scales + packed
    path.write_bytes(blob)
    return len(blob), q


def unpack_matrix(path):
    b = path.read_bytes()
    magic, version, rows, cols, nscale = HEADER.unpack_from(b)
    if magic != b"T58X" or version != 1 or nscale != rows:
        raise ValueError(f"invalid ternary matrix: {path}")
    off = HEADER.size
    scales = np.frombuffer(b, dtype="<f2", count=rows, offset=off).astype(np.float32)
    sym = np.frombuffer(b, dtype=np.uint8, offset=off + 2 * rows)
    codes = np.empty(len(sym) * 4, np.uint8)
    codes[0::4] = sym & 3; codes[1::4] = (sym >> 2) & 3
    codes[2::4] = (sym >> 4) & 3; codes[3::4] = (sym >> 6) & 3
    codes = codes[:rows * cols].reshape(rows, cols)
    vals = np.where(codes == 1, 1., np.where(codes == 2, -1., 0.)).astype(np.float32)
    return vals * scales[:, None]


def run(args):
    if args.synthetic:
        xtr, ytr, xv, yv, wt, wv, provenance = data_synthetic(args.seed)
    else:
        xtr, ytr, xv, yv, wt, wv, provenance = load_dataset(args.dataset)
        provenance["kind"] = "routed-document-capture"
    params, logs, history = train(xtr, ytr, xv, yv, wt, wv, args.hidden, args.epochs, args.batch_size, args.lr, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    gb, _ = pack_matrix(args.out / "gate-up.t58", params[0], logs[0])
    db, _ = pack_matrix(args.out / "down.t58", params[1], logs[1])
    wg, wd = unpack_matrix(args.out / "gate-up.t58"), unpack_matrix(args.out / "down.t58")
    pred = swiglu(xv, wg, wd)
    controls = {}
    if args.synthetic:
        # Compare packed decodes on these exact heldout rows; labels remain synthetic.
        sys.path.insert(0, str(ROOT))
        import scale_vq
        base = ROOT / "champion"
        bg = scale_vq.decode_matrix((base / "gate-up.svq").read_bytes())
        bd = scale_vq.decode_matrix((base / "down.svq").read_bytes())
        bp = swiglu(xv, bg, bd)
        controls["latest_local_verified_svq1"] = {"files": [str(base / "gate-up.svq"), str(base / "down.svq")],
            "packed_bytes": (base / "gate-up.svq").stat().st_size + (base / "down.svq").stat().st_size,
            "heldout_relative_output_mse": rel_mse(bp, yv), "heldout_relative_routed_output_mse": routed_mse(bp, yv, wv),
            "student_to_svq1_output_mse_ratio": rel_mse(pred, yv) / max(rel_mse(bp, yv), 1e-20)}
        narrow = ROOT / "experiments/narrow-expert-distill/synthetic-smoke"
        if (narrow / "gate-up.hyb").is_file() and (narrow / "down.hyb").is_file():
            sys.path.insert(0, str(ROOT / "experiments/narrow-expert-distill"))
            import distill as narrow_distill
            npred, nmse, nrouted = narrow_distill.evaluate_packed(
                xv, yv, wv, narrow / "gate-up.hyb", narrow / "down.hyb")
            controls["narrow_student_same_synthetic_rows"] = {"files": [str(narrow / "gate-up.hyb"), str(narrow / "down.hyb")],
                "packed_bytes": (narrow / "gate-up.hyb").stat().st_size + (narrow / "down.hyb").stat().st_size + (narrow / "student.json").stat().st_size,
                "heldout_relative_output_mse": nmse, "heldout_relative_routed_output_mse": nrouted,
                "student_to_narrow_output_mse_ratio": rel_mse(pred, yv) / max(nmse, 1e-20)}
    meta = {"label": "BitNet b1.58-inspired ternary SwiGLU student", "format": "T58X",
            "weight_values": [-1, 0, 1], "learned_scale": "FP16 per output row", "input_width": WIDTH,
            "output_width": WIDTH, "student_intermediate": args.hidden, "packed_files": {"gate-up.t58": gb, "down.t58": db},
            "metadata_byte_counting": "student.json file size is included; matrix headers/scales/payloads are included",
            "data_kind": provenance.get("kind"), "provenance": provenance, "byte_cap": CAP}
    manifest = args.out / "student.json"
    meta["model_specific_bytes_including_metadata"] = 0
    for _ in range(8):
        manifest.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        nbytes = gb + db + manifest.stat().st_size
        if nbytes == meta["model_specific_bytes_including_metadata"]:
            break
        meta["model_specific_bytes_including_metadata"] = nbytes
    manifest.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    total = gb + db + manifest.stat().st_size
    if total > CAP:
        raise RuntimeError(f"packed model exceeds cap: {total} > {CAP}")
    result = {"label": "SYNTHETIC SMOKE ONLY" if args.synthetic else "routed document-disjoint behavioral result",
              "objective": "teacher output MSE with router-weighted training/reporting when weights are present",
              "widths": {"input": WIDTH, "hidden": args.hidden, "output": WIDTH}, "train_rows": len(xtr), "heldout_rows": len(xv),
              "provenance": provenance, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
              "seed": args.seed, "blas_threads": 1, "packed_bytes_including_metadata": total,
              "byte_cap": CAP, "heldout_relative_output_mse": rel_mse(pred, yv),
              "heldout_relative_routed_output_mse": routed_mse(pred, yv, wv), "same_rows_packed_controls": controls,
              "history": history}
    (args.out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path, nargs="?")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "synthetic-smoke")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=53)
    a = p.parse_args()
    if a.synthetic == (a.dataset is not None):
        p.error("provide exactly one dataset path or --synthetic")
    if not 1 <= a.hidden < WIDTH:
        p.error("hidden must be in 1..1535")
    run(a)


if __name__ == "__main__":
    main()
