#!/usr/bin/env python3
"""Distill a narrower SwiGLU student, then score its serialized HYB1 factors.

Teacher data uses the routed-data/teacher-qat NPZ contract (1536-wide x and y).
The synthetic mode is only a pipeline smoke test and is labeled in every output.
All BLAS runtimes are pinned to one thread before NumPy is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import compress

CAP = 158632
WIDTH = 1536


def swiglu(x, wg, wd, return_cache=False):
    gu = x @ wg.T
    raw_gate, raw_up = np.split(gu, 2, axis=-1)
    gate = np.minimum(raw_gate, 10.0)
    up = np.clip(raw_up, -10.0, 10.0)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(gate, -80, 80)))
    silu = gate * sigmoid
    hidden = silu * up
    y = hidden @ wd.T
    if return_cache:
        return y, (raw_gate, raw_up, gate, up, sigmoid, silu, hidden)
    return y


def rel_mse(pred, target):
    return float(np.square(pred - target).sum(dtype=np.float64) /
                 max(np.square(target).sum(dtype=np.float64), 1e-20))


def routed_rel_mse(pred, target, weight):
    if weight is None:
        return None
    w = np.asarray(weight, np.float32).reshape(-1, 1)
    return rel_mse(pred * w, target * w)


def synthetic_data(seed=53):
    """Generate explicitly synthetic 1536-wide routed expert targets."""
    rng = np.random.default_rng(seed)
    ntrain, nval, teacher_hidden = 96, 48, 48
    wg = rng.normal(0, 1 / math.sqrt(WIDTH), (2 * teacher_hidden, WIDTH)).astype(np.float32)
    wd = rng.normal(0, 1 / math.sqrt(teacher_hidden), (WIDTH, teacher_hidden)).astype(np.float32)
    xtr = rng.normal(size=(ntrain, WIDTH)).astype(np.float32)
    xv = rng.normal(size=(nval, WIDTH)).astype(np.float32)
    ytr = swiglu(xtr, wg, wd)
    yv = swiglu(xv, wg, wd)
    wtr = rng.uniform(.1, 1, ntrain).astype(np.float32)
    wv = rng.uniform(.1, 1, nval).astype(np.float32)
    return xtr, ytr, xv, yv, wtr, wv, {"kind": "synthetic", "seed": seed,
        "teacher_intermediate": teacher_hidden, "training_document_count": 0,
        "heldout_document_count": 0}


def load_data(path):
    with np.load(path, allow_pickle=False) as data:
        required = ("x_train", "y_train", "x_val", "y_val")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"dataset missing keys: {missing}")
        xtr, ytr, xv, yv = [np.asarray(data[k], np.float32) for k in required]
        wtr = np.asarray(data["route_weight_train"], np.float32) if "route_weight_train" in data else None
        wv = np.asarray(data["route_weight_val"], np.float32) if "route_weight_val" in data else None
    if any(a.ndim != 2 for a in (xtr, ytr, xv, yv)):
        raise ValueError("expected flattened rank-two inputs and targets")
    if xtr.shape[1] != WIDTH or xv.shape[1] != WIDTH or ytr.shape[1] != WIDTH or yv.shape[1] != WIDTH:
        raise ValueError("layer-1 expert input/output dimensions must both remain 1536")
    if len(xtr) != len(ytr) or len(xv) != len(yv):
        raise ValueError("input and target counts differ")
    if wtr is not None and (len(wtr) != len(xtr) or len(wv) != len(xv)):
        raise ValueError("route weights do not align with examples")
    sidecar = path.with_suffix(".metadata.json")
    provenance = json.loads(sidecar.read_text()) if sidecar.exists() else {"kind": "routed-capture", "dataset": str(path)}
    return xtr, ytr, xv, yv, wtr, wv, provenance


def train(xtr, ytr, xv, yv, wtr, wv, hidden, epochs, batch, lr, seed,
          training_steps=None, training_examples=None):
    if hidden <= 0 or hidden >= WIDTH:
        raise ValueError("student intermediate width must be in 1..1535")
    rng = np.random.default_rng(seed)
    wg = rng.normal(0, math.sqrt(2 / WIDTH), (2 * hidden, WIDTH)).astype(np.float32)
    wd = rng.normal(0, math.sqrt(1 / hidden), (WIDTH, hidden)).astype(np.float32)
    params = [wg, wd]
    ms = [np.zeros_like(p) for p in params]
    vs = [np.zeros_like(p) for p in params]
    history = []
    step = 0
    if training_steps is not None and training_examples is not None:
        raise ValueError("choose either a training step count or example count")
    if training_examples is not None:
        loops = [math.ceil(training_examples / batch)]
    else:
        loops = [training_steps] if training_steps is not None else [math.ceil(len(xtr) / batch)] * epochs
    order = rng.permutation(len(xtr))
    cursor = 0
    examples_left = training_examples
    for loop_index, steps_this_loop in enumerate(loops):
        if training_steps is None and training_examples is None:
            order = rng.permutation(len(xtr))
        for batch_index in range(steps_this_loop):
            if training_steps is None and training_examples is None:
                idx = order[batch_index * batch:(batch_index + 1) * batch]
            else:
                batch_now = min(batch, examples_left) if examples_left is not None else batch
                if cursor + batch_now > len(order):
                    order = rng.permutation(len(xtr))
                    cursor = 0
                idx = order[cursor:cursor + batch_now]
                cursor += len(idx)
                if examples_left is not None:
                    examples_left -= len(idx)
            xb, yb = xtr[idx], ytr[idx]
            pred, cache = swiglu(xb, wg, wd, True)
            raw_g, raw_u, gate, up, sig, silu, h = cache
            if wtr is None:
                d_out = (pred - yb) * (2.0 / pred.size)
            else:
                weight = wtr[idx].reshape(-1, 1)
                d_out = ((pred - yb) * (weight * weight)) * (2.0 / pred.size)
            grad_wd = d_out.T @ h
            d_h = d_out @ wd
            d_g = d_h * up * (sig * (1 + gate * (1 - sig))) * (raw_g <= 10.0)
            d_u = d_h * silu * ((raw_u >= -10.0) & (raw_u <= 10.0))
            grad_wg = np.concatenate((d_g, d_u), axis=-1).T @ xb
            step += 1
            for p, g, m, v in zip(params, (grad_wg, grad_wd), ms, vs):
                m *= .9; m += .1 * g
                v *= .999; v += .001 * g * g
                p -= lr * (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
        pred = swiglu(xv, wg, wd)
        metric = {"validation_relative_output_mse": rel_mse(pred, yv),
                  "validation_relative_routed_output_mse": routed_rel_mse(pred, yv, wv)}
        if training_steps is None and training_examples is None:
            metric["epoch"] = loop_index + 1
        elif training_steps is not None:
            metric["training_steps"] = training_steps
        else:
            metric["training_examples"] = training_examples
        history.append(metric)
    return wg, wd, history


def rank_bytes(m, n, rank, per_row=True):
    return compress.packed_size(m, n, rank, int(per_row))


def pack_at_rank(matrix, rank, out, per_row=True, sign_sweeps=2):
    m, n = matrix.shape
    bpw = 8.0 * rank_bytes(m, n, rank, per_row) / (m * n) + 1e-8
    blob, actual, _ = compress.fit(matrix, bpw=bpw, per_row=per_row, sign_sweeps=sign_sweeps)
    if actual != rank:
        raise RuntimeError(f"requested HYB rank {rank}, compressor emitted {actual}")
    out.write_bytes(blob)
    return blob


def model_meta_bytes(payload):
    return len((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def pack_student(wg, wd, out, provenance, hidden, quantization_scheme="hyb1-row-scale", sign_sweeps=2):
    out.mkdir(parents=True, exist_ok=True)
    m1, n1 = wg.shape
    m2, n2 = wd.shape
    rank_max = min(m1, n1, m2, n2)
    per_row = quantization_scheme == "hyb1-row-scale"
    if quantization_scheme not in ("hyb1-row-scale", "hyb1-global-scale"):
        raise ValueError(f"unsupported quantization scheme {quantization_scheme}")
    rank = max(r for r in range(1, rank_max + 1)
               if rank_bytes(m1, n1, r, per_row) + rank_bytes(m2, n2, r, per_row) + 4096 <= CAP)
    gpath, dpath = out / "gate-up.hyb", out / "down.hyb"
    while rank > 0:
        gblob = pack_at_rank(wg, rank, gpath, per_row, sign_sweeps)
        dblob = pack_at_rank(wd, rank, dpath, per_row, sign_sweeps)
        meta = {"label": "narrow-student-distillation", "student_intermediate": hidden,
                "input_width": WIDTH, "output_width": WIDTH, "format": "HYB1",
                "quantization_scheme": quantization_scheme, "sign_sweeps": sign_sweeps,
                "gate_up_rank": rank, "down_rank": rank,
                "packed_files": {"gate-up.hyb": len(gblob), "down.hyb": len(dblob)},
                "data_kind": provenance.get("kind", "routed-capture"),
                "split_unit": provenance.get("split_unit", "synthetic-example"),
                "dataset_sha256": provenance.get("dataset_sha256")}
        meta["byte_cap"] = CAP

        def save_manifest_and_count():
            manifest = out / "student.json"
            meta["model_specific_bytes_including_metadata"] = 0
            manifest.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            for _ in range(4):
                total_bytes = gpath.stat().st_size + dpath.stat().st_size + manifest.stat().st_size
                if meta["model_specific_bytes_including_metadata"] == total_bytes:
                    return total_bytes
                meta["model_specific_bytes_including_metadata"] = total_bytes
                manifest.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            raise RuntimeError("could not stabilize byte count in model metadata")

        total = save_manifest_and_count()
        if total <= CAP:
            break
        rank -= 1
    if rank <= 0:
        raise RuntimeError("no HYB1 rank fits byte cap including metadata")
    return rank, total, gpath, dpath


def evaluate_packed(x, y, weights, gate_path, down_path):
    # Score only factors read back from the bytes that will be shipped.
    def decode(path):
        raw = path.read_bytes()
        if raw[:4] == b"SVQ1":
            import scale_vq
            return scale_vq.decode_factors(raw)
        return compress.unpack(raw)
    a, b = decode(gate_path)
    c, d = decode(down_path)
    pred = swiglu(x, (a @ b).astype(np.float32), (c @ d).astype(np.float32))
    return pred, rel_mse(pred, y), routed_rel_mse(pred, y, weights)


def has_full_expert_pair(gate_path, down_path):
    try:
        gm, gn = compress.HEADER.unpack_from(gate_path.read_bytes())[1:3]
        dm, dn = compress.HEADER.unpack_from(down_path.read_bytes())[1:3]
        return (gm, gn, dm, dn) == (2 * WIDTH, WIDTH, WIDTH, WIDTH)
    except (OSError, ValueError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path, nargs="?")
    p.add_argument("--synthetic", action="store_true", help="run a labeled synthetic 1536-wide pipeline smoke")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "synthetic-smoke")
    p.add_argument("--baseline-gate", type=Path)
    p.add_argument("--baseline-down", type=Path)
    p.add_argument("--teacher-qat", type=Path, default=ROOT / "experiments/teacher-qat/qat-routed")
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=53)
    args = p.parse_args()
    if args.synthetic == (args.dataset is not None):
        p.error("provide exactly one dataset or --synthetic")
    if args.synthetic:
        xtr, ytr, xv, yv, wtr, wv, provenance = synthetic_data(args.seed)
        provenance["kind"] = "synthetic-smoke-only"
    else:
        xtr, ytr, xv, yv, wtr, wv, provenance = load_data(args.dataset)
        provenance["kind"] = "routed-document-capture"
    if args.baseline_gate is None or args.baseline_down is None:
        manifest_path = ROOT / "champion/result.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            projections = manifest.get("projections", {})
            args.baseline_gate = args.baseline_gate or ROOT / "champion" / projections.get("gate-up", {}).get("file", "gate-up.hyb")
            args.baseline_down = args.baseline_down or ROOT / "champion" / projections.get("down", {}).get("file", "down.hyb")
    if args.baseline_gate.is_file() and args.baseline_down.is_file():
        base_pred, base_mse, base_routed = evaluate_packed(xv, yv, wv, args.baseline_gate, args.baseline_down)
        baseline = {"gate": str(args.baseline_gate), "down": str(args.baseline_down),
                    "packed_bytes": args.baseline_gate.stat().st_size + args.baseline_down.stat().st_size,
                    "format": args.baseline_gate.read_bytes()[:4].decode("ascii", "replace"),
                    "published_raw_weight_score_separate": (json.loads((ROOT / "champion/result.json").read_text()).get("score")
                                                             if (ROOT / "champion/result.json").is_file() else None),
                    "heldout_relative_output_mse": base_mse,
                    "heldout_relative_routed_output_mse": base_routed}
    else:
        base_pred = None
        baseline = {"available": False, "reason": "baseline HYB1 files not found"}
    wg, wd, history = train(xtr, ytr, xv, yv, wtr, wv, args.hidden,
                            args.epochs, args.batch_size, args.lr, args.seed)
    rank, model_bytes, gpath, dpath = pack_student(wg, wd, args.out, provenance, args.hidden)
    pred, student_mse, student_routed = evaluate_packed(xv, yv, wv, gpath, dpath)
    distilled_control = {"available": False, "reason": "real routed teacher-QAT pair is not available yet"}
    qgate, qdown = args.teacher_qat / "gate-up.hyb", args.teacher_qat / "down.hyb"
    if qgate.is_file() and qdown.is_file():
        if has_full_expert_pair(qgate, qdown):
            _, q_mse, q_routed = evaluate_packed(xv, yv, wv, qgate, qdown)
            distilled_control = {"available": True, "files": [str(qgate), str(qdown)],
                "bytes": qgate.stat().st_size + qdown.stat().st_size,
                "heldout_relative_output_mse": q_mse,
                "heldout_relative_routed_output_mse": q_routed}
        else:
            distilled_control = {"available": False,
                "reason": "found teacher-QAT files do not match the 1536-wide expert; likely a synthetic fixture"}
    result = {"label": "SYNTHETIC SMOKE ONLY" if args.synthetic else "routed document-disjoint behavioral result",
              "objective": "standalone teacher expert-output MSE; route-weighted metric additionally reported",
              "input_width": WIDTH, "output_width": WIDTH, "student_intermediate": args.hidden,
              "dataset": str(args.dataset) if args.dataset else "generated synthetic fixture",
              "train_rows": len(xtr), "heldout_rows": len(xv), "provenance": provenance,
              "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
              "seed": args.seed, "blas_threads": 1,
              "packed_hyb_ranks": {"gate_up": rank, "down": rank},
              "student_model_bytes_including_metadata": model_bytes, "byte_cap": CAP,
              "baseline": baseline,
              "teacher_qat_packed_student": distilled_control,
              "student_heldout_relative_output_mse": student_mse,
              "student_heldout_relative_routed_output_mse": student_routed,
              "relative_output_mse_vs_baseline": (student_mse / max(base_mse, 1e-20) if base_pred is not None else None),
              "relative_routed_output_mse_vs_baseline": (student_routed / max(base_routed, 1e-20)
                    if base_pred is not None and student_routed is not None and base_routed is not None else None),
              "relative_output_mse_vs_teacher_qat": (student_mse / max(distilled_control["heldout_relative_output_mse"], 1e-20)
                    if distilled_control.get("available") else None),
              "relative_routed_output_mse_vs_teacher_qat": (student_routed / max(distilled_control["heldout_relative_routed_output_mse"], 1e-20)
                    if distilled_control.get("available") and student_routed is not None and distilled_control["heldout_relative_routed_output_mse"] is not None else None),
              "history": history}
    (args.out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
