#!/usr/bin/env python3
"""Bounded RF-guided vs uniform-random search for narrow SwiGLU students.

Both policies use the same six initial trials. Each then receives four new
trials from the same configuration space. Calibration documents are split by
document into train and tuning-validation; the data sidecar's heldout documents
are opened only after both policies have selected a winner.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import distill


def split_training_documents(data_path, x, y, route_weight, provenance, seed):
    docs = provenance.get("training_document_rows")
    if not docs:
        raise ValueError("real-data search requires training_document_rows in dataset metadata")
    nonempty = [d for d in docs if int(d["routed_rows"]) > 0]
    if len(nonempty) < 2:
        raise ValueError("need at least two calibration documents to reserve document-disjoint tuning validation")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(nonempty))
    nval = max(1, min(len(nonempty) - 1, round(.25 * len(nonempty))))
    val_docs = {nonempty[int(i)]["doc_id"] for i in perm[:nval]}
    train_ix, tune_ix = [], []
    cursor = 0
    for doc in docs:
        n = int(doc["routed_rows"])
        idx = list(range(cursor, cursor + n))
        (tune_ix if doc["doc_id"] in val_docs else train_ix).extend(idx)
        cursor += n
    if cursor != len(x):
        raise ValueError("document row metadata does not cover calibration arrays")
    if not train_ix or not tune_ix:
        raise ValueError("document split produced an empty train or tuning set")
    wi = route_weight
    def take(a, ids):
        return a[np.asarray(ids, dtype=np.int64)]
    split = {"split_unit": "document", "training_documents": [d["doc_id"] for d in docs if d["doc_id"] not in val_docs],
             "tuning_validation_documents": sorted(val_docs), "final_test_documents": provenance.get("heldout_documents", []),
             "training_rows": len(train_ix), "tuning_validation_rows": len(tune_ix),
             "final_test_rows": int(provenance.get("heldout_routed_rows", 0)), "seed": seed}
    return take(x, train_ix), take(y, train_ix), take(x, tune_ix), take(y, tune_ix), \
           (take(wi, train_ix) if wi is not None else None), (take(wi, tune_ix) if wi is not None else None), split


def load_explicit_tuning_dataset(path):
    """Load a shared three-way aggregate without deriving a competing split."""
    sidecar_path = path.with_suffix(".metadata.json")
    provenance = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
    with np.load(path, allow_pickle=False) as data:
        needed = ("x_train", "y_train", "x_tune", "y_tune", "x_val", "y_val")
        missing = [k for k in needed if k not in data]
        if missing:
            raise ValueError(f"explicit tuning arrays missing from aggregate: {missing}")
        xtr, ytr, xtune, ytune, xtest, ytest = [np.asarray(data[k], np.float32) for k in needed]
        wtr = np.asarray(data["route_weight_train"], np.float32) if "route_weight_train" in data else None
        wtune = np.asarray(data["route_weight_tune"], np.float32) if "route_weight_tune" in data else None
        wtest = np.asarray(data["route_weight_val"], np.float32) if "route_weight_val" in data else None
    if any(a.ndim != 2 for a in (xtr, ytr, xtune, ytune, xtest, ytest)):
        raise ValueError("expected flattened rank-two arrays for fit/tune/test")
    if any(a.shape[1] != distill.WIDTH for a in (xtr, ytr, xtune, ytune, xtest, ytest)):
        raise ValueError("fit/tune/test input and target widths must all be 1536")
    if len(xtr) != len(ytr) or len(xtune) != len(ytune) or len(xtest) != len(ytest):
        raise ValueError("fit/tune/test input and target counts differ")
    for weight, rows, name in ((wtr, xtr, "fit"), (wtune, xtune, "tune"), (wtest, xtest, "test")):
        if weight is not None and len(weight) != len(rows):
            raise ValueError(f"{name} route weights do not align with examples")
    if any(weight is not None for weight in (wtr, wtune, wtest)) and not all(
            weight is not None for weight in (wtr, wtune, wtest)):
        raise ValueError("route weights must be supplied for fit, tune and test together")
    docs = {k: provenance.get(k, []) for k in ("training_documents", "tuning_documents", "heldout_documents")}
    sets = [set(docs[k]) for k in docs]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("fit, tuning and final-test document IDs overlap")
    split = {"split_unit": provenance.get("split_unit", "document"),
             "training_documents": docs["training_documents"],
             "tuning_validation_documents": docs["tuning_documents"],
             "final_test_documents": docs["heldout_documents"],
             "training_rows": len(xtr), "tuning_validation_rows": len(xtune),
             "final_test_rows": len(xtest), "seed": provenance.get("split_seed")}
    if split["split_unit"] != "document":
        raise ValueError("shared fit/tune/test split must be document-disjoint")
    for docs_key, rows_key, array in (("training_document_rows", "training_rows", xtr),
                                      ("tuning_document_rows", "tuning_validation_rows", xtune),
                                      ("heldout_document_rows", "final_test_rows", xtest)):
        if docs_key in provenance and sum(int(d["routed_rows"]) for d in provenance[docs_key]) != len(array):
            raise ValueError(f"{docs_key} metadata does not cover its split array")
    return xtr, ytr, xtune, ytune, wtr, wtune, xtest, ytest, wtest, split, provenance


def synthetic_splits(seed):
    xtr, ytr, xtest, ytest, wtr, wtest, _ = distill.synthetic_data(seed)
    rng = np.random.default_rng(seed + 1)
    ix = rng.permutation(len(xtr))
    tune = ix[:max(1, len(ix) // 4)]
    train = ix[len(tune):]
    return (xtr[train], ytr[train], xtr[tune], ytr[tune], wtr[train], wtr[tune],
            xtest, ytest, wtest,
            {"split_unit": "synthetic-example", "training_documents": [],
             "tuning_validation_documents": [], "final_test_documents": [],
             "training_rows": len(train), "tuning_validation_rows": len(tune),
             "final_test_rows": len(xtest), "seed": seed, "label": "SYNTHETIC SMOKE ONLY"})


def synthetic_document_splits(seed):
    x, y, xtest, ytest, w, wtest, _ = distill.synthetic_data(seed)
    provenance = {"kind": "synthetic-document-fixture", "split_unit": "document",
        "training_document_rows": [{"doc_id": f"cal-{i}", "routed_rows": 32} for i in range(3)],
        "heldout_documents": ["test-a", "test-b"], "heldout_routed_rows": len(xtest)}
    xtr, ytr, xtune, ytune, wtr, wtune, split = split_training_documents(
        None, x, y, w, provenance, seed)
    split["label"] = "SYNTHETIC DOCUMENT-SPLIT PIPELINE SMOKE ONLY"
    return xtr, ytr, xtune, ytune, wtr, wtune, xtest, ytest, wtest, split, provenance


def rf_predict_and_disagree(configs, losses, candidates, seed, trees=48):
    """Tiny randomized CART forest using only NumPy; std across trees is disagreement."""
    rng = np.random.default_rng(seed)
    x = np.asarray([[c["hidden"] / 384, int(c["scheme"] == "hyb1-global-scale"),
                     math.log10(c["lr"])] for c in configs], dtype=np.float64)
    q = np.asarray([[c["hidden"] / 384, int(c["scheme"] == "hyb1-global-scale"),
                     math.log10(c["lr"])] for c in candidates], dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)

    def build(xb, yb, depth):
        if depth == 0 or len(yb) < 3 or np.ptp(yb) < 1e-12:
            return float(np.mean(yb))
        best = None
        features = rng.choice(x.shape[1], size=max(1, int(math.sqrt(x.shape[1]))), replace=False)
        for feature in features:
            vals = np.unique(xb[:, feature])
            if len(vals) < 2:
                continue
            for cut in rng.choice(vals[:-1] + (vals[1:] - vals[:-1]) / 2,
                                  size=min(8, len(vals) - 1), replace=False):
                left = xb[:, feature] <= cut
                if left.sum() < 2 or (~left).sum() < 2:
                    continue
                score = np.square(yb[left] - yb[left].mean()).sum() + np.square(yb[~left] - yb[~left].mean()).sum()
                if best is None or score < best[0]:
                    best = (score, feature, cut, left)
        if best is None:
            return float(np.mean(yb))
        _, feature, cut, left = best
        return (feature, cut, build(xb[left], yb[left], depth - 1), build(xb[~left], yb[~left], depth - 1))

    def predict(tree, row):
        while isinstance(tree, tuple):
            feature, cut, left, right = tree
            tree = left if row[feature] <= cut else right
        return tree

    preds = []
    for _ in range(trees):
        indices = rng.integers(0, len(y), size=len(y))
        tree = build(x[indices], y[indices], 3)
        preds.append([predict(tree, row) for row in q])
    pred = np.asarray(preds)
    return pred.mean(axis=0), pred.std(axis=0)


def run_trial(config, index, root, train_data, tune_data, epochs, batch, seed, data_sha):
    xtr, ytr, wtr = train_data
    xtune, ytune, wtune = tune_data
    # Equalize the dominant SwiGLU training MAC proxy across widths.
    reference_steps = epochs * math.ceil(len(xtr) / batch)
    reference_examples = reference_steps * min(batch, len(xtr))
    training_examples = max(1, round(reference_examples * 256 / config["hidden"]))
    steps = math.ceil(training_examples / batch)
    wg, wd, history = distill.train(xtr, ytr, xtune, ytune, wtr, wtune,
                                    config["hidden"], 1, batch, config["lr"], seed,
                                    training_examples=training_examples)
    out = root / f"trial-{index:02d}"
    rank, nbytes, gp, dp = distill.pack_student(wg, wd, out, {"kind": "search-trial", "dataset_sha256": data_sha},
                                                config["hidden"], config["scheme"], config["sign_sweeps"])
    pred, loss, routed = distill.evaluate_packed(xtune, ytune, wtune, gp, dp)
    result = {"config": config, "seed": seed, "reference_epochs_at_hidden_256": epochs,
              "training_steps": steps, "training_examples": training_examples,
              "training_compute_proxy_examples_x_hidden": training_examples * config["hidden"],
              "batch_size": batch,
              "tuning_validation_relative_output_mse": loss,
              "tuning_validation_relative_routed_output_mse": routed,
              "actual_model_bytes_including_metadata": nbytes, "hyb_rank": rank,
              "history": history, "artifact_dir": str(out)}
    (out / "search-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def evaluate_winner(result, test_x, test_y, test_w):
    path = Path(result["artifact_dir"])
    _, mse, routed = distill.evaluate_packed(test_x, test_y, test_w, path / "gate-up.hyb", path / "down.hyb")
    result["final_test_relative_output_mse"] = mse
    result["final_test_relative_routed_output_mse"] = routed


def model_pair_has_full_expert_shapes(gate_path, down_path):
    try:
        gm, gn = distill.compress.HEADER.unpack_from(gate_path.read_bytes())[1:3]
        dm, dn = distill.compress.HEADER.unpack_from(down_path.read_bytes())[1:3]
        return (gm, gn, dm, dn) == (2 * distill.WIDTH, distill.WIDTH,
                                     distill.WIDTH, distill.WIDTH)
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path, nargs="?")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-documents", action="store_true",
                    help="exercise the real document-disjoint split path using generated data")
    ap.add_argument("--out", type=Path, default=HERE / "search-results")
    ap.add_argument("--initial-trials", type=int, default=6)
    ap.add_argument("--additional-trials", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=811)
    ap.add_argument("--baseline-gate", type=Path)
    ap.add_argument("--baseline-down", type=Path)
    ap.add_argument("--teacher-qat", type=Path, default=ROOT / "experiments/teacher-qat/qat-routed")
    args = ap.parse_args()
    if sum((args.synthetic, args.synthetic_documents, args.dataset is not None)) != 1:
        ap.error("provide exactly one dataset, --synthetic, or --synthetic-documents")
    if args.baseline_gate is None or args.baseline_down is None:
        manifest_path = ROOT / "champion/result.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            projections = manifest.get("projections", {})
            args.baseline_gate = args.baseline_gate or ROOT / "champion" / projections.get("gate-up", {}).get("file", "gate-up.hyb")
            args.baseline_down = args.baseline_down or ROOT / "champion" / projections.get("down", {}).get("file", "down.hyb")
    if args.synthetic:
        xtr, ytr, xtune, ytune, wtr, wtune, xtest, ytest, wtest, split = synthetic_splits(args.seed)
        provenance = {"kind": "synthetic-smoke-only"}
        data_sha = None
    elif args.synthetic_documents:
        xtr, ytr, xtune, ytune, wtr, wtune, xtest, ytest, wtest, split, provenance = synthetic_document_splits(args.seed)
        data_sha = None
    else:
        with np.load(args.dataset, allow_pickle=False) as probe:
            explicit_tune = "x_tune" in probe and "y_tune" in probe
        if explicit_tune:
            xtr, ytr, xtune, ytune, wtr, wtune, xtest, ytest, wtest, split, provenance = load_explicit_tuning_dataset(args.dataset)
        else:
            x, y, xtest, ytest, wtr_all, wtest, provenance = distill.load_data(args.dataset)
            xtr, ytr, xtune, ytune, wtr, wtune, split = split_training_documents(
                args.dataset, x, y, wtr_all, provenance, args.seed)
        data_sha = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    configs = [dict(hidden=h, scheme=q, lr=lr, sign_sweeps=sw)
               for h, q, lr, sw in itertools.product((128, 192, 256, 320, 384),
                   ("hyb1-row-scale", "hyb1-global-scale"), (3e-4, 1e-3, 3e-3), (0, 2))]
    rng = np.random.default_rng(args.seed)
    initial_ids = rng.choice(len(configs), size=min(args.initial_trials, len(configs)), replace=False).tolist()
    initial = [configs[i] for i in initial_ids]
    trial_root = args.out / "trials"
    trial_root.mkdir(parents=True, exist_ok=True)
    trained = {}
    run_index = 0
    for config in initial:
        key = json.dumps(config, sort_keys=True)
        trained[key] = run_trial(config, run_index, trial_root,
            (xtr, ytr, wtr), (xtune, ytune, wtune), args.epochs, args.batch_size,
            args.seed + 1000 + run_index, data_sha)
        run_index += 1

    remaining = [c for c in configs if json.dumps(c, sort_keys=True) not in trained]
    random_additions = []
    random_pool = list(remaining)
    for _ in range(min(args.additional_trials, len(random_pool))):
        pick = int(rng.integers(len(random_pool)))
        random_additions.append(random_pool.pop(pick))
    random_policy = list(initial)
    for config in random_additions:
        key = json.dumps(config, sort_keys=True)
        if key not in trained:
            trained[key] = run_trial(config, run_index, trial_root,
                (xtr, ytr, wtr), (xtune, ytune, wtune), args.epochs, args.batch_size,
                args.seed + 1000 + run_index, data_sha)
            run_index += 1
        random_policy.append(config)

    rf_policy = list(initial)
    rf_additions = []
    rf_pool = list(remaining)
    for _ in range(min(args.additional_trials, len(rf_pool))):
        # The RF policy learns only from the shared seeds and its own prior proposals.
        seen = list(rf_policy)
        results = [trained[json.dumps(c, sort_keys=True)] for c in seen]
        means, disagreements = rf_predict_and_disagree(seen,
            [r["tuning_validation_relative_routed_output_mse"] or r["tuning_validation_relative_output_mse"] for r in results],
            rf_pool, args.seed + 2000 + len(rf_additions))
        # Minimize predicted error while rewarding tree disagreement; scale uncertainty to the observed objective range.
        spread = max(1e-12, float(np.std([r["tuning_validation_relative_output_mse"] for r in results])))
        acquisition = means - .5 * spread * disagreements
        choice = int(np.argmin(acquisition))
        config = rf_pool.pop(choice)
        rf_additions.append({"config": config, "predicted_error": float(means[choice]),
                             "tree_disagreement": float(disagreements[choice]),
                             "acquisition": float(acquisition[choice]), "objective_scale": spread})
        key = json.dumps(config, sort_keys=True)
        if key not in trained:
            trained[key] = run_trial(config, run_index, trial_root,
                (xtr, ytr, wtr), (xtune, ytune, wtune), args.epochs, args.batch_size,
                args.seed + 1000 + run_index, data_sha)
            run_index += 1
        rf_policy.append(config)

    policies = {}
    for name, configs_for_policy in (("random", random_policy), ("random_forest_guided", rf_policy)):
        runs = [trained[json.dumps(c, sort_keys=True)] for c in configs_for_policy]
        winner = min(runs, key=lambda r: (r["tuning_validation_relative_routed_output_mse"]
                                          if r["tuning_validation_relative_routed_output_mse"] is not None
                                          else r["tuning_validation_relative_output_mse"]))
        # The final test is accessed only now, after policy winners are fixed.
        evaluate_winner(winner, xtest, ytest, wtest)
        policies[name] = {"trial_count_including_shared_initial": len(runs),
                          "selected_tuning_winner": winner}

    # Evaluate current packed baseline and the teacher-QAT packed control only on the final test split.
    controls = {}
    if args.baseline_gate.is_file() and args.baseline_down.is_file():
        _, mse, routed = distill.evaluate_packed(xtest, ytest, wtest, args.baseline_gate, args.baseline_down)
        controls["current_packed_baseline"] = {"files": [str(args.baseline_gate), str(args.baseline_down)],
            "bytes": args.baseline_gate.stat().st_size + args.baseline_down.stat().st_size,
            "format": args.baseline_gate.read_bytes()[:4].decode("ascii", "replace"),
            "published_raw_weight_score_separate": (json.loads((ROOT / "champion/result.json").read_text()).get("score")
                                                     if (ROOT / "champion/result.json").is_file() else None),
            "final_test_relative_output_mse": mse, "final_test_relative_routed_output_mse": routed}
    if (args.teacher_qat / "gate-up.hyb").is_file() and (args.teacher_qat / "down.hyb").is_file():
        qgate, qdown = args.teacher_qat / "gate-up.hyb", args.teacher_qat / "down.hyb"
        if model_pair_has_full_expert_shapes(qgate, qdown):
            _, mse, routed = distill.evaluate_packed(xtest, ytest, wtest, qgate, qdown)
            controls["teacher_qat_packed_student"] = {"files": [str(qgate), str(qdown)],
                "bytes": qgate.stat().st_size + qdown.stat().st_size,
                "final_test_relative_output_mse": mse, "final_test_relative_routed_output_mse": routed}
        else:
            controls["teacher_qat_packed_student"] = {"available": False,
                "reason": "found artifact dimensions do not match the 1536-wide expert; likely a synthetic fixture"}
    else:
        controls["teacher_qat_packed_student"] = {"available": False,
            "reason": "real routed teacher-QAT pair is not available yet"}
    for policy in policies.values():
        winner = policy["selected_tuning_winner"]
        for control_name, control in controls.items():
            if control.get("available") is False:
                continue
            control_mse = control["final_test_relative_output_mse"]
            control_routed = control["final_test_relative_routed_output_mse"]
            winner.setdefault("relative_output_mse_vs_controls", {})[control_name] = (
                winner["final_test_relative_output_mse"] / max(control_mse, 1e-20))
            if winner["final_test_relative_routed_output_mse"] is not None and control_routed is not None:
                winner.setdefault("relative_routed_output_mse_vs_controls", {})[control_name] = (
                    winner["final_test_relative_routed_output_mse"] / max(control_routed, 1e-20))
    synthetic_label = args.synthetic or args.synthetic_documents or "synthetic" in str(provenance.get("kind", "")).lower()
    synthetic_document_label = args.synthetic_documents or (synthetic_label and split.get("split_unit") == "document")
    result = {"label": ("SYNTHETIC DOCUMENT-SPLIT PIPELINE SMOKE ONLY" if synthetic_document_label else
                        "SYNTHETIC SMOKE ONLY" if synthetic_label else
                        "document-disjoint behavioral HPO; raw-weight scoreboard excluded"),
        "search": {"initial_shared_random_trials": len(initial), "additional_trials_per_policy": args.additional_trials,
                   "reference_epochs_at_hidden_256": args.epochs, "batch_size": args.batch_size,
                   "equalized_training_compute_proxy": "training examples * hidden width (rounding error under 0.5 hidden-width units)",
                   "space": {"hidden": [128, 192, 256, 320, 384],
                             "quantization_scheme": ["hyb1-row-scale", "hyb1-global-scale"],
                             "learning_rate": [3e-4, 1e-3, 3e-3], "sign_sweeps": [0, 2]},
                   "rf": {"trees": 48, "max_depth": 3, "acquisition": "predicted routed error - 0.5 * tree disagreement"},
                   "rf_guided_choices": rf_additions,
                   "compute_runs": run_index, "shared_initial_trials_reused": True},
        "document_split": split, "data_sha256": data_sha, "controls": controls, "policies": policies,
        "blas_threads": 1}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "search-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
