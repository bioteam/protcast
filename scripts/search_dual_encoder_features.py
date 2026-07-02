"""search_dual_encoder_features.py

Per-level feature search for the ORDER-embedding dual encoder.

Motivation
----------
The KNN and flat-NN comparisons show classical FVs (esp. PseKRAAC) boost ESM-C
at *every* GO level, with the best companion descriptor changing by depth
(PseKRAAC_3B at L5-6, PseKRAAC_10 at L7, ...). The order dual-encoder, by
contrast, has so far only clearly benefited at L6 — but it was always fed a
single *fixed* feature set (type_7 + type_3B + type_8). This harness tests
whether giving the order dual-encoder the per-level-optimal descriptor unlocks
the all-level gains the other architectures already enjoy.

Efficiency
----------
Only the dual arm (Arm B) depends on the feature choice; the ESM-only order NN
baseline (Arm A) is feature-invariant. So we train Arm A **once** on a fixed
protein set, then loop candidate feature sets training only Arm B and compare
each to that single baseline. For an N-descriptor search this is ~(N+1) trains
instead of ~2N — the saving grows with N.

Method notes
------------
* All arms train on one global protein set = ESM ∩ annotations ∩ (FVs valid
  under *every* candidate set), so Arm A and every Arm B share the exact same
  train/val split (seeded). Any ΔFmax is attributable to the feature block.
  The count of proteins dropped to enforce this intersection is logged.
* This is a *ranking* tool. The winning descriptor should then be confirmed
  with a rigorous per-set run (its own maximal intersection) via
  compare_knn_vs_multilabel.py --order --dual-encoder --feature_algorithms ...
* Per-set model/encoder files are overwritten between iterations (the dual arm
  reuses one id); only the metrics — captured in the results JSON — are kept.

Example
-------
python3 scripts/search_dual_encoder_features.py \\
    -d mf_go_terms-level-7 \\
    -p ProtCastDataset.bin \\
    -o dual_feature_search \\
    --feature-sets PseKRAAC_type_3B PseKRAAC_type_10 PseKRAAC_type_7 \\
                   PseKRAAC_type_7+PseKRAAC_type_3B+PseKRAAC_type_8 \\
    --seed 42 -v
"""

import os

# TF determinism must be set before TF is imported (transitively, below).
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import re
import gc
import json
import time
import random
import argparse
import importlib.util
import pathlib

import numpy as np

from protcast.preprocessing.protcast_dataset import ProtCastDataset
from protcast.config.model_config import ConfigManager


def _load_sibling(mod_name):
    """Load a sibling script in scripts/ by file path.

    Mirrors the importlib pattern compare_knn_vs_multilabel.py already uses, so
    we don't depend on scripts/ being an importable package.
    """
    spec = importlib.util.spec_from_file_location(
        mod_name, pathlib.Path(__file__).parent / f"{mod_name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_feature_sets(specs):
    """Turn ["A", "A+B"] into [("A", ["A"]), ("A+B", ["A", "B"])]."""
    parsed = []
    for spec in specs:
        algos = [a for a in spec.split("+") if a]
        parsed.append((spec, algos))
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Search classical feature descriptors for the ORDER dual-encoder: "
            "train the ESM-only order NN once, then rank candidate feature sets "
            "by the dual-encoder ΔFmax they produce over that baseline."
        )
    )
    parser.add_argument("-d", "--input_dir", required=True,
                        help="Directory of pre-computed ESM embedding .pkl files")
    parser.add_argument("-p", "--protcast_dataset", required=True,
                        help="Path to serialised ProtCastDataset (.bin)")
    parser.add_argument("--feature-sets", nargs="+", default=[
                            "PseKRAAC_type_3B",
                            "PseKRAAC_type_10",
                            "PseKRAAC_type_7",
                            "PseKRAAC_type_8",
                            "PseKRAAC_type_7+PseKRAAC_type_3B+PseKRAAC_type_8",
                        ],
                        help=("Candidate descriptor sets; combine algorithms with '+'. "
                              "The '+'-joined default reproduces the current fixed set "
                              "as a reference point."))
    parser.add_argument("--order-variant", choices=["soft", "hard"], default="soft",
                        help="Order-violation loss variant for all arms (default soft)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the train/val split (default 42)")
    parser.add_argument("-o", "--output_dir", default="dual_feature_search",
                        help="Directory for output files")
    # Opt-in architecture / optimisation overrides (applied to every arm).
    parser.add_argument("--fv-hidden", type=int, default=None)
    parser.add_argument("--fv-dropout", type=float, default=None)
    parser.add_argument("--gated-fusion", action="store_true")
    parser.add_argument("--order-weight", type=float, default=None)
    parser.add_argument("--order-dim", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lr-schedule", choices=["cosine"], default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=None)
    parser.add_argument("--use_mlflow", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    config = ConfigManager.load_config()

    # Same opt-in override mechanism as compare_knn_vs_multilabel.py.
    _overrides = {
        "FV_HIDDEN": args.fv_hidden,
        "FV_DROPOUT": args.fv_dropout,
        "ORDER_WEIGHT": args.order_weight,
        "ORDER_DIM": args.order_dim,
        "LEARNING_RATE": args.learning_rate,
        "PATIENCE": args.patience,
        "MIN_DELTA": args.min_delta,
    }
    for k, v in _overrides.items():
        if v is not None:
            config[k] = v
    if args.gated_fusion:
        config["GATED_FUSION"] = True
    if args.lr_schedule:
        config["LR_SCHEDULE"] = args.lr_schedule

    start = time.time()

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)
    except Exception:
        pass

    input_dir = os.path.abspath(args.input_dir)
    protcast_dataset = os.path.abspath(args.protcast_dataset)
    name = os.path.basename(input_dir.rstrip("/"))
    level_match = re.search(r"level-(\d+)", name)
    level = int(level_match.group(1)) if level_match else None

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    results_file = f"{name}_dual_feature_search_results.json"

    # ── Sibling helpers ─────────────────────────────────────────────────────
    cmp = _load_sibling("compare_knn_vs_multilabel")
    knn_esm = _load_sibling("compare_knn_esm_vs_knn_combined")
    load_flat_embeddings = cmp.load_flat_embeddings
    train_multilabel = cmp.train_multilabel
    compute_classical_feature_vectors = knn_esm.compute_classical_feature_vectors

    feature_sets = parse_feature_sets(args.feature_sets)

    # ── Resume support ──────────────────────────────────────────────────────
    results = None
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
    if results is None:
        results = {
            "seed": args.seed,
            "level": level,
            "order_variant": args.order_variant,
            "baseline_esm_order": {},
            "feature_sets": {},
        }
    results.setdefault("feature_sets", {})

    # ── Load data ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    dataset = ProtCastDataset.load_serialized_file(protcast_dataset)
    protein_embeddings, protein_go_terms, go_ids = load_flat_embeddings(
        input_dir, args.verbose
    )
    if not protein_embeddings:
        print(f"Error: no embeddings loaded from {input_dir}")
        return
    go_dag = getattr(dataset, "annotated_dag", None)
    if go_dag is None:
        print("Warning: no annotated_dag — order-violation loss will have zero "
              "edges (order arms degenerate to flat).")

    sequences = {
        pid: dataset.proteins[pid].sequence
        for pid in protein_embeddings
        if pid in dataset.proteins
    }

    # ── Compute FVs for every candidate set, then fix a global protein set ──
    print("\nComputing feature vectors for all candidate sets...")
    fv_by_set = {}
    for set_name, algos in feature_sets:
        try:
            fv_by_set[set_name] = compute_classical_feature_vectors(
                sequences, algos, verbose=args.verbose
            )
            dim = next(iter(fv_by_set[set_name].values())).shape[0]
            print(f"  {set_name}: dim={dim}, proteins={len(fv_by_set[set_name])}")
        except Exception as e:
            print(f"  {set_name}: FAILED to compute ({e}) — skipping")
            results["feature_sets"][set_name] = {"status": f"error: fv compute: {e}"}

    live_sets = [(n, a) for (n, a) in feature_sets if n in fv_by_set]
    if not live_sets:
        print("Error: no candidate feature set produced valid vectors")
        return

    # Global protein set shared by the baseline and every dual arm.
    global_common = set(protein_embeddings) & set(protein_go_terms)
    for set_name, _ in live_sets:
        global_common &= set(fv_by_set[set_name])
    global_common = sorted(global_common)
    if not global_common:
        print("Error: empty intersection across ESM / annotations / all FV sets")
        return

    dropped = len(protein_embeddings) - len(global_common)
    print(f"\nGlobal protein set (baseline + all dual arms): {len(global_common)} "
          f"(dropped {dropped} to keep every arm on identical proteins)")

    base_embeddings = {p: protein_embeddings[p] for p in global_common}
    base_go_terms = {p: protein_go_terms[p] for p in global_common}
    results["esm_dim"] = int(next(iter(base_embeddings.values())).shape[0])
    results["n_proteins"] = len(global_common)
    results["n_proteins_dropped"] = dropped

    # ── Arm A: ESM-only order NN baseline (trained once) ────────────────────
    if results.get("baseline_esm_order", {}).get("status") != "ok":
        print("\n" + "=" * 60)
        print(f"BASELINE: ESM-only ORDER NN ({args.order_variant})")
        print("=" * 60)
        try:
            results["baseline_esm_order"] = train_multilabel(
                base_embeddings, base_go_terms, go_ids,
                config, name, args.seed, go_dag,
                use_order=True, use_mlflow=args.use_mlflow, verbose=args.verbose,
                order_variant=args.order_variant,
            )
            b = results["baseline_esm_order"]
            print(f"Baseline — Fmax: {b['fmax']:.4f}  Smin: {b['smin']:.4f}  "
                  f"Epochs: {b['epochs']}")
        except Exception as e:
            print(f"FAILED: baseline — {e}")
            results["baseline_esm_order"] = {"status": f"error: {e}"}
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    baseline = results.get("baseline_esm_order", {})
    base_fmax = baseline.get("fmax")
    base_smin = baseline.get("smin")

    # ── Arm B: dual-encoder, one per candidate feature set ──────────────────
    for i, (set_name, algos) in enumerate(live_sets, 1):
        if results["feature_sets"].get(set_name, {}).get("status") == "ok":
            print(f"\n[{i}/{len(live_sets)}] {set_name}: already done, skipping")
            continue

        print("\n" + "=" * 60)
        print(f"[{i}/{len(live_sets)}] DUAL-ENCODER + {set_name}")
        print("=" * 60)
        fv_set = {p: fv_by_set[set_name][p] for p in global_common}
        try:
            res = train_multilabel(
                base_embeddings, base_go_terms, go_ids,
                config, name, args.seed, go_dag,
                use_order=True, use_mlflow=args.use_mlflow, verbose=args.verbose,
                order_variant=args.order_variant,
                use_dual_encoder=True, fv_embeddings=fv_set,
            )
            res["fv_dim"] = int(next(iter(fv_set.values())).shape[0])
            res["algorithms"] = algos
            if base_fmax is not None and res.get("fmax") is not None:
                res["delta_fmax"] = float(res["fmax"]) - float(base_fmax)
            if base_smin is not None and res.get("smin") is not None:
                res["delta_smin"] = float(res["smin"]) - float(base_smin)
            results["feature_sets"][set_name] = res
            d = res.get("delta_fmax")
            print(f"{set_name} — Fmax: {res['fmax']:.4f}  "
                  f"ΔFmax vs baseline: {d:+.4f}" if d is not None
                  else f"{set_name} — Fmax: {res['fmax']:.4f}")
        except Exception as e:
            print(f"FAILED: {set_name} — {e}")
            results["feature_sets"][set_name] = {"status": f"error: {e}"}

        results["elapsed"] = round(time.time() - start)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        gc.collect()

    # ── Ranking table ───────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"DUAL-ENCODER FEATURE SEARCH — {name} (seed {args.seed})")
    print("=" * 78)
    print(f"ESM-only order baseline Fmax: {_fmt(base_fmax)}  Smin: {_fmt(base_smin)}")
    print("-" * 78)
    print(f"{'Feature set':<44} {'dim':>5} {'Fmax':>8} {'ΔFmax':>9} {'ΔSmin':>9}")
    print("-" * 78)
    ranked = sorted(
        ((n, r) for n, r in results["feature_sets"].items() if r.get("status") == "ok"),
        key=lambda kv: kv[1].get("delta_fmax", float("-inf")),
        reverse=True,
    )
    for set_name, r in ranked:
        print(f"{set_name:<44} {r.get('fv_dim', '?'):>5} "
              f"{_fmt(r.get('fmax')):>8} {_signed(r.get('delta_fmax')):>9} "
              f"{_signed(r.get('delta_smin')):>9}")
    failed = [n for n, r in results["feature_sets"].items() if r.get("status") != "ok"]
    if failed:
        print(f"\n(failed/skipped: {', '.join(failed)})")
    print("=" * 78)
    print(f"\nResults saved to {results_file}")


def _fmt(v):
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "---"


def _signed(v):
    try:
        return f"{float(v):+.4f}"
    except (TypeError, ValueError):
        return "---"


if __name__ == "__main__":
    main()
