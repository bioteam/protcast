"""compare_knn_vs_multilabel.py

Three-way CAFA comparison of protein function prediction approaches using
pre-computed ESM-C embeddings:

  1. KNN          – nearest-neighbour voting over raw ESM-C embedding space
  2. Flat NN      – multi-label sigmoid neural network trained on ESM-C embeddings
  3. Box NN       – same network with BoxEmbeddingLayer + GO-DAG containment
                    loss, enforcing ontological hierarchy (--box flag required)

Primary question: does the extra computational cost of the neural approaches
yield a meaningful Fmax gain over simple KNN retrieval—especially on specific
(deep) GO terms where training data is sparse?

All models are trained on the same train/val split (controlled by --seed) and
evaluated with CAFA-standard Fmax and Smin.  Results are broken down by GO term
depth and annotation frequency to surface where each model wins or loses.

Inputs (not modified):
    - Pre-computed ESM embeddings (.pkl files) in the -d/--input_dir directory.
    - Serialized ProtCastDataset (.bin file) containing protein sequences
      and GO annotations.

Saved to the output directory (-o/--output_dir, default: knn_vs_multilabel):
    - {name}_knn_vs_multilabel_results.json   All results (updated after each model)
    - KNN model:   {name}_knn_comparison_knn.joblib
    - Flat NN:     {name}_multilabel_flat_comparison_multilabel.keras
    - Box NN:      {name}_multilabel_box_comparison_multilabel.keras  (--box only)
    - GOEncoder files for each neural model

    If the results JSON already exists with all requested models completed, the
    script loads and prints without retraining.  Individual models already present
    are skipped on re-run (allows resuming interrupted runs).

Example usage:

python3 scripts/compare_knn_vs_multilabel.py \\
    -d mf_go_terms-level-8 \\
    -p ProtcastDataset.bin \\
    -o knn_vs_multilabel \\
    --seed 42 \\
    -v

python3 scripts/compare_knn_vs_multilabel.py \\
    -d mf_go_terms-level-8 \\
    -p ProtcastDataset.bin \\
    --box \\
    --use_mlflow \\
    --seed 42 \\
    -v
"""

import os

# Enable TensorFlow determinism BEFORE any TF import (which happens transitively
# below via the classifier modules).  Together these flags make GPU training
# deterministic for a given seed: identical Fmax across re-runs of the same
# seed.  This eliminates within-seed (training) variance — so a single run per
# seed is statistically sufficient, and all observed variance is attributable
# to the train/val split (between-seed variance).  Costs ~5–10% throughput.
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import re
import gc
import json
import time
import pickle
import random
import argparse
from collections import defaultdict

import numpy as np

from protcast.model.knn_classifier import KNNClassifier
from protcast.model.multilabel_classifier import MultiLabelClassifier
from protcast.model.stats.utils import calculate_fmax, calculate_smin
from protcast.preprocessing.protcast_dataset import ProtCastDataset
from protcast.config.model_config import ConfigManager


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_flat_embeddings(input_dir, verbose=False):
    """Load pre-computed ESM embeddings into flat protein-centric dicts.

    Returns
    -------
    protein_embeddings : dict
        {protein_id: np.ndarray} mapping each protein to its ESM embedding.
    protein_go_terms : dict
        {protein_id: set[str]} mapping each protein to its GO term annotations.
    go_ids : list[str]
        Ordered list of GO term IDs found in the embedding files.
    """
    protein_embeddings = {}
    protein_go_terms = defaultdict(set)
    go_ids = []
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".pkl"):
            continue
        match = re.match(r"(GO_\d+)", filename)
        if not match:
            continue
        # Restore canonical GO:XXXXXXX format — filenames use underscores
        # (GO_0008150.pkl) but the GO DAG and annotations use colons.
        go_id = match.group(1).replace("_", ":", 1)
        go_ids.append(go_id)
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "rb") as f:
            embeddings_dict = pickle.load(f)
        for pid, embedding in embeddings_dict.items():
            if pid not in protein_embeddings:
                protein_embeddings[pid] = embedding
            protein_go_terms[pid].add(go_id)
    if verbose:
        print(f"Loaded {len(protein_embeddings)} proteins, {len(go_ids)} GO terms")
    return protein_embeddings, dict(protein_go_terms), go_ids


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_knn(
    protein_embeddings, protein_go_terms, go_ids,
    config, name, seed, go_dag, use_mlflow, verbose=False,
):
    """Fit KNNClassifier and return a serialisable result dict."""
    classifier = KNNClassifier(
        verbose=verbose,
        protein_embeddings=protein_embeddings,
        protein_go_terms=protein_go_terms,
        go_ids=go_ids,
        config=config,
        id=f"{name}_knn_comparison",
        use_mlflow=use_mlflow,
        go_dag=go_dag,
        random_state=seed,
    )
    classifier.run()

    depth_metrics = classifier.compute_depth_metrics(
        classifier.y_val, classifier.y_val_pred
    )
    freq_metrics = classifier.compute_frequency_metrics(
        classifier.y_val, classifier.y_val_pred
    )

    # Always save locally so the model can be used for inference later.
    # When use_mlflow=True, log_model() inside run() also uploads to MLflow.
    classifier.save_model()

    result = {
        "fmax": float(classifier.best_fmax),
        "fmax_threshold": float(classifier.best_threshold),
        "smin": float(classifier.best_smin),
        "smin_threshold": float(classifier.smin_threshold),
        "training_time": round(classifier.training_time, 2),
        "depth_metrics": {str(k): v for k, v in depth_metrics.items()},
        "frequency_metrics": freq_metrics,
        "status": "ok",
    }
    del classifier
    gc.collect()
    return result


def train_multilabel(
    protein_embeddings, protein_go_terms, go_ids,
    config, name, seed, go_dag, use_box, use_mlflow, verbose=False,
):
    """Fit MultiLabelClassifier (flat or box) and return a serialisable result dict.

    Parameters
    ----------
    use_box : bool
        When True, overrides USE_BOX_EMBEDDINGS in config so a copy of config
        is used for this run—config.json on disk is never modified.
    """
    label = "box" if use_box else "flat"

    # Work from a config copy so the flag doesn't persist between runs.
    run_config = dict(config)
    run_config["USE_BOX_EMBEDDINGS"] = use_box

    classifier = MultiLabelClassifier(
        verbose=verbose,
        protein_embeddings=protein_embeddings,
        protein_go_terms=protein_go_terms,
        go_ids=go_ids,
        config=run_config,
        id=f"{name}_multilabel_{label}_comparison",
        use_mlflow=use_mlflow,
        go_dag=go_dag,
        random_state=seed,
    )
    classifier.run()

    # Predict on the held-out validation set once; reuse for all metrics.
    y_pred = classifier.model.predict(classifier.X_val, verbose=0)
    fmax, fmax_threshold = calculate_fmax(classifier.y_val, y_pred)
    smin, smin_threshold = calculate_smin(classifier.y_val, y_pred)

    depth_metrics = classifier.compute_depth_metrics(classifier.y_val, y_pred)
    freq_metrics = classifier.compute_frequency_metrics(classifier.y_val, y_pred)

    # Save locally for inference (MLflow upload happens inside run() if enabled).
    classifier.save_model()

    result = {
        "fmax": float(fmax),
        "fmax_threshold": float(fmax_threshold),
        "smin": float(smin),
        "smin_threshold": float(smin_threshold),
        "best_loss": float(min(classifier.history.history["val_loss"])),
        "epochs": len(classifier.history.history["loss"]),
        "training_time": round(classifier.training_time, 2),
        "depth_metrics": {str(k): v for k, v in depth_metrics.items()},
        "frequency_metrics": freq_metrics,
        "status": "ok",
    }
    del classifier
    gc.collect()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Results display
# ──────────────────────────────────────────────────────────────────────────────

def _nan_safe(value, fmt=".4f"):
    """Format a float, returning '---' for NaN or missing values."""
    try:
        if np.isnan(float(value)):
            return "---"
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return "---"


def _delta_str(model_fmax, knn_fmax):
    """Format Fmax delta relative to KNN (+/- sign)."""
    try:
        d = float(model_fmax) - float(knn_fmax)
        return f"{d:>+9.4f}"
    except (TypeError, ValueError):
        return f"{'---':>9}"


def print_results(results):
    """Print three-section comparison: overall, depth breakdown, frequency breakdown."""
    # Derive has_box from actual data, not the run_box flag — the flag can be
    # stale if --box was added on a resume run of a previously flag-less run.
    has_box = results.get("multilabel_box", {}).get("status") == "ok"

    knn  = results.get("knn", {})
    flat = results.get("multilabel_flat", {})
    box  = results.get("multilabel_box", {}) if has_box else {}

    knn_fmax = knn.get("fmax")

    sep = "=" * (104 if has_box else 88)
    thin = "-" * (104 if has_box else 88)

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + sep)
    print("KNN vs MULTILABEL COMPARISON  (CAFA Metrics)")
    print(sep)
    print(
        f"Level : {results.get('level', '?')}   "
        f"Seed  : {results['seed']}   "
        f"ESM dim : {results['esm_dim']}   "
        f"Box embeddings : {'yes' if has_box else 'no'}"
    )

    # ── Section 1: Overall ────────────────────────────────────────────────────
    print()
    print("── OVERALL METRICS ──")
    if has_box:
        hdr = f"{'Model':<22} {'Fmax':>8} {'Thr':>6} {'Smin':>8} {'Epochs':>7} {'Time':>8}   {'Flat-KNN':>9}  {'Box-KNN':>9}"
    else:
        hdr = f"{'Model':<22} {'Fmax':>8} {'Thr':>6} {'Smin':>8} {'Epochs':>7} {'Time':>8}   {'Flat-KNN':>9}"
    print(hdr)
    print(thin)

    def _overall_row(label, r, is_knn=False):
        fmax = _nan_safe(r.get("fmax"))
        thr  = _nan_safe(r.get("fmax_threshold"), ".2f")
        smin = _nan_safe(r.get("smin"))
        t    = _nan_safe(r.get("training_time"), ".1f")
        ep   = r.get("epochs")
        ep_s = f"{ep:>7d}" if isinstance(ep, int) else f"{'---':>7}"

        if is_knn:
            deltas = f"{'---':>9}"
            if has_box:
                deltas += f"  {'---':>9}"
        else:
            deltas = _delta_str(r.get("fmax"), knn_fmax)
            if has_box:
                # second delta only makes sense for box row
                if r is box:
                    deltas = f"{'---':>9}  " + _delta_str(r.get("fmax"), knn_fmax)
                else:
                    deltas += f"  {'---':>9}"

        print(f"{label:<22} {fmax:>8} {thr:>6} {smin:>8} {ep_s} {t:>7}s   {deltas}")

    if knn:
        _overall_row("KNN", knn, is_knn=True)
    if flat:
        _overall_row("MultiLabel flat", flat)
    if box:
        _overall_row("MultiLabel + box", box)

    # ── Section 2: Depth breakdown ────────────────────────────────────────────
    print()
    print("── DEPTH BREAKDOWN  (higher depth = more specific GO terms) ──")

    all_depths = set()
    for r in [knn, flat, box]:
        if r:
            all_depths.update(int(d) for d in r.get("depth_metrics", {}))

    if all_depths:
        if has_box:
            hdr = (f"{'Depth':>5}  {'N terms':>7}  {'Avg ann':>8}  "
                   f"{'KNN':>8}  {'Flat':>8}  {'Box':>8}  {'Flat-KNN':>9}  {'Box-KNN':>9}")
        else:
            hdr = (f"{'Depth':>5}  {'N terms':>7}  {'Avg ann':>8}  "
                   f"{'KNN':>8}  {'Flat':>8}  {'Flat-KNN':>9}")
        print(hdr)
        print("-" * len(hdr))

        for depth in sorted(all_depths):
            ds = str(depth)
            knn_d  = knn.get("depth_metrics", {}).get(ds, {})
            flat_d = flat.get("depth_metrics", {}).get(ds, {})
            box_d  = box.get("depth_metrics", {}).get(ds, {}) if has_box else {}

            meta    = knn_d or flat_d or box_d
            n_terms = meta.get("n_terms", "?")
            avg_ann = _nan_safe(meta.get("avg_train_count"), ".1f")

            knn_f  = knn_d.get("fmax")
            flat_f = flat_d.get("fmax")
            box_f  = box_d.get("fmax") if has_box else None

            try:
                flat_knn = f"{float(flat_f) - float(knn_f):>+9.4f}"
            except (TypeError, ValueError):
                flat_knn = f"{'---':>9}"

            row = (f"{depth:>5}  {n_terms:>7}  {avg_ann:>8}  "
                   f"{_nan_safe(knn_f):>8}  {_nan_safe(flat_f):>8}")
            if has_box:
                try:
                    box_knn = f"{float(box_f) - float(knn_f):>+9.4f}"
                except (TypeError, ValueError):
                    box_knn = f"{'---':>9}"
                row += f"  {_nan_safe(box_f):>8}  {flat_knn}  {box_knn}"
            else:
                row += f"  {flat_knn}"
            print(row)
    else:
        print("  (no depth metrics — go_dag not available in ProtCastDataset)")

    # ── Section 3: Frequency breakdown ────────────────────────────────────────
    print()
    print("── FREQUENCY BREAKDOWN  (by training annotation count) ──")

    bucket_labels = [
        ("rare_lt50",      "Rare  (<50)     "),
        ("medium_50_500",  "Medium (50–500) "),
        ("common_gt500",   "Common (>500)   "),
    ]

    any_freq = any(r.get("frequency_metrics") for r in [knn, flat, box] if r)
    if any_freq:
        if has_box:
            hdr = (f"{'Bucket':<18}  {'N terms':>7}  {'Avg ann':>8}  "
                   f"{'KNN':>8}  {'Flat':>8}  {'Box':>8}  {'Flat-KNN':>9}  {'Box-KNN':>9}")
        else:
            hdr = (f"{'Bucket':<18}  {'N terms':>7}  {'Avg ann':>8}  "
                   f"{'KNN':>8}  {'Flat':>8}  {'Flat-KNN':>9}")
        print(hdr)
        print("-" * len(hdr))

        for bucket, label in bucket_labels:
            knn_b  = knn.get("frequency_metrics", {}).get(bucket, {})
            flat_b = flat.get("frequency_metrics", {}).get(bucket, {})
            box_b  = box.get("frequency_metrics", {}).get(bucket, {}) if has_box else {}

            meta = knn_b or flat_b or box_b
            if not meta:
                continue

            n_terms = meta.get("n_terms", "?")
            avg_ann = _nan_safe(meta.get("avg_train_count"), ".1f")

            knn_f  = knn_b.get("fmax")
            flat_f = flat_b.get("fmax")
            box_f  = box_b.get("fmax") if has_box else None

            try:
                flat_knn = f"{float(flat_f) - float(knn_f):>+9.4f}"
            except (TypeError, ValueError):
                flat_knn = f"{'---':>9}"

            row = (f"{label:<18}  {n_terms:>7}  {avg_ann:>8}  "
                   f"{_nan_safe(knn_f):>8}  {_nan_safe(flat_f):>8}")
            if has_box:
                try:
                    box_knn = f"{float(box_f) - float(knn_f):>+9.4f}"
                except (TypeError, ValueError):
                    box_knn = f"{'---':>9}"
                row += f"  {_nan_safe(box_f):>8}  {flat_knn}  {box_knn}"
            else:
                row += f"  {flat_knn}"
            print(row)
    else:
        print("  (no frequency metrics)")

    print()
    print(f"Total elapsed time: {results.get('elapsed', '?')}s")
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare KNN vs MultiLabel flat vs MultiLabel+box on ESM-C embeddings "
            "using CAFA Fmax/Smin with depth and frequency breakdowns."
        )
    )
    parser.add_argument(
        "-d", "--input_dir", required=True,
        help="Path to directory containing pre-computed ESM embedding .pkl files",
    )
    parser.add_argument(
        "-p", "--protcast_dataset", required=True,
        help="Path to serialised ProtCastDataset (.bin file)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split (default: 42)",
    )
    parser.add_argument(
        "-o", "--output_dir", default="knn_vs_multilabel",
        help="Directory for all output files (default: knn_vs_multilabel)",
    )
    parser.add_argument(
        "--box", action="store_true",
        help="Also train MultiLabelClassifier with BoxEmbeddingLayer (3-way comparison)",
    )
    parser.add_argument(
        "--use_mlflow", action="store_true",
        help="Log each model run to MLflow",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    config = ConfigManager.load_config()
    start  = time.time()

    # Seed Python, NumPy, and TF RNGs so determinism flags actually take effect.
    # Combined with TF_DETERMINISTIC_OPS at the top of this file, this makes
    # the same --seed reproduce the same Fmax bit-for-bit across re-runs.
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)
    except Exception:
        pass  # KNN-only runs don't need TF

    # Resolve to absolute paths now — os.chdir() later would silently break
    # any relative paths that are used after the working directory changes.
    input_dir = os.path.abspath(args.input_dir)
    protcast_dataset = os.path.abspath(args.protcast_dataset)
    name   = os.path.basename(input_dir.rstrip("/"))

    level_match = re.search(r"level-(\d+)", name)
    level = int(level_match.group(1)) if level_match else None

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    results_file = f"{name}_knn_vs_multilabel_results.json"

    # ── Resume support ─────────────────────────────────────────────────────
    results = None
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)

    expected = {"knn", "multilabel_flat"}
    if args.box:
        expected.add("multilabel_box")

    if results is not None:
        done = {k for k in expected if results.get(k, {}).get("status") == "ok"}
        if done == expected:
            print(f"All models already completed in {results_file}")
            print_results(results)
            return
        remaining = expected - done
        print(f"Resuming: completed={sorted(done)}, remaining={sorted(remaining)}")
    else:
        results = {
            "seed":    args.seed,
            "level":   level,
            "esm_dim": None,
            "run_box": args.box,
        }

    # ── Validate inputs ────────────────────────────────────────────────────
    if not os.path.isdir(input_dir):
        print(f"Error: embeddings directory not found: {input_dir}")
        return
    if not [f for f in os.listdir(input_dir) if f.endswith(".pkl")]:
        print(f"Error: no .pkl embedding files found in {input_dir}")
        return
    if not os.path.exists(protcast_dataset):
        print(f"Error: ProtCast dataset not found: {protcast_dataset}")
        return

    # ── Load data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    try:
        dataset = ProtCastDataset.load_serialized_file(protcast_dataset)
    except Exception as e:
        print(f"Error loading ProtCastDataset from {protcast_dataset}: {e}")
        return

    try:
        protein_embeddings, protein_go_terms, go_ids = load_flat_embeddings(
            input_dir, args.verbose
        )
    except Exception as e:
        print(f"Error loading embeddings from {input_dir}: {e}")
        return

    if not protein_embeddings:
        print(f"Error: no protein embeddings loaded from {input_dir}")
        return

    go_dag = getattr(dataset, "annotated_dag", None)
    if go_dag is None:
        print(
            "Warning: ProtCastDataset has no annotated_dag — depth metrics "
            "and box containment loss will be unavailable."
        )

    if results.get("esm_dim") is None:
        results["esm_dim"] = int(next(iter(protein_embeddings.values())).shape[0])

    # ── Parent MLflow run ────────────────────────────────────────────────────
    # Wrap the three model runs under a single parent so they appear grouped
    # in the MLflow UI as one comparison.  Each child run starts nested under
    # this parent (handled inside the classifier .run() methods).
    parent_mlflow = None
    parent_run_ctx = None
    if args.use_mlflow:
        try:
            from protcast.utils.mlflow_utils import init_mlflow
            parent_mlflow = init_mlflow(
                experiment_name=config.get("EXPERIMENT_NAME", "Default Experiment"),
                repo_owner=config.get("DAGSHUB_REPO_OWNER", "aakpan"),
                repo_name=config.get("DAGSHUB_REPO_NAME", "my-first-repo"),
                verbose=args.verbose,
            )
            parent_run_name = f"comparison_{name}_seed{args.seed}"
            parent_run_ctx = parent_mlflow.start_run(run_name=parent_run_name)
            parent_mlflow.set_tag("comparison_type", "knn_vs_multilabel")
            parent_mlflow.set_tag("level", str(level))
            parent_mlflow.log_param("seed", args.seed)
            parent_mlflow.log_param("box_enabled", args.box)
            parent_mlflow.log_param("esm_dim", results["esm_dim"])
        except Exception as e:
            print(f"Warning: could not start parent MLflow run: {e}")
            parent_mlflow = None

    # ── Model 1: KNN ───────────────────────────────────────────────────────
    if results.get("knn", {}).get("status") != "ok":
        print("\n" + "=" * 60)
        print("MODEL 1 / 3: KNN")
        print("=" * 60)
        try:
            results["knn"] = train_knn(
                protein_embeddings, protein_go_terms, go_ids,
                config, name, args.seed, go_dag, args.use_mlflow, args.verbose,
            )
            print(
                f"KNN — Fmax: {results['knn']['fmax']:.4f}  "
                f"Smin: {results['knn']['smin']:.4f}  "
                f"Time: {results['knn']['training_time']:.1f}s"
            )
        except Exception as e:
            print(f"FAILED: KNN — {e}")
            results["knn"] = {"status": f"error: {e}"}

        results["elapsed"] = round(time.time() - start)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    # ── Model 2: MultiLabel flat ───────────────────────────────────────────
    if results.get("multilabel_flat", {}).get("status") != "ok":
        print("\n" + "=" * 60)
        print("MODEL 2 / 3: MULTILABEL (flat)")
        print("=" * 60)
        try:
            results["multilabel_flat"] = train_multilabel(
                protein_embeddings, protein_go_terms, go_ids,
                config, name, args.seed, go_dag,
                use_box=False, use_mlflow=args.use_mlflow, verbose=args.verbose,
            )
            r = results["multilabel_flat"]
            print(
                f"Flat NN — Fmax: {r['fmax']:.4f}  "
                f"Smin: {r['smin']:.4f}  "
                f"Epochs: {r['epochs']}  "
                f"Time: {r['training_time']:.1f}s"
            )
        except Exception as e:
            print(f"FAILED: MultiLabel flat — {e}")
            results["multilabel_flat"] = {"status": f"error: {e}"}

        results["elapsed"] = round(time.time() - start)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    # ── Model 3: MultiLabel + box (optional) ──────────────────────────────
    if args.box and results.get("multilabel_box", {}).get("status") != "ok":
        print("\n" + "=" * 60)
        print("MODEL 3 / 3: MULTILABEL + BOX EMBEDDINGS")
        print("=" * 60)
        try:
            results["multilabel_box"] = train_multilabel(
                protein_embeddings, protein_go_terms, go_ids,
                config, name, args.seed, go_dag,
                use_box=True, use_mlflow=args.use_mlflow, verbose=args.verbose,
            )
            r = results["multilabel_box"]
            print(
                f"Box NN — Fmax: {r['fmax']:.4f}  "
                f"Smin: {r['smin']:.4f}  "
                f"Epochs: {r['epochs']}  "
                f"Time: {r['training_time']:.1f}s"
            )
        except Exception as e:
            print(f"FAILED: MultiLabel box — {e}")
            results["multilabel_box"] = {"status": f"error: {e}"}

        results["elapsed"] = round(time.time() - start)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    results["elapsed"] = round(time.time() - start)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Close the parent MLflow run after all child runs have finished.
    if parent_mlflow is not None:
        try:
            for key in ("knn", "multilabel_flat", "multilabel_box"):
                r = results.get(key, {})
                if r.get("status") == "ok":
                    parent_mlflow.log_metric(f"{key}_fmax", r["fmax"])
                    parent_mlflow.log_metric(f"{key}_smin", r["smin"])
                    parent_mlflow.log_metric(f"{key}_training_time", r["training_time"])
            parent_mlflow.log_metric("total_elapsed_seconds", results["elapsed"])
            parent_mlflow.end_run()
        except Exception as e:
            print(f"Warning: error finalising parent MLflow run: {e}")

    print_results(results)


if __name__ == "__main__":
    main()
