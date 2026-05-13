"""compare_knn_esm_vs_knn_combined.py

Two-way CAFA comparison of KNN protein-function prediction:

  1. KNN (ESM-C)            – nearest-neighbour voting over raw ESM-C
                              embeddings (same as compare_knn_vs_multilabel.py).
  2. KNN (ESM-C + classical) – nearest-neighbour voting over a concatenation
                              of ESM-C embeddings and classical feature
                              vectors (CTriad, Moran, CTDD by default).
                              The two feature blocks are standardised
                              separately (fit on the training split only)
                              before concatenation, so the high-dimensional
                              ESM block does not numerically dominate the
                              distance metric.

Primary question: do classical descriptors carry information that
complements ESM-C embeddings under a simple non-parametric model (KNN),
or is ESM-C alone already saturating what KNN can extract?

Both KNN models are trained and evaluated on the *same* protein set,
the *same* train/val split (controlled by --seed), and the *same*
KNN hyper-parameters, so any Fmax/Smin delta is attributable to the
feature representation, not the data or the algorithm.

Inputs (not modified):
    - Pre-computed ESM embeddings (.pkl files) in -d/--input_dir.
    - Serialized ProtCastDataset (.bin file) with sequences + GO terms.

Saved to the output directory (-o/--output_dir):
    - {name}_knn_esm_vs_combined_results.json
    - {name}_knn_esm_comparison_knn.joblib
    - {name}_knn_combined_comparison_knn.joblib
    - {name}_knn_combined_scalers.pkl  (ESM + FV StandardScalers, plus
                                        the train-protein-id list used to
                                        fit them, for reproducible inference)

    Re-running with an existing results JSON resumes — completed models
    are skipped, so the script is safe to re-invoke after interruption.

Example usage:

python3 scripts/compare_knn_esm_vs_knn_combined.py \\
    -d mf_go_terms-level-4 \\
    -p ProtCastDataset.bin \\
    -o knn_esm_vs_combined \\
    --feature_algorithms CTriad Moran CTDD \\
    --seed 42 \\
    -v
"""

import os

# Match the determinism flags used in compare_knn_vs_multilabel.py so the
# environments are identical, even though KNN itself does not use TF.
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
from sklearn.preprocessing import StandardScaler

from protcast.model.knn_classifier import KNNClassifier
from protcast.preprocessing.protcast_dataset import ProtCastDataset
from protcast.config.model_config import ConfigManager


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_flat_embeddings(input_dir, verbose=False):
    """Load pre-computed ESM embeddings into flat protein-centric dicts.

    Identical to load_flat_embeddings() in compare_knn_vs_multilabel.py:
    one entry per protein, GO ids normalised to canonical GO:XXXXXXX form.
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


def compute_classical_feature_vectors(sequences, feature_algorithms, verbose=False):
    """Compute the concatenated classical FV block for every protein.

    Parameters
    ----------
    sequences : dict
        {protein_id: str} mapping. Order is not preserved by the calculator,
        so we look entries up by pid afterwards.
    feature_algorithms : list[str]
        Algorithms passed one-by-one to Calculator.get_feature_vectors,
        e.g. ["CTriad", "Moran", "CTDD"].

    Returns
    -------
    fv_dict : dict
        {protein_id: np.ndarray} — only contains protein ids that had a
        valid encoding under every algorithm (the intersection).
    """
    from protein_feature_vectors import Calculator
    fv = Calculator(verbose=verbose)
    algo_encodings = {}
    for algo in feature_algorithms:
        fv.get_feature_vectors(algo, pdict=sequences)
        if fv.encodings is None:
            raise ValueError(f"No {algo} encodings generated")
        algo_encodings[algo] = fv.encodings

    valid_pids = [
        pid for pid in sequences
        if all(pid in enc.index for enc in algo_encodings.values())
    ]
    if verbose:
        skipped = len(sequences) - len(valid_pids)
        if skipped:
            print(f"Classical FV: dropped {skipped} proteins missing in ≥1 encoding")

    fv_dict = {}
    for pid in valid_pids:
        parts = [
            algo_encodings[algo].loc[pid].values.astype(np.float32)
            for algo in feature_algorithms
        ]
        fv_dict[pid] = np.concatenate(parts).astype(np.float32)
    return fv_dict


def build_combined_embeddings(
    protein_embeddings, fv_dict, train_pids, verbose=False,
):
    """Concatenate scaled ESM and scaled classical FVs into one vector per pid.

    Scalers are fit on the train pids only (no validation leakage) and then
    used to transform every protein. We return the resulting per-pid dict
    plus the fitted scalers so they can be persisted for inference.
    """
    common_pids = sorted(set(protein_embeddings) & set(fv_dict))
    if not common_pids:
        raise ValueError("No proteins have both ESM and classical FVs")

    esm_dim = next(iter(protein_embeddings.values())).shape[0]
    fv_dim  = next(iter(fv_dict.values())).shape[0]

    # Stack the train slice for fitting; stack everything for transforming.
    train_pid_set = set(train_pids)
    train_only = [p for p in common_pids if p in train_pid_set]
    if not train_only:
        raise ValueError("No training proteins survived the FV intersection")

    esm_train = np.vstack([protein_embeddings[p] for p in train_only]).astype(np.float32)
    fv_train  = np.vstack([fv_dict[p]            for p in train_only]).astype(np.float32)

    esm_scaler = StandardScaler().fit(esm_train)
    fv_scaler  = StandardScaler().fit(fv_train)

    combined = {}
    for pid in common_pids:
        esm_s = esm_scaler.transform(protein_embeddings[pid].reshape(1, -1).astype(np.float32))
        fv_s  = fv_scaler.transform(fv_dict[pid].reshape(1, -1).astype(np.float32))
        fv_s  = np.nan_to_num(fv_s, nan=0.0, posinf=0.0, neginf=0.0)
        combined[pid] = np.concatenate([esm_s.ravel(), fv_s.ravel()]).astype(np.float32)

    if verbose:
        print(
            f"Combined feature dim: {esm_dim + fv_dim} "
            f"(ESM: {esm_dim} + FV: {fv_dim})  over {len(common_pids)} proteins"
        )
    return combined, esm_scaler, fv_scaler, common_pids


# ──────────────────────────────────────────────────────────────────────────────
# Training helper
# ──────────────────────────────────────────────────────────────────────────────

def train_knn(
    protein_embeddings, protein_go_terms, go_ids,
    config, name, variant_tag, seed, go_dag, use_mlflow, verbose=False,
):
    """Fit KNNClassifier on the given embedding dict and return a result dict."""
    classifier = KNNClassifier(
        verbose=verbose,
        protein_embeddings=protein_embeddings,
        protein_go_terms=protein_go_terms,
        go_ids=go_ids,
        config=config,
        id=f"{name}_knn_{variant_tag}_comparison",
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

    classifier.save_model()

    result = {
        "fmax": float(classifier.best_fmax),
        "fmax_threshold": float(classifier.best_threshold),
        "smin": float(classifier.best_smin),
        "smin_threshold": float(classifier.smin_threshold),
        "training_time": round(classifier.training_time, 2),
        "vector_length": int(classifier.vector_length),
        "depth_metrics": {str(k): v for k, v in depth_metrics.items()},
        "frequency_metrics": freq_metrics,
        "status": "ok",
    }
    del classifier
    gc.collect()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Train-pid extraction
# ──────────────────────────────────────────────────────────────────────────────

def get_train_pids(protein_ids, validation_split, random_state):
    """Reproduce the exact train split that KNNClassifier.train_model uses.

    KNNClassifier sorts protein ids inside prepare_data() and then passes the
    matching X / y rows to sklearn.train_test_split with the supplied
    random_state. We mirror that here so we can fit the combined-feature
    scalers on the same training proteins the KNN will later learn from.

    Invariant: if the classifier's sort order ever drifts from `sorted()`,
    this function silently fits scalers on the wrong split. The dict-key
    invariant (we feed the classifier a pre-filtered dict whose sorted keys
    are `protein_ids`) is what makes the splits agree.
    """
    from sklearn.model_selection import train_test_split
    sorted_pids = sorted(protein_ids)
    train_pids, _ = train_test_split(
        sorted_pids, test_size=validation_split, random_state=random_state,
    )
    return train_pids


# ──────────────────────────────────────────────────────────────────────────────
# Results display
# ──────────────────────────────────────────────────────────────────────────────

def _nan_safe(value, fmt=".4f"):
    try:
        if np.isnan(float(value)):
            return "---"
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return "---"


def _delta_str(a, b):
    try:
        d = float(a) - float(b)
        return f"{d:>+9.4f}"
    except (TypeError, ValueError):
        return f"{'---':>9}"


def print_results(results):
    esm  = results.get("knn_esm", {})
    comb = results.get("knn_combined", {})
    esm_fmax = esm.get("fmax")

    sep  = "=" * 96
    thin = "-" * 96
    print("\n" + sep)
    print("KNN(ESM-C) vs KNN(ESM-C + classical)  (CAFA Metrics)")
    print(sep)
    print(
        f"Level : {results.get('level', '?')}   "
        f"Seed  : {results['seed']}   "
        f"ESM dim : {results.get('esm_dim', '?')}   "
        f"Combined dim : {results.get('combined_dim', '?')}   "
        f"Features : {', '.join(results.get('feature_algorithms', []))}"
    )

    # ── Overall ───────────────────────────────────────────────────────────────
    print()
    print("── OVERALL METRICS ──")
    hdr = (f"{'Model':<26} {'Fmax':>8} {'Thr':>6} {'Smin':>8} "
           f"{'Time':>8}   {'Δ vs ESM':>10}")
    print(hdr)
    print(thin)

    def _row(label, r, is_baseline=False):
        fmax = _nan_safe(r.get("fmax"))
        thr  = _nan_safe(r.get("fmax_threshold"), ".2f")
        smin = _nan_safe(r.get("smin"))
        t    = _nan_safe(r.get("training_time"), ".1f")
        delta = f"{'---':>10}" if is_baseline else _delta_str(r.get("fmax"), esm_fmax)
        print(f"{label:<26} {fmax:>8} {thr:>6} {smin:>8} {t:>7}s   {delta}")

    if esm:
        _row("KNN (ESM-C)",            esm,  is_baseline=True)
    if comb:
        _row("KNN (ESM-C + classical)", comb)

    # ── Depth breakdown ───────────────────────────────────────────────────────
    print()
    print("── DEPTH BREAKDOWN  (higher depth = more specific GO terms) ──")
    all_depths = set()
    for r in (esm, comb):
        if r:
            all_depths.update(int(d) for d in r.get("depth_metrics", {}))

    if all_depths:
        hdr = (f"{'Depth':>5}  {'N terms':>7}  {'Avg ann':>8}  "
               f"{'ESM':>8}  {'Combined':>9}  {'Δ vs ESM':>10}")
        print(hdr)
        print("-" * len(hdr))
        for depth in sorted(all_depths):
            ds = str(depth)
            esm_d  = esm.get("depth_metrics", {}).get(ds, {})
            comb_d = comb.get("depth_metrics", {}).get(ds, {})
            meta   = esm_d or comb_d
            n_terms = meta.get("n_terms", "?")
            avg_ann = _nan_safe(meta.get("avg_train_count"), ".1f")
            esm_f  = esm_d.get("fmax")
            comb_f = comb_d.get("fmax")
            print(
                f"{depth:>5}  {n_terms:>7}  {avg_ann:>8}  "
                f"{_nan_safe(esm_f):>8}  {_nan_safe(comb_f):>9}  "
                f"{_delta_str(comb_f, esm_f)}"
            )
    else:
        print("  (no depth metrics — go_dag not available)")

    # ── Frequency breakdown ───────────────────────────────────────────────────
    print()
    print("── FREQUENCY BREAKDOWN  (by training annotation count) ──")
    bucket_labels = [
        ("rare_lt50",     "Rare  (<50)     "),
        ("medium_50_500", "Medium (50–500) "),
        ("common_gt500",  "Common (>500)   "),
    ]
    any_freq = any(r.get("frequency_metrics") for r in (esm, comb) if r)
    if any_freq:
        hdr = (f"{'Bucket':<18}  {'N terms':>7}  {'Avg ann':>8}  "
               f"{'ESM':>8}  {'Combined':>9}  {'Δ vs ESM':>10}")
        print(hdr)
        print("-" * len(hdr))
        for bucket, label in bucket_labels:
            esm_b  = esm.get("frequency_metrics", {}).get(bucket, {})
            comb_b = comb.get("frequency_metrics", {}).get(bucket, {})
            meta = esm_b or comb_b
            if not meta:
                continue
            n_terms = meta.get("n_terms", "?")
            avg_ann = _nan_safe(meta.get("avg_train_count"), ".1f")
            esm_f  = esm_b.get("fmax")
            comb_f = comb_b.get("fmax")
            print(
                f"{label:<18}  {n_terms:>7}  {avg_ann:>8}  "
                f"{_nan_safe(esm_f):>8}  {_nan_safe(comb_f):>9}  "
                f"{_delta_str(comb_f, esm_f)}"
            )
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
            "Compare KNN(ESM-C) vs KNN(ESM-C + classical feature vectors) "
            "using CAFA Fmax/Smin with depth and frequency breakdowns."
        )
    )
    parser.add_argument("-d", "--input_dir", required=True,
                        help="Directory of pre-computed ESM embedding .pkl files")
    parser.add_argument("-p", "--protcast_dataset", required=True,
                        help="Path to serialised ProtCastDataset (.bin)")
    parser.add_argument("--feature_algorithms", nargs="+",
                        default=["CTriad", "Moran", "CTDD"],
                        help="Classical FV algorithms (default: CTriad Moran CTDD)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split (default: 42)")
    parser.add_argument("-o", "--output_dir", default="knn_esm_vs_combined",
                        help="Directory for output files (default: knn_esm_vs_combined)")
    parser.add_argument("--use_mlflow", action="store_true",
                        help="Log each model run to MLflow")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    config = ConfigManager.load_config()
    start  = time.time()

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)
    except Exception:
        pass  # not required for KNN

    input_dir        = os.path.abspath(args.input_dir)
    protcast_dataset = os.path.abspath(args.protcast_dataset)
    name             = os.path.basename(input_dir.rstrip("/"))

    level_match = re.search(r"level-(\d+)", name)
    level = int(level_match.group(1)) if level_match else None

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    results_file = f"{name}_knn_esm_vs_combined_results.json"

    # ── Resume support ─────────────────────────────────────────────────────
    results = None
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)

    expected = {"knn_esm", "knn_combined"}
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
            "seed": args.seed,
            "level": level,
            "feature_algorithms": args.feature_algorithms,
            "esm_dim": None,
            "combined_dim": None,
        }

    if not os.path.isdir(input_dir):
        print(f"Error: embeddings directory not found: {input_dir}")
        return
    if not [f for f in os.listdir(input_dir) if f.endswith(".pkl")]:
        print(f"Error: no .pkl embedding files found in {input_dir}")
        return
    if not os.path.exists(protcast_dataset):
        print(f"Error: ProtCast dataset not found: {protcast_dataset}")
        return

    # ── Load ESM embeddings + dataset ──────────────────────────────────────
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    try:
        dataset = ProtCastDataset.load_serialized_file(protcast_dataset)
    except Exception as e:
        print(f"Error loading ProtCastDataset: {e}")
        return

    try:
        protein_embeddings, protein_go_terms, go_ids = load_flat_embeddings(
            input_dir, args.verbose,
        )
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return
    if not protein_embeddings:
        print("Error: no embeddings loaded")
        return

    go_dag = getattr(dataset, "annotated_dag", None)
    if go_dag is None:
        print("Warning: ProtCastDataset has no annotated_dag — depth metrics unavailable.")

    # ── Compute classical FVs once for the intersection of pids ────────────
    # Both KNN variants must train on the SAME protein set, otherwise the
    # train/val split (which is driven by sorted pid order) would differ.
    sequences = {
        pid: dataset.proteins[pid].sequence
        for pid in protein_embeddings
        if pid in dataset.proteins
    }
    missing_seq = len(protein_embeddings) - len(sequences)
    if missing_seq:
        print(f"Warning: {missing_seq} embedded proteins absent from ProtCastDataset")

    print("Computing classical feature vectors...")
    fv_dict = compute_classical_feature_vectors(
        sequences, args.feature_algorithms, verbose=args.verbose,
    )

    # Restrict the protein set to the intersection so both variants train on
    # identical data. The KNN baseline must therefore drop any pids that
    # lack a valid classical FV.
    common_pids = sorted(set(protein_embeddings) & set(fv_dict) & set(protein_go_terms))
    if not common_pids:
        print("Error: empty intersection of ESM / FV / annotated proteins")
        return
    if args.verbose:
        print(f"Common protein set (used for both variants): {len(common_pids)}")

    protein_embeddings = {p: protein_embeddings[p] for p in common_pids}
    protein_go_terms   = {p: protein_go_terms[p]   for p in common_pids}

    if results.get("esm_dim") is None:
        results["esm_dim"] = int(next(iter(protein_embeddings.values())).shape[0])

    # ── Variant 1: KNN on ESM-C only ───────────────────────────────────────
    if results.get("knn_esm", {}).get("status") != "ok":
        print("\n" + "=" * 60)
        print("MODEL 1 / 2: KNN (ESM-C)")
        print("=" * 60)
        try:
            results["knn_esm"] = train_knn(
                protein_embeddings, protein_go_terms, go_ids,
                config, name, "esm", args.seed, go_dag,
                args.use_mlflow, args.verbose,
            )
            print(
                f"KNN ESM — Fmax: {results['knn_esm']['fmax']:.4f}  "
                f"Smin: {results['knn_esm']['smin']:.4f}  "
                f"Time: {results['knn_esm']['training_time']:.1f}s"
            )
        except Exception as e:
            print(f"FAILED: KNN ESM — {e}")
            results["knn_esm"] = {"status": f"error: {e}"}

        results["elapsed"] = round(time.time() - start)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    # ── Variant 2: KNN on (scaled ESM-C ⊕ scaled classical FVs) ───────────
    if results.get("knn_combined", {}).get("status") != "ok":
        print("\n" + "=" * 60)
        print(f"MODEL 2 / 2: KNN (ESM-C + {', '.join(args.feature_algorithms)})")
        print("=" * 60)
        try:
            validation_split = config.get("VALIDATION_SPLIT", 0.2)
            train_pids = get_train_pids(common_pids, validation_split, args.seed)
            combined_embeddings, esm_scaler, fv_scaler, _ = build_combined_embeddings(
                protein_embeddings, fv_dict, train_pids, verbose=args.verbose,
            )

            # Persist scalers (+ the train pids used to fit them) so that
            # inference on new proteins is exactly reproducible.
            scalers_path = f"{name}_knn_combined_scalers.pkl"
            with open(scalers_path, "wb") as f:
                pickle.dump(
                    {
                        "esm_scaler": esm_scaler,
                        "fv_scaler":  fv_scaler,
                        "feature_algorithms": args.feature_algorithms,
                        "train_pids": train_pids,
                    },
                    f,
                )
            if args.verbose:
                print(f"Saved combined scalers to {scalers_path}")

            results["combined_dim"] = int(
                next(iter(combined_embeddings.values())).shape[0]
            )

            results["knn_combined"] = train_knn(
                combined_embeddings, protein_go_terms, go_ids,
                config, name, "combined", args.seed, go_dag,
                args.use_mlflow, args.verbose,
            )
            print(
                f"KNN Combined — Fmax: {results['knn_combined']['fmax']:.4f}  "
                f"Smin: {results['knn_combined']['smin']:.4f}  "
                f"Time: {results['knn_combined']['training_time']:.1f}s"
            )
        except Exception as e:
            print(f"FAILED: KNN Combined — {e}")
            results["knn_combined"] = {"status": f"error: {e}"}

        results["elapsed"] = round(time.time() - start)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

    results["elapsed"] = round(time.time() - start)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    print_results(results)


if __name__ == "__main__":
    main()
