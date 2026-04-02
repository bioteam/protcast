"""scan_feature_vectors_only.py

Benchmark every individual feature vector algorithm from protein-feature-vectors
through MultiLabelClassifier (sigmoid, multi-label) WITHOUT ESM embeddings.

Reports CAFA-standard Fmax and Smin, plus macro F1 for each algorithm.
Includes an ESM-only baseline for comparison.

Inputs (not modified):
    - Pre-computed ESM embeddings (.pkl files) in the -d/--input_dir directory.
      Used to determine the GO term scope and protein set (and for the
      ESM-only baseline). Feature-vector-only runs use sequences from
      the ProtCastDataset, not the embeddings.
    - Serialized ProtCastDataset (.bin file) containing protein sequences
      and GO annotations.

Saved to the output directory (-o/--output_dir, default: fv_only_scan):
    - {name}_fv_only_scan_results.json   All results (updated after each algorithm)
    - Model/encoder files for each algorithm that completes successfully

    If the results JSON already exists with all algorithms completed, the script
    loads and prints it without retraining. Individual algorithms already in the
    results file are skipped on re-run (allows resuming interrupted scans).

Example usage:

python3 scripts/scan_feature_vectors_only.py \\
    -d mf_go_terms-level-8 \\
    -p ProtcastDataset.bin \\
    -o fv_only_scan \\
    --seed 42 \\
    -v
"""

import os
import time
import argparse
import pickle
import json
import re
import gc
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from protein_feature_vectors import Calculator

from protcast.model.multilabel_classifier import MultiLabelClassifier
from protcast.model.stats.utils import calculate_fmax, calculate_smin
from protcast.preprocessing.protcast_dataset import ProtCastDataset
from protcast.config.model_config import ConfigManager


# Complete list of algorithms from protein-feature-vectors README
ALL_ALGORITHMS = [
    "AAC",
    "AC",
    "ACC",
    "APAAC",
    "ASDC",
    "CC",
    "CKSAAGP_type_1",
    "CKSAAGP_type_2",
    "CKSAAP_type_1",
    "CKSAAP_type_2",
    "CKSAAP_type_3",
    "CTDC",
    "CTDD",
    "CTDT",
    "CTriad",
    "DDE",
    "DistancePair",
    "DPC_type_1",
    "DPC_type_2",
    "GAAC",
    "GDPC_type_1",
    "GDPC_type_2",
    "Geary",
    "GTPC_type_1",
    "GTPC_type_2",
    "K1TPC",
    "K2TPC",
    "KSCTriad",
    "Moran",
    "NMBroto",
    "PAAC",
    "PseKRAAC_type_1",
    "PseKRAAC_type_2",
    "PseKRAAC_type_3A",
    "PseKRAAC_type_3B",
    "PseKRAAC_type_4",
    "PseKRAAC_type_5",
    "PseKRAAC_type_6A",
    "PseKRAAC_type_6B",
    "PseKRAAC_type_6C",
    "PseKRAAC_type_7",
    "PseKRAAC_type_8",
    "PseKRAAC_type_9",
    "PseKRAAC_type_10",
    "PseKRAAC_type_11",
    "PseKRAAC_type_12",
    "PseKRAAC_type_13",
    "PseKRAAC_type_14",
    "PseKRAAC_type_15",
    "PseKRAAC_type_16",
    "QSOrder",
    "SOCNumber",
    "TPC_type_1",
    "TPC_type_2",
    "TPC_type_3",
]


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
        go_id = match.group(1)
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


def build_fv_embeddings(dataset, protein_ids, algo, verbose=False):
    """Compute and normalize feature vectors for a single algorithm.

    Parameters
    ----------
    dataset : ProtCastDataset
        Dataset containing protein sequences.
    protein_ids : list[str]
        Protein IDs to compute feature vectors for.
    algo : str
        Feature vector algorithm name (e.g. "CTriad").
    verbose : bool
        Print progress.

    Returns
    -------
    fv_embeddings : dict
        {protein_id: np.ndarray} with normalized feature vectors.
    fv_dim : int
        Dimension of the feature vectors.
    """
    # Collect sequences
    sequences = {}
    for pid in protein_ids:
        if pid in dataset.proteins:
            sequences[pid] = dataset.proteins[pid].sequence

    if not sequences:
        raise ValueError("No proteins have sequences in the dataset")

    # Compute feature vectors
    calc = Calculator(verbose=verbose)
    calc.get_feature_vectors(algo, pdict=sequences)
    if calc.encodings is None:
        raise ValueError(f"No encodings generated for algorithm {algo}")

    # Keep only proteins present in encodings
    final_pids = [pid for pid in sequences if pid in calc.encodings.index]
    if not final_pids:
        raise ValueError(f"No valid proteins for algorithm {algo}")

    skipped = len(sequences) - len(final_pids)
    if skipped > 0 and verbose:
        print(f"  Skipped {skipped} proteins missing from {algo} encodings")

    # Build array and normalize
    fv_list = [calc.encodings.loc[pid].values.astype(np.float32) for pid in final_pids]
    fv_array = np.vstack(fv_list)
    fv_dim = fv_array.shape[1]

    scaler = StandardScaler()
    fv_scaled = scaler.fit_transform(fv_array).astype(np.float32)

    fv_embeddings = {pid: fv_scaled[i] for i, pid in enumerate(final_pids)}

    if verbose:
        print(f"  {algo}: {len(final_pids)} proteins, dim={fv_dim}")

    return fv_embeddings, fv_dim


def compute_metrics(classifier):
    """Compute Fmax, Smin, and macro F1 on the validation set."""
    y_pred = classifier.model.predict(classifier.X_val, verbose=0)
    fmax, fmax_threshold = calculate_fmax(classifier.y_val, y_pred)
    smin, smin_threshold = calculate_smin(classifier.y_val, y_pred)

    # Macro F1 at the Fmax-optimal threshold
    y_pred_binary = (y_pred >= fmax_threshold).astype(np.float32)
    f1_macro = float(f1_score(
        classifier.y_val, y_pred_binary, average="macro", zero_division=0
    ))

    return {
        "best_fmax": float(fmax),
        "fmax_threshold": float(fmax_threshold),
        "smin": float(smin),
        "smin_threshold": float(smin_threshold),
        "f1_macro": float(f1_macro),
    }


def train_esm_baseline(protein_embeddings, protein_go_terms, go_ids, config, name, seed, verbose=False):
    """Train ESM-only baseline using MultiLabelClassifier."""
    classifier = MultiLabelClassifier(
        verbose=verbose,
        protein_embeddings=protein_embeddings,
        protein_go_terms=protein_go_terms,
        go_ids=go_ids,
        config=config,
        id=f"{name}_fvscan_esm_only",
        random_state=seed,
    )
    classifier.run()

    metrics = compute_metrics(classifier)
    result = {
        "algorithm": "ESM_only",
        "fv_dim": classifier.vector_length,
        **metrics,
        "best_loss": float(min(classifier.history.history["val_loss"])),
        "epochs": len(classifier.history.history["loss"]),
        "training_time": classifier.training_time,
        "status": "ok",
    }
    del classifier
    gc.collect()
    return result


def train_fv_only(protein_go_terms, go_ids, dataset, protein_ids, config, name, algo, seed, verbose=False):
    """Train a single feature-vector-only model using MultiLabelClassifier."""
    fv_embeddings, fv_dim = build_fv_embeddings(dataset, protein_ids, algo, verbose)

    # Filter protein_go_terms to only proteins with feature vectors
    filtered_go_terms = {
        pid: terms
        for pid, terms in protein_go_terms.items()
        if pid in fv_embeddings
    }

    classifier = MultiLabelClassifier(
        verbose=verbose,
        protein_embeddings=fv_embeddings,
        protein_go_terms=filtered_go_terms,
        go_ids=go_ids,
        config=config,
        id=f"{name}_fvscan_{algo}",
        random_state=seed,
    )
    classifier.run()

    metrics = compute_metrics(classifier)
    result = {
        "algorithm": algo,
        "fv_dim": fv_dim,
        **metrics,
        "best_loss": float(min(classifier.history.history["val_loss"])),
        "epochs": len(classifier.history.history["loss"]),
        "training_time": classifier.training_time,
        "status": "ok",
    }
    del classifier, fv_embeddings
    gc.collect()
    return result


def print_results(results):
    """Print scan results as a sorted table."""
    print("\n" + "=" * 112)
    print("FEATURE VECTORS ONLY — SCAN RESULTS (CAFA Metrics)")
    print("=" * 112)
    print(f"Level: {results.get('level', '?')}")
    print(f"Seed: {results['seed']}")
    print()

    baseline = results["baseline"]
    algo_results = results["algorithms"]

    # Sort by Fmax descending
    sorted_results = sorted(algo_results, key=lambda x: x["best_fmax"], reverse=True)

    print(
        f"{'Rank':<5} {'Algorithm':<22} {'Dim':>7} {'Fmax':>8} {'dFmax':>8} "
        f"{'Thr':>6} {'Smin':>8} {'F1':>8} {'Loss':>8} {'Epochs':>7} {'Time':>7} {'Status':<6}"
    )
    print("-" * 112)

    # Print baseline first
    print(
        f"{'---':<5} {'ESM_only (baseline)':<22} {baseline['fv_dim']:>7d} "
        f"{baseline['best_fmax']:>8.4f} {'---':>8} "
        f"{baseline['fmax_threshold']:>6.2f} {baseline['smin']:>8.4f} "
        f"{baseline['f1_macro']:>8.4f} {baseline['best_loss']:>8.4f} "
        f"{baseline['epochs']:>7d} {baseline['training_time']:>6.1f}s {'ok':<6}"
    )
    print("-" * 112)

    for rank, r in enumerate(sorted_results, 1):
        if r["status"] != "ok":
            print(
                f"{rank:<5} {r['algorithm']:<22} {'---':>7} {'---':>8} {'---':>8} "
                f"{'---':>6} {'---':>8} {'---':>8} {'---':>8} {'---':>7} {'---':>7} {r['status']:<6}"
            )
            continue
        fmax_diff = r["best_fmax"] - baseline["best_fmax"]
        print(
            f"{rank:<5} {r['algorithm']:<22} {r['fv_dim']:>7d} "
            f"{r['best_fmax']:>8.4f} {fmax_diff:>+8.4f} "
            f"{r['fmax_threshold']:>6.2f} {r['smin']:>8.4f} "
            f"{r['f1_macro']:>8.4f} {r['best_loss']:>8.4f} "
            f"{r['epochs']:>7d} {r['training_time']:>6.1f}s {r['status']:<6}"
        )

    # Summary
    ok_results = [r for r in sorted_results if r["status"] == "ok"]
    print()
    print(f"Completed: {len(ok_results)}/{len(sorted_results)} algorithms")
    if ok_results:
        best = ok_results[0]
        print(
            f"Best FV-only: {best['algorithm']} "
            f"(Fmax {best['best_fmax']:.4f}, F1 {best['f1_macro']:.4f})"
        )
        print(
            f"ESM baseline: Fmax {baseline['best_fmax']:.4f}, "
            f"F1 {baseline['f1_macro']:.4f}"
        )
    print(f"\nTotal elapsed time: {results['elapsed']}s")


def main():
    parser = argparse.ArgumentParser(
        description="Scan all individual feature algorithms (without ESM) through multilabel classification"
    )
    parser.add_argument(
        "-d", "--input_dir", required=True, help="Path to embeddings directory (defines GO term scope)"
    )
    parser.add_argument(
        "-p", "--protcast_dataset", required=True, help="Path to ProtCast dataset"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for train/test split"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="fv_only_scan",
        help="Directory for all output files (default: fv_only_scan)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Specific algorithms to test (default: all)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    args = parser.parse_args()

    config = ConfigManager.load_config()
    start = time.time()
    input_dir = args.input_dir
    name = os.path.basename(input_dir)
    algorithms = args.algorithms or ALL_ALGORITHMS

    # Extract level from directory name (e.g. mf_go_terms-level-8 -> 8)
    level_match = re.search(r"level-(\d+)", name)
    level = int(level_match.group(1)) if level_match else None

    # Create output directory and work from there
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    results_file = f"{name}_fv_only_scan_results.json"

    # Load existing results for resume support
    results = None
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)

        completed = {r["algorithm"] for r in results.get("algorithms", [])}
        remaining = [a for a in algorithms if a not in completed]
        if not remaining and results.get("baseline"):
            print(f"All algorithms already completed in {results_file}")
            print_results(results)
            return
        print(f"Resuming: {len(completed)} done, {len(remaining)} remaining")
    else:
        results = {
            "seed": args.seed,
            "level": level,
            "baseline": None,
            "algorithms": [],
        }
        remaining = algorithms

    # Validate input files exist
    if not os.path.isdir(input_dir):
        print(f"Error: Embeddings directory not found: {input_dir}")
        return
    pkl_files = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]
    if not pkl_files:
        print(f"Error: No .pkl embedding files found in {input_dir}")
        return
    if not os.path.exists(args.protcast_dataset):
        print(f"Error: ProtCast dataset not found: {args.protcast_dataset}")
        return

    # Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    dataset = ProtCastDataset.load_serialized_file(args.protcast_dataset)
    protein_embeddings, protein_go_terms, go_ids = load_flat_embeddings(
        input_dir, args.verbose
    )
    protein_ids = list(protein_embeddings.keys())

    # --- Baseline: ESM embeddings only ---
    if not results["baseline"]:
        print("\n" + "=" * 60)
        print("BASELINE: ESM EMBEDDINGS ONLY")
        print("=" * 60)
        baseline = train_esm_baseline(
            protein_embeddings, protein_go_terms, go_ids,
            config, name, args.seed, args.verbose,
        )
        results["baseline"] = baseline
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(
            f"Baseline Fmax: {baseline['best_fmax']:.4f}, "
            f"Smin: {baseline['smin']:.4f}, F1: {baseline['f1_macro']:.4f}"
        )

    # --- Scan individual algorithms (feature vectors only) ---
    total = len(algorithms)
    completed_count = total - len(remaining)

    for i, algo in enumerate(remaining):
        idx = completed_count + i + 1
        print(f"\n{'=' * 60}")
        print(f"[{idx}/{total}] FV only: {algo}")
        print("=" * 60)

        try:
            result = train_fv_only(
                protein_go_terms, go_ids, dataset, protein_ids,
                config, name, algo, args.seed, args.verbose,
            )
        except Exception as e:
            print(f"FAILED: {algo} - {e}")
            result = {
                "algorithm": algo,
                "fv_dim": 0,
                "best_fmax": 0.0,
                "fmax_threshold": 0.0,
                "smin": 0.0,
                "smin_threshold": 0.0,
                "f1_macro": 0.0,
                "best_loss": 0.0,
                "epochs": 0,
                "training_time": 0.0,
                "status": f"error: {e}",
            }

        results["algorithms"].append(result)

        # Save after each algorithm for resume support
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
