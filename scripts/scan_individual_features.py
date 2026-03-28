"""scan_individual_features.py

Test every individual feature vector algorithm from protein-feature-vectors
combined with ESM-C embeddings, comparing each against an ESM-only baseline.

For each algorithm, trains a model using ESM embeddings + that single algorithm's
feature vectors, then compares against ESM-only performance.

Inputs (not modified):
    - Pre-computed ESM embeddings (.pkl files) in the -d/--input_dir directory.
    - Serialized ProtCastDataset (.bin file) containing protein sequences
      and GO annotations.

Saved to the output directory (-o/--output_dir, default: feature_scan):
    - {name}_feature_scan_results.json   All results (updated after each algorithm)
    - Model/encoder/scaler files for each algorithm that completes successfully

    If the results JSON already exists with all algorithms completed, the script
    loads and prints it without retraining. Individual algorithms already in the
    results file are skipped on re-run (allows resuming interrupted scans).

Example usage:

python3 scripts/scan_individual_features.py \\
    -d mf_go_terms-level-8 \\
    -p ProtcastDataset.bin \\
    -o feature_scan \\
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
from collections import defaultdict

from protcast.model.multi_classifier import MultiClassifier
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


def load_embeddings(input_dir, verbose=False):
    """Load pre-computed ESM embeddings from pickle files."""
    proteins = defaultdict(dict)
    for filename in os.listdir(input_dir):
        if not filename.endswith(".pkl"):
            continue
        match = re.match(r"(GO_\d+)", filename)
        if not match:
            continue
        go_id = match.group(1)
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "rb") as f:
            p = pickle.load(f)
        proteins[go_id] = p
    if verbose:
        total = sum(len(v) for v in proteins.values())
        print(f"Loaded embeddings for {len(proteins)} GO terms, {total} proteins")
    return proteins


def build_combined_proteins(proteins, dataset, verbose=False):
    """Pair embeddings with sequences from the dataset for combined mode."""
    combined = defaultdict(dict)
    missing = 0
    for go_id, pid_embeddings in proteins.items():
        for pid, embedding in pid_embeddings.items():
            if pid in dataset.proteins:
                combined[go_id][pid] = {
                    "embedding": embedding,
                    "sequence": dataset.proteins[pid].sequence,
                }
            else:
                missing += 1
    if missing > 0:
        print(f"Warning: {missing} proteins skipped (no sequence in dataset)")
    if verbose:
        total = sum(len(v) for v in combined.values())
        print(f"Combined data: {len(combined)} GO terms, {total} proteins")
    return combined


def train_esm_only(proteins, config, name, seed, verbose=False):
    """Train ESM-only baseline model."""
    model = MultiClassifier(
        algorithm="esm",
        verbose=verbose,
        proteins=proteins,
        config=config,
        id=f"{name}_scan_esm_only",
        input_source="esm_embeddings",
        random_state=seed,
    )
    model.run()
    history = model.history.history
    result = {
        "algorithm": "ESM_only",
        "fv_dim": 0,
        "combined_dim": model.vector_length,
        "best_f1": float(max(history["val_f1_score"])),
        "best_acc": float(max(history["val_accuracy"])),
        "best_loss": float(min(history["val_loss"])),
        "epochs": len(history["loss"]),
        "training_time": model.training_time,
        "status": "ok",
    }
    del model
    gc.collect()
    return result


def train_combined_single(combined_proteins, config, name, algo, seed, verbose=False):
    """Train ESM + single feature algorithm."""
    model = MultiClassifier(
        algorithm="esm_combined",
        verbose=verbose,
        proteins=combined_proteins,
        config=config,
        id=f"{name}_scan_{algo}",
        input_source="combined",
        feature_algorithms=[algo],
        random_state=seed,
    )
    model.run()
    history = model.history.history
    result = {
        "algorithm": algo,
        "fv_dim": model.fv_dim,
        "combined_dim": model.vector_length,
        "best_f1": float(max(history["val_f1_score"])),
        "best_acc": float(max(history["val_accuracy"])),
        "best_loss": float(min(history["val_loss"])),
        "epochs": len(history["loss"]),
        "training_time": model.training_time,
        "status": "ok",
    }
    del model
    gc.collect()
    return result


def print_results(results):
    """Print scan results as a sorted table."""
    print("\n" + "=" * 90)
    print("FEATURE SCAN RESULTS")
    print("=" * 90)
    print(f"Level: {results.get('level', '?')}")
    print(f"Seed: {results['seed']}")
    print(f"ESM dimension: {results['esm_dim']}")
    print()

    baseline = results["baseline"]
    algo_results = results["algorithms"]

    # Sort by F1 score descending
    sorted_results = sorted(algo_results, key=lambda x: x["best_f1"], reverse=True)

    print(
        f"{'Rank':<5} {'Algorithm':<22} {'FV Dim':>7} {'F1':>8} {'dF1':>8} "
        f"{'Acc':>8} {'Loss':>8} {'Epochs':>7} {'Time':>7} {'Status':<6}"
    )
    print("-" * 98)

    # Print baseline first
    print(
        f"{'---':<5} {'ESM_only (baseline)':<22} {'---':>7} "
        f"{baseline['best_f1']:>8.4f} {'---':>8} "
        f"{baseline['best_acc']:>8.4f} {baseline['best_loss']:>8.4f} "
        f"{baseline['epochs']:>7d} {baseline['training_time']:>6.1f}s {'ok':<6}"
    )
    print("-" * 98)

    for rank, r in enumerate(sorted_results, 1):
        if r["status"] != "ok":
            print(f"{rank:<5} {r['algorithm']:<22} {'---':>7} {'---':>8} {'---':>8} {'---':>8} {'---':>8} {'---':>7} {'---':>7} {r['status']:<6}")
            continue
        f1_diff = r["best_f1"] - baseline["best_f1"]
        print(
            f"{rank:<5} {r['algorithm']:<22} {r['fv_dim']:>7d} "
            f"{r['best_f1']:>8.4f} {f1_diff:>+8.4f} "
            f"{r['best_acc']:>8.4f} {r['best_loss']:>8.4f} "
            f"{r['epochs']:>7d} {r['training_time']:>6.1f}s {r['status']:<6}"
        )

    # Summary
    ok_results = [r for r in sorted_results if r["status"] == "ok"]
    improved = [r for r in ok_results if r["best_f1"] > baseline["best_f1"] + 0.005]
    print()
    print(f"Completed: {len(ok_results)}/{len(sorted_results)} algorithms")
    print(f"Improved over baseline (>0.5% F1): {len(improved)}")
    if improved:
        print(f"Best: {improved[0]['algorithm']} (F1 {improved[0]['best_f1']:.4f}, "
              f"+{improved[0]['best_f1'] - baseline['best_f1']:.4f})")
    print(f"\nTotal elapsed time: {results['elapsed']}s")


def main():
    parser = argparse.ArgumentParser(
        description="Scan all individual feature algorithms combined with ESM embeddings"
    )
    parser.add_argument(
        "-d", "--input_dir", required=True, help="Path to embeddings directory"
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
        default="feature_scan",
        help="Directory for all output files (default: feature_scan)",
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
    results_file = f"{name}_feature_scan_results.json"

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
            "esm_dim": None,
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
    proteins = load_embeddings(input_dir, args.verbose)
    combined_proteins = build_combined_proteins(proteins, dataset, args.verbose)

    # --- Baseline: ESM embeddings only ---
    if not results["baseline"]:
        print("\n" + "=" * 60)
        print("BASELINE: ESM EMBEDDINGS ONLY")
        print("=" * 60)
        baseline = train_esm_only(proteins, config, name, args.seed, args.verbose)
        results["baseline"] = baseline
        results["esm_dim"] = baseline["combined_dim"]
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Baseline F1: {baseline['best_f1']:.4f}")

    # --- Scan individual algorithms ---
    total = len(algorithms)
    completed_count = total - len(remaining)

    for i, algo in enumerate(remaining):
        idx = completed_count + i + 1
        print(f"\n{'=' * 60}")
        print(f"[{idx}/{total}] ESM + {algo}")
        print("=" * 60)

        try:
            result = train_combined_single(
                combined_proteins, config, name, algo, args.seed, args.verbose
            )
        except Exception as e:
            print(f"FAILED: {algo} - {e}")
            result = {
                "algorithm": algo,
                "fv_dim": 0,
                "combined_dim": 0,
                "best_f1": 0.0,
                "best_acc": 0.0,
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
