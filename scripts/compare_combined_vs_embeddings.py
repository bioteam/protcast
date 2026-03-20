"""compare_combined_vs_embeddings.py

Compare the effectiveness of ESM-C embeddings alone vs. ESM-C embeddings
combined with traditional feature vectors (CTriad, Moran, CTDD) for
predicting molecular function (GO term classification).

Both models are trained on the same data with the same train/test split
(controlled by --seed) so that metrics are directly comparable.

Example usage:

python3 scripts/compare_combined_vs_embeddings.py \
    -d mf_go_terms-level-4 \
    -p ProtcastDataset.bin \
    --feature_algorithms CTriad Moran CTDD \
    --seed 42 \
    -v
"""

import sys
import os
import time
import argparse
import pickle
import re
import gc
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from protcast.model.multi_classifier import MultiClassifier  # noqa: E402
from protcast.preprocessing.protcast_dataset import (  # noqa: E402
    ProtCastDataset,
)
from protcast.config.model_config import ConfigManager  # noqa: E402


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


def main():
    parser = argparse.ArgumentParser(
        description="Compare ESM embeddings vs combined (ESM + feature vectors)"
    )
    parser.add_argument(
        "-d", "--input_dir", required=True, help="Path to embeddings directory"
    )
    parser.add_argument(
        "-p", "--protcast_dataset", required=True, help="Path to ProtCast dataset"
    )
    parser.add_argument(
        "--feature_algorithms",
        nargs="+",
        default=["CTriad", "Moran", "CTDD"],
        help="Feature vector algorithms for combined mode (default: CTriad Moran CTDD)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for train/test split"
    )
    parser.add_argument("--use_mlflow", action="store_true", help="Use MLFlow")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    args = parser.parse_args()

    config = ConfigManager.load_config()
    start = time.time()

    # Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    dataset = ProtCastDataset.load_serialized_file(args.protcast_dataset)
    proteins = load_embeddings(args.input_dir, args.verbose)
    combined_proteins = build_combined_proteins(proteins, dataset, args.verbose)
    name = os.path.basename(args.input_dir)

    # --- Model A: ESM embeddings only ---
    print("\n" + "=" * 60)
    print("MODEL A: ESM EMBEDDINGS ONLY")
    print("=" * 60)
    model_a = MultiClassifier(
        algorithm="esm",
        verbose=args.verbose,
        proteins=proteins,
        config=config,
        id=f"{name}_esm_only",
        use_mlflow=args.use_mlflow,
        input_source="esm_embeddings",
        random_state=args.seed,
    )
    model_a.run()
    history_a = model_a.history.history

    # Extract best metrics for Model A
    best_f1_a = max(history_a["val_f1_score"])
    best_acc_a = max(history_a["val_accuracy"])
    best_loss_a = min(history_a["val_loss"])
    epochs_a = len(history_a["loss"])

    gc.collect()

    # --- Model B: Combined (ESM + feature vectors) ---
    print("\n" + "=" * 60)
    print(f"MODEL B: COMBINED (ESM + {', '.join(args.feature_algorithms)})")
    print("=" * 60)
    model_b = MultiClassifier(
        algorithm="esm_combined",
        verbose=args.verbose,
        proteins=combined_proteins,
        config=config,
        id=f"{name}_combined",
        use_mlflow=args.use_mlflow,
        input_source="combined",
        feature_algorithms=args.feature_algorithms,
        random_state=args.seed,
    )
    model_b.run()
    history_b = model_b.history.history

    # Extract best metrics for Model B
    best_f1_b = max(history_b["val_f1_score"])
    best_acc_b = max(history_b["val_accuracy"])
    best_loss_b = min(history_b["val_loss"])
    epochs_b = len(history_b["loss"])

    gc.collect()

    # --- Results ---
    elapsed = round(time.time() - start)
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Random seed: {args.seed}")
    print(f"Feature algorithms: {', '.join(args.feature_algorithms)}")
    print(f"Input dimensions: ESM={model_a.vector_length}, Combined={model_b.vector_length}")
    print()
    print(f"{'Metric':<25} {'ESM Only':>12} {'Combined':>12} {'Diff':>12}")
    print("-" * 61)

    f1_diff = best_f1_b - best_f1_a
    acc_diff = best_acc_b - best_acc_a
    loss_diff = best_loss_b - best_loss_a

    print(f"{'Best val F1 score':<25} {best_f1_a:>12.4f} {best_f1_b:>12.4f} {f1_diff:>+12.4f}")
    print(f"{'Best val accuracy':<25} {best_acc_a:>12.4f} {best_acc_b:>12.4f} {acc_diff:>+12.4f}")
    print(f"{'Best val loss':<25} {best_loss_a:>12.4f} {best_loss_b:>12.4f} {loss_diff:>+12.4f}")
    print(f"{'Epochs trained':<25} {epochs_a:>12d} {epochs_b:>12d}")
    print(f"{'Training time (s)':<25} {model_a.training_time:>12.1f} {model_b.training_time:>12.1f}")
    print()

    if f1_diff > 0.005:
        print(f">> Combined mode IMPROVED F1 score by {f1_diff:+.4f}")
    elif f1_diff < -0.005:
        print(f">> Combined mode DECREASED F1 score by {f1_diff:+.4f}")
    else:
        print(f">> No significant difference in F1 score ({f1_diff:+.4f})")

    print(f"\nTotal elapsed time: {elapsed}s")
    print("Done!")


if __name__ == "__main__":
    main()
