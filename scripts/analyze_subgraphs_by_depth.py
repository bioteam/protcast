import sys
import argparse
import random
import json
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from protcast.preprocessing.protcast_dataset import (  # noqa: E402
    ProtCastDataset,
)
from protcast.model.multi_classifier import MultiClassifier  # noqa: E402


"""analyze_subgraphs_by_depth.py

This script analyzes GO term subgraphs by depth level in the molecular function ontology.

It takes:
- Path to serialized ProtCastDataset file
- Minimum number of sequences required for analysis
- Maximum number of sequences used to train a model

For each depth level in the Molecular Function graph:
    For each GO term at that depth:
      - Get the subgraph of GO terms for the GO term
      - Get the sequences associated with all the GO terms in the subgraph
      - Divide the sequences into training and test sets
      - Train a binary classifier model (subgraph vs. non-subgraph sequences)
      - Evaluate the F1 score of the model using the test set

Outputs summary statistics including F1 scores by depth level.
"""


def collect_subgraph_sequences(dataset, go_term, verbose=False):
    """Collect all protein sequences associated with a GO term's subgraph.

    Args:
        dataset: ProtCastDataset instance
        go_term: AnnotatedGOTerm instance
        verbose: Whether to print progress information

    Returns:
        dict: protein_id -> sequence mapping for subgraph sequences
    """
    subgraph_go_ids = dataset.get_subgraph(go_term.go_id)

    if verbose:
        print(f"      Subgraph contains {len(subgraph_go_ids)} GO terms")

    subgraph_sequences = {}

    for go_id in subgraph_go_ids:
        term = dataset.get_term(go_id)
        if term and term.annotations:
            protein_ids = term.get_all_pids()
            if protein_ids:
                for pid in protein_ids:
                    if pid in dataset.proteins:
                        protein = dataset.proteins[pid]
                        if protein.sequence and len(protein.sequence) > 0:
                            subgraph_sequences[pid] = protein.sequence

    if verbose:
        print(f"      Found {len(subgraph_sequences)} sequences in subgraph")

    return subgraph_sequences


def prepare_training_data(
    subgraph_sequences, dataset, max_seqs, verbose=False
):
    """Prepare balanced training data from subgraph and negative sequences.

    Args:
        subgraph_sequences: dict of protein_id -> sequence for positive samples
        dataset: ProtCastDataset instance
        max_seqs: Maximum number of sequences to use
        verbose: Whether to print progress information

    Returns:
        tuple: (positive_sequences, negative_sequences)
    """
    positive_sequences = list(subgraph_sequences.values())
    positive_ids = list(subgraph_sequences.keys())

    # Limit sequences if we have more than max_seqs
    if len(positive_sequences) > max_seqs:
        if verbose:
            print(
                f"      Limiting to {max_seqs} sequences (from {len(positive_sequences)})"
            )
        random.seed(42)  # For reproducibility
        # Sample sequences and their corresponding IDs together
        combined = list(zip(positive_sequences, positive_ids))
        sampled_combined = random.sample(combined, max_seqs)
        positive_sequences, positive_ids = zip(*sampled_combined)
        positive_sequences = list(positive_sequences)
        positive_ids = list(positive_ids)

    # Get negative sequences (not in subgraph)
    all_protein_ids = set(dataset.proteins.keys())
    negative_protein_ids = all_protein_ids - set(positive_ids)
    negative_protein_ids = list(negative_protein_ids)

    # Sample equal number of negative sequences
    random.seed(42)  # For reproducibility
    if len(negative_protein_ids) >= len(positive_ids):
        sampled_negative_ids = random.sample(
            negative_protein_ids, len(positive_ids)
        )
    else:
        sampled_negative_ids = negative_protein_ids

    negative_sequences = []
    for pid in sampled_negative_ids:
        if pid in dataset.proteins and dataset.proteins[pid].sequence:
            negative_sequences.append(dataset.proteins[pid].sequence)

    return positive_sequences, negative_sequences


def create_classifier_datasets(positive_sequences, negative_sequences, go_id):
    """Create datasets for MultiClassifier training and testing.

    Args:
        positive_sequences: List of positive sequence strings
        negative_sequences: List of negative sequence strings
        go_id: GO term ID for positive class

    Returns:
        tuple: (train_proteins_dict, test_proteins_dict, test_labels)
    """
    # Combine sequences and create labels
    all_sequences = positive_sequences + negative_sequences
    labels = [1] * len(positive_sequences) + [0] * len(negative_sequences)

    # Split into training and test sets
    train_sequences, test_sequences, train_labels, test_labels = (
        train_test_split(
            all_sequences,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
    )

    # Prepare training data for MultiClassifier
    negative_go_id = "GO:negative"
    train_proteins_dict = {go_id: {}, negative_go_id: {}}
    test_proteins_dict = {go_id: {}, negative_go_id: {}}

    train_pos_count = 0
    train_neg_count = 0
    test_pos_count = 0
    test_neg_count = 0

    for seq, label in zip(train_sequences, train_labels):
        if label == 1:
            train_proteins_dict[go_id][f"train_pos_{train_pos_count}"] = seq
            train_pos_count += 1
        else:
            train_proteins_dict[negative_go_id][
                f"train_neg_{train_neg_count}"
            ] = seq
            train_neg_count += 1

    for seq, label in zip(test_sequences, test_labels):
        if label == 1:
            test_proteins_dict[go_id][f"test_pos_{test_pos_count}"] = seq
            test_pos_count += 1
        else:
            test_proteins_dict[negative_go_id][
                f"test_neg_{test_neg_count}"
            ] = seq
            test_neg_count += 1

    return train_proteins_dict, test_proteins_dict, test_labels


def train_and_evaluate_model(
    train_proteins_dict,
    test_proteins_dict,
    test_labels,
    config,
    go_id,
    verbose=False,
):
    """Train a MultiClassifier model and evaluate its performance.

    Args:
        train_proteins_dict: Training data for MultiClassifier
        test_proteins_dict: Test data for MultiClassifier
        test_labels: True labels for test data
        config: Configuration dictionary for MultiClassifier
        go_id: GO term ID being processed
        verbose: Whether to print progress information

    Returns:
        float: F1 score of the trained model
    """
    if verbose:
        print(f"      Training model for {go_id}")

    # Train model
    classifier = MultiClassifier(
        algorithm="AAC",  # Amino Acid Composition
        verbose=False,  # Reduce verbosity for batch processing
        proteins=train_proteins_dict,
        config=config,
        use_mlflow=False,
        use_tensorboard=False,
    )
    classifier.run()

    # Evaluate model
    if verbose:
        print(f"      Evaluating model for {go_id}")

    test_classifier = MultiClassifier(
        algorithm="AAC",
        verbose=False,
        proteins=test_proteins_dict,
        config=config,
        use_mlflow=False,
        use_tensorboard=False,
    )

    # Get feature vectors for test data
    test_classifier.get_feature_vectors()
    test_classifier.prepare_data()

    # Make predictions
    predictions = classifier.model.predict(test_classifier.X)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate F1 score
    f1 = f1_score(test_labels, predicted_classes, average="binary")

    return f1


def process_go_term(term, dataset, config, args):
    """Process a single GO term: collect sequences, train model, evaluate.

    Args:
        term: AnnotatedGOTerm instance to process
        dataset: ProtCastDataset instance
        config: Configuration dictionary
        args: Command line arguments

    Returns:
        dict: Results dictionary with metrics, or None if processing failed
    """
    if args.verbose:
        print(f"    Processing term: {term.go_id} ({term.name})")

    # Collect subgraph sequences
    subgraph_sequences = collect_subgraph_sequences(
        dataset, term, args.verbose
    )

    # Check if we have minimum required sequences
    if len(subgraph_sequences) < args.minimum_seqs:
        if args.verbose:
            print(
                f"      Skipping {term.go_id}: only {len(subgraph_sequences)} sequences (minimum: {args.minimum_seqs})"
            )
        return None

    # Prepare training data
    positive_sequences, negative_sequences = prepare_training_data(
        subgraph_sequences, dataset, args.max_seqs, args.verbose
    )

    if len(negative_sequences) < len(positive_sequences) // 2:
        if args.verbose:
            print(
                f"      Skipping {term.go_id}: insufficient negative samples"
            )
        return None

    if args.verbose:
        print(
            f"      Dataset: {len(positive_sequences)} positive, {len(negative_sequences)} negative"
        )

    # Create datasets for classifier
    train_proteins_dict, test_proteins_dict, test_labels = (
        create_classifier_datasets(
            positive_sequences, negative_sequences, term.go_id
        )
    )

    try:
        # Train and evaluate model
        f1 = train_and_evaluate_model(
            train_proteins_dict,
            test_proteins_dict,
            test_labels,
            config,
            term.go_id,
            args.verbose,
        )

        # Calculate subgraph size
        subgraph_go_ids = dataset.get_subgraph(term.go_id)

        result = {
            "go_id": term.go_id,
            "go_name": term.name,
            "depth": term.depth,
            "subgraph_size": len(subgraph_go_ids),
            "total_sequences": len(subgraph_sequences),
            "train_positive": len(
                [seq for seq in train_proteins_dict[term.go_id]]
            ),
            "train_negative": len(
                [seq for seq in train_proteins_dict["GO:negative"]]
            ),
            "test_positive": len(
                [seq for seq in test_proteins_dict[term.go_id]]
            ),
            "test_negative": len(
                [seq for seq in test_proteins_dict["GO:negative"]]
            ),
            "f1_score": f1,
        }

        if args.verbose:
            print(f"      F1 Score: {f1:.4f}")

        return result

    except Exception as e:
        if args.verbose:
            print(f"      Error processing {term.go_id}: {str(e)}")
        return None


def print_results_summary(results):
    """Print a formatted summary of all results.

    Args:
        results: List of result dictionaries
    """
    print("\nResults Summary:")
    print("=" * 80)
    print(
        f"{'GO ID':<12} {'Name':<30} {'Depth':<6} {'Subgraph':<9} {'Sequences':<10} {'F1 Score':<10}"
    )
    print("-" * 80)

    for result in results:
        print(
            f"{result['go_id']:<12} {result['go_name'][:29]:<30} {result['depth']:<6} "
            f"{result['subgraph_size']:<9} {result['total_sequences']:<10} {result['f1_score']:<10.4f}"
        )

    print(f"\nProcessed {len(results)} GO terms successfully")

    if results:
        avg_f1 = sum(r["f1_score"] for r in results) / len(results)
        print(f"Average F1 Score: {avg_f1:.4f}")

        # Group by depth and show averages
        print("\nResults by Depth:")
        depth_results = defaultdict(list)
        for result in results:
            depth_results[result["depth"]].append(result["f1_score"])

        for depth in sorted(depth_results.keys()):
            scores = depth_results[depth]
            avg_score = sum(scores) / len(scores)
            print(
                f"  Depth {depth}: {len(scores)} terms, avg F1 = {avg_score:.4f}"
            )


def main():
    """Main function to coordinate the analysis workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--minimum_seqs",
        default=500,
        help="Minimum number of sequences",
        type=int,
    )
    parser.add_argument(
        "--max_seqs",
        default=2000,
        help="Maximum number of sequences to use for training",
        type=int,
    )
    parser.add_argument(
        "-p",
        "--protcast_dataset",
        required=True,
        help="Path to serialized dataset",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.json",
        help="Path to configuration file",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    args = parser.parse_args()

    # Load configuration from file
    with open(args.config, "r") as f:
        config = json.load(f)

    dataset = ProtCastDataset.load_serialized_file(args.protcast_dataset)

    if args.verbose:
        print(f"Loaded dataset with {len(dataset.proteins)} proteins")

    # Get all molecular function terms and organize by depth
    molecular_function_terms = dataset.get_all_terms(
        namespace="molecular_function"
    )
    if args.verbose:
        print(
            f"Found {len(molecular_function_terms)} molecular function terms"
        )

    # Group terms by depth level
    depth_levels = defaultdict(list)
    for term in molecular_function_terms:
        depth_levels[term.depth].append(term)

    if args.verbose:
        print(f"Depth levels found: {sorted(depth_levels.keys())}")
        for depth in sorted(depth_levels.keys()):
            print(f"  Depth {depth}: {len(depth_levels[depth])} terms")

    # Results storage
    results = []

    # Iterate through each depth level
    for depth in sorted(depth_levels.keys()):
        if args.verbose:
            print(f"\nProcessing depth level {depth}")

        terms_at_depth = depth_levels[depth]

        if args.verbose:
            print(f"  Processing {len(terms_at_depth)} terms at depth {depth}")

        # Process each GO term at this depth
        for term in terms_at_depth:
            result = process_go_term(term, dataset, config, args)
            if result:
                results.append(result)

    # Print results summary
    print_results_summary(results)


if __name__ == "__main__":
    main()
