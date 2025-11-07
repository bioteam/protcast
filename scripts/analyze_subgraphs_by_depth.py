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
        print(f"Subgraph contains {len(subgraph_go_ids)} GO terms")

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
        print(f"Found {len(subgraph_sequences)} sequences in subgraph")

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
                f"Limiting to {max_seqs} sequences (from {len(positive_sequences)})"
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

    # Filter to only include proteins that have valid sequences
    valid_negative_ids = []
    for pid in negative_protein_ids:
        if pid in dataset.proteins:
            protein = dataset.proteins[pid]
            if protein.sequence and len(protein.sequence) > 0:
                valid_negative_ids.append(pid)

    # Sample equal number of negative sequences
    random.seed(42)  # For reproducibility
    if len(valid_negative_ids) >= len(positive_ids):
        sampled_negative_ids = random.sample(
            valid_negative_ids, len(positive_ids)
        )
    else:
        sampled_negative_ids = valid_negative_ids

    # Collect negative sequences with validation
    negative_sequences = []
    for pid in sampled_negative_ids:
        seq = dataset.proteins[pid].sequence
        if seq and len(seq) > 0:
            negative_sequences.append(seq)

    # Final validation to ensure equal counts
    if len(positive_sequences) != len(negative_sequences):
        if verbose:
            print("WARNING: Sequence count mismatch!")
            print(
                f"Positive: {len(positive_sequences)}, Negative: {len(negative_sequences)}"
            )
        # Adjust to match the smaller count
        min_count = min(len(positive_sequences), len(negative_sequences))
        positive_sequences = positive_sequences[:min_count]
        negative_sequences = negative_sequences[:min_count]
        if verbose:
            print(f"Adjusted both to: {min_count}")

    return positive_sequences, negative_sequences


def create_classifier_datasets(positive_sequences, negative_sequences, go_id):
    """Create datasets for MultiClassifier training and testing.

    Args:
        positive_sequences: List of positive sequence strings
        negative_sequences: List of negative sequence strings
        go_id: GO term ID for positive class

    Returns:
        tuple: (train_proteins_dict, test_proteins_dict)
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

    return train_proteins_dict, test_proteins_dict


def train_and_evaluate_model(
    train_proteins_dict,
    test_proteins_dict,
    config,
    algorithm,
    go_id,
    verbose=False,
):
    """Train a MultiClassifier model and evaluate its performance.

    Args:
        train_proteins_dict: Training data for MultiClassifier
        test_proteins_dict: Test data for MultiClassifier
        config: Configuration dictionary for MultiClassifier
        algorithm: Feature vector algorithm to use
        go_id: GO term ID being processed
        verbose: Whether to print progress information

    Returns:
        float: F1 score of the trained model
    """
    if verbose:
        print(f"Training model for {go_id}")

    # Train model
    classifier = MultiClassifier(
        algorithm=algorithm,
        verbose=False,  # Reduce verbosity for batch processing
        proteins=train_proteins_dict,
        config=config,
        id=go_id,
        use_mlflow=False,
        use_tensorboard=False,
    )
    classifier.run()

    # Evaluate model
    if verbose:
        print(f"Evaluating model for {go_id}")

    test_classifier = MultiClassifier(
        algorithm=algorithm,  # Use same algorithm as training
        verbose=False,
        proteins=test_proteins_dict,
        config=config,
        id=go_id,
        use_mlflow=False,
        use_tensorboard=False,
    )

    # Get feature vectors for test data
    test_classifier.get_feature_vectors()
    test_classifier.prepare_data()

    # Regenerate test labels based on actual samples that made it through feature extraction
    # test_classifier.y contains the labels for samples that successfully generated features
    actual_test_labels = np.argmax(test_classifier.y, axis=1)

    # Make predictions
    predictions = classifier.model.predict(test_classifier.X)
    predicted_classes = np.argmax(predictions, axis=1)

    # Verify matching sample counts
    if len(actual_test_labels) != len(predicted_classes):
        print(
            f"ERROR: Label count mismatch: {len(actual_test_labels)} vs {len(predicted_classes)}"
        )
        return 0.0

    # Calculate F1 score using the actual labels that match the predictions
    f1 = f1_score(actual_test_labels, predicted_classes, average="binary")

    return f1


def analyze_go_term(term, dataset, config, algorithm, args):
    """Process a single GO term: collect sequences, train model, evaluate.

    Args:
        term: AnnotatedGOTerm instance to process
        dataset: ProtCastDataset instance
        config: Configuration dictionary
        args: Command line arguments

    Returns:
        dict: Results dictionary with metrics, or None if processing failed
    """
    modelname = f"{term.go_id}_{algorithm}.keras"
    if Path(modelname).is_file():
        if args.verbose:
            print(f"Model {modelname} already exists, skipping...")
        return None

    if args.verbose:
        print(f"Processing term: {term.go_id} ({term.name})")

    # Collect subgraph sequences
    subgraph_sequences = collect_subgraph_sequences(
        dataset, term, args.verbose
    )

    # Check if we have minimum required sequences
    if len(subgraph_sequences) < args.minimum_seqs:
        if args.verbose:
            print(
                f"Skipping {term.go_id}: only {len(subgraph_sequences)} sequences (minimum: {args.minimum_seqs})"
            )
        return None

    # Prepare training data
    positive_sequences, negative_sequences = prepare_training_data(
        subgraph_sequences, dataset, args.max_seqs, args.verbose
    )

    if len(negative_sequences) < len(positive_sequences) // 2:
        if args.verbose:
            print(f"Skipping {term.go_id}: insufficient negative samples")
        return None

    if args.verbose:
        print(
            f"Dataset: {len(positive_sequences)} positive, {len(negative_sequences)} negative"
        )

    # Create datasets for classifier
    train_proteins_dict, test_proteins_dict = create_classifier_datasets(
        positive_sequences, negative_sequences, term.go_id
    )

    try:
        # Train and evaluate model
        f1 = train_and_evaluate_model(
            train_proteins_dict,
            test_proteins_dict,
            config,
            algorithm,
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
            "algorithm": args.algorithm,
        }

        if args.verbose:
            print(f"F1 Score: {f1:.4f}")

        return result

    except Exception as e:
        if args.verbose:
            print(f"Error processing {term.go_id}: {str(e)}")
        return None


def write_result(result):
    """Write a result to TSV.

    Args:
        result: result dictionary
    """
    filename = f"{result['go_id']}_{result['algorithm']}.tsv"
    with open(filename, "w") as f:
        f.write("GO ID\tName\tDepth\tSubgraph\tSequences\tF1 Score\tAlgorithm")
        f.write(f"{result['go_id']}\t{result['go_name']}\t{result['depth']}\t")
        f.write(
            f"{result['subgraph_size']}\t{result['total_sequences']}\t{result['f1_score']}\t{result['algorithm']}"
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
        "-a",
        "--algorithm",
        required=True,
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
            print(f"Depth {depth}: {len(depth_levels[depth])} terms")

    # Iterate through each depth level
    for depth in sorted(depth_levels.keys()):
        if args.verbose:
            print(f"\nProcessing depth level {depth}")

        terms_at_depth = depth_levels[depth]

        if args.verbose:
            print(f"Processing {len(terms_at_depth)} terms at depth {depth}")

        # Process each GO term at this depth
        for term in terms_at_depth:
            result = analyze_go_term(
                term, dataset, config, args.algorithm, args
            )
            if result:
                write_result(result)


if __name__ == "__main__":
    main()
