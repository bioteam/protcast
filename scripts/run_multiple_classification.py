import sys
import argparse
from collections import defaultdict
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from protcast.model.multi_classifier import MultiClassifier  # noqa: E402
from protcast.preprocessing.protcast_dataset import (  # noqa: E402
    load_serialized_file,
)

if __name__ == "__main__":
    """"run_multiple_classification.py
    This script uses TensorFlow and Keras FeatureSpace to classify sequences.
    Provide a text file with GO ids, the subgraph sequences will be used
    for training and testing. Example:

    python3 scripts/run_multiple_classification.py \
    -g test/data/go-terms.txt \
    -p ProtcastDataset.bin \
    -a qsorder -f ifeatpro -s
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g", "--go_ids_file", required=True, help="Path to GO ids file"
    )
    parser.add_argument(
        "-a", "--algorithm", default="ctriad", help="Feature vector algorithm"
    )
    parser.add_argument(
        "-p",
        "--protcast_dataset",
        required=True,
        help="Path to ProtCast dataset",
    )
    parser.add_argument(
        "-f",
        "--feature_creator",
        default="iFeatureOmega",
        choices=["iFeatureOmega", "ifeatpro"],
        help="Feature vector creator",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    parser.add_argument("-s", "--save", action="store_true", help="Save model")
    args = parser.parse_args()

    dataset = load_serialized_file(args.protcast_dataset)

    go_ids = [
        line.strip()
        for line in open(args.go_ids_file, "r")
        if line.startswith("GO:")
    ]
    # Primary keys are GO ids, secondary keys are protein ids, values are sequences
    proteins = defaultdict(dict)
    for go_id in go_ids:
        subgraph_ids = dataset.get_subgraph(go_id)
        for subid in subgraph_ids:
            seqs = dataset.get_term(subid).get_all_sequences()
            proteins[go_id].update(seqs)

    classifier = MultiClassifier(
        args.algorithm,
        args.feature_creator,
        args.verbose,
        args.save,
        proteins,
    )
    classifier.run()
    classifier.test_model()
    classifier.save_model()
