import sys
import argparse
from pathlib import Path
from Bio import SeqIO

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.model.binary_classifier import BinaryClassifier  # noqa: E402

if __name__ == "__main__":
    """"binary_classify.py
    This script uses Keras and Keras FeatureSpace to classify sequences.
    Provide a *fasta file with some group of related sequences ("target")
    and a second *fasta file with unrelated or control sequences ("non-target").
    Example:

    python scripts/binary_classify.py \
    -t gpcr-sequences.fasta \
    -nt non-gpcr-sequences.fasta \
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--target", required=True, help="Path to target sequences")
    parser.add_argument("-nt","--non_target", required=True, help="Path to non-target sequences")
    parser.add_argument("-a", "--algorithm", default="ctriad", help="Feature vector algorithm")
    parser.add_argument("-n", "--name", default="test", help="Name")
    parser.add_argument("-v", action="store_true", help="Verbose")
    args = parser.parse_args()

    target_seqs = SeqIO.to_dict(SeqIO.parse(args.target, "fasta"))
    non_target_seqs = SeqIO.to_dict(SeqIO.parse(args.non_target, "fasta"))

    classifier = BinaryClassifier(args.name, target_seqs, non_target_seqs, alg)
    classifier.run()
    classifier.test_model()
    classifier.save_model()
