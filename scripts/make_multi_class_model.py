import re
import sys
import os
import argparse
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from protcast.model.multi_classifier import MultiClassifier  # noqa: E402
from protcast.preprocessing.protcast_dataset import (  # noqa: E402
    ProtCastDataset,
)

if __name__ == "__main__":
    """"make_multi_class_model.py
    Provide a Fasta file, or a text file with GO ids and a ProtCastDataset file. 
    Example to use a ProtCastDataset:

    python3 scripts/make_multi_class_model.py \
    -g test/data/go-terms.txt \
    -p ProtcastDataset.bin \
    -a qsorder -f ifeatpro -s

    Or use a sequence (Fasta) file:

    python3 scripts/make_multi_class_model.py \
    -s test/data/random-level-4.fa \
    -v

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--go_ids_file", help="Path to GO ids file")
    parser.add_argument("-s", "--seq_file", help="Path to Fasta file")
    parser.add_argument(
        "-a", "--algorithm", default="qsorder", help="Feature vector algorithm"
    )
    parser.add_argument(
        "-p",
        "--protcast_dataset",
        help="Path to ProtCast dataset",
    )
    parser.add_argument(
        "-f",
        "--feature_creator",
        default="ifeatpro",
        choices=["iFeatureOmega", "ifeatpro"],
        help="Feature vector creator",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    args = parser.parse_args()

    # Primary keys are GO ids, secondary keys are protein ids, values are sequences
    proteins = defaultdict(dict)

    # GO id subgraph sequences are collected from a ProtCastDataset
    if args.go_ids_file is not None and args.protcast_dataset is not None:
        dataset = ProtCastDataset.load_serialized_file(args.protcast_dataset)
        go_ids = [
            line.strip()
            for line in open(args.go_ids_file, "r")
            if line.startswith("GO:")
        ]
        for go_id in go_ids:
            subgraph_go_ids = dataset.get_subgraph(go_id)
            for subid in subgraph_go_ids:
                pids = dataset.get_term(subid).get_all_pids()
                if pids:
                    seqs = {
                        pid: dataset.proteins[pid].sequence
                        for pid in pids
                        if pid in dataset.proteins
                    }
                    proteins[go_id].update(seqs)
    # Sequences are collected from a fasta file where the description contains a GO id
    elif args.seq_file is not None:
        from Bio import SeqIO

        for seq in SeqIO.parse(args.seq_file, "fasta"):
            go_id = re.search("GO:\d+", seq.description)[0]
            if go_id is not None:
                proteins[go_id][seq.id] = str(seq.seq)
    else:
        sys.exit("Need input files")

    classifier = MultiClassifier(
        args.algorithm,
        args.feature_creator,
        args.verbose,
        proteins,
    )
    classifier.run()
    # Not necessary with the checkpoints in place
    # classifier.save_model()
