import sys
import argparse
import random
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.protcast_dataset import (  # noqa: E402
    ProtCastDataset,
)


""""create_subgraph_sequences_files.py

This script takes:

- Path to serialized ProtCastDataset file
- Path to a text file with a list of GO IDs

It returns for each GO id:

- A FASTA file containing all the sequences in the DAG associated with the GO term
- A FASTA file containing an equal number of sequences in the DAG not associated with the GO term

Example:

python scripts/create_subgraph_sequences_files.py \
-p data/dataset/u-2021-04-g-2021-10-26/dataset.bin \
-g GO_ids.txt 
"""
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_seqs", default=500, help="Number of sequences")
parser.add_argument(
    "-p", "--protcast_dataset", required=True, help="Path to serialized dataset"
)
parser.add_argument(
    "-g", "--go_ids", required=True, help="Path to file with a list of GO ids"
)
parser.add_argument("-v", action="store_true", help="Verbose")
args = parser.parse_args()

dataset = ProtCastDataset.load_serialized_file(args.protcast_dataset)

go_ids = list()
with open(args.go_ids, "r") as f:
    for line in f:
        go_ids.append(line.strip())


for go_id in go_ids:
    target_go_ids = dataset.get_subgraph(go_id)
    go_terms = dataset.get_terms(target_go_ids)
    annots = [go_term.get_all_annotations() for go_term in go_terms]
    target_seq_ids = [
        annot.protein_id for annot in [x for y in annots for x in y]
    ]

    non_target_go_ids = dataset.get_inverse_subgraph(go_id)
    go_terms = dataset.get_terms(non_target_go_ids)
    annots = [go_term.get_all_annotations() for go_term in go_terms]
    non_target_seq_ids = [
        annot.protein_id for annot in [x for y in annots for x in y]
    ]

    random_non_target_seq_ids = [
        random.choice(non_target_go_ids) for x in range(len(target_seq_ids))
    ]

    with open(f"{id}_subgraph.fa", "w") as target_seq_file:
        for id in target_seq_ids:
            target_seq_file.write(f">{id}\n")
            target_seq_file.write(f"{dataset.proteins[id].sequence}\n")

    with open(f"{id}_inv_subgraph.fa", "w") as non_target_seq_file:
        for id in random_non_target_seq_ids:
            non_target_seq_file.write(f">{id}\n")
            non_target_seq_file.write(f"{dataset.proteins[id].sequence}\n")
