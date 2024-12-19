import sys
import argparse
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.protcast_dataset import (  # noqa: E402
    ProtCastDataset,
)


""""create_subgraph_sequences_files.py

This script takes:
- Serialized ProtCastDataset file
- A text file with a list of GO IDs

It returns:
- One or FASTAs file containing the input sequences

Example:

python scripts/create_subgraph_sequences_files.py \
-p data/dataset/u-2021-04-g-2021-10-26/dataset.bin \
-g GO_ids.txt 
"""
parser = argparse.ArgumentParser()
parser.add_argument("p", "protcast_dataset", help="Path to serialized dataset")
parser.add_argument("g", "go_ids", help="Path to file with a list of GO ids")
parser.add_argument("-v", action="store_true", help="Verbose")
args = parser.parse_args()

dataset = ProtCastDataset.load_serialized_file(args.protcast_dataset)

with open(args.go_ids, "r") as input_file:
    with open(args.output, "w") as output_seq_file:
        with open(args.output + ".fasta", "w") as output_fasta_file:
            for in_seq in input_file.readlines():
                in_seq = in_seq.strip()
                out_seq = dataset.accessions[in_seq]
                output_seq_file.write(out_seq + "\n")
                output_fasta_file.write(f">{out_seq}\n")
                output_fasta_file.write(
                    f"{dataset.proteins[out_seq].sequence}\n"
                )
