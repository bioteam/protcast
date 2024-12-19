import sys
import argparse
import logging
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.protcast_dataset import (
    ProtCastDataset,
)  # noqa: E402


def main():
    """"create_subgraph_sequences_files.py

    This script takes:
    - Serialized ProtCastDataset file
    - A text file with a list of GO IDs

    It returns:
    - A FASTA file containing the input sequences

    Example:

    python scripts/create_subgraph_sequences_files.py \
    -p data/dataset/u-2021-04-g-2021-10-26/dataset.bin \
    -i GO_ids.txt 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to serialized dataset")
    parser.add_argument("seq_ids", help="Path to file with a list of GO ids")
    parser.add_argument("output", help="Output file")
    parser.add_argument("-v", action="store_true", help="Verbose")
    args = parser.parse_args()

    if args.v:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info("Deserializing 'ProtCastDataset'...")
    dataset = ProtCastDataset.load_serialized_file(args.dataset)
    logging.info("Done deserializing 'ProtCastDataset'...")

    logging.info(
        "Converting input sequences IDs to their primary accessions..."
    )
    with open(args.seq_ids, "r") as input_file:
        with open(args.output, "w") as output_seq_file:
            with open(args.output + ".fasta", "w") as output_fasta_file:
                for in_seq in input_file.readlines():
                    in_seq = in_seq.strip()
                    out_seq = dataset.accessions[in_seq]
                    if in_seq != out_seq:
                        logging.info(
                            f"Found primary accession: {out_seq} and secondary accession: {in_seq}"
                        )
                    output_seq_file.write(out_seq + "\n")
                    output_fasta_file.write(f">{out_seq}\n")
                    output_fasta_file.write(
                        f"{dataset.proteins[out_seq].sequence}\n"
                    )


if __name__ == "__main__":
    main()
