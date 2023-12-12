import sys

import argparse
import logging

from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.dataset import Dataset


def main():
    """"create_annotation_files.py
    Creates annotation files from the input databases or from a serialized serialized 'Dataset'
    file. Example:

    python -O preprocessing-scripts/create_annotation_files.py \
    -o data/ontology/t0/go_20211026.obo \
    -s0 data/uniprotkb/t0/uniprot_sprot.dat \
    -s1 data/uniprotkb/t1/uniprot_sprot.dat \
    -t data/trembl/trembl_seqs_found_in_goa_2021_10_26.fasta \
    -g data/goa/t0/goa_uniprot_all_noiea.gaf \
    data/annotations/annotations_u_2021_04_g_2021_10_26

    or:

    python -O preprocessing-scripts/create_annotation_files.py \
    -f data/dataset/u-2021-04-g-2021-10-26/dataset.bin \
    test
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--ontology", help="Path to ontology file (.obo file)")
    parser.add_argument(
        "-s0", "--swissprot_t0", help="Path to the SwissProt DB at t0 (.dat file)"
    )
    parser.add_argument(
        "-s1", "--swissprot_t1", help="Path to the SwissProt DB at t1 (.dat file)"
    )
    parser.add_argument("-t", "--trembl", help="Path to TrEMBL DB (.fasta file)")
    parser.add_argument(
        "-g", "--goa", help="Path to Gene Ontology Annotation (.gaf file)"
    )
    parser.add_argument("-f", "--file", help="Path to serialized 'Dataset' file")
    parser.add_argument("output", help="Output annotation file")
    parser.add_argument("-v", default=False, action="store_true", help="Verbose")
    args = parser.parse_args()

    all_source_flags = (
        args.ontology
        and args.swissprot_t0
        and args.swissprot_t1
        and args.trembl
        and args.goa
    )
    serialized_flags = args.file
    if (all_source_flags and serialized_flags) or (
        not serialized_flags and not all_source_flags
    ):
        print("Provide either '-f' flag or '-o', '-s0', '-s1', '-t' and '-g' flags")
        exit(1)

    if args.v:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.file:
        logging.info(f"Deserializing dataset from file: {args.file}")
        dataset = Dataset.from_serialized_file(args.file)
    else:
        logging.info(f"Building dataset from input DBs.")
        dataset = Dataset(
            Path(args.ontology),
            Path(args.swissprot_t0),
            Path(args.swissprot_t1),
            Path(args.trembl),
            Path(args.goa),
        )

    logging.info(f"Creating annotation files {args.output}_{{bpo, cco, mfo}}.tsv")
    dataset.create_annotation_files(args.output)


if __name__ == "__main__":
    main()
