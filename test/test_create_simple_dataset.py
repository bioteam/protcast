import argparse
import os
import sys
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.simple_dataset import SimpleDataset


def main():
    """"test_create_simple_dataset.py
    Creates a dataset that can be used by Keras FeatureSpace, which expects
    protein features and GO terms to be passed in as a dataframe.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--ontology", help="Path to Gene Ontology file (*.obo)",
        default='data/go.obo'
    )
    parser.add_argument(
        "-s", "--swissprot", help="Path to the SwissProt file (*.dat)",
        default='data/uniprot_mini.dat'
    )
    parser.add_argument(
        "-t", "--trembl", help="Path to the TrEMBL file (*.fa)",
        default='data/uniprot_trembl_mini.fasta'
    )
    parser.add_argument(
        "-g", "--gaf", help="Path to GOA format file (*.gaf)",
        default='data/goa_uniprot_mini.gaf'
    )
    parser.add_argument(
        "--output_dir", help="Output directory",
        default='data/'
    )
    parser.add_argument(
        "-v", "--verbose", default=True,
        help="Create DEBUG log",
    )
    args = parser.parse_args()

    dataset = SimpleDataset(
        Path(args.ontology),
        Path(args.swissprot),
        Path(args.trembl),
        Path(args.gaf),
        Path(args.output_dir),
        args.verbose,
    )

    dataset.save()

    os.unlink(Path(args.output_dir, 'SimpleDataset.bin'))
    os.unlink(Path(args.output_dir, 'SimpleDataset.log'))

if __name__ == "__main__":
    main()
