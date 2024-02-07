import argparse
import sys
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.simple_dataset import SimpleDataset  # noqa: E402


def main():
    """"create_simple_dataset.py
    Creates a dataset that can be used by Keras FeatureSpace, which expects
    protein features and GO terms to be passed in as a dataframe. Example:

    python3 preprocessing-scripts/create_simple_dataset.py \
    -o /data/GO/2023-11-15/go.obo \
    -s /data/UniProt/2023-11-15/uniprot_sprot.dat \
    -t /data/UniProt/2023-11-15/uniprot_trembl.fasta \
    -g /data/GO/2023-11-15/filtered_goa_uniprot_all_noiea.gaf \
    -O /data/11-28-2023
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--ontology",
        type=Path,
        help="Path to Gene Ontology file (*.obo)",
        required=True
    )
    parser.add_argument(
        "-s",
        "--swissprot",
        type=Path,
        help="Path to the SwissProt file (*.dat)",
        required=True
    )
    parser.add_argument(
        "-t", "--trembl", type=Path, help="Path to the TrEMBL file (*.fa)", required=True
    )
    parser.add_argument(
        "-g",
        "--gaf",
        type=Path,
        help="Path to GO Annotation Format file (*.gaf)",
        required=True
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Create DEBUG log",
    )
    parser.add_argument(
        "-n",
        "--no_propogate",
        default=False,
        action="store_true",
        help="Propogate annotations",
    )
    parser.add_argument(
        "-O", "--output_dir", type=Path, help="Output directory"
    )
    args = parser.parse_args()

    dataset = SimpleDataset(
        args.ontology,
        args.swissprot,
        args.trembl,
        args.gaf,
        args.output_dir,
        args.no_propogate,
        args.verbose,
    )

    dataset.save()


if __name__ == "__main__":
    main()
