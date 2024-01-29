import argparse
import logging as log
from pathlib import Path
import sys

file = Path(__file__).resolve()
package_root_directory = file.parents[2]
sys.path.append(str(package_root_directory))

from protcast.stats import stats  # noqa: E402


def main():
    """
    This scripts generate statistics from an already built dataset. The files
    generated are:
    - general.txt: Contains:
        - Creation time: Time when the dataset was generated
        - Ontology File: Relative path to the OBO file used to generate the
        dataset and its hash.
        - Swiss-Prot File: Relative path to the '.dat' used to generate the
        dataset and its hash.
        - GOA File: Relative path to the '.dat' used to generate the dataset
        and its hash.

    Example:
    python preprocessing/stats/create_dataset_stats.py \
    -i data/dataset/u-2021-04-g-2021-10-26/dataset.bin \
    """
    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to serialized dataset file")
    args = parser.parse_args()

    log.info(f"Generating dataset statistics from: {args.input}")
    stats.generate_dataset_stats(args.input)


if __name__ == "__main__":
    main()
