import argparse
import logging as log
from pathlib import Path
import sys

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from preprocessing.stats.create_stats_files import create_stats_files  # noqa: E402


def main():
    """
    Generate statistics files for an existing SimpleDataset. Example:
    
    python utility-scripts/create_dataset_stats.py \
    -i data/dataset/u-2021-04-g-2021-10-26/dataset.bin \
    """
    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to serialized dataset file", required=True)
    args = parser.parse_args()

    log.info(f"Generating dataset statistics from: {args.input}")
    create_stats_files(args.input)


if __name__ == "__main__":
    main()
