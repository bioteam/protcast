from __future__ import annotations

import sys

import argparse
import logging as log

from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.protcast_dataset import DeepredDataset


def main():
    """write_protcast_dataset_to_file.py
    This script takes a DeepredDataset serialized file and writes it to text
    for model training.

    Example:
    python preprocessing-scripts/write_protcast_dataset_to_file.py \
        data/protcast-dataset/u-2021-04-g-2021-10-26/protcast_dataset.bin \
        data/protcast-dataset/u-2021-04-g-2021-10-26/hdf5/ \
        -d

    python preprocessing-scripts/write_protcast_dataset_to_file.py \
        data/protcast-dataset/u-2021-04-g-2021-10-26/protcast_dataset.bin \
        data/protcast-dataset/u-2021-04-g-2021-10-26/text-files/ \
        -t
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "protcast_file", help="Path to serialized 'DeepredDataset'"
    )
    parser.add_argument(
        "output_dir", help="Output directory for the Deepred dataset"
    )
    parser.add_argument(
        "-t", default=False, action="store_true", help="Generate text files"
    )
    parser.add_argument(
        "-d", default=False, action="store_true", help="Generate hdf5 files"
    )
    parser.add_argument(
        "-v", default=False, action="store_true", help="Verbose"
    )
    args = parser.parse_args()

    if not (args.t or args.d):
        print("Provide either '-t flag or '-d' flags")
        exit(1)

    if args.v:
        log.basicConfig(level=log.DEBUG)
    else:
        log.basicConfig(level=log.INFO)

    log.info(f"Deserializing DeepredDataset from file: {args.protcast_file}")
    model_dataset: DeepredDataset = (
        DeepredDataset.from_serialized_protcast_dataset(Path(args.protcast_file))
    )

    if args.t:
        log.info("Writing submodels dataset to text files")
        model_dataset.write_datasets_to_files(Path(args.output_dir))

    if args.d:
        log.info("Writing submodels dataset to HDF5 file")
        model_dataset.write_datasets_to_hdf5(Path(args.output_dir))


if __name__ == "__main__":
    main()
