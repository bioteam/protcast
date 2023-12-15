import os
import sys

import argparse
import logging as log
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[2]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.deepred_dataset import DeepredDataset
from protcast.stats import stats


def main():
    """
    This scripts generate statistics from an already built DeepredDataset. The
    files generated are:
    - general.txt: Contains:
        - Creation time: Time when the dataset was generated.
        - Ontology File: Relative path to the OBO file used to generate the
        dataset and its hash.
        - Swissprot File: Relative path to the '.dat' used to generate the
        dataset and its hash.
        - GOA File: Relative path to the '.dat' used to generate the dataset
        and its hash.

        Example:
        python preprocessing-scripts/stats/create_protcast_dataset_stats.py \
            data/protcast-dataset/u-2021-04-g-2021-10-26/protcast_dataset.bin \
            data/protcast-dataset/u-2021-04-g-2021-10-26/stats/ \
            -w
    """
    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", help="Path to serialized protcast dataset file"
    )
    parser.add_argument("output_dir", help="Output dataset dir")
    parser.add_argument(
        "-w", default=False, action="store_true", help="Overwrite dir"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        if args.w:
            os.makedirs(output_dir)
        else:
            print(f"Output directory: '{output_dir}' not found")
            exit(1)

    if len(os.listdir(output_dir)) > 0:
        if not args.w:
            print(
                "Output directory not empty. Provide the '-w' flag to "
                "overwrite"
            )
            exit(1)

    log.info(
        f"Deserializing DeepredDataset from file: {args.dataset}"
    )
    dataset = DeepredDataset.from_serialized_protcast_dataset(
        args.dataset
    )
    log.info("Generating DeepredDataset statistics")
    stats.generate_protcast_dataset_stats(dataset, output_dir)


if __name__ == "__main__":
    main()
