from __future__ import annotations

import sys

import argparse
import logging
import numpy as np

from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.model.protcast import Deepred

from protcast import BP, CC, MF
from protcast.preprocessing.oracle_dataset import OracleDataset


def main():
    """train_submodel.py
    This script can build a network from:
    - Deepred Dataset: A serialized file of an DeepredDataset
    - Dataset: A serialized file of a Dataset

    Example:

    python3 training-scripts/train_submodel.py -d data/dataset/dataset.bin
    """
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-d", help="Input 'Dataset' serialized file"
    )
    input_group.add_argument(
        "-m", help="Input 'DeepredDataset' serialized file"
    )
    parser.add_argument(
        "-v", default=False, action="store_true", help="Verbose"
    )
    args = parser.parse_args()

    if args.v:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.d:
        model_dataset = OracleDataset.from_serialized_dataset(args.d)
    else:
        model_dataset = OracleDataset.from_serialized_oracle_dataset(
            args.m
        )

    # model_dataset.summary()
    with open(
        "ctriad/model_biological_process_2_30_0_x_hat.mat", "r"
    ) as f:
        x_hat = [
            [float(x) for x in line.strip("\n").split(" ")]
            for line in f.readlines()
        ]
        x_hat = np.array(x_hat)

    with open(
        "ctriad/model_biological_process_2_30_0_y_hat.mat", "r"
    ) as f:
        y_hat = [
            [int(x) for x in line.strip("\n").split(" ")]
            for line in f.readlines()
        ]
        y_hat = np.array(y_hat)

    model = Deepred(model_dataset)
    model.fit_submodel(
        namespace=BP,
        level=2,
        bucket=30,
        index=0,
        x_hat=x_hat,
        y_hat=y_hat,
    )


if __name__ == "__main__":
    main()
