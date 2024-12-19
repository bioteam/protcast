import argparse
import sys
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.protcast_dataset import (
    ProtCastDataset,
)  # noqa: E402


def main():
    """ProtCastDataset2obo.py
    Create an *obo file from a serialized ProtCastDataset.
    The *xref* lines are the ids of the proteins annotated
    with that term.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Input file",
        default="ProtCastDataset.bin",
    )
    args = parser.parse_args()

    dataset = ProtCastDataset.load_serialized_file(args.input)
    dataset.to_obo()


if __name__ == "__main__":
    main()
