import argparse

from protcast.preprocessing.protcast_dataset import (  # noqa: E402
    ProtCastDataset,
)


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
