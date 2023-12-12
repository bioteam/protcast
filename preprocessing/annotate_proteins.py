import sys

import argparse
import pickle

from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.dataset import Dataset

def main():
    """annotate_proteins.py
    ...

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("proteins_path", help="Path to serialized proteins dictionary")
    parser.add_argument("goa", help="Path to GOA databse")
    parser.add_argument("output", help="Output swissprot file")
    parser.add_argument("-v", default=False, action="store_true", help="Verbose")

    args = parser.parse_args()

    with open(args.proteins_path, "rb") as f:
        proteins = pickle.load(f)

    annotate_proteins_from_goa(proteins, args.goa)


if __name__ == "__main__":
    main()
