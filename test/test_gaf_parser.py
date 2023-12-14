import sys

import argparse

from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from preprocessing.parse_gaf import parse_gaf


if __name__ == "__main__":
    """test_gaf_parser.py
    Checks that a GAF file is parsed correctly
    """
    parser = argparse.ArgumentParser(
        description="Checks that a GAF file is parsed correctly"
    )
    parser.add_argument(
        "-i", "--input", default="data/goa_uniprot_mini.gaf"
    )
    args = parser.parse_args()

    annotations = parse_gaf(args.input)

    assert len(annotations) == 995
    assert annotations[0]["DB_Object_ID"] == "A0A1Z4V764"
    assert annotations[990]["DB_Object_ID"] == "A0A6N9GJR9"
