import sys

import argparse
import logging

from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.ontology import Ontology


if __name__ == "__main__":
    """"ontology_to_files.py
    This script can be used to create the ontology files needed by the CAFA2 
    repository function pfp_ontbuild() to build an ontology. Example:

    python -O preprocessing-scripts/ontology_to_files.py \
    data/ontology/t0/go_20211026.obo \
    data/ontology/t0/ontology_20211026
    """
    parser = argparse.ArgumentParser(
        description="Write an ontology to CAFA2 files"
    )
    parser.add_argument("input_obo_file")
    parser.add_argument("output_files")
    parser.add_argument("-v", action="store_true", help="Verbose")
    args = parser.parse_args()

    if args.v:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ontology = Ontology(args.input_obo_file)
    ontology.to_files(args.output_files)
