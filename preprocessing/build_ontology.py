import sys

import argparse
import logging

from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.ontology import Ontology


if __name__ == "__main__":
    """build_ontology.py
    Builds an Ontology object and saves it to disk given an input *obo file.
    Example:

    python3 preprocessing-scripts/write_protcast_dataset_to_file.py \
    test.obo \
    test.bin
    """
    parser = argparse.ArgumentParser(
        description="Build an ontology from an \
        `.obo` file"
    )
    parser.add_argument("input_obo_file")
    parser.add_argument("output_serialized_ontology")
    parser.add_argument("-v", default=False, action="store_true", help="Verbose")
    args = parser.parse_args()

    if args.v:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ontology = Ontology(args.input_obo_file)
    ontology.save(args.output_serialized_ontology)
