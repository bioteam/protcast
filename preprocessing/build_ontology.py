from protcast.preprocessing.ontology import Ontology
import sys
import argparse
import logging
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))


if __name__ == "__main__":
    """build_ontology.py
    Builds an Ontology object and saves it to disk given an input *obo file.
    Example:

    python3 preprocessing-scripts/build_ontology.py data/go.obo data/go.obo.bin
    """
    parser = argparse.ArgumentParser(
        description="Build an ontology from an *.obo file and save"
    )
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    parser.add_argument(
        "-v", default=False, action="store_true", help="Verbose"
    )
    args = parser.parse_args()

    if args.v:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ontology = Ontology(args.input)
    ontology.save(args.output)
