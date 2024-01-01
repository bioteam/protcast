import sys
import argparse
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.ontology import GOTerm, Ontology  # noqa: E402

"""
This test showcases traversing the DAG from a given GOTerm to the root of the tree
"""


def path_to_root(go_term: GOTerm) -> str:
    """test_path_to_root
    ...

    Parameters
    ----------
    name: type
        ...
    ...:
        ...

    Returns
    -------
    None
    """
    parents = list(go_term.get_parents())
    if not parents:
        return go_term.id
    else:
        return go_term.id + "->" + path_to_root(parents[0])


if __name__ == "__main__":
    """path_to_root.py
    Print a path from a GO term to the root
    """
    parser = argparse.ArgumentParser(
        description="Print a path from a GO term to the root"
    )
    parser.add_argument(
        "input", metavar="<path/to/input_serialized_ontology_file>"
    )
    parser.add_argument("id")

    args = parser.parse_args()

    ontology = Ontology.load_ontology(args.input)

    go_term = ontology.get_primary_term(args.id)

    print(path_to_root(go_term))
