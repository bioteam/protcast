import sys

import argparse

from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.ontology import Ontology


if __name__ == "__main__":
    """test_ontology_topology.py
    Checks that the levels of the GO terms are correctly populated
    """
    parser = argparse.ArgumentParser(
        description="Checks that the levels of the GO terms are correctly populated"
    )
    parser.add_argument(
        "input", metavar="<path/to/input_serialized_ontology_file>"
    )
    args = parser.parse_args()

    ontology = Ontology.load_ontology(args.input)

    # Levels

    # Assert roots are at level 0
    assert (
        ontology.get_primary_term("GO:0008150").level == 0
    )  # Biological Process
    assert (
        ontology.get_primary_term("GO:0005575").level == 0
    )  # Cellular Component
    assert (
        ontology.get_primary_term("GO:0003674").level == 0
    )  # Molecular Function

    # Assert multi parent path to root use topological sorting
    assert (
        # fatty acid ligase activity
        ontology.get_primary_term("GO:0015645").level
        == 5
    )

    # Parents of GO:0015645
    assert (
        ontology.get_primary_term("GO:0140657").level == 1
    )  # ATP-dependent activity
    assert (
        ontology.get_primary_term("GO:0016878").level == 4
    )  # acid-thiol ligase activity

    # Ancestry

    # Roots should have no ancestors
    assert not ontology.get_primary_term(
        "GO:0008150"
    ).ancestors  # Biological Process
    assert not ontology.get_primary_term(
        "GO:0005575"
    ).ancestors  # Biological Process
    assert not ontology.get_primary_term(
        "GO:0003674"
    ).ancestors  # Biological Process

    # GO:0015645
    ancestors = set(
        [
            "GO:0003674",
            "GO:0003824",
            "GO:0016874",
            "GO:0016877",
            "GO:0016878",
            "GO:0140657",
        ]
    )
    # fatty acid ligase activity
    assert set(ontology.get_primary_term("GO:0015645").ancestors) == ancestors

    # GO:0140657
    ancestors = set(["GO:0003674"])
    assert (
        set(ontology.get_primary_term("GO:0140657").ancestors) == ancestors
    )  # fatty acid ligase activity

    # GO:0016878
    ancestors = set(["GO:0003674", "GO:0003824", "GO:0016874", "GO:0016877"])
    assert (
        set(ontology.get_primary_term("GO:0016878").ancestors) == ancestors
    )  # fatty acid ligase activity
