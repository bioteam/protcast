import argparse
import logging
import os
from pathlib import Path
import sys

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.ontology import Ontology  # noqa: E402


if __name__ == "__main__":
    """test_ontology.py
    Checks the levels of GO terms, ancestors, and children.
    """
    parser = argparse.ArgumentParser(
        description="Checks the levels of GO terms, ancestors, and children"
    )
    parser.add_argument("-i", "--input", default="data/go.obo")
    parser.add_argument("-o", "--output", default="data/go.obo.bin")
    parser.add_argument("-v", default=False, action="store_true", help="Verbose")
    args = parser.parse_args()

    if args.v:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ontology = Ontology(args.input)
    ontology.save(args.output)

    assert os.path.isfile(Path(args.output))
    os.remove(args.output)

    # level
    # Assert roots are level 0
    assert ontology.get_primary_term("GO:0008150").level == 0 # Biological Process
    assert ontology.get_primary_term("GO:0005575").level == 0 # Cellular Component
    assert ontology.get_primary_term("GO:0003674").level == 0 # Molecular Function
    # Assert levels of ancestors of GO:0031957
    assert ontology.get_primary_term("GO:0003824").level == 1 # catalytic activity
    assert ontology.get_primary_term("GO:0016874").level == 2 # ligase activity
    assert ontology.get_primary_term("GO:0016877").level == 3 # ligase activity, forming carbon-sulfur bonds
    # The 2 parents of GO:0015645
    assert ontology.get_primary_term("GO:0140657").level == 1 # ATP-dependent activity
    assert ontology.get_primary_term("GO:0016878").level == 4 # acid-thiol ligase activity
    # Has 2 parents thus could be 2 or 5
    assert ontology.get_primary_term("GO:0015645").level == 2 # fatty acid ligase activity
    # Has 2 parents, one of which also has 2 parents
    assert ontology.get_primary_term("GO:0031957").level == 3 # very long-chain fatty acid-CoA ligase activity
 
    # ancestors
    # Roots should have no ancestors
    assert not ontology.get_primary_term("GO:0008150").ancestors # Biological Process
    assert not ontology.get_primary_term("GO:0005575").ancestors # Biological Process
    assert not ontology.get_primary_term("GO:0003674").ancestors # Biological Process

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
    assert set(ontology.get_primary_term("GO:0015645").ancestors) == ancestors

    ancestors = set(["GO:0003674"])
    assert (
        set(ontology.get_primary_term("GO:0140657").ancestors) == ancestors
    ) 

    ancestors = set([
        "GO:0003674", 
        "GO:0003824", 
        "GO:0016874", 
        "GO:0016877"
        ])
    assert (
        set(ontology.get_primary_term("GO:0016878").ancestors) == ancestors
    )  

    # children
    assert not ontology.get_primary_term("GO:0031957").children
    assert len(ontology.get_primary_term("GO:0015645").children) == 9

    # parents
    # GO:0016405 CoA-ligase activity, GO:0015645 fatty acid ligase activity
    assert len(ontology.get_primary_term("GO:0031957").parents) == 2
    assert len(ontology.get_primary_term("GO:0015645").parents) == 2
    assert not ontology.get_primary_term("GO:0003674").parents

    # is_leaf 
    assert ontology.get_primary_term("GO:0031957").is_leaf
    assert not ontology.get_primary_term("GO:0015645").is_leaf
