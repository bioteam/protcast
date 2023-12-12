import argparse
from pathlib import Path
import sys

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from preprocessing.parse_swissprot import parse_swissprot
from protcast.preprocessing.ontology import Ontology


if __name__ == "__main__":
    """test_swissprot_parser.py
    Checks that a Swissprot file is parsed correctly
    """
    parser = argparse.ArgumentParser(
        description="Checks that a Swissprot file is parsed correctly"
    )
    parser.add_argument("-i", "--input", default="data/uniprot_mini.dat")
    parser.add_argument("-o", "--ontology", default="data/go.obo")
    args = parser.parse_args()

    ontology = Ontology(args.ontology)

    proteins, go_terms_not_found, accessions = parse_swissprot(ontology, Path(args.input))

    assert len(go_terms_not_found) == 0    
    assert len(proteins) == 141
    assert proteins['A0A016QRH0'].id == 'A0A016QRH0'
    assert len(proteins['A0A016QRH0'].annotations.keys()) == 3
    assert proteins['A0A0A2ZXP0'].sequence == "MADTFKEIDAQNAWQLVQERQAFLVDVRDIQRFAYSHPQAAFHLTNQSYGEFCQRCDFEDPIVVICYHGNSSRNVAQFLVEQGFDEVYSVRGGFDAWCKAELPLEQGL"
