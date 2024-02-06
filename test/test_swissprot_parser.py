import argparse
from pathlib import Path
import sys

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from preprocessing.parse_swissprot import parse_swissprot  # noqa: E402
from protcast.preprocessing.annotated_godag import Ontology  # noqa: E402

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

    proteins, go_terms_not_found, accessions = parse_swissprot(
        ontology, Path(args.input)
    )

    assert len(go_terms_not_found) == 0
    assert len(proteins) == 142
    assert len(accessions) == 143

    assert accessions["A0A017QRH0"] == "A0A016QRH0"
    assert accessions["A0A016QRH0"] == "A0A016QRH0"

    assert proteins["A0A016QRH0"].id == "A0A016QRH0"
    assert len(proteins["A0A016QRH0"].annotations) == 3
    annots = proteins["A0A016QRH0"].get_all_annotations()
    assert len(annots) == 3
    assert annots[0].evidence_code == "IEA"
    assert annots[0].is_manual is False
    assert annots[2].go_term_id == "GO:0015379"
    manuals = proteins["A0A016QRH0"].get_manual_annotations()
    assert manuals == []
    assert len(proteins["A0A016QRH0"].get_electronic_annotations()) == 3

    annots = proteins["A0A1D6P109"].get_all_annotations()
    assert len(annots) == 9
    assert annots[2].evidence_code == "IEA"
    assert annots[8].evidence_code == "IBA"
    assert annots[8].is_manual is True
    manuals = proteins["A0A1D6P109"].get_manual_annotations()
    assert len(manuals) == 8
    assert len(proteins["A0A1D6P109"].get_electronic_annotations()) == 1

    assert (
        proteins["A0A0A2ZXP0"].sequence
        == "MADTFKEIDAQNAWQLVQERQAFLVDVRDIQRFAYSHPQAAFHLTNQSYGEFCQRCDFEDPIVVICYHGNSSRNVAQFLVEQGFDEVYSVRGGFDAWCKAELPLEQGL"
    )
