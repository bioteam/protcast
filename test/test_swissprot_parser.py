from pathlib import Path

from protcast.preprocessing.parse_swissprot import parse_swissprot  # noqa: E402
from protcast.preprocessing.annotated_godag import AnnotatedGODag  # noqa: E402


def test_swissprot_parser_parses_swissprot_file():
    """Checks that a Swissprot file is parsed correctly using fixtures in test/data."""
    input_path = Path("test/data/uniprot_mini.dat")
    ontology_path = Path("test/data/go-2023-11-15.obo")

    ontology = AnnotatedGODag(ontology_path)

    annotations, proteins = parse_swissprot(input_path)

    # Totals
    # All 5 terms are obsolete
    # assert len(go_terms_not_found) == 5
    assert len(annotations) == 444
    assert len(proteins) == 142

    assert proteins["A0A016QRH0"].id == "A0A016QRH0"
    assert len(proteins["A0A016QRH0"].annotations) == 3
    annots = proteins["A0A016QRH0"].get_all_annotations()
    assert len(annots) == 3
    assert annots[0].evidence_code == "IEA"
    assert annots[0].is_manual is False
    assert annots[2].go_id == "GO:0015379"
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
