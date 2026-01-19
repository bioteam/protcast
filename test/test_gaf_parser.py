from protcast.preprocessing.parse_gaf import parse_gaf  # noqa: E402


def test_gaf_parser_parses_gaf_file():
    """Checks that a GAF file is parsed correctly from test fixtures."""
    input_path = "test/data/goa_uniprot_mini.gaf"

    annotations = parse_gaf(input_path)

    assert len(annotations) == 995
    assert annotations[0]["DB_Object_ID"] == "A0A1Z4V764"
    assert annotations[990]["DB_Object_ID"] == "A0A6N9GJR9"
    assert annotations[994]["DB_Object_ID"] == "M5BGM1"
    assert annotations[994]["GO_ID"] == "GO:0046872"
