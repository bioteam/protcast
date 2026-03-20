import pytest

from protcast.utils.blast_to_go import BlastToGo  # noqa: E402

pytestmark = pytest.mark.integration


def test_blast_to_go_returns_expected_number_of_go_terms():
    """Run BlastToGo against a short sequence and assert we get GO IDs.
    This is an integration test that hits external UniProt/GO APIs.
    The exact count may change as UniProt annotations are updated,
    so we assert at least 1 GO term is returned.
    """
    seq = (
        "MADTFKEIDAQNAWQLVQERQAFLVDVRDIQRFAYSHPQAAFHLTNQSYGEFCQRCDFEDPIVV"
        "ICYHGNSSRNVAQFLVEQGFDEVYSVRGGFDAWCKAELPLEQGL"
    )

    app = BlastToGo(verbose=True)
    go_ids = app.blast_to_go(seq)
    assert len(go_ids) >= 1, f"Expected at least 1 GO term, got {go_ids}"
