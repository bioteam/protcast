import pytest

from protcast.utils.blast_to_go import BlastToGo  # noqa: E402

pytestmark = pytest.mark.integration


def test_blast_to_go_returns_expected_number_of_go_terms():
    """Run BlastToGo against a short sequence and assert we get 3 GO IDs.
    This is an integration test that hits external UniProt/GO APIs.
    """
    seq = (
        "MADTFKEIDAQNAWQLVQERQAFLVDVRDIQRFAYSHPQAAFHLTNQSYGEFCQRCDFEDPIVV"
        "ICYHGNSSRNVAQFLVEQGFDEVYSVRGGFDAWCKAELPLEQGL"
    )

    app = BlastToGo(verbose=True)
    go_ids = app.blast_to_go(seq)
    assert len(go_ids) == 3
