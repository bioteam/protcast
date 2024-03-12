import sys
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.model.stats.blast_to_go import BlastToGo  # noqa: E402


if __name__ == "__main__":
    """test_blast_to_go.py
    Run BlastToGo, which gets GO terms from best, non-identical match using Uniprot API
    """

    seq = "MADTFKEIDAQNAWQLVQERQAFLVDVRDIQRFAYSHPQAAFHLTNQSYGEFCQRCDFEDPIVVICYHGNSSRNVAQFLVEQGFDEVYSVRGGFDAWCKAELPLEQGL"

    app = BlastToGo(verbose=True)
    go_ids = app.blast_to_go(seq)
    assert len(go_ids) == 3