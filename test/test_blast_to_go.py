import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from protcast.utils.blast_to_go import BlastToGo  # noqa: E402


"""test_blast_to_go.py
Run BlastToGo, which gets GO terms from best, non-identical match using Uniprot API
"""

seq = "MADTFKEIDAQNAWQLVQERQAFLVDVRDIQRFAYSHPQAAFHLTNQSYGEFCQRCDFEDPIVVICYHGNSSRNVAQFLVEQGFDEVYSVRGGFDAWCKAELPLEQGL"

app = BlastToGo(verbose=True)
go_ids = app.blast_to_go(seq)
assert len(go_ids) == 3
