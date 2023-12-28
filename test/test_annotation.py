import sys
import logging

from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.ontology import GOTerm
from protcast.preprocessing.protein import Protein
from protcast.preprocessing.annotation import Annotation


if __name__ == "__main__":
    """test_annotation.py
    Builds Protein and Annotation objects and tests.
    Example:

    python3 test_annotation.py
    """
    logging.basicConfig(level=logging.DEBUG)

    protein1 = Protein("abc", "MLK")
    term1 = GOTerm("GO:123456", "molecular function", "kinase", None, 3, True)
    term2 = GOTerm(
        "GO:456789", "molecular function", "phosphatase", None, 3, True
    )
    annot1 = Annotation("abc", "IEA", term1)
    annot2 = Annotation("abc", "IEA", term2)
    annot3 = Annotation("abc", "EXP", term1)
    annot4 = Annotation("abc", "IEA", term1)

    protein1.add_annotation(annot1)
    assert len(protein1.get_all_annotations()) == 1
    assert protein1.has_annotation(annot1)
    assert protein1.has_annotation(annot3) == False
    assert protein1.has_annotation(annot2) == False
    assert protein1.has_annotation(annot4)

    protein1.add_annotation(annot2)
    assert len(protein1.get_all_annotations()) == 2
    assert protein1.has_annotation(annot1)
    assert protein1.has_annotation(annot2)
    assert protein1.has_annotation(annot3) == False
    assert protein1.has_annotation(annot4)

    protein1.add_annotation(annot3)
    assert len(protein1.get_all_annotations()) == 3
    assert protein1.has_annotation(annot1)
    assert protein1.has_annotation(annot2)
    assert protein1.has_annotation(annot3)
    assert protein1.has_annotation(annot4)

    assert len(protein1.get_all_go_ids()) == 3
    assert len(set(protein1.get_all_go_ids())) == 2
