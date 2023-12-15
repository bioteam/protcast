import sys

import argparse

from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.simple_dataset import SimpleDataset


if __name__ == "__main__":
    """test_annot_propagation.py
    Checks that a protein is annotated with all ancestor GO Terms
    """
    parser = argparse.ArgumentParser(
        description="Checks that a protein is annotated with all ancestor GO Terms"
    )
    parser.add_argument(
        "input", metavar="<path/to/input_serialized_dataset_file>"
    )
    args = parser.parse_args()

    dataset = Dataset.from_serialized_file(args.input)

    # Original Annot: Q8HZM6 -> GO:0002548
    protein = dataset.proteins.get("Q8HZM6")
    go_term_ancestors = dataset.ontology.get_primary_term(
        "GO:0002548"
    ).ancestors

    protein_annots = set(protein.annotations.keys())

    assert protein_annots.issuperset(go_term_ancestors)
