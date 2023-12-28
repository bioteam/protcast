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
        "input",
        metavar="<path/to/input_serialized_dataset_file>",
    )
    args = parser.parse_args()

    dataset = SimpleDataset.from_serialized_file(args.input)

    # A0A098C095 is annotated with a single GO term (GO:0009289)
    protein = dataset.proteins.get("A0A098C095")
    # ['GO:0005575', 'GO:0110165', 'GO:0042995']
    go_term_ancestors = dataset.ontology.get_primary_term(
        "GO:0009289"
    ).ancestors
    # {'GO:0009289', 'GO:0005575', 'GO:0110165', 'GO:0042995'}
    go_ids = set(protein.get_all_go_ids())

    assert go_ids.issuperset(set(go_term_ancestors))
    assert set(go_term_ancestors).issubset(go_ids)
    assert len(go_ids) > len(set(go_term_ancestors))
