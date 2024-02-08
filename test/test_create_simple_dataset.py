import argparse
import os
import sys
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.simple_dataset import SimpleDataset  # noqa: E402


def main():
    """test_create_simple_dataset.py
    Create a SimpleDataset and test parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--ontology",
        type=Path,
        help="Path to Gene Ontology file (*.obo)",
        default="data/go.obo",
    )
    parser.add_argument(
        "-s",
        "--swissprot",
        type=Path,
        help="Path to the SwissProt file (*.dat)",
        default="data/uniprot_mini.dat",
    )
    parser.add_argument(
        "-t",
        "--trembl",
        type=Path,
        help="Path to the TrEMBL file (*.fa)",
        default="data/uniprot_trembl_mini.fasta",
    )
    parser.add_argument(
        "-g",
        "--gaf",
        type=Path,
        help="Path to GOA format file (*.gaf)",
        default="data/goa_uniprot_mini.gaf",
    )
    parser.add_argument(
        "-O",
        "--output_dir",
        type=Path,
        help="Output directory",
        default="data/",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Create DEBUG log",
    )
    parser.add_argument(
        "-k",
        "--keep",
        default=False,
        action="store_true",
        help="Keep *bin and *log files",
    )
    args = parser.parse_args()

    dataset = SimpleDataset(
        args.ontology,
        args.swissprot,
        args.trembl,
        args.gaf,
        args.output_dir,
        args.verbose,
    )

    # Test Swissprot and *gaf parsing without propagation
    # There are near-duplicate lines for A0A016QRH0 in the *gaf file
    sw_protein = dataset.proteins.get("A0A016QRH0")
    assert len(sw_protein.annotations) == 3
    annots = sw_protein.get_all_annotations()
    assert len(annots) == 3
    assert annots[0].evidence_code == "IEA"
    assert annots[0].is_manual is False
    assert annots[2].go_id == "GO:0015379"
    manuals = sw_protein.get_manual_annotations()
    assert manuals == []
    assert len(sw_protein.get_electronic_annotations()) == 3

    # Test TrEMBL parsing
    trembl_protein = dataset.proteins.get("M5BGM1")
    annots = trembl_protein.get_all_annotations()
    assert len(annots) == 0
    trembl_protein.sequence == "GTGTEELKSLFNXTATLWCVHQRIDIKDTKEALDKVEEXQNKSKQKTQQAAAAAGSSSQNYPIVQNAQGQMTHQSMSPRTLNAWVKVIEEKASAQK"

    # Test association of AnnotatedGOTerms and Annotations
    annots = dataset.annotated_dag.get_term("GO:0015379").annotations
    assert len(annots) == 1
    assert annots[0].protein_id == "A0A016QRH0"

    annots = dataset.annotated_dag.get_term("GO:0070469").annotations
    assert len(annots) == 3
    assert annots[0].protein_id == "A0A2U4Z3V2"
    assert annots[1].protein_id == "A0A7H0LCT9"
    assert annots[2].protein_id == "N0GT22"

    dataset.to_obo()
    assert os.path.isfile(Path(args.output_dir, "SimpleDataset.obo"))

    dataset.save()
    # Output files
    assert os.path.isfile(Path(args.output_dir, "SimpleDataset.bin"))
    assert os.path.isfile(Path(args.output_dir, "SimpleDataset.log"))

    if not args.keep:
        os.unlink(Path(args.output_dir, "SimpleDataset.bin"))
        os.unlink(Path(args.output_dir, "SimpleDataset.log"))
        os.unlink(Path(args.output_dir, "SimpleDataset.obo"))


if __name__ == "__main__":
    main()
