import argparse
import os
import sys
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from protcast.preprocessing.simple_dataset import SimpleDataset


def main():
    """ "test_create_simple_dataset.py
    Creates a dataset that can be used by Keras FeatureSpace, which expects
    protein features and GO terms to be passed in as a dataframe.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--ontology",
        help="Path to Gene Ontology file (*.obo)",
        default="data/go.obo",
    )
    parser.add_argument(
        "-s",
        "--swissprot",
        help="Path to the SwissProt file (*.dat)",
        default="data/uniprot_mini.dat",
    )
    parser.add_argument(
        "-t",
        "--trembl",
        help="Path to the TrEMBL file (*.fa)",
        default="data/uniprot_trembl_mini.fasta",
    )
    parser.add_argument(
        "-g",
        "--gaf",
        help="Path to GOA format file (*.gaf)",
        default="data/goa_uniprot_mini.gaf",
    )
    parser.add_argument(
        "--output_dir", help="Output directory", default="data/"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Create DEBUG log",
    )
    parser.add_argument(
        "-r",
        "--remove",
        default=False,
        action="store_true",
        help="Remove files",
    )
    parser.add_argument(
        "-n",
        "--no_propogate",
        default=False,
        action="store_true",
        help="Propogate annotations",
    )
    args = parser.parse_args()

    # This must be run with "-n", without propogation
    dataset = SimpleDataset(
        Path(args.ontology),
        Path(args.swissprot),
        Path(args.trembl),
        Path(args.gaf),
        Path(args.output_dir),
        args.no_propogate,
        args.verbose,
    )

    dataset.save()

    assert os.path.isfile(Path(args.output_dir, "SimpleDataset.bin"))
    assert os.path.isfile(Path(args.output_dir, "SimpleDataset.log"))

    # Test Swissprot and *gaf parsing
    # There are near-duplicate lines for A0A016QRH0 in the *gaf file
    sw_protein = dataset.proteins.get("A0A016QRH0")
    assert len(sw_protein.annotations) == 10
    annots = sw_protein.get_all_annotations()
    assert len(annots) == 10
    assert annots[0].evidence_code == "IEA"
    assert annots[0].is_manual == False
    assert annots[2].go_term_id == "GO:0015379"
    manuals = sw_protein.get_manual_annotations()
    assert manuals == []
    assert len(sw_protein.get_electronic_annotations()) == 10

    # Test TrEMBL parsing
    trembl_protein = dataset.proteins.get("M5BGM1")
    annots = trembl_protein.get_all_annotations()
    assert len(annots) == 1
    assert annots[0].evidence_code == "IEA"
    trembl_protein.sequence == "GTGTEELKSLFNXTATLWCVHQRIDIKDTKEALDKVEEXQNKSKQKTQQAAAAAGSSSQNYPIVQNAQGQMTHQSMSPRTLNAWVKVIEEKASAQK"

    dataset.to_obo(args.output_dir)
    assert os.path.isfile(Path(args.output_dir, "terms.obo"))

    if args.remove:
        os.unlink(Path(args.output_dir, "SimpleDataset.bin"))
        os.unlink(Path(args.output_dir, "SimpleDataset.log"))
        os.unlink(Path(args.output_dir, "terms.obo"))


if __name__ == "__main__":
    main()
