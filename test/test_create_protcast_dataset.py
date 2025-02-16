import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from protcast.preprocessing.protcast_dataset import (  # noqa: E402
    ProtCastDataset,
)


"""test_create_protcast_dataset.py
Create a ProtCastDataset and test parsing
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "-o",
    "--ontology",
    type=Path,
    help="Path to Gene Ontology file (*.obo)",
    default="test/data/go-2023-11-15.obo",
)
parser.add_argument(
    "-s",
    "--swissprot",
    type=Path,
    help="Path to the SwissProt file (*.dat)",
    default="test/data/uniprot_mini.dat",
)
parser.add_argument(
    "-t",
    "--trembl",
    type=Path,
    help="Path to the TrEMBL file (*.fa)",
    default="test/data/uniprot_trembl_mini.fasta",
)
parser.add_argument(
    "-g",
    "--gaf",
    type=Path,
    help="Path to GOA format file (*.gaf)",
    default="test/data/goa_uniprot_mini.gaf",
)
parser.add_argument(
    "-O",
    "--output_dir",
    type=Path,
    help="Output directory",
    default="test/data/",
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

dataset = ProtCastDataset(
    args.ontology,
    args.swissprot,
    args.trembl,
    args.gaf,
    args.output_dir,
    args.verbose,
)

# Test Swissprot and *gaf parsing
# There are near-duplicate lines for A0A016QRH0 in the *gaf file
# *gaf file has 995 entries, SwissProt file has 438 annotations
sw_protein = dataset.proteins.get("A0A016QRH0")
assert len(sw_protein.annotations) == 3
annots = sw_protein.get_all_annotations()
assert len(annots) == 3
assert annots[0].evidence_code == "IEA"
assert annots[0].is_manual is False
assert annots[2].go_id == "GO:0015379"
assert len(sw_protein.get_manual_annotations()) == 0
assert len(sw_protein.get_electronic_annotations()) == 3
assert len(sw_protein.accessions) == 2
assert sw_protein.accessions[0] == "A0A016QRH0"
assert sw_protein.accessions[1] == "A0A017QRH0"
assert sw_protein.sequence.startswith(
    "MTRPPTPPASGRQGPDAPVPRVRKPLFSRVSPPQLIALSFALAILVGGVLLSLPITHGAG"
)

# Test TrEMBL parsing, *fasta file has 15 sequences
trembl_protein = dataset.proteins.get("M5BGM1")
assert not trembl_protein.get_all_annotations()
assert (
    trembl_protein.sequence
    == "GTGTEELKSLFNXTATLWCVHQRIDIKDTKEALDKVEEXQNKSKQKTQQAAAAAGSSSQNYPIVQNAQGQMTHQSMSPRTLNAWVKVIEEKASAQK"
)

# Test association of AnnotatedGOTerms and Annotations
annots = dataset.get_term("GO:0015379").annotations
assert len(annots) == 1
assert annots[0].protein_id == "A0A016QRH0"

annots = dataset.get_term("GO:0070469").annotations
assert len(annots) == 3
assert annots[0].protein_id == "A0A2U4Z3V2"
assert annots[1].protein_id == "A0A7H0LCT9"
assert annots[2].protein_id == "N0GT22"

# Check that we have Proteins for all the Annotations
for annot in dataset.get_all_annotations():
    assert dataset.accessions.get(annot.protein_id)
    assert dataset.proteins.get(annot.protein_id)

# Totals
assert len(dataset.get_all_annotations()) == 773

# level
assert dataset.get_term("GO:0008150").level == 0  # Biological Process
assert dataset.get_term("GO:0005575").level == 0  # Cellular Component
assert dataset.get_term("GO:0003674").level == 0  # Molecular Function
# Assert levels of get_all_ancestors() of GO:0031957
assert dataset.get_term("GO:0003824").level == 1  # catalytic activity
assert dataset.get_term("GO:0016874").level == 2  # ligase activity
assert (
    dataset.get_term("GO:0016877").level == 3
)  # ligase activity, forming carbon-sulfur bonds
# The 2 parents of GO:0015645
assert dataset.get_term("GO:0140657").level == 1  # ATP-dependent activity
assert dataset.get_term("GO:0016878").level == 4  # acid-thiol ligase activity
# Has 2 parents thus could be 2 or 5
assert dataset.get_term("GO:0015645").level == 2  # fatty acid ligase activity
# Has 2 parents, one of which also has 2 parents
assert (
    dataset.get_term("GO:0031957").level == 3
)  # very long-chain fatty acid-CoA ligase activity

# children
assert not dataset.get_term("GO:0031957").children
assert len(dataset.get_term("GO:0031955").children) == 0
assert len(dataset.get_term("GO:0015645").children) == 9

# Subgraphs
assert len(dataset.get_subgraph("GO:0031957")) == 1
assert len(dataset.get_subgraph("GO:0031955")) == 1
assert len(dataset.get_subgraph("GO:0022857")) == 5628
# Subgraph numbers validated by goatools
assert len(dataset.get_subgraph("GO:0015645")) == 16
assert len(dataset.get_subgraph("GO:0031956")) == 3
assert len(dataset.get_subgraph("GO:0004467")) == 5
assert len(dataset.get_subgraph("GO:0016405")) == 58

# parents
# GO:0016405 (CoA-ligase activity), GO:0015645 (fatty acid ligase activity)
assert len(dataset.get_term("GO:0031957").parents) == 2
parents = dataset.get_term("GO:0015645").parents
assert len(parents) == 2
# The order of the parents is not deterministic
assert parents[0] == "GO:0016878" or parents[0] == "GO:0140657"
assert parents[1] == "GO:0016878" or parents[1] == "GO:0140657"
assert not dataset.get_term("GO:0003674").parents
# Roots should have no parents
assert not dataset.get_term("GO:0008150").parents  # Biological Process
assert not dataset.get_term("GO:0005575").parents  # Biological Process
assert not dataset.get_term("GO:0003674").parents  # Biological Process

dataset.to_obo()
assert os.path.isfile(Path(args.output_dir, "ProtCastDataset.obo"))

dataset.save()
# Output files
assert os.path.isfile(Path(args.output_dir, "ProtCastDataset.bin"))
assert os.path.isfile(Path(args.output_dir, "ProtCastDataset.log"))

if not args.keep:
    os.unlink(Path(args.output_dir, "ProtCastDataset.bin"))
    os.unlink(Path(args.output_dir, "ProtCastDataset.log"))
    os.unlink(Path(args.output_dir, "ProtCastDataset.obo"))
