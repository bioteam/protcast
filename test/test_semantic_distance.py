"""
test_semantic_distance.py

Test the SemanticDistance class using the GO OBO file in test/data/.

Usage:
    python3 test/test_semantic_distance.py -v
    python3 test/test_semantic_distance.py -s test/data/random-level-4.fa -v
"""

import sys
import argparse
import re
from pathlib import Path
from collections import defaultdict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.integration

from protcast.utils.mlflow_utils import SemanticDistance  # noqa: E402

OBO_FILE = str(Path(__file__).resolve().parent / "data" / "go-2023-11-15.obo")


@pytest.fixture(scope="module")
def sd():
    """Create a SemanticDistance instance from the test OBO file."""
    return SemanticDistance(OBO_FILE)


def test_same_term(sd, verbose=False):
    """Distance from a term to itself should be 0."""
    d = sd.shortest_path("GO:0010181", "GO:0010181")
    assert d == 0, f"Expected 0, got {d}"
    if verbose:
        print(f"  PASS  same term -> 0")


def test_to_mf_root(sd, verbose=False):
    """GO:0010181 is level 4 MF, so distance to MF root GO:0003674 should be 4."""
    d = sd.shortest_path("GO:0010181", "GO:0003674")
    assert d == 4, f"Expected 4, got {d}"
    if verbose:
        print(f"  PASS  GO:0010181 -> MF root (GO:0003674) = {d}")


def test_sibling_terms(sd, verbose=False):
    """Two terms sharing a direct parent should have distance 2."""
    # GO:0010181 (FMN binding) and GO:0032565 (dGMP binding) are both
    # children of GO:0043168 (anion binding)
    d = sd.shortest_path("GO:0010181", "GO:0032565")
    if verbose:
        print(f"  INFO  GO:0010181 -> GO:0032565 = {d}")
    assert d == 2, f"Expected 2 (siblings via GO:0043168), got {d}"
    if verbose:
        print(f"  PASS  sibling terms -> 2")


def test_cross_branch(sd, verbose=False):
    """Terms in different MF branches should have a larger distance."""
    # FMN binding vs sulfurtransferase are in different branches
    d = sd.shortest_path("GO:0010181", "GO:0016783")
    assert d > 2, f"Expected > 2 for cross-branch terms, got {d}"
    if verbose:
        print(f"  PASS  GO:0010181 -> GO:0016783 (cross-branch) = {d}")


def test_unknown_term(sd, verbose=False):
    """Unknown terms should return -1."""
    d = sd.shortest_path("GO:9999999", "GO:0010181")
    assert d == -1, f"Expected -1 for unknown term, got {d}"
    if verbose:
        print(f"  PASS  unknown term -> -1")


def test_batch(sd, verbose=False):
    """batch_distances should return a list of correct distances."""
    true_ids = ["GO:0010181", "GO:0016783", "GO:0010181"]
    pred_ids = ["GO:0010181", "GO:0010181", "GO:0016783"]
    dists = sd.batch_distances(true_ids, pred_ids)
    assert len(dists) == 3
    assert dists[0] == 0, f"Same term distance should be 0, got {dists[0]}"
    assert dists[1] == dists[2], f"Distance should be symmetric, got {dists[1]} vs {dists[2]}"
    assert dists[1] > 0
    if verbose:
        print(f"  PASS  batch_distances = {dists}")


def run_fasta_simulation(sd, seq_file, verbose=False):
    """
    Simulate inference: for each sequence, randomly 'predict' one of the
    GO terms from the file, then compute distances.  This exercises the
    full pipeline on real GO terms from the test data.
    """
    import random

    # Parse GO IDs from FASTA headers without requiring Biopython
    go_ids_in_file = []
    with open(seq_file) as f:
        for line in f:
            if line.startswith(">"):
                match = re.search(r"GO:\d+", line)
                if match:
                    go_ids_in_file.append(match.group(0))

    unique_go_ids = list(set(go_ids_in_file))
    if verbose:
        print(f"\n  Simulating inference on {len(go_ids_in_file)} sequences "
              f"across {len(unique_go_ids)} GO terms")

    # "Predict" by randomly picking a GO term for each sequence
    random.seed(42)
    pred_ids = [random.choice(unique_go_ids) for _ in go_ids_in_file]

    dists = sd.batch_distances(go_ids_in_file, pred_ids)
    valid = [d for d in dists if d >= 0]
    correct = sum(1 for d in dists if d == 0)
    mean_dist = sum(valid) / len(valid) if valid else 0

    if verbose:
        print(f"  Results:")
        print(f"    Total predictions:  {len(dists)}")
        print(f"    Correct (dist=0):   {correct}")
        print(f"    Mean distance:      {mean_dist:.2f}")
        print(f"    Max distance:       {max(valid)}")
        print(f"    Unknown terms:      {sum(1 for d in dists if d < 0)}")

    # Build a distance distribution
    dist_counts = defaultdict(int)
    for d in valid:
        dist_counts[d] += 1
    if verbose:
        print(f"    Distance distribution:")
        for dist_val in sorted(dist_counts.keys()):
            bar = "#" * (dist_counts[dist_val] // 5)
            print(f"      {dist_val:3d}: {dist_counts[dist_val]:4d} {bar}")

    # All pairwise distances between the 25 GO terms
    if verbose:
        print(f"\n  Pairwise distance matrix (unique GO terms):")
        sorted_ids = sorted(unique_go_ids)
        # Header (abbreviated GO IDs)
        header = "          " + "  ".join(
            gid.replace("GO:", "")[:5] for gid in sorted_ids
        )
        print(header)
        for gid_a in sorted_ids:
            row = f"  {gid_a}  "
            for gid_b in sorted_ids:
                d = sd.shortest_path(gid_a, gid_b)
                row += f"{d:5d} "
            print(row)

    assert len(valid) == len(dists), "All test GO terms should be in the OBO file"
    assert mean_dist > 0, "Random predictions should have non-zero mean distance"
    print(f"  PASS  FASTA simulation ({len(dists)} predictions, mean dist={mean_dist:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test SemanticDistance with GO OBO file"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "-s", "--seq_file", default=None,
        help="Optional FASTA file to simulate inference distances",
    )
    parser.add_argument(
        "--obo_file", default=OBO_FILE,
        help=f"Path to OBO file (default: {OBO_FILE})",
    )
    args = parser.parse_args()

    print(f"Loading GO DAG from {args.obo_file}...")
    sd = SemanticDistance(args.obo_file)
    print("GO DAG loaded.\n")

    print("Running unit tests:")
    test_same_term(sd, args.verbose)
    test_to_mf_root(sd, args.verbose)
    test_sibling_terms(sd, args.verbose)
    test_cross_branch(sd, args.verbose)
    test_unknown_term(sd, args.verbose)
    test_batch(sd, args.verbose)

    if args.seq_file:
        print("\nRunning FASTA simulation test:")
        run_fasta_simulation(sd, args.seq_file, args.verbose)

    print("\nAll tests passed.")
