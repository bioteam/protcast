import argparse
import random

from protcast.preprocessing.protcast_dataset import (  # noqa: E402
    ProtCastDataset,
)
from protcast.config.model_config import ConfigManager  # noqa: E402


""""create_subgraph_sequences_files.py

This script takes:

- Path to serialized ProtCastDataset file
- Path to a text file with a list of GO IDs

It returns for each GO id:

- A FASTA file containing all the sequences from the DAG in the GO term's subgraph
- A FASTA file containing an equal number of sequences not in the GO term's subgraph

Example:

python scripts/create_subgraph_sequences_files.py \
-p data/dataset/u-2021-04-g-2021-10-26/dataset.bin \
-g GO_ids.txt 
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--minimum_seqs",
    default=500,
    help="Minumum number of sequences",
    type=int,
)
parser.add_argument(
    "-p",
    "--protcast_dataset",
    required=True,
    help="Path to serialized dataset",
)
parser.add_argument(
    "-g", "--go_ids", required=True, help="Path to file with a list of GO ids"
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help=(
        "Random seed for sampling. If unset, falls back to RANDOM_SEED in "
        "config.json, then to 42."
    ),
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()

# Resolve seed: CLI flag > config["RANDOM_SEED"] > 42. Wrapped in try so
# absence of config.json is non-fatal — sampling still works with default 42.
if args.seed is None:
    try:
        _config = ConfigManager.load_config()
        args.seed = int(_config.get("RANDOM_SEED", 42))
    except FileNotFoundError:
        args.seed = 42
if args.verbose:
    print(f"Random seed: {args.seed}")

# One Random instance reused across all GO terms in the loop below, so the
# sampling sequence is deterministic given a fixed seed and a fixed
# iteration order over `go_ids`.
rng = random.Random(args.seed)

dataset = ProtCastDataset.load_serialized_file(args.protcast_dataset)

with open(args.go_ids, "r") as f:
    go_ids = [line.strip() for line in f]


for go_id in go_ids:
    go_terms = dataset.get_terms(dataset.get_subgraph(go_id))
    all_annots = list()
    for go_term in go_terms:
        annots = go_term.get_all_annotations()
        if annots:
            all_annots.extend(annots)
    target_seq_ids = set([annot.protein_id for annot in all_annots])

    if len(target_seq_ids) < args.minimum_seqs:
        if args.verbose:
            print(
                f"Not enough subgraph sequences for {go_id}: {len(target_seq_ids)}"
            )
        continue

    go_terms = dataset.get_terms(dataset.get_inverse_subgraph(go_id))
    all_annots = list()
    for go_term in go_terms:
        annots = go_term.get_all_annotations()
        if annots:
            all_annots.extend(annots)
    non_target_seq_ids = set([annot.protein_id for annot in all_annots])

    if len(non_target_seq_ids) < args.minimum_seqs:
        if args.verbose:
            print(
                f"Not enough inverse-subgraph sequences for {go_id}: {len(non_target_seq_ids)}"
            )
        continue

    # Sort the set-derived id lists before sampling. Set iteration order is
    # non-deterministic across processes (PYTHONHASHSEED is randomised), so
    # `list(target_seq_ids)` alone would yield a different order — and thus
    # a different sample — on each run, even with a fixed seed.
    min_target_seq_ids = rng.sample(
        sorted(target_seq_ids), args.minimum_seqs
    )
    min_non_target_seq_ids = rng.sample(
        sorted(non_target_seq_ids), args.minimum_seqs
    )

    go_term = dataset.get_term(go_id)

    with open(f"{go_id}_subgraph.fa", "w") as target_seq_file:
        for id in min_target_seq_ids:
            target_seq_file.write(
                f">{id} {go_id} {go_term.name}\n{dataset.proteins[id].sequence}\n"
            )

    with open(f"{go_id}_inv_subgraph.fa", "w") as non_target_seq_file:
        for id in min_non_target_seq_ids:
            non_target_seq_file.write(
                f">{id} Not in {go_id} {go_term.name} subgraph\n{dataset.proteins[id].sequence}\n"
            )
