import os
from os.path import basename
import time
import random
import argparse
import re
import gc
from collections import defaultdict


from protcast.model.multi_classifier import MultiClassifier  # noqa: E402
from protcast.preprocessing.protcast_dataset import (  # noqa: E402
    ProtCastDataset,
)
from protcast.config.model_config import ConfigManager  # noqa: E402

config = ConfigManager.load_config()


""""make_multi_class_model.py
Provide a text file with GO ids and a ProtCastDataset file. 

python3 scripts/make_multi_class_model.py \
-g test/data/go-terms.txt \
-p ProtcastDataset.bin \
-a CTriad
"""
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--go_ids_file", help="Path to GO ids file")
parser.add_argument(
    "--use_tensorboard", action="store_true", help="Use TensorBoard"
)
parser.add_argument("--use_mlflow", action="store_true", help="Use MLFlow")
parser.add_argument(
    "-a", "--algorithm", default="CTriad", help="Feature vector algorithm"
)
parser.add_argument(
    "-p",
    "--protcast_dataset",
    help="Path to ProtCast dataset",
)
parser.add_argument(
    "--minimum_seqs",
    default=500,
    help="Minimum number of sequences",
    type=int,
)
parser.add_argument(
    "--maximum_seqs",
    default=2000,
    help="Maximum number of sequences to use for training",
    type=int,
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()

start = time.time()


# Add memory usage monitoring
def get_memory_usage():
    """Return current memory usage in GB"""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1e9
        return f"{memory_gb:.2f} GB"
    except ImportError:
        return "psutil not installed"


if args.verbose:
    print(f"Starting memory usage: {get_memory_usage()}")

# Primary keys are GO ids from a given level, secondary keys are subgraph protein ids, values are sequences
proteins = defaultdict(dict)

# GO id subgraph sequences are collected from a ProtCastDataset
if args.verbose:
    print(f"Loading ProtCastDataset from {args.protcast_dataset}")

dataset = ProtCastDataset.load_serialized_file(args.protcast_dataset)

if args.verbose:
    print(f"Memory after loading dataset: {get_memory_usage()}")
    print(f"Reading GO IDs from {args.go_ids_file}")

# Set of GO ids to process
go_ids = {
    match.group(0)
    for line in open(args.go_ids_file, "r")
    if (match := re.search(r"GO:\d+", line))
}
for go_id in go_ids:
    subgraph_go_ids = dataset.get_subgraph(go_id)
    for subid in subgraph_go_ids:
        pids = dataset.get_term(subid).get_all_pids()
        if pids:
            seqs = {
                pid: dataset.proteins[pid].sequence
                for pid in pids
                if pid in dataset.proteins
            }
            # Update the proteins dict so GO id has protein id keys with sequence values
            proteins[go_id].update(seqs)
    if len(proteins[go_id]) > args.minimum_seqs:
        # If we've collected too many sequences, sample down to the maximum
        if len(proteins[go_id]) > args.maximum_seqs:
            num_to_sample = args.maximum_seqs
            sampled_items = random.sample(
                list(proteins[go_id].items()), num_to_sample
            )
            # Replace the proteins dictionary for this GO ID with the sampled subset
            proteins[go_id] = dict(sampled_items)
        if args.verbose:
            print(
                f"GO subgraph {go_id}: {len(proteins[go_id])} sequences collected"
            )
    else:
        if args.verbose:
            print(
                f"GO subgraph {go_id} skipped: < {args.minimum_seqs} sequences"
            )
        # Remove GO IDs with insufficient samples from the proteins dictionary
        del proteins[go_id]

name = basename(args.go_ids_file).replace(".tsv", "")

classifier = MultiClassifier(
    args.algorithm,
    args.verbose,
    proteins,
    config,
    name,
    args.use_mlflow,
    args.use_tensorboard,
)
classifier.run()
# Not necessary with the checkpoints in place
# classifier.save_model()

# Force garbage collection to free memory
gc.collect()

end = time.time()
print(f"Elapsed {args.algorithm} time: {round(end - start)}s")
if args.verbose:
    print(f"Final memory usage: {get_memory_usage()}")
print("Done!")
