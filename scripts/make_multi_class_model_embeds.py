import sys
import os
from os.path import basename
import time
import argparse
import pickle
import re
import gc
from collections import defaultdict


sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from protcast.model.multi_classifier import MultiClassifier  # noqa: E402
from protcast.preprocessing.protcast_dataset import (  # noqa: E402
    ProtCastDataset,
)
from protcast.config.model_config import ConfigManager  # noqa: E402

config = ConfigManager.load_config()


""""make_multi_class_model_embeds.py
Provide an input directory containing embeddings files and a ProtCastDataset file. 

python3 scripts/make_multi_class_model_embeds.py \
-d mf_go_terms-level-4 \
-p ProtcastDataset.bin \
--input_source esm_embeddings
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "--use_tensorboard", action="store_true", help="Use TensorBoard"
)
parser.add_argument("--use_mlflow", action="store_true", help="Use MLFlow")
parser.add_argument(
    "--input_source",
    default="feature_vectors",
    help="feature_vectors or esm_embeddings",
)
parser.add_argument(
    "--algorithm", default="esm", help="feature_vectors or esm_embeddings"
)
parser.add_argument(
    "-d",
    "--input_dir",
    help="Path to embeddings files",
)
parser.add_argument(
    "-p",
    "--protcast_dataset",
    help="Path to ProtCast dataset",
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


# GO id subgraph sequences are collected from a ProtCastDataset
if args.verbose:
    print(f"Loading ProtCastDataset from {args.protcast_dataset}")

dataset = ProtCastDataset.load_serialized_file(args.protcast_dataset)

"""
>>> with open("GO_0140078.pkl", "rb") as f:
    p = pickle.load(open(f)
>>> p['P23396']
array([-0.01154739,  0.00231701,  0.00114867, ...,  0.0028205 ,
        0.01398835,  0.00365072], dtype=float32)
"""
# Primary keys are GO ids from a given level, secondary keys are subgraph protein ids, values are embeddings
proteins = defaultdict(dict)
for filename in os.listdir(args.input_dir):
    if not filename.endswith(".pkl"):
        continue
    match = re.match(r"(GO_\d+)", filename)
    if not match:
        continue
    go_id = match.group(1)
    filepath = os.path.join(args.input_dir, filename)
    with open(filepath, "rb") as f:
        p = pickle.load(f)
    proteins[go_id] = p

name = basename(args.input_dir)

classifier = MultiClassifier(
    args.algorithm,
    args.verbose,
    proteins,
    config,
    name,
    args.use_mlflow,
    args.use_tensorboard,
    args.input_source,
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
