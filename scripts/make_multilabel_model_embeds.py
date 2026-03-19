"""make_multilabel_model_embeds.py

Train a multi-label classifier using pre-computed ESM embeddings.

Unlike the multi-class script (which assigns ONE GO term per protein),
this script creates multi-hot labels so each protein can be annotated
with ALL of its GO terms simultaneously.

Input: A directory of per-GO-term embedding pickle files, where each
file is {protein_id: np.ndarray}. The script merges these into a
single protein-centric dataset with multi-hot labels.

Example:

python3 scripts/make_multilabel_model_embeds.py \
    -d mf_go_terms-level-4 \
    -p ProtcastDataset.bin \
    --use_mlflow \
    -v
"""

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

from protcast.model.multilabel_classifier import MultiLabelClassifier  # noqa: E402
from protcast.preprocessing.protcast_dataset import (  # noqa: E402
    ProtCastDataset,
)
from protcast.config.model_config import ConfigManager  # noqa: E402

config = ConfigManager.load_config()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use_tensorboard", action="store_true", help="Use TensorBoard"
)
parser.add_argument("--use_mlflow", action="store_true", help="Use MLFlow")
parser.add_argument(
    "-d",
    "--input_dir",
    required=True,
    help="Path to directory containing per-GO-term embedding pickle files",
)
parser.add_argument(
    "-p",
    "--protcast_dataset",
    help="Path to ProtCast dataset (optional, for metadata)",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()

start = time.time()


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

# Step 1: Load all per-GO-term embedding files and build:
#   - protein_embeddings: {protein_id: np.ndarray}  (one embedding per protein)
#   - protein_go_terms:   {protein_id: set[str]}     (all GO terms per protein)
#   - go_ids:             list[str]                   (all GO terms seen)

protein_embeddings = {}
protein_go_terms = defaultdict(set)
go_ids = []

if args.verbose:
    print(f"Loading embeddings from {args.input_dir}")

for filename in sorted(os.listdir(args.input_dir)):
    if not filename.endswith(".pkl"):
        continue
    match = re.match(r"(GO_\d+)", filename)
    if not match:
        continue

    go_id = match.group(1)
    filepath = os.path.join(args.input_dir, filename)

    with open(filepath, "rb") as f:
        embeddings_dict = pickle.load(f)

    go_ids.append(go_id)

    for pid, embedding in embeddings_dict.items():
        # Store the embedding (same protein may appear in multiple GO files;
        # the embedding should be identical since it's from the same model)
        if pid not in protein_embeddings:
            protein_embeddings[pid] = embedding
        # Record this GO term for the protein
        protein_go_terms[pid].add(go_id)

    if args.verbose:
        print(f"  {go_id}: {len(embeddings_dict)} proteins")

if args.verbose:
    print(f"\nTotal unique proteins: {len(protein_embeddings)}")
    print(f"Total GO terms: {len(go_ids)}")
    all_counts = [len(gos) for gos in protein_go_terms.values()]
    print(f"Avg GO terms per protein: {sum(all_counts)/len(all_counts):.1f}")
    print(f"Max GO terms per protein: {max(all_counts)}")
    print(f"Proteins with >1 GO term: {sum(1 for c in all_counts if c > 1)}")
    print(f"Memory after loading: {get_memory_usage()}")

name = basename(args.input_dir)

classifier = MultiLabelClassifier(
    verbose=args.verbose,
    protein_embeddings=protein_embeddings,
    protein_go_terms=dict(protein_go_terms),
    go_ids=go_ids,
    config=config,
    id=name,
    use_mlflow=args.use_mlflow,
    use_tensorboard=args.use_tensorboard,
)
classifier.run()

gc.collect()

end = time.time()
print(f"Elapsed time: {round(end - start)}s")
if args.verbose:
    print(f"Final memory usage: {get_memory_usage()}")
print("Done!")
