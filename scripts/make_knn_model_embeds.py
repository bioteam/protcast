"""make_knn_model_embeds.py

Train a KNN classifier using pre-computed ESM embeddings.

Mirrors make_multilabel_model_embeds.py exactly — same input format,
same embedding loading, same data structures — but uses KNNClassifier
instead of MultiLabelClassifier.

Input: A directory of per-GO-term embedding pickle files, where each
file is {protein_id: np.ndarray}. The script merges these into a
single protein-centric dataset with multi-hot labels.

Example:

python3 scripts/make_knn_model_embeds.py \
    -d mf_go_terms-level-4 \
    -p ProtcastDataset.bin \
    --obo data/go-basic.obo \
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

from protcast.model.knn_classifier import KNNClassifier  # noqa: E402
from protcast.preprocessing.protcast_dataset import (  # noqa: E402
    ProtCastDataset,
)
from protcast.config.model_config import ConfigManager  # noqa: E402

config = ConfigManager.load_config()

parser = argparse.ArgumentParser()
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
parser.add_argument(
    "--obo",
    type=str,
    default=None,
    help="Path to GO OBO file for depth-level metric breakdowns",
)
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

# Step 1: Load all per-GO-term embedding files
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
        if pid not in protein_embeddings:
            protein_embeddings[pid] = embedding
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

# Load GO DAG for depth-level metrics (optional)
go_dag = None
if args.obo:
    from protcast.preprocessing.annotated_godag import AnnotatedGODag

    if args.verbose:
        print(f"Loading GO DAG from {args.obo}")
    go_dag = AnnotatedGODag(args.obo)
    if args.verbose:
        print(f"GO DAG loaded: {len(go_dag.go_terms_map)} terms")

classifier = KNNClassifier(
    verbose=args.verbose,
    protein_embeddings=protein_embeddings,
    protein_go_terms=dict(protein_go_terms),
    go_ids=go_ids,
    config=config,
    id=name,
    use_mlflow=args.use_mlflow,
    go_dag=go_dag,
)
classifier.run()

gc.collect()

end = time.time()
print(f"Elapsed time: {round(end - start)}s")
if args.verbose:
    print(f"Final memory usage: {get_memory_usage()}")
print("Done!")
