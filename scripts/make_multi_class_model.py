import re
import sys
import os
import time
import json
import argparse
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from protcast.model.multi_classifier import MultiClassifier  # noqa: E402
from protcast.preprocessing.protcast_dataset import (  # noqa: E402
    ProtCastDataset,
)
from protcast.config.model_config import ConfigManager  # noqa: E402

config = ConfigManager.load_config()


""""make_multi_class_model.py
Provide a Fasta file, or a text file with GO ids and a ProtCastDataset file. 
Example to use a ProtCastDataset:

python3 scripts/make_multi_class_model.py \
-g test/data/go-terms.txt \
-p ProtcastDataset.bin \
-a qsorder
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
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()

start = time.time()

# Primary keys are GO ids, secondary keys are protein ids, values are sequences
proteins = defaultdict(dict)

# GO id subgraph sequences are collected from a ProtCastDataset
dataset = ProtCastDataset.load_serialized_file(args.protcast_dataset)
go_ids = [
    line.strip()
    for line in open(args.go_ids_file, "r")
    if line.startswith("GO:")
]
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
            proteins[go_id].update(seqs)

classifier = MultiClassifier(
    args.algorithm,
    args.verbose,
    proteins,
    config,
    args.use_mlflow,
    args.use_tensorboard,
)
classifier.run()
# Not necessary with the checkpoints in place
# classifier.save_model()

end = time.time()
print(f"Elapsed {args.algorithm} time: {round(end - start)}s")
