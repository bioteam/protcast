import re
import sys
import os
import time
import json
import argparse
from collections import defaultdict
from Bio import SeqIO
from pathlib import Path
import inspect

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from protcast.model.multi_classifier import MultiClassifier  # noqa: E402

print("MultiClassifier defined in:", inspect.getfile(MultiClassifier))

# config_path = os.path.join(os.getcwd(), "mlflow_config.json")

# path to the current script
script_path = Path(__file__).resolve()

# Get the parent directory of the script
parent_dir = script_path.parent.parent

# Full path to the config file
config_path = parent_dir / "mlflow_config.json"

# Load the configuration file
with open(config_path, "r") as f:
    config = json.load(f)


""""test_multi_classifier.py
python3 scripts/test_multi_classifier.py \
-s test/data/random-level-4.fa \
-v
"""
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seq_file", help="Path to Fasta file")
parser.add_argument(
    "--use_tensorboard", action="store_true", help="Use TensorBoard"
)
parser.add_argument("--use_mlflow", action="store_true", help="Use MLFlow")
parser.add_argument(
    "-a", "--algorithm", default="CTriad", help="Feature vector algorithm"
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()

start = time.time()

# Primary keys are GO ids, secondary keys are protein ids, values are sequences
proteins = defaultdict(dict)

# Sequences are collected from a fasta file where the description contains a GO id
for seq in SeqIO.parse(args.seq_file, "fasta"):
    go_id = re.search("GO:\d+", seq.description)[0]  # type: ignore
    if go_id is not None:
        proteins[go_id][seq.id] = str(seq.seq)
if args.verbose:
    print(f"Number of GO ids: {len(proteins.keys())}")
    print(
        f"Number of proteins: {len([v for inner_dict in proteins.values() for v in inner_dict.values()])}"
    )


classifier = MultiClassifier(
    algorithm=args.algorithm,
    verbose=args.verbose,
    proteins=proteins,
    optimizer=config["OPTIMIZER"],
    loss=config["LOSS"],
    metrics=config["METRICS"],
    epochs=config["EPOCHS"],
    batch_size=config["BATCH_SIZE"],
    neurons=config["NEURONS"],
    dropout=config["DROPOUT"],
    pred_threshold=config["PRED_THRESHOLD"],
    validation_split=config["VALIDATION_SPLIT"],
    patience=config["PATIENCE"],
    use_mlflow=args.use_mlflow,
    use_tensorboard=args.use_tensorboard,
)
classifier.run()
# Not necessary with the checkpoints in place
# classifier.save_model()

end = time.time()

if args.verbose:
    print(f"Elapsed {args.algorithm} time: {round(end - start)}s")
