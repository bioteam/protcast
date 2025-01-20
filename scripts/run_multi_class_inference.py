import re
import sys
import os
import argparse
import numpy as np
import pickle
from pathlib import Path
from Bio import SeqIO
from protcast.model.feature_vector import (
    get_ifeatpro_features,
    get_ifeatureomega_features,
)

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from protcast.model.multi_classifier import MultiClassifier  # noqa: E402
from protcast.model.multi_classifier import GOEncoder  # noqa: E402

""""run_multi_class_inference.py
Provide the name of a model file and the name of sequence file. 
Example:

python3 scripts/run_multi_class_inference.py \
-s unknown-seqs.fa \
-m 02-01-2024_ctriad.h5 \ 
-g 02-01-2024_goencoder.pickle \
-v
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model_file", required=True, help="Path to model file"
)
parser.add_argument(
    "-g", "--goencoder_file", required=True, help="Path to GOEncoder file"
)
parser.add_argument(
    "-s", "--seq_file", required=True, help="Path to Fasta file"
)
parser.add_argument(
    "-f",
    "--feature_creator",
    default="ifeatpro",
    help="Feature creator package",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()

model = MultiClassifier.load_model(Path(args.model_file))
algorithm = re.search(r"\d+-\d+-\d+-\d+-\d+-\d+_(.+)\.h5$", args.model_file)[1]
go_encoder = GOEncoder.load(args.goencoder_file)


def get_feature_vector(seq):
    if args.feature_creator == "ifeatpro":
        return get_ifeatpro_features(algorithm, seq)
    elif args.feature_creator == "ifeatureomega":
        return get_ifeatureomega_features(algorithm, seq)


print("Protein\tActual GO\tPredicted GO\tProbability\tName")
for seq in SeqIO.parse(args.seq_file, "fasta"):
    fv = get_feature_vector({seq.id: str(seq.seq)})
    # A feature vector may not be created if the sequence is too short
    if len(fv[0]) == 0:
        continue
    # The feature vector is a 1D array, but the model expects a 2D array where the first dimension
    # is the batch size (None means it can be any number), and the second dimension is the length
    # of the feature vector. -1 in the reshape() tells numpy to infer the second dimension.
    actual_go_id = re.search(r"(GO:\d+)", seq.description)[1]
    name = re.search(r"GO:\d+ (.+) MF", seq.description)[1].strip()
    X = np.array(fv[0][0]).reshape(1, -1)
    result = model.predict(X)
    tup = go_encoder.decode_probabilities(result, top_k=1)[0][0]
    print(f"{seq.id}\t{actual_go_id}\t{tup[0]}\t{tup[1]}\t{name}")
