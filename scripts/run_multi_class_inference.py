import re
import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from Bio import SeqIO

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from protcast.model.multi_classifier import MultiClassifier  # noqa: E402
from protcast.model.multi_classifier import GOEncoder  # noqa: E402
from protcast.model.feature_vector import FeatureVector

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
    choices=["ifeatpro", "iFeatureOmega"],
    help="Feature creator package",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()

start = time.time()

model = MultiClassifier.load_model(Path(args.model_file))
algorithm = re.search(
    r"\d+-\d+-\d+-\d+-\d+-\d+_(.+)\.keras$", args.model_file
)[1]
go_encoder = GOEncoder.load(args.goencoder_file)
fv_factory = FeatureVector(algorithm, args.feature_creator)

# Collect data for F1 score calculation
true = list()
pred = list()
names = dict()

print("Protein\tActual GO\tPredicted GO\tProbability\tName")
for seq in SeqIO.parse(args.seq_file, "fasta"):
    fv = fv_factory.get_feature_vectors({seq.id: str(seq.seq)})
    # A feature vector may not be created if the sequence is too short
    if len(fv[0]) == 0:
        continue
    # The feature vector is a 1D array, but the model expects a 2D array where the first dimension
    # is the batch size (None means it can be any number), and the second dimension is the length
    # of the feature vector. -1 in the reshape() tells numpy to infer the second dimension.
    X_test = np.array(fv[0][0]).reshape(1, -1)
    y_pred_ps = model.predict(X_test)
    pred_tup = go_encoder.decode_probabilities(y_pred_ps, top_k=1)[0][0]
    # Get actual GO and name from the sequence description
    actual_go_id = re.search(r"(GO:\d+)", seq.description)[1]
    actual_go_name = re.search(r"GO:\d+ (.+) MF", seq.description)[1].strip()
    names[actual_go_id] = actual_go_name
    print(
        f"{seq.id}\t{actual_go_id}\t{pred_tup[0]}\t{pred_tup[1]}\t{actual_go_name}"
    )
    true.append(go_encoder.encode(actual_go_id))
    pred.append(go_encoder.encode(pred_tup[0]))

y_true = np.array(true)
y_pred = np.array(pred)
# Calculate F1 score (weighted average, there is also "micro" and "macro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")
# Calculate F1 score for each class
print(f"Weighted F1 score\t{f1_weighted:.4f}")
print("F1 scores per class")
f1_per_class = f1_score(y_true, y_pred, average=None)
for i, f1 in enumerate(f1_per_class):
    go_id = go_encoder.decode(i)
    print(f"{go_id}\t{names[go_id]}\t{f1:.4f}")

end = time.time()
print(f"Elapsed inference time: {round(end - start)}")
