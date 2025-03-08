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
from protein_feature_vectors import Calculator

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
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()

start = time.time()

model = MultiClassifier.load_model(Path(args.model_file))
algorithm = re.search(
    r"\d+-\d+-\d+-\d+-\d+-\d+_(.+)\.keras$", args.model_file
)[1]
go_decoder = GOEncoder.load(args.goencoder_file)
fv_calculator = Calculator(verbose=True)

# Collect data for F1 score calculation
true = list()
pred = list()
names = dict()

print("Protein\tActual GO\tPredicted GO\tProbability\tName")
for seq in SeqIO.parse(args.seq_file, "fasta"):
    seqstr = str(seq.seq).upper()
    fv_calculator.get_feature_vectors(algorithm, pdict={seq.id: seqstr})
    # No result: sequence is too short or has non-standard aa's
    if len(fv_calculator.encodings) == 0:
        continue
    """
    The feature vector is a 1D array, but the model expects a 2D array where
    dimension 1 is the number of samples and dimension 2 is the length of
    the feature vector. -1 in reshape() tells numpy to infer dimension 2.
    """
    X_test = np.array(fv_calculator.encodings.values[0]).reshape(1, -1)
    """
    y_pred - probability for each class per sample:
    array([[2.0167906e-08, 1.0642472e-07, 2.1713373e-07, 5.4475471e-08,
        5.5837597e-07, 4.9242377e-08, 1.5946955e-07, 1.2771345e-05,
        7.0346204e-08, 9.9994290e-01, 6.5556578e-09, 2.2713012e-07,
        1.6319204e-06, 8.0122229e-09, 2.5287613e-06, 8.6221941e-10,
        1.3849089e-07, 1.1159210e-08, 3.9416241e-07, 1.4942275e-08,
        2.5876445e-05, 5.4921088e-07, 7.5157303e-08, 7.4607419e-06,
        4.1306234e-06]], dtype=float32)
    """
    y_pred = model.predict(X_test)
    pred_tup = go_decoder.decode_probabilities(y_pred, top_k=1)[0][0]
    # Get actual GO and name from the sequence description
    actual_go_id = re.search(r"(GO:\d+)", seq.description)[1]
    actual_go_name = re.search(r"GO:\d+ (.+) MF", seq.description)[1].strip()
    names[actual_go_id] = actual_go_name
    print(
        f"{seq.id}\t{actual_go_id}\t{pred_tup[0]}\t{pred_tup[1]}\t{actual_go_name}"
    )
    true.append(go_decoder.encode(actual_go_id))
    pred.append(go_decoder.encode(pred_tup[0]))

y_true = np.array(true)
y_pred = np.array(pred)
# Get overall F1 score ("weighted average", there is also "micro" and "macro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")
print(f"Weighted F1 score\t{f1_weighted:.4f}")
# Get F1 score for each class
print("F1 scores per class")
f1_per_class = f1_score(y_true, y_pred, average=None)
for i, f1 in enumerate(f1_per_class):
    go_id = go_decoder.decode(i)
    print(f"{go_id}\t{names[go_id]}\t{f1:.4f}")

end = time.time()
print(f"Elapsed {algorithm} inference time: {round(end - start)}")
