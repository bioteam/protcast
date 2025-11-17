import re
import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path
from Bio import SeqIO

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from protcast.model.multi_classifier import MultiClassifier  # noqa: E402
from protcast.model.multi_classifier import GOEncoder  # noqa: E402
from protein_feature_vectors import Calculator


def get_confidence_level(probability):
    """
    Categorize prediction confidence based on probability.

    Parameters
    ----------
    probability : float
        Prediction probability (0.0 to 1.0)

    Returns
    -------
    str : Confidence level category
    """
    if probability >= 0.9:
        return "VERY_HIGH"
    elif probability >= 0.7:
        return "HIGH"
    elif probability >= 0.5:
        return "MEDIUM"
    elif probability >= 0.3:
        return "LOW"
    else:
        return "VERY_LOW"


""""run_multi_class_inference.py
Provide the name of a model file and the name of sequence file. 
Example:

python3 scripts/run_multi_class_inference-nof1.py \
-s unknown-seqs.fa \
-m 02-01-2024_ctriad.keras \
-v
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model_file", required=True, help="Path to model file"
)
parser.add_argument(
    "-s", "--seq_file", required=True, help="Path to Fasta file"
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
args = parser.parse_args()

start = time.time()

model = MultiClassifier.load_model(Path(args.model_file))
algorithm = re.search(r"([^_]+)\.keras$", args.model_file)[1]  # type: ignore
go_decoder = GOEncoder.load(args.model_file.replace(".keras", ".pickle"))
fv_calculator = Calculator(verbose=True)

print("Protein\tPredicted GO\tProbability\tConfidence")
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
    # Get confidence level
    confidence = get_confidence_level(pred_tup[1])

    print(f"{seq.id}\t{pred_tup[0]}\t{pred_tup[1]:.4f}\t{confidence}")

end = time.time()
print(f"Elapsed {algorithm} inference time: {round(end - start)}")
