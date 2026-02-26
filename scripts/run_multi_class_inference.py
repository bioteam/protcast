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
from protcast.config.model_config import ConfigManager  # noqa: E402
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


""""test_multi_class_inference.py
Provide the name of a model file and the name of sequence file.
Example:

python3 test/test_multi_class_inference.py \
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
parser.add_argument("--use_mlflow", action="store_true", help="Use MLFlow")
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
skipped_count = 0
processed_count = 0

print("Protein\tActual GO\tPredicted GO\tProbability\tConfidence\tName")
for seq in SeqIO.parse(args.seq_file, "fasta"):
    seqstr = str(seq.seq).upper()
    fv_calculator.get_feature_vectors(algorithm, pdict={seq.id: seqstr})
    # No result: sequence is too short or has non-standard aa's
    if len(fv_calculator.encodings) == 0:
        skipped_count += 1
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

    # Get confidence level
    confidence = get_confidence_level(pred_tup[1])

    print(
        f"{seq.id}\t{actual_go_id}\t{pred_tup[0]}\t{pred_tup[1]:.4f}\t{confidence}\t{actual_go_name}"
    )
    true.append(go_decoder.encode(actual_go_id))
    pred.append(go_decoder.encode(pred_tup[0]))
    processed_count += 1

y_true = np.array(true)
y_pred = np.array(pred)
# Get overall F1 score ("weighted average", there is also "micro" and "macro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")
f1_macro = f1_score(y_true, y_pred, average="macro")
print(f"Weighted F1 score\t{f1_weighted:.4f}")
# Get F1 score for each class
print("F1 scores per class")
f1_per_class = f1_score(y_true, y_pred, average=None)
for i, f1 in enumerate(f1_per_class):
    go_id = go_decoder.decode(i)
    print(f"{go_id}\t{names[go_id]}\t{f1:.4f}")

end = time.time()
inference_time = round(end - start)
print(f"Elapsed {algorithm} inference time: {inference_time}")

# --- MLflow logging for inference ---
if args.use_mlflow:
    config = ConfigManager.load_config()
    from protcast.utils.mlflow_utils import init_mlflow, log_inference_results

    mlflow = init_mlflow(
        experiment_name=config.get("EXPERIMENT_NAME", "Default Experiment"),
        verbose=args.verbose,
    )
    if mlflow is not None:
        log_inference_results(
            mlflow,
            params={
                "model_file": args.model_file,
                "goencoder_file": args.goencoder_file,
                "seq_file": args.seq_file,
                "algorithm": algorithm,
                "input_source": "feature_vectors",
                "num_classes": go_decoder.num_classes,
            },
            metrics={
                "f1_weighted": round(f1_weighted, 4),
                "f1_macro": round(f1_macro, 4),
                "processed_sequences": processed_count,
                "skipped_sequences": skipped_count,
                "inference_time_seconds": inference_time,
            },
            tags={
                "Inference Info": "MultiClassifier feature-vector inference",
                "input_source": "feature_vectors",
            },
        )
        print("MLflow inference logging complete.")
