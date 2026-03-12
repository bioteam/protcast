import sys
import time
import argparse
import json
from collections import defaultdict
from pathlib import Path
from Bio import SeqIO

# Add project root to sys.path for protcast imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

pytestmark = pytest.mark.integration

from protcast.model.multi_classifier import MultiClassifier
from protcast.config.model_config import ConfigManager


def collect_sequences(seq_file, verbose=False):
    proteins = []
    for seq in SeqIO.parse(seq_file, "fasta"):
        proteins.append((seq.id, str(seq.seq)))
    if verbose:
        print(f"Collected {len(proteins)} sequences from {seq_file}")
    return proteins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MultiClassifier inference")
    parser.add_argument("-m", "--model_path", required=True, help="Path to saved keras model (e.g. model.keras)")
    parser.add_argument("-g", "--encoder_path", required=True, help="Path to encoder/pickle file (e.g. encoder.pkl)")
    parser.add_argument("-s", "--seq_file", required=True, help="Path to FASTA file for inference")
    parser.add_argument("--config_path", type=str, help="Path to custom config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    start = time.time()

    if args.config_path:
        config = ConfigManager.load_config(args.config_path)
    else:
        config = ConfigManager.load_config()

    proteins = collect_sequences(args.seq_file, verbose=args.verbose)
    if not proteins:
        print("No sequences found for inference")
        sys.exit(1)

    # The MultiClassifier expects proteins grouped by GO id for training,
    # but for inference we provide a flat list via the `run_inference` API.
    # Here we construct a minimal stub and rely on the classifier to support
    # inference with loaded model and encoder.

    test_id = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())

    clf = MultiClassifier(
        "INFERENCE",  # algorithm name is irrelevant for loading a model
        args.verbose,
        {},  # no training proteins required for inference
        config,
        test_id,
        use_mlflow=False,
        use_tensorboard=False,
    )

    # Load model and encoder then run inference
    try:
        if args.verbose:
            print(f"Loading model from {args.model_path} and encoder from {args.encoder_path}")
        clf.load_model(args.model_path, args.encoder_path)
    except Exception as e:
        print(f"Failed to load model or encoder: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    seq_ids = [sid for sid, _ in proteins]
    sequences = [seq for _, seq in proteins]

    try:
        predictions = clf.run_inference(sequences, seq_ids=seq_ids)
    except AttributeError:
        print("The MultiClassifier implementation does not provide a `run_inference` method.")
        print("Please adapt this test to your project's inference API.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during inference: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Print a concise summary of results
    print(f"Inference completed for {len(predictions)} sequences")
    # predictions assumed to be a dict-like mapping seq_id -> top predictions
    shown = 0
    for sid, preds in predictions.items():
        if shown >= 10:
            break
        print(f"{sid}: {preds}")
        shown += 1

    elapsed = round(time.time() - start, 2)
    if args.verbose:
        print(f"Total inference time: {elapsed}s")
