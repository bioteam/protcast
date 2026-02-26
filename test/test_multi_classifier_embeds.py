"""
test_multi_classifier_embeds.py

Test MultiClassifier training and inference using ESM-C embeddings.
Reads a FASTA file, generates ESM-C embeddings on the fly, then
trains a multiclass model — analogous to test_multi_classifier.py
but for the ESM embedding pathway.

Example usage:

python3 test/test_multi_classifier_embeds.py \
    -s test/data/random-level-4.fa \
    --use_mlflow \
    --model_type esmc_300m \
    -v
"""

import re
import sys
import time
import json
import argparse
import numpy as np
import torch
from collections import defaultdict
from Bio import SeqIO
from pathlib import Path

# Add project root to sys.path for protcast imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

pytestmark = pytest.mark.integration

from protcast.model.multi_classifier import MultiClassifier  # noqa: E402
from protcast.config.model_config import ConfigManager  # noqa: E402


def load_esm_model(model_name, verbose=False):
    """Load an ESM-C model."""
    from esm.models.esmc import ESMC

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Loading ESM-C model: {model_name} on {device}")

    model = ESMC.from_pretrained(model_name, device=device)
    model.eval()

    if verbose:
        print(f"ESM-C model loaded successfully")
    return model


def generate_embedding(esm_model, sequence, protein_id, verbose=False):
    """Generate an ESM-C embedding for a single protein sequence."""
    from esm.sdk.api import ESMProtein

    try:
        protein = ESMProtein(sequence=sequence)
        with torch.no_grad():
            protein_tensor = esm_model.encode(protein)
            output = esm_model.forward(
                sequence_tokens=protein_tensor.sequence.unsqueeze(0)
            )
            sequence_embeddings = output.embeddings.squeeze(0).to(
                dtype=torch.float32
            )
            embedding = sequence_embeddings.mean(dim=0).cpu().numpy()
            if verbose:
                print(f"  {protein_id}: embedding shape {embedding.shape}")
            return embedding
    except Exception as e:
        if verbose:
            print(f"  Error embedding {protein_id}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test MultiClassifier with ESM-C embeddings"
    )
    parser.add_argument(
        "-s", "--seq_file", required=True, help="Path to FASTA file"
    )
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        help="Use TensorBoard logging",
    )
    parser.add_argument(
        "--use_mlflow", action="store_true", help="Use MLFlow logging"
    )
    parser.add_argument(
        "--model_type",
        default="esmc_300m",
        choices=["esm3_c", "esmc_300m", "esmc_600m"],
        help="ESM-C model type (default: esmc_300m for faster testing)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--config_override",
        type=str,
        help="JSON string to override config values",
    )
    parser.add_argument(
        "--config_path", type=str, help="Path to custom config file"
    )
    args = parser.parse_args()

    start = time.time()

    # Load ESM model
    esm_model = load_esm_model(args.model_type, args.verbose)

    # Read sequences and generate embeddings grouped by GO ID
    # proteins dict: {go_id: {protein_id: embedding_array}}
    proteins = defaultdict(dict)
    skipped = 0

    if args.verbose:
        print(f"\nGenerating embeddings from {args.seq_file}...")

    for seq in SeqIO.parse(args.seq_file, "fasta"):
        match = re.search(r"GO:\d+", seq.description)
        if not match:
            if args.verbose:
                print(f"Warning: No GO ID found in sequence {seq.id}")
            skipped += 1
            continue

        go_id = match.group(0)
        sequence = str(seq.seq).upper()

        embedding = generate_embedding(
            esm_model, sequence, seq.id, verbose=args.verbose
        )
        if embedding is not None:
            proteins[go_id][seq.id] = embedding
        else:
            skipped += 1

    # Free ESM model memory before training
    del esm_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc

    gc.collect()

    if not proteins:
        print("Error: No sequences with GO IDs found in the input file")
        sys.exit(1)

    if args.verbose:
        print(f"\nEmbedding generation complete:")
        print(f"  GO terms: {len(proteins)}")
        total_proteins = sum(len(v) for v in proteins.values())
        print(f"  Proteins: {total_proteins}")
        print(f"  Skipped: {skipped}")
        print(f"  GO IDs: {list(proteins.keys())}")

    # Load config
    if args.config_path is not None:
        config = ConfigManager.load_config(args.config_path)
        if args.verbose:
            print(f"Using config file: {args.config_path}")
    else:
        config = ConfigManager.load_config()

    if args.config_override:
        try:
            config_override = json.loads(args.config_override)
            if args.verbose:
                print(
                    f"Config overrides: {json.dumps(config_override, indent=2)}"
                )
            config.update(config_override)
        except json.JSONDecodeError as e:
            print(f"Error parsing config_override JSON: {e}")
            sys.exit(1)

    # Generate a unique ID for this test run
    test_id = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())

    classifier = MultiClassifier(
        args.model_type,  # algorithm name is the ESM model type
        args.verbose,
        proteins,
        config,
        test_id,
        use_mlflow=args.use_mlflow,
        use_tensorboard=args.use_tensorboard,
        input_source="esm_embeddings",
    )

    if args.verbose:
        print(f"\nTraining MultiClassifier with ESM-C embeddings ({args.model_type})")

    try:
        classifier.run()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    elapsed_time = round(time.time() - start, 2)

    if args.verbose:
        print("\nTraining Summary:")
        print(f"\tESM model type: {args.model_type}")
        print(f"\tModel training time: {round(classifier.training_time, 2)}s")
        print(f"\tMLflow logging time: {round(classifier.logging_time, 2)}s")
        print(f"\tTotal elapsed time: {elapsed_time}s")
        if hasattr(classifier, "history") and classifier.history.history:
            history = classifier.history.history

            best_val_f1 = max(history["val_f1_score"])
            best_epoch = history["val_f1_score"].index(best_val_f1) + 1
            epochs_run = len(history["val_f1_score"])

            print(
                f"\tBest validation F1 Score: {round(best_val_f1, 4)} (at Epoch {best_epoch})"
            )
            print(
                f"\tTotal epochs run: {epochs_run} (out of {classifier.epochs})"
            )
        if hasattr(classifier, "model"):
            print(f"\tModel parameters: {classifier.model.count_params():,}")
