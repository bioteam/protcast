"""
run_multi_class_inference_embeds.py

Perform multi-class inference using a model trained with ESM-C embeddings.

This script generates ESM-C embeddings for query sequences and then classifies
them using a pre-trained model. The model must have been trained with ESM-C
embeddings (using input_source="esm_embeddings" in MultiClassifier).

Example usage:

python3 scripts/run_multi_class_inference_embeds.py \
    -m model_esm.keras \
    -g model_esm_GOEncoder.pickle \
    -s query_sequences.fa \
    --model_type esmc_600m \
    -v
"""

import re
import sys
import os
import argparse
import time
import numpy as np
import pickle
import torch
from pathlib import Path
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from protcast.model.multi_classifier import MultiClassifier  # noqa: E402
from protcast.model.multi_classifier import GOEncoder  # noqa: E402


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


def load_esm_model(model_name, verbose=False):
    """Load an ESM-C model using the correct ESM 3.2.1 API."""
    try:
        if verbose:
            print(f"Loading ESM-C model: {model_name}")

        # Use GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f"Using device: {device}")

        # Load the model
        model = ESMC.from_pretrained(model_name, device=device)
        model.eval()

        if verbose:
            print("✓ ESM-C model loaded successfully!")
            print(f"Model name: {model_name}")

        return model

    except Exception as e:
        print(f"Error loading ESM-C model: {e}")
        sys.exit(1)


def get_embedding_for_sequence(model, sequence, protein_id, verbose=False):
    """
    Generate ESM-C embedding for a single protein sequence.

    Parameters
    ----------
    model : ESMC
        ESM-C model instance
    sequence : str
        Protein sequence
    protein_id : str
        Protein identifier
    verbose : bool
        Whether to print verbose output

    Returns
    -------
    np.ndarray
        Embedding vector (shape: [embedding_dim])
    """
    try:
        # Create ESMProtein object
        protein = ESMProtein(sequence=sequence)

        if verbose:
            print(
                f"Creating embedding for {protein_id} (length: {len(sequence)})"
            )

        # Get embeddings using ESM-C
        with torch.no_grad():
            # Encode protein to get tokens
            protein_tensor = model.encode(protein)

            # Get embeddings through forward pass
            # Add batch dimension (unsqueeze) for the sequence tokens
            output = model.forward(
                sequence_tokens=protein_tensor.sequence.unsqueeze(0)
            )

            # Extract embeddings and mean pool over sequence length
            # Shape: [batch=1, seq_len, embed_dim] -> [embed_dim]
            sequence_embeddings = output.embeddings.squeeze(0).to(
                dtype=torch.float32
            )  # Remove batch dim, convert from bfloat16
            protein_embedding = sequence_embeddings.mean(dim=0).cpu().numpy()

            if verbose:
                print(f"  ✓ Embedding shape: {protein_embedding.shape}")

            return protein_embedding

    except Exception as e:
        print(f"❌ Error processing protein {protein_id}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Multi-class inference using ESM-C embeddings"
    )
    parser.add_argument(
        "-m", "--model_file", required=True, help="Path to model file (.keras)"
    )
    parser.add_argument(
        "-g",
        "--goencoder_file",
        required=True,
        help="Path to GOEncoder pickle file",
    )
    parser.add_argument(
        "-s", "--seq_file", required=True, help="Path to FASTA file with sequences"
    )
    parser.add_argument(
        "--model_type",
        default="esmc_600m",
        choices=["esm3_c", "esmc_300m", "esmc_600m"],
        help="ESM-C model type to use for generating embeddings",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    start = time.time()

    # Load the trained model and GO encoder
    if args.verbose:
        print("Loading trained model and GOEncoder...")
    model = MultiClassifier.load_model(Path(args.model_file))
    go_decoder = GOEncoder.load(args.goencoder_file)

    if args.verbose:
        print(f"Model loaded from {args.model_file}")
        print(f"GOEncoder loaded from {args.goencoder_file}")
        print(f"Number of GO classes: {go_decoder.num_classes}")

    # Load ESM-C model for generating embeddings
    esm_model = load_esm_model(args.model_type, args.verbose)

    if args.verbose:
        print("\nProcessing sequences...")

    print("Protein\tPredicted GO\tProbability\tConfidence")

    processed_count = 0
    skipped_count = 0

    for seq in SeqIO.parse(args.seq_file, "fasta"):
        seqstr = str(seq.seq).upper()

        # Validate sequence
        if len(seqstr) == 0:
            if args.verbose:
                print(f"⚠ Skipping {seq.id}: empty sequence")
            skipped_count += 1
            continue

        # Generate ESM-C embedding for this sequence
        embedding = get_embedding_for_sequence(
            esm_model, seqstr, seq.id, verbose=args.verbose
        )

        if embedding is None:
            if args.verbose:
                print(f"⚠ Skipping {seq.id}: embedding generation failed")
            skipped_count += 1
            continue

        # Reshape embedding for model input: [embedding_dim] -> [1, embedding_dim]
        X_test = embedding.reshape(1, -1).astype(np.float32)

        # Get predictions from the model
        y_pred = model.predict(X_test, verbose="auto")

        # Decode predictions to GO ID with probability
        pred_tup = go_decoder.decode_probabilities(y_pred, top_k=1)[0][0]

        # Get confidence level
        confidence = get_confidence_level(pred_tup[1])

        print(f"{seq.id}\t{pred_tup[0]}\t{pred_tup[1]:.4f}\t{confidence}")

        processed_count += 1

    end = time.time()

    if args.verbose:
        print(f"\n--- Summary ---")
        print(f"Processed: {processed_count} sequences")
        print(f"Skipped: {skipped_count} sequences")
        print(f"Total time: {round(end - start)}s")


if __name__ == "__main__":
    main()
