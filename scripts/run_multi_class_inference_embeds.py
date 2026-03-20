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
from collections import Counter
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein
from protein_feature_vectors import Calculator

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
    parser.add_argument("--use_mlflow", action="store_true", help="Use MLFlow")
    parser.add_argument(
        "--input_source",
        default="esm_embeddings",
        choices=["esm_embeddings", "combined"],
        help="Input source used during training (default: esm_embeddings)",
    )
    parser.add_argument(
        "--scalers_file",
        default=None,
        help="Path to scalers pickle file (required for combined mode)",
    )
    args = parser.parse_args()

    if args.input_source == "combined" and not args.scalers_file:
        parser.error("--scalers_file is required when using --input_source combined")

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

    # Load scalers for combined mode
    scalers = None
    fv_calc = None
    if args.input_source == "combined":
        scalers = MultiClassifier.load_scalers(args.scalers_file)
        fv_calc = Calculator(verbose=args.verbose)
        if args.verbose:
            print(f"Scalers loaded from {args.scalers_file}")
            print(f"Feature algorithms: {scalers['feature_algorithms']}")
            print(
                f"Expected dimensions: ESM={scalers['embedding_dim']}, "
                f"FV={scalers['fv_dim']}"
            )

    if args.verbose:
        print("\nProcessing sequences...")

    print("Protein\tPredicted GO\tProbability\tConfidence")

    processed_count = 0
    skipped_count = 0
    probabilities = []
    pred_counts = Counter()
    confidence_counts = {
        "VERY_HIGH": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "VERY_LOW": 0,
    }

    for seq in SeqIO.parse(args.seq_file, "fasta"):
        seqstr = str(seq.seq).upper()

        # Validate sequence
        if len(seqstr) == 0:
            if args.verbose:
                print(f"Skipping {seq.id}: empty sequence")
            skipped_count += 1
            continue

        # Generate ESM-C embedding for this sequence
        embedding = get_embedding_for_sequence(
            esm_model, seqstr, seq.id, verbose=args.verbose
        )

        if embedding is None:
            if args.verbose:
                print(f"Skipping {seq.id}: embedding generation failed")
            skipped_count += 1
            continue

        if args.input_source == "combined":
            # Generate feature vectors for this sequence
            pdict = {seq.id: seqstr}
            fv_parts = []
            for algo in scalers["feature_algorithms"]:
                fv_calc.get_feature_vectors(algo, pdict=pdict)
                if fv_calc.encodings is None:
                    if args.verbose:
                        print(f"Skipping {seq.id}: {algo} encoding failed")
                    skipped_count += 1
                    continue
                fv_parts.append(
                    fv_calc.encodings.iloc[0].values.astype(np.float32)
                )

            if len(fv_parts) != len(scalers["feature_algorithms"]):
                continue  # Skip if any feature algorithm failed

            fv_vector = np.concatenate(fv_parts)

            # Apply saved scalers (transform only, no fitting)
            emb_scaled = scalers["emb_scaler"].transform(
                embedding.reshape(1, -1)
            )
            fv_scaled = scalers["fv_scaler"].transform(
                fv_vector.reshape(1, -1)
            )
            X_test = np.hstack([emb_scaled, fv_scaled]).astype(np.float32)
        else:
            # Reshape embedding for model input: [embedding_dim] -> [1, embedding_dim]
            X_test = embedding.reshape(1, -1).astype(np.float32)

        # Get predictions from the model
        y_pred = model.predict(X_test, verbose="auto")

        # Decode predictions to GO ID with probability
        pred_tup = go_decoder.decode_probabilities(y_pred, top_k=1)[0][0]

        # Get confidence level
        confidence = get_confidence_level(pred_tup[1])
        confidence_counts[confidence] += 1
        probabilities.append(pred_tup[1])
        pred_counts[pred_tup[0]] += 1

        print(f"{seq.id}\t{pred_tup[0]}\t{pred_tup[1]:.4f}\t{confidence}")

        processed_count += 1

    end = time.time()
    inference_time = round(end - start)

    if args.verbose:
        print(f"\n--- Summary ---")
        print(f"Processed: {processed_count} sequences")
        print(f"Skipped: {skipped_count} sequences")
        print(f"Total time: {inference_time}s")

    # --- MLflow logging for inference ---
    if args.use_mlflow:
        from protcast.config.model_config import ConfigManager
        from protcast.utils.mlflow_utils import (
            init_mlflow,
            log_inference_results,
            load_run_metadata,
        )

        config = ConfigManager.load_config()
        mlflow = init_mlflow(
            experiment_name=config.get("EXPERIMENT_NAME", "Default Experiment"),
            repo_owner=config.get("DAGSHUB_REPO_OWNER", "aakpan"),
            repo_name=config.get("DAGSHUB_REPO_NAME", "my-first-repo"),
            verbose=args.verbose,
        )
        if mlflow is not None:
            # Build params, including training run link if available
            params = {
                "model_file": args.model_file,
                "goencoder_file": args.goencoder_file,
                "seq_file": args.seq_file,
                "esm_model_type": args.model_type,
                "input_source": args.input_source,
                "num_classes": go_decoder.num_classes,
            }
            run_meta = load_run_metadata(args.model_file)
            if run_meta:
                params["training_run_id"] = run_meta["training_run_id"]

            # Build metrics
            metrics = {
                "processed_sequences": processed_count,
                "skipped_sequences": skipped_count,
                "inference_time_seconds": inference_time,
            }
            # Confidence distribution
            for level, count in confidence_counts.items():
                metrics[f"confidence_{level.lower()}"] = count
            # Mean probability
            if probabilities:
                metrics["mean_top_probability"] = round(
                    float(np.mean(probabilities)), 4
                )
            # Per-class prediction counts
            for go_id, count in pred_counts.items():
                metrics[f"pred_count_{go_id}"] = count

            log_inference_results(
                mlflow,
                params=params,
                metrics=metrics,
                tags={
                    "Inference Info": f"MultiClassifier {args.input_source} inference",
                    "input_source": args.input_source,
                },
            )
            print("MLflow inference logging complete.")


if __name__ == "__main__":
    main()
