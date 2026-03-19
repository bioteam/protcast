"""
run_multilabel_inference_embeds.py

Perform multi-label inference using a model trained with ESM-C embeddings.

Unlike multi-class inference (which predicts ONE GO term per protein),
this returns ALL GO terms above a confidence threshold for each protein.

Example usage:

python3 scripts/run_multilabel_inference_embeds.py \
    -m model_multilabel.keras \
    -g model_multilabel_GOEncoder.pkl \
    -s query_sequences.fa \
    --model_type esmc_600m \
    --threshold 0.5 \
    -v
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from collections import Counter
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from protcast.model.multilabel_classifier import (  # noqa: E402
    MultiLabelClassifier,
    GOEncoder,
)


def load_esm_model(model_name, verbose=False):
    """Load an ESM-C model."""
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f"Loading ESM-C model: {model_name} on {device}")
        model = ESMC.from_pretrained(model_name, device=device)
        model.eval()
        if verbose:
            print("ESM-C model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading ESM-C model: {e}")
        sys.exit(1)


def get_embedding_for_sequence(model, sequence, protein_id, verbose=False):
    """Generate ESM-C embedding for a single protein sequence."""
    try:
        protein = ESMProtein(sequence=sequence)
        with torch.no_grad():
            protein_tensor = model.encode(protein)
            output = model.forward(
                sequence_tokens=protein_tensor.sequence.unsqueeze(0)
            )
            sequence_embeddings = output.embeddings.squeeze(0).to(
                dtype=torch.float32
            )
            protein_embedding = sequence_embeddings.mean(dim=0).cpu().numpy()
            return protein_embedding
    except Exception as e:
        print(f"Error processing protein {protein_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Multi-label inference using ESM-C embeddings"
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
        "-s", "--seq_file", required=True, help="Path to FASTA file"
    )
    parser.add_argument(
        "--model_type",
        default="esmc_600m",
        choices=["esm3_c", "esmc_300m", "esmc_600m"],
        help="ESM-C model type for embedding generation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Prediction threshold (default: use best threshold from training)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    parser.add_argument("--use_mlflow", action="store_true", help="Use MLFlow")
    args = parser.parse_args()

    start = time.time()

    # Load model and encoder
    if args.verbose:
        print("Loading model and GOEncoder...")
    model = MultiLabelClassifier.load_model(Path(args.model_file))
    go_decoder = GOEncoder.load(args.goencoder_file)

    if args.verbose:
        print(f"GO classes: {go_decoder.num_classes}")

    # Determine threshold
    threshold = args.threshold
    if threshold is None:
        # Try to load best threshold from encoder metadata
        threshold = getattr(go_decoder, "best_threshold", 0.5)
        if args.verbose:
            print(f"Using threshold: {threshold}")

    # Load ESM model
    esm_model = load_esm_model(args.model_type, args.verbose)

    # Header
    print("Protein\tGO_Terms\tProbabilities\tNum_Predictions")

    processed_count = 0
    skipped_count = 0
    total_predictions = 0
    go_term_counts = Counter()

    for seq in SeqIO.parse(args.seq_file, "fasta"):
        seqstr = str(seq.seq).upper()

        if len(seqstr) == 0:
            skipped_count += 1
            continue

        # Generate embedding
        embedding = get_embedding_for_sequence(
            esm_model, seqstr, seq.id, verbose=args.verbose
        )

        if embedding is None:
            skipped_count += 1
            continue

        X_test = embedding.reshape(1, -1).astype(np.float32)

        # Get sigmoid predictions
        y_pred = model.predict(X_test, verbose=0)[0]

        # Decode all terms above threshold
        predictions = go_decoder.decode_multilabel(y_pred, threshold=threshold)

        if predictions:
            go_terms = ",".join(go_id for go_id, _ in predictions)
            probs = ",".join(f"{prob:.4f}" for _, prob in predictions)
            print(f"{seq.id}\t{go_terms}\t{probs}\t{len(predictions)}")
            total_predictions += len(predictions)
            for go_id, _ in predictions:
                go_term_counts[go_id] += 1
        else:
            print(f"{seq.id}\tNONE\t-\t0")

        processed_count += 1

    end = time.time()

    if args.verbose:
        print(f"\n--- Summary ---")
        print(f"Processed: {processed_count} sequences")
        print(f"Skipped: {skipped_count} sequences")
        print(f"Total GO predictions: {total_predictions}")
        if processed_count > 0:
            print(f"Avg predictions per protein: {total_predictions/processed_count:.1f}")
        print(f"Unique GO terms predicted: {len(go_term_counts)}")
        print(f"Threshold used: {threshold}")
        print(f"Total time: {round(end - start)}s")

    # MLflow logging
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
            params = {
                "model_file": args.model_file,
                "goencoder_file": args.goencoder_file,
                "seq_file": args.seq_file,
                "esm_model_type": args.model_type,
                "model_type": "multi_label",
                "threshold": threshold,
                "num_classes": go_decoder.num_classes,
            }
            run_meta = load_run_metadata(args.model_file)
            if run_meta:
                params["training_run_id"] = run_meta["training_run_id"]

            metrics = {
                "processed_sequences": processed_count,
                "skipped_sequences": skipped_count,
                "total_predictions": total_predictions,
                "unique_go_terms_predicted": len(go_term_counts),
                "inference_time_seconds": round(end - start),
            }
            if processed_count > 0:
                metrics["avg_predictions_per_protein"] = round(
                    total_predictions / processed_count, 2
                )

            log_inference_results(
                mlflow,
                params=params,
                metrics=metrics,
                tags={
                    "Inference Info": "MultiLabelClassifier ESM-embedding inference",
                    "model_type": "multi_label",
                },
            )
            print("MLflow inference logging complete.")


if __name__ == "__main__":
    main()
