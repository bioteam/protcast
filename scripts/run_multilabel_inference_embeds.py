"""
run_multilabel_inference_embeds.py

Perform multi-label inference using a model trained with ESM-C embeddings.

Unlike multi-class inference (which predicts ONE GO term per protein),
this returns ALL GO terms above a confidence threshold for each protein,
with human-readable GO term names, namespaces, and calibrated confidence.

Example usage:

python3 scripts/run_multilabel_inference_embeds.py \
    -m model_multilabel.keras \
    -g model_multilabel_GOEncoder.pkl \
    -s query_sequences.fa \
    --model_type esmc_600m \
    --obo test/data/go-2023-11-15.obo \
    -v

Without OBO file (no GO term names, just IDs):

python3 scripts/run_multilabel_inference_embeds.py \
    -m model_multilabel.keras \
    -g model_multilabel_GOEncoder.pkl \
    -s query_sequences.fa \
    --model_type esmc_600m \
    -v
"""

import sys
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from collections import Counter
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein

from protcast.model.multilabel_classifier import (  # noqa: E402
    MultiLabelClassifier,
    GOEncoder,
    get_confidence_label,
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


def load_go_metadata(obo_path, verbose=False):
    """Load GO term names and namespaces from an OBO file.

    Returns
    -------
    dict
        {GO_ID: {"name": str, "namespace": str}} or empty dict if loading fails.
    """
    try:
        from goatools.obo_parser import GODag
        dag = GODag(obo_path, prt=None)
        metadata = {}
        for go_id, term in dag.items():
            if not term.is_obsolete:
                metadata[go_id] = {
                    "name": term.name,
                    "namespace": term.namespace,
                }
        if verbose:
            print(f"Loaded {len(metadata)} GO terms from {obo_path}")
        return metadata
    except Exception as e:
        if verbose:
            print(f"Warning: Could not load OBO file: {e}")
        return {}


# Namespace abbreviations for compact output
NS_SHORT = {
    "biological_process": "BP",
    "cellular_component": "CC",
    "molecular_function": "MF",
}


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
    parser.add_argument(
        "--obo",
        type=str,
        default=None,
        help="Path to GO OBO file for term names and namespaces",
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
        threshold = getattr(go_decoder, "best_threshold", 0.5)
    if args.verbose:
        print(f"Using threshold: {threshold}")

    # Load GO metadata if OBO file provided
    go_metadata = {}
    if args.obo:
        go_metadata = load_go_metadata(args.obo, args.verbose)

    # Load ESM model
    esm_model = load_esm_model(args.model_type, args.verbose)

    # Header — columns depend on whether we have GO metadata
    if go_metadata:
        print("Protein\tGO_ID\tGO_Name\tNamespace\tScore\tConfidence")
    else:
        print("Protein\tGO_ID\tScore\tConfidence")

    processed_count = 0
    skipped_count = 0
    total_predictions = 0
    go_term_counts = Counter()
    confidence_counts = Counter()

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
            for go_id, prob in predictions:
                confidence = get_confidence_label(prob, threshold)
                confidence_counts[confidence] += 1
                go_term_counts[go_id] += 1

                if go_metadata and go_id in go_metadata:
                    meta = go_metadata[go_id]
                    ns = NS_SHORT.get(meta["namespace"], meta["namespace"])
                    print(
                        f"{seq.id}\t{go_id}\t{meta['name']}\t{ns}"
                        f"\t{prob:.4f}\t{confidence}"
                    )
                else:
                    print(f"{seq.id}\t{go_id}\t{prob:.4f}\t{confidence}")

            total_predictions += len(predictions)
        else:
            if go_metadata:
                print(f"{seq.id}\t-\t-\t-\t-\tNO_PREDICTION")
            else:
                print(f"{seq.id}\t-\t-\tNO_PREDICTION")

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
        print(f"Confidence distribution:")
        for level in ["VERY_HIGH", "HIGH", "MEDIUM", "LOW"]:
            count = confidence_counts.get(level, 0)
            print(f"  {level}: {count}")
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
            for level, count in confidence_counts.items():
                metrics[f"confidence_{level.lower()}"] = count

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
