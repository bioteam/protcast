"""test_multilabel_small_scale.py

Small-scale end-to-end training and inference for the MultiLabelClassifier.
Uses the test FASTA data (which has GO terms in headers) to run the full
pipeline: parse sequences -> generate embeddings -> train -> evaluate -> infer.

Supports both fake embeddings (for quick testing without ESM-C) and real
ESM-C embeddings (matching the pattern from test_multi_classifier_embeds.py).

Usage:

    # Quick test with fake embeddings (no GPU needed, ~1 second)
    python3 test/test_multilabel_small_scale.py \
        -s test/data/random-level-4_200.fa \
        -v

    # Real ESM-C embeddings (requires ESM-C model)
    python3 test/test_multilabel_small_scale.py \
        -s test/data/random-level-4_200.fa \
        --use_esm \
        --model_type esmc_300m \
        -v

    # Real embeddings + MLflow logging
    python3 test/test_multilabel_small_scale.py \
        -s test/data/random-level-4_200.fa \
        --use_esm \
        --model_type esmc_300m \
        --use_mlflow \
        -v

    # With custom config overrides
    python3 test/test_multilabel_small_scale.py \
        -s test/data/random-level-4_200.fa \
        --use_esm \
        --config_override '{"EPOCHS": 50, "PATIENCE": 15}' \
        -v
"""

import re
import sys
import os
import time
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protcast.model.multilabel_classifier import (
    MultiLabelClassifier,
    GOEncoder,
)
from protcast.model.stats.utils import calculate_fmax, calculate_smin
from protcast.config.model_config import ConfigManager


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def fake_esm_embedding(sequence, embed_dim=128, seed=None):
    """Generate a deterministic fake embedding from a protein sequence.

    Uses a hash of the sequence as the random seed so the same sequence
    always produces the same embedding. This lets us test the full pipeline
    without needing ESM-C installed.
    """
    if seed is None:
        seed = hash(sequence) % (2**31)
    rng = np.random.RandomState(seed)
    return rng.randn(embed_dim).astype(np.float32)


def load_esm_model(model_name, verbose=False):
    """Load an ESM-C model."""
    import torch
    from esm.models.esmc import ESMC

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Loading ESM-C model: {model_name} on {device}")

    model = ESMC.from_pretrained(model_name, device=device)
    model.eval()

    if verbose:
        print("ESM-C model loaded successfully")
    return model


def generate_esm_embedding(esm_model, sequence, protein_id, verbose=False):
    """Generate an ESM-C embedding for a single protein sequence."""
    import torch
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


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def parse_fasta_multilabel(fasta_path, verbose=False):
    """Parse a FASTA file where headers contain GO terms.

    Returns protein-centric data structures for multi-label training:
      - protein_sequences: {protein_id: sequence}
      - protein_go_terms:  {protein_id: set[GO_ID, ...]}
      - go_ids:            list of all unique GO IDs

    If the same protein appears under multiple GO terms, its GO terms
    are merged (this is the key difference from the multiclass parser).
    """
    protein_sequences = {}
    protein_go_terms = defaultdict(set)

    for seq in SeqIO.parse(fasta_path, "fasta"):
        match = re.search(r"GO:\d+", seq.description)
        if not match:
            if verbose:
                print(f"Warning: No GO ID in {seq.id}, skipping")
            continue

        go_id = match.group(0)
        pid = seq.id
        protein_sequences[pid] = str(seq.seq)
        protein_go_terms[pid].add(go_id)

    go_ids = sorted(set().union(*protein_go_terms.values()))

    if verbose:
        n_multi = sum(1 for gos in protein_go_terms.values() if len(gos) > 1)
        print(f"Parsed {len(protein_sequences)} unique proteins")
        print(f"Found {len(go_ids)} GO terms: {go_ids}")
        print(f"Proteins with >1 GO term: {n_multi}")
        for go_id in go_ids:
            count = sum(1 for gos in protein_go_terms.values() if go_id in gos)
            print(f"  {go_id}: {count} proteins")

    return protein_sequences, dict(protein_go_terms), go_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Small-scale MultiLabelClassifier training and inference"
    )
    parser.add_argument(
        "-s", "--seq_file", required=True,
        help="Path to FASTA file with GO terms in headers"
    )
    parser.add_argument(
        "--use_esm", action="store_true",
        help="Use real ESM-C embeddings (default: fake embeddings)"
    )
    parser.add_argument(
        "--model_type",
        default="esmc_300m",
        choices=["esm3_c", "esmc_300m", "esmc_600m"],
        help="ESM-C model type (default: esmc_300m for faster testing)",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=128,
        help="Fake embedding dimension when --use_esm is not set (default: 128)"
    )
    parser.add_argument(
        "--use_tensorboard", action="store_true", help="Use TensorBoard logging"
    )
    parser.add_argument(
        "--use_mlflow", action="store_true", help="Use MLFlow logging"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--config_override", type=str,
        help="JSON string to override config values"
    )
    parser.add_argument(
        "--config_path", type=str, help="Path to custom config file"
    )
    args = parser.parse_args()

    start = time.time()

    # --- Step 1: Parse FASTA into multi-label format ---
    print("=" * 60)
    print("STEP 1: Parsing FASTA (multi-label)")
    print("=" * 60)
    protein_sequences, protein_go_terms, go_ids = parse_fasta_multilabel(
        args.seq_file, verbose=args.verbose
    )

    if not protein_sequences:
        print("Error: No sequences with GO IDs found")
        sys.exit(1)

    # --- Step 2: Generate embeddings ---
    print("\n" + "=" * 60)
    if args.use_esm:
        print(f"STEP 2: Generating ESM-C embeddings ({args.model_type})")
    else:
        print(f"STEP 2: Generating fake embeddings (dim={args.embed_dim})")
    print("=" * 60)

    protein_embeddings = {}
    skipped = 0

    if args.use_esm:
        esm_model = load_esm_model(args.model_type, args.verbose)

        for pid, seq in protein_sequences.items():
            embedding = generate_esm_embedding(
                esm_model, seq, pid, verbose=args.verbose
            )
            if embedding is not None:
                protein_embeddings[pid] = embedding
            else:
                skipped += 1

        # Free ESM model memory before training
        import torch
        del esm_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        if args.verbose:
            print(f"\nEmbedding generation complete:")
            print(f"  Embedded: {len(protein_embeddings)}")
            print(f"  Skipped:  {skipped}")
            if protein_embeddings:
                sample = next(iter(protein_embeddings.values()))
                print(f"  Embedding dim: {sample.shape[0]}")

        # Remove skipped proteins from go_terms
        for pid in list(protein_go_terms.keys()):
            if pid not in protein_embeddings:
                del protein_go_terms[pid]

    else:
        for pid, seq in protein_sequences.items():
            protein_embeddings[pid] = fake_esm_embedding(
                seq, embed_dim=args.embed_dim
            )

        if args.verbose:
            sample_pid = next(iter(protein_embeddings))
            print(f"Embedding dim: {args.embed_dim}")
            print(f"Sample ({sample_pid}): shape={protein_embeddings[sample_pid].shape}")

    if not protein_embeddings:
        print("Error: No embeddings generated")
        sys.exit(1)

    # --- Step 3: Load config ---
    if args.config_path is not None:
        config = ConfigManager.load_config(args.config_path)
    else:
        config = ConfigManager.load_config()

    if args.config_override:
        try:
            config.update(json.loads(args.config_override))
            if args.verbose:
                print(f"Config overrides applied: {args.config_override}")
        except json.JSONDecodeError as e:
            print(f"Error parsing config_override: {e}")
            sys.exit(1)

    # --- Step 4: Train ---
    print("\n" + "=" * 60)
    print("STEP 3: Training MultiLabelClassifier")
    print("=" * 60)

    test_id = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())
    classifier = MultiLabelClassifier(
        verbose=args.verbose,
        protein_embeddings=protein_embeddings,
        protein_go_terms=protein_go_terms,
        go_ids=go_ids,
        config=config,
        id=test_id,
        use_mlflow=args.use_mlflow,
        use_tensorboard=args.use_tensorboard,
    )

    try:
        classifier.run()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # --- Step 5: Evaluate ---
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation")
    print("=" * 60)

    y_val_pred = classifier.model.predict(classifier.X_val, verbose=0)
    fmax, fmax_threshold = calculate_fmax(classifier.y_val, y_val_pred)
    smin, smin_threshold = calculate_smin(classifier.y_val, y_val_pred)

    print(f"Validation Fmax:  {fmax:.4f}  (threshold={fmax_threshold:.2f})")
    print(f"Validation Smin:  {smin:.4f}  (threshold={smin_threshold:.2f})")

    # --- Step 6: Inference demo ---
    print("\n" + "=" * 60)
    print("STEP 5: Inference demo (first 5 proteins)")
    print("=" * 60)

    go_encoder = classifier.go_encoder
    demo_pids = list(protein_embeddings.keys())[:5]

    print(f"{'Protein':<15} {'True GO terms':<35} {'Predicted GO terms':<35}")
    print("-" * 85)
    for pid in demo_pids:
        embedding = protein_embeddings[pid].reshape(1, -1)
        y_pred = classifier.model.predict(embedding, verbose=0)[0]

        predictions = go_encoder.decode_multilabel(
            y_pred, threshold=classifier.best_threshold
        )
        pred_str = ", ".join(
            f"{go}({p:.2f})" for go, p in predictions
        ) or "NONE"

        true_str = ", ".join(sorted(protein_go_terms[pid]))

        print(f"{pid:<15} {true_str:<35} {pred_str:<35}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    elapsed = round(time.time() - start, 2)
    embed_dim = next(iter(protein_embeddings.values())).shape[0]
    print(f"  Input source:       {'ESM-C ' + args.model_type if args.use_esm else 'fake embeddings'}")
    print(f"  Proteins:           {len(protein_embeddings)}")
    print(f"  GO terms:           {len(go_ids)}")
    print(f"  Embedding dim:      {embed_dim}")
    print(f"  Model params:       {classifier.model.count_params():,}")
    print(f"  Training time:      {round(classifier.training_time, 2)}s")
    print(f"  Validation Fmax:    {fmax:.4f}")
    print(f"  Validation Smin:    {smin:.4f}")
    print(f"  Best threshold:     {classifier.best_threshold:.2f}")
    print(f"  Total elapsed:      {elapsed}s")

    if hasattr(classifier, "history") and classifier.history.history:
        history = classifier.history.history
        epochs_run = len(history.get("loss", []))
        best_val_loss = min(history.get("val_loss", [float("inf")]))
        print(f"  Epochs run:         {epochs_run}")
        print(f"  Best val loss:      {best_val_loss:.4f}")

    print("\nDone!")
