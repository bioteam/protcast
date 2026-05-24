"""
ProtCast ESM3 Integration

This script demonstrates how to integrate ESM-C (Cambrian) embeddings with ProtCast for protein
function prediction. These embeddings represent proteins in vector space, which can be helpful
for predicting protein functions, properties, or other downstream tasks.

ESM-C sizes and dimensions:

esm3_c:  2560 dimensions, 2B parameters, 80 layers
esmc_300m: 960 dimensions, 300M parameters, 30 layers
esmc_600m: 1152 dimensions, 600M parameters, 36 layers

The embedding dimension increases with model size, allowing larger models to capture more complex protein representations.

Per-residue vs. whole-protein embeddings:

For a protein with length L, the per-residue embeddings would have shape (L, D) where D is the embedding dimension
When using average pooling to get a whole-protein representation, you get a vector of size (D)

These embeddings are contextual, meaning each amino acid's representation is influenced by its surrounding sequence

Pooling strategies (--pooling):

mean         (default) per-dimension average across residues → vector of size D.
             Standard practice for ESM-family embeddings.
mean_max_std concatenation of per-dimension mean, max, and standard deviation
             across residues → vector of size 3*D. Captures dispersion and
             extreme-value information that mean pooling discards.
"""

import torch
import numpy  # noqa: F401
import argparse
import pickle
import random
import sys
from pathlib import Path
from collections import defaultdict
from esm.models.esmc import ESMC

from protcast.preprocessing.protcast_dataset import ProtCastDataset
from protcast.config.model_config import ConfigManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ESM3 embeddings for proteins in a ProtCastDataset"
    )
    parser.add_argument(
        "-p",
        "--protcast_dataset",
        required=True,
        help="Path to ProtCastDataset binary file",
    )
    parser.add_argument(
        "-g",
        "--go_ids_file",
        required=True,
        help="Path to file with GO IDs to process",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Directory where per-GO-term embedding pickles will be written",
    )
    parser.add_argument(
        "--model_type",
        default="esmc_600m",
        choices=["esm3_c", "esmc_300m", "esmc_600m"],
        help="ESM-C model type to use (esm3_c defaults to esmc_600m)",
    )
    parser.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "mean_max_std"],
        help=(
            "Strategy for reducing per-residue ESM-C embeddings to a "
            "single per-protein vector. 'mean' (default) produces a "
            "D-dimensional vector; 'mean_max_std' concatenates per-dimension "
            "mean, max, and standard deviation, producing a 3*D-dimensional "
            "vector. Use a separate --output_dir for non-default pooling, "
            "since the output filenames do not encode the strategy."
        ),
    )
    parser.add_argument(
        "--minimum_seqs",
        default=10,
        help="Minimum number of sequences per GO term (terms below this are skipped)",
        type=int,
    )
    parser.add_argument(
        "--maximum_seqs",
        default=None,
        help="Optional cap on sequences per GO term; if unset, all sequences are used",
        type=int,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for per-GO-term down-sampling when --maximum_seqs "
            "is set. If unset, falls back to RANDOM_SEED in config.json, "
            "then to 42."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing embeddings for GO term",
    )
    return parser.parse_args()


def load_go_ids(go_ids_file):
    """Load GO IDs from a file."""
    import re

    go_ids = set()
    with open(go_ids_file, "r") as f:
        for line in f:
            match = re.search(r"GO:\d+", line)
            if match:
                go_ids.add(match.group(0))

    if not go_ids:
        print(f"Warning: No GO IDs found in {go_ids_file}")

    return go_ids


def get_proteins_for_go_terms(
    dataset, go_ids, minimum_seqs, maximum_seqs, rng, verbose=False
):
    """
    Get protein sequences for specified GO terms.

    Parameters:
    -----------
    dataset : ProtCastDataset
        Dataset containing protein sequences and GO annotations
    go_ids : set
        Set of GO IDs to process
    minimum_seqs : int
        Minimum number of proteins per GO term
    maximum_seqs : int
        Maximum number of proteins per GO term
    rng : random.Random
        Seeded RNG used for down-sampling when a GO term has more than
        `maximum_seqs` proteins. A single instance is reused across GO terms
        so the full sampling sequence is deterministic for a fixed seed.
    verbose : bool
        Whether to print verbose output

    Returns:
    --------
    proteins_by_go : dict
        Dictionary mapping GO IDs to dictionaries of protein_id: sequence
    """
    proteins_by_go = defaultdict(dict)

    # `go_ids` is a set, so its iteration order is non-deterministic across
    # processes. Sorting fixes the order; this matters because `rng` is
    # reused across GO terms, so the order of GO terms determines which
    # protein samples each one gets.
    for go_id in sorted(go_ids):
        # Get subgraph of GO terms (the term and all its descendants)
        subgraph_go_ids = dataset.get_subgraph(go_id)

        # Collect protein IDs and sequences from the subgraph
        for subid in subgraph_go_ids:
            term = dataset.get_term(subid)
            if term is not None:
                pids = term.get_all_pids()
                if pids:
                    # Add protein sequences to the collection for this GO term
                    for pid in pids:
                        if pid in dataset.proteins:
                            proteins_by_go[go_id][pid] = dataset.proteins[
                                pid
                            ].sequence
            else:
                if verbose:
                    print(f"Warning: GO term {subid} not found in dataset")

        # Check if we have enough proteins for this GO term
        if len(proteins_by_go[go_id]) < minimum_seqs:
            if verbose:
                print(
                    f"GO term {go_id} skipped: Only {len(proteins_by_go[go_id])} proteins, minimum {minimum_seqs} required"
                )
            del proteins_by_go[go_id]
        elif maximum_seqs is not None and len(proteins_by_go[go_id]) > maximum_seqs:
            # Sample down to maximum number of proteins. Sort by pid first
            # so the sample is reproducible across runs — dict iteration is
            # insertion-order stable within a process, but the upstream pid
            # collection path can differ between processes (e.g. if GO term
            # traversal order changes), so we don't rely on it.
            protein_items = sorted(proteins_by_go[go_id].items())
            sampled_items = rng.sample(protein_items, maximum_seqs)
            proteins_by_go[go_id] = dict(sampled_items)
            if verbose:
                print(
                    f"GO term {go_id}: Sampled down from {len(protein_items)} to {maximum_seqs} proteins"
                )
        else:
            if verbose:
                print(
                    f"GO term {go_id}: Using {len(proteins_by_go[go_id])} proteins"
                )

    return proteins_by_go


def pool_embeddings(sequence_embeddings, strategy):
    """Reduce per-residue embeddings of shape (L, D) to a single vector.

    strategy:
      mean         -> (D,) per-dimension mean across residues.
      mean_max_std -> (3*D,) concatenation of mean, max, and std across
                       residues. std uses unbiased=False so the result is
                       defined (zero) for L=1.
    """
    if strategy == "mean":
        return sequence_embeddings.mean(dim=0)
    if strategy == "mean_max_std":
        mean = sequence_embeddings.mean(dim=0)
        max_ = sequence_embeddings.max(dim=0).values
        std = sequence_embeddings.std(dim=0, unbiased=False)
        return torch.cat([mean, max_, std], dim=0)
    raise ValueError(f"Unknown pooling strategy: {strategy}")


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


def get_embeddings_for_term(model, sequences_dict, go_id, pooling="mean", verbose=False):
    """
    Process protein sequences using ESM-C API to generate embeddings.

    Parameters:
    -----------
    model : ESMCInferenceClient
        ESM-C inference client
    sequences_dict : dict
        Dictionary mapping protein IDs to sequences
    go_id : str
        GO term identifier
    verbose : bool
        Whether to print verbose output

    Returns:
    --------
    embeddings_dict : dict
        Dictionary mapping protein IDs to ESM-C embeddings
    """

    from esm.sdk.api import ESMProtein

    embeddings_dict = {}

    if verbose:
        print(
            f"Processing {len(sequences_dict)} sequences for GO term {go_id}..."
        )

    # ESM-C processes sequences individually
    for protein_id, sequence in sequences_dict.items():
        try:
            # Create ESMProtein object
            protein = ESMProtein(sequence=sequence)

            if verbose:
                print(
                    f"Creating embedding for {protein_id} (length: {len(sequence)}, GO term: {go_id})"
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

                # Extract per-residue embeddings, then reduce to a single
                # per-protein vector with the chosen pooling strategy.
                # Shape: [batch=1, seq_len, embed_dim] -> pooled vector.
                sequence_embeddings = output.embeddings.squeeze(
                    0
                ).to(  # Remove batch dim
                    dtype=torch.float32
                )  # Convert from bfloat16 before numpy conversion
                protein_embedding = (
                    pool_embeddings(sequence_embeddings, pooling)
                    .cpu()
                    .numpy()
                )

                embeddings_dict[protein_id] = protein_embedding

                if verbose:
                    print(
                        f"  ✓ Processed {protein_id}: embedding shape {protein_embedding.shape}"
                    )

        except Exception as e:
            print(f"❌ Error processing protein {protein_id}: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            continue

    if verbose:
        print(
            f"✓ Successfully processed {len(embeddings_dict)} proteins for GO term {go_id}"
        )
    return embeddings_dict


def clean_go_id(go_id):
    """Return a filesystem-friendly GO term identifier."""

    return go_id.replace(":", "_")


def main():
    args = parse_args()

    # Resolve seed: CLI flag > config["RANDOM_SEED"] > 42. config.json
    # absence is non-fatal so the script still works in environments where
    # the config file isn't deployed alongside it.
    if args.seed is None:
        try:
            _config = ConfigManager.load_config()
            args.seed = int(_config.get("RANDOM_SEED", 42))
        except FileNotFoundError:
            args.seed = 42
    if args.verbose:
        print(f"Random seed: {args.seed}")
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load GO IDs
    go_ids = load_go_ids(args.go_ids_file)
    if args.verbose:
        print(f"Loaded {len(go_ids)} GO terms from {args.go_ids_file}")

    # Load ProtCastDataset
    if args.verbose:
        print(f"Loading ProtCastDataset from {args.protcast_dataset}")
    dataset = ProtCastDataset.load_serialized_file(args.protcast_dataset)

    # Get proteins for each GO term
    proteins_by_go = get_proteins_for_go_terms(
        dataset,
        go_ids,
        args.minimum_seqs,
        args.maximum_seqs,
        rng,
    )

    if args.verbose:
        print(
            f"Found {len(proteins_by_go)} GO terms with > {args.minimum_seqs} proteins"
        )

    if not proteins_by_go:
        if args.verbose:
            print("No GO terms with sufficient proteins found. Exiting.")
        return

    # Load ESM model
    model = load_esm_model(args.model_type, args.verbose)

    # Process each GO term separately to manage memory
    saved_terms = 0

    for go_id, proteins in proteins_by_go.items():
        go_path = output_dir / f"{clean_go_id(go_id)}.pkl"

        if go_path.exists() and not args.force:
            if args.verbose:
                print(
                    f"Embeddings for GO term {go_id} already exist at {go_path}, skipping."
                )
            continue

        if args.verbose:
            print(f"Generating embeddings for GO term {go_id}")
        # Get embeddings for all proteins for this GO term
        embeddings = get_embeddings_for_term(
            model, proteins, go_id, args.pooling, args.verbose
        )

        with open(go_path, "wb") as f:
            pickle.dump(embeddings, f)

        saved_terms += 1

        if args.verbose:
            print(f"Saved embeddings for {go_id} to {go_path}")

    print(f"Saved embeddings for {saved_terms} GO terms in {output_dir}")


if __name__ == "__main__":
    main()
