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
"""

import torch
import numpy  # noqa: F401
import argparse
import os
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from protcast.preprocessing.protcast_dataset import ProtCastDataset


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
        "--minimum_seqs",
        default=500,
        help="Minimum number of sequences",
        type=int,
    )
    parser.add_argument(
        "--maximum_seqs",
        default=2000,
        help="Maximum number of sequences to use for training",
        type=int,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
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
    dataset, go_ids, minimum_seqs, maximum_seqs, verbose=False
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
    verbose : bool
        Whether to print verbose output

    Returns:
    --------
    proteins_by_go : dict
        Dictionary mapping GO IDs to dictionaries of protein_id: sequence
    """
    proteins_by_go = defaultdict(dict)

    for go_id in tqdm(go_ids, desc="Processing GO terms", disable=not verbose):
        # Get subgraph of GO terms (the term and all its descendants)
        subgraph_go_ids = dataset.get_subgraph(go_id)

        # Collect protein IDs and sequences from the subgraph
        for subid in subgraph_go_ids:
            pids = dataset.get_term(subid).get_all_pids()
            if pids:
                # Add protein sequences to the collection for this GO term
                for pid in pids:
                    if pid in dataset.proteins:
                        seq = dataset.proteins[pid].sequence
                        # DIAGNOSTIC: Check what we're getting from the dataset
                        if not isinstance(seq, str):
                            print(
                                f"\n=== WARNING: Non-string sequence in dataset ==="
                            )
                            print(f"  Protein ID: {pid}")
                            print(f"  GO term: {go_id}")
                            print(f"  Sequence type: {type(seq)}")
                            print(f"  Sequence value: {seq}")
                            print(f"  Protein object: {dataset.proteins[pid]}")
                            print(
                                f"  Protein object type: {type(dataset.proteins[pid])}"
                            )
                            if hasattr(dataset.proteins[pid], "__dict__"):
                                print(
                                    f"  Protein attributes: {dataset.proteins[pid].__dict__}"
                                )
                            print(f"=== END WARNING ===")
                        elif len(seq) > 10000:
                            print(
                                f"\n=== WARNING: Very long sequence in dataset ==="
                            )
                            print(f"  Protein ID: {pid}")
                            print(f"  GO term: {go_id}")
                            print(f"  Sequence length: {len(seq)}")
                            print(f"  First 100 chars: {seq[:100]}")
                            print(f"  Last 100 chars: {seq[-100:]}")
                            print(f"=== END WARNING ===")
                        proteins_by_go[go_id][pid] = seq

        # Check if we have enough proteins for this GO term
        if len(proteins_by_go[go_id]) < minimum_seqs:
            if verbose:
                print(
                    f"GO term {go_id} skipped: Only {len(proteins_by_go[go_id])} proteins, minimum {minimum_seqs} required"
                )
            del proteins_by_go[go_id]
        elif len(proteins_by_go[go_id]) > maximum_seqs:
            # Sample down to maximum number of proteins
            import random

            protein_items = list(proteins_by_go[go_id].items())
            sampled_items = random.sample(protein_items, maximum_seqs)
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


def load_esm_model(model_name, verbose=False):
    """Load an ESM-C model using the correct ESM 3.2.1 API."""

    try:
        from esm.models.esmc import ESMC

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


def get_embeddings_for_term(model, sequences_dict, go_id, verbose=False):
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
            # DIAGNOSTIC: Check what we actually got from the dataset
            if len(sequence) > 10000:
                print(f"\n=== DIAGNOSTIC INFO for {protein_id} ===")
                print(f"Sequence type: {type(sequence)}")
                print(f"Sequence length: {len(sequence)}")
                print(f"First 100 chars: {str(sequence)[:100]}")
                print(f"Last 100 chars: {str(sequence)[-100:]}")
                print(f"Is string: {isinstance(sequence, str)}")
                if hasattr(sequence, "__dict__"):
                    print(f"Object attributes: {sequence.__dict__}")
                print(f"=== END DIAGNOSTIC ===")
                raise ValueError(
                    f"Sequence too long: {len(sequence)} amino acids"
                )

            if len(sequence) == 0:
                print(f"ERROR: Empty sequence for {protein_id}")
                raise ValueError(f"Empty sequence")

            if not isinstance(sequence, str):
                print(f"ERROR: Sequence is not a string for {protein_id}")
                print(f"  Type: {type(sequence)}")
                print(f"  Value: {sequence}")
                raise ValueError(f"Sequence is not a string")

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

                # Extract embeddings and mean pool over sequence length
                # Shape: [batch=1, seq_len, embed_dim] -> [embed_dim]
                sequence_embeddings = output.embeddings.squeeze(
                    0
                ).to(  # Remove batch dim
                    dtype=torch.float32
                )  # Convert from bfloat16 before numpy conversion
                protein_embedding = (
                    sequence_embeddings.mean(dim=0).cpu().numpy()
                )  # Mean pool

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

    for go_id, proteins in tqdm(
        proteins_by_go.items(), desc="Creating embeddings for GO term"
    ):
        # Get embeddings for all proteins for this GO term
        embeddings = get_embeddings_for_term(
            model, proteins, go_id, args.verbose
        )

        go_path = output_dir / f"{clean_go_id(go_id)}.pkl"
        with open(go_path, "wb") as f:
            pickle.dump(embeddings, f)

        saved_terms += 1

        if args.verbose:
            print(f"Saved embeddings for {go_id} to {go_path}")

    print(f"Saved embeddings for {saved_terms} GO terms in {output_dir}")


if __name__ == "__main__":
    main()
