"""
ProtCast ESM3 Integration

This script demonstrates how to integrate ESM embeddings with ProtCast for protein
function prediction.

ESM-3-C is a contrastive learning variant of the ESM-3 model from Meta AI Research.
These embeddings can be used to represent proteins in vector space, which can be helpful
for predicting protein functions, properties, or other downstream tasks.

Dimensionality of ESM Embeddings

For ESM-2 Models:

ESM-2 (t33, 650M parameters): 1280 dimensions

Other ESM-2 models also use embedding dimensions that match their model size.

For ESM-3 Models (based on Meta AI documentation):

ESM-3-C (640M parameters): 1280 dimensions
ESM-3 (650M parameters): 1280 dimensions
ESM-3 (3B parameters): 2560 dimensions
ESM-3 (14B parameters): 5120 dimensions

The embedding dimension increases with model size, allowing larger models to capture more complex protein representations.

Important Notes:
Per-residue vs. whole-protein embeddings:

For a protein with length L, the per-residue embeddings would have shape (L, D) where D is the embedding dimension
When using average pooling to get a whole-protein representation, you get a vector of size (D)

These embeddings are contextual, meaning each amino acid's representation is influenced by its surrounding sequence
"""

import torch
import esm
import numpy as np
import argparse
import os
from pathlib import Path
import pickle
import sys
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
        default="esm3c_embeddings",
        help="Directory to save embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing sequences",
    )
    parser.add_argument(
        "--model_type",
        default="esm3_c",
        choices=["esm3_c", "esm3_650m", "esm3_3b", "esm3_14b"],
        help="ESM3 model type to use",
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
                        proteins_by_go[go_id][pid] = dataset.proteins[
                            pid
                        ].sequence

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


def load_esm_model(model_type):
    """Load an ESM-3 or ESM-C model."""
    model_mapping = {
        "esm3_c": "esm3_c_640m_combined",
        "esm3_650m": "esm3_650m",
        "esm3_3b": "esm3_3b",
        "esm3_14b": "esm3_14b",
    }

    model_name = model_mapping[model_type]
    print(f"Loading ESM model: {model_name}")

    try:
        # Load the model
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model.eval()

        # Use GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)

        return model, alphabet, device
    except Exception as e:
        print(f"Error loading ESM model: {e}")
        print(
            "Please make sure you have installed the latest esm package and have internet access."
        )
        sys.exit(1)


def process_sequences_in_batches(
    model,
    alphabet,
    sequences_dict,
    batch_size,
    device,
    verbose=False,
):
    """
    Process protein sequences in batches to generate ESM embeddings.

    Parameters:
    -----------
    model : esm.model.ProteinBertModel
        ESM model to use for embeddings
    alphabet : esm.data.Alphabet
        Alphabet for tokenizing sequences
    sequences_dict : dict
        Dictionary mapping protein IDs to sequences
    batch_size : int
        Number of sequences to process at once
    device : torch.device
        Device to use for computation
    verbose : bool
        Whether to print verbose output

    Returns:
    --------
    embeddings_dict : dict
        Dictionary mapping protein IDs to ESM embeddings
    """
    # Filter out sequences that are too long
    #filtered_sequences = {
    #    pid: seq
    #    for pid, seq in sequences_dict.items()
    #    if len(seq) <= max_seq_length
    #}
    # if len(filtered_sequences) < len(sequences_dict):
    #     if verbose:
    #         print(
    #             f"Filtered out {len(sequences_dict) - len(filtered_sequences)} sequences longer than {max_seq_length}"
    #         )

    # Convert dictionary to list of tuples for batch processing
    sequences_list = list(sequences_dict.items())
    embeddings_dict = {}

    # Create batch converter
    batch_converter = alphabet.get_batch_converter()

    # Process in batches
    for i in tqdm(
        range(0, len(sequences_list), batch_size),
        desc="Processing batches",
        disable=not verbose,
    ):
        batch = sequences_list[i : i + batch_size]

        # Convert to format expected by the model
        batch_data = [(pid, seq) for pid, seq in batch]
        _, _, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        # Generate embeddings
        with torch.no_grad():
            results = model(
                batch_tokens,
                repr_layers=[model.num_layers],
                return_contacts=False,
            )
            token_embeddings = results["representations"][model.num_layers]

            # Process each sequence in the batch
            for j, (pid, seq) in enumerate(batch):
                # Get embeddings for this sequence (excluding special tokens)
                seq_embeddings = (
                    token_embeddings[j, 1 : len(seq) + 1, :].cpu().numpy()
                )
                # Average pooling to get a single vector per protein
                protein_embedding = np.mean(seq_embeddings, axis=0)
                embeddings_dict[pid] = protein_embedding
                if verbose:
                    print(
                        f"Processed {pid}: sequence length {len(seq)}, embedding shape {protein_embedding.shape}"
                    )

    return embeddings_dict


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

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
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"Found {len(proteins_by_go)} GO terms with sufficient proteins")

    if not proteins_by_go:
        if args.verbose:
            print("No GO terms with sufficient proteins found. Exiting.")
        return

    # Load ESM model
    model, alphabet, device = load_esm_model(args.model_type)

    # Process each GO term separately to manage memory
    all_embeddings = {}
    for go_id, proteins in tqdm(
        proteins_by_go.items(), desc="Processing GO terms"
    ):
        # Get embeddings for all proteins for this GO term
        embeddings = process_sequences_in_batches(
            model,
            alphabet,
            proteins,
            args.batch_size,
            device,
            verbose=args.verbose,
        )

        # Save embeddings for this GO term
        output_file = os.path.join(
            args.output_dir, f"{go_id.replace(':', '_')}_embeddings.pkl"
        )
        with open(output_file, "wb") as f:
            pickle.dump(embeddings, f)

        if args.verbose:
            print(
                f"Saved {len(embeddings)} embeddings for {go_id} to {output_file}"
            )

        # Add to complete dictionary
        all_embeddings[go_id] = embeddings

    # Save combined embeddings
    combined_output = os.path.join(
        args.output_dir, f"all_go_terms_{args.model_type}_embeddings.pkl"
    )
    with open(combined_output, "wb") as f:
        pickle.dump(all_embeddings, f)

    print(
        f"Saved combined embeddings for {len(all_embeddings)} GO terms to {combined_output}"
    )
    print("Done!")


if __name__ == "__main__":
    main()
