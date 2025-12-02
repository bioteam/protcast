import torch
import esm
import numpy as np
from pathlib import Path
import os

def get_esm3_embedding(sequence, model_type="esm3_c", layer_idx=None, avg_pooling=True):
    """
    Generate embeddings for a protein sequence using ESM-3 or ESM-C models.
    
    Parameters:
    -----------
    sequence : str
        The amino acid sequence to embed
    model_type : str, default="esm3_c"
        Type of model to use: "esm3_650m", "esm3_3b", "esm3_14b", or "esm3_c"
    layer_idx : int, optional
        Index of the layer to extract embeddings from.
        If None, uses the default representation layer for the model.
    avg_pooling : bool, default=True
        If True, returns the average of the token embeddings
        If False, returns per-residue embeddings for each amino acid
    
    Returns:
    --------
    embedding : numpy.ndarray
        If avg_pooling is True: a single vector of shape (embed_dim,)
        If avg_pooling is False: a matrix of shape (seq_len, embed_dim)
    """
    # Model selection
    model_mapping = {
        "esm3_c": "esm3_c_640m_combined",       # Contrastive model, 640M parameters
        "esm3_650m": "esm3_650m",               # Base ESM-3 model, 650M parameters
        "esm3_3b": "esm3_3b",                   # Larger ESM-3, 3B parameters
        "esm3_14b": "esm3_14b",                 # Largest ESM-3, 14B parameters
    }
    
    if model_type not in model_mapping:
        raise ValueError(f"Model type {model_type} not recognized. Choose from: {list(model_mapping.keys())}")
    
    model_name = model_mapping[model_type]
    print(f"Loading ESM model: {model_name}")
    
    # Check input
    if not sequence:
        raise ValueError("Empty sequence provided")
    
    try:
        # Load the model
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model.eval()  # Set to evaluation mode
        
        # Use GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Using device: {device}")
        
        # Prepare the sequence as a batch
        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
        batch_tokens = batch_tokens.to(device)
        
        print(f"Processing sequence of length {len(sequence)}")
        
        # Extract the embeddings
        with torch.no_grad():
            # If layer_idx is None, use all layers and the model's default behavior
            if layer_idx is None:
                # ESM-3 models provide a standard way to get representations
                results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
                token_embeddings = results["representations"][model.num_layers]
            else:
                results = model(batch_tokens, repr_layers=[layer_idx], return_contacts=False)
                token_embeddings = results["representations"][layer_idx]
        
        # Remove batch dimension and special tokens (excluding the start token <cls>)
        token_embeddings = token_embeddings[0, 1:len(sequence)+1, :]
        
        # Convert to numpy for easier handling
        embeddings = token_embeddings.cpu().numpy()
        
        if avg_pooling:
            # Average pooling over all amino acids to get a fixed-size vector
            embeddings = np.mean(embeddings, axis=0)
            print(f"Generated sequence embedding with shape: {embeddings.shape}")
        else:
            print(f"Generated per-residue embeddings with shape: {embeddings.shape}")
        
        return embeddings
        
    except Exception as e:
        # Handle the case where the model might not be found
        if "No such model" in str(e):
            print(f"Error: {e}")
            print("You may need to download the model manually or use an available model.")
            print("For ESM-3, you can download models from https://github.com/facebookresearch/esm")
            return None
        else:
            raise e


def download_esm_model(model_type="esm3_c"):
    """
    Attempt to download an ESM-3 model.
    
    Parameters:
    -----------
    model_type : str, default="esm3_c"
        Type of model to download: "esm3_650m", "esm3_3b", "esm3_14b", or "esm3_c"
    """
    model_mapping = {
        "esm3_c": "esm3_c_640m_combined", 
        "esm3_650m": "esm3_650m",
        "esm3_3b": "esm3_3b",
        "esm3_14b": "esm3_14b",
    }
    
    if model_type not in model_mapping:
        raise ValueError(f"Model type {model_type} not recognized. Choose from: {list(model_mapping.keys())}")
    
    model_name = model_mapping[model_type]
    
    try:
        # This will download the model automatically
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        print(f"Successfully downloaded model: {model_name}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


def save_embedding(embedding, output_file):
    """Save embedding as a numpy file."""
    np.save(output_file, embedding)
    print(f"Saved embedding to {output_file}")


def batch_embedding_esm3(sequences_dict, model_type="esm3_c"):
    """
    Generate embeddings for multiple sequences using ESM-3 or ESM-C models.
    
    Parameters:
    -----------
    sequences_dict : dict
        Dictionary mapping sequence IDs to amino acid sequences
    model_type : str, default="esm3_c"
        Type of model to use
    
    Returns:
    --------
    embeddings_dict : dict
        Dictionary mapping sequence IDs to embeddings
    """
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
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        batch_converter = alphabet.get_batch_converter()
        
        # Prepare the data in the format expected by the batch converter
        batch_data = [(seq_id, seq) for seq_id, seq in sequences_dict.items()]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        
        print(f"Processing batch of {len(sequences_dict)} sequences")
        
        embeddings_dict = {}
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
            token_embeddings = results["representations"][model.num_layers]
            
            for i, (seq_id, seq) in enumerate(sequences_dict.items()):
                # Extract embeddings for this sequence (removing special tokens)
                seq_embeddings = token_embeddings[i, 1:len(seq)+1, :].cpu().numpy()
                # Average pooling
                seq_embedding = np.mean(seq_embeddings, axis=0)
                embeddings_dict[seq_id] = seq_embedding
        
        print("Batch processing complete")
        return embeddings_dict
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return {}


def example_usage():
    # Example protein sequence (insulin)
    insulin_sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
    
    # Attempt to download/load the model first
    if download_esm_model(model_type="esm3_c"):
        # 1. Generate a per-protein embedding (average pooling) with ESM-3-C
        protein_embedding = get_esm3_embedding(
            insulin_sequence, 
            model_type="esm3_c",
            avg_pooling=True
        )
        
        if protein_embedding is not None:
            save_embedding(protein_embedding, "insulin_esm3c_embedding.npy")
            
            # 2. Generate per-residue embeddings
            residue_embeddings = get_esm3_embedding(
                insulin_sequence,
                model_type="esm3_c",
                avg_pooling=False
            )
            save_embedding(residue_embeddings, "insulin_esm3c_residue_embeddings.npy")
            
            # Show how to load and use the embeddings
            loaded_embedding = np.load("insulin_esm3c_embedding.npy")
            print(f"Loaded embedding shape: {loaded_embedding.shape}")
            print(f"First 5 values: {loaded_embedding[:5]}")
            
            # Demonstrate batch processing
            print("\nDemonstrating batch processing")
            test_sequences = {
                "protein1": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "protein2": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTR",
                "protein3": "MPIGSEDYLVNLVVDGKPIIIFEPGRVYEVIINDLTKPIMIIKESLGMCYSHDIKVIFK"
            }
            embeddings = batch_embedding_esm3(test_sequences, model_type="esm3_c")
            print(f"Generated embeddings for {len(embeddings)} proteins")
            for seq_id, embedding in embeddings.items():
                print(f"{seq_id}: embedding shape {embedding.shape}")
    else:
        print("Could not download/load the model. Please check your internet connection or model availability.")


def integrate_with_protcast(sequence_dict, model_type="esm3_c"):
    """
    Example of how to integrate ESM-3 embeddings with ProtCast.
    
    Parameters:
    -----------
    sequence_dict : dict
        Dictionary mapping protein IDs to sequences
    model_type : str
        ESM-3 model type to use
        
    Returns:
    --------
    embeddings_dict : dict
        Dictionary mapping protein IDs to ESM-3 embeddings
    """
    # Get embeddings for all sequences
    embeddings = batch_embedding_esm3(sequence_dict, model_type=model_type)
    
    # Example of how you might use these with ProtCast
    print(f"Generated {len(embeddings)} ESM-3 embeddings for ProtCast integration")
    
    # Save embeddings to a file for later use
    np.save(f"protcast_esm3_{model_type}_embeddings.npy", embeddings)
    
    # Return the embeddings dictionary for immediate use
    return embeddings


if __name__ == "__main__":
    print("Running ESM-3/ESM-C embedding example")
    example_usage()