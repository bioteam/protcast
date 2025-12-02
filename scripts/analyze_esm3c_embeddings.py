"""
Analyze ESM3-C Embeddings

This script provides utilities to analyze and visualize ESM3-C embeddings generated 
with the protcast_esm3c_integration.py script.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze ESM3-C embeddings')
    parser.add_argument('-i', '--input_file', required=True, help='Path to embeddings pickle file')
    parser.add_argument('-o', '--output_dir', default='embedding_analysis', help='Output directory for visualizations')
    parser.add_argument('--visualization', choices=['tsne', 'pca'], default='tsne',
                        help='Visualization method for embeddings')
    parser.add_argument('--n_components', type=int, default=2, help='Number of components for dimensionality reduction')
    parser.add_argument('--perplexity', type=int, default=30, help='Perplexity for t-SNE')
    parser.add_argument('--sample_size', type=int, default=1000, help='Maximum number of proteins to visualize per class')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    return parser.parse_args()


def load_embeddings(input_file):
    """Load embeddings from a pickle file."""
    with open(input_file, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def reduce_dimensionality(embeddings_dict, method='tsne', n_components=2, perplexity=30, verbose=False):
    """
    Reduce dimensionality of embeddings for visualization.
    
    Parameters:
    -----------
    embeddings_dict : dict
        Dictionary where keys are GO IDs and values are dicts mapping protein IDs to embeddings
    method : str
        Dimensionality reduction method ('tsne' or 'pca')
    n_components : int
        Number of dimensions in the output
    perplexity : int
        Perplexity parameter for t-SNE
    verbose : bool
        Whether to print verbose output
    
    Returns:
    --------
    reduced_data : dict
        Dictionary with reduced embeddings and metadata
    """
    # Collect all embeddings and their labels
    all_embeddings = []
    all_labels = []
    go_id_to_idx = {}
    protein_ids = []
    
    for go_idx, (go_id, proteins) in enumerate(embeddings_dict.items()):
        go_id_to_idx[go_id] = go_idx
        for protein_id, embedding in proteins.items():
            all_embeddings.append(embedding)
            all_labels.append(go_idx)
            protein_ids.append(protein_id)
    
    # Convert to numpy array
    X = np.array(all_embeddings)
    y = np.array(all_labels)
    
    if verbose:
        print(f"Reducing dimensionality of {X.shape[0]} embeddings from {X.shape[1]} dimensions to {n_components} dimensions")
    
    # Apply dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, verbose=verbose)
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    X_reduced = reducer.fit_transform(X)
    
    if verbose:
        if method == 'pca':
            explained_variance = reducer.explained_variance_ratio_
            print(f"Explained variance ratio: {explained_variance}")
            print(f"Total explained variance: {np.sum(explained_variance):.2f}")
    
    # Prepare results
    result = {
        'X_reduced': X_reduced,
        'labels': y,
        'go_id_to_idx': go_id_to_idx,
        'idx_to_go_id': {v: k for k, v in go_id_to_idx.items()},
        'protein_ids': protein_ids,
        'method': method,
        'n_components': n_components
    }
    
    return result


def visualize_embeddings(reduced_data, output_file, sample_size=1000, verbose=False):
    """
    Create a visualization of the reduced embeddings.
    
    Parameters:
    -----------
    reduced_data : dict
        Dictionary with reduced embeddings and metadata
    output_file : str
        Path to save the visualization
    sample_size : int
        Maximum number of proteins to visualize per class
    verbose : bool
        Whether to print verbose output
    """
    X_reduced = reduced_data['X_reduced']
    labels = reduced_data['labels']
    idx_to_go_id = reduced_data['idx_to_go_id']
    method = reduced_data['method']
    n_components = reduced_data['n_components']
    
    # Sample points if there are too many
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    if X_reduced.shape[0] > sample_size and n_classes > 0:
        if verbose:
            print(f"Sampling {sample_size} points from {X_reduced.shape[0]} total points for visualization")
        
        # Try to sample evenly from each class
        samples_per_class = max(1, sample_size // n_classes)
        sampled_indices = []
        
        for label in unique_labels:
            class_indices = np.where(labels == label)[0]
            n_samples = min(samples_per_class, len(class_indices))
            sampled_indices.extend(np.random.choice(class_indices, size=n_samples, replace=False))
        
        # Limit to sample_size
        if len(sampled_indices) > sample_size:
            sampled_indices = sampled_indices[:sample_size]
        
        X_reduced = X_reduced[sampled_indices]
        labels = labels[sampled_indices]
    
    # Set up the plot
    plt.figure(figsize=(12, 10))
    
    if n_components == 2:
        # 2D plot
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='tab10',
                             alpha=0.7, s=50, edgecolors='w', linewidths=0.5)
        
        # Add legend for class labels
        legend_elements = []
        for i, label in enumerate(unique_labels):
            go_id = idx_to_go_id[label]
            count = np.sum(labels == label)
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=scatter.cmap(scatter.norm(label)),
                                             markersize=10, label=f'{go_id} ({count} proteins)'))
        
        plt.legend(handles=legend_elements, title='GO Terms', 
                  loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.xlabel(f'Component 1')
        plt.ylabel(f'Component 2')
        
    elif n_components == 3:
        # 3D plot
        ax = plt.figure().add_subplot(111, projection='3d')
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                           c=labels, cmap='tab10', alpha=0.7, s=50, edgecolors='w', linewidths=0.5)
        
        # Add legend for class labels
        legend_elements = []
        for i, label in enumerate(unique_labels):
            go_id = idx_to_go_id[label]
            count = np.sum(labels == label)
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=scatter.cmap(scatter.norm(label)),
                                             markersize=10, label=f'{go_id} ({count} proteins)'))
        
        plt.legend(handles=legend_elements, title='GO Terms', 
                  loc='center left', bbox_to_anchor=(1, 0.5))
        
        ax.set_xlabel(f'Component 1')
        ax.set_ylabel(f'Component 2')
        ax.set_zlabel(f'Component 3')
    
    # Set title and save
    plt.title(f'ESM3-C Embeddings Visualization using {method.upper()}')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if verbose:
        print(f"Saved visualization to {output_file}")


def analyze_embedding_similarity(embeddings_dict, output_file, verbose=False):
    """
    Analyze similarity between GO terms based on their embeddings.
    
    Parameters:
    -----------
    embeddings_dict : dict
        Dictionary where keys are GO IDs and values are dicts mapping protein IDs to embeddings
    output_file : str
        Path to save the similarity matrix visualization
    verbose : bool
        Whether to print verbose output
    """
    go_ids = list(embeddings_dict.keys())
    n_go_terms = len(go_ids)
    
    # Compute mean embedding for each GO term
    mean_embeddings = {}
    for go_id, proteins in embeddings_dict.items():
        embeddings = np.array(list(proteins.values()))
        mean_embeddings[go_id] = np.mean(embeddings, axis=0)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.zeros((n_go_terms, n_go_terms))
    for i, go_id1 in enumerate(go_ids):
        for j, go_id2 in enumerate(go_ids):
            emb1 = mean_embeddings[go_id1]
            emb2 = mean_embeddings[go_id2]
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarity_matrix[i, j] = similarity
    
    # Visualize similarity matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, xticklabels=go_ids, yticklabels=go_ids, 
                annot=False, cmap='viridis', vmin=0, vmax=1)
    plt.title('GO Term Embedding Similarity Matrix (Cosine Similarity)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if verbose:
        print(f"Saved similarity matrix to {output_file}")
    
    return similarity_matrix, go_ids


def analyze_dimensionality(embeddings_dict, output_file, verbose=False):
    """
    Analyze the dimensionality of embeddings using PCA.
    
    Parameters:
    -----------
    embeddings_dict : dict
        Dictionary where keys are GO IDs and values are dicts mapping protein IDs to embeddings
    output_file : str
        Path to save the dimensionality analysis plot
    verbose : bool
        Whether to print verbose output
    """
    # Collect all embeddings
    all_embeddings = []
    for go_id, proteins in embeddings_dict.items():
        all_embeddings.extend(list(proteins.values()))
    
    # Convert to numpy array
    X = np.array(all_embeddings)
    
    if verbose:
        print(f"Analyzing dimensionality of {X.shape[0]} embeddings with {X.shape[1]} features")
    
    # Run PCA
    pca = PCA().fit(X)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    
    # Cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components needed for different variance thresholds
    thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
    components_needed = []
    for threshold in thresholds:
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        components_needed.append(n_components)
        if verbose:
            print(f"{threshold*100:.0f}% of variance explained by {n_components} components")
    
    # Plot individual explained variance
    plt.subplot(1, 2, 1)
    plt.plot(pca.explained_variance_ratio_[:50], 'o-', linewidth=2, markersize=8)
    plt.title('Explained Variance by Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    
    # Plot cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_variance[:100], 'o-', linewidth=2, markersize=8)
    
    # Add horizontal lines for thresholds
    for i, threshold in enumerate(thresholds):
        plt.axhline(y=threshold, color=f'C{i+1}', linestyle='--', alpha=0.7)
        plt.text(components_needed[i] + 5, threshold, f'{threshold*100:.0f}% ({components_needed[i]} components)', 
                 verticalalignment='center')
    
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    if verbose:
        print(f"Saved dimensionality analysis to {output_file}")
    
    return components_needed, thresholds


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load embeddings
    print(f"Loading embeddings from {args.input_file}")
    embeddings_dict = load_embeddings(args.input_file)
    
    # Check format of embeddings
    if not isinstance(embeddings_dict, dict):
        print("Error: Embeddings file has unexpected format. Expected a dictionary.")
        return
    
    # Count proteins and GO terms
    go_terms = len(embeddings_dict)
    protein_count = sum(len(proteins) for proteins in embeddings_dict.values())
    print(f"Loaded embeddings for {protein_count} proteins across {go_terms} GO terms")
    
    # Perform dimensionality reduction
    reduced_data = reduce_dimensionality(
        embeddings_dict,
        method=args.visualization,
        n_components=args.n_components,
        perplexity=args.perplexity,
        verbose=args.verbose
    )
    
    # Create visualizations
    viz_output = os.path.join(args.output_dir, f"{args.visualization}_visualization.png")
    visualize_embeddings(
        reduced_data,
        viz_output,
        sample_size=args.sample_size,
        verbose=args.verbose
    )
    
    # Analyze similarity between GO terms
    similarity_output = os.path.join(args.output_dir, "go_term_similarity.png")
    analyze_embedding_similarity(
        embeddings_dict,
        similarity_output,
        verbose=args.verbose
    )
    
    # Analyze dimensionality of embeddings
    dim_output = os.path.join(args.output_dir, "dimensionality_analysis.png")
    analyze_dimensionality(
        embeddings_dict,
        dim_output,
        verbose=args.verbose
    )
    
    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()