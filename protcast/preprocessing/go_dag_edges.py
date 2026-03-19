"""go_dag_edges.py

Extract parent-child edge pairs from the GO DAG for box containment loss.

Given an AnnotatedGODag and a set of GO IDs used in the model, produces
integer index pairs suitable for the containment regularization loss
in box_embeddings.py.
"""

from __future__ import annotations

import numpy as np


def extract_dag_edges(go_dag, go_ids, go_encoder):
    """Extract parent-child index pairs from the GO DAG.

    For each GO term in go_ids, looks up its parents and children in the
    DAG. Only edges where BOTH parent and child are in go_ids (i.e., both
    are being predicted by the model) are included.

    Parameters
    ----------
    go_dag : AnnotatedGODag
        The GO DAG loaded from an OBO file.
    go_ids : list of str
        GO term IDs used by the model (e.g., ["GO:0003674", "GO:0005575"]).
    go_encoder : GOEncoder
        Encoder mapping GO IDs to integer indices.

    Returns
    -------
    np.ndarray
        Shape (num_edges, 2) of [parent_idx, child_idx] integer pairs.
        Empty array with shape (0, 2) if no edges are found.
    """
    go_id_set = set(go_ids)
    edges = []

    for go_id in go_ids:
        if go_id not in go_dag.go_terms_map:
            continue

        term = go_dag.go_terms_map[go_id]

        # Add edges where this term is the child
        for parent_id in term.parents:
            if parent_id in go_id_set:
                parent_idx = go_encoder.go_to_int[parent_id]
                child_idx = go_encoder.go_to_int[go_id]
                edges.append([parent_idx, child_idx])

    if not edges:
        return np.zeros((0, 2), dtype=np.int32)

    return np.array(edges, dtype=np.int32)
