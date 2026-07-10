"""Shared ESM-C pooling utilities.

This module is the single source of truth for turning ESM-C's per-residue
output into one fixed-length per-protein vector. Every script that generates
ESM-C embeddings — the training-set generator (``make_esm_embeddings.py``) and
the inference-time generators (``run_knn_inference_embeds.py``,
``run_multilabel_inference_embeds.py``) — must call :func:`embed_sequence` so
that the pooling is byte-for-byte identical across them.

Why that matters: a classifier is trained on vectors produced by one pooling
recipe and then, at prediction time, is fed vectors produced by another script.
If the two recipes disagree (different pooling strategy, layer, or special-token
handling), the query vectors live in a different feature space than the training
vectors and predictions are silently garbage. Centralising the recipe here makes
that class of bug impossible.

Pooling strategies
------------------
``mean``
    Per-dimension average across residues -> vector of length D. The field
    standard for ESM-family embeddings and the strongest, most defensible
    default for diverse-protein tasks such as GO-term function prediction.
``mean_max_std``
    Concatenation of per-dimension mean, max and (population) std across
    residues -> vector of length 3*D. Captures peak and dispersion signal that
    the mean discards. Note that ``max`` is sensitive to outlier positions, so
    it should only ever be used with ``strip_special_tokens=True`` (otherwise
    the max can simply report the BOS/EOS activation).

Special tokens
--------------
ESM-C's ``encode`` tokenises with ``add_special_tokens=True``, so the model
output has a prepended BOS and an appended EOS position: shape ``[L+2, D]`` for
an ``L``-residue protein. Pooling over those two positions contaminates the
representation — negligibly for ``mean`` on long proteins, but substantially for
``max`` (special tokens often carry extreme activations) and for short
sequences. :func:`embed_sequence` therefore strips them by default.

Layer selection
---------------
``model.forward`` returns ``hidden_states`` with shape ``[n_layers, B, L, D]`` —
every transformer layer, computed on every call. Passing ``layer=None`` (the
default) uses ``output.embeddings`` (the final representation, matching historical
behaviour); passing an integer selects ``output.hidden_states[layer]`` so that
mid-layer representations — which often beat the final layer for function
prediction — can be pooled without any extra forward compute.
"""

from __future__ import annotations

POOLING_STRATEGIES = ("mean", "mean_max_std")


def pool_embeddings(sequence_embeddings, strategy):
    """Reduce per-residue embeddings of shape (L, D) to a single vector.

    Parameters
    ----------
    sequence_embeddings : torch.Tensor
        Per-residue representations, shape ``(L, D)``. Special tokens should
        already have been removed by the caller (see :func:`embed_sequence`).
    strategy : str
        One of :data:`POOLING_STRATEGIES`.

    Returns
    -------
    torch.Tensor
        ``mean`` -> shape ``(D,)``; ``mean_max_std`` -> shape ``(3*D,)``
        (mean, then max, then std, concatenated in that order).

    Notes
    -----
    ``std`` uses ``unbiased=False`` (population std) so the result is defined
    (zero) when ``L == 1`` rather than NaN from a 1/(n-1) division.
    """
    import torch

    if strategy == "mean":
        return sequence_embeddings.mean(dim=0)
    if strategy == "mean_max_std":
        mean = sequence_embeddings.mean(dim=0)
        max_ = sequence_embeddings.max(dim=0).values
        std = sequence_embeddings.std(dim=0, unbiased=False)
        return torch.cat([mean, max_, std], dim=0)
    raise ValueError(
        f"Unknown pooling strategy: {strategy!r} "
        f"(expected one of {POOLING_STRATEGIES})"
    )


def select_layer_representation(output, layer):
    """Return the ``[B, L, D]`` representation for the requested layer.

    ``layer=None`` returns ``output.embeddings`` (the final representation,
    matching historical behaviour). An integer indexes ``output.hidden_states``
    (shape ``[n_layers, B, L, D]``) and supports negative indices, so ``-1`` is
    the last hidden layer and ``0`` the first.
    """
    if layer is None:
        if output.embeddings is None:
            raise ValueError("Model output has no `embeddings` field to pool.")
        return output.embeddings

    hidden = getattr(output, "hidden_states", None)
    if hidden is None:
        raise ValueError(
            "Layer selection requested but the model output has no "
            "`hidden_states`. This ESM-C build does not expose per-layer "
            "representations; omit --layer to pool the final embeddings."
        )
    n_layers = hidden.shape[0]
    idx = layer if layer >= 0 else n_layers + layer
    if not (0 <= idx < n_layers):
        raise ValueError(
            f"Requested layer {layer} is out of range for a model with "
            f"{n_layers} hidden layers (valid: -{n_layers}..{n_layers - 1})."
        )
    return hidden[idx]


def reduce_forward_output(
    output,
    *,
    pooling="mean",
    layer=None,
    strip_special_tokens=True,
):
    """Reduce a model ``forward()`` output to a pooled 1-D numpy vector.

    This is the layer-selection -> squeeze -> strip-special-tokens -> pool
    pipeline, split out from :func:`embed_sequence` so it can be unit-tested
    with a plain tensor container, without a live ESM-C model (which needs a
    GPU and downloaded weights).

    Parameters
    ----------
    output : object
        Anything with an ``embeddings`` attribute (shape ``[B, L(+2), D]``) and,
        for layer selection, a ``hidden_states`` attribute
        (shape ``[n_layers, B, L(+2), D]``) — e.g. an ESM-C ``ESMCOutput``.
    pooling, layer, strip_special_tokens
        See :func:`embed_sequence`.

    Returns
    -------
    numpy.ndarray
        The pooled per-protein embedding (float32).
    """
    import torch

    # [B=1, L(+2), D] for the chosen layer -> drop batch dim -> [L(+2), D].
    # Cast from bfloat16 before any numpy conversion downstream.
    reps = select_layer_representation(output, layer)
    sequence_embeddings = reps.squeeze(0).to(dtype=torch.float32)

    if strip_special_tokens:
        # Positions 0 and -1 are BOS/EOS (encode adds them). Guard against a
        # sequence too short to have any residues left after stripping.
        if sequence_embeddings.shape[0] <= 2:
            raise ValueError(
                "Sequence is too short to strip special tokens "
                f"(only {sequence_embeddings.shape[0]} token positions). "
                "Pass strip_special_tokens=False for such inputs."
            )
        sequence_embeddings = sequence_embeddings[1:-1]

    pooled = pool_embeddings(sequence_embeddings, pooling)
    return pooled.cpu().numpy()


def embed_sequence(
    model,
    sequence,
    *,
    pooling="mean",
    layer=None,
    strip_special_tokens=True,
):
    """Encode one protein sequence with ESM-C and return a pooled vector.

    This is the full, shared encode -> forward -> :func:`reduce_forward_output`
    pipeline. All embedding-generation scripts call this so training and
    inference use an identical recipe.

    Parameters
    ----------
    model : ESMC
        A loaded ESM-C model (``esm.models.esmc.ESMC``).
    sequence : str
        The amino-acid sequence.
    pooling : str
        Pooling strategy; see :data:`POOLING_STRATEGIES`.
    layer : int or None
        ``None`` pools the final ``embeddings``; an int pools that hidden layer.
    strip_special_tokens : bool
        If True (default), drop the BOS (first) and EOS (last) positions before
        pooling. Strongly recommended, and effectively required for
        ``mean_max_std`` because ``max`` is otherwise dominated by special
        tokens.

    Returns
    -------
    numpy.ndarray
        The pooled per-protein embedding (float32).
    """
    import torch
    from esm.sdk.api import ESMProtein

    protein = ESMProtein(sequence=sequence)
    with torch.no_grad():
        protein_tensor = model.encode(protein)
        output = model.forward(
            sequence_tokens=protein_tensor.sequence.unsqueeze(0)
        )
        return reduce_forward_output(
            output,
            pooling=pooling,
            layer=layer,
            strip_special_tokens=strip_special_tokens,
        )
