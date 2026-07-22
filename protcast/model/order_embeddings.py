"""order_embeddings.py

Order embedding layer and model builder for hierarchy-aware GO term prediction.

Each GO term is represented as a single point in the non-negative orthant of
R^d.  The GO DAG hierarchy is encoded by the *reversed product order*: a term
``a`` entails a term ``b`` (``a`` is more specific than ``b``) when ``a``
dominates ``b`` coordinate-wise, i.e. ``a_k >= b_k`` for all k.  General terms
sit near the origin; specific terms are pushed outward.

A protein embedding ``p`` is also projected into the orthant and must
*dominate* every GO term it is annotated with.  Membership is scored from the
per-dimension order violation:

    E(p entails t)  = || max(0, t - p) ||^2          (zero iff p dominates t)
    score           = exp(-temperature * E)          (in (0, 1], drop-in for BCE)

Unlike box embeddings, the partial order *is* the scoring function — there is
no box volume to estimate and no separate containment regularizer competing
with the binary-crossentropy objective.  The order relation is antisymmetric
and transitive by construction, which is what makes it a natural fit for a
directed acyclic graph like the Gene Ontology and gives ancestral closure
("one true path") for free.

The DAG hierarchy is enforced with an order-violation loss that, for each
(parent, child) edge, penalizes the child for *not* dominating the parent —
i.e. for being less specific than its parent.

To keep gradients alive everywhere (the failure mode of hard-ReLU box
containment, where a satisfied constraint has exactly zero gradient), the
default "soft" variant replaces ``max(0, x)`` with ``softplus(beta * x) / beta``.

References
----------
- Vendrov et al., "Order-Embeddings of Images and Language" (ICLR 2016)
- Lai & Hockenmaier, "Learning to Predict Denotational Probabilities for
  Modeling Entailment" (EACL 2017) — the probabilistic / soft-violation form
- Li et al., "Smoothing the Geometry of Probabilistic Box Embeddings"
  (ICLR 2019) — the softplus gradient-flow fix, applied here to order violations
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf
from keras import layers


@keras.utils.register_keras_serializable(package="ProtCast")
class OrderEmbeddingLayer(layers.Layer):
    """Represents GO terms as points in the non-negative orthant and scores
    protein membership via the order-violation energy.

    Each GO term j has a single learned vector ``raw_t_j`` in R^d; the term's
    position in the orthant is ``t_j = softplus(raw_t_j)`` (always positive,
    but unconstrained for the optimizer).

    For a protein point ``p`` (also mapped into the orthant), the membership
    score for term j is:

        E_j   = sum_k [ max(0, t_jk - p_k) ]^2
        score_j = exp(-temperature * E_j)

    The energy is zero (score 1) exactly when the protein dominates the term
    on every dimension — i.e. the protein "entails" the term.

    Parameters
    ----------
    num_classes : int
        Number of GO terms.
    order_dim : int
        Dimensionality of the order embedding space.
    temperature : float
        Scaling factor inside the exp (higher = sharper boundaries).  Plays
        the same role as ``temperature`` did for the box layer.
    """

    def __init__(self, num_classes, order_dim, temperature=10.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.order_dim = order_dim
        self.temperature = temperature

    def build(self, input_shape):
        # One vector per GO term (vs center + offset for boxes — half the params)
        self.raw_terms = self.add_weight(
            name="raw_terms",
            shape=(self.num_classes, self.order_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, protein_embedding):
        """Compute membership scores for each GO term.

        Parameters
        ----------
        protein_embedding : tf.Tensor
            Shape (batch_size, order_dim) — protein projected into order space.
            Expected non-negative (the projection head uses a softplus/relu).

        Returns
        -------
        tf.Tensor
            Shape (batch_size, num_classes) — membership scores in (0, 1].
        """
        # Term positions in the non-negative orthant
        terms = tf.nn.softplus(self.raw_terms)  # (num_classes, order_dim)

        # Broadcast: protein (batch, 1, order_dim), terms (1, num_classes, order_dim)
        p = tf.expand_dims(protein_embedding, axis=1)
        t = tf.expand_dims(terms, axis=0)

        # Order violation per dimension: positive where the term is NOT
        # dominated by the protein (t_k > p_k).  Squared, summed over dims.
        violation = tf.nn.relu(t - p)  # (batch, num_classes, order_dim)
        energy = tf.reduce_sum(tf.square(violation), axis=-1)  # (batch, num_classes)

        # Map energy -> (0, 1]: 1 when fully entailed (energy 0), -> 0 otherwise
        scores = tf.exp(-self.temperature * energy)

        return scores

    def get_term_positions(self):
        """Return the term points in the orthant for visualization/analysis.

        Returns
        -------
        np.ndarray
            Shape (num_classes, order_dim).
        """
        return tf.nn.softplus(self.raw_terms).numpy()

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "order_dim": self.order_dim,
            "temperature": self.temperature,
        })
        return config


def order_violation_loss(order_layer, parent_child_indices):
    """Hard order-violation loss for the GO DAG hierarchy.

    For each (parent, child) edge, the child term should be *more specific*
    than the parent, which in the reversed product order means the child
    must **dominate** the parent coordinate-wise: ``child_k >= parent_k``.

    This penalizes the violation — dimensions where the child fails to
    dominate the parent:

        penalty = mean over edges of:
            sum_k [ max(0, parent_k - child_k) ]^2

    Hard variant: uses ReLU, so once the child already dominates the parent
    on a dimension the gradient there is exactly zero — analogous to the box
    containment flat-gradient problem.  Prefer ``order_violation_loss_soft``.

    Parameters
    ----------
    order_layer : OrderEmbeddingLayer
        The layer containing the term parameters.
    parent_child_indices : tf.Tensor
        Shape (num_edges, 2) of integer index pairs [parent_idx, child_idx].
        Same format produced by ``extract_dag_edges`` for the box model.

    Returns
    -------
    tf.Tensor
        Scalar loss value.
    """
    if parent_child_indices is None or tf.shape(parent_child_indices)[0] == 0:
        return tf.constant(0.0)

    terms = tf.nn.softplus(order_layer.raw_terms)

    parent_idx = parent_child_indices[:, 0]
    child_idx = parent_child_indices[:, 1]

    parent = tf.gather(terms, parent_idx)
    child = tf.gather(terms, child_idx)

    # Child must dominate parent: penalize where parent_k > child_k
    violation = tf.nn.relu(parent - child)
    per_edge = tf.reduce_sum(tf.square(violation), axis=-1)
    return tf.reduce_mean(per_edge)


def order_violation_loss_soft(order_layer, parent_child_indices, beta=5.0):
    """Soft order-violation loss using softplus instead of ReLU.

    The hard ReLU violation has zero gradient once the child already
    dominates the parent on a dimension, so the optimizer gets no signal to
    push the child *further* into specificity.  Replacing ``max(0, x)`` with
    ``softplus(beta * x) / beta`` gives a non-zero gradient everywhere:
    asymptotically ReLU for large positive violations, smoothly decaying for
    negative ones, and non-zero at the boundary.

    This is the probabilistic-order / smoothed-geometry fix (Lai &
    Hockenmaier 2017; Li et al. 2019) and is the reason an order model is
    expected to train where the hard-containment box model stalled.

    Parameters
    ----------
    order_layer : OrderEmbeddingLayer
        The layer containing the term parameters.
    parent_child_indices : tf.Tensor
        Shape (num_edges, 2) of integer index pairs [parent_idx, child_idx].
    beta : float
        Smoothness parameter.  Higher beta -> closer to hard ReLU; lower
        beta -> smoother but more leakage at non-violating points.

    Returns
    -------
    tf.Tensor
        Scalar loss value.
    """
    if parent_child_indices is None or tf.shape(parent_child_indices)[0] == 0:
        return tf.constant(0.0)

    terms = tf.nn.softplus(order_layer.raw_terms)

    parent_idx = parent_child_indices[:, 0]
    child_idx = parent_child_indices[:, 1]

    parent = tf.gather(terms, parent_idx)
    child = tf.gather(terms, child_idx)

    # Softplus-smoothed violation — non-zero gradient everywhere
    violation = tf.math.softplus(beta * (parent - child)) / beta
    per_edge = tf.reduce_sum(tf.square(violation), axis=-1)
    return tf.reduce_mean(per_edge)


def build_order_embedding_model_dual(
    esm_dim,
    fv_dim,
    num_classes,
    hidden_layers,
    dropout_rate,
    order_dim,
    temperature=10.0,
    fv_hidden=32,
):
    """Build a dual-encoder order embedding model.

    Gives the small PseKRAAC feature block its own MLP branch before merging
    with the ESM-C branch.  This prevents the 12 PseKRAAC dimensions from
    being drowned out by the 1152 ESM-C dimensions in the first weight matrix,
    which happens when both blocks are naively concatenated and fed into a
    single dense layer.

    Architecture:
        ESM-C (esm_dim)    → [Dense+Dropout]* ─────────────┐
                                                             ├─ Concatenate
        PseKRAAC (fv_dim)  → Dense(fv_hidden, relu) ────────┘
                                                → Dense(order_dim, softplus)
                                                → OrderEmbeddingLayer → scores

    Parameters
    ----------
    esm_dim : int
        Dimension of ESM-C input (e.g. 1152).
    fv_dim : int
        Dimension of PseKRAAC feature vector (e.g. 12).
    num_classes : int
        Number of GO terms.
    hidden_layers : list of int
        Units in each hidden Dense layer applied to the ESM-C branch.
    dropout_rate : float
        Dropout rate after each ESM-C hidden layer.
    order_dim : int
        Dimensionality of the order embedding space.
    temperature : float
        Temperature for the membership score exponential.
    fv_hidden : int
        Units in the PseKRAAC encoder Dense layer (default 32).

    Returns
    -------
    tuple of (keras.Model, OrderEmbeddingLayer)
        The model and order layer (needed for the order-violation loss).
        The model accepts a list of two inputs: [esm_tensor, fv_tensor].
    """
    esm_input = layers.Input(shape=(esm_dim,), name="esm_input")
    fv_input  = layers.Input(shape=(fv_dim,),  name="fv_input")

    # ESM-C branch — same hidden stack as the single-encoder model
    x = esm_input
    for units in hidden_layers:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

    # PseKRAAC branch — dedicated small MLP so these 12 dims get equal footing
    fv = layers.Dense(fv_hidden, activation="relu", name="fv_encoder")(fv_input)

    # Merge both branches then project into the non-negative orthant
    merged = layers.Concatenate(name="feature_merge")([x, fv])
    merged = layers.Dense(
        order_dim, activation="softplus", name="order_projection"
    )(merged)

    order_layer = OrderEmbeddingLayer(
        num_classes=num_classes,
        order_dim=order_dim,
        temperature=temperature,
        name="order_embeddings",
    )
    scores = order_layer(merged)

    model = keras.Model(
        inputs=[esm_input, fv_input],
        outputs=scores,
        name="order_embedding_model_dual",
    )
    return model, order_layer


def build_order_embedding_model(
    input_dim,
    num_classes,
    hidden_layers,
    dropout_rate,
    order_dim,
    temperature=10.0,
):
    """Build a Keras Functional model with an order embedding output.

    Architecture:
        Input(input_dim) -> [Dense+Dropout]* -> Dense(order_dim, softplus)
                         -> OrderEmbeddingLayer -> scores

    The hidden layers process the ESM embedding, a projection layer maps to
    ``order_dim`` and (via softplus) into the non-negative orthant so the
    protein point lives in the same space as the term points, and the
    OrderEmbeddingLayer computes membership scores.

    Parameters
    ----------
    input_dim : int
        Dimension of input protein embeddings.
    num_classes : int
        Number of GO terms.
    hidden_layers : list of int
        Units in each hidden Dense layer.
    dropout_rate : float
        Dropout rate after each hidden layer.
    order_dim : int
        Dimensionality of the order embedding space.
    temperature : float
        Temperature for the membership score exponential.

    Returns
    -------
    tuple of (keras.Model, OrderEmbeddingLayer)
        The model and order layer (needed for the order-violation loss).
    """
    inputs = layers.Input(shape=(input_dim,))
    x = inputs

    # Hidden layers (same as flat model)
    for units in hidden_layers:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

    # Project into the non-negative orthant so the protein point can be
    # compared to the (softplus-positive) term points under the product order.
    x = layers.Dense(order_dim, activation="softplus", name="order_projection")(x)

    order_layer = OrderEmbeddingLayer(
        num_classes=num_classes,
        order_dim=order_dim,
        temperature=temperature,
        name="order_embeddings",
    )
    scores = order_layer(x)

    model = keras.Model(inputs=inputs, outputs=scores, name="order_embedding_model")

    return model, order_layer
