"""box_embeddings.py

Box embedding layer and model builder for hierarchy-aware GO term prediction.

Each GO term is represented as an axis-aligned hyperrectangle (box) in
d-dimensional space, parameterized by a center and a positive offset.
A protein embedding's membership score for a GO term is computed from
the signed distance to the box boundaries using a smooth approximation.

The GO DAG hierarchy is enforced via a containment regularization loss
that penalizes child boxes that extend outside their parent boxes.

References
----------
- Vilnis et al., "Probabilistic Embedding of Knowledge Graphs with Box
  Lattice Measures" (ACL 2018)
- Li et al., "Smoothing the Geometry of Probabilistic Box Embeddings"
  (ICLR 2019)
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf
from keras import layers


@keras.utils.register_keras_serializable(package="ProtCast")
class BoxEmbeddingLayer(layers.Layer):
    """Represents GO terms as boxes and scores protein membership.

    Each GO term j has:
      - center_j: center of the box in R^d
      - log_offset_j: log of the half-widths (softplus → positive offset)

    The box for term j spans:
      [center_j - offset_j, center_j + offset_j]

    For a protein point p, membership score for term j is:
      score_j = sigmoid(temperature * smooth_min_k(signed_distance_jk))

    where signed_distance_jk = offset_jk - |p_k - center_jk| and
    smooth_min approximates the minimum across dimensions.

    Parameters
    ----------
    num_classes : int
        Number of GO terms (boxes).
    box_dim : int
        Dimensionality of box space.
    temperature : float
        Scaling factor for sigmoid (higher = sharper boundaries).
    """

    def __init__(self, num_classes, box_dim, temperature=10.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.box_dim = box_dim
        self.temperature = temperature

    def build(self, input_shape):
        self.centers = self.add_weight(
            name="centers",
            shape=(self.num_classes, self.box_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.log_offsets = self.add_weight(
            name="log_offsets",
            shape=(self.num_classes, self.box_dim),
            initializer=keras.initializers.Constant(0.0),
            trainable=True,
        )

    def call(self, protein_embedding):
        """Compute membership scores for each GO term.

        Parameters
        ----------
        protein_embedding : tf.Tensor
            Shape (batch_size, box_dim) — protein projected into box space.

        Returns
        -------
        tf.Tensor
            Shape (batch_size, num_classes) — sigmoid membership scores.
        """
        # offsets are always positive via softplus
        offsets = tf.nn.softplus(self.log_offsets)  # (num_classes, box_dim)

        # Expand dims for broadcasting:
        # protein: (batch, 1, box_dim), centers: (1, num_classes, box_dim)
        p = tf.expand_dims(protein_embedding, axis=1)
        c = tf.expand_dims(self.centers, axis=0)
        o = tf.expand_dims(offsets, axis=0)

        # Signed distance: positive inside box, negative outside
        # signed_dist_k = offset_k - |p_k - center_k|
        signed_dist = o - tf.abs(p - c)  # (batch, num_classes, box_dim)

        # Smooth minimum across dimensions (LogSumExp approximation)
        # min ≈ -1/temp * log(sum(exp(-temp * x)))
        # We use a moderate softness to keep gradients flowing
        softness = 5.0
        smooth_min = -tf.reduce_logsumexp(
            -softness * signed_dist, axis=-1
        ) / softness  # (batch, num_classes)

        # Membership score via sigmoid
        scores = tf.sigmoid(self.temperature * smooth_min)

        return scores

    def get_box_bounds(self):
        """Return box min/max corners for visualization or analysis.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            (box_min, box_max) each of shape (num_classes, box_dim).
        """
        offsets = tf.nn.softplus(self.log_offsets).numpy()
        centers = self.centers.numpy()
        return centers - offsets, centers + offsets

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "box_dim": self.box_dim,
            "temperature": self.temperature,
        })
        return config


def containment_loss(box_layer, parent_child_indices):
    """Compute containment regularization loss for GO DAG hierarchy.

    For each (parent, child) pair, penalizes the child box for extending
    beyond the parent box boundaries. This enforces the ontology constraint
    that child GO terms are more specific than parents — so the child box
    should be contained within the parent box.

    penalty = mean over pairs of:
        sum_k [ max(0, parent_min_k - child_min_k)^2
              + max(0, child_max_k - parent_max_k)^2 ]

    Parameters
    ----------
    box_layer : BoxEmbeddingLayer
        The layer containing box parameters.
    parent_child_indices : tf.Tensor
        Shape (num_edges, 2) of integer index pairs [parent_idx, child_idx].

    Returns
    -------
    tf.Tensor
        Scalar loss value.
    """
    if parent_child_indices is None or tf.shape(parent_child_indices)[0] == 0:
        return tf.constant(0.0)

    offsets = tf.nn.softplus(box_layer.log_offsets)

    # Gather parent and child parameters
    parent_idx = parent_child_indices[:, 0]
    child_idx = parent_child_indices[:, 1]

    parent_centers = tf.gather(box_layer.centers, parent_idx)
    parent_offsets = tf.gather(offsets, parent_idx)
    child_centers = tf.gather(box_layer.centers, child_idx)
    child_offsets = tf.gather(offsets, child_idx)

    # Box bounds
    parent_min = parent_centers - parent_offsets
    parent_max = parent_centers + parent_offsets
    child_min = child_centers - child_offsets
    child_max = child_centers + child_offsets

    # Violation: child extends below parent's min or above parent's max
    lower_violation = tf.nn.relu(parent_min - child_min)
    upper_violation = tf.nn.relu(child_max - parent_max)

    # Sum violations across dimensions, mean across edges
    per_edge = tf.reduce_sum(
        tf.square(lower_violation) + tf.square(upper_violation), axis=-1
    )
    return tf.reduce_mean(per_edge)


def build_box_embedding_model(
    input_dim,
    num_classes,
    hidden_layers,
    dropout_rate,
    box_dim,
    temperature=10.0,
):
    """Build a Keras Functional model with box embedding output.

    Architecture:
        Input(input_dim) → [Dense+Dropout]* → Dense(box_dim) → BoxEmbeddingLayer → scores

    The hidden layers process the ESM embedding, then a projection layer
    maps to box_dim, and the BoxEmbeddingLayer computes membership scores.

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
    box_dim : int
        Dimensionality of box space.
    temperature : float
        Temperature for box membership sigmoid.

    Returns
    -------
    tuple of (keras.Model, BoxEmbeddingLayer)
        The model and box layer (needed for containment loss computation).
    """
    inputs = layers.Input(shape=(input_dim,))
    x = inputs

    # Hidden layers (same as flat model)
    for units in hidden_layers:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

    # Project to box space
    x = layers.Dense(box_dim, activation="relu", name="box_projection")(x)

    # Box membership scores
    box_layer = BoxEmbeddingLayer(
        num_classes=num_classes,
        box_dim=box_dim,
        temperature=temperature,
        name="box_embeddings",
    )
    scores = box_layer(x)

    model = keras.Model(inputs=inputs, outputs=scores, name="box_embedding_model")

    return model, box_layer
