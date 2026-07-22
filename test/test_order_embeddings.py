"""Tests for order embeddings: OrderEmbeddingLayer, order-violation loss,
DAG edges, and end-to-end training with the order model path."""

import numpy as np
import pytest

import tensorflow as tf
from protcast.model.order_embeddings import (
    OrderEmbeddingLayer,
    build_order_embedding_model,
    order_violation_loss,
    order_violation_loss_soft,
)
from protcast.model.multilabel_classifier import (
    MultiLabelClassifier,
    GOEncoder,
)
from protcast.preprocessing.go_dag_edges import extract_dag_edges


# ---------------------------------------------------------------------------
# OrderEmbeddingLayer tests
# ---------------------------------------------------------------------------


class TestOrderEmbeddingLayer:
    def test_output_shape(self):
        layer = OrderEmbeddingLayer(num_classes=5, order_dim=8)
        x = tf.random.normal((4, 8))
        out = layer(x)
        assert out.shape == (4, 5)

    def test_output_range(self):
        """Membership scores are exp(-energy), so they lie in (0, 1]."""
        layer = OrderEmbeddingLayer(num_classes=3, order_dim=16)
        # Use non-negative inputs (the projection head emits softplus output)
        x = tf.abs(tf.random.normal((10, 16)))
        out = layer(x).numpy()
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_dominating_point_scores_high(self):
        """A protein point that dominates a term coordinate-wise should score
        higher than one that fails to dominate it.

        In the reversed product order, the protein must be >= the term on
        every dimension to fully entail it (energy 0, score 1)."""
        layer = OrderEmbeddingLayer(num_classes=1, order_dim=4, temperature=10.0)
        _ = layer(tf.zeros((1, 4)))  # build

        # Place the term at a small positive position (softplus(-0.433)≈0.5).
        layer.raw_terms.assign(tf.constant([[-0.433, -0.433, -0.433, -0.433]]))

        # Dominating point: well above the term on every dim
        dominating = tf.constant([[5.0, 5.0, 5.0, 5.0]])
        # Failing point: at the origin, below the (positive) term
        failing = tf.zeros((1, 4))

        score_dom = layer(dominating).numpy()[0, 0]
        score_fail = layer(failing).numpy()[0, 0]
        assert score_dom > score_fail
        # A fully-dominating point should be near 1 (energy ~0)
        assert score_dom > 0.99

    def test_get_term_positions(self):
        layer = OrderEmbeddingLayer(num_classes=3, order_dim=4)
        _ = layer(tf.zeros((1, 4)))
        positions = layer.get_term_positions()
        assert positions.shape == (3, 4)
        # softplus output is strictly positive
        assert np.all(positions > 0.0)

    def test_serialization(self):
        """Layer should be serializable for model save/load."""
        layer = OrderEmbeddingLayer(num_classes=5, order_dim=8)
        config = layer.get_config()
        assert config["num_classes"] == 5
        assert config["order_dim"] == 8
        assert config["temperature"] == 10.0

        restored = OrderEmbeddingLayer.from_config(config)
        assert restored.num_classes == 5
        assert restored.order_dim == 8


# ---------------------------------------------------------------------------
# Order-violation loss tests
# ---------------------------------------------------------------------------


class TestOrderViolationLoss:
    def test_no_edges_returns_zero(self):
        layer = OrderEmbeddingLayer(num_classes=3, order_dim=4)
        _ = layer(tf.zeros((1, 4)))
        empty_edges = tf.constant(np.zeros((0, 2), dtype=np.int32))
        loss_val = order_violation_loss(layer, empty_edges)
        assert float(loss_val) == 0.0

    def test_child_dominates_parent_low_loss(self):
        """When the child dominates the parent coordinate-wise (child is more
        specific), the order-violation loss should be ~0."""
        layer = OrderEmbeddingLayer(num_classes=2, order_dim=4)
        _ = layer(tf.zeros((1, 4)))

        # term 0 = parent (small coords), term 1 = child (larger coords).
        # softplus(0)≈0.69 (parent), softplus(3)≈3.05 (child) -> child dominates.
        layer.raw_terms.assign(tf.constant([
            [0.0, 0.0, 0.0, 0.0],   # parent: ~0.69 per dim
            [3.0, 3.0, 3.0, 3.0],   # child:  ~3.05 per dim (dominates parent)
        ]))

        edges = tf.constant([[0, 1]], dtype=tf.int32)  # [parent_idx, child_idx]
        loss_val = float(order_violation_loss(layer, edges))
        assert loss_val < 0.01

    def test_child_below_parent_high_loss(self):
        """When the child fails to dominate the parent (parent coords larger),
        the order-violation loss should be large."""
        layer = OrderEmbeddingLayer(num_classes=2, order_dim=4)
        _ = layer(tf.zeros((1, 4)))

        # Parent has large coords, child has small coords -> child violates.
        layer.raw_terms.assign(tf.constant([
            [3.0, 3.0, 3.0, 3.0],     # parent: ~3.05 per dim
            [-2.0, -2.0, -2.0, -2.0],  # child:  ~0.13 per dim (fails to dominate)
        ]))

        edges = tf.constant([[0, 1]], dtype=tf.int32)
        loss_val = float(order_violation_loss(layer, edges))
        assert loss_val > 1.0

    def test_soft_variant_nonzero_gradient_when_satisfied(self):
        """The soft variant should keep a non-zero gradient even when the child
        already dominates the parent — the gradient-flow property the hard
        ReLU variant lacks."""
        layer = OrderEmbeddingLayer(num_classes=2, order_dim=4)
        _ = layer(tf.zeros((1, 4)))
        # Child already dominates parent (constraint satisfied)
        layer.raw_terms.assign(tf.constant([
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 3.0, 3.0],
        ]))
        edges = tf.constant([[0, 1]], dtype=tf.int32)

        with tf.GradientTape() as tape:
            loss_soft = order_violation_loss_soft(layer, edges, beta=5.0)
        grad_soft = tape.gradient(loss_soft, layer.raw_terms)

        with tf.GradientTape() as tape:
            loss_hard = order_violation_loss(layer, edges)
        grad_hard = tape.gradient(loss_hard, layer.raw_terms)

        # Hard ReLU gives an (almost) zero gradient once satisfied; soft does not.
        soft_norm = float(tf.norm(grad_soft))
        hard_norm = float(tf.norm(grad_hard)) if grad_hard is not None else 0.0
        assert soft_norm > hard_norm
        assert soft_norm > 0.0


# ---------------------------------------------------------------------------
# build_order_embedding_model tests
# ---------------------------------------------------------------------------


class TestBuildOrderModel:
    def test_model_builds(self):
        model, order_layer = build_order_embedding_model(
            input_dim=64, num_classes=5, hidden_layers=[32, 16],
            dropout_rate=0.3, order_dim=8,
        )
        assert model is not None
        assert isinstance(order_layer, OrderEmbeddingLayer)

    def test_model_output_shape(self):
        model, _ = build_order_embedding_model(
            input_dim=64, num_classes=10, hidden_layers=[32],
            dropout_rate=0.2, order_dim=16,
        )
        x = np.random.randn(3, 64).astype(np.float32)
        y = model.predict(x, verbose=0)
        assert y.shape == (3, 10)

    def test_model_output_range(self):
        model, _ = build_order_embedding_model(
            input_dim=32, num_classes=4, hidden_layers=[16],
            dropout_rate=0.1, order_dim=8,
        )
        x = np.random.randn(5, 32).astype(np.float32)
        y = model.predict(x, verbose=0)
        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)


# ---------------------------------------------------------------------------
# extract_dag_edges tests
# ---------------------------------------------------------------------------


class FakeGOTerm:
    """Minimal stand-in for AnnotatedGOTerm."""
    def __init__(self, go_id, parents=None, children=None):
        self.go_id = go_id
        self.parents = parents or []
        self.children = children or []


class FakeGODag:
    """Minimal stand-in for AnnotatedGODag."""
    def __init__(self, terms):
        self.go_terms_map = {t.go_id: t for t in terms}


class TestExtractDagEdges:
    def test_simple_hierarchy(self):
        # GO:0001 is parent of GO:0002 and GO:0003
        terms = [
            FakeGOTerm("GO:0001", parents=[], children=["GO:0002", "GO:0003"]),
            FakeGOTerm("GO:0002", parents=["GO:0001"], children=[]),
            FakeGOTerm("GO:0003", parents=["GO:0001"], children=[]),
        ]
        dag = FakeGODag(terms)
        encoder = GOEncoder("test")
        encoder.fit(["GO:0001", "GO:0002", "GO:0003"])

        edges = extract_dag_edges(dag, ["GO:0001", "GO:0002", "GO:0003"], encoder)
        assert edges.shape[1] == 2
        assert len(edges) == 2  # two parent-child pairs

        # Both edges should have parent_idx=0 (GO:0001)
        parent_indices = set(edges[:, 0])
        assert 0 in parent_indices  # GO:0001 index

    def test_no_edges_when_disjoint(self):
        terms = [
            FakeGOTerm("GO:0001", parents=[], children=["GO:0005"]),
            FakeGOTerm("GO:0002", parents=["GO:0006"], children=[]),
        ]
        dag = FakeGODag(terms)
        encoder = GOEncoder("test")
        encoder.fit(["GO:0001", "GO:0002"])

        edges = extract_dag_edges(dag, ["GO:0001", "GO:0002"], encoder)
        assert edges.shape == (0, 2)

    def test_filters_terms_not_in_model(self):
        # GO:0001 -> GO:0002 -> GO:0003, but model only has GO:0001 and GO:0003
        terms = [
            FakeGOTerm("GO:0001", parents=[], children=["GO:0002"]),
            FakeGOTerm("GO:0002", parents=["GO:0001"], children=["GO:0003"]),
            FakeGOTerm("GO:0003", parents=["GO:0002"], children=[]),
        ]
        dag = FakeGODag(terms)
        encoder = GOEncoder("test")
        encoder.fit(["GO:0001", "GO:0003"])

        edges = extract_dag_edges(dag, ["GO:0001", "GO:0003"], encoder)
        # GO:0002 not in model, so no edges should exist
        assert edges.shape == (0, 2)


# ---------------------------------------------------------------------------
# MultiLabelClassifier order mode integration test
# ---------------------------------------------------------------------------


class TestMultiLabelClassifierOrderMode:
    def _make_config(self, use_order=True, order_variant="soft"):
        return {
            "USER": "test",
            "EXPERIMENT_NAME": "test",
            "OPTIMIZER": "adam",
            "LOSS": "binary_crossentropy",
            "METRICS": ["accuracy"],
            "EPOCHS": 3,
            "BATCH_SIZE": 16,
            "HIDDEN_LAYERS": [32, 16],
            "DROPOUT": 0.3,
            "PRED_THRESHOLD": 50.0,
            "VALIDATION_SPLIT": 0.2,
            "PATIENCE": 2,
            "USE_ORDER_EMBEDDINGS": use_order,
            "ORDER_DIM": 8,
            "ORDER_TEMPERATURE": 10.0,
            "ORDER_WEIGHT": 0.1,
            "ORDER_VARIANT": order_variant,
            "ORDER_BETA": 5.0,
        }

    def _make_synthetic_data(self, n_proteins=100, n_go_terms=5, embed_dim=32):
        np.random.seed(42)
        go_ids = [f"GO:{i:07d}" for i in range(n_go_terms)]
        protein_embeddings = {}
        protein_go_terms = {}

        for i in range(n_proteins):
            pid = f"P{i:05d}"
            protein_embeddings[pid] = np.random.randn(embed_dim).astype(np.float32)
            n_labels = np.random.randint(1, min(4, n_go_terms + 1))
            protein_go_terms[pid] = set(
                np.random.choice(go_ids, size=n_labels, replace=False)
            )
        return protein_embeddings, protein_go_terms, go_ids

    def _make_fake_dag(self, go_ids):
        """Create a simple chain DAG: go_ids[0] -> go_ids[1] -> ... -> go_ids[-1]."""
        terms = []
        for i, go_id in enumerate(go_ids):
            parents = [go_ids[i - 1]] if i > 0 else []
            children = [go_ids[i + 1]] if i < len(go_ids) - 1 else []
            terms.append(FakeGOTerm(go_id, parents=parents, children=children))
        return FakeGODag(terms)

    def test_order_model_trains(self, tmp_path):
        """Full training run with order embeddings and a fake DAG."""
        embeddings, go_terms, go_ids = self._make_synthetic_data()
        dag = self._make_fake_dag(go_ids)

        clf = MultiLabelClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=self._make_config(use_order=True),
            id=str(tmp_path / "test_order"),
            go_dag=dag,
        )
        clf.run()

        assert clf.model is not None
        assert hasattr(clf, "best_threshold")
        assert 0.0 < clf.best_threshold < 1.0
        assert clf._order_layer is not None

        # Predictions should be valid (0, 1] scores
        X_test = np.random.randn(3, 32).astype(np.float32)
        y_pred = clf.model.predict(X_test, verbose=0)
        assert y_pred.shape == (3, 5)
        assert np.all(y_pred >= 0)
        assert np.all(y_pred <= 1)

    def test_order_model_hard_variant_trains(self, tmp_path):
        """The hard (ReLU) order-violation variant should also train end-to-end."""
        embeddings, go_terms, go_ids = self._make_synthetic_data()
        dag = self._make_fake_dag(go_ids)

        clf = MultiLabelClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=self._make_config(use_order=True, order_variant="hard"),
            id=str(tmp_path / "test_order_hard"),
            go_dag=dag,
        )
        clf.run()
        assert clf.model is not None
        assert clf._order_layer is not None

    def test_order_model_without_dag(self, tmp_path):
        """Order model should work without a DAG (no order-violation loss)."""
        embeddings, go_terms, go_ids = self._make_synthetic_data()

        clf = MultiLabelClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=self._make_config(use_order=True),
            id=str(tmp_path / "test_order_nodag"),
            go_dag=None,
        )
        clf.run()
        assert clf.model is not None
        assert len(clf._dag_edges) == 0

    def test_flat_model_still_works(self, tmp_path):
        """Verify flat path is unaffected by order config being present."""
        embeddings, go_terms, go_ids = self._make_synthetic_data()

        clf = MultiLabelClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=self._make_config(use_order=False),
            id=str(tmp_path / "test_flat"),
        )
        clf.run()
        assert clf.model is not None
        assert clf._order_layer is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
