"""Tests for box embeddings: BoxEmbeddingLayer, containment loss, DAG edges,
and end-to-end training with the box model path."""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
from protcast.model.box_embeddings import (
    BoxEmbeddingLayer,
    build_box_embedding_model,
    containment_loss,
)
from protcast.model.multilabel_classifier import (
    MultiLabelClassifier,
    GOEncoder,
)
from protcast.preprocessing.go_dag_edges import extract_dag_edges


# ---------------------------------------------------------------------------
# BoxEmbeddingLayer tests
# ---------------------------------------------------------------------------


class TestBoxEmbeddingLayer:
    def test_output_shape(self):
        layer = BoxEmbeddingLayer(num_classes=5, box_dim=8)
        x = tf.random.normal((4, 8))
        out = layer(x)
        assert out.shape == (4, 5)

    def test_output_range(self):
        """Membership scores should be in (0, 1) since they are sigmoid."""
        layer = BoxEmbeddingLayer(num_classes=3, box_dim=16)
        x = tf.random.normal((10, 16))
        out = layer(x).numpy()
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_point_inside_box_scores_high(self):
        """A point at the center of a box should score higher than one far away."""
        layer = BoxEmbeddingLayer(num_classes=1, box_dim=4, temperature=10.0)
        # Build by calling once
        _ = layer(tf.zeros((1, 4)))

        # Set box to center=0, offset=1 (box spans [-1, 1] in each dim)
        layer.centers.assign(tf.zeros((1, 4)))
        layer.log_offsets.assign(tf.ones((1, 4)))  # softplus(1) ≈ 1.31

        center_point = tf.zeros((1, 4))
        far_point = tf.constant([[10.0, 10.0, 10.0, 10.0]])

        score_center = layer(center_point).numpy()[0, 0]
        score_far = layer(far_point).numpy()[0, 0]
        assert score_center > score_far

    def test_get_box_bounds(self):
        layer = BoxEmbeddingLayer(num_classes=3, box_dim=4)
        _ = layer(tf.zeros((1, 4)))
        box_min, box_max = layer.get_box_bounds()
        assert box_min.shape == (3, 4)
        assert box_max.shape == (3, 4)
        assert np.all(box_min <= box_max)

    def test_serialization(self):
        """Layer should be serializable for model save/load."""
        layer = BoxEmbeddingLayer(num_classes=5, box_dim=8)
        config = layer.get_config()
        assert config["num_classes"] == 5
        assert config["box_dim"] == 8
        assert config["temperature"] == 10.0

        restored = BoxEmbeddingLayer.from_config(config)
        assert restored.num_classes == 5
        assert restored.box_dim == 8


# ---------------------------------------------------------------------------
# Containment loss tests
# ---------------------------------------------------------------------------


class TestContainmentLoss:
    def test_no_edges_returns_zero(self):
        layer = BoxEmbeddingLayer(num_classes=3, box_dim=4)
        _ = layer(tf.zeros((1, 4)))
        empty_edges = tf.constant(np.zeros((0, 2), dtype=np.int32))
        loss_val = containment_loss(layer, empty_edges)
        assert float(loss_val) == 0.0

    def test_child_inside_parent_low_loss(self):
        """When child box is fully inside parent, loss should be ~0."""
        layer = BoxEmbeddingLayer(num_classes=2, box_dim=4)
        _ = layer(tf.zeros((1, 4)))

        # Parent: center=0, large offset → big box
        # Child: center=0, small offset → small box inside parent
        layer.centers.assign(tf.zeros((2, 4)))
        # softplus(3) ≈ 3.05, softplus(-2) ≈ 0.13
        layer.log_offsets.assign(tf.constant([
            [3.0, 3.0, 3.0, 3.0],   # parent: big box
            [-2.0, -2.0, -2.0, -2.0],  # child: tiny box
        ]))

        edges = tf.constant([[0, 1]], dtype=tf.int32)
        loss_val = float(containment_loss(layer, edges))
        assert loss_val < 0.01

    def test_child_outside_parent_high_loss(self):
        """When child box extends far beyond parent, loss should be large."""
        layer = BoxEmbeddingLayer(num_classes=2, box_dim=4)
        _ = layer(tf.zeros((1, 4)))

        # Parent: small box at center=0
        # Child: huge box far from parent
        layer.centers.assign(tf.constant([
            [0.0, 0.0, 0.0, 0.0],    # parent
            [10.0, 10.0, 10.0, 10.0],  # child far away
        ]))
        layer.log_offsets.assign(tf.constant([
            [-2.0, -2.0, -2.0, -2.0],  # parent: tiny
            [3.0, 3.0, 3.0, 3.0],      # child: huge
        ]))

        edges = tf.constant([[0, 1]], dtype=tf.int32)
        loss_val = float(containment_loss(layer, edges))
        assert loss_val > 1.0


# ---------------------------------------------------------------------------
# build_box_embedding_model tests
# ---------------------------------------------------------------------------


class TestBuildBoxModel:
    def test_model_builds(self):
        model, box_layer = build_box_embedding_model(
            input_dim=64, num_classes=5, hidden_layers=[32, 16],
            dropout_rate=0.3, box_dim=8,
        )
        assert model is not None
        assert isinstance(box_layer, BoxEmbeddingLayer)

    def test_model_output_shape(self):
        model, _ = build_box_embedding_model(
            input_dim=64, num_classes=10, hidden_layers=[32],
            dropout_rate=0.2, box_dim=16,
        )
        x = np.random.randn(3, 64).astype(np.float32)
        y = model.predict(x, verbose=0)
        assert y.shape == (3, 10)

    def test_model_output_range(self):
        model, _ = build_box_embedding_model(
            input_dim=32, num_classes=4, hidden_layers=[16],
            dropout_rate=0.1, box_dim=8,
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
# MultiLabelClassifier box mode integration test
# ---------------------------------------------------------------------------


class TestMultiLabelClassifierBoxMode:
    def _make_config(self, use_boxes=True):
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
            "USE_BOX_EMBEDDINGS": use_boxes,
            "BOX_DIM": 8,
            "BOX_TEMPERATURE": 10.0,
            "CONTAINMENT_WEIGHT": 0.1,
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

    def test_box_model_trains(self, tmp_path):
        """Full training run with box embeddings and a fake DAG."""
        embeddings, go_terms, go_ids = self._make_synthetic_data()
        dag = self._make_fake_dag(go_ids)

        clf = MultiLabelClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=self._make_config(use_boxes=True),
            id=str(tmp_path / "test_box"),
            go_dag=dag,
        )
        clf.run()

        assert clf.model is not None
        assert hasattr(clf, "best_threshold")
        assert 0.0 < clf.best_threshold < 1.0
        assert clf._box_layer is not None

        # Predictions should be valid sigmoid outputs
        X_test = np.random.randn(3, 32).astype(np.float32)
        y_pred = clf.model.predict(X_test, verbose=0)
        assert y_pred.shape == (3, 5)
        assert np.all(y_pred >= 0)
        assert np.all(y_pred <= 1)

    def test_box_model_without_dag(self, tmp_path):
        """Box model should work without DAG (no containment loss)."""
        embeddings, go_terms, go_ids = self._make_synthetic_data()

        clf = MultiLabelClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=self._make_config(use_boxes=True),
            id=str(tmp_path / "test_box_nodag"),
            go_dag=None,
        )
        clf.run()
        assert clf.model is not None
        assert len(clf._dag_edges) == 0

    def test_flat_model_still_works(self, tmp_path):
        """Verify flat path is unaffected by box config being present."""
        embeddings, go_terms, go_ids = self._make_synthetic_data()

        clf = MultiLabelClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=self._make_config(use_boxes=False),
            id=str(tmp_path / "test_flat"),
        )
        clf.run()
        assert clf.model is not None
        assert clf._box_layer is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
