"""Tests for MultiLabelClassifier and multi-label evaluation metrics."""

import numpy as np
import pytest

from protcast.model.multilabel_classifier import MultiLabelClassifier, GOEncoder
from protcast.model.stats.utils import calculate_fmax, calculate_smin


# --- GOEncoder tests ---


class TestGOEncoder:
    def setup_method(self):
        self.encoder = GOEncoder("test")
        self.go_ids = ["GO:0001", "GO:0002", "GO:0003", "GO:0004"]
        self.encoder.fit(self.go_ids)

    def test_fit(self):
        assert self.encoder.num_classes == 4
        assert self.encoder.go_to_int["GO:0001"] == 0
        assert self.encoder.int_to_go[0] == "GO:0001"

    def test_encode_multilabel(self):
        label = self.encoder.encode_multilabel({"GO:0001", "GO:0003"})
        assert label.shape == (4,)
        assert label[0] == 1.0  # GO:0001
        assert label[1] == 0.0  # GO:0002
        assert label[2] == 1.0  # GO:0003
        assert label[3] == 0.0  # GO:0004

    def test_encode_multilabel_unknown_ignored(self):
        label = self.encoder.encode_multilabel({"GO:0001", "GO:9999"})
        assert label.sum() == 1.0  # Only GO:0001 encoded

    def test_decode_multilabel(self):
        probs = np.array([0.9, 0.1, 0.8, 0.3])
        results = self.encoder.decode_multilabel(probs, threshold=0.5)
        assert len(results) == 2
        assert results[0] == ("GO:0001", 0.9)
        assert results[1] == ("GO:0003", 0.8)

    def test_decode_multilabel_none_above_threshold(self):
        probs = np.array([0.1, 0.1, 0.1, 0.1])
        results = self.encoder.decode_multilabel(probs, threshold=0.5)
        assert len(results) == 0

    def test_decode_probabilities_backward_compat(self):
        probs = np.array([[0.9, 0.1, 0.8, 0.3]])
        results = self.encoder.decode_probabilities(probs, top_k=2)
        assert len(results) == 1
        assert len(results[0]) == 2
        assert results[0][0][0] == "GO:0001"

    def test_save_load(self, tmp_path):
        self.encoder.id = str(tmp_path / "test")
        self.encoder.save()
        loaded = GOEncoder.load(f"{tmp_path / 'test'}_GOEncoder.pkl")
        assert loaded.num_classes == 4
        assert loaded.go_to_int == self.encoder.go_to_int


# --- Fmax / Smin tests ---


class TestFmax:
    def test_perfect_predictions(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]])
        fmax, threshold = calculate_fmax(y_true, y_pred)
        assert fmax > 0.9
        assert 0.0 < threshold < 1.0

    def test_worst_predictions(self):
        y_true = np.array([[1, 0, 0], [0, 0, 1]])
        y_pred = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        fmax, _ = calculate_fmax(y_true, y_pred)
        assert fmax < 0.5

    def test_single_class(self):
        y_true = np.array([[1], [1], [0]])
        y_pred = np.array([[0.8], [0.7], [0.2]])
        fmax, _ = calculate_fmax(y_true, y_pred)
        assert fmax > 0.5

    def test_returns_tuple(self):
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])
        result = calculate_fmax(y_true, y_pred)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestSmin:
    def test_perfect_predictions(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]])
        smin, _ = calculate_smin(y_true, y_pred)
        assert smin < 1.0

    def test_returns_tuple(self):
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])
        result = calculate_smin(y_true, y_pred)
        assert isinstance(result, tuple)
        assert len(result) == 2


# --- MultiLabelClassifier integration test ---


class TestMultiLabelClassifier:
    """Integration test with synthetic data."""

    def _make_config(self):
        return {
            "USER": "test",
            "EXPERIMENT_NAME": "test",
            "OPTIMIZER": "adam",
            "LOSS": "binary_crossentropy",
            "METRICS": ["accuracy"],
            "EPOCHS": 5,
            "BATCH_SIZE": 16,
            "NEURONS": 32,
            "DROPOUT": 0.3,
            "PRED_THRESHOLD": 50.0,
            "VALIDATION_SPLIT": 0.2,
            "PATIENCE": 3,
        }

    def _make_synthetic_data(self, n_proteins=200, n_go_terms=10, embed_dim=64):
        """Create synthetic protein embeddings and multi-label annotations."""
        np.random.seed(42)
        go_ids = [f"GO:{i:07d}" for i in range(n_go_terms)]

        protein_embeddings = {}
        protein_go_terms = {}

        for i in range(n_proteins):
            pid = f"P{i:05d}"
            protein_embeddings[pid] = np.random.randn(embed_dim).astype(np.float32)
            # Each protein gets 1-4 random GO terms
            n_labels = np.random.randint(1, min(5, n_go_terms + 1))
            protein_go_terms[pid] = set(
                np.random.choice(go_ids, size=n_labels, replace=False)
            )

        return protein_embeddings, protein_go_terms, go_ids

    def test_prepare_data_shapes(self):
        embeddings, go_terms, go_ids = self._make_synthetic_data()
        clf = MultiLabelClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=self._make_config(),
            id="test",
        )
        clf.prepare_data()

        assert clf.X.shape == (200, 64)
        assert clf.y.shape == (200, 10)
        # Multi-hot: each row should have >= 1 label
        assert (clf.y.sum(axis=1) >= 1).all()
        # Some rows should have > 1 label
        assert (clf.y.sum(axis=1) > 1).any()

    def test_full_training_run(self, tmp_path):
        embeddings, go_terms, go_ids = self._make_synthetic_data()
        clf = MultiLabelClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=self._make_config(),
            id=str(tmp_path / "test"),
        )
        clf.run()

        # Model should exist
        assert clf.model is not None
        # Should have computed Fmax
        assert hasattr(clf, "best_threshold")
        assert 0.0 < clf.best_threshold < 1.0

        # Test prediction shape
        X_test = np.random.randn(5, 64).astype(np.float32)
        y_pred = clf.model.predict(X_test, verbose=0)
        assert y_pred.shape == (5, 10)
        # Sigmoid outputs should be in [0, 1]
        assert (y_pred >= 0).all()
        assert (y_pred <= 1).all()
        # Should NOT sum to 1 (unlike softmax) in general
        # (they could by coincidence, but very unlikely with 10 classes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
