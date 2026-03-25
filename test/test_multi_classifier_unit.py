"""Unit tests for MultiClassifier and its GOEncoder.

These tests exercise the ESM-embeddings pathway of MultiClassifier
(no external model downloads, no GPU, no network) and the GOEncoder
label-encoding logic.
"""

from pathlib import Path

import numpy as np
import pytest

from protcast.model.multi_classifier import MultiClassifier, GOEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Return a minimal config dict suitable for MultiClassifier."""
    cfg = {
        "USER": "test",
        "EXPERIMENT_NAME": "unit_test",
        "REGISTERED_MODEL_NAME": "test_model",
        "OPTIMIZER": "adam",
        "LOSS": "categorical_crossentropy",
        "METRICS": ["accuracy"],
        "EPOCHS": 3,
        "BATCH_SIZE": 16,
        "HIDDEN_LAYERS": [64, 32],
        "DROPOUT": 0.3,
        "PRED_THRESHOLD": 50.0,
        "VALIDATION_SPLIT": 0.2,
        "PATIENCE": 2,
    }
    cfg.update(overrides)
    return cfg


def _make_esm_proteins(n_per_class=30, embed_dim=32, n_classes=4, seed=42):
    """Create synthetic ESM-style protein data: {go_id: {pid: np.array}}."""
    rng = np.random.RandomState(seed)
    proteins = {}
    for c in range(n_classes):
        go_id = f"GO:{c:07d}"
        proteins[go_id] = {}
        for i in range(n_per_class):
            pid = f"P{c}_{i:04d}"
            proteins[go_id][pid] = rng.randn(embed_dim).astype(np.float32)
    return proteins


# ---------------------------------------------------------------------------
# GOEncoder tests
# ---------------------------------------------------------------------------

class TestGOEncoder:
    def setup_method(self):
        self.encoder = GOEncoder("unit_test")
        self.go_ids = ["GO:0000001", "GO:0000002", "GO:0000003"]
        self.encoder.fit(self.go_ids)

    def test_fit_creates_mappings(self):
        assert self.encoder.num_classes == 3
        assert set(self.encoder.go_to_int.keys()) == set(self.go_ids)
        assert set(self.encoder.int_to_go.values()) == set(self.go_ids)

    def test_encode_returns_integer(self):
        idx = self.encoder.encode("GO:0000001")
        assert isinstance(idx, int)
        assert 0 <= idx < 3

    def test_encode_unknown_raises(self):
        with pytest.raises(KeyError):
            self.encoder.encode("GO:9999999")

    def test_decode_single_integer(self):
        idx = self.encoder.encode("GO:0000002")
        assert self.encoder.decode(idx) == "GO:0000002"

    def test_decode_one_hot(self):
        one_hot = np.array([[0, 0, 1], [1, 0, 0]])
        decoded = self.encoder.decode(one_hot)
        assert len(decoded) == 2
        assert decoded[0] == "GO:0000003"
        assert decoded[1] == "GO:0000001"

    def test_decode_probabilities(self):
        probs = np.array([[0.1, 0.8, 0.1]])
        results = self.encoder.decode_probabilities(probs, top_k=1)
        assert len(results) == 1
        assert results[0][0][0] == "GO:0000002"

    def test_decode_probabilities_top_k(self):
        probs = np.array([[0.1, 0.7, 0.2]])
        results = self.encoder.decode_probabilities(probs, top_k=2)
        assert len(results[0]) == 2
        # Top result should be GO:0000002 (index 1, prob 0.7)
        assert results[0][0][0] == "GO:0000002"

    def test_save_load(self, tmp_path):
        self.encoder.id = str(tmp_path / "enc")
        self.encoder.save()
        loaded = GOEncoder.load(f"{tmp_path / 'enc'}_GOEncoder.pkl")
        assert loaded.num_classes == self.encoder.num_classes
        assert loaded.go_to_int == self.encoder.go_to_int
        assert loaded.int_to_go == self.encoder.int_to_go

    def test_fit_not_called_raises(self):
        fresh = GOEncoder("fresh")
        with pytest.raises(ValueError):
            fresh.encode("GO:0000001")


# ---------------------------------------------------------------------------
# MultiClassifier — ESM embeddings pathway
# ---------------------------------------------------------------------------

class TestMultiClassifierESM:
    """Tests using synthetic ESM embeddings (no external deps)."""

    def _build_classifier(self, tmp_path, **config_overrides):
        proteins = _make_esm_proteins()
        config = _make_config(**config_overrides)
        return MultiClassifier(
            algorithm="esmc_300m",
            verbose=False,
            proteins=proteins,
            config=config,
            id=str(tmp_path / "test"),
            input_source="esm_embeddings",
            random_state=42,
        )

    def test_init_sets_attributes(self, tmp_path):
        clf = self._build_classifier(tmp_path)
        assert clf.input_source == "esm_embeddings"
        assert clf.algorithm == "esmc_300m"
        assert clf.random_state == 42

    def test_invalid_input_source_raises(self, tmp_path):
        proteins = _make_esm_proteins()
        with pytest.raises(ValueError, match="input_source must be one of"):
            MultiClassifier(
                algorithm="test",
                verbose=False,
                proteins=proteins,
                config=_make_config(),
                id="test",
                input_source="invalid_source",
            )

    def test_get_esm_embeddings(self, tmp_path):
        clf = self._build_classifier(tmp_path)
        clf.get_feature_vectors()
        # 4 classes
        assert len(clf.go_ids) == 4
        assert len(clf.features) == 4
        assert len(clf.pids) == 4
        # 30 proteins per class
        assert all(len(f) == 30 for f in clf.features)
        # Vector length recorded
        assert clf.vector_length == 32

    def test_prepare_data_shapes(self, tmp_path):
        clf = self._build_classifier(tmp_path)
        clf.get_feature_vectors()
        clf.prepare_data()
        # 4 classes * 30 proteins = 120 total
        assert clf.X.shape == (120, 32)
        assert clf.y.shape == (120, 4)
        # One-hot: each row sums to 1
        assert np.allclose(clf.y.sum(axis=1), 1.0)

    def test_build_model_architecture(self, tmp_path):
        clf = self._build_classifier(tmp_path)
        clf.get_feature_vectors()
        clf.prepare_data()
        clf.build_model()
        assert clf.model is not None
        # Output layer should have 4 units (one per class)
        output_shape = clf.model.output_shape
        assert output_shape[-1] == 4

    def test_full_training_run(self, tmp_path):
        clf = self._build_classifier(tmp_path, EPOCHS=3, PATIENCE=2)
        clf.run()
        assert clf.model is not None
        assert clf.training_time > 0
        # History should exist
        assert hasattr(clf, "history")
        assert "loss" in clf.history.history

    def test_prediction_shape_and_range(self, tmp_path):
        clf = self._build_classifier(tmp_path, EPOCHS=3)
        clf.run()
        X_test = np.random.randn(5, 32).astype(np.float32)
        preds = clf.model.predict(X_test, verbose=0)
        assert preds.shape == (5, 4)
        # Softmax: rows sum to ~1
        assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-5)
        # All values in [0, 1]
        assert (preds >= 0).all()
        assert (preds <= 1).all()

    def test_save_and_load_model(self, tmp_path):
        clf = self._build_classifier(tmp_path, EPOCHS=3)
        clf.run()
        model_path = Path(f"{clf.get_name()}.keras")
        loaded = MultiClassifier.load_model(model_path)
        X_test = np.random.randn(3, 32).astype(np.float32)
        loaded_preds = loaded.predict(X_test, verbose=0)
        # Loaded model should produce valid softmax outputs
        assert loaded_preds.shape == (3, 4)
        assert np.allclose(loaded_preds.sum(axis=1), 1.0, atol=1e-5)

    def test_get_name(self, tmp_path):
        clf = self._build_classifier(tmp_path)
        name = clf.get_name()
        assert "esmc_300m" in name

    def test_random_state_reproducibility(self, tmp_path):
        """Two classifiers with same random_state should split data identically."""
        clf1 = self._build_classifier(tmp_path)
        clf1.get_feature_vectors()
        clf1.prepare_data()

        clf2 = self._build_classifier(tmp_path)
        clf2.get_feature_vectors()
        clf2.prepare_data()

        np.testing.assert_array_equal(clf1.X, clf2.X)
        np.testing.assert_array_equal(clf1.y, clf2.y)


# ---------------------------------------------------------------------------
# MultiClassifier — combined mode (mocked feature vectors)
# ---------------------------------------------------------------------------

class TestMultiClassifierCombined:
    """Test the combined input_source path with mocked Calculator."""

    def test_combined_rejects_without_proper_data(self, tmp_path):
        """Combined mode expects {go_id: {pid: {'embedding': ..., 'sequence': ...}}}."""
        # Provide ESM-style data (no 'embedding'/'sequence' keys) — should fail
        proteins = _make_esm_proteins(n_per_class=5)
        config = _make_config()
        clf = MultiClassifier(
            algorithm="combined",
            verbose=False,
            proteins=proteins,
            config=config,
            id=str(tmp_path / "test"),
            input_source="combined",
        )
        with pytest.raises((TypeError, KeyError, AttributeError, IndexError)):
            clf.get_feature_vectors()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
