"""test_knn_classifier.py

Unit tests for KNNClassifier — mirrors test_multilabel_classifier.py structure.
"""

import pytest
import numpy as np

from protcast.model.knn_classifier import KNNClassifier
from protcast.model.multilabel_classifier import GOEncoder
from protcast.model.stats.utils import calculate_fmax, calculate_smin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Return a minimal config dict for KNNClassifier."""
    config = {
        "USER": "test",
        "EXPERIMENT_NAME": "knn_test",
        "OPTIMIZER": "adam",
        "VALIDATION_SPLIT": 0.2,
        "KNN_N_NEIGHBORS": 5,
        "KNN_METRIC": "cosine",
        "KNN_WEIGHTS": "distance",
        "KNN_ALGORITHM": "brute",
    }
    config.update(overrides)
    return config


def _make_synthetic_data(n_proteins=200, n_go_terms=10, embed_dim=64, seed=42):
    """Generate synthetic protein embeddings and GO annotations.

    Creates clusters in embedding space so KNN can learn real signal.
    """
    rng = np.random.RandomState(seed)

    go_ids = [f"GO:{i:07d}" for i in range(1, n_go_terms + 1)]
    protein_embeddings = {}
    protein_go_terms = {}

    # Create cluster centers for each GO term
    centers = rng.randn(n_go_terms, embed_dim).astype(np.float32)

    for i in range(n_proteins):
        pid = f"PROT_{i:04d}"
        # Assign 1-3 GO terms per protein
        n_terms = rng.randint(1, min(4, n_go_terms + 1))
        term_indices = rng.choice(n_go_terms, size=n_terms, replace=False)
        protein_go_terms[pid] = {go_ids[j] for j in term_indices}

        # Embedding is a noisy version of the mean of assigned cluster centers
        center = centers[term_indices].mean(axis=0)
        noise = rng.randn(embed_dim).astype(np.float32) * 0.3
        protein_embeddings[pid] = center + noise

    return protein_embeddings, protein_go_terms, go_ids


# ---------------------------------------------------------------------------
# GOEncoder tests (reuses same encoder as multilabel)
# ---------------------------------------------------------------------------

class TestGOEncoder:
    def test_fit_and_encode(self):
        enc = GOEncoder("test")
        enc.fit(["GO:0001", "GO:0002", "GO:0003"])
        assert enc.num_classes == 3
        assert enc.encode("GO:0001") == 0

    def test_encode_multilabel(self):
        enc = GOEncoder("test")
        enc.fit(["GO:0001", "GO:0002", "GO:0003"])
        vec = enc.encode_multilabel({"GO:0001", "GO:0003"})
        assert vec.shape == (3,)
        assert vec[0] == 1.0
        assert vec[1] == 0.0
        assert vec[2] == 1.0

    def test_decode_multilabel(self):
        enc = GOEncoder("test")
        enc.fit(["GO:0001", "GO:0002", "GO:0003"])
        probs = np.array([0.9, 0.1, 0.8])
        results = enc.decode_multilabel(probs, threshold=0.5)
        assert len(results) == 2
        assert results[0][0] == "GO:0001"


# ---------------------------------------------------------------------------
# KNNClassifier tests
# ---------------------------------------------------------------------------

class TestKNNClassifier:
    def test_prepare_data_shapes(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        config = _make_config()
        clf = KNNClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=config,
            id="test",
        )
        clf.prepare_data()
        assert clf.X.shape == (200, 64)
        assert clf.y.shape == (200, 10)

    def test_full_training_run(self, tmp_path):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        config = _make_config()
        clf = KNNClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=config,
            id="test",
        )
        clf.prepare_data()
        clf.build_model()
        clf.train_model()

        assert clf.best_fmax > 0.0
        assert 0 < clf.best_threshold < 1
        assert clf.best_smin >= 0

    def test_predict_scores_in_range(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        config = _make_config()
        clf = KNNClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=config,
            id="test",
        )
        clf.prepare_data()
        clf.build_model()
        clf.train_model()

        # Predict on 5 random samples
        scores = clf.predict(clf.X_val[:5])
        assert scores.shape == (5, 10)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_save_load(self, tmp_path):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        config = _make_config()
        clf = KNNClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=config,
            id="test",
        )
        clf.prepare_data()
        clf.build_model()
        clf.train_model()

        save_path = str(tmp_path / "model.joblib")
        clf.save_model(save_path)

        artifact = KNNClassifier.load_model(save_path)
        assert "nn" in artifact
        assert "y_train" in artifact
        assert artifact["best_threshold"] == clf.best_threshold

        # Predictions should match
        scores_orig = clf.predict(clf.X_val[:3])
        scores_loaded = KNNClassifier.predict_from_artifact(artifact, clf.X_val[:3])
        np.testing.assert_array_almost_equal(scores_orig, scores_loaded)

    def test_uniform_vs_distance_weighting(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()

        clf_dist = KNNClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=_make_config(KNN_WEIGHTS="distance"),
            id="test_dist",
        )
        clf_dist.prepare_data()
        clf_dist.build_model()
        clf_dist.train_model()

        clf_uni = KNNClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=_make_config(KNN_WEIGHTS="uniform"),
            id="test_uni",
        )
        clf_uni.prepare_data()
        clf_uni.build_model()
        clf_uni.train_model()

        # Scores should differ between weighting schemes
        scores_dist = clf_dist.predict(clf_dist.X_val[:5])
        scores_uni = clf_uni.predict(clf_uni.X_val[:5])
        assert not np.allclose(scores_dist, scores_uni)

    def test_varying_k(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()

        clf_k3 = KNNClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=_make_config(KNN_N_NEIGHBORS=3),
            id="test_k3",
        )
        clf_k3.prepare_data()
        clf_k3.build_model()
        clf_k3.train_model()

        clf_k20 = KNNClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=_make_config(KNN_N_NEIGHBORS=20),
            id="test_k20",
        )
        clf_k20.prepare_data()
        clf_k20.build_model()
        clf_k20.train_model()

        # Different K should produce different predictions
        scores_k3 = clf_k3.predict(clf_k3.X_val[:5])
        scores_k20 = clf_k20.predict(clf_k20.X_val[:5])
        assert not np.allclose(scores_k3, scores_k20)

    def test_k_clamped_for_small_dataset(self):
        """K should be clamped when dataset is smaller than requested K."""
        embeddings, go_terms, go_ids = _make_synthetic_data(n_proteins=15)
        config = _make_config(KNN_N_NEIGHBORS=100)
        clf = KNNClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=config,
            id="test",
        )
        clf.prepare_data()
        clf.build_model()
        # K should have been reduced
        assert clf.knn_n_neighbors < 100

    def test_frequency_metrics(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        config = _make_config()
        clf = KNNClassifier(
            verbose=False,
            protein_embeddings=embeddings,
            protein_go_terms=go_terms,
            go_ids=go_ids,
            config=config,
            id="test",
        )
        clf.prepare_data()
        clf.build_model()
        clf.train_model()

        freq_metrics = clf.compute_frequency_metrics(clf.y_val, clf.y_val_pred)
        # With 200 proteins and 10 terms, all should be in "rare_lt50"
        assert "rare_lt50" in freq_metrics
        assert freq_metrics["rare_lt50"]["n_terms"] > 0
        assert 0 <= freq_metrics["rare_lt50"]["fmax"] <= 1


# ---------------------------------------------------------------------------
# Fmax / Smin metric tests (shared with multilabel)
# ---------------------------------------------------------------------------

class TestFmax:
    def test_perfect_prediction(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        y_pred = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], dtype=np.float32)
        fmax, threshold = calculate_fmax(y_true, y_pred)
        assert fmax > 0.9

    def test_random_prediction(self):
        rng = np.random.RandomState(42)
        y_true = (rng.rand(50, 10) > 0.7).astype(np.float32)
        y_pred = rng.rand(50, 10).astype(np.float32)
        fmax, threshold = calculate_fmax(y_true, y_pred)
        assert 0 <= fmax <= 1
        assert 0 < threshold < 1


class TestSmin:
    def test_perfect_prediction(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        y_pred = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], dtype=np.float32)
        smin, _ = calculate_smin(y_true, y_pred)
        assert smin < 0.5

    def test_random_prediction(self):
        rng = np.random.RandomState(42)
        y_true = (rng.rand(50, 10) > 0.7).astype(np.float32)
        y_pred = rng.rand(50, 10).astype(np.float32)
        smin, threshold = calculate_smin(y_true, y_pred)
        assert smin >= 0
