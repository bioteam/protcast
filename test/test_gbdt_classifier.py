"""test_gbdt_classifier.py

Unit tests for GBDTClassifier (LightGBM one-vs-rest) — mirrors
test_knn_classifier.py. The whole module is skipped when lightgbm is not
installed, so CI without the optional dependency stays green.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from protcast.model.gbdt_classifier import GBDTClassifier  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _run_in_tmp(tmp_path, monkeypatch):
    """GOEncoder.save() and save_importances() write into CWD — keep it clean."""
    monkeypatch.chdir(tmp_path)


def _make_config(**overrides):
    """Minimal config: a small boosting budget so tests run in seconds."""
    config = {
        "USER": "test",
        "EXPERIMENT_NAME": "gbdt_test",
        "VALIDATION_SPLIT": 0.2,
        "GBDT_N_ESTIMATORS": 40,
        "GBDT_NUM_LEAVES": 7,
        "GBDT_MIN_CHILD_SAMPLES": 5,
        "GBDT_COLSAMPLE_BYTREE": 0.5,
        "GBDT_N_JOBS": 2,
    }
    config.update(overrides)
    return config


def _make_synthetic_data(n_proteins=200, n_go_terms=10, embed_dim=64, seed=42):
    """Clustered embeddings (same generator as test_knn_classifier.py)."""
    rng = np.random.RandomState(seed)
    go_ids = [f"GO:{i:07d}" for i in range(1, n_go_terms + 1)]
    protein_embeddings, protein_go_terms = {}, {}
    centers = rng.randn(n_go_terms, embed_dim).astype(np.float32)
    for i in range(n_proteins):
        pid = f"PROT_{i:04d}"
        n_terms = rng.randint(1, min(4, n_go_terms + 1))
        term_indices = rng.choice(n_go_terms, size=n_terms, replace=False)
        protein_go_terms[pid] = {go_ids[j] for j in term_indices}
        center = centers[term_indices].mean(axis=0)
        noise = rng.randn(embed_dim).astype(np.float32) * 0.3
        protein_embeddings[pid] = center + noise
    return protein_embeddings, protein_go_terms, go_ids


def _make_block_data(n_proteins=300, esm_dim=32, fv_dim=6, n_go_terms=10, seed=0):
    """ESM block = pure noise; FV block encodes the GO cluster centres.

    The FV block therefore MUST carry the gain — the assertion that pins the
    block-share attribution.
    """
    rng = np.random.RandomState(seed)
    go_ids = [f"GO:{i:07d}" for i in range(1, n_go_terms + 1)]
    centers = rng.randn(n_go_terms, fv_dim).astype(np.float32)
    protein_embeddings, protein_go_terms = {}, {}
    for i in range(n_proteins):
        pid = f"P{i:04d}"
        k = rng.randint(1, 4)
        idx = rng.choice(n_go_terms, size=k, replace=False)
        protein_go_terms[pid] = {go_ids[j] for j in idx}
        fv = centers[idx].mean(axis=0) + 0.3 * rng.randn(fv_dim).astype(np.float32)
        esm = rng.randn(esm_dim).astype(np.float32)
        protein_embeddings[pid] = np.concatenate([esm, fv])
    return protein_embeddings, protein_go_terms, go_ids


def _fit(embeddings, go_terms, go_ids, config, id="test"):
    clf = GBDTClassifier(
        verbose=False,
        protein_embeddings=embeddings,
        protein_go_terms=go_terms,
        go_ids=go_ids,
        config=config,
        id=id,
    )
    clf.prepare_data()
    clf.build_model()
    clf.train_model()
    return clf


# ---------------------------------------------------------------------------
# GBDTClassifier tests
# ---------------------------------------------------------------------------

class TestGBDTClassifier:
    def test_prepare_data_shapes(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        clf = GBDTClassifier(
            verbose=False, protein_embeddings=embeddings, protein_go_terms=go_terms,
            go_ids=go_ids, config=_make_config(), id="test",
        )
        clf.prepare_data()
        assert clf.X.shape == (200, 64)
        assert clf.y.shape == (200, 10)
        assert clf.esm_dim is None
        assert clf.fv_dim == 0

    def test_full_training_run(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        clf = _fit(embeddings, go_terms, go_ids, _make_config())

        assert clf.best_fmax > 0.3
        assert 0 < clf.best_threshold < 1
        assert clf.best_smin >= 0
        assert clf.y_val_pred.shape == clf.y_val.shape
        assert np.all(clf.y_val_pred >= 0) and np.all(clf.y_val_pred <= 1)
        assert len(clf.model_strings) == 10
        assert all(s is not None for s in clf.model_strings)
        assert clf.constant_cols == {}
        # fixed budget when early stopping is off
        assert clf.best_iterations == [40] * 10
        assert clf.gain_importance.shape == (64,)

    def test_predict_scores_in_range(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        clf = _fit(embeddings, go_terms, go_ids, _make_config())
        scores = clf.predict(clf.X_val[:5])
        assert scores.shape == (5, 10)
        assert np.all(scores >= 0) and np.all(scores <= 1)
        # predict() must reproduce the val predictions made during training
        np.testing.assert_array_almost_equal(scores, clf.y_val_pred[:5], decimal=5)

    def test_degenerate_columns_become_constant_prior(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        go_ids = go_ids + ["GO:9999999", "GO:0000000"]   # never / always annotated
        for pid in go_terms:
            go_terms[pid].add("GO:0000000")
        clf = _fit(embeddings, go_terms, go_ids, _make_config())

        j_always = clf.go_ids.index("GO:0000000")
        j_never = clf.go_ids.index("GO:9999999")
        assert clf.constant_cols[j_always] == 1.0
        assert clf.constant_cols[j_never] == 0.0
        assert clf.model_strings[j_always] is None
        assert clf.model_strings[j_never] is None
        assert len(clf.constant_cols) == 2

        scores = clf.predict(clf.X_val[:4])
        assert np.all(scores[:, j_always] == 1.0)
        assert np.all(scores[:, j_never] == 0.0)
        # the other ten labels still got real boosters
        assert sum(s is not None for s in clf.model_strings) == 10

    def test_save_load_roundtrip(self, tmp_path):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        clf = _fit(embeddings, go_terms, go_ids, _make_config())
        clf.fv_matrix = np.ones((len(clf.protein_ids), 3), dtype=np.float32)
        clf.feature_algorithms = ["PseKRAAC_type_7"]

        path = str(tmp_path / "model.joblib")
        clf.save_model(path)
        art = GBDTClassifier.load_model(path)

        for key in ("format", "model_strings", "constant_cols", "go_ids",
                    "protein_ids", "gain_importance", "split_importance",
                    "hyperparams", "best_iterations", "fv_matrix"):
            assert key in art
        assert art["best_threshold"] == clf.best_threshold
        assert art["protein_ids"] == sorted(embeddings)
        assert art["fv_matrix"].shape == (200, 3)
        assert art["feature_algorithms"] == ["PseKRAAC_type_7"]

        scores_orig = clf.predict(clf.X_val[:3])
        scores_loaded = GBDTClassifier.predict_from_artifact(art, clf.X_val[:3])
        np.testing.assert_array_almost_equal(scores_orig, scores_loaded)

    def test_save_importances_writes_npy(self, tmp_path):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        clf = _fit(embeddings, go_terms, go_ids, _make_config())
        paths = clf.save_importances(prefix=str(tmp_path / "imp"))
        assert len(paths) == 2
        gain = np.load(paths[0])
        assert gain.shape == (64,)
        np.testing.assert_allclose(gain, clf.gain_importance)

    def test_block_gain_shares_favor_signal_block(self):
        embeddings, go_terms, go_ids = _make_block_data(esm_dim=32, fv_dim=6)
        clf = _fit(embeddings, go_terms, go_ids, _make_config(ESM_DIM=32))
        assert clf.esm_dim == 32 and clf.fv_dim == 6

        shares = clf.compute_block_shares()
        assert shares["esm_dim"] == 32 and shares["fv_dim"] == 6
        assert shares["fv_gain_share"] > 0.5            # FV carries the signal
        assert abs(shares["esm_gain_share"] + shares["fv_gain_share"] - 1.0) < 1e-6
        assert shares["fv_gain_per_dim_ratio"] > 1.0     # per-dim, FV >> noise ESM
        assert len(shares["fv_gain_by_feature"]) == 6
        assert 0.0 < shares["fv_split_share"] <= 1.0
        assert clf.gain_importance.shape == (38,)

    def test_no_esm_dim_gives_single_block_shares(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        clf = _fit(embeddings, go_terms, go_ids, _make_config())
        shares = clf.compute_block_shares()
        assert shares["fv_dim"] == 0
        assert shares["fv_gain_share"] is None
        assert shares["fv_gain_by_feature"] == []
        assert abs(shares["esm_gain_share"] - 1.0) < 1e-9

    def test_duplicate_go_ids_are_deduplicated(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        clf = GBDTClassifier(
            verbose=False, protein_embeddings=embeddings, protein_go_terms=go_terms,
            go_ids=go_ids + [go_ids[0]], config=_make_config(), id="test",
        )
        clf.prepare_data()
        assert clf.go_ids == sorted(go_ids)
        assert clf.y.shape == (200, 10)

    def test_esm_dim_out_of_range_raises(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        clf = GBDTClassifier(
            verbose=False, protein_embeddings=embeddings, protein_go_terms=go_terms,
            go_ids=go_ids, config=_make_config(ESM_DIM=999), id="test",
        )
        with pytest.raises(ValueError):
            clf.prepare_data()

    def test_early_stopping_records_best_iterations(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        clf = _fit(
            embeddings, go_terms, go_ids,
            _make_config(GBDT_EARLY_STOPPING_ROUNDS=3, GBDT_N_ESTIMATORS=200),
        )
        assert len(clf.best_iterations) == 10
        assert all(1 <= b <= 200 for b in clf.best_iterations)
        assert any(b < 200 for b in clf.best_iterations)   # something stopped early
        # predictions round-trip with the truncated models too
        art = GBDTClassifier.load_model(clf.save_model("es.joblib"))
        np.testing.assert_array_almost_equal(
            clf.predict(clf.X_val[:3]),
            GBDTClassifier.predict_from_artifact(art, clf.X_val[:3]),
        )

    def test_hyperparameters_change_predictions(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        clf_a = _fit(embeddings, go_terms, go_ids, _make_config(GBDT_N_ESTIMATORS=5), id="a")
        clf_b = _fit(embeddings, go_terms, go_ids, _make_config(GBDT_N_ESTIMATORS=60), id="b")
        assert not np.allclose(clf_a.predict(clf_a.X_val[:5]), clf_b.predict(clf_b.X_val[:5]))

    def test_frequency_and_depth_metrics(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        clf = _fit(embeddings, go_terms, go_ids, _make_config())
        freq = clf.compute_frequency_metrics(clf.y_val, clf.y_val_pred)
        # 200 proteins × 10 terms → every term is rare (<50 train annotations)
        assert "rare_lt50" in freq
        assert freq["rare_lt50"]["n_terms"] == 10
        assert 0 <= freq["rare_lt50"]["fmax"] <= 1
        # no go_dag → no depth breakdown
        assert clf.compute_depth_metrics(clf.y_val, clf.y_val_pred) == {}

    def test_hyperparams_dict_is_json_safe(self):
        embeddings, go_terms, go_ids = _make_synthetic_data()
        clf = _fit(embeddings, go_terms, go_ids, _make_config())
        import json
        hp = clf.hyperparams()
        json.dumps(hp)
        assert hp["n_estimators"] == 40 and hp["num_leaves"] == 7


# ---------------------------------------------------------------------------
# Driver helpers (scripts/compare_knn_vs_multilabel.py)
# ---------------------------------------------------------------------------

_TF_ENV_FLAGS = ("TF_DETERMINISTIC_OPS", "TF_CUDNN_DETERMINISTIC", "PYTHONHASHSEED")


def _load_driver():
    """The comparison logic lives in a script, not the package — load by path.

    The driver enables TensorFlow op determinism at import (env flags read by
    TF at first use). Left in place, that leaks into later tests in the same
    process: unseeded random ops (Dropout in the MLP tests) then raise. Snapshot
    and restore the flags, and switch TF's determinism back off.
    """
    import os
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_knn_vs_multilabel.py"
    )
    saved = {k: os.environ.get(k) for k in _TF_ENV_FLAGS}
    try:
        spec = importlib.util.spec_from_file_location("compare_knn_vs_multilabel", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        try:
            from tensorflow.python.framework import config as _tf_config
            if _tf_config.is_op_determinism_enabled():
                _tf_config.disable_op_determinism()
        except Exception:  # pragma: no cover - TF absent or API moved
            pass
    return mod


class TestDriverHelpers:
    def test_build_raw_combined_embeddings(self):
        drv = _load_driver()
        rng = np.random.RandomState(1)
        esm = {f"P{i:03d}": (rng.randn(8) * 5 + 20).astype(np.float32) for i in range(20)}
        fv = {f"P{i:03d}": (rng.rand(3) * 0.01).astype(np.float32) for i in range(15)}
        fv["P003"][1] = np.nan                          # must be cleaned
        go = {f"P{i:03d}": {"GO:0000001"} for i in range(2, 20)}

        esm_common, combined, common_pids, esm_dim, fv_dim = (
            drv.build_raw_combined_embeddings(esm, fv, go)
        )
        assert common_pids == sorted(set(esm) & set(fv) & set(go))
        assert len(common_pids) == 13
        assert (esm_dim, fv_dim) == (8, 3)
        assert set(esm_common) == set(combined) == set(common_pids)
        for p in common_pids:
            assert combined[p].shape == (11,)
            assert combined[p].dtype == np.float32
            # ESM first, byte-identical, unscaled
            np.testing.assert_array_equal(combined[p][:8], esm[p])
            np.testing.assert_array_equal(esm_common[p], esm[p])
            # FV block raw except NaN → 0
            np.testing.assert_array_equal(combined[p][8:], np.nan_to_num(fv[p], nan=0.0))
        assert combined["P003"][9] == 0.0

    def test_build_raw_combined_embeddings_empty_intersection_raises(self):
        drv = _load_driver()
        with pytest.raises(ValueError):
            drv.build_raw_combined_embeddings(
                {"A": np.zeros(2, np.float32)}, {"B": np.zeros(1, np.float32)}, {"A": {"GO:1"}}
            )

    def test_shuffle_fv_embeddings_is_seeded_permutation(self):
        drv = _load_driver()
        fv = {f"P{i}": np.full(2, i, dtype=np.float32) for i in range(30)}
        s1 = drv.shuffle_fv_embeddings(fv, seed=7)
        s2 = drv.shuffle_fv_embeddings(fv, seed=7)
        assert set(s1) == set(fv)
        # same multiset of vectors, deterministic under the seed, actually moved
        assert sorted(v[0] for v in s1.values()) == sorted(v[0] for v in fv.values())
        assert all(np.array_equal(s1[p], s2[p]) for p in fv)
        assert any(not np.array_equal(s1[p], fv[p]) for p in fv)
