"""Regression tests for the KNN comparison's ESM-scaling fix
(``scripts/compare_knn_esm_vs_knn_combined.py``).

These pin the scientific invariant behind the three-arm design: the scaled-ESM
baseline (arm 2) and the ESM half of the combined vector (arm 3) are produced by
ONE StandardScaler, fit on the training fold only. If they ever diverge, the
baseline-vs-combined delta would again conflate "added classical features" with
"standardised the ESM block" — exactly the confound this fix removes.
"""

import importlib.util
from pathlib import Path

import numpy as np

# The comparison logic lives in a script (scripts/), not an installed package,
# so load it by file path.
_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_knn_esm_vs_knn_combined.py"
)
_SPEC = importlib.util.spec_from_file_location("compare_knn_esm_vs_knn_combined", _SCRIPT)
cmp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cmp)


def _toy_data(n=20, esm_dim=8, fv_dim=3, seed=0):
    """ESM on a large offset scale, FV on a tiny scale — mimics the real
    magnitude mismatch that motivates per-block standardisation."""
    rng = np.random.RandomState(seed)
    pids = [f"P{i:03d}" for i in range(n)]
    protein_embeddings = {
        p: (rng.randn(esm_dim) * 5 + 20).astype(np.float32) for p in pids
    }
    fv_dict = {p: (rng.rand(fv_dim) * 0.01).astype(np.float32) for p in pids}
    return pids, protein_embeddings, fv_dict


def test_combined_esm_block_equals_scaled_esm():
    """The first esm_dim columns of `combined` must be byte-identical to the
    scaled-ESM baseline vector for every protein."""
    pids, emb, fv = _toy_data()
    scaled_esm, combined, _, _, common = cmp.build_scaled_representations(
        emb, fv, train_pids=pids[:16]
    )
    esm_dim = next(iter(emb.values())).shape[0]
    for p in common:
        np.testing.assert_array_equal(combined[p][:esm_dim], scaled_esm[p])


def test_scaler_fit_on_train_fold_only():
    """The ESM scaler's mean must match the TRAIN rows, not all rows
    (no validation leakage)."""
    pids, emb, fv = _toy_data()
    train_pids = pids[:16]
    _, _, esm_scaler, _, _ = cmp.build_scaled_representations(
        emb, fv, train_pids=train_pids
    )
    train_mean = np.mean([emb[p] for p in train_pids], axis=0)
    all_mean = np.mean([emb[p] for p in pids], axis=0)
    np.testing.assert_allclose(esm_scaler.mean_, train_mean, rtol=1e-5)
    # If it had leaked, mean_ would equal the all-protein mean instead.
    assert not np.allclose(esm_scaler.mean_, all_mean)


def test_scaled_esm_is_standardised_on_train():
    pids, emb, fv = _toy_data()
    train_pids = pids[:16]
    scaled_esm, _, _, _, _ = cmp.build_scaled_representations(
        emb, fv, train_pids=train_pids
    )
    train_matrix = np.vstack([scaled_esm[p] for p in train_pids])
    np.testing.assert_allclose(train_matrix.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(train_matrix.std(axis=0), 1.0, atol=1e-4)


def test_combined_dim_is_esm_plus_fv():
    pids, emb, fv = _toy_data(esm_dim=8, fv_dim=3)
    scaled_esm, combined, _, _, _ = cmp.build_scaled_representations(
        emb, fv, train_pids=pids[:16]
    )
    assert next(iter(combined.values())).shape[0] == 8 + 3
    assert next(iter(scaled_esm.values())).shape[0] == 8
