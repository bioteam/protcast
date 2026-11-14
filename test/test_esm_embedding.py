"""Tests for the shared ESM-C pooling module
(``protcast.preprocessing.esm_embedding``).

The pooling recipe in that module is shared by the training-set embedder
(``make_esm_embeddings.py``) and both inference-time embedders, so these tests
pin the contract that keeps the train/inference feature spaces identical:
the pooling maths, BOS/EOS special-token stripping, and hidden-layer selection.

``torch`` is not installed on CPU-only dev boxes, so the whole module is skipped
there and runs on the GPU nodes / CI where torch is present.
"""

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from protcast.preprocessing.esm_embedding import (  # noqa: E402
    pool_embeddings,
    select_layer_representation,
    reduce_forward_output,
    POOLING_STRATEGIES,
)


def _fake_output(embeddings, hidden_states=None):
    """Stand-in for an ESMCOutput: just needs .embeddings / .hidden_states."""
    return types.SimpleNamespace(
        embeddings=embeddings, hidden_states=hidden_states
    )


# ── pool_embeddings ────────────────────────────────────────────────────────

def test_pool_mean_shape_and_values():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (L=3, D=2)
    out = pool_embeddings(x, "mean")
    assert tuple(out.shape) == (2,)
    assert torch.allclose(out, torch.tensor([3.0, 4.0]))


def test_pool_mean_max_std_shape_and_values():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (L=3, D=2)
    out = pool_embeddings(x, "mean_max_std")
    assert tuple(out.shape) == (6,)  # 3 * D
    expected = torch.cat(
        [
            torch.tensor([3.0, 4.0]),          # mean
            torch.tensor([5.0, 6.0]),          # max
            x.std(dim=0, unbiased=False),      # population std
        ]
    )
    assert torch.allclose(out, expected)


def test_pool_std_defined_for_single_residue():
    # unbiased=False => std of a length-1 sequence is 0, not NaN.
    x = torch.tensor([[2.0, 9.0]])  # (L=1, D=2)
    out = pool_embeddings(x, "mean_max_std")
    assert not torch.isnan(out).any()
    assert torch.allclose(out[4:], torch.zeros(2))  # std block is zero


def test_pool_unknown_strategy_raises():
    with pytest.raises(ValueError):
        pool_embeddings(torch.zeros(2, 2), "bogus")


def test_pooling_strategies_constant_exposed():
    assert "mean" in POOLING_STRATEGIES
    assert "mean_max_std" in POOLING_STRATEGIES


# ── select_layer_representation ──────────────────────────────────────────────

def test_select_layer_none_returns_final_embeddings():
    emb = torch.randn(1, 5, 4)
    hs = torch.randn(3, 1, 5, 4)
    assert torch.equal(select_layer_representation(_fake_output(emb, hs), None), emb)


def test_select_layer_indexes_hidden_states():
    emb = torch.randn(1, 5, 4)
    hs = torch.randn(3, 1, 5, 4)
    out = _fake_output(emb, hs)
    assert torch.equal(select_layer_representation(out, 0), hs[0])
    assert torch.equal(select_layer_representation(out, 2), hs[2])
    assert torch.equal(select_layer_representation(out, -1), hs[-1])


def test_select_layer_out_of_range_raises():
    out = _fake_output(torch.randn(1, 5, 4), torch.randn(3, 1, 5, 4))
    with pytest.raises(ValueError):
        select_layer_representation(out, 3)
    with pytest.raises(ValueError):
        select_layer_representation(out, -4)


def test_select_layer_missing_hidden_states_raises():
    out = _fake_output(torch.randn(1, 5, 4), hidden_states=None)
    with pytest.raises(ValueError):
        select_layer_representation(out, 0)


# ── reduce_forward_output (strip + layer + pool end to end) ──────────────────

def test_reduce_strips_special_tokens():
    # BOS/EOS carry extreme activations; stripping must exclude them.
    residues = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])  # mean = 2
    bos = torch.full((1, 2), 100.0)
    eos = torch.full((1, 2), -100.0)
    seq = torch.cat([bos, residues, eos], dim=0).unsqueeze(0)  # (1, L+2, D)
    out = _fake_output(seq)

    stripped = reduce_forward_output(out, pooling="mean", strip_special_tokens=True)
    kept = reduce_forward_output(out, pooling="mean", strip_special_tokens=False)

    assert np.allclose(stripped, [2.0, 2.0])       # residues only
    assert not np.allclose(kept, [2.0, 2.0])       # BOS/EOS drag the mean


def test_reduce_layer_selection_changes_output():
    emb = torch.zeros(1, 5, 4)
    # hidden layer i is filled with the constant i.
    hs = torch.stack([torch.full((1, 5, 4), float(i)) for i in range(3)], dim=0)
    out = _fake_output(emb, hs)

    assert np.allclose(reduce_forward_output(out, layer=None), 0.0)
    assert np.allclose(reduce_forward_output(out, layer=0), 0.0)
    assert np.allclose(reduce_forward_output(out, layer=2), 2.0)


def test_reduce_too_short_to_strip_raises():
    seq = torch.randn(1, 2, 4)  # BOS + EOS only, no residues
    with pytest.raises(ValueError):
        reduce_forward_output(_fake_output(seq), strip_special_tokens=True)


def test_reduce_mean_max_std_dim():
    seq = torch.randn(1, 6, 4)  # L+2 = 6, D = 4
    r = reduce_forward_output(
        _fake_output(seq), pooling="mean_max_std", strip_special_tokens=True
    )
    assert r.shape == (12,)  # 3 * D
