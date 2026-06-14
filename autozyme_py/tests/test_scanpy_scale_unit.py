"""Unit tests for autozyme.scanpy._scale.

Covers the numba kernels (_accumulate_stats, _fused_scale_clip) directly,
the back-compat alias, and the fast_scale dispatch + correctness path.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from autozyme.scanpy import _scale as S


pytestmark = pytest.mark.skipif(not S.HAS_NUMBA, reason="numba not installed")


# --------------------------------------------------------------------------
# _accumulate_stats — per-column sum + sum-of-squares from CSR (indices,data)
# --------------------------------------------------------------------------

def test_accumulate_stats_matches_numpy():
    X = np.array([[1.0, 0.0, 2.0],
                  [3.0, 4.0, 0.0]], dtype=np.float32)
    csr = sparse.csr_matrix(X)
    sums, sumsq = S._accumulate_stats(csr.indices.astype(np.int32),
                                      csr.data.astype(np.float32), 3)
    np.testing.assert_allclose(sums, X.sum(axis=0))
    np.testing.assert_allclose(sumsq, (X * X).sum(axis=0))


def test_accumulate_stats_empty():
    sums, sumsq = S._accumulate_stats(
        np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), 4)
    np.testing.assert_array_equal(sums, np.zeros(4))
    np.testing.assert_array_equal(sumsq, np.zeros(4))


def test_accumulate_stats_repeated_column():
    # Two entries in the same column accumulate.
    indices = np.array([0, 0, 1], dtype=np.int32)
    data = np.array([2.0, 3.0, 5.0], dtype=np.float32)
    sums, sumsq = S._accumulate_stats(indices, data, 2)
    np.testing.assert_allclose(sums, [5.0, 5.0])
    np.testing.assert_allclose(sumsq, [4.0 + 9.0, 25.0])


# --------------------------------------------------------------------------
# _fused_scale_clip — (x - mean) * inv_std, clipped to [-max, +max]
# --------------------------------------------------------------------------

def test_fused_scale_clip_basic_standardization():
    x = np.array([[0.0, 10.0], [2.0, 20.0]], dtype=np.float32)
    mean = np.array([1.0, 15.0], dtype=np.float32)
    inv_std = np.array([0.5, 0.1], dtype=np.float32)
    S._fused_scale_clip(x, mean, inv_std, np.float32(np.inf))
    expected = np.array([[(0 - 1) * 0.5, (10 - 15) * 0.1],
                         [(2 - 1) * 0.5, (20 - 15) * 0.1]], dtype=np.float32)
    np.testing.assert_allclose(x, expected, rtol=1e-6)


def test_fused_scale_clip_symmetric_clip():
    # The 2026-05-21 fix: clip is two-sided. Large positive and negative
    # standardized values both get capped.
    x = np.array([[100.0, -100.0]], dtype=np.float32)
    mean = np.array([0.0, 0.0], dtype=np.float32)
    inv_std = np.array([1.0, 1.0], dtype=np.float32)
    S._fused_scale_clip(x, mean, inv_std, np.float32(3.0))
    np.testing.assert_allclose(x, [[3.0, -3.0]])


def test_fused_scale_clip_no_clip_within_range():
    x = np.array([[1.5, -2.0]], dtype=np.float32)
    mean = np.array([0.0, 0.0], dtype=np.float32)
    inv_std = np.array([1.0, 1.0], dtype=np.float32)
    S._fused_scale_clip(x, mean, inv_std, np.float32(10.0))
    np.testing.assert_allclose(x, [[1.5, -2.0]])


def test_alias_is_same_function():
    # _fused_scale_clip_upper is the back-compat alias for the now-symmetric kernel.
    assert S._fused_scale_clip_upper is S._fused_scale_clip


# --------------------------------------------------------------------------
# _orig_scale — returns sc.pp.scale (or its __autozyme_original__)
# --------------------------------------------------------------------------

def test_orig_scale_returns_callable():
    pytest.importorskip("scanpy")
    fn = S._orig_scale()
    assert callable(fn)


# --------------------------------------------------------------------------
# fast_scale — dispatch logic (delegations) + numeric correctness
# --------------------------------------------------------------------------

def test_fast_scale_zyme_false_delegates(monkeypatch):
    called = {}

    def fake():
        def _fn(data, **kw):
            called["hit"] = True
            return "ORIG"
        return _fn

    monkeypatch.setattr(S, "_orig_scale", fake)
    assert S.fast_scale("DATA", zyme=False) == "ORIG"
    assert called["hit"]


def test_fast_scale_non_anndata_delegates(monkeypatch):
    monkeypatch.setattr(S, "_orig_scale", lambda: (lambda data, **kw: "ORIG"))
    # A bare ndarray is not an AnnData -> delegate.
    assert S.fast_scale(np.zeros((2, 2), dtype=np.float32)) == "ORIG"


def test_fast_scale_zero_center_false_delegates(monkeypatch):
    ad = pytest.importorskip("anndata")
    monkeypatch.setattr(S, "_orig_scale", lambda: (lambda data, **kw: "ORIG"))
    a = ad.AnnData(sparse.csr_matrix(np.eye(3, dtype=np.float32)))
    assert S.fast_scale(a, zero_center=False) == "ORIG"


def test_fast_scale_layer_delegates(monkeypatch):
    ad = pytest.importorskip("anndata")
    monkeypatch.setattr(S, "_orig_scale", lambda: (lambda data, **kw: "ORIG"))
    a = ad.AnnData(sparse.csr_matrix(np.eye(3, dtype=np.float32)))
    assert S.fast_scale(a, layer="foo") == "ORIG"


def test_fast_scale_dense_X_delegates(monkeypatch):
    ad = pytest.importorskip("anndata")
    monkeypatch.setattr(S, "_orig_scale", lambda: (lambda data, **kw: "ORIG"))
    a = ad.AnnData(np.ones((3, 3), dtype=np.float32))  # dense .X, not CSR
    assert S.fast_scale(a) == "ORIG"


def test_fast_scale_csr_numeric_matches_zscore():
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(3)
    X = rng.random((40, 6), dtype=np.float32) * 5.0
    csr = sparse.csr_matrix(X)
    a = ad.AnnData(csr)
    S.fast_scale(a, max_value=None)  # in-place
    # Reference: per-gene z-score with ddof=1 (n/(n-1) correction).
    n = X.shape[0]
    mean = X.mean(axis=0)
    var = X.var(axis=0) * n / (n - 1)
    std = np.sqrt(var)
    std[std == 0] = 1.0
    ref = (X - mean) / std
    np.testing.assert_allclose(a.X, ref, rtol=1e-4, atol=1e-4)
    # var annotations recorded.
    np.testing.assert_allclose(a.var["mean"].values, mean, rtol=1e-4)
    np.testing.assert_allclose(a.var["var"].values, var, rtol=1e-4)


def test_fast_scale_copy_returns_new_and_clips():
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(5)
    X = rng.random((30, 5), dtype=np.float32) * 100.0
    a = ad.AnnData(sparse.csr_matrix(X))
    before = a.X.copy()
    out = S.fast_scale(a, max_value=2.0, copy=True)
    assert out is not None and out is not a
    # Input untouched under copy=True.
    np.testing.assert_allclose(a.X.toarray(), before.toarray())
    # Output clipped to [-2, 2].
    assert out.X.max() <= 2.0 + 1e-5
    assert out.X.min() >= -2.0 - 1e-5
