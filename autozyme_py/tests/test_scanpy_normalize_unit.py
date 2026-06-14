"""Unit tests for autozyme.scanpy._normalize.

Tests the numba CSR kernels directly on small fixed inputs (no AnnData),
plus the pure-python helpers (_n_threads, _maybe_warn_about_dtype,
_ensure_csr_float32) and the fast_normalize_total / fast_log1p dispatch logic.

Env recipe required (see BRIEFING): KMP_DUPLICATE_LIB_OK=TRUE
NUMBA_THREADING_LAYER=workqueue OMP_NUM_THREADS=2.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from scipy import sparse

from autozyme.scanpy import _normalize as N


# --------------------------------------------------------------------------
# numba kernels (pure functions on CSR data/indptr)
# --------------------------------------------------------------------------

def _csr(rows):
    """Build a tiny float32 CSR matrix from a list of dense rows."""
    m = sparse.csr_matrix(np.asarray(rows, dtype=np.float32))
    m.indptr = m.indptr.astype(np.int32)
    m.indices = m.indices.astype(np.int32)
    return m


def test_row_sums_matches_scipy_sum():
    X = _csr([[1, 2, 0], [0, 0, 3], [4, 5, 6]])
    sums = np.zeros(X.shape[0], dtype=np.float32)
    N._row_sums(X.data, X.indptr, sums)
    ref = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    np.testing.assert_allclose(sums, ref, rtol=1e-6)
    np.testing.assert_allclose(sums, [3.0, 3.0, 15.0], rtol=1e-6)


def test_row_sums_empty_row_is_zero():
    # Middle row has no nonzeros.
    X = _csr([[1, 2], [0, 0], [3, 0]])
    sums = np.zeros(3, dtype=np.float32)
    N._row_sums(X.data, X.indptr, sums)
    np.testing.assert_allclose(sums, [3.0, 0.0, 3.0])


def test_row_sums_single_row():
    X = _csr([[2, 3, 4]])
    sums = np.zeros(1, dtype=np.float32)
    N._row_sums(X.data, X.indptr, sums)
    assert sums[0] == pytest.approx(9.0)


def test_scale_only_scales_to_target():
    X = _csr([[1, 1, 0], [0, 2, 2]])  # row sums 2, 4
    sums = np.array([2.0, 4.0], dtype=np.float32)
    target = np.float32(10.0)
    N._scale_only(X.data, X.indptr, sums, target)
    # Each row should now sum to target.
    new = np.zeros(2, dtype=np.float32)
    N._row_sums(X.data, X.indptr, new)
    np.testing.assert_allclose(new, [10.0, 10.0], rtol=1e-5)


def test_scale_only_skips_zero_sum_rows():
    # Row with sum 0 must be left untouched (no div-by-zero).
    X = _csr([[0, 0], [1, 1]])
    sums = np.array([0.0, 2.0], dtype=np.float32)
    before = X.data.copy()
    N._scale_only(X.data, X.indptr, sums, np.float32(10.0))
    # First row has no data; second scaled. Confirm no NaN/inf leaked.
    assert np.all(np.isfinite(X.data))
    # The only datum (row 1) was scaled 10/2 = 5x.
    np.testing.assert_allclose(np.sort(X.data), np.sort(before * 5.0))


def test_fused_normalize_only_equals_rowsum_then_scale():
    rows = [[1, 3, 0], [0, 0, 5], [2, 2, 2]]
    X1 = _csr(rows)
    X2 = _csr(rows)
    target = np.float32(1e4)
    # Fused path.
    N._fused_normalize_only(X1.data, X1.indptr, target)
    # Two-step reference.
    sums = np.zeros(3, dtype=np.float32)
    N._row_sums(X2.data, X2.indptr, sums)
    N._scale_only(X2.data, X2.indptr, sums, target)
    np.testing.assert_allclose(X1.data, X2.data, rtol=1e-5)


def test_fused_normalize_only_zero_row_unchanged():
    X = _csr([[0, 0, 0], [1, 1, 0]])
    # all-zero first row has no stored data; second row sum 2 -> scaled.
    N._fused_normalize_only(X.data, X.indptr, np.float32(4.0))
    new = np.zeros(2, dtype=np.float32)
    N._row_sums(X.data, X.indptr, new)
    np.testing.assert_allclose(new, [0.0, 4.0], rtol=1e-5)


def test_log1p_inplace_matches_numpy():
    data = np.array([0.0, 1.0, 9.0, 1e3, 0.5], dtype=np.float32)
    ref = np.log1p(data.astype(np.float64)).astype(np.float32)
    work = data.copy()
    N._log1p_inplace(work)
    np.testing.assert_allclose(work, ref, rtol=1e-5, atol=1e-6)


def test_log1p_inplace_empty():
    data = np.empty(0, dtype=np.float32)
    N._log1p_inplace(data)  # must not raise
    assert data.shape == (0,)


# --------------------------------------------------------------------------
# _n_threads — env var resolution
# --------------------------------------------------------------------------

def test_n_threads_reads_zyme_threads(monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
              "SCANPY_TURBO_THREADS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ZYME_THREADS", "5")
    assert N._n_threads() == 5


def test_n_threads_precedence_order(monkeypatch):
    monkeypatch.setenv("ZYME_THREADS", "3")
    monkeypatch.setenv("AUTOZYME_THREADS", "7")
    monkeypatch.setenv("OMP_NUM_THREADS", "9")
    # ZYME_THREADS wins (first in the precedence tuple).
    assert N._n_threads() == 3


def test_n_threads_malformed_falls_through(monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
              "SCANPY_TURBO_THREADS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ZYME_THREADS", "not-a-number")
    monkeypatch.setenv("AUTOZYME_THREADS", "4")
    assert N._n_threads() == 4


def test_n_threads_nonpositive_skipped(monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
              "SCANPY_TURBO_THREADS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ZYME_THREADS", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    assert N._n_threads() == 6


def test_n_threads_falls_back_to_cpu_count(monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
              "SCANPY_TURBO_THREADS"):
        monkeypatch.delenv(v, raising=False)
    n = N._n_threads()
    assert n >= 1
    assert n == (os.cpu_count() or 8)


# --------------------------------------------------------------------------
# _ensure_csr_float32
# --------------------------------------------------------------------------

def test_ensure_csr_float32_from_dense():
    X = np.array([[1, 2], [3, 4]], dtype=np.float64)
    out = N._ensure_csr_float32(X)
    assert sparse.isspmatrix_csr(out)
    assert out.dtype == np.float32
    assert out.indptr.dtype == np.int32
    assert out.indices.dtype == np.int32
    np.testing.assert_allclose(out.toarray(), X)


def test_ensure_csr_float32_from_csc():
    X = sparse.csc_matrix(np.array([[1, 0], [0, 2]], dtype=np.float64))
    out = N._ensure_csr_float32(X)
    assert sparse.isspmatrix_csr(out)
    assert out.dtype == np.float32


def test_ensure_csr_float32_already_canonical_keeps_dtypes():
    X = _csr([[1, 0], [0, 2]])
    out = N._ensure_csr_float32(X)
    assert out.dtype == np.float32
    assert out.indptr.dtype == np.int32
    assert out.indices.dtype == np.int32


# --------------------------------------------------------------------------
# _maybe_warn_about_dtype  (session-global flag)
# --------------------------------------------------------------------------

def test_maybe_warn_about_dtype_warns_once_for_float64(monkeypatch):
    monkeypatch.setattr(N, "_DTYPE_WARNED", False, raising=False)
    X = sparse.csr_matrix(np.array([[1.0, 2.0]], dtype=np.float64))
    with pytest.warns(RuntimeWarning, match="non-canonical dtypes"):
        N._maybe_warn_about_dtype(X)
    # Second call is a no-op (already warned) — no warning expected.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        N._maybe_warn_about_dtype(X)  # must not raise (no warning)


def test_maybe_warn_about_dtype_silent_for_canonical(monkeypatch):
    monkeypatch.setattr(N, "_DTYPE_WARNED", False, raising=False)
    X = _csr([[1, 2], [3, 4]])  # float32 + int32 already
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        N._maybe_warn_about_dtype(X)  # canonical -> no warning
    # Flag stays False because nothing was warned.
    assert N._DTYPE_WARNED is False


def test_maybe_warn_about_dtype_int64_indptr_triggers(monkeypatch):
    monkeypatch.setattr(N, "_DTYPE_WARNED", False, raising=False)
    X = sparse.csr_matrix(np.array([[1, 2], [3, 4]], dtype=np.float32))
    X.indptr = X.indptr.astype(np.int64)
    X.indices = X.indices.astype(np.int64)
    with pytest.warns(RuntimeWarning):
        N._maybe_warn_about_dtype(X)


# --------------------------------------------------------------------------
# fast_normalize_total / fast_log1p  — dispatch + correctness via AnnData
# --------------------------------------------------------------------------

def _tiny_adata():
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(0)
    X = sparse.random(20, 15, density=0.4, format="csr",
                      dtype=np.float32, random_state=rng).astype(np.float32)
    X.data = rng.integers(1, 10, size=X.nnz).astype(np.float32)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    return ad.AnnData(X)


def test_fast_normalize_total_inplace_returns_none():
    pytest.importorskip("anndata")
    a = _tiny_adata()
    out = N.fast_normalize_total(a, target_sum=1e4)
    assert out is None
    sums = np.asarray(a.X.sum(axis=1)).ravel()
    # Rows with any data should sum to target.
    np.testing.assert_allclose(sums[sums > 0], 1e4, rtol=1e-4)


def test_fast_normalize_total_copy_returns_new_anndata():
    pytest.importorskip("anndata")
    a = _tiny_adata()
    before = a.X.copy()
    out = N.fast_normalize_total(a, target_sum=1e4, copy=True)
    assert out is not None
    assert out is not a
    # Original is unmodified under copy=True.
    np.testing.assert_allclose(a.X.toarray(), before.toarray())


def test_fast_normalize_total_median_target_when_none():
    pytest.importorskip("anndata")
    a = _tiny_adata()
    pre_sums = np.asarray(a.X.sum(axis=1)).ravel()
    median_target = float(np.median(pre_sums[pre_sums > 0]))
    N.fast_normalize_total(a)  # target_sum=None -> median of row sums
    post = np.asarray(a.X.sum(axis=1)).ravel()
    np.testing.assert_allclose(post[post > 0], median_target, rtol=1e-3)


def test_fast_normalize_total_zyme_false_delegates(monkeypatch):
    # zyme=False short-circuits to _orig; intercept via monkeypatch so we
    # don't need a real registered patch.
    called = {}

    def fake_orig(name):
        def _fn(adata, **kw):
            called["name"] = name
            called["kw"] = kw
            return "ORIG"
        return _fn

    monkeypatch.setattr(N, "_orig", fake_orig)
    out = N.fast_normalize_total("ADATA", target_sum=7, zyme=False)
    assert out == "ORIG"
    assert called["name"] == "normalize_total"


def test_fast_normalize_total_inplace_false_delegates(monkeypatch):
    called = {}

    def fake_orig(name):
        def _fn(*a, **k):
            called["hit"] = True
            return "ORIG"
        return _fn

    monkeypatch.setattr(N, "_orig", fake_orig)
    out = N.fast_normalize_total("ADATA", inplace=False)
    assert out == "ORIG"
    assert called.get("hit")


def test_fast_normalize_total_extra_kwargs_delegate(monkeypatch):
    monkeypatch.setattr(N, "_orig", lambda name: (lambda *a, **k: "ORIG"))
    out = N.fast_normalize_total("ADATA", exclude_highly_expressed=True)
    assert out == "ORIG"


def test_fast_log1p_matches_numpy_log1p():
    pytest.importorskip("anndata")
    a = _tiny_adata()
    ref = np.log1p(a.X.toarray())
    out = N.fast_log1p(a)
    assert out is None
    np.testing.assert_allclose(a.X.toarray(), ref, rtol=1e-5, atol=1e-6)
    assert a.uns["log1p"] == {"base": None}


def test_fast_log1p_dense_path():
    ad = pytest.importorskip("anndata")
    # Compute reference BEFORE the in-place mutation (AnnData shares the array).
    ref = np.log1p(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    a = ad.AnnData(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))
    N.fast_log1p(a)
    np.testing.assert_allclose(a.X, ref, rtol=1e-6)


def test_fast_log1p_copy_returns_new():
    pytest.importorskip("anndata")
    a = _tiny_adata()
    before = a.X.copy()
    out = N.fast_log1p(a, copy=True)
    assert out is not None and out is not a
    np.testing.assert_allclose(a.X.toarray(), before.toarray())


def test_fast_log1p_zyme_false_delegates(monkeypatch):
    monkeypatch.setattr(N, "_orig", lambda name: (lambda *a, **k: "ORIG"))
    assert N.fast_log1p("ADATA", zyme=False) == "ORIG"


def test_fast_log1p_kwargs_delegate(monkeypatch):
    monkeypatch.setattr(N, "_orig", lambda name: (lambda *a, **k: "ORIG"))
    assert N.fast_log1p("ADATA", base=2) == "ORIG"
