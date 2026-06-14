"""Wave-3 ``scale`` + ``normalize`` coverage — the reachable pure-python lines
wave-1/wave-2 left in ``_scale.py`` and ``_normalize.py``.

``_normalize._ensure_csr_float32``: lines 188-191 (the int64 indptr / indices
-> int32 down-cast branch) only run when the input CSR already has float32-or-
casting data AND int64 index arrays with nnz <= INT32_MAX. Wave-2's e2e inputs
used int32 indices, so those casts never fired. We hit them with a direct call
on a float64 + int64-indexed CSR matrix, plus an already-canonical no-op case.

``_scale``: the remaining "missing" lines are the numba @njit kernel bodies
(_accumulate_stats / _fused_scale_clip) — invisible to coverage.py, already
exercised in test_scanpy_scale_unit.py — and the ``except ImportError`` numba
guard (unreachable while numba is installed). We add a fast-path numeric
parity assertion (mask_obs / obsm delegation already covered in wave-2) to
keep the wrapper exercised, but no new VISIBLE line is recoverable there.

NOT reachable: ``fast_normalize_total``'s ``X.nnz > _INT32_MAX`` overflow
branch (lines 221-228) — needs a >2-billion-nonzero matrix, infeasible in a
unit test; documented in the agent report.

Env recipe: KMP_DUPLICATE_LIB_OK=TRUE NUMBA_THREADING_LAYER=workqueue
NUMBA_NUM_THREADS=14 OMP_NUM_THREADS=2.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from autozyme.scanpy import _normalize as N
from autozyme.scanpy import _scale as S


# --------------------------------------------------------------------------
# _ensure_csr_float32 — int64 index down-cast branch (lines 188-191)
# --------------------------------------------------------------------------

def test_ensure_csr_float32_downcasts_int64_indices():
    # Must be float32 ALREADY so the `astype(float32)` at line 185 is skipped
    # (scipy's astype silently downcasts indices to int32, which would hide the
    # explicit index-cast branch). With float32 data + int64 indices, lines
    # 188-191 are the only place the int32 down-cast happens.
    X = sparse.csr_matrix(np.array([[1.0, 0.0, 2.0],
                                    [0.0, 3.0, 0.0]], dtype=np.float32))
    X.indptr = X.indptr.astype(np.int64)
    X.indices = X.indices.astype(np.int64)
    assert X.dtype == np.float32 and X.indptr.dtype == np.int64
    out = N._ensure_csr_float32(X)
    assert out.dtype == np.float32
    assert out.indptr.dtype == np.int32   # line 189 ran
    assert out.indices.dtype == np.int32  # line 191 ran
    np.testing.assert_allclose(out.toarray(), X.toarray())


def test_ensure_csr_float32_already_int32_noop_on_indices():
    # float32 + int32 already -> the dtype != int32 branches are skipped.
    X = sparse.csr_matrix(np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32))
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    out = N._ensure_csr_float32(X)
    assert out.indptr.dtype == np.int32
    assert out.indices.dtype == np.int32


def test_ensure_csr_float32_float64_int32_only_casts_data():
    # float64 data but int32 indices -> only the data cast runs; index branches
    # are no-ops (already int32).
    X = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    out = N._ensure_csr_float32(X)
    assert out.dtype == np.float32
    assert out.indptr.dtype == np.int32


def test_ensure_csr_float32_from_dense_array():
    # Non-sparse input -> wrapped to CSR first (line 182).
    X = np.array([[1, 2], [3, 4]], dtype=np.float64)
    out = N._ensure_csr_float32(X)
    assert sparse.isspmatrix_csr(out)
    assert out.dtype == np.float32


def test_ensure_csr_float32_from_csc():
    # Non-CSR sparse -> tocsr() branch (line 184).
    X = sparse.csc_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64))
    out = N._ensure_csr_float32(X)
    assert sparse.isspmatrix_csr(out)
    assert out.dtype == np.float32


# --------------------------------------------------------------------------
# fast_scale — fast-path numeric parity (keeps the wrapper + kernel exercised)
# --------------------------------------------------------------------------

def test_fast_scale_csr_parity_with_upstream():
    pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")
    import scanpy as sc
    import autozyme

    rng = np.random.default_rng(7)
    dense = (rng.random((40, 10)) * 5.0).astype(np.float32)
    a_fast = ad.AnnData(sparse.csr_matrix(dense))
    a_orig = a_fast.copy()
    # Fast path (direct call).
    S.fast_scale(a_fast, max_value=10)
    with autozyme.disabled():
        sc.pp.scale(a_orig, max_value=10)
    fast_X = a_fast.X.toarray() if sparse.issparse(a_fast.X) else np.asarray(a_fast.X)
    orig_X = a_orig.X.toarray() if sparse.issparse(a_orig.X) else np.asarray(a_orig.X)
    np.testing.assert_allclose(fast_X, orig_X, rtol=1e-3, atol=1e-3)


def test_fast_scale_n_obs_one_skips_bessel():
    # n_obs == 1 -> the `if n_obs > 1` Bessel correction is skipped (the
    # else-of-that-branch). std collapses to 1.0 everywhere (var == 0).
    ad = pytest.importorskip("anndata")
    X = sparse.csr_matrix(np.array([[2.0, 4.0, 6.0]], dtype=np.float32))
    a = ad.AnnData(X)
    S.fast_scale(a, max_value=None)
    # Single row: mean == the row, (x - mean) == 0 -> all zeros after scaling.
    out = a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)
    np.testing.assert_allclose(out, np.zeros((1, 3)), atol=1e-6)
    np.testing.assert_allclose(a.var["std"].values, [1.0, 1.0, 1.0])


# --------------------------------------------------------------------------
# fast_normalize_total — already-CSR-float32 path (no dtype warning, no cast)
# --------------------------------------------------------------------------

def test_fast_normalize_canonical_input_no_warning():
    # Canonical float32 + int32 CSR -> _maybe_warn_about_dtype stays silent and
    # _ensure_csr_float32 is a near no-op. Confirms the happy fast path.
    ad = pytest.importorskip("anndata")
    import warnings

    rng = np.random.default_rng(8)
    dense = rng.integers(0, 6, size=(30, 12)).astype(np.float32)
    dense[:, 0] += 1
    X = sparse.csr_matrix(dense)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    a = ad.AnnData(X)
    # Reset the session-global flag so a stray earlier warn doesn't mask this.
    N._DTYPE_WARNED = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        N.fast_normalize_total(a, target_sum=1e4)  # must not warn
    sums = np.asarray(a.X.sum(axis=1)).ravel()
    np.testing.assert_allclose(sums[sums > 0], 1e4, rtol=1e-4)
