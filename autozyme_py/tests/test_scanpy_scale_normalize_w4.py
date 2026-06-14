"""Wave-4 ``scale`` + ``normalize`` coverage / hardening.

For ``_scale.py`` the reachable pure-python wrapper (``fast_scale``) is already
100% line-covered by waves 1-3; the only "missing" lines are the
``except ImportError`` numba guard (16-17) and the ``@njit`` kernel bodies
(``_accumulate_stats`` / ``_fused_scale_clip``, 29-36 / 50-59) which coverage.py
cannot see (exercised in test_scanpy_scale_unit.py).

For ``_normalize.py`` the only genuinely-reachable line still missing after
waves 1-3 is the ``X.nnz > _INT32_MAX`` int64-overflow delegation branch
(lines 221-228). It normally needs a >2.1-billion-nonzero matrix, but we drive
it deterministically by monkeypatching the module-level ``_INT32_MAX`` constant
down to a tiny value so a small CSR trips the branch — the branch then delegates
to upstream ``normalize_total`` (still correct). We cover it for both
``copy=False`` and ``copy=True`` (the two return-contract arms). The remaining
"missing" lines are the four ``@njit`` kernels (44-51 / 63-72 / 84-95 / 107-108).

The rest of these tests harden untested dtype / format / param combinations
(asserting parity vs the unpatched original) for both modules.

Env recipe: KMP_DUPLICATE_LIB_OK=TRUE NUMBA_THREADING_LAYER=workqueue
NUMBA_NUM_THREADS=14 OMP_NUM_THREADS=2.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import sparse

from autozyme.scanpy import _normalize as N
from autozyme.scanpy import _scale as S


def _dense(adata):
    X = adata.X
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


# ==========================================================================
# fast_normalize_total — the nnz > _INT32_MAX overflow branch (221-228)
# ==========================================================================

def test_normalize_total_nnz_overflow_inplace(monkeypatch):
    # Force the int64-overflow branch by shrinking the threshold so a small CSR
    # has nnz > _INT32_MAX. The wrapper delegates to upstream in-place and
    # returns None (copy=False). Patch _orig so we don't need a registered patch.
    ad = pytest.importorskip("anndata")
    called = {}

    def fake_orig(name):
        def _fn(adata, **kw):
            called["name"] = name
            called["copy"] = kw.get("copy")
            called["inplace"] = kw.get("inplace")
            return None
        return _fn

    monkeypatch.setattr(N, "_orig", fake_orig)
    monkeypatch.setattr(N, "_INT32_MAX", 1)  # any nnz >= 2 now "overflows"

    rng = np.random.default_rng(0)
    dense = rng.integers(0, 5, size=(8, 6)).astype(np.float32)
    dense[:, 0] += 1
    X = sparse.csr_matrix(dense)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    a = ad.AnnData(X)
    assert X.nnz > N._INT32_MAX  # precondition: branch will fire

    out = N.fast_normalize_total(a, target_sum=1e4)
    assert out is None
    # Upstream was invoked in-place with copy=False.
    assert called["name"] == "normalize_total"
    assert called["copy"] is False and called["inplace"] is True


def test_normalize_total_nnz_overflow_copy_returns_new(monkeypatch):
    # Same overflow branch but copy=True -> the wrapper copies adata first, then
    # delegates in-place, then returns the local handle (line 228 `return adata`).
    ad = pytest.importorskip("anndata")

    def fake_orig(name):
        def _fn(adata, **kw):
            return None  # pretend upstream normalized in place
        return _fn

    monkeypatch.setattr(N, "_orig", fake_orig)
    monkeypatch.setattr(N, "_INT32_MAX", 1)

    dense = np.array([[1, 2, 0], [0, 3, 4]], dtype=np.float32)
    X = sparse.csr_matrix(dense)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    a = ad.AnnData(X)
    out = N.fast_normalize_total(a, target_sum=1e4, copy=True)
    assert out is not None and out is not a  # the copied handle


# ==========================================================================
# fast_normalize_total / fast_log1p — dtype + format hardening
# ==========================================================================

def _tiny_counts(n_obs=24, n_vars=18, seed=0, dtype=np.float32):
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.6, size=(n_obs, n_vars)).astype(dtype)
    dense[:, 0] += 1  # no all-zero rows
    X = sparse.csr_matrix(dense)
    if dtype == np.float32:
        X.indptr = X.indptr.astype(np.int32)
        X.indices = X.indices.astype(np.int32)
    return ad.AnnData(X)


def test_normalize_then_log1p_full_pipeline_parity():
    # The canonical tutorial pair on the fast path vs vanilla, end to end.
    pytest.importorskip("scanpy")
    import scanpy as sc
    import autozyme

    a_fast = _tiny_counts(seed=1)
    a_van = a_fast.copy()
    N.fast_normalize_total(a_fast, target_sum=1e4)
    N.fast_log1p(a_fast)
    with autozyme.disabled():
        sc.pp.normalize_total(a_van, target_sum=1e4)
        sc.pp.log1p(a_van)
    np.testing.assert_allclose(_dense(a_fast), _dense(a_van), rtol=1e-4, atol=1e-5)
    assert a_fast.uns["log1p"] == {"base": None}


def test_normalize_float64_int64_warns_then_casts(monkeypatch):
    # float64 data + int64 indices -> one-shot dtype warning + cast to f32/i32.
    monkeypatch.setattr(N, "_DTYPE_WARNED", False, raising=False)
    a = _tiny_counts(seed=2, dtype=np.float64)
    a.X.indptr = a.X.indptr.astype(np.int64)
    a.X.indices = a.X.indices.astype(np.int64)
    with pytest.warns(RuntimeWarning, match="non-canonical dtypes"):
        N.fast_normalize_total(a, target_sum=1e4)
    assert a.X.dtype == np.float32
    sums = np.asarray(a.X.sum(axis=1)).ravel()
    np.testing.assert_allclose(sums[sums > 0], 1e4, rtol=1e-4)


def test_log1p_csc_float64_double_coercion():
    # Non-CSR + float64 -> both the tocsr() and astype(float32) coercion arms.
    a = _tiny_counts(seed=3, dtype=np.float64)
    a.X = a.X.tocsc()
    ref = np.log1p(a.X.toarray())
    N.fast_log1p(a)
    assert sparse.isspmatrix_csr(a.X)
    assert a.X.dtype == np.float32
    np.testing.assert_allclose(a.X.toarray(), ref, rtol=1e-5, atol=1e-6)


def test_log1p_already_log_records_base_none():
    # Calling fast_log1p records uns['log1p']={'base':None} (the marker used by
    # downstream double-log1p detection).
    a = _tiny_counts(seed=4)
    N.fast_normalize_total(a, target_sum=1e4)
    N.fast_log1p(a)
    assert a.uns["log1p"] == {"base": None}


# ==========================================================================
# fast_scale — dtype + clip + var-stats hardening
# ==========================================================================

def test_scale_max_value_clips_both_tails_parity():
    pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")
    import scanpy as sc
    import autozyme

    rng = np.random.default_rng(5)
    # Heavy-tailed values so scaling produces both large +/- z-scores to clip.
    dense = (rng.standard_normal((50, 12)) * 3.0 + 2.0).astype(np.float32)
    dense = np.clip(dense, 0, None)  # keep nonneg (log-norm-like)
    a_fast = ad.AnnData(sparse.csr_matrix(dense))
    a_van = a_fast.copy()
    S.fast_scale(a_fast, max_value=1.5)
    with autozyme.disabled():
        sc.pp.scale(a_van, max_value=1.5)
    fast_X = _dense(a_fast)
    np.testing.assert_allclose(fast_X, _dense(a_van), rtol=1e-3, atol=1e-3)
    # Two-sided clip honored.
    assert fast_X.max() <= 1.5 + 1e-5
    assert fast_X.min() >= -1.5 - 1e-5


def test_scale_float64_input_casts_and_matches():
    pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")
    import scanpy as sc
    import autozyme

    rng = np.random.default_rng(6)
    dense = (rng.random((30, 8)) * 4.0).astype(np.float64)
    a_fast = ad.AnnData(sparse.csr_matrix(dense))  # float64 CSR
    a_van = a_fast.copy()
    S.fast_scale(a_fast, max_value=10)
    with autozyme.disabled():
        sc.pp.scale(a_van, max_value=10)
    np.testing.assert_allclose(_dense(a_fast), _dense(a_van), rtol=1e-3, atol=1e-3)


def test_scale_zero_variance_gene_std_one():
    # A constant gene has var==0 -> std set to 1.0, scaled column is all zeros.
    ad = pytest.importorskip("anndata")
    dense = np.array([[5.0, 1.0],
                      [5.0, 2.0],
                      [5.0, 3.0],
                      [5.0, 4.0]], dtype=np.float32)
    a = ad.AnnData(sparse.csr_matrix(dense))
    S.fast_scale(a, max_value=None)
    out = _dense(a)
    # Constant first column -> all zeros after centering.
    np.testing.assert_allclose(out[:, 0], np.zeros(4), atol=1e-6)
    assert a.var["std"].values[0] == pytest.approx(1.0)


def test_scale_mask_obs_delegates(monkeypatch):
    # mask_obs set -> the fast path declines and delegates to upstream.
    ad = pytest.importorskip("anndata")
    monkeypatch.setattr(S, "_orig_scale", lambda: (lambda data, **kw: "ORIG"))
    a = ad.AnnData(sparse.csr_matrix(np.eye(4, dtype=np.float32)))
    assert S.fast_scale(a, mask_obs=np.array([True, True, False, False])) == "ORIG"


def test_scale_obsm_delegates(monkeypatch):
    # obsm set -> delegate to upstream.
    ad = pytest.importorskip("anndata")
    monkeypatch.setattr(S, "_orig_scale", lambda: (lambda data, **kw: "ORIG"))
    a = ad.AnnData(sparse.csr_matrix(np.eye(3, dtype=np.float32)))
    assert S.fast_scale(a, obsm="X_pca") == "ORIG"
