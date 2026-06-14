"""End-to-end ``sc.pp.normalize_total`` / ``sc.pp.log1p`` patched-path tests.

Drives a tiny real AnnData through the patched functions and asserts PARITY
against the upstream original (``zyme=False``). This exercises the
pure-python wrapper / dispatch / dtype-handling / copy-vs-inplace / return-value
lines in ``_normalize.py`` that coverage can see (the numba kernel bodies stay
invisible to coverage but are reached via the fast path).

Env recipe: KMP_DUPLICATE_LIB_OK=TRUE NUMBA_THREADING_LAYER=workqueue
NUMBA_NUM_THREADS=14 OMP_NUM_THREADS=2 (see BRIEFING2). Do NOT pin a small
thread count: let the patched code resolve its own thread pool.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

sc = pytest.importorskip("scanpy")
ad = pytest.importorskip("anndata")
import autozyme  # noqa: E402


def _counts(n_obs=50, n_vars=70, seed=1, dtype=np.float32, int_idx=True):
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.5, size=(n_obs, n_vars)).astype(dtype)
    # Guarantee no all-zero row so target_sum=None median path is well defined.
    dense[:, 0] += 1
    X = sparse.csr_matrix(dense)
    if int_idx and dtype == np.float32:
        X.indptr = X.indptr.astype(np.int32)
        X.indices = X.indices.astype(np.int32)
    adata = ad.AnnData(X)
    adata.var_names = [f"g{j}" for j in range(n_vars)]
    return adata


def _dense(adata):
    X = adata.X
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


@pytest.fixture(autouse=True)
def _activate():
    autozyme.activate("scanpy")
    yield


# --------------------------------------------------------------------------
# normalize_total
# --------------------------------------------------------------------------

def test_normalize_total_explicit_target_parity():
    a_fast = _counts()
    a_orig = a_fast.copy()
    out_fast = sc.pp.normalize_total(a_fast, target_sum=1e4, copy=True)
    out_orig = sc.pp.normalize_total(a_orig, target_sum=1e4, copy=True, zyme=False)
    np.testing.assert_allclose(_dense(out_fast), _dense(out_orig), rtol=1e-4, atol=1e-3)
    # copy=True returns a new object, original untouched.
    assert out_fast is not a_fast


def test_normalize_total_target_none_median_parity():
    a_fast = _counts(seed=2)
    a_orig = a_fast.copy()
    sc.pp.normalize_total(a_fast)            # target_sum=None → median path
    sc.pp.normalize_total(a_orig, zyme=False)
    np.testing.assert_allclose(_dense(a_fast), _dense(a_orig), rtol=1e-3, atol=1e-3)


def test_normalize_total_inplace_returns_none():
    a = _counts()
    ret = sc.pp.normalize_total(a, target_sum=1e4)
    assert ret is None  # default inplace+no-copy mutates and returns None


def test_normalize_total_inplace_false_delegates_to_upstream():
    # inplace=False returns a dict and is delegated to upstream by the wrapper.
    a = _counts()
    res = sc.pp.normalize_total(a, target_sum=1e4, inplace=False)
    assert isinstance(res, dict)
    assert "X" in res or "norm_factor" in res


def test_normalize_total_zyme_false_equals_upstream():
    a_fast = _counts(seed=3)
    a_z = a_fast.copy()
    sc.pp.normalize_total(a_fast, target_sum=1e4, zyme=False)
    # Compare against a freshly-deactivated upstream call.
    autozyme.deactivate("scanpy")
    sc.pp.normalize_total(a_z, target_sum=1e4)
    np.testing.assert_allclose(_dense(a_fast), _dense(a_z), rtol=1e-5)


def test_normalize_total_extra_kwarg_delegates():
    # Passing an extra kwarg (key_added) takes the `kwargs` delegation branch.
    a = _counts()
    sc.pp.normalize_total(a, target_sum=1e4, key_added="nrm")
    assert "nrm" in a.obs


def test_normalize_total_dense_input_coerced_to_csr():
    a = _counts()
    a.X = a.X.toarray().astype(np.float32)  # dense
    sc.pp.normalize_total(a, target_sum=1e4)
    assert sparse.issparse(a.X)  # wrapper coerces dense → CSR


def test_normalize_total_float64_int64_warns_and_casts():
    # Non-canonical dtypes trigger the one-shot dtype warning + cast branch.
    # Reset module flag so the warning is emitted in this test.
    from autozyme.scanpy import _normalize as N
    N._DTYPE_WARNED = False
    a = _counts(dtype=np.float64, int_idx=False)
    with pytest.warns(RuntimeWarning):
        sc.pp.normalize_total(a, target_sum=1e4)
    assert a.X.dtype == np.float32


def test_normalize_total_csc_input_tocsr():
    a = _counts()
    a.X = a.X.tocsc()  # non-CSR sparse
    sc.pp.normalize_total(a, target_sum=1e4)
    assert sparse.isspmatrix_csr(a.X)


# --------------------------------------------------------------------------
# log1p
# --------------------------------------------------------------------------

def test_log1p_parity_and_uns_recorded():
    a_fast = _counts(seed=4)
    sc.pp.normalize_total(a_fast, target_sum=1e4)
    a_orig = a_fast.copy()
    sc.pp.log1p(a_fast)
    sc.pp.log1p(a_orig, zyme=False)
    np.testing.assert_allclose(_dense(a_fast), _dense(a_orig), rtol=1e-5, atol=1e-6)
    # Fast path records the upstream-compatible log1p marker.
    assert a_fast.uns["log1p"]["base"] is None


def test_log1p_copy_true_returns_new():
    a = _counts()
    sc.pp.normalize_total(a, target_sum=1e4)
    out = sc.pp.log1p(a, copy=True)
    assert out is not a
    assert "log1p" in out.uns


def test_log1p_inplace_returns_none():
    a = _counts()
    sc.pp.normalize_total(a, target_sum=1e4)
    assert sc.pp.log1p(a) is None


def test_log1p_dense_input_path():
    a = _counts()
    a.X = a.X.toarray().astype(np.float32)  # dense → np.log1p out= branch
    sc.pp.log1p(a)
    # Values are log-scaled (max < original max which was small counts).
    assert np.isfinite(a.X).all()


def test_log1p_csc_and_float64_coercion():
    a = _counts(dtype=np.float64, int_idx=False)
    a.X = a.X.tocsc()  # non-CSR + float64 → both coercion branches
    sc.pp.log1p(a)
    assert sparse.isspmatrix_csr(a.X)
    assert a.X.dtype == np.float32


def test_log1p_extra_kwarg_delegates():
    a = _counts()
    sc.pp.normalize_total(a, target_sum=1e4)
    # base= kwarg goes through the kwargs delegation branch to upstream.
    sc.pp.log1p(a, base=2)
    assert a.uns["log1p"]["base"] == 2
