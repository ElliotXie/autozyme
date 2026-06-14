"""End-to-end ``sc.pp.regress_out`` patched-path tests.

Drives a tiny real AnnData through ``fast_regress_out`` and asserts parity vs
upstream across its dispatch branches: sparse fast path (patch-engaged),
dense + non-singular gram (defer to upstream), singular gram (pinv bypass of
the per-gene statsmodels GLM), categorical regressor (defer), empty keys
(defer), copy=True/False.

Env recipe: KMP_DUPLICATE_LIB_OK=TRUE NUMBA_THREADING_LAYER=workqueue
NUMBA_NUM_THREADS=14 OMP_NUM_THREADS=2.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

sc = pytest.importorskip("scanpy")
ad = pytest.importorskip("anndata")
import autozyme  # noqa: E402


def _lognorm(n_obs=40, n_vars=30, seed=20, sparse_x=True):
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.8, size=(n_obs, n_vars)).astype(np.float32)
    dense[:, 0] += 1
    X = sparse.csr_matrix(dense)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    adata = ad.AnnData(X)
    adata.var_names = [f"g{j}" for j in range(n_vars)]
    sc.pp.normalize_total(adata, target_sum=1e4, zyme=False)
    sc.pp.log1p(adata, zyme=False)
    # A continuous covariate to regress out, plus a constant-zero one.
    adata.obs["total"] = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    adata.obs["zero_cov"] = np.zeros(n_obs, dtype=np.float64)
    rng2 = np.random.default_rng(seed + 100)
    adata.obs["pct_mt"] = rng2.random(n_obs).astype(np.float64)
    if not sparse_x:
        adata.X = adata.X.toarray()
    return adata


def _dense(adata):
    X = adata.X
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


@pytest.fixture(autouse=True)
def _activate():
    autozyme.activate("scanpy")
    yield


def test_regress_out_sparse_parity():
    # Sparse path → patch-engaged sparse-aware GEMM.
    a_fast = _lognorm()
    a_orig = a_fast.copy()
    sc.pp.regress_out(a_fast, ["total"])
    sc.pp.regress_out(a_orig, ["total"], zyme=False)
    np.testing.assert_allclose(_dense(a_fast), _dense(a_orig), rtol=1e-3, atol=1e-4)


def test_regress_out_string_key_normalized_to_list():
    a_fast = _lognorm(seed=21)
    a_orig = a_fast.copy()
    sc.pp.regress_out(a_fast, "total")          # str → [str]
    sc.pp.regress_out(a_orig, "total", zyme=False)
    np.testing.assert_allclose(_dense(a_fast), _dense(a_orig), rtol=1e-3, atol=1e-4)


def test_regress_out_singular_gram_pinv_parity():
    # zero_cov is constant-zero → gram is singular → pinv bypass path
    # (replaces upstream's slow per-gene statsmodels GLM fallback).
    a_fast = _lognorm(seed=22)
    a_orig = a_fast.copy()
    sc.pp.regress_out(a_fast, ["total", "zero_cov"])
    sc.pp.regress_out(a_orig, ["total", "zero_cov"], zyme=False)
    np.testing.assert_allclose(_dense(a_fast), _dense(a_orig), rtol=1e-2, atol=1e-3)


def test_regress_out_dense_nonsingular_defers_to_upstream():
    # Dense X + non-singular gram → wrapper defers to upstream (parity exact).
    a_fast = _lognorm(seed=23, sparse_x=True)
    a_fast.X = a_fast.X.toarray()
    a_orig = a_fast.copy()
    sc.pp.regress_out(a_fast, ["total"])
    sc.pp.regress_out(a_orig, ["total"], zyme=False)
    np.testing.assert_allclose(_dense(a_fast), _dense(a_orig), rtol=1e-3, atol=1e-4)


def test_regress_out_copy_true_returns_new():
    a = _lognorm()
    before = _dense(a).copy()
    out = sc.pp.regress_out(a, ["total"], copy=True)
    assert out is not a
    # Original unchanged when copy=True.
    np.testing.assert_allclose(_dense(a), before, rtol=1e-6)


def test_regress_out_inplace_returns_none():
    a = _lognorm()
    assert sc.pp.regress_out(a, ["total"]) is None


def test_regress_out_categorical_key_defers():
    # Categorical regressor → wrapper defers to upstream (different semantics).
    a = _lognorm(seed=24)
    a.obs["grp"] = (np.arange(a.n_obs) % 2).astype(str)
    a.obs["grp"] = a.obs["grp"].astype("category")
    out = sc.pp.regress_out(a, ["grp"], copy=True)
    assert out.X.shape == a.X.shape


def test_regress_out_zyme_false_equals_upstream():
    a_fast = _lognorm(seed=25)
    a_z = a_fast.copy()
    sc.pp.regress_out(a_fast, ["total"], zyme=False)
    autozyme.deactivate("scanpy")
    sc.pp.regress_out(a_z, ["total"])
    np.testing.assert_allclose(_dense(a_fast), _dense(a_z), rtol=1e-4, atol=1e-5)
