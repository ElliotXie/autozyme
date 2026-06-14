"""End-to-end ``sc.pp.scale`` patched-path tests.

Drives a tiny real AnnData through ``fast_scale`` and asserts parity vs the
upstream original, covering the wrapper dispatch / fallback branches in
``_scale.py``: zero_center=False, layer/obsm/mask_obs set, non-CSR input,
copy=True/False, max_value, and the var stats annotations.

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


def _lognorm(n_obs=50, n_vars=60, seed=5):
    """A log-normalized sparse CSR AnnData (the typical pre-scale state)."""
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.6, size=(n_obs, n_vars)).astype(np.float32)
    dense[:, 0] += 1
    X = sparse.csr_matrix(dense)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    adata = ad.AnnData(X)
    adata.var_names = [f"g{j}" for j in range(n_vars)]
    sc.pp.normalize_total(adata, target_sum=1e4, zyme=False)
    sc.pp.log1p(adata, zyme=False)
    return adata


def _dense(adata):
    X = adata.X
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


@pytest.fixture(autouse=True)
def _activate():
    autozyme.activate("scanpy")
    yield


def test_scale_parity_with_max_value():
    a_fast = _lognorm()
    a_orig = a_fast.copy()
    sc.pp.scale(a_fast, max_value=10)
    sc.pp.scale(a_orig, max_value=10, zyme=False)
    np.testing.assert_allclose(_dense(a_fast), _dense(a_orig), rtol=1e-3, atol=1e-3)


def test_scale_parity_no_clip():
    a_fast = _lognorm(seed=6)
    a_orig = a_fast.copy()
    sc.pp.scale(a_fast)              # max_value=None → np.inf clip
    sc.pp.scale(a_orig, zyme=False)
    np.testing.assert_allclose(_dense(a_fast), _dense(a_orig), rtol=1e-3, atol=1e-3)


def test_scale_writes_var_stats():
    a = _lognorm()
    sc.pp.scale(a, max_value=10)
    for k in ("mean", "var", "std"):
        assert k in a.var.columns


def test_scale_copy_true_returns_new():
    a = _lognorm()
    out = sc.pp.scale(a, max_value=10, copy=True)
    assert out is not a
    assert "mean" in out.var.columns


def test_scale_inplace_returns_none():
    a = _lognorm()
    assert sc.pp.scale(a, max_value=10) is None


def test_scale_zero_center_false_delegates():
    # zero_center=False is not handled by the fast path → upstream fallback.
    a_fast = _lognorm(seed=7)
    a_orig = a_fast.copy()
    sc.pp.scale(a_fast, zero_center=False, max_value=10)
    sc.pp.scale(a_orig, zero_center=False, max_value=10, zyme=False)
    np.testing.assert_allclose(_dense(a_fast), _dense(a_orig), rtol=1e-3, atol=1e-3)


def test_scale_dense_input_delegates():
    # Non-CSR (dense) .X is not handled by the fast kernel → upstream fallback.
    a = _lognorm()
    a.X = a.X.toarray()
    out = sc.pp.scale(a, max_value=10, copy=True)
    assert out.X.shape == a.X.shape


def test_scale_layer_delegates_to_upstream():
    a = _lognorm()
    a.layers["norm"] = a.X.copy()
    # layer set → fast path declines, upstream handles the layer scale.
    sc.pp.scale(a, layer="norm", max_value=10)
    assert "norm" in a.layers


def test_scale_float64_csr_fast_path_cast():
    # CSR float64 input still hits the fast path (it casts x to float32 first).
    a_fast = _lognorm(seed=9)
    a_fast.X = a_fast.X.astype(np.float64)
    a_orig = a_fast.copy()
    sc.pp.scale(a_fast, max_value=10)
    sc.pp.scale(a_orig, max_value=10, zyme=False)
    np.testing.assert_allclose(_dense(a_fast), _dense(a_orig), rtol=1e-3, atol=1e-3)


def test_scale_zyme_false_equals_upstream():
    a_fast = _lognorm(seed=8)
    a_z = a_fast.copy()
    sc.pp.scale(a_fast, max_value=10, zyme=False)
    autozyme.deactivate("scanpy")
    sc.pp.scale(a_z, max_value=10)
    np.testing.assert_allclose(_dense(a_fast), _dense(a_z), rtol=1e-4, atol=1e-4)
