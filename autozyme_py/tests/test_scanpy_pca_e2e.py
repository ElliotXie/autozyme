"""End-to-end ``sc.tl.pca`` patched-path tests.

Drives a tiny real AnnData through ``fast_pca`` and asserts parity (up to the
inherent eigenvector sign ambiguity) vs upstream. Covers the wrapper dispatch /
scope-guard / HVG-subset / copy / n_comps / scope-guard-fallback lines in
``_pca.py``.

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


def _scaled(n_obs=60, n_vars=80, seed=10, with_hvg=False):
    """A scaled dense AnnData ready for PCA."""
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.7, size=(n_obs, n_vars)).astype(np.float32)
    dense[:, 0] += 1
    X = sparse.csr_matrix(dense)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    adata = ad.AnnData(X)
    adata.var_names = [f"g{j}" for j in range(n_vars)]
    sc.pp.normalize_total(adata, target_sum=1e4, zyme=False)
    sc.pp.log1p(adata, zyme=False)
    if with_hvg:
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=40, zyme=False)
    sc.pp.scale(adata, max_value=10, zyme=False)
    return adata


def _abs_align(a, b):
    """Compare PCA embeddings up to per-component sign flip."""
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape
    for k in range(a.shape[1]):
        col_a, col_b = a[:, k], b[:, k]
        same = np.allclose(col_a, col_b, atol=1e-2, rtol=1e-2)
        flip = np.allclose(col_a, -col_b, atol=1e-2, rtol=1e-2)
        assert same or flip, f"PC {k} mismatch"


@pytest.fixture(autouse=True)
def _activate():
    autozyme.activate("scanpy")
    yield


def test_pca_parity_default_ncomps():
    a_fast = _scaled()
    a_orig = a_fast.copy()
    sc.tl.pca(a_fast, n_comps=20, use_highly_variable=False)
    sc.tl.pca(a_orig, n_comps=20, use_highly_variable=False, zyme=False)
    _abs_align(a_fast.obsm["X_pca"], a_orig.obsm["X_pca"])


def test_pca_uns_keys_present():
    a = _scaled()
    sc.tl.pca(a, n_comps=15, use_highly_variable=False)
    assert "X_pca" in a.obsm
    assert "PCs" in a.varm
    assert "pca" in a.uns
    assert a.uns["pca"]["variance"].shape[0] == 15


def test_pca_variance_ratio_matches_upstream():
    a_fast = _scaled(seed=11)
    a_orig = a_fast.copy()
    sc.tl.pca(a_fast, n_comps=10, use_highly_variable=False)
    sc.tl.pca(a_orig, n_comps=10, use_highly_variable=False, zyme=False)
    np.testing.assert_allclose(
        a_fast.uns["pca"]["variance_ratio"],
        a_orig.uns["pca"]["variance_ratio"],
        rtol=1e-2, atol=1e-3,
    )


def test_pca_copy_true_returns_new():
    a = _scaled()
    out = sc.tl.pca(a, n_comps=10, use_highly_variable=False, copy=True)
    assert out is not a
    assert "X_pca" in out.obsm
    assert "X_pca" not in a.obsm


def test_pca_inplace_returns_none():
    a = _scaled()
    assert sc.tl.pca(a, n_comps=10, use_highly_variable=False) is None


def test_pca_hvg_subset_path():
    # use_highly_variable left default with HVG present → wrapper subsets,
    # runs fast PCA on the subset, then lifts PCs back into full var space.
    a = _scaled(with_hvg=True)
    sc.tl.pca(a, n_comps=10)
    assert a.varm["PCs"].shape[0] == a.n_vars  # full gene space
    assert a.uns["pca"]["params"]["use_highly_variable"] is True
    # Non-HVG rows of PCs are zero-filled.
    mask = a.var["highly_variable"].values
    assert np.allclose(a.varm["PCs"][~mask], 0.0)


def test_pca_zero_center_false_delegates():
    # zero_center=False trips the scope guard → upstream PCA.
    a = _scaled()
    sc.tl.pca(a, n_comps=10, use_highly_variable=False, zero_center=False)
    assert "X_pca" in a.obsm


def test_pca_return_info_delegates():
    # return_info=True trips the scope guard → upstream PCA.
    a = _scaled()
    sc.tl.pca(a, n_comps=10, use_highly_variable=False, return_info=True)
    assert "X_pca" in a.obsm


def test_pca_sparse_input_densified():
    # Fast path on still-sparse .X (no scale) exercises the X.toarray() branch.
    rng = np.random.default_rng(12)
    dense = rng.poisson(0.7, size=(60, 70)).astype(np.float32)
    X = sparse.csr_matrix(dense)
    a = ad.AnnData(X)
    sc.tl.pca(a, n_comps=10, use_highly_variable=False)
    assert "X_pca" in a.obsm


def test_pca_zyme_false_equals_upstream():
    a_fast = _scaled(seed=13)
    a_z = a_fast.copy()
    sc.tl.pca(a_fast, n_comps=10, use_highly_variable=False, zyme=False)
    autozyme.deactivate("scanpy")
    sc.tl.pca(a_z, n_comps=10, use_highly_variable=False)
    _abs_align(a_fast.obsm["X_pca"], a_z.obsm["X_pca"])
