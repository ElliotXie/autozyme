"""Unit tests for autozyme.scanpy._pca.

Covers the BLAS helpers (_accelerate_sgemm_gram / _accelerate_sgemm where
available, else the numpy fallback path used inside fast_pca), the dispatch
guards (kwargs that force a vanilla fall-through), and numeric parity of the
Gram-matrix PCA against sklearn/numpy on a tiny matrix.
"""
from __future__ import annotations

import platform

import numpy as np
import pytest
from scipy import sparse

from autozyme.scanpy import _pca as PCA


# --------------------------------------------------------------------------
# Module-level constants / detection
# --------------------------------------------------------------------------

def test_gene_limit_constant():
    assert PCA._GENE_LIMIT == 8000


def test_use_accelerate_only_on_darwin():
    if platform.system() != "Darwin":
        assert PCA._USE_ACCELERATE is False


# --------------------------------------------------------------------------
# Accelerate sgemm helpers (Darwin only) — exact GEMM parity with numpy
# --------------------------------------------------------------------------

@pytest.mark.skipif(not PCA._USE_ACCELERATE,
                    reason="Apple Accelerate not available")
def test_accelerate_sgemm_gram_matches_numpy():
    rng = np.random.default_rng(0)
    X = np.ascontiguousarray(rng.random((20, 6)).astype(np.float32))
    alpha = 0.5
    out = PCA._accelerate_sgemm_gram(X, alpha=alpha)
    ref = alpha * (X.T @ X)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not PCA._USE_ACCELERATE,
                    reason="Apple Accelerate not available")
def test_accelerate_sgemm_matches_numpy():
    rng = np.random.default_rng(1)
    A = np.ascontiguousarray(rng.random((7, 5)).astype(np.float32))
    B = np.ascontiguousarray(rng.random((5, 3)).astype(np.float32))
    out = PCA._accelerate_sgemm(A, B)
    np.testing.assert_allclose(out, A @ B, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------
# Dispatch guards — each must defer to upstream
# --------------------------------------------------------------------------

def test_pca_zyme_false_delegates(monkeypatch):
    called = {}
    monkeypatch.setattr(
        PCA, "_orig_pca",
        lambda: (lambda adata, **kw: called.setdefault("hit", True) and "ORIG" or "ORIG"))
    out = PCA.fast_pca("ADATA", n_comps=10, zyme=False)
    assert out == "ORIG"


@pytest.mark.parametrize("kw", [
    {"layer": "foo"},
    {"zero_center": False},
    {"mask_var": np.array([True, False])},
    {"chunked": True},
    {"return_info": True},
    {"key_added": "mypca"},
    {"dtype": "float64"},
])
def test_pca_guard_kwargs_delegate(monkeypatch, kw):
    monkeypatch.setattr(PCA, "_orig_pca", lambda: (lambda adata, **k: "ORIG"))
    assert PCA.fast_pca("ADATA", n_comps=5, **kw) == "ORIG"


def test_pca_gene_limit_delegates(monkeypatch):
    ad = pytest.importorskip("anndata")
    # n_genes above the limit -> vanilla ARPACK fall-through.
    monkeypatch.setattr(PCA, "_GENE_LIMIT", 3, raising=False)
    called = {}

    def fake_orig():
        def _fn(adata, n_comps=50, copy=False, **kw):
            called["hit"] = True
        return _fn

    monkeypatch.setattr(PCA, "_orig_pca", fake_orig)
    X = np.zeros((10, 5), dtype=np.float32)  # 5 genes > limit(3)
    a = ad.AnnData(X)
    PCA.fast_pca(a, n_comps=2)
    assert called.get("hit")


# --------------------------------------------------------------------------
# Numeric parity — Gram-matrix PCA vs sklearn on a tiny dense matrix
# --------------------------------------------------------------------------

def test_fast_pca_variance_ratio_and_shapes():
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(7)
    n_cells, n_genes, n_comps = 60, 8, 4
    X = (rng.random((n_cells, n_genes)).astype(np.float32) * 3.0)
    a = ad.AnnData(X.copy())
    out = PCA.fast_pca(a, n_comps=n_comps, use_highly_variable=False)
    assert out is None
    assert a.obsm["X_pca"].shape == (n_cells, n_comps)
    assert a.varm["PCs"].shape == (n_genes, n_comps)
    vr = a.uns["pca"]["variance_ratio"]
    assert len(vr) == n_comps
    # Variance ratios are sorted descending and bounded in (0, 1].
    assert np.all(np.diff(vr) <= 1e-5)
    assert np.all(vr > 0) and np.all(vr <= 1.0 + 1e-5)


def test_fast_pca_matches_sklearn_eigenvalues():
    ad = pytest.importorskip("anndata")
    skd = pytest.importorskip("sklearn.decomposition")
    rng = np.random.default_rng(11)
    n_cells, n_genes, n_comps = 80, 6, 4
    X = (rng.random((n_cells, n_genes)).astype(np.float32) * 2.0)
    a = ad.AnnData(X.copy())
    PCA.fast_pca(a, n_comps=n_comps, use_highly_variable=False)

    ref = skd.PCA(n_components=n_comps, svd_solver="full")
    ref.fit(X.astype(np.float64))
    # Explained variance (eigenvalues of covariance) should match.
    np.testing.assert_allclose(
        a.uns["pca"]["variance"], ref.explained_variance_, rtol=1e-2, atol=1e-3)
    # X_pca columns match sklearn up to sign.
    got = np.asarray(a.obsm["X_pca"])
    want = ref.transform(X.astype(np.float64))
    for k in range(n_comps):
        col_g, col_w = got[:, k], want[:, k]
        sign = np.sign(np.dot(col_g, col_w)) or 1.0
        np.testing.assert_allclose(col_g * sign, col_w, rtol=1e-2, atol=1e-2)


def test_fast_pca_copy_returns_new():
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(13)
    X = rng.random((30, 5)).astype(np.float32)
    a = ad.AnnData(X.copy())
    out = PCA.fast_pca(a, n_comps=3, use_highly_variable=False, copy=True)
    assert out is not None and out is not a
    assert "X_pca" not in a.obsm  # original untouched
    assert "X_pca" in out.obsm


def test_fast_pca_sparse_input():
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(17)
    X = (rng.random((40, 6)) * (rng.random((40, 6)) > 0.5)).astype(np.float32)
    a = ad.AnnData(sparse.csr_matrix(X))
    PCA.fast_pca(a, n_comps=3, use_highly_variable=False)
    assert a.obsm["X_pca"].shape == (40, 3)


def test_fast_pca_hvg_subset_lifts_pcs_to_full_space():
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(19)
    n_cells, n_genes = 50, 10
    X = rng.random((n_cells, n_genes)).astype(np.float32)
    a = ad.AnnData(X.copy())
    hv = np.zeros(n_genes, dtype=bool)
    hv[:4] = True  # only 4 HVG
    a.var["highly_variable"] = hv
    PCA.fast_pca(a, n_comps=3)  # use_highly_variable defaults to True here
    # PCs lifted back into full gene space: non-HVG rows are zero.
    pcs = np.asarray(a.varm["PCs"])
    assert pcs.shape[0] == n_genes
    assert np.allclose(pcs[~hv], 0.0)
    assert a.uns["pca"]["params"]["use_highly_variable"] is True
