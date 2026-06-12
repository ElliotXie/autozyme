"""Contract tests for ``sc.tl.pca``.

``fast_pca(adata, n_comps=50, *, zyme=True, **kwargs)`` accepts the full
upstream kwarg set via ``**kwargs``. Risk class: same as the May 2026
``normalize_total`` bug -- ``copy=True`` could silently drop through
without producing a return value.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def log_norm_adata(tiny_adata):
    import autozyme
    import scanpy as sc
    a = tiny_adata.copy()
    with autozyme.disabled():
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    return a


def test_pca_default_return_matches_vanilla(log_norm_adata):
    import autozyme
    import scanpy as sc

    a_fast = log_norm_adata.copy()
    a_vanilla = log_norm_adata.copy()
    with autozyme.disabled():
        ref = sc.tl.pca(a_vanilla, n_comps=10)
    out = sc.tl.pca(a_fast, n_comps=10)

    assert type(out) is type(ref)
    assert "X_pca" in a_fast.obsm
    assert a_fast.obsm["X_pca"].shape == (log_norm_adata.n_obs, 10)


def test_pca_copy_true_returns_distinct(log_norm_adata):
    """``sc.tl.pca(adata, copy=True)`` must return a new AnnData, not None.

    Same bug class as the fixed ``normalize_total(copy=True)`` regression.
    Regression target — if a future refactor of fast_pca drops the copy
    kwarg from **kwargs, this fails immediately.
    """
    import scanpy as sc

    out = sc.tl.pca(log_norm_adata, n_comps=10, copy=True)
    assert out is not None, "copy=True must not return None"
    assert out is not log_norm_adata
    assert "X_pca" in out.obsm


def test_pca_copy_false_returns_none(log_norm_adata):
    import scanpy as sc

    assert sc.tl.pca(log_norm_adata, n_comps=10) is None
    assert "X_pca" in log_norm_adata.obsm


def test_pca_zyme_false_delegates(log_norm_adata):
    import autozyme
    import numpy as np
    import scanpy as sc

    a_escape = log_norm_adata.copy()
    a_vanilla = log_norm_adata.copy()
    with autozyme.disabled():
        sc.tl.pca(a_vanilla, n_comps=10, random_state=0)
    sc.tl.pca(a_escape, n_comps=10, random_state=0, zyme=False)

    # PCA is sign-ambiguous; compare up-to-sign via absolute cosine.
    P_v = a_vanilla.obsm["X_pca"]
    P_e = a_escape.obsm["X_pca"]
    for k in range(min(5, P_v.shape[1])):
        cos = abs(np.dot(P_v[:, k], P_e[:, k])) / (
            np.linalg.norm(P_v[:, k]) * np.linalg.norm(P_e[:, k]) + 1e-12
        )
        assert cos > 0.99, f"zyme=False PC{k} cos={cos:.3f} differs from vanilla"
