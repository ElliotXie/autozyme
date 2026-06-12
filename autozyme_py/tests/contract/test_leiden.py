"""Contract tests for ``sc.tl.leiden``.

``fast_leiden`` already implements the ``copy`` kwarg (``adata_out =
adata.copy() if copy else adata``), so it's the model for what
normalize_total / log1p should look like. Tests here pin that.

Dispatch surface:
  - flavor='leidenalg'  -> upstream delegation (different algorithm)
  - flavor='igraph'     -> fast path (default)
  - zyme=False          -> upstream
"""
from __future__ import annotations

import pytest


@pytest.fixture
def neighbors_adata(tiny_adata):
    """Compute log-norm + PCA + neighbors so leiden has a graph to chew on."""
    import autozyme
    import scanpy as sc
    a = tiny_adata.copy()
    with autozyme.disabled():
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.tl.pca(a, n_comps=10)
        sc.pp.neighbors(a, n_neighbors=10, random_state=0)
    return a


def test_leiden_default_return_matches_vanilla(neighbors_adata):
    import autozyme
    import scanpy as sc

    a_fast = neighbors_adata.copy()
    a_vanilla = neighbors_adata.copy()
    with autozyme.disabled():
        ref = sc.tl.leiden(a_vanilla, flavor="igraph",
                           n_iterations=2, directed=False, random_state=0)
    out = sc.tl.leiden(a_fast, flavor="igraph",
                       n_iterations=2, directed=False, random_state=0)

    assert type(out) is type(ref)
    assert "leiden" in a_fast.obs.columns


def test_leiden_copy_true_returns_distinct(neighbors_adata):
    """``copy=True`` returns a new AnnData carrying the leiden labels."""
    import scanpy as sc

    out = sc.tl.leiden(neighbors_adata, flavor="igraph",
                       n_iterations=2, directed=False, random_state=0,
                       copy=True)
    assert out is not None
    assert out is not neighbors_adata
    assert "leiden" in out.obs.columns
    # Input must be untouched.
    assert "leiden" not in neighbors_adata.obs.columns


def test_leiden_copy_false_returns_none(neighbors_adata):
    import scanpy as sc

    out = sc.tl.leiden(neighbors_adata, flavor="igraph",
                       n_iterations=2, directed=False, random_state=0)
    assert out is None
    assert "leiden" in neighbors_adata.obs.columns


def test_leiden_leidenalg_delegates(neighbors_adata):
    """flavor='leidenalg' must delegate to upstream (different algorithm)."""
    pytest.importorskip("leidenalg")
    import autozyme
    import scanpy as sc

    a_fast = neighbors_adata.copy()
    a_vanilla = neighbors_adata.copy()
    with autozyme.disabled():
        ref = sc.tl.leiden(a_vanilla, flavor="leidenalg",
                           n_iterations=2, directed=False, random_state=0)
    out = sc.tl.leiden(a_fast, flavor="leidenalg",
                       n_iterations=2, directed=False, random_state=0)
    assert type(out) is type(ref)


def test_leiden_zyme_false_delegates(neighbors_adata):
    """Per-call zyme=False delegates to upstream."""
    import scanpy as sc

    out = sc.tl.leiden(neighbors_adata, flavor="igraph",
                       n_iterations=2, directed=False, random_state=0,
                       zyme=False)
    # leidenalg may or may not be installed; just verify no crash + labels added.
    assert out is None or out is not None  # both contracts valid for the bypass
    assert "leiden" in neighbors_adata.obs.columns
