"""Contract tests for ``sc.tl.rank_genes_groups``.

``_fast_rank_genes_groups`` has a fast path gated on:
  - method='wilcoxon'
  - reference='rest'
  - tie_correct=False
  - sparse input
Anything else falls through to upstream.

Tests pin:
  - default fast path: return type + uns slot populated
  - copy=True returns distinct AnnData
  - method != 'wilcoxon'  -> delegated; same return shape
  - reference != 'rest'   -> delegated; same return shape
  - zyme=False            -> upstream
"""
from __future__ import annotations

import pytest


@pytest.fixture
def clustered_adata(tiny_adata):
    """Need pre-existing group labels in .obs for rank_genes to test against."""
    import autozyme
    import numpy as np
    import scanpy as sc
    a = tiny_adata.copy()
    with autozyme.disabled():
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    # Synthetic 3-group assignment as Categorical (the canonical scanpy
    # pattern; both vanilla and autozyme require this dtype).
    import pandas as pd
    rng = np.random.default_rng(0)
    a.obs["group"] = pd.Categorical(rng.integers(0, 3, size=a.n_obs).astype(str))
    return a


def test_rank_genes_default_return_matches_vanilla(clustered_adata):
    import autozyme
    import scanpy as sc

    a_fast = clustered_adata.copy()
    a_vanilla = clustered_adata.copy()
    with autozyme.disabled():
        ref = sc.tl.rank_genes_groups(a_vanilla, groupby="group",
                                      method="wilcoxon")
    out = sc.tl.rank_genes_groups(a_fast, groupby="group",
                                  method="wilcoxon")

    assert type(out) is type(ref)
    assert "rank_genes_groups" in a_fast.uns


def test_rank_genes_copy_true_returns_distinct(clustered_adata):
    """copy=True returns a new AnnData."""
    import scanpy as sc

    out = sc.tl.rank_genes_groups(clustered_adata, groupby="group",
                                  method="wilcoxon", copy=True)
    assert out is not None
    assert out is not clustered_adata
    assert "rank_genes_groups" in out.uns
    assert "rank_genes_groups" not in clustered_adata.uns


def test_rank_genes_t_test_delegates(clustered_adata):
    """method='t-test' is not on the fast path -> upstream delegation."""
    import autozyme
    import scanpy as sc

    a_fast = clustered_adata.copy()
    a_vanilla = clustered_adata.copy()
    with autozyme.disabled():
        ref = sc.tl.rank_genes_groups(a_vanilla, groupby="group",
                                      method="t-test")
    out = sc.tl.rank_genes_groups(a_fast, groupby="group",
                                  method="t-test")
    assert type(out) is type(ref)


def test_rank_genes_specific_reference_delegates(clustered_adata):
    """reference!='rest' (e.g. reference='0') -> upstream delegation."""
    import autozyme
    import scanpy as sc

    a_fast = clustered_adata.copy()
    a_vanilla = clustered_adata.copy()
    with autozyme.disabled():
        ref = sc.tl.rank_genes_groups(a_vanilla, groupby="group",
                                      method="wilcoxon", reference="0")
    out = sc.tl.rank_genes_groups(a_fast, groupby="group",
                                  method="wilcoxon", reference="0")
    assert type(out) is type(ref)


def test_rank_genes_rankby_abs_full_output_matches_vanilla(clustered_adata):
    """rankby_abs=True without n_genes must rank by |score|, like upstream.

    Regression (scope audit, 2026-06-12): the full-output fast-path branch
    sorted by signed score and ignored ``rankby_abs``, so a full-output call
    with ``rankby_abs=True`` silently diverged from vanilla (the top-N gene
    set per group was wrong). The top-N (n_genes) branch already honored it.
    """
    import autozyme
    import numpy as np
    import scanpy as sc

    a_fast = clustered_adata.copy()
    a_vanilla = clustered_adata.copy()
    with autozyme.disabled():
        sc.tl.rank_genes_groups(a_vanilla, groupby="group",
                                method="wilcoxon", rankby_abs=True)
    sc.tl.rank_genes_groups(a_fast, groupby="group",
                            method="wilcoxon", rankby_abs=True)

    rf = a_fast.uns["rank_genes_groups"]
    rv = a_vanilla.uns["rank_genes_groups"]
    for g in rf["names"].dtype.names:
        # The sort key is |score|; it must match vanilla at every rank position.
        sf = np.abs(np.asarray(rf["scores"][g], dtype=np.float64))
        sv = np.abs(np.asarray(rv["scores"][g], dtype=np.float64))
        np.testing.assert_allclose(sf, sv, rtol=0, atol=1e-5)
        # And the actual top-N ranked gene sets must agree.
        k = min(20, len(sf))
        top_f = set(np.asarray(rf["names"][g])[:k])
        top_v = set(np.asarray(rv["names"][g])[:k])
        assert top_f == top_v, f"group {g}: top-{k} gene set diverged"


def test_rank_genes_zyme_false_delegates(clustered_adata):
    import scanpy as sc

    a = clustered_adata.copy()
    out = sc.tl.rank_genes_groups(a, groupby="group",
                                  method="wilcoxon", zyme=False)
    assert out is None
    assert "rank_genes_groups" in a.uns
