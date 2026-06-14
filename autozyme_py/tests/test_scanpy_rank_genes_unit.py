"""Unit tests for autozyme.scanpy._rank_genes.

Tests the numba kernels directly:
  _dual_sort / _partial_dual_sort  — sorting primitives
  _batch_top_n                     — top-n indices per row by descending score
  _bh_correct_2d / _bh_gather      — Benjamini-Hochberg correction
  _presort_csc_columns             — per-column value sort
  _fused_stats_rank_sums_csc       — group stats + Wilcoxon rank sums
  _fused_all_csc                   — fused rank sums + z-scores + p-values
  _compute_topn_logfc              — log2 fold change
  _sparse_rankdata_csc             — full per-cell rank matrix

Plus _fast_rank_genes_groups dispatch + end-to-end parity vs vanilla scanpy
Wilcoxon on a tiny CSR matrix.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse

from autozyme.scanpy import _rank_genes as RG


pytestmark = pytest.mark.skipif(not RG.HAS_NUMBA, reason="numba not installed")


# ==========================================================================
# _dual_sort — sorts vals[0:n] and idx[0:n] together by vals
# ==========================================================================

def test_dual_sort_basic():
    vals = np.array([3.0, 1.0, 2.0, 0.0], dtype=np.float64)
    idx = np.array([0, 1, 2, 3], dtype=np.int64)
    RG._dual_sort(vals, idx, 4)
    np.testing.assert_array_equal(vals, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(idx, [3, 1, 2, 0])


def test_dual_sort_already_sorted():
    vals = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    idx = np.array([0, 1, 2], dtype=np.int64)
    RG._dual_sort(vals, idx, 3)
    np.testing.assert_array_equal(vals, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(idx, [0, 1, 2])


def test_dual_sort_reverse():
    n = 50
    rng = np.random.default_rng(0)
    base = rng.random(n)
    vals = base.copy()
    idx = np.arange(n, dtype=np.int64)
    RG._dual_sort(vals, idx, n)
    np.testing.assert_allclose(vals, np.sort(base))
    np.testing.assert_array_equal(vals, base[idx])  # idx tracks original positions


def test_dual_sort_with_ties():
    vals = np.array([2.0, 1.0, 2.0, 1.0], dtype=np.float64)
    idx = np.array([0, 1, 2, 3], dtype=np.int64)
    RG._dual_sort(vals, idx, 4)
    np.testing.assert_array_equal(vals, [1.0, 1.0, 2.0, 2.0])


def test_dual_sort_n_le_1_noop():
    vals = np.array([5.0], dtype=np.float64)
    idx = np.array([0], dtype=np.int64)
    RG._dual_sort(vals, idx, 1)
    assert vals[0] == 5.0


def test_dual_sort_large_random_matches_numpy():
    rng = np.random.default_rng(1)
    base = rng.random(200)
    vals = base.copy()
    idx = np.arange(200, dtype=np.int64)
    RG._dual_sort(vals, idx, 200)
    np.testing.assert_allclose(vals, np.sort(base))
    # idx is a valid permutation.
    assert sorted(idx.tolist()) == list(range(200))


# ==========================================================================
# _partial_dual_sort — smallest k elements sorted at front
# ==========================================================================

def test_partial_dual_sort_smallest_k():
    rng = np.random.default_rng(2)
    base = rng.random(100)
    vals = base.copy()
    idx = np.arange(100, dtype=np.int64)
    k = 5
    RG._partial_dual_sort(vals, idx, 100, k)
    # First k are the k smallest, in ascending order.
    np.testing.assert_allclose(vals[:k], np.sort(base)[:k])


def test_partial_dual_sort_k_ge_n_full_sort():
    base = np.array([3.0, 1.0, 2.0], dtype=np.float64)
    vals = base.copy()
    idx = np.array([0, 1, 2], dtype=np.int64)
    RG._partial_dual_sort(vals, idx, 3, 5)  # k >= n
    np.testing.assert_array_equal(vals, [1.0, 2.0, 3.0])


def test_partial_dual_sort_k_zero_noop():
    base = np.array([3.0, 1.0, 2.0], dtype=np.float64)
    vals = base.copy()
    idx = np.array([0, 1, 2], dtype=np.int64)
    RG._partial_dual_sort(vals, idx, 3, 0)
    np.testing.assert_array_equal(vals, base)  # unchanged


# ==========================================================================
# _batch_top_n — top-n indices per row by descending score
# ==========================================================================

def test_batch_top_n_descending():
    scores = np.array([[1.0, 5.0, 3.0, 2.0],
                       [9.0, 0.0, 7.0, 8.0]], dtype=np.float64)
    result = np.empty((2, 2), dtype=np.int64)
    RG._batch_top_n(scores, 2, result)
    # Row 0 top-2 by descending score: indices 1 (5.0), 2 (3.0).
    np.testing.assert_array_equal(result[0], [1, 2])
    # Row 1 top-2: indices 0 (9.0), 3 (8.0).
    np.testing.assert_array_equal(result[1], [0, 3])


def test_batch_top_n_full():
    scores = np.array([[3.0, 1.0, 2.0]], dtype=np.float64)
    result = np.empty((1, 3), dtype=np.int64)
    RG._batch_top_n(scores, 3, result)
    np.testing.assert_array_equal(result[0], [0, 2, 1])  # 3,2,1 -> idx 0,2,1


# ==========================================================================
# _bh_correct_2d — Benjamini-Hochberg per row, matches statsmodels
# ==========================================================================

def test_bh_correct_2d_matches_reference():
    pvals = np.array([[0.01, 0.5, 0.04, 0.2, 0.8]], dtype=np.float64)
    adj = np.empty_like(pvals)
    RG._bh_correct_2d(pvals, adj, 1, 5)
    # Reference BH (step-up): sort, p*n/rank, cumulative min from the top.
    n = 5
    order = np.argsort(pvals[0])
    sorted_p = pvals[0][order]
    ref_sorted = np.minimum.accumulate(
        (sorted_p * n / np.arange(1, n + 1))[::-1])[::-1]
    ref_sorted = np.clip(ref_sorted, 0, 1)
    ref = np.empty(n)
    ref[order] = ref_sorted
    np.testing.assert_allclose(adj[0], ref, rtol=1e-9)


def test_bh_correct_2d_all_ones_stay_one():
    pvals = np.ones((1, 4), dtype=np.float64)
    adj = np.empty_like(pvals)
    RG._bh_correct_2d(pvals, adj, 1, 4)
    np.testing.assert_allclose(adj[0], np.ones(4))


# ==========================================================================
# _presort_csc_columns
# ==========================================================================

def test_presort_csc_columns_sorts_each_column():
    # CSC: column 0 has values [3,1], column 1 has [2].
    X = sparse.csc_matrix(np.array([[3.0, 0.0],
                                    [1.0, 2.0]], dtype=np.float32))
    data = X.data.copy()
    indices = X.indices.copy()
    RG._presort_csc_columns(X.indptr.astype(np.int32),
                            indices.astype(np.int32), data)
    # Column 0 data sorted ascending.
    col0 = data[X.indptr[0]:X.indptr[1]]
    assert np.all(np.diff(col0) >= 0)


# ==========================================================================
# _fused_stats_rank_sums_csc — Wilcoxon rank sums match scipy
# ==========================================================================

def test_fused_rank_sums_match_scipy_ranksums():
    scipy_stats = pytest.importorskip("scipy.stats")
    # One gene, two groups. Dense column with zeros (sparse-aware ranks).
    col = np.array([0.0, 0.0, 1.0, 3.0, 5.0, 2.0], dtype=np.float64)
    groups = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)  # 3 vs 3
    n_cells = 6

    X = sparse.csc_matrix(col.reshape(-1, 1))
    X.data = X.data.astype(np.float64)
    indptr = X.indptr.astype(np.int64)
    indices = X.indices.astype(np.int64)

    group_sizes = np.array([3, 3], dtype=np.int64)
    group_sums = np.zeros((2, 1), dtype=np.float64)
    group_sq = np.zeros((2, 1), dtype=np.float64)
    group_nnz = np.zeros((2, 1), dtype=np.int64)
    rank_sums = np.zeros((2, 1), dtype=np.float64)

    RG._fused_stats_rank_sums_csc(
        indptr, indices, X.data, n_cells, groups, group_sizes, 2,
        group_sums, group_sq, group_nnz, rank_sums)

    # scipy: rank all cells, sum ranks per group (average ranks for ties).
    ranks = scipy_stats.rankdata(col)
    ref_rank_sum_g0 = ranks[groups == 0].sum()
    ref_rank_sum_g1 = ranks[groups == 1].sum()
    np.testing.assert_allclose(rank_sums[0, 0], ref_rank_sum_g0, rtol=1e-9)
    np.testing.assert_allclose(rank_sums[1, 0], ref_rank_sum_g1, rtol=1e-9)
    # Group sums / nnz also correct.
    np.testing.assert_allclose(group_sums[0, 0], col[groups == 0].sum())
    np.testing.assert_allclose(group_sums[1, 0], col[groups == 1].sum())


def test_fused_rank_sums_total_invariant():
    # Sum of rank sums across groups == sum of all ranks (n(n+1)/2).
    rng = np.random.default_rng(3)
    n_cells = 20
    col = (rng.random(n_cells) * (rng.random(n_cells) > 0.5)).astype(np.float64)
    groups = (rng.random(n_cells) > 0.5).astype(np.int64)
    X = sparse.csc_matrix(col.reshape(-1, 1))
    indptr = X.indptr.astype(np.int64)
    indices = X.indices.astype(np.int64)
    group_sizes = np.bincount(groups, minlength=2).astype(np.int64)
    gs = np.zeros((2, 1)); gq = np.zeros((2, 1))
    gn = np.zeros((2, 1), dtype=np.int64); rs = np.zeros((2, 1))
    RG._fused_stats_rank_sums_csc(
        indptr, indices, X.data.astype(np.float64), n_cells, groups,
        group_sizes, 2, gs, gq, gn, rs)
    total_rank = n_cells * (n_cells + 1) / 2.0
    np.testing.assert_allclose(rs.sum(), total_rank, rtol=1e-9)


# ==========================================================================
# _fused_all_csc — z-scores and p-values
# ==========================================================================

def test_fused_all_csc_zscore_pvalue_consistency():
    # z-score and p-value must satisfy p = erfc(|z|/sqrt2).
    rng = np.random.default_rng(4)
    n_cells, n_genes = 30, 4
    dense = (rng.random((n_cells, n_genes)) *
             (rng.random((n_cells, n_genes)) > 0.4)).astype(np.float64)
    groups = (rng.random(n_cells) > 0.5).astype(np.int64)
    X = sparse.csc_matrix(dense)
    # presort columns as the real path does
    indptr = X.indptr.astype(np.int32)
    indices = X.indices.astype(np.int32)
    data = X.data.astype(np.float64)
    RG._presort_csc_columns(indptr, indices, data)
    group_sizes = np.bincount(groups, minlength=2).astype(np.int64)

    scores = np.empty((2, n_genes)); pvals = np.empty((2, n_genes))
    g_sum = np.empty((2, n_genes)); total = np.empty(n_genes)
    RG._fused_all_csc(indptr, indices, data, n_cells, groups, group_sizes, 2,
                      scores, pvals, g_sum, total)
    # Relationship p = erfc(|z| / sqrt(2)) holds elementwise.
    ref_p = np.vectorize(lambda z: math.erfc(abs(z) / math.sqrt(2.0)))(scores)
    np.testing.assert_allclose(pvals, ref_p, rtol=1e-9)
    # total_sum equals the column-wise sum of the dense matrix.
    np.testing.assert_allclose(total, dense.sum(axis=0), rtol=1e-9)


# ==========================================================================
# _compute_topn_logfc
# ==========================================================================

def test_compute_topn_logfc_matches_formula():
    n_groups, n_genes, n_top = 1, 3, 2
    g_sum = np.array([[10.0, 4.0, 6.0]], dtype=np.float64)
    total = np.array([20.0, 8.0, 12.0], dtype=np.float64)
    top_idx = np.array([[0, 2]], dtype=np.int64)
    out = np.empty((n_top, n_groups), dtype=np.float64)
    inv_n_g = np.array([0.1], dtype=np.float64)     # n_g = 10
    inv_n_rest = np.array([0.05], dtype=np.float64)  # n_rest = 20
    RG._compute_topn_logfc(g_sum, total, top_idx, out, inv_n_g, inv_n_rest,
                           1.0, n_groups, n_top)
    # gene 0: mean_g = 10*0.1 = 1.0; mean_rest = (20-10)*0.05 = 0.5
    mg, mr = 1.0, 0.5
    ref0 = math.log2((math.expm1(mg) + 1e-9) / (math.expm1(mr) + 1e-9))
    assert out[0, 0] == pytest.approx(ref0, rel=1e-9)
    # gene 2: mean_g = 6*0.1 = 0.6; mean_rest = (12-6)*0.05 = 0.3
    mg2, mr2 = 0.6, 0.3
    ref2 = math.log2((math.expm1(mg2) + 1e-9) / (math.expm1(mr2) + 1e-9))
    assert out[1, 0] == pytest.approx(ref2, rel=1e-9)


# ==========================================================================
# _sparse_rankdata_csc — full per-cell rank matrix matches scipy
# ==========================================================================

def test_sparse_rankdata_csc_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    col = np.array([0.0, 2.0, 0.0, 5.0, 1.0], dtype=np.float64)
    n_cells = 5
    X = sparse.csc_matrix(col.reshape(-1, 1))
    ranks = np.empty((n_cells, 1), dtype=np.float64)
    RG._sparse_rankdata_csc(X.indptr.astype(np.int64),
                            X.indices.astype(np.int64),
                            X.data.astype(np.float64), n_cells, ranks)
    ref = scipy_stats.rankdata(col)
    np.testing.assert_allclose(ranks[:, 0], ref, rtol=1e-9)


def test_sparse_rankdata_csc_all_zero_column():
    n_cells = 4
    X = sparse.csc_matrix(np.zeros((n_cells, 1), dtype=np.float64))
    ranks = np.empty((n_cells, 1), dtype=np.float64)
    RG._sparse_rankdata_csc(X.indptr.astype(np.int64),
                            X.indices.astype(np.int64),
                            X.data.astype(np.float64), n_cells, ranks)
    # All-zero column: every cell gets the average rank (n+1)/2 = 2.5.
    np.testing.assert_allclose(ranks[:, 0], 2.5)


# ==========================================================================
# _bh_gather — fused BH + top-N gather
# ==========================================================================

def test_bh_gather_outputs():
    n_groups, n_genes, n_top = 1, 4, 2
    pvals = np.array([[0.01, 0.5, 0.04, 0.2]], dtype=np.float64)
    scores = np.array([[5.0, 1.0, 4.0, 2.0]], dtype=np.float64)
    top_idx = np.array([[0, 2]], dtype=np.int64)  # by descending score
    out_s = np.empty((n_top, n_groups))
    out_p = np.empty((n_top, n_groups))
    out_pa = np.empty((n_top, n_groups))
    RG._bh_gather(pvals, scores, top_idx, out_s, out_p, out_pa,
                  n_groups, n_genes, n_top)
    # Gathered scores/pvals for top indices.
    np.testing.assert_allclose(out_s[:, 0], [5.0, 4.0])
    np.testing.assert_allclose(out_p[:, 0], [0.01, 0.04])
    # Adjusted pvals match a full BH reference at those indices.
    ref_adj = np.empty_like(pvals)
    RG._bh_correct_2d(pvals.copy(), ref_adj, 1, n_genes)
    np.testing.assert_allclose(out_pa[:, 0], [ref_adj[0, 0], ref_adj[0, 2]],
                               rtol=1e-9)


# ==========================================================================
# _fast_rank_genes_groups dispatch + end-to-end parity
# ==========================================================================

def test_rank_genes_non_wilcoxon_delegates(monkeypatch):
    monkeypatch.setattr(RG, "_orig_rank_genes",
                        lambda: (lambda adata, groupby, **kw: "ORIG"))
    assert RG._fast_rank_genes_groups("A", "grp", method="t-test") == "ORIG"


def test_rank_genes_reference_not_rest_delegates(monkeypatch):
    monkeypatch.setattr(RG, "_orig_rank_genes",
                        lambda: (lambda adata, groupby, **kw: "ORIG"))
    out = RG._fast_rank_genes_groups("A", "grp", reference="0")
    assert out == "ORIG"


def test_rank_genes_tie_correct_delegates(monkeypatch):
    monkeypatch.setattr(RG, "_orig_rank_genes",
                        lambda: (lambda adata, groupby, **kw: "ORIG"))
    out = RG._fast_rank_genes_groups("A", "grp", tie_correct=True)
    assert out == "ORIG"


def _adata_groups(seed=0, n_cells=60, n_genes=20):
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.7, size=(n_cells, n_genes)).astype(np.float32)
    # Make group "1" over-express the first 3 genes.
    labels = np.array(["0", "1"] * (n_cells // 2))
    counts[labels == "1", :3] += rng.poisson(3, size=((labels == "1").sum(), 3))
    X = sparse.csr_matrix(counts)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    a = ad.AnnData(X)
    a.obs["grp"] = pd.Categorical(labels)
    a.var_names = [f"g{i}" for i in range(n_genes)]
    import scanpy as sc
    with __import__("autozyme").disabled():
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    return a


def test_rank_genes_scores_match_vanilla_wilcoxon():
    sc = pytest.importorskip("scanpy")
    import autozyme
    a = _adata_groups(seed=1)
    a_van = a.copy()
    with autozyme.disabled():
        sc.tl.rank_genes_groups(a_van, "grp", method="wilcoxon")
    RG._fast_rank_genes_groups(a, "grp", method="wilcoxon")

    # Compare z-scores for group "1": fast path stores names sorted by score;
    # vanilla does too. Build per-gene score dicts and compare.
    def score_map(adata, group):
        rec = adata.uns["rank_genes_groups"]
        names = rec["names"][group]
        scores = rec["scores"][group]
        return dict(zip(map(str, names), np.asarray(scores, dtype=np.float64)))

    fast = score_map(a, "1")
    van = score_map(a_van, "1")
    common = set(fast) & set(van)
    assert len(common) == a.n_vars
    for g in common:
        np.testing.assert_allclose(fast[g], van[g], rtol=1e-3, atol=1e-3)


def test_rank_genes_top_n_path_names_ordered():
    sc = pytest.importorskip("scanpy")
    a = _adata_groups(seed=2)
    RG._fast_rank_genes_groups(a, "grp", method="wilcoxon", n_genes=5)
    rec = a.uns["rank_genes_groups"]
    # n_genes=5 -> exactly 5 names per group, ordered by descending score.
    names1 = rec["names"]["1"]
    scores1 = np.asarray(rec["scores"]["1"], dtype=np.float64)
    assert len(names1) == 5
    assert np.all(np.diff(scores1) <= 1e-6)  # descending


def test_rank_genes_copy_returns_new():
    sc = pytest.importorskip("scanpy")
    a = _adata_groups(seed=3)
    out = RG._fast_rank_genes_groups(a, "grp", method="wilcoxon", copy=True)
    assert out is not None and out is not a
    assert "rank_genes_groups" not in a.uns
    assert "rank_genes_groups" in out.uns


def test_rank_genes_bonferroni_correction():
    sc = pytest.importorskip("scanpy")
    a = _adata_groups(seed=4)
    RG._fast_rank_genes_groups(a, "grp", method="wilcoxon",
                               corr_method="bonferroni")
    rec = a.uns["rank_genes_groups"]
    pvals = np.asarray(rec["pvals"]["1"], dtype=np.float64)
    pvals_adj = np.asarray(rec["pvals_adj"]["1"], dtype=np.float64)
    # Bonferroni: adj = min(p * n_genes, 1).
    np.testing.assert_allclose(
        pvals_adj, np.minimum(pvals * a.n_vars, 1.0), rtol=1e-9)
