"""Unit tests for the pure numpy/numba/pandas helpers in autozyme.cellphonedb.

The contract test (test_cellphonedb.py) drives the statistical-analysis pipeline
end to end. Here we test the self-contained kernels directly:

  - _build_onehot_flat           numba one-hot scatter, vs numpy
  - _gather_count_kernel_active  fused gather+predicate+count, vs numpy
  - _CompactStats                sentinel container
  - fast_build_clusters          one-hot matmul cluster means, vs groupby
  - fast_filter_interactions_by_counts / fast_interacting_pair_build  pandas paths
  - fast_shuffle_meta            permutation preserves the value multiset
  - fast_build_percent_result    compact-path p-value computation

cellphonedb must import for the module to load; none of the kernels above use
cellphonedb internals at runtime.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("numba")
pytest.importorskip("cellphonedb")

from autozyme import cellphonedb as azcpdb


# --------------------------------------------------------------------------
# _build_onehot_flat
# --------------------------------------------------------------------------
def test_build_onehot_flat_matches_numpy():
    rng = np.random.default_rng(0)
    b_size, n_cells, n_clusters = 4, 12, 3
    codes_batch = rng.integers(0, n_clusters, size=(b_size, n_cells)).astype(np.int64)
    onehot = np.empty((n_cells, b_size * n_clusters), dtype=np.float32)
    azcpdb._build_onehot_flat(codes_batch, n_clusters, onehot)
    # Reference: each column block b holds the one-hot of codes_batch[b].
    ref = np.zeros((n_cells, b_size * n_clusters), dtype=np.float32)
    for b in range(b_size):
        for c in range(n_cells):
            ref[c, b * n_clusters + codes_batch[b, c]] = 1.0
    np.testing.assert_array_equal(onehot, ref)


def test_build_onehot_flat_rows_sum_to_one_per_block():
    rng = np.random.default_rng(1)
    b_size, n_cells, n_clusters = 3, 10, 4
    codes_batch = rng.integers(0, n_clusters, size=(b_size, n_cells)).astype(np.int64)
    onehot = np.empty((n_cells, b_size * n_clusters), dtype=np.float32)
    azcpdb._build_onehot_flat(codes_batch, n_clusters, onehot)
    blocks = onehot.reshape(n_cells, b_size, n_clusters)
    np.testing.assert_array_equal(blocks.sum(axis=2), np.ones((n_cells, b_size)))


# --------------------------------------------------------------------------
# _gather_count_kernel_active
# --------------------------------------------------------------------------
def test_gather_count_kernel_active_matches_numpy():
    rng = np.random.default_rng(2)
    b_size, n_g, n_c = 20, 5, 4
    all_means = rng.standard_normal((b_size, n_g, n_c)).astype(np.float32)
    n_active = 7
    g1 = rng.integers(0, n_g, n_active).astype(np.int64)
    g2 = rng.integers(0, n_g, n_active).astype(np.int64)
    c1 = rng.integers(0, n_c, n_active).astype(np.int64)
    c2 = rng.integers(0, n_c, n_active).astype(np.int64)
    two_real = rng.standard_normal(n_active).astype(np.float32)
    # Output grid indices (interaction i, cluster-pair j).
    out_i = np.arange(n_active, dtype=np.int64)
    out_j = np.zeros(n_active, dtype=np.int64)
    count_acc = np.zeros((n_active, 1), dtype=np.int32)
    azcpdb._gather_count_kernel_active(
        all_means, g1, g2, c1, c2, out_i, out_j, two_real, count_acc
    )
    # Reference count: per active entry, count batches with x>0 & y>0 & x+y>tr.
    for k in range(n_active):
        x = all_means[:, g1[k], c1[k]]
        y = all_means[:, g2[k], c2[k]]
        ref = int(np.sum((x > 0) & (y > 0) & (x + y > two_real[k])))
        assert count_acc[k, 0] == ref


def test_gather_count_kernel_accumulates():
    # Two active entries writing to the same (i, j) slot must sum.
    b_size = 10
    all_means = np.ones((b_size, 2, 2), dtype=np.float32)  # all > 0
    g1 = np.array([0, 1], dtype=np.int64)
    g2 = np.array([0, 1], dtype=np.int64)
    c1 = np.array([0, 1], dtype=np.int64)
    c2 = np.array([0, 1], dtype=np.int64)
    two_real = np.array([0.0, 0.0], dtype=np.float32)  # 1+1 > 0 always true
    out_i = np.array([0, 0], dtype=np.int64)
    out_j = np.array([0, 0], dtype=np.int64)
    count_acc = np.zeros((1, 1), dtype=np.int32)
    azcpdb._gather_count_kernel_active(
        all_means, g1, g2, c1, c2, out_i, out_j, two_real, count_acc
    )
    assert count_acc[0, 0] == 2 * b_size


# --------------------------------------------------------------------------
# _CompactStats
# --------------------------------------------------------------------------
def test_compact_stats_holds_count_and_iters():
    count = np.array([[1, 2], [3, 4]], dtype=np.int32)
    cs = azcpdb._CompactStats(count, 1000)
    assert cs.n_iters == 1000
    np.testing.assert_array_equal(cs.count, count)


# --------------------------------------------------------------------------
# fast_build_clusters
# --------------------------------------------------------------------------
def test_build_clusters_means_match_groupby():
    rng = np.random.default_rng(3)
    n_cells, n_genes, n_clusters = 30, 6, 3
    counts_vals = rng.random((n_genes, n_cells)).astype(np.float32)
    cell_types = rng.integers(0, n_clusters, n_cells)
    gene_names = [f"g{i}" for i in range(n_genes)]
    cell_names = [f"c{i}" for i in range(n_cells)]
    counts = pd.DataFrame(counts_vals, index=gene_names, columns=cell_names)
    meta = pd.DataFrame(
        {"cell_type": [f"ct{t}" for t in cell_types]}, index=cell_names
    )
    res = azcpdb.fast_build_clusters(meta, counts, {}, skip_percent=False)

    # Reference cluster means via pandas groupby on the transpose.
    long = counts.T.copy()
    long["cell_type"] = [f"ct{t}" for t in cell_types]
    ref_means = long.groupby("cell_type", observed=True).mean().T
    got = res["means"][ref_means.columns]
    np.testing.assert_allclose(
        got.values.astype(np.float64), ref_means.values.astype(np.float64),
        rtol=1e-5, atol=1e-6,
    )
    # Percents = fraction of cells with count > 0 per cluster.
    pos = (counts.T > 0).astype(float)
    pos["cell_type"] = [f"ct{t}" for t in cell_types]
    ref_pct = pos.groupby("cell_type", observed=True).mean().T
    got_pct = res["percents"][ref_pct.columns]
    np.testing.assert_allclose(
        got_pct.values.astype(np.float64), ref_pct.values.astype(np.float64),
        rtol=1e-5, atol=1e-6,
    )


def test_build_clusters_skip_percent_empty_pcts():
    rng = np.random.default_rng(4)
    counts = pd.DataFrame(
        rng.random((4, 10)).astype(np.float32),
        index=[f"g{i}" for i in range(4)],
        columns=[f"c{i}" for i in range(10)],
    )
    meta = pd.DataFrame(
        {"cell_type": (["a"] * 5 + ["b"] * 5)},
        index=[f"c{i}" for i in range(10)],
    )
    res = azcpdb.fast_build_clusters(meta, counts, {}, skip_percent=True)
    # percents frame exists but holds no numeric data (all NaN).
    assert res["percents"].isna().all().all()


# --------------------------------------------------------------------------
# fast_filter_interactions_by_counts
# --------------------------------------------------------------------------
def test_filter_interactions_by_counts():
    interactions = pd.DataFrame({
        "multidata_1_id": [1, 2, 99, 3],
        "multidata_2_id": [2, 3, 4, 100],
    })
    counts = pd.DataFrame(index=[1, 2, 3, 4], data={"x": [0, 0, 0, 0]})
    empty_complex = pd.DataFrame(columns=["complex_multidata_id", "protein_multidata_id"])
    out = azcpdb.fast_filter_interactions_by_counts(interactions, counts, empty_complex)
    # row 0 (1,2) and row 1 (2,3) are fully covered; rows 2,3 reference 99/100.
    assert list(out.index) == [0, 1]


# --------------------------------------------------------------------------
# fast_interacting_pair_build
# --------------------------------------------------------------------------
def test_interacting_pair_build_chooses_complex_or_gene_name():
    interactions = pd.DataFrame({
        "is_complex_1": [True, False],
        "is_complex_2": [False, True],
        "name_1": ["CX1", "CXa"],
        "gene_name_1": ["g1", "g2"],
        "name_2": ["CX2", "CXb"],
        "gene_name_2": ["h1", "h2"],
    })
    pair = azcpdb.fast_interacting_pair_build(interactions)
    # row0: is_complex_1=True -> name_1 (CX1); is_complex_2=False -> gene_name_2 (h1)
    assert pair.iloc[0] == "CX1_h1"
    # row1: is_complex_1=False -> gene_name_1 (g2); is_complex_2=True -> name_2 (CXb)
    assert pair.iloc[1] == "g2_CXb"
    assert pair.name == "interacting_pair"


# --------------------------------------------------------------------------
# fast_shuffle_meta
# --------------------------------------------------------------------------
def test_shuffle_meta_preserves_value_multiset():
    np.random.seed(0)
    meta = pd.DataFrame({
        "cell_type": pd.Categorical(["a", "b", "a", "c", "b", "a"]),
    })
    out = azcpdb.fast_shuffle_meta(meta)
    # Same categories, same value counts; original frame untouched.
    assert sorted(out["cell_type"].tolist()) == sorted(meta["cell_type"].tolist())
    assert list(out["cell_type"].cat.categories) == list(meta["cell_type"].cat.categories)
    assert meta["cell_type"].tolist() == ["a", "b", "a", "c", "b", "a"]


# --------------------------------------------------------------------------
# fast_build_percent_result (compact path)
# --------------------------------------------------------------------------
def test_build_percent_result_compact_path():
    real_mean = pd.DataFrame([[2.0, 0.0], [1.0, 3.0]])
    real_pct = np.array([[1, 1], [1, 0]])
    count = np.array([[400, 0], [250, 600]], dtype=np.int32)
    stats = [azcpdb._CompactStats(count, 1000)]
    base = pd.DataFrame(index=["i0", "i1"], columns=["p0", "p1"])
    out = azcpdb.fast_build_percent_result(
        real_mean, real_pct, stats, None, None, base, "|"
    )
    # p-value = count / n_iters, then forced to 1 where real_mean==0 or pct==0.
    expected = count.astype(float) / 1000.0
    mask = (real_mean.values == 0) | (real_pct == 0)
    expected[mask] = 1.0
    np.testing.assert_allclose(out.values, expected)
    assert list(out.index) == ["i0", "i1"]
