"""Wave-3 heavy-path tests for autozyme.cellphonedb.fast_shuffled_analysis.

Wave-1 (`test_cellphonedb_unit.py`) tested the numba kernels and helper
functions in isolation. Wave-2 (`test_cellphonedb_e2e.py`) covered the
pure-python wrapper/dispatch lines (fast_call scope guard, percent_analysis,
build_percent_result, add_multidata, build_clusters, save_dfs_as_tsv).

What NEITHER reached is `fast_shuffled_analysis` (lines 192-320), the heaviest
end-to-end patch path: the fused permutation inner loop that batches BATCH=50
shuffles through one sgemm + the active-only numba gather/count kernel and
returns a `_CompactStats` count grid. Wave-2 explicitly skipped it as too heavy.

This file drives it directly with the smallest synthetic CellphoneDB-shaped
inputs (a few genes / cells / cluster-pairs) and asserts the patched
shuffled-count grid is BIT-EXACT against an independent brute-force numpy
permutation reference that replicates the same RNG consumption order (seed +
BATCH=50 tile-then-shuffle). It covers:

  - the simple-gene-only permutation count path (no complexes),
  - the complex-aggregation path (complex_to_protein_ids -> min over protein
    rows, the referenced_complex / complex_protein_rows_ref machinery),
  - the `_real_pct_var is None` active-mask fallback (active = real_mean > 0),
  - the full _CompactStats -> fast_build_percent_result consumption (p = count
    / n_iters, masked to 1 where real_mean / pct is 0).

cellphonedb must import for the module to load; the kernels themselves use no
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
# Independent brute-force reference: replays the SAME RNG consumption order as
# fast_shuffled_analysis (np.random.seed already set by caller; BATCH=50 tile
# + per-row np.random.shuffle), computes per-permutation cluster means via a
# one-hot matmul, and counts (x>0 & y>0 & x+y>2*real) per active entry.
# --------------------------------------------------------------------------
def _brute_force_counts(iterations, meta, counts, interactions,
                        cluster_combinations, complex_to_protein_ids,
                        real_mean, active_mask):
    cat = meta["cell_type"].astype("category")
    cluster_names = cat.cat.categories.to_numpy()
    codes_orig = cat.cat.codes.to_numpy().copy().astype(np.int64)
    n_cells = codes_orig.shape[0]
    n_clusters = len(cluster_names)
    inv = (1.0 / np.bincount(codes_orig, minlength=n_clusters)).astype(np.float32)

    counts_vals = np.ascontiguousarray(counts.values, dtype=np.float32)
    counts_index = list(counts.index.to_numpy())
    n_simple = len(counts_index)
    complex_ids = list(complex_to_protein_ids.keys()) if complex_to_protein_ids else []

    id_to_row = {g: i for i, g in enumerate(counts_index)}
    for i, cid in enumerate(complex_ids):
        id_to_row[cid] = n_simple + i
    name_to_col = {n: i for i, n in enumerate(cluster_names)}

    g1 = np.array([id_to_row[g] for g in interactions["multidata_1_id"].values])
    g2 = np.array([id_to_row[g] for g in interactions["multidata_2_id"].values])
    c1 = np.array([name_to_col[c] for c in cluster_combinations[:, 0]])
    c2 = np.array([name_to_col[c] for c in cluster_combinations[:, 1]])
    two_real = (real_mean.values * 2.0).astype(np.float32)

    n_i, n_j = real_mean.shape
    count_ref = np.zeros((n_i, n_j), dtype=np.int32)

    BATCH = 50
    for batch_start in range(0, iterations, BATCH):
        b_size = min(BATCH, iterations - batch_start)
        codes_batch = np.tile(codes_orig, (b_size, 1))
        for b in range(b_size):
            np.random.shuffle(codes_batch[b])
        for b in range(b_size):
            onehot = np.zeros((n_cells, n_clusters), dtype=np.float32)
            onehot[np.arange(n_cells), codes_batch[b]] = 1.0
            simple_means = (counts_vals @ onehot) * inv
            all_means = np.empty((n_simple + len(complex_ids), n_clusters),
                                 dtype=np.float32)
            all_means[:n_simple] = simple_means
            for ci, cid in enumerate(complex_ids):
                prows = np.asarray(complex_to_protein_ids[cid])
                all_means[n_simple + ci] = simple_means[prows].min(axis=0)
            for i in range(n_i):
                for j in range(n_j):
                    if not active_mask[i, j]:
                        continue
                    x = all_means[g1[i], c1[j]]
                    y = all_means[g2[i], c2[j]]
                    if (x > 0.0) and (y > 0.0) and (x + y > two_real[i, j]):
                        count_ref[i, j] += 1
    return count_ref


# --------------------------------------------------------------------------
# Tiny synthetic CellphoneDB-shaped inputs.
# --------------------------------------------------------------------------
def _simple_inputs(seed=0):
    n_cells = 6
    meta = pd.DataFrame(
        {"cell_type": pd.Categorical(["A", "A", "A", "B", "B", "B"])},
        index=[f"c{i}" for i in range(n_cells)],
    )
    gene_ids = [10, 11, 12, 13]
    rng = np.random.default_rng(seed)
    counts = pd.DataFrame(
        rng.random((len(gene_ids), n_cells)).astype(np.float32),
        index=gene_ids, columns=meta.index,
    )
    interactions = pd.DataFrame(
        {"multidata_1_id": [10, 11, 12], "multidata_2_id": [11, 13, 10]},
        index=["i0", "i1", "i2"],
    )
    cc = np.array([["A", "A"], ["A", "B"], ["B", "A"], ["B", "B"]], dtype=object)
    real_mean = pd.DataFrame(
        np.abs(rng.random((3, 4))).astype(np.float64), index=interactions.index
    )
    return meta, counts, interactions, cc, real_mean


def test_shuffled_analysis_simple_path_matches_brute_force():
    """fast_shuffled_analysis (no complexes) count grid is bit-exact vs an
    independent numpy permutation reference using the same seed."""
    meta, counts, interactions, cc, real_mean = _simple_inputs(seed=0)
    real_pct = np.ones(real_mean.shape, dtype=int)
    active_mask = (real_mean.values != 0) & (real_pct != 0)

    azcpdb._real_pct_var.set(real_pct)
    np.random.seed(123)
    out = azcpdb.fast_shuffled_analysis(
        100, meta, counts, interactions, cc, {}, real_mean,
        threads=1, separator="|",
    )
    assert isinstance(out, list) and len(out) == 1
    stats = out[0]
    assert isinstance(stats, azcpdb._CompactStats)
    assert stats.n_iters == 100
    assert stats.count.shape == real_mean.shape

    np.random.seed(123)
    ref = _brute_force_counts(
        100, meta, counts, interactions, cc, {}, real_mean, active_mask
    )
    np.testing.assert_array_equal(stats.count, ref)


def test_shuffled_analysis_complex_path_matches_brute_force():
    """The complex-aggregation branch (complex id -> min over protein rows) is
    bit-exact vs the brute-force reference that builds the same complex rows."""
    n_cells = 8
    meta = pd.DataFrame(
        {"cell_type": pd.Categorical(["A", "A", "A", "A", "B", "B", "B", "B"])},
        index=[f"c{i}" for i in range(n_cells)],
    )
    gene_ids = [10, 11, 12, 13]
    rng = np.random.default_rng(3)
    counts = pd.DataFrame(
        rng.random((len(gene_ids), n_cells)).astype(np.float32),
        index=gene_ids, columns=meta.index,
    )
    # complex 999 = elementwise min over positional protein rows 0 and 1.
    complex_to_protein_ids = {999: [0, 1]}
    interactions = pd.DataFrame(
        {"multidata_1_id": [999, 11, 12], "multidata_2_id": [12, 999, 13]},
        index=["i0", "i1", "i2"],
    )
    cc = np.array([["A", "A"], ["A", "B"], ["B", "B"]], dtype=object)
    real_mean = pd.DataFrame(
        np.abs(rng.random((3, 3))).astype(np.float64), index=interactions.index
    )
    real_pct = np.ones(real_mean.shape, dtype=int)
    active_mask = (real_mean.values != 0) & (real_pct != 0)

    azcpdb._real_pct_var.set(real_pct)
    np.random.seed(7)
    out = azcpdb.fast_shuffled_analysis(
        100, meta, counts, interactions, cc, complex_to_protein_ids, real_mean,
        threads=1, separator="|",
    )
    stats = out[0]

    np.random.seed(7)
    ref = _brute_force_counts(
        100, meta, counts, interactions, cc, complex_to_protein_ids,
        real_mean, active_mask,
    )
    np.testing.assert_array_equal(stats.count, ref)


def test_shuffled_analysis_no_pct_var_active_mask_fallback():
    """When _real_pct_var is None, the active mask falls back to real_mean > 0
    (line 238-239). Entries with real_mean == 0 are inactive and stay at count
    0; active entries match the brute-force reference."""
    meta, counts, interactions, cc, real_mean = _simple_inputs(seed=5)
    # Force a zero real-mean entry so the >0 active mask actually excludes it.
    rm = real_mean.copy()
    rm.iloc[0, 1] = 0.0
    active_mask = rm.values > 0

    azcpdb._real_pct_var.set(None)
    np.random.seed(321)
    out = azcpdb.fast_shuffled_analysis(
        80, meta, counts, interactions, cc, {}, rm, threads=1, separator="|",
    )
    stats = out[0]
    # The zeroed (inactive) entry is never counted.
    assert stats.count[0, 1] == 0

    np.random.seed(321)
    ref = _brute_force_counts(
        80, meta, counts, interactions, cc, {}, rm, active_mask
    )
    np.testing.assert_array_equal(stats.count, ref)


def test_shuffled_analysis_feeds_build_percent_result():
    """End-to-end: the _CompactStats from fast_shuffled_analysis flows into
    fast_build_percent_result and yields p = count / n_iters, masked to 1 where
    real_mean or real_pct is 0 -- the real handoff the patch relies on."""
    meta, counts, interactions, cc, real_mean = _simple_inputs(seed=2)
    real_pct = np.ones(real_mean.shape, dtype=int)
    # Make one entry's real_pct 0 so the percent mask is exercised.
    real_pct[2, 0] = 0

    azcpdb._real_pct_var.set(real_pct)
    np.random.seed(99)
    stats_list = azcpdb.fast_shuffled_analysis(
        100, meta, counts, interactions, cc, {}, real_mean,
        threads=1, separator="|",
    )
    base = pd.DataFrame(
        index=interactions.index,
        columns=(pd.Series(cc[:, 0]) + "|" + pd.Series(cc[:, 1])).values,
    )
    pvals = azcpdb.fast_build_percent_result(
        real_mean, real_pct, stats_list, interactions, cc, base, "|"
    )
    count = stats_list[0].count
    expected = count.astype(np.float64) / 100.0
    mask = (real_mean.values == 0) | (real_pct == 0)
    expected[mask] = 1.0
    np.testing.assert_allclose(pvals.values, expected)
    # p-values are valid probabilities.
    assert np.all((pvals.values >= 0.0) & (pvals.values <= 1.0))
