"""End-to-end / wrapper-line tests for autozyme.cellphonedb.

Wave-1 (`test_cellphonedb_unit.py`) tested the numba kernels, build_clusters
(simple), filter, interacting_pair, shuffle_meta, and build_percent_result
(compact). This file covers the remaining COVERAGE-VISIBLE pure-python helpers
that the kernels + the wave-1/contract files don't reach:

  - `fast_call`: the scope guard's score_interactions=False (delegate) AND
    score_interactions=True (run-fully-upstream-under-disabled) branches, with
    `_orig_call` stubbed so no real cpdb database is needed.
  - `fast_percent_analysis`: the percent-mask + ContextVar stash.
  - `fast_build_percent_result`: the non-compact (packbits/unpackbits) fallback.
  - `fast_add_multidata_and_means_to_counts`: the merge + already-unique path.
  - `fast_build_clusters`: the complex-aggregation branch.
  - `fast_save_dfs_as_tsv`: passthrough to the captured upstream writer.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("cellphonedb")

import autozyme
from autozyme import cellphonedb as azcpdb


@pytest.fixture(autouse=True)
def _activate():
    autozyme.activate("cellphonedb")
    yield


def test_fast_call_default_delegates(monkeypatch):
    """score_interactions=False (default) calls _orig_call directly (no
    disabled() wrap)."""
    state = {"disabled": None}

    def _fake_orig(*args, **kwargs):
        state["disabled"] = autozyme.is_disabled()
        return "RESULT"

    monkeypatch.setattr(azcpdb, "_orig_call", _fake_orig)
    out = azcpdb.fast_call(cpdb_file_path="x", score_interactions=False)
    assert out == "RESULT"
    assert state["disabled"] is False  # ran with patches still active


def test_fast_call_score_interactions_runs_fully_upstream(monkeypatch):
    """score_interactions=True wraps the original in autozyme.disabled() so the
    untested scoring stage runs fully upstream (patches off)."""
    state = {"disabled": None}

    def _fake_orig(*args, **kwargs):
        state["disabled"] = autozyme.is_disabled()
        return "SCORED"

    monkeypatch.setattr(azcpdb, "_orig_call", _fake_orig)
    out = azcpdb.fast_call(score_interactions=True)
    assert out == "SCORED"
    assert state["disabled"] is True  # ran inside disabled()


def test_fast_percent_analysis_mask_and_contextvar():
    """fast_percent_analysis stashes the int 0/1 mask in the ContextVar and
    returns a DataFrame of the same shape, matching a direct threshold count."""
    azcpdb._real_pct_var.set(None)
    # 3 interactions x 2 cluster-pairs.
    interactions = pd.DataFrame(
        {"multidata_1_id": [10, 11, 12], "multidata_2_id": [20, 21, 22]},
        index=["i0", "i1", "i2"],
    )
    # percents indexed by gene id, columns are cluster names; cover every
    # referenced gene + cluster.
    percents = pd.DataFrame(
        {
            "cA": {10: 0.5, 11: 0.2, 12: 0.0, 20: 0.6, 21: 0.05, 22: 0.3},
            "cB": {10: 0.05, 11: 0.4, 12: 0.9, 20: 0.2, 21: 0.5, 22: 0.7},
        }
    )
    clusters = {"percents": percents}
    cluster_combinations = np.array([["cA", "cB"], ["cB", "cA"]], dtype=object)
    out = azcpdb.fast_percent_analysis(
        clusters, threshold=0.1, interactions=interactions,
        cluster_combinations=cluster_combinations, separator="|",
    )
    assert out.shape == (3, 2)
    stashed = azcpdb._real_pct_var.get()
    assert stashed is not None and stashed.shape == (3, 2)
    assert set(np.unique(stashed)).issubset({0, 1})


def test_build_percent_result_fallback_packbits_path():
    """When statistical_mean_analysis is a list of packed uint8 arrays (NOT a
    _CompactStats), build_percent_result takes the unpackbits accumulation
    branch and forces p=1 where real mean / pct is zero."""
    real_mean = pd.DataFrame([[2.0, 0.0], [1.0, 3.0]])
    real_pct = np.array([[1, 1], [1, 0]])
    # Two permutations, each a packbits of a (2x2)=4-bit count grid.
    grid0 = np.array([1, 0, 1, 1], dtype=np.uint8)
    grid1 = np.array([0, 0, 1, 1], dtype=np.uint8)
    packed = [np.packbits(grid0), np.packbits(grid1)]
    base = pd.DataFrame(index=["i0", "i1"], columns=["p0", "p1"])

    out = azcpdb.fast_build_percent_result(
        real_mean, real_pct, packed, None, None, base, "|"
    )
    # Reference: mean of unpacked grids, then masked to 1.
    ref = (grid0.reshape(2, 2).astype(float) + grid1.reshape(2, 2)) / 2.0
    mask = (real_mean.values == 0) | (real_pct == 0)
    ref[mask] = 1.0
    np.testing.assert_allclose(out.values, ref)


def test_add_multidata_and_means_to_counts_unique_index():
    """fast_add_multidata_and_means_to_counts merges gene metadata and (with a
    unique id_multidata) skips the groupby().mean()."""
    counts = pd.DataFrame(
        {"cell0": [1.0, 2.0], "cell1": [3.0, 4.0]},
        index=["ENSG1", "ENSG2"],
    ).astype(np.float32)
    genes = pd.DataFrame({
        "ensembl": ["ENSG1", "ENSG2"],
        "id_multidata": [100, 200],
        "gene_name": ["G1", "G2"],
        "hgnc_symbol": ["G1", "G2"],
    })
    out_counts, relations = azcpdb.fast_add_multidata_and_means_to_counts(
        counts, genes, counts_data="ensembl"
    )
    assert list(out_counts.index) == [100, 200]
    assert set(out_counts.columns) == {"cell0", "cell1"}
    assert out_counts.index.is_unique
    assert set(relations.columns) >= {"id_multidata", "ensembl", "gene_name"}


def test_build_clusters_with_complexes():
    """fast_build_clusters' complex-aggregation branch (min over protein rows)
    produces the extra complex rows in means + percents."""
    meta = pd.DataFrame(
        {"cell_type": pd.Categorical(["A", "A", "B", "B"])},
        index=[f"c{i}" for i in range(4)],
    )
    counts = pd.DataFrame(
        {f"c{i}": [float(i + 1), float(i + 2), float(i)] for i in range(4)},
        index=[10, 11, 12],
    ).astype(np.float32)
    # complex "cx" is the min over protein rows 0 and 1 (positional).
    complex_to_protein_row_ids = {999: [0, 1]}
    res = azcpdb.fast_build_clusters(
        meta, counts, complex_to_protein_row_ids, skip_percent=False
    )
    assert 999 in res["means"].index
    assert 999 in res["percents"].index
    # complex mean row == elementwise min of the two protein-row means.
    simple = res["means"].loc[[10, 11]]
    np.testing.assert_allclose(
        res["means"].loc[999].values, simple.min(axis=0).values, rtol=1e-6
    )


def test_save_dfs_as_tsv_passthrough(monkeypatch):
    """fast_save_dfs_as_tsv forwards verbatim to the captured upstream writer."""
    seen = {}

    def _fake(out, suffix, analysis_name, name2df):
        seen["args"] = (out, suffix, analysis_name, name2df)
        return "WROTE"

    monkeypatch.setattr(azcpdb, "_orig_save_dfs_as_tsv", _fake)
    res = azcpdb.fast_save_dfs_as_tsv("/tmp/x", "suf", "name", {"a": 1})
    assert res == "WROTE"
    assert seen["args"] == ("/tmp/x", "suf", "name", {"a": 1})
