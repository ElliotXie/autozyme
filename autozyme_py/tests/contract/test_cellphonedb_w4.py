"""Wave-4 reachable-orchestration tests for autozyme.cellphonedb.

Waves 1-3 covered the numba kernels, the helper functions, the fast_call scope
guard, percent_analysis/build_percent_result/build_clusters, and the heavy
`fast_shuffled_analysis` fused permutation loop. This file mops up the remaining
COVERAGE-VISIBLE orchestration lines neither reached:

  - `fast_filter_interactions_by_counts` with a NON-empty `complex_composition`
    (src lines 395-396): complex/protein multidata ids are folded into the
    membership set, so an interaction referencing only a complex id is kept.
  - `fast_add_multidata_and_means_to_counts`:
      * the int->float32 cast branch (src line 416) when counts isn't float32,
      * the NON-unique multidata-index `groupby(...).mean()` branch (src line 418)
        when two genes map to the same `id_multidata`.
  - the smoke recipe IO ends: `_smoke_load` (src 452-477, incl. the
    FileNotFoundError guard for a missing input) and `_smoke_save` (src 524-526,
    writing means/pvalues/significant_means tsv from the in-memory result dict).

NOT covered here (documented in the report): `_smoke_call` (src 498-516) runs a
real `cpdb_statistical_analysis_method.call(... iterations=1000 ...)` which needs
the real CellphoneDB v5 database zip + correctly-formatted counts/meta TSVs on
disk -- large data, multi-second; out of reach for a self-contained contract
test. The numba kernel bodies (`_build_onehot_flat` / `_gather_count_kernel_active`,
src 100-138) are invisible to coverage.py even though waves 1+3 exercise them.
"""
from __future__ import annotations

import os
import tempfile

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("numba")
pytest.importorskip("cellphonedb")

from autozyme import cellphonedb as azcpdb


# --------------------------------------------------------------------------
# fast_filter_interactions_by_counts: the non-empty complex_composition branch
# --------------------------------------------------------------------------
def test_filter_interactions_by_counts_with_complex_composition():
    """A non-empty complex_composition folds complex + protein multidata ids into
    the membership set, so an interaction whose partner is a *complex* id (not a
    direct counts row) is retained."""
    counts = pd.DataFrame(np.arange(12).reshape(3, 4), index=[10, 11, 12])
    interactions = pd.DataFrame(
        {
            # row 0: 10 & 12 both direct counts rows  -> keep
            # row 1: 11 direct, 100 is a complex id    -> keep (via complex set)
            # row 2: 99 absent everywhere              -> drop
            "multidata_1_id": [10, 11, 99],
            "multidata_2_id": [12, 100, 10],
        }
    )
    complex_composition = pd.DataFrame(
        {"complex_multidata_id": [100], "protein_multidata_id": [11]}
    )
    out = azcpdb.fast_filter_interactions_by_counts(
        interactions, counts, complex_composition
    )
    assert list(out.index) == [0, 1]


def test_filter_interactions_empty_complex_composition_branch():
    """The empty-complex_composition branch (no complex/protein ids folded in)
    only keeps interactions whose BOTH partners are direct counts rows."""
    counts = pd.DataFrame(np.arange(8).reshape(2, 4), index=[10, 11])
    interactions = pd.DataFrame(
        {"multidata_1_id": [10, 10], "multidata_2_id": [11, 999]}
    )
    out = azcpdb.fast_filter_interactions_by_counts(
        interactions, counts, pd.DataFrame()
    )
    assert list(out.index) == [0]


# --------------------------------------------------------------------------
# fast_add_multidata_and_means_to_counts: int->float32 + non-unique groupby
# --------------------------------------------------------------------------
def test_add_multidata_int_cast_and_nonunique_groupby():
    """Two genes mapping to the same `id_multidata` make the post-merge index
    non-unique (src line 418 `groupby(index).mean()`), and an int counts frame
    triggers the float32 cast (src line 416)."""
    genes = pd.DataFrame(
        {
            "id_multidata": [1, 1, 2],   # g0, g1 -> same multidata 1 (non-unique)
            "ensembl": ["g0", "g1", "g2"],
            "gene_name": ["G0", "G1", "G2"],
            "hgnc_symbol": ["G0", "G1", "G2"],
        },
        index=["g0", "g1", "g2"],
    )
    counts = pd.DataFrame(
        {"cellA": [1, 2, 3], "cellB": [4, 5, 6]},  # int dtype -> astype float32
        index=["g0", "g1", "g2"],
    )
    out_counts, relations = azcpdb.fast_add_multidata_and_means_to_counts(
        counts, genes, counts_data="ensembl"
    )
    # multidata 1 = mean(g0, g1); multidata 2 = g2.
    assert list(out_counts.index) == [1, 2]
    assert out_counts.index.is_unique
    assert all(dt == np.float32 for dt in out_counts.dtypes)
    # multidata 1, cellA = mean(1, 2) = 1.5
    assert out_counts.loc[1, "cellA"] == pytest.approx(1.5)
    assert len(relations) == 3


# --------------------------------------------------------------------------
# Smoke recipe IO ends: _smoke_load (+ FNF guard) and _smoke_save
# --------------------------------------------------------------------------
def _make_smoke_task_dir(with_inputs=True):
    import yaml

    td = tempfile.mkdtemp(prefix="autozyme_cpdb_w4_")
    os.makedirs(os.path.join(td, "data", "_refs"), exist_ok=True)
    os.makedirs(os.path.join(td, "data", "small"), exist_ok=True)
    if with_inputs:
        # _smoke_load only checks os.path.isfile on these three paths.
        open(os.path.join(td, "data", "_refs", "cellphonedb_v5.0.0.zip"), "w").close()
        open(os.path.join(td, "data", "small", "counts.tsv"), "w").close()
        open(os.path.join(td, "data", "small", "meta.tsv"), "w").close()
    with open(os.path.join(td, "task.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"datasets": [{"tier": "small", "path": "data/small"}]}, f
        )
    return td


def test_smoke_load_resolves_inputs_and_seeds():
    """`_smoke_load` reads task.yaml, locates the db/counts/meta paths, sets the
    mp start method + RNG seed, and returns the three input paths."""
    td = _make_smoke_task_dir(with_inputs=True)
    inputs = azcpdb._smoke_load(td, "small")
    assert set(inputs.keys()) == {"cpdb_db", "counts_file", "meta_file"}
    for v in inputs.values():
        assert os.path.isfile(v)


def test_smoke_load_raises_on_missing_input():
    """`_smoke_load` raises FileNotFoundError when an input is missing (the
    `not os.path.isfile` guard, src lines 461-463)."""
    td = _make_smoke_task_dir(with_inputs=False)
    with pytest.raises(FileNotFoundError):
        azcpdb._smoke_load(td, "small")


def test_smoke_save_writes_three_frames():
    """`_smoke_save` writes means/pvalues/significant_means tsv from the in-memory
    result dict (src lines 524-526)."""
    result = {
        "means": pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
        "pvalues": pd.DataFrame({"a": [0.1, 0.2]}),
        "significant_means": pd.DataFrame({"a": [1.0, np.nan]}),
    }
    out_dir = tempfile.mkdtemp(prefix="autozyme_cpdb_w4_out_")
    azcpdb._smoke_save(result, out_dir)
    for name in ("means.tsv", "pvalues.tsv", "significant_means.tsv"):
        path = os.path.join(out_dir, name)
        assert os.path.isfile(path)
        # tab-separated, round-trippable.
        back = pd.read_csv(path, sep="\t")
        assert len(back) == 2
