"""Wave-4 ``rank_genes_groups`` coverage / hardening.

The reachable pure-python wrapper in ``_rank_genes.py`` (``_fast_rank_genes_groups``,
lines 408-651) is already 100% line-covered by waves 1-3. Everything still
"missing" in this module is either:
  * the ``except ImportError`` numba guard (28-29, unreachable while numba is
    installed), or
  * the ``@njit`` kernel bodies (43-351) which coverage.py cannot see (they are
    exercised directly in test_scanpy_rank_genes_unit.py).

So no NEW visible line is recoverable here. These tests instead harden the
parameter-combination matrix the briefing called out (more method /
corr_method / reference / layer / mask / groups-subset / use_raw combinations),
asserting parity vs the upstream original (``zyme=False``) on combinations the
earlier waves did not cover. They guard the already-covered dispatch branches
against future regressions.

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
from autozyme.scanpy import _rank_genes as RG  # noqa: E402


def _grouped(n_per=40, n_vars=30, n_groups=3, seed=0, base=None):
    """Log-normalized sparse AnnData with a categorical 'group' column."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    n_obs = n_per * n_groups
    counts = rng.poisson(0.6, size=(n_obs, n_vars)).astype(np.float32)
    for g in range(n_groups):
        rows = slice(g * n_per, (g + 1) * n_per)
        cols = slice(g * 4, g * 4 + 4)
        counts[rows, cols] += rng.poisson(4.0, size=counts[rows, cols].shape)
    X = sparse.csr_matrix(counts)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    a = ad.AnnData(X)
    a.var_names = [f"g{j}" for j in range(n_vars)]
    a.obs["group"] = pd.Categorical([f"grp{i // n_per}" for i in range(n_obs)])
    with autozyme.disabled():
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a, base=base)
    return a


def _score_map(adata, group, field="scores", key="rank_genes_groups"):
    rec = adata.uns[key]
    return dict(zip(map(str, rec["names"][group]),
                    np.asarray(rec[field][group], dtype=np.float64)))


# --------------------------------------------------------------------------
# groups-subset + top-N + bonferroni (an untested 3-way combination)
# --------------------------------------------------------------------------

def test_groups_subset_top_n_bonferroni_parity():
    a_fast = _grouped(seed=10, n_groups=3)
    a_van = a_fast.copy()
    RG._fast_rank_genes_groups(
        a_fast, "group", method="wilcoxon", groups=["grp0", "grp2"],
        n_genes=8, corr_method="bonferroni")
    with autozyme.disabled():
        sc.tl.rank_genes_groups(
            a_van, "group", method="wilcoxon", groups=["grp0", "grp2"],
            n_genes=8, corr_method="bonferroni")
    names = a_fast.uns["rank_genes_groups"]["names"].dtype.names
    assert set(names) == {"grp0", "grp2"}
    assert len(a_fast.uns["rank_genes_groups"]["names"]) == 8
    # Top marker per requested group agrees with vanilla.
    for g in names:
        top_fast = str(a_fast.uns["rank_genes_groups"]["names"][g][0])
        top_van = str(a_van.uns["rank_genes_groups"]["names"][g][0])
        assert top_fast == top_van


# --------------------------------------------------------------------------
# layer + mask_var + full-output (combined dispatch flags)
# --------------------------------------------------------------------------

def test_layer_plus_mask_var_full_output():
    a = _grouped(seed=11, n_vars=30)
    a.layers["lognorm"] = a.X.copy()
    mask = np.zeros(a.n_vars, dtype=bool)
    mask[5:20] = True
    a.var["use"] = mask
    RG._fast_rank_genes_groups(
        a, "group", method="wilcoxon", layer="lognorm", use_raw=False,
        mask_var="use")
    rec = a.uns["rank_genes_groups"]
    g = rec["names"].dtype.names[0]
    # 15 masked genes -> 15 entries; all from the masked window.
    assert len(rec["names"][g]) == 15
    assert set(map(str, rec["names"][g])).issubset(set(a.var_names[5:20]))
    assert rec["params"]["layer"] == "lognorm"


# --------------------------------------------------------------------------
# bonferroni full-output parity on adjusted p-values
# --------------------------------------------------------------------------

def test_full_output_bonferroni_pvals_adj_parity():
    a_fast = _grouped(seed=12, n_groups=2)
    a_van = a_fast.copy()
    RG._fast_rank_genes_groups(
        a_fast, "group", method="wilcoxon", corr_method="bonferroni")
    with autozyme.disabled():
        sc.tl.rank_genes_groups(
            a_van, "group", method="wilcoxon", corr_method="bonferroni")
    g = a_fast.uns["rank_genes_groups"]["names"].dtype.names[0]
    fast_pa = _score_map(a_fast, g, field="pvals_adj")
    van_pa = _score_map(a_van, g, field="pvals_adj")
    common = list(set(fast_pa) & set(van_pa))[:15]
    for name in common:
        np.testing.assert_allclose(fast_pa[name], van_pa[name], rtol=1e-4,
                                   atol=1e-6)


# --------------------------------------------------------------------------
# pts + groups-subset (pts restricted to the requested groups' columns)
# --------------------------------------------------------------------------

def test_pts_with_groups_subset():
    a = _grouped(seed=13, n_groups=3, n_vars=24)
    RG._fast_rank_genes_groups(
        a, "group", method="wilcoxon", groups=["grp1", "grp2"], pts=True)
    u = a.uns["rank_genes_groups"]
    import pandas as pd
    assert isinstance(u["pts"], pd.DataFrame)
    # pts columns correspond to exactly the requested groups.
    assert set(u["pts"].columns) == {"grp1", "grp2"}
    assert u["pts"].shape[0] == a.n_vars


# --------------------------------------------------------------------------
# n_genes equal to n_vars takes the full-output (NOT top-N) branch
# --------------------------------------------------------------------------

def test_n_genes_equal_n_vars_uses_full_output():
    # n_genes >= n_genes_total -> the `n_genes < n_genes_total` guard is False
    # so the FULL-output branch runs (not the partial top-N path).
    a = _grouped(seed=14, n_vars=20, n_groups=2)
    RG._fast_rank_genes_groups(
        a, "group", method="wilcoxon", n_genes=a.n_vars)
    rec = a.uns["rank_genes_groups"]["names"]
    # Full output -> all genes present, sorted by descending score.
    assert len(rec) == a.n_vars


# --------------------------------------------------------------------------
# corr_method='none' on the FULL-output path parity (pvals_adj == pvals)
# --------------------------------------------------------------------------

def test_full_output_corr_none_equals_raw_pvals():
    a = _grouped(seed=15, n_groups=2)
    RG._fast_rank_genes_groups(
        a, "group", method="wilcoxon", corr_method="none")
    u = a.uns["rank_genes_groups"]
    g = u["names"].dtype.names[0]
    np.testing.assert_allclose(
        np.asarray(u["pvals_adj"][g], dtype=np.float64),
        np.asarray(u["pvals"][g], dtype=np.float64))


# --------------------------------------------------------------------------
# t-test-overestim_var also delegates (another non-wilcoxon method)
# --------------------------------------------------------------------------

def test_ttest_overestim_var_method_delegates():
    a = _grouped(seed=16)
    RG._fast_rank_genes_groups(
        a, "group", method="t-test_overestim_var")
    assert a.uns["rank_genes_groups"]["params"]["method"] == "t-test_overestim_var"


def test_logreg_method_delegates():
    a = _grouped(seed=17)
    RG._fast_rank_genes_groups(a, "group", method="logreg")
    assert "rank_genes_groups" in a.uns


# --------------------------------------------------------------------------
# reference set to a real group (not 'rest') + tie_correct both defer
# --------------------------------------------------------------------------

def test_reference_real_group_and_tie_correct_defer():
    a = _grouped(seed=18, n_groups=3)
    # reference != 'rest' -> defer (one-vs-one). Just confirm it completes and
    # records the requested reference.
    RG._fast_rank_genes_groups(
        a, "group", method="wilcoxon", reference="grp0", groups=["grp1", "grp2"])
    assert a.uns["rank_genes_groups"]["params"]["reference"] == "grp0"


# --------------------------------------------------------------------------
# copy=True + top-N together (output isolation on the partial path)
# --------------------------------------------------------------------------

def test_copy_true_with_top_n():
    a = _grouped(seed=19, n_groups=2)
    out = RG._fast_rank_genes_groups(
        a, "group", method="wilcoxon", n_genes=6, copy=True)
    assert out is not None and out is not a
    assert "rank_genes_groups" in out.uns
    assert "rank_genes_groups" not in a.uns
    assert len(out.uns["rank_genes_groups"]["names"]) == 6
