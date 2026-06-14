"""End-to-end ``sc.tl.rank_genes_groups`` patched-path tests.

Drives a tiny real AnnData through ``_fast_rank_genes_groups`` (wilcoxon +
rest + sparse + no tie_correct = the fused fast path) and asserts parity vs
upstream, covering the wrapper dispatch / param-handling / fallback lines:
  * full output (n_genes=None) vs top-N (n_genes set), the BH / bonferroni /
    none corr_method branches, groups subset, pts, copy, key_added, layer,
    and the fallbacks (tie_correct, reference != 'rest', dense input).

Env recipe: KMP_DUPLICATE_LIB_OK=TRUE NUMBA_THREADING_LAYER=workqueue
NUMBA_NUM_THREADS=14 OMP_NUM_THREADS=2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sc = pytest.importorskip("scanpy")
ad = pytest.importorskip("anndata")
import autozyme  # noqa: E402


def _grouped(n_per=40, n_vars=60, n_groups=3, seed=50):
    """Log-normalized sparse AnnData with a categorical 'group' column.

    Each group gets a distinct set of up-regulated marker genes so the
    wilcoxon ranking is non-degenerate.
    """
    rng = np.random.default_rng(seed)
    n_obs = n_per * n_groups
    base = rng.poisson(0.6, size=(n_obs, n_vars)).astype(np.float32)
    for g in range(n_groups):
        rows = slice(g * n_per, (g + 1) * n_per)
        cols = slice(g * 5, g * 5 + 5)
        base[rows, cols] += rng.poisson(4.0, size=base[rows, cols].shape)
    X = sparse.csr_matrix(base)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    adata = ad.AnnData(X)
    adata.var_names = [f"g{j}" for j in range(n_vars)]
    adata.obs["group"] = pd.Categorical(
        [f"grp{i // n_per}" for i in range(n_obs)]
    )
    sc.pp.normalize_total(adata, target_sum=1e4, zyme=False)
    sc.pp.log1p(adata, zyme=False)
    return adata


def _top_names(adata, key="rank_genes_groups", group=None):
    rec = adata.uns[key]["names"]
    if group is None:
        group = rec.dtype.names[0]
    return list(rec[group])


@pytest.fixture(autouse=True)
def _activate():
    autozyme.activate("scanpy")
    yield


# --------------------------------------------------------------------------
# full-output fast path
# --------------------------------------------------------------------------

def test_rank_genes_full_output_top_marker_parity():
    a_fast = _grouped()
    a_orig = a_fast.copy()
    sc.tl.rank_genes_groups(a_fast, "group", method="wilcoxon")
    sc.tl.rank_genes_groups(a_orig, "group", method="wilcoxon", zyme=False)
    # Top-ranked gene per group should agree (markers are well separated).
    for g in a_fast.uns["rank_genes_groups"]["names"].dtype.names:
        top_fast = _top_names(a_fast, group=g)[0]
        top_orig = _top_names(a_orig, group=g)[0]
        assert top_fast == top_orig


def test_rank_genes_uns_keys_present():
    a = _grouped()
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon")
    u = a.uns["rank_genes_groups"]
    for k in ("names", "scores", "pvals", "pvals_adj", "logfoldchanges", "params"):
        assert k in u


def test_rank_genes_scores_parity():
    a_fast = _grouped(seed=51)
    a_orig = a_fast.copy()
    sc.tl.rank_genes_groups(a_fast, "group", method="wilcoxon")
    sc.tl.rank_genes_groups(a_orig, "group", method="wilcoxon", zyme=False)
    g = a_fast.uns["rank_genes_groups"]["names"].dtype.names[0]
    # The per-gene score arrays (sorted by gene name) should match closely.
    def by_name(adata):
        rec = adata.uns["rank_genes_groups"]
        return dict(zip(rec["names"][g], rec["scores"][g]))
    sf, so = by_name(a_fast), by_name(a_orig)
    common = list(sf.keys())[:10]
    for name in common:
        assert np.isclose(sf[name], so[name], rtol=1e-2, atol=1e-2)


# --------------------------------------------------------------------------
# top-N path (n_genes set)
# --------------------------------------------------------------------------

def test_rank_genes_top_n_bh():
    a = _grouped()
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", n_genes=10)
    rec = a.uns["rank_genes_groups"]["names"]
    assert len(rec) == 10


def test_rank_genes_top_n_bonferroni():
    a = _grouped()
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", n_genes=10,
                            corr_method="bonferroni")
    u = a.uns["rank_genes_groups"]
    g = u["names"].dtype.names[0]
    # bonferroni-adjusted pvals are >= raw pvals.
    assert np.all(np.asarray(u["pvals_adj"][g]) >= np.asarray(u["pvals"][g]) - 1e-9)


def test_rank_genes_full_bonferroni():
    a = _grouped()
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon",
                            corr_method="bonferroni")
    assert "pvals_adj" in a.uns["rank_genes_groups"]


def test_rank_genes_full_corr_none():
    # corr_method='none' → full-output branch where pvals_adj == pvals.
    a = _grouped()
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", corr_method="none")
    u = a.uns["rank_genes_groups"]
    g = u["names"].dtype.names[0]
    np.testing.assert_allclose(np.asarray(u["pvals_adj"][g]),
                               np.asarray(u["pvals"][g]))


def test_rank_genes_top_n_corr_none():
    # corr_method='none' top-N branch (out_pa = out_p.copy()).
    a = _grouped()
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", n_genes=10,
                            corr_method="none")
    assert len(a.uns["rank_genes_groups"]["names"]) == 10


# --------------------------------------------------------------------------
# groups subset
# --------------------------------------------------------------------------

def test_rank_genes_groups_subset():
    a = _grouped()
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon",
                            groups=["grp0", "grp1"])
    names = a.uns["rank_genes_groups"]["names"].dtype.names
    assert set(names) == {"grp0", "grp1"}


def test_rank_genes_groups_scalar_raises():
    a = _grouped()
    # A scalar groups arg must raise (the wrapper validates this).
    with pytest.raises(ValueError):
        sc.tl.rank_genes_groups(a, "group", method="wilcoxon", groups="grp0")


# --------------------------------------------------------------------------
# pts
# --------------------------------------------------------------------------

def test_rank_genes_pts():
    a = _grouped()
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", pts=True)
    u = a.uns["rank_genes_groups"]
    assert "pts" in u and "pts_rest" in u
    assert isinstance(u["pts"], pd.DataFrame)


# --------------------------------------------------------------------------
# copy / key_added / use_raw
# --------------------------------------------------------------------------

def test_rank_genes_copy_true_returns_new():
    a = _grouped()
    out = sc.tl.rank_genes_groups(a, "group", method="wilcoxon", copy=True)
    assert out is not a
    assert "rank_genes_groups" in out.uns
    assert "rank_genes_groups" not in a.uns


def test_rank_genes_key_added():
    a = _grouped()
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", key_added="markers")
    assert "markers" in a.uns


def test_rank_genes_use_raw_path():
    a = _grouped()
    a.raw = a  # snapshot log-normalized sparse matrix
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", use_raw=True)
    assert "rank_genes_groups" in a.uns


def test_rank_genes_key_added_none_defaults():
    # key_added=None → wrapper normalizes to 'rank_genes_groups'.
    a = _grouped()
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", key_added=None)
    assert "rank_genes_groups" in a.uns


def test_rank_genes_layer_path():
    # layer set → X is pulled from the layer instead of .X.
    a = _grouped()
    a.layers["lognorm"] = a.X.copy()
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", layer="lognorm",
                            use_raw=False)
    assert a.uns["rank_genes_groups"]["params"]["layer"] == "lognorm"


def test_rank_genes_only_positive_kwd():
    # only_positive=True is consumed by the wrapper (flips rankby_abs).
    a = _grouped()
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", n_genes=10,
                            only_positive=True)
    assert len(a.uns["rank_genes_groups"]["names"]) == 10


def test_rank_genes_mask_var_path():
    # mask_var restricts the gene set the fast path ranks over.
    a = _grouped()
    mask = np.zeros(a.n_vars, dtype=bool)
    mask[:30] = True
    a.var["use"] = mask
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", mask_var="use")
    # Only the 30 masked genes appear in the ranking output.
    g = a.uns["rank_genes_groups"]["names"].dtype.names[0]
    assert len(a.uns["rank_genes_groups"]["names"][g]) == 30


# --------------------------------------------------------------------------
# fallback branches (delegate to upstream)
# --------------------------------------------------------------------------

def test_rank_genes_tie_correct_defers():
    a = _grouped()
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", tie_correct=True)
    assert "rank_genes_groups" in a.uns


def test_rank_genes_reference_group_defers():
    a = _grouped()
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", reference="grp0")
    assert "rank_genes_groups" in a.uns


def test_rank_genes_ttest_method_defers():
    a = _grouped()
    sc.tl.rank_genes_groups(a, "group", method="t-test")
    assert a.uns["rank_genes_groups"]["params"]["method"] == "t-test"


def test_rank_genes_dense_input_defers():
    a = _grouped()
    a.X = a.X.toarray()  # dense → fast path declines, upstream handles it
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon")
    assert "rank_genes_groups" in a.uns


def test_rank_genes_zyme_false_equals_upstream():
    a_fast = _grouped(seed=52)
    a_z = a_fast.copy()
    sc.tl.rank_genes_groups(a_fast, "group", method="wilcoxon", zyme=False)
    autozyme.deactivate("scanpy")
    sc.tl.rank_genes_groups(a_z, "group", method="wilcoxon")
    g = a_fast.uns["rank_genes_groups"]["names"].dtype.names[0]
    assert _top_names(a_fast, group=g)[0] == _top_names(a_z, group=g)[0]
