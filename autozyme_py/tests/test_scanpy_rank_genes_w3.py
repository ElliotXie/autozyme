"""Wave-3 ``rank_genes_groups`` coverage — the reachable pure-python lines
wave-1/wave-2 missed in ``_rank_genes.py``.

The dominant genuinely-uncovered pure-python gap is the
``log_base_factor != 1.0`` branch in the *full-output* logfc path
(lines 581-583): it only runs when ``adata.uns['log1p']['base']`` is a real
base (e.g. ``log1p(base=2)``). Wave-2 only logged with the default base=None,
so that branch never fired. We also exercise the same non-default base on the
top-N path (``_compute_topn_logfc`` with a non-unit factor) and a couple of
param combinations (groups subset by integer label, mask_var + top-N).

Everything else "missing" in this module is numba @njit kernel bodies
(_fused_*_csc / _dual_sort / _bh_* / _sparse_rankdata_csc / etc.) which
coverage.py cannot see; those are exercised in
test_scanpy_rank_genes_unit.py.

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


def _grouped(n_per=40, n_vars=30, n_groups=2, seed=0, base=None):
    """Log-normalized sparse AnnData; ``base`` controls the log1p base.

    A non-None ``base`` makes the fast path's recorded ``log_base_factor``
    differ from 1.0, which routes the logfc computation through the
    ``base``-correcting branch.
    """
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


# --------------------------------------------------------------------------
# log_base_factor != 1.0 — full-output logfc branch (lines 581-583)
# --------------------------------------------------------------------------

def test_full_output_log_base2_logfc_branch():
    # base=2 → log_base_factor = ln(2) != 1.0 → the base-correcting full-output
    # logfc branch fires. Run direct (full output, n_genes=None).
    a = _grouped(seed=1, base=2)
    assert a.uns["log1p"]["base"] == 2
    RG._fast_rank_genes_groups(a, "group", method="wilcoxon")
    rec = a.uns["rank_genes_groups"]
    g = rec["names"].dtype.names[0]
    logfc = np.asarray(rec["logfoldchanges"][g], dtype=np.float64)
    assert logfc.shape[0] == a.n_vars
    assert np.all(np.isfinite(logfc))


def test_full_output_log_base2_logfc_matches_vanilla():
    # Parity check: the base-corrected logfc should agree with vanilla scanpy
    # (which also reads uns['log1p']['base']).
    a_fast = _grouped(seed=2, base=2)
    a_van = a_fast.copy()
    RG._fast_rank_genes_groups(a_fast, "group", method="wilcoxon")
    with autozyme.disabled():
        sc.tl.rank_genes_groups(a_van, "group", method="wilcoxon")
    g = a_fast.uns["rank_genes_groups"]["names"].dtype.names[0]

    def by_name(adata, field):
        rec = adata.uns["rank_genes_groups"]
        return dict(zip(rec["names"][g], rec[field][g]))

    fast_lfc = by_name(a_fast, "logfoldchanges")
    van_lfc = by_name(a_van, "logfoldchanges")
    common = list(set(fast_lfc) & set(van_lfc))[:10]
    for name in common:
        np.testing.assert_allclose(
            float(fast_lfc[name]), float(van_lfc[name]), rtol=1e-2, atol=1e-2)


def test_top_n_log_base2_logfc_branch():
    # base != None also flows into _compute_topn_logfc with a non-unit factor
    # on the top-N path (n_genes set). Asserts finite, correct count.
    a = _grouped(seed=3, base=10)
    RG._fast_rank_genes_groups(a, "group", method="wilcoxon", n_genes=8)
    rec = a.uns["rank_genes_groups"]
    assert len(rec["names"]) == 8
    g = rec["names"].dtype.names[0]
    assert np.all(np.isfinite(np.asarray(rec["logfoldchanges"][g], dtype=np.float64)))


# --------------------------------------------------------------------------
# groups subset specified by integer labels (lines 491-502 int-coercion)
# --------------------------------------------------------------------------

def test_groups_subset_with_integer_args_string_categories():
    # The categories are strings ("0", "1", "2") — scanpy's normal convention —
    # but the user passes integer group ids. The wrapper's
    # `str(g) if isinstance(g, int)` coercion (line 493) maps int 0 -> "0"
    # before the np.where category lookup.
    import pandas as pd
    rng = np.random.default_rng(4)
    counts = rng.poisson(0.6, size=(60, 20)).astype(np.float32)
    counts[:20, :3] += 5
    X = sparse.csr_matrix(counts)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    a = ad.AnnData(X)
    a.var_names = [f"g{j}" for j in range(20)]
    a.obs["grp"] = pd.Categorical(["0", "1", "2"] * 20)  # string categories
    with autozyme.disabled():
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    # Pass integer group ids → the `str(g) if isinstance(g, int)` path.
    RG._fast_rank_genes_groups(a, "grp", method="wilcoxon", groups=[0, 1])
    names = a.uns["rank_genes_groups"]["names"].dtype.names
    assert set(names) == {"0", "1"}


# --------------------------------------------------------------------------
# mask_var restricting the gene set on the top-N path
# --------------------------------------------------------------------------

def test_mask_var_with_top_n():
    a = _grouped(seed=5, n_vars=30)
    mask = np.zeros(a.n_vars, dtype=bool)
    mask[:15] = True
    a.var["use"] = mask
    RG._fast_rank_genes_groups(
        a, "group", method="wilcoxon", n_genes=5, mask_var="use")
    rec = a.uns["rank_genes_groups"]["names"]
    assert len(rec) == 5
    # All returned gene names come from the masked subset.
    g = rec.dtype.names[0]
    masked_names = set(a.var_names[:15])
    assert set(map(str, rec[g])).issubset(masked_names)


# --------------------------------------------------------------------------
# pts on the top-N path (combined with n_genes)
# --------------------------------------------------------------------------

def test_pts_with_top_n_full_matrix():
    # pts uses the FULL n_genes_total matrix regardless of n_genes; confirm the
    # pts / pts_rest DataFrames cover all genes even when n_genes is set.
    a = _grouped(seed=6, n_vars=24)
    RG._fast_rank_genes_groups(
        a, "group", method="wilcoxon", n_genes=6, pts=True)
    u = a.uns["rank_genes_groups"]
    import pandas as pd
    assert isinstance(u["pts"], pd.DataFrame)
    assert u["pts"].shape[0] == a.n_vars  # one row per gene
    assert "pts_rest" in u
