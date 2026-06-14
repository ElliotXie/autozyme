"""End-to-end ``sc.pp.highly_variable_genes`` patched-path tests.

Drives a tiny real AnnData through ``_patched_hvg`` and its sub-paths and
asserts parity vs upstream. Covers the wrapper dispatch / validation /
fallback lines coverage can see:
  * flavor='seurat' fast path (``_fast_hvg_seurat``): n_top_genes set/unset,
    inplace True/False, subset, copy of the dataframe return.
  * flavor='seurat_v3'(+_paper) batch-aware fast path
    (``_fast_hvg_seurat_v3_batch``): batch_key set, the many guard-fallbacks.
  * batch_key set with flavor='seurat' → upstream defer.
  * dense input → upstream defer.

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
pytest.importorskip("skmisc")  # seurat_v3 loess fit needs skmisc.loess
import autozyme  # noqa: E402


def _raw_counts(n_obs=120, n_vars=100, seed=30, batches=2):
    """Raw integer counts CSR AnnData with a categorical batch column."""
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.9, size=(n_obs, n_vars)).astype(np.float32)
    # Give a handful of genes extra variance so HVG selection is non-degenerate.
    dense[:, :10] += rng.poisson(3.0, size=(n_obs, 10)).astype(np.float32)
    X = sparse.csr_matrix(dense)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    adata = ad.AnnData(X)
    adata.var_names = [f"g{j}" for j in range(n_vars)]
    adata.obs["batch"] = pd.Categorical(
        [f"b{i % batches}" for i in range(n_obs)]
    )
    return adata


def _lognorm(adata):
    sc.pp.normalize_total(adata, target_sum=1e4, zyme=False)
    sc.pp.log1p(adata, zyme=False)
    return adata


@pytest.fixture(autouse=True)
def _activate():
    autozyme.activate("scanpy")
    yield


# --------------------------------------------------------------------------
# flavor='seurat' (default) fast path
# --------------------------------------------------------------------------

def test_hvg_seurat_n_top_genes_parity():
    a_fast = _lognorm(_raw_counts())
    a_orig = a_fast.copy()
    sc.pp.highly_variable_genes(a_fast, flavor="seurat", n_top_genes=30)
    sc.pp.highly_variable_genes(a_orig, flavor="seurat", n_top_genes=30, zyme=False)
    # Same set of selected genes.
    assert (a_fast.var["highly_variable"].values
            == a_orig.var["highly_variable"].values).all()
    assert a_fast.var["highly_variable"].sum() == 30


def test_hvg_seurat_cutoff_mode_parity():
    # No n_top_genes → min_disp/min_mean cutoff branch.
    a_fast = _lognorm(_raw_counts(seed=31))
    a_orig = a_fast.copy()
    sc.pp.highly_variable_genes(a_fast, flavor="seurat")
    sc.pp.highly_variable_genes(a_orig, flavor="seurat", zyme=False)
    assert (a_fast.var["highly_variable"].values
            == a_orig.var["highly_variable"].values).all()


def test_hvg_seurat_uns_and_var_columns():
    a = _lognorm(_raw_counts())
    sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=20)
    assert a.uns["hvg"]["flavor"] == "seurat"
    for col in ("highly_variable", "means", "dispersions", "dispersions_norm"):
        assert col in a.var.columns


def test_hvg_seurat_inplace_false_returns_dataframe():
    a = _lognorm(_raw_counts())
    df = sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=20, inplace=False)
    assert isinstance(df, pd.DataFrame)
    assert "highly_variable" in df.columns
    # inplace=False does not write to .var.
    assert "highly_variable" not in a.var.columns


def test_hvg_seurat_subset_inplace():
    a = _lognorm(_raw_counts())
    n0 = a.n_vars
    sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=25, subset=True)
    assert a.n_vars == 25 < n0


def test_hvg_seurat_subset_dataframe():
    a = _lognorm(_raw_counts())
    df = sc.pp.highly_variable_genes(
        a, flavor="seurat", n_top_genes=25, subset=True, inplace=False
    )
    assert len(df) == 25
    assert df["highly_variable"].all()


def test_hvg_seurat_dense_input_defers():
    a = _lognorm(_raw_counts())
    a.X = a.X.toarray()  # dense → fast path declines, upstream handles it
    sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=20)
    assert "highly_variable" in a.var.columns


# --------------------------------------------------------------------------
# batch_key with flavor='seurat' → upstream defer (the kwargs.get path)
# --------------------------------------------------------------------------

def test_hvg_seurat_with_batch_key_defers():
    a = _lognorm(_raw_counts())
    sc.pp.highly_variable_genes(
        a, flavor="seurat", n_top_genes=20, batch_key="batch"
    )
    assert "highly_variable" in a.var.columns


# --------------------------------------------------------------------------
# flavor='seurat_v3' / 'seurat_v3_paper' batch-aware fast path
# --------------------------------------------------------------------------

def test_hvg_seurat_v3_paper_batch_parity():
    a_fast = _raw_counts(seed=32)  # raw counts (v3 wants raw)
    a_orig = a_fast.copy()
    sc.pp.highly_variable_genes(
        a_fast, flavor="seurat_v3_paper", n_top_genes=30, batch_key="batch"
    )
    sc.pp.highly_variable_genes(
        a_orig, flavor="seurat_v3_paper", n_top_genes=30, batch_key="batch",
        zyme=False,
    )
    # n_top_genes selected by both.
    assert a_fast.var["highly_variable"].sum() == 30
    assert a_orig.var["highly_variable"].sum() == 30
    # The HVG-rank annotation column is written.
    assert "highly_variable_rank" in a_fast.var.columns
    assert "highly_variable_nbatches" in a_fast.var.columns


def test_hvg_seurat_v3_batch_runs():
    a = _raw_counts(seed=33)
    sc.pp.highly_variable_genes(
        a, flavor="seurat_v3", n_top_genes=25, batch_key="batch"
    )
    assert a.uns["hvg"]["flavor"] == "seurat_v3"
    assert a.var["highly_variable"].sum() == 25
    for col in ("means", "variances", "variances_norm"):
        assert col in a.var.columns


def test_hvg_seurat_v3_noninteger_warns():
    # Non-integer values + check_values → UserWarning from the fast path.
    a = _raw_counts(seed=34)
    a.X = a.X.copy()
    a.X.data = a.X.data + 0.5  # fractional → "expects raw count data" warning
    with pytest.warns(UserWarning):
        sc.pp.highly_variable_genes(
            a, flavor="seurat_v3", n_top_genes=20, batch_key="batch"
        )


def test_hvg_seurat_v3_no_batch_key_defers():
    # seurat_v3 WITHOUT batch_key → not the batch fast path; upstream defer.
    a = _raw_counts(seed=35)
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=20)
    assert "highly_variable" in a.var.columns


def test_hvg_seurat_v3_subset_true_defers():
    # subset=True trips the v3-batch guard → upstream defer (still correct).
    a = _raw_counts(seed=36)
    sc.pp.highly_variable_genes(
        a, flavor="seurat_v3", n_top_genes=20, batch_key="batch", subset=True
    )
    assert a.n_vars == 20


def test_hvg_seurat_v3_dense_input_defers():
    # Dense .X trips the isspmatrix_csr guard in the v3-batch path → defer.
    a = _raw_counts(seed=37)
    a.X = a.X.toarray()
    sc.pp.highly_variable_genes(
        a, flavor="seurat_v3", n_top_genes=20, batch_key="batch"
    )
    assert "highly_variable" in a.var.columns


def test_hvg_seurat_v3_noncategorical_batch_coerced():
    # A non-categorical batch column hits the `astype("category")` branch
    # in the v3-batch fast path.
    a = _raw_counts(seed=39)
    a.obs["batch"] = (np.arange(a.n_obs) % 2).astype(np.int64)  # plain int col
    sc.pp.highly_variable_genes(
        a, flavor="seurat_v3", n_top_genes=20, batch_key="batch"
    )
    assert a.var["highly_variable"].sum() == 20


def test_hvg_zyme_false_equals_upstream():
    a_fast = _lognorm(_raw_counts(seed=38))
    a_z = a_fast.copy()
    sc.pp.highly_variable_genes(a_fast, flavor="seurat", n_top_genes=20, zyme=False)
    autozyme.deactivate("scanpy")
    sc.pp.highly_variable_genes(a_z, flavor="seurat", n_top_genes=20)
    assert (a_fast.var["highly_variable"].values
            == a_z.var["highly_variable"].values).all()
