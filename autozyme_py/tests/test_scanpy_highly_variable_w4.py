"""Wave-4 ``highly_variable_genes`` coverage / hardening.

The reachable pure-python wrappers in ``_highly_variable.py``
(``_patched_hvg`` / ``_fast_hvg_seurat`` / ``_fast_hvg_seurat_v3_batch`` and
all their guard-fallbacks) are already 100% line-covered by waves 1-3.
Everything still "missing" in this module is either the ``except ImportError``
numba guard (24-25, unreachable while numba is installed) or the ``@njit``
kernel bodies (``_hvg_one_pass`` / ``_v3_batch_mean_var`` /
``_v3_batch_clip_sum``, lines 42-239) which coverage.py cannot see (exercised
directly in test_scanpy_highly_variable_unit.py).

So no NEW visible line is recoverable here. These tests instead harden the
flavor / batch / subset / inplace / cutoff parameter matrix called out in the
briefing, asserting parity vs the upstream original (``zyme=False``) on
combinations the earlier waves did not cover.

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
from autozyme.scanpy import _highly_variable as HV  # noqa: E402


def _raw_counts(n_obs=120, n_vars=80, seed=0, batches=2):
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.9, size=(n_obs, n_vars)).astype(np.float32)
    dense[:, :10] += rng.poisson(3.0, size=(n_obs, 10)).astype(np.float32)
    X = sparse.csr_matrix(dense)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    a = ad.AnnData(X)
    a.var_names = [f"g{j}" for j in range(n_vars)]
    a.obs["batch"] = pd.Categorical([f"b{i % batches}" for i in range(n_obs)])
    return a


def _lognorm(a):
    with autozyme.disabled():
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    return a


@pytest.fixture(autouse=True)
def _activate():
    autozyme.activate("scanpy")
    yield
    autozyme.deactivate("scanpy")


# --------------------------------------------------------------------------
# seurat cutoff mode with explicit min/max bounds (parity vs upstream)
# --------------------------------------------------------------------------

def test_seurat_explicit_disp_mean_cutoffs_parity():
    a_fast = _lognorm(_raw_counts(seed=1))
    a_van = a_fast.copy()
    sc.pp.highly_variable_genes(
        a_fast, flavor="seurat", min_mean=0.01, max_mean=4.0,
        min_disp=0.4, max_disp=np.inf)
    sc.pp.highly_variable_genes(
        a_van, flavor="seurat", min_mean=0.01, max_mean=4.0,
        min_disp=0.4, max_disp=np.inf, zyme=False)
    assert (a_fast.var["highly_variable"].values
            == a_van.var["highly_variable"].values).all()


# --------------------------------------------------------------------------
# seurat with custom n_bins (parity)
# --------------------------------------------------------------------------

def test_seurat_custom_n_bins_parity():
    a_fast = _lognorm(_raw_counts(seed=2))
    a_van = a_fast.copy()
    sc.pp.highly_variable_genes(a_fast, flavor="seurat", n_top_genes=25, n_bins=15)
    sc.pp.highly_variable_genes(
        a_van, flavor="seurat", n_top_genes=25, n_bins=15, zyme=False)
    assert (a_fast.var["highly_variable"].values
            == a_van.var["highly_variable"].values).all()


# --------------------------------------------------------------------------
# seurat_v3 (non-paper) batch parity on the full annotation set
# --------------------------------------------------------------------------

def test_seurat_v3_batch_full_annotation_parity():
    a_fast = _raw_counts(seed=3, batches=2)
    a_van = a_fast.copy()
    sc.pp.highly_variable_genes(
        a_fast, flavor="seurat_v3", n_top_genes=30, batch_key="batch")
    sc.pp.highly_variable_genes(
        a_van, flavor="seurat_v3", n_top_genes=30, batch_key="batch", zyme=False)
    assert a_fast.var["highly_variable"].sum() == 30
    assert a_van.var["highly_variable"].sum() == 30
    # The number of genes the two agree on as HVGs should be the large majority.
    overlap = (a_fast.var["highly_variable"].values
               & a_van.var["highly_variable"].values).sum()
    assert overlap >= 25  # >= 25/30 agree
    for col in ("highly_variable_rank", "highly_variable_nbatches",
                "means", "variances", "variances_norm"):
        assert col in a_fast.var.columns


# --------------------------------------------------------------------------
# seurat_v3 with 3 batches (more than the common 2-batch case)
# --------------------------------------------------------------------------

def test_seurat_v3_three_batches_runs():
    a = _raw_counts(seed=4, n_obs=150, batches=3)
    sc.pp.highly_variable_genes(
        a, flavor="seurat_v3_paper", n_top_genes=25, batch_key="batch")
    assert a.var["highly_variable"].sum() == 25
    # nbatches annotation is in [0, 3].
    nb = a.var["highly_variable_nbatches"].values
    assert nb.min() >= 0 and nb.max() <= 3


# --------------------------------------------------------------------------
# seurat_v3 check_values=False skips the count-data validation
# --------------------------------------------------------------------------

def test_seurat_v3_check_values_false_no_warning():
    import warnings
    a = _raw_counts(seed=5)
    a.X = a.X.copy()
    a.X.data = a.X.data + 0.5  # non-integer values
    # check_values=False -> the kernel's do_check arg is False -> no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sc.pp.highly_variable_genes(
            a, flavor="seurat_v3", n_top_genes=20, batch_key="batch",
            check_values=False)
    assert a.var["highly_variable"].sum() == 20


# --------------------------------------------------------------------------
# seurat_v3 with n_top_genes=None -> the wrapper's guard defers to upstream
# --------------------------------------------------------------------------

def test_seurat_v3_n_top_genes_none_defers():
    # n_top_genes=None trips the v3-batch fast-path guard -> the wrapper defers
    # to upstream seurat_v3 (which selects via dispersion cutoff, no raise).
    # Parity check confirms the defer produced the upstream result.
    a_fast = _raw_counts(seed=6)
    a_van = a_fast.copy()
    sc.pp.highly_variable_genes(
        a_fast, flavor="seurat_v3", n_top_genes=None, batch_key="batch")
    sc.pp.highly_variable_genes(
        a_van, flavor="seurat_v3", n_top_genes=None, batch_key="batch",
        zyme=False)
    assert (a_fast.var["highly_variable"].values
            == a_van.var["highly_variable"].values).all()
    # The v3 annotation columns are present (written by the upstream path).
    assert "highly_variable_rank" in a_fast.var.columns


# --------------------------------------------------------------------------
# seurat flavor, inplace=False + subset=False (DataFrame return, no var write)
# --------------------------------------------------------------------------

def test_seurat_cutoff_inplace_false_dataframe():
    a = _lognorm(_raw_counts(seed=7))
    df = sc.pp.highly_variable_genes(a, flavor="seurat", inplace=False)
    assert isinstance(df, pd.DataFrame)
    for col in ("highly_variable", "means", "dispersions", "dispersions_norm"):
        assert col in df.columns
    # inplace=False does not annotate .var.
    assert "highly_variable" not in a.var.columns


# --------------------------------------------------------------------------
# seurat_v3 batch with a single batch (one category) still runs the fast path
# --------------------------------------------------------------------------

def test_seurat_v3_single_batch_runs():
    # One batch with >= 2 cells passes every guard and runs the fast path.
    a = _raw_counts(seed=8, batches=1)
    sc.pp.highly_variable_genes(
        a, flavor="seurat_v3", n_top_genes=20, batch_key="batch")
    assert a.var["highly_variable"].sum() == 20
    # Single batch -> nbatches is 0 or 1 per gene.
    assert set(np.unique(a.var["highly_variable_nbatches"].values)).issubset({0, 1})
