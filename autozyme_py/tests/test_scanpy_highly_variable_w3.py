"""Wave-3 ``highly_variable_genes`` coverage — the reachable pure-python lines
wave-1/wave-2 missed in ``_highly_variable.py``.

Targets (line numbers as of this revision):
  * ``_fast_hvg_seurat``: CSC->CSR conversion (266), float64->float32 cast
    (268), and the ``n_top_genes >= n_genes`` cutoff=-inf branch (291).
  * ``_fast_hvg_seurat_v3_batch`` guard-fallbacks that need a *direct* call
    with a degenerate batch layout to reach: tiny matrix (<2 rows / 0 cols)
    (390), negative batch codes (406), a singleton batch (batch_sizes<2)
    (421), and the per-batch "<2 non-constant genes" loess guard (460).
  * the skmisc-missing ImportError fallback (363-364) via monkeypatch.

Everything else "missing" in this module is numba @njit kernel bodies
(_hvg_one_pass / _v3_batch_mean_var / _v3_batch_clip_sum) which coverage.py
cannot see; those are exercised in test_scanpy_highly_variable_unit.py.

Env recipe: KMP_DUPLICATE_LIB_OK=TRUE NUMBA_THREADING_LAYER=workqueue
NUMBA_NUM_THREADS=14 OMP_NUM_THREADS=2.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from autozyme.scanpy import _highly_variable as HV

pytestmark = pytest.mark.skipif(not HV.HAS_NUMBA, reason="numba not installed")


def _lognorm_adata(n_cells=60, n_genes=24, seed=0, fmt="csr", dtype=np.float32):
    """Tiny log-normalized AnnData in the requested sparse format / dtype."""
    ad = pytest.importorskip("anndata")
    sc = pytest.importorskip("scanpy")
    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.6, size=(n_cells, n_genes)).astype(np.float32)
    X = sparse.csr_matrix(counts)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    a = ad.AnnData(X)
    import autozyme
    with autozyme.disabled():
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    if fmt == "csc":
        a.X = a.X.tocsc()
    if dtype != np.float32:
        a.X = a.X.astype(dtype)
    return a


# --------------------------------------------------------------------------
# _fast_hvg_seurat — input-coercion + cutoff branches
# --------------------------------------------------------------------------

def test_seurat_csc_input_converted_to_csr():
    # CSC .X trips the `not isspmatrix_csr -> tocsr()` branch (line 266).
    pytest.importorskip("scanpy")
    a = _lognorm_adata(seed=10, fmt="csc")
    assert sparse.isspmatrix_csc(a.X)
    HV._fast_hvg_seurat(a, n_top_genes=8)
    assert "highly_variable" in a.var.columns
    assert int(a.var["highly_variable"].sum()) >= 8


def test_seurat_float64_input_cast_to_float32():
    # float64 CSR trips the `x.dtype != float32 -> astype` branch (line 268).
    pytest.importorskip("scanpy")
    a = _lognorm_adata(seed=11, dtype=np.float64)
    assert a.X.dtype == np.float64
    HV._fast_hvg_seurat(a, n_top_genes=8)
    assert "highly_variable" in a.var.columns


def test_seurat_n_top_genes_ge_n_genes_selects_all():
    # n_top_genes >= n_genes → the `cutoff = -inf` branch (line 291); every
    # gene with a finite normalized dispersion is selected.
    pytest.importorskip("scanpy")
    a = _lognorm_adata(seed=12, n_genes=20)
    HV._fast_hvg_seurat(a, n_top_genes=20)  # == n_genes
    # cutoff -inf -> disp_norm_clean >= -inf is True everywhere -> all selected.
    assert int(a.var["highly_variable"].sum()) == 20


def test_seurat_n_top_genes_gt_n_genes_selects_all():
    pytest.importorskip("scanpy")
    a = _lognorm_adata(seed=13, n_genes=18)
    HV._fast_hvg_seurat(a, n_top_genes=50)  # > n_genes
    assert int(a.var["highly_variable"].sum()) == 18


# --------------------------------------------------------------------------
# _fast_hvg_seurat_v3_batch — guard fallbacks (direct calls)
# --------------------------------------------------------------------------

def _raw_batch_adata(n_cells, n_genes, batch_codes, seed=0):
    """Raw-count CSR AnnData with a categorical 'batch' obs from explicit codes."""
    ad = pytest.importorskip("anndata")
    import pandas as pd
    rng = np.random.default_rng(seed)
    counts = rng.poisson(1.0, size=(n_cells, n_genes)).astype(np.float32)
    X = sparse.csr_matrix(counts)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    a = ad.AnnData(X)
    a.var_names = [f"g{j}" for j in range(n_genes)]
    a.obs["batch"] = pd.Categorical([f"b{c}" for c in batch_codes])
    return a


def test_v3_batch_tiny_matrix_defers(monkeypatch):
    # x.shape[0] < 2 → the small-matrix guard (line 390) defers to upstream.
    captured = {}
    monkeypatch.setattr(
        HV, "_orig_hvg",
        lambda: (lambda adata, **kw: captured.setdefault("deferred", True)))
    a = _raw_batch_adata(1, 5, [0], seed=20)  # only 1 row
    HV._fast_hvg_seurat_v3_batch(
        a, flavor="seurat_v3", n_top_genes=3, batch_key="batch")
    assert captured.get("deferred")


def test_v3_batch_zero_cols_defers(monkeypatch):
    # x.shape[1] == 0 also trips the same guard (line 390).
    captured = {}
    monkeypatch.setattr(
        HV, "_orig_hvg",
        lambda: (lambda adata, **kw: captured.setdefault("deferred", True)))
    a = _raw_batch_adata(6, 0, [0, 0, 1, 1, 0, 1], seed=21)
    HV._fast_hvg_seurat_v3_batch(
        a, flavor="seurat_v3", n_top_genes=3, batch_key="batch")
    assert captured.get("deferred")


def test_v3_batch_singleton_batch_defers(monkeypatch):
    # A batch with a single cell -> batch_sizes < 2 guard (line 421) defers.
    captured = {}
    monkeypatch.setattr(
        HV, "_orig_hvg",
        lambda: (lambda adata, **kw: captured.setdefault("deferred", True)))
    # codes: batch 0 has 5 cells, batch 1 has exactly 1 cell.
    a = _raw_batch_adata(6, 8, [0, 0, 0, 0, 0, 1], seed=22)
    HV._fast_hvg_seurat_v3_batch(
        a, flavor="seurat_v3", n_top_genes=3, batch_key="batch")
    assert captured.get("deferred")


def test_v3_batch_negative_codes_defers(monkeypatch):
    # A NaN/unused category yields a -1 code -> the `codes < 0` guard (line 406).
    import pandas as pd
    captured = {}
    monkeypatch.setattr(
        HV, "_orig_hvg",
        lambda: (lambda adata, **kw: captured.setdefault("deferred", True)))
    a = _raw_batch_adata(6, 8, [0, 0, 0, 1, 1, 1], seed=23)
    # Inject a missing batch label (NaN) → cat code -1 for that row.
    cats = pd.Categorical(["b0", "b0", "b0", "b1", "b1", None])
    a.obs["batch"] = cats
    HV._fast_hvg_seurat_v3_batch(
        a, flavor="seurat_v3", n_top_genes=3, batch_key="batch")
    assert captured.get("deferred")


def test_v3_batch_too_few_nonconstant_genes_defers(monkeypatch):
    # A per-batch matrix with <2 non-constant genes hits the loess guard
    # (line 460). Use a near-constant matrix: only one gene varies.
    ad = pytest.importorskip("anndata")
    import pandas as pd
    captured = {}
    monkeypatch.setattr(
        HV, "_orig_hvg",
        lambda: (lambda adata, **kw: captured.setdefault("deferred", True)))
    n_cells, n_genes = 8, 6
    dense = np.full((n_cells, n_genes), 3.0, dtype=np.float32)  # constant
    dense[:, 0] = np.arange(n_cells, dtype=np.float32)          # one varying gene
    X = sparse.csr_matrix(dense)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    a = ad.AnnData(X)
    a.var_names = [f"g{j}" for j in range(n_genes)]
    a.obs["batch"] = pd.Categorical(["b0"] * (n_cells // 2) + ["b1"] * (n_cells // 2))
    HV._fast_hvg_seurat_v3_batch(
        a, flavor="seurat_v3", n_top_genes=3, batch_key="batch")
    assert captured.get("deferred")


def test_v3_batch_skmisc_missing_defers(monkeypatch):
    # Simulate skmisc.loess being unavailable -> the ImportError fallback
    # (lines 363-364). Force the `from skmisc.loess import loess` to raise.
    captured = {}
    monkeypatch.setattr(
        HV, "_orig_hvg",
        lambda: (lambda adata, **kw: captured.setdefault("deferred", True)))

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "skmisc.loess" or name.startswith("skmisc"):
            raise ImportError("no skmisc")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    a = _raw_batch_adata(8, 6, [0, 0, 0, 0, 1, 1, 1, 1], seed=24)
    HV._fast_hvg_seurat_v3_batch(
        a, flavor="seurat_v3", n_top_genes=3, batch_key="batch")
    assert captured.get("deferred")
