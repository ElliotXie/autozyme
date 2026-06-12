"""Contract tests for ``sc.pp.highly_variable_genes``.

Dispatch surface:
  - flavor in {"seurat"}                + numba  -> fast path
  - flavor in {"seurat_v3","seurat_v3_paper"} + batch_key  -> v3-batch fast path
  - flavor with batch_key  -> upstream
  - other flavors          -> upstream

Contract test pins:
  - return type matches vanilla for the user-facing flavors
  - inplace=True default mutates and returns None
  - zyme=False per-call delegates cleanly
"""
from __future__ import annotations

import pytest


@pytest.fixture
def log_norm_adata(tiny_adata):
    """HVG (flavor='seurat'/'cell_ranger') expects log-normalized input."""
    import autozyme
    import scanpy as sc
    a = tiny_adata.copy()
    # log-normalize via vanilla path to avoid coupling this fixture to
    # the autozyme fast-path correctness (which has its own tests).
    with autozyme.disabled():
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    return a


def test_hvg_seurat_default_return_matches_vanilla(log_norm_adata):
    """flavor='seurat' fast path return must match vanilla (None, in-place)."""
    import autozyme
    import scanpy as sc

    a_fast = log_norm_adata.copy()
    a_vanilla = log_norm_adata.copy()

    with autozyme.disabled():
        ref = sc.pp.highly_variable_genes(a_vanilla, flavor="seurat",
                                          n_top_genes=30)
    out = sc.pp.highly_variable_genes(a_fast, flavor="seurat",
                                      n_top_genes=30)

    assert type(out) is type(ref)
    assert "highly_variable" in a_fast.var.columns
    assert "highly_variable" in a_vanilla.var.columns


def test_hvg_cell_ranger_delegates(log_norm_adata):
    """flavor='cell_ranger' is not on the fast path -> upstream delegation."""
    import autozyme
    import scanpy as sc

    a_fast = log_norm_adata.copy()
    a_vanilla = log_norm_adata.copy()

    with autozyme.disabled():
        ref = sc.pp.highly_variable_genes(a_vanilla, flavor="cell_ranger",
                                          n_top_genes=30)
    out = sc.pp.highly_variable_genes(a_fast, flavor="cell_ranger",
                                      n_top_genes=30)

    assert type(out) is type(ref)
    assert "highly_variable" in a_fast.var.columns


def test_hvg_zyme_false_delegates(log_norm_adata):
    """Per-call ``zyme=False`` must match vanilla output on HVG flags."""
    import autozyme
    import numpy as np
    import scanpy as sc

    a_escape = log_norm_adata.copy()
    a_vanilla = log_norm_adata.copy()
    with autozyme.disabled():
        sc.pp.highly_variable_genes(a_vanilla, flavor="seurat",
                                    n_top_genes=30)
    sc.pp.highly_variable_genes(a_escape, flavor="seurat",
                                n_top_genes=30, zyme=False)

    np.testing.assert_array_equal(
        a_escape.var["highly_variable"].to_numpy(),
        a_vanilla.var["highly_variable"].to_numpy(),
    )
