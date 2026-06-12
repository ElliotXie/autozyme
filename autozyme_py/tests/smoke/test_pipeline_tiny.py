"""End-to-end pipeline smoke for the scanpy patch on a tiny synthetic input.

Per Cat 3 design: pipeline-shape libraries (scanpy, seurat) get an
end-to-end smoke; other patches only get contract tests. This file is the
scanpy version. The Seurat counterpart lives in
``autozyme_r/tests/testthat/test-smoke_pipeline.R``.

Goal: run normalize -> log -> HVG -> PCA -> neighbors -> leiden through
the patched namespace on a < 1k-cell synthetic AnnData in under 30s. We
do NOT compare to vanilla here — that's test_scanpy_smoke.py's parity
job. The smoke catches regressions where a patch silently breaks chaining
(e.g. May 2026 ``normalize_total(copy=True) -> None`` cascade-crashing
downstream).
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
sc = pytest.importorskip("scanpy")
pytest.importorskip("scipy")
pytest.importorskip("anndata")


@pytest.fixture(scope="module")
def synthetic_adata():
    """700 cells x 600 genes Poisson-distributed CSR, structured into 4 groups."""
    import anndata as ad
    from scipy import sparse

    rng = np.random.default_rng(42)
    n_cells, n_genes = 700, 600
    # Per-group mean shift gives leiden something to cluster.
    group_means = np.array([1.0, 2.0, 3.5, 1.5], dtype=np.float32)
    group_assign = rng.integers(0, 4, size=n_cells)
    base = np.repeat(group_means[group_assign, None], n_genes, axis=1)
    noise = rng.poisson(base + 0.5).astype(np.float32)
    # Sparsify ~50% of entries.
    mask = rng.random((n_cells, n_genes)) < 0.5
    noise[mask] = 0.0
    X = sparse.csr_matrix(noise)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    return ad.AnnData(X)


def test_pipeline_end_to_end_patched(synthetic_adata):
    """Full scanpy chain under autozyme.activate must not crash and must populate
    all expected obsm/uns slots. Catches Bug 2-class chain-break regressions."""
    import autozyme
    autozyme.activate("scanpy")

    a = synthetic_adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=200, flavor="seurat")
    sc.tl.pca(a, n_comps=20)
    sc.pp.neighbors(a, n_neighbors=15)
    sc.tl.leiden(a, flavor="igraph", n_iterations=2, directed=False,
                 random_state=0)

    assert "highly_variable" in a.var.columns
    assert "X_pca" in a.obsm
    assert a.obsm["X_pca"].shape == (700, 20)
    assert "neighbors" in a.uns
    assert "leiden" in a.obs.columns
    assert a.obs["leiden"].nunique() >= 2  # synthetic groups should separate


def test_pipeline_end_to_end_copy_chain(synthetic_adata):
    """``copy=True`` chain pattern must work end-to-end.

    This is the standard scanpy tutorial pattern for keeping intermediates
    around. It was broken in the May 2026 ``normalize_total(copy=True)``
    bug — every downstream step received ``None``.
    """
    import autozyme
    autozyme.activate("scanpy")

    a_raw = synthetic_adata.copy()
    a_norm = sc.pp.normalize_total(a_raw, target_sum=1e4, copy=True)
    assert a_norm is not None and a_norm is not a_raw

    a_log = sc.pp.log1p(a_norm, copy=True)
    assert a_log is not None and a_log is not a_norm

    # Confirm raw input is untouched by the copy chain.
    assert np.array_equal(
        a_raw.X.toarray(), synthetic_adata.X.toarray()
    )
