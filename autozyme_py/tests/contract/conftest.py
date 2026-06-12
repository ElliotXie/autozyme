"""Shared fixtures for per-API contract tests.

A contract test asserts that a patched function's PUBLIC SHAPE matches
upstream: signature, return type, mutation semantics, kwarg permutations.
It does NOT assert speed or numerical fidelity (those live in
test_scanpy_smoke.py's parity block). Bugs of the form "copy=True returns
None" are exactly what these tests catch and what whole-pipeline smoke
will miss because the smoke chains in-place by default.

Pattern: every patched function gets one file ``test_<fn>.py`` that walks
the cartesian product of its kwargs and, for each, asserts:
  - return type matches upstream
  - identity (is / is not the input adata) matches upstream
  - the ``zyme=False`` escape hatch still delegates to upstream

Add a new patched function -> add a new test_<fn>.py here. The pattern
file (``test_normalize_total.py``) is the canonical template.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _activate_autozyme():
    """Every contract test runs with autozyme.activate('scanpy') applied."""
    import autozyme
    autozyme.activate("scanpy")
    try:
        yield
    finally:
        autozyme.deactivate_all()


@pytest.fixture
def tiny_adata():
    """Minimal CSR AnnData with raw-count-shaped data — under 1 ms to build."""
    np = pytest.importorskip("numpy")
    sparse = pytest.importorskip("scipy.sparse")
    ad = pytest.importorskip("anndata")

    rng = np.random.default_rng(0)
    n_cells, n_genes = 80, 120
    X = sparse.random(n_cells, n_genes, density=0.25, format="csr",
                      dtype=np.float32, random_state=rng).astype(np.float32)
    X.data = rng.integers(1, 10, size=X.nnz).astype(np.float32)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    return ad.AnnData(X)


@pytest.fixture
def tiny_dense_adata():
    """Dense (ndarray) AnnData — exercises non-sparse fallback paths."""
    np = pytest.importorskip("numpy")
    ad = pytest.importorskip("anndata")

    rng = np.random.default_rng(1)
    X = rng.random((50, 80), dtype=np.float32) * 5.0
    return ad.AnnData(X)
