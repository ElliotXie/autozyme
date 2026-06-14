"""End-to-end ``sc.tl.leiden`` patched-path tests.

Drives a tiny real AnnData (with a neighbors graph) through ``fast_leiden``
and asserts it produces a valid partition, covering the wrapper dispatch /
graph-build / fork-run / defer-to-upstream branches in ``_leiden.py``.

The igraph C Leiden and leidenalg are different algorithms, so we assert
structural validity + key bookkeeping rather than label-for-label parity.

Env recipe: KMP_DUPLICATE_LIB_OK=TRUE NUMBA_THREADING_LAYER=workqueue
NUMBA_NUM_THREADS=14 OMP_NUM_THREADS=2.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

sc = pytest.importorskip("scanpy")
ad = pytest.importorskip("anndata")
pytest.importorskip("igraph")
pytest.importorskip("leidenalg")
import autozyme  # noqa: E402


def _graph_adata(n_per=25, seed=40):
    """AnnData with two well-separated blobs + a kNN neighbors graph."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.3, size=(n_per, 12))
    b = rng.normal(6, 0.3, size=(n_per, 12))
    X = np.vstack([a, b]).astype(np.float32)
    adata = ad.AnnData(X)
    adata.obsm["X_pca"] = X  # use X directly as the embedding
    sc.pp.neighbors(adata, n_neighbors=10, use_rep="X_pca", random_state=0)
    return adata


@pytest.fixture(autouse=True)
def _activate():
    autozyme.activate("scanpy")
    yield


def test_leiden_fast_path_two_clusters():
    a = _graph_adata()
    sc.tl.leiden(a, flavor="igraph", n_iterations=2, directed=False, random_state=0)
    labels = a.obs["leiden"]
    # Two separated blobs → at least 2 clusters, and the params bookkeeping.
    assert labels.nunique() >= 2
    assert a.uns["leiden"]["params"]["random_state"] == 0


def test_leiden_inplace_returns_none():
    a = _graph_adata()
    assert sc.tl.leiden(a, flavor="igraph", directed=False, n_iterations=2) is None


def test_leiden_copy_true_returns_new():
    a = _graph_adata()
    out = sc.tl.leiden(a, flavor="igraph", directed=False, n_iterations=2, copy=True)
    assert out is not a
    assert "leiden" in out.obs.columns
    assert "leiden" not in a.obs.columns


def test_leiden_key_added_custom():
    a = _graph_adata()
    sc.tl.leiden(a, flavor="igraph", directed=False, n_iterations=2,
                 key_added="clusters")
    assert "clusters" in a.obs.columns
    assert "clusters" in a.uns


def test_leiden_no_weights_path():
    a = _graph_adata()
    sc.tl.leiden(a, flavor="igraph", directed=False, n_iterations=2,
                 use_weights=False)
    assert a.obs["leiden"].nunique() >= 2


def test_leiden_auto_iterations_default():
    # n_iterations=-1 (auto) stays on the fast path (capped to 2 internally).
    a = _graph_adata()
    sc.tl.leiden(a, flavor="igraph", directed=False, n_iterations=-1)
    assert "leiden" in a.obs.columns


def test_leiden_leidenalg_flavor_defers():
    # flavor='leidenalg' → wrapper defers to upstream (different algorithm).
    a = _graph_adata()
    sc.tl.leiden(a, flavor="leidenalg", n_iterations=2, directed=False)
    assert "leiden" in a.obs.columns


def test_leiden_directed_defers():
    # directed=True trips the wrapper's defer guard → upstream sc.tl.leiden,
    # which (with flavor='igraph') itself rejects a directed graph. Either way
    # the wrapper's defer branch is exercised; we just confirm the call reaches
    # upstream by asserting upstream's own ValueError propagates unchanged.
    a = _graph_adata()
    with pytest.raises(ValueError, match="directed graph"):
        sc.tl.leiden(a, flavor="igraph", directed=True, n_iterations=2)


def test_leiden_high_iterations_defers():
    # n_iterations > 2 → wrapper defers to upstream rather than under-converge.
    a = _graph_adata()
    sc.tl.leiden(a, flavor="igraph", directed=False, n_iterations=5)
    assert "leiden" in a.obs.columns


def test_leiden_zyme_false_uses_upstream():
    a = _graph_adata()
    sc.tl.leiden(a, flavor="igraph", directed=False, n_iterations=2,
                 random_state=0, zyme=False)
    assert "leiden" in a.obs.columns
