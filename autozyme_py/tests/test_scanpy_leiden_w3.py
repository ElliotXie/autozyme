"""Wave-3 ``leiden`` coverage — the reachable pure-python lines wave-1/wave-2
missed in ``_leiden.py``.

Targets:
  * ``_graph_cache_key`` exception guard (lines 71-72) — adjacency whose
    ``.shape`` access raises returns None.
  * the graph cache hit (line 250) and eviction-when-full (lines 268-271)
    branches in ``_build_simple_graph`` (env-gated via
    ``AUTOZYME_SCBLAS_LEIDEN_CACHE``).
  * ``_run_leiden_direct`` (lines 277-280) — the Windows / direct execution
    path, callable directly on Unix with a real igraph Graph.
  * the fork child-failure ``RuntimeError`` (line 336) — community_leiden that
    raises in the child leaves the DONE marker unwritten, so the parent raises.

NOT reachable here (documented in the agent report):
  * the entire scBLAS native path (_load_scblas_leiden body, _scblas_upper_edges
    with a loaded lib, lines 94-156/168-241/259) — the scBLAS shared library is
    not built/shipped in autozyme_release (no ``lib_scblas/`` dir), so
    ``_load_scblas_leiden`` short-circuits to None and the SciPy-triu fallback
    is always taken. Wave-1 already covered the env-parsing helpers and the
    "lib unavailable -> None" returns.
  * the fork *child* body (lines 298-316) — runs in a forked child process and
    is invisible to the parent's coverage tracer.

Env recipe: KMP_DUPLICATE_LIB_OK=TRUE NUMBA_THREADING_LAYER=workqueue
NUMBA_NUM_THREADS=14 OMP_NUM_THREADS=2.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from autozyme.scanpy import _leiden as LD


def _block_adjacency():
    """Two disconnected triangles (block-diagonal symmetric adjacency)."""
    A = np.zeros((6, 6), dtype=np.float64)
    for i, j in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        A[i, j] = A[j, i] = 1.0
    return sparse.csr_matrix(A)


# --------------------------------------------------------------------------
# _graph_cache_key — exception guard (lines 71-72)
# --------------------------------------------------------------------------

def test_graph_cache_key_returns_none_on_shape_error(monkeypatch):
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_CACHE", "1")  # cache enabled

    class _BadAdj:
        nnz = 0

        @property
        def shape(self):
            raise RuntimeError("no shape")

    # The try/except around shape/nnz extraction swallows the error -> None.
    assert LD._graph_cache_key(_BadAdj(), True) is None


# --------------------------------------------------------------------------
# _build_simple_graph — cache hit + eviction (lines 250, 268-271)
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_cache():
    # Keep the module-level graph cache from leaking between tests.
    LD._GRAPH_CACHE.clear()
    yield
    LD._GRAPH_CACHE.clear()


def test_build_simple_graph_cache_hit_returns_same_object(monkeypatch):
    pytest.importorskip("igraph")
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_CACHE", "1")
    adj = _block_adjacency()
    g1 = LD._build_simple_graph(adj, use_weights=True)
    g2 = LD._build_simple_graph(adj, use_weights=True)
    # Same cache key (same id/shape/nnz/use_weights) -> cached object reused.
    assert g1 is g2
    assert len(LD._GRAPH_CACHE) == 1


def test_build_simple_graph_cache_eviction_at_limit(monkeypatch):
    pytest.importorskip("igraph")
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_CACHE", "1")
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_CACHE_SIZE", "2")  # cap = 2
    # Distinct adjacency objects -> distinct cache keys (different id()).
    adjs = [_block_adjacency() for _ in range(3)]
    for adj in adjs:
        LD._build_simple_graph(adj, use_weights=True)
    # Cap is 2: inserting the 3rd evicts the oldest -> cache size stays <= 2.
    assert len(LD._GRAPH_CACHE) == 2


def test_build_simple_graph_cache_disabled_no_storage(monkeypatch):
    pytest.importorskip("igraph")
    monkeypatch.delenv("AUTOZYME_SCBLAS_LEIDEN_CACHE", raising=False)
    LD._build_simple_graph(_block_adjacency(), use_weights=True)
    assert len(LD._GRAPH_CACHE) == 0  # nothing cached when disabled


# --------------------------------------------------------------------------
# _run_leiden_direct — the direct (non-fork) execution path (lines 277-280)
# --------------------------------------------------------------------------

def test_run_leiden_direct_clusters_block_graph(monkeypatch):
    pytest.importorskip("igraph")
    monkeypatch.delenv("AUTOZYME_SCBLAS_LEIDEN", raising=False)
    g = LD._build_simple_graph(_block_adjacency(), use_weights=True)
    membership = LD._run_leiden_direct(
        g, random_state=0,
        clustering_args={
            "objective_function": "modularity",
            "n_iterations": 2,
            "weights": "weight",
            "resolution": 1.0,
        },
    )
    assert membership.dtype == np.int32
    assert membership.shape == (6,)
    # Two disconnected triangles -> exactly two communities, split 0-2 vs 3-5.
    assert len(set(membership.tolist())) == 2
    assert set(membership[:3].tolist()).isdisjoint(set(membership[3:].tolist()))


# --------------------------------------------------------------------------
# _run_leiden_in_fork — child failure -> parent RuntimeError (line 336)
# --------------------------------------------------------------------------

@pytest.mark.skipif(LD._IS_WINDOWS, reason="fork path is Unix-only")
def test_run_leiden_in_fork_child_failure_raises():
    pytest.importorskip("igraph")

    class _RaisingGraph:
        """A stand-in graph whose community_leiden blows up in the child."""

        def vcount(self):
            return 6

        def community_leiden(self, **kwargs):
            raise RuntimeError("deliberate child failure")

    # The child catches the exception, os._exit(1) without the DONE marker,
    # so the parent raises a RuntimeError about the missing completion marker.
    with pytest.raises(RuntimeError, match="without writing"):
        LD._run_leiden_in_fork(
            _RaisingGraph(), 6, random_state=0,
            clustering_args={"objective_function": "modularity",
                             "n_iterations": 2},
        )


@pytest.mark.skipif(LD._IS_WINDOWS, reason="fork path is Unix-only")
def test_run_leiden_in_fork_success_returns_membership():
    pytest.importorskip("igraph")
    g = LD._build_simple_graph(_block_adjacency(), use_weights=True)
    membership = LD._run_leiden_in_fork(
        g, g.vcount(), random_state=0,
        clustering_args={
            "objective_function": "modularity",
            "n_iterations": 2,
            "weights": "weight",
            "resolution": 1.0,
        },
    )
    assert membership.shape == (6,)
    assert len(set(membership.tolist())) == 2
