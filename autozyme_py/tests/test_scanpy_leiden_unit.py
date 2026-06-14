"""Unit tests for autozyme.scanpy._leiden.

The scBLAS native path is env-gated (default OFF) and the fork path is hard to
unit-test deterministically, so we focus on:
  - env-var parsing helpers (_scblas_enabled, _scblas_threads,
    _graph_cache_enabled, _graph_cache_limit, _graph_cache_key)
  - candidate-path discovery (_scblas_candidate_paths)
  - _build_simple_graph (upper-triangle simple igraph build, weight doubling)
  - fast_leiden dispatch guards + one real end-to-end clustering on a tiny
    block-diagonal graph (compared to vanilla scanpy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from autozyme.scanpy import _leiden as LD


# ==========================================================================
# Env-var parsing helpers
# ==========================================================================

@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("on", True), ("yes", True),
    ("TRUE", True), ("Yes", True),
    ("0", False), ("false", False), ("off", False), ("no", False),
    ("", False), ("garbage", False),
])
def test_scblas_enabled(monkeypatch, val, expected):
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN", val)
    assert LD._scblas_enabled() is expected


def test_scblas_enabled_default_false(monkeypatch):
    monkeypatch.delenv("AUTOZYME_SCBLAS_LEIDEN", raising=False)
    assert LD._scblas_enabled() is False


def test_scblas_threads_parses_int(monkeypatch):
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_THREADS", "8")
    assert LD._scblas_threads() == 8


def test_scblas_threads_default_one(monkeypatch):
    monkeypatch.delenv("AUTOZYME_SCBLAS_LEIDEN_THREADS", raising=False)
    assert LD._scblas_threads() == 1


def test_scblas_threads_malformed_returns_one(monkeypatch):
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_THREADS", "abc")
    assert LD._scblas_threads() == 1


@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("on", True), ("yes", True),
    ("0", False), ("", False), ("nope", False),
])
def test_graph_cache_enabled(monkeypatch, val, expected):
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_CACHE", val)
    assert LD._graph_cache_enabled() is expected


def test_graph_cache_limit_default(monkeypatch):
    monkeypatch.delenv("AUTOZYME_SCBLAS_LEIDEN_CACHE_SIZE", raising=False)
    assert LD._graph_cache_limit() == 4


def test_graph_cache_limit_parses(monkeypatch):
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_CACHE_SIZE", "10")
    assert LD._graph_cache_limit() == 10


def test_graph_cache_limit_floor_one(monkeypatch):
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_CACHE_SIZE", "0")
    assert LD._graph_cache_limit() == 1  # max(1, value)


def test_graph_cache_limit_malformed_default(monkeypatch):
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_CACHE_SIZE", "xyz")
    assert LD._graph_cache_limit() == 4


# ==========================================================================
# _graph_cache_key
# ==========================================================================

def test_graph_cache_key_none_when_cache_disabled(monkeypatch):
    monkeypatch.delenv("AUTOZYME_SCBLAS_LEIDEN_CACHE", raising=False)
    adj = sparse.csr_matrix(np.eye(3, dtype=np.float64))
    assert LD._graph_cache_key(adj, True) is None


def test_graph_cache_key_tuple_when_enabled(monkeypatch):
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_CACHE", "1")
    adj = sparse.csr_matrix(np.eye(4, dtype=np.float64))
    key = LD._graph_cache_key(adj, True)
    assert key is not None
    assert key[1] == (4, 4)        # shape
    assert key[2] == int(adj.nnz)  # nnz
    assert key[3] is True          # use_weights


def test_graph_cache_key_distinguishes_use_weights(monkeypatch):
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_CACHE", "1")
    adj = sparse.csr_matrix(np.eye(3, dtype=np.float64))
    k_true = LD._graph_cache_key(adj, True)
    k_false = LD._graph_cache_key(adj, False)
    assert k_true != k_false


# ==========================================================================
# _scblas_candidate_paths
# ==========================================================================

def test_candidate_paths_yields_env_first(monkeypatch):
    monkeypatch.setenv("SCBLAS_LIB", "/tmp/my_scblas.dylib")
    monkeypatch.delenv("AUTOZYME_SCBLAS_LIB", raising=False)
    paths = list(LD._scblas_candidate_paths())
    assert Path("/tmp/my_scblas.dylib") in paths
    assert paths[0] == Path("/tmp/my_scblas.dylib")


def test_candidate_paths_uses_autozyme_scblas_lib(monkeypatch):
    monkeypatch.delenv("SCBLAS_LIB", raising=False)
    monkeypatch.setenv("AUTOZYME_SCBLAS_LIB", "/tmp/alt.so")
    paths = list(LD._scblas_candidate_paths())
    assert Path("/tmp/alt.so") in paths


def test_candidate_paths_no_env_does_not_crash(monkeypatch):
    monkeypatch.delenv("SCBLAS_LIB", raising=False)
    monkeypatch.delenv("AUTOZYME_SCBLAS_LIB", raising=False)
    # Returns a (possibly empty) iterable without raising.
    list(LD._scblas_candidate_paths())


# ==========================================================================
# _ptr
# ==========================================================================

def test_ptr_returns_pointer():
    import ctypes
    arr = np.array([1, 2, 3], dtype=np.int64)
    p = LD._ptr(arr, ctypes.c_int64)
    assert isinstance(p, ctypes.POINTER(ctypes.c_int64))


# ==========================================================================
# _scblas_upper_edges — returns None when lib unavailable (default state)
# ==========================================================================

def test_scblas_upper_edges_none_without_lib(monkeypatch):
    # Default: scBLAS disabled -> _load_scblas_leiden returns None -> None.
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN", "0")
    adj = sparse.csr_matrix(np.eye(3, dtype=np.float64))
    assert LD._scblas_upper_edges(adj) is None


def test_load_scblas_leiden_none_when_disabled(monkeypatch):
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN", "0")
    assert LD._load_scblas_leiden() is None


# ==========================================================================
# _build_simple_graph — upper triangle, weight doubling
# ==========================================================================

def _block_adjacency():
    """Two disconnected triangles (block-diagonal symmetric adjacency)."""
    A = np.zeros((6, 6), dtype=np.float64)
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        A[i, j] = A[j, i] = 1.0
    for i, j in [(3, 4), (4, 5), (3, 5)]:
        A[i, j] = A[j, i] = 1.0
    return sparse.csr_matrix(A)


def test_build_simple_graph_edge_count_is_upper_triangle(monkeypatch):
    pytest.importorskip("igraph")
    monkeypatch.delenv("AUTOZYME_SCBLAS_LEIDEN", raising=False)  # use scipy triu
    adj = _block_adjacency()
    g = LD._build_simple_graph(adj, use_weights=True)
    assert g.vcount() == 6
    # 3 edges per triangle, upper triangle only -> 6 edges total.
    assert g.ecount() == 6


def test_build_simple_graph_doubles_weights(monkeypatch):
    pytest.importorskip("igraph")
    monkeypatch.delenv("AUTOZYME_SCBLAS_LEIDEN", raising=False)
    A = np.zeros((3, 3), dtype=np.float64)
    A[0, 1] = A[1, 0] = 2.5
    adj = sparse.csr_matrix(A)
    g = LD._build_simple_graph(adj, use_weights=True)
    # weight = upper.data * 2 = 5.0
    assert g.es["weight"] == [5.0]


def test_build_simple_graph_no_weights(monkeypatch):
    pytest.importorskip("igraph")
    monkeypatch.delenv("AUTOZYME_SCBLAS_LEIDEN", raising=False)
    adj = _block_adjacency()
    g = LD._build_simple_graph(adj, use_weights=False)
    assert "weight" not in g.es.attributes()


# ==========================================================================
# fast_leiden dispatch guards
# ==========================================================================

def test_fast_leiden_zyme_false_delegates(monkeypatch):
    monkeypatch.setattr(LD, "_orig_leiden", lambda: (lambda adata, **kw: "ORIG"))
    assert LD.fast_leiden("ADATA", zyme=False) == "ORIG"


@pytest.mark.parametrize("kw", [
    {"flavor": "leidenalg"},
    {"restrict_to": ("g", ["a"])},
    {"partition_type": object()},
    {"directed": True},
    {"n_iterations": 5},  # > 2 -> defer to upstream igraph
])
def test_fast_leiden_guard_kwargs_delegate(monkeypatch, kw):
    pytest.importorskip("anndata")
    ad = pytest.importorskip("anndata")
    monkeypatch.setattr(LD, "_orig_leiden", lambda: (lambda adata, **k: "ORIG"))
    a = ad.AnnData(np.zeros((4, 2), dtype=np.float32))
    assert LD.fast_leiden(a, **kw) == "ORIG"


# ==========================================================================
# fast_leiden end-to-end on a tiny block-diagonal graph (vs vanilla)
# ==========================================================================

def _adata_with_neighbors():
    ad = pytest.importorskip("anndata")
    sc = pytest.importorskip("scanpy")
    rng = np.random.default_rng(0)
    # Two well-separated blobs -> two clusters.
    a_pts = rng.normal(0, 0.2, size=(20, 5))
    b_pts = rng.normal(8, 0.2, size=(20, 5))
    X = np.vstack([a_pts, b_pts]).astype(np.float32)
    a = ad.AnnData(X)
    with __import__("autozyme").disabled():
        sc.pp.neighbors(a, n_neighbors=10, use_rep="X")
    return a


def test_fast_leiden_finds_two_clusters():
    sc = pytest.importorskip("scanpy")
    pytest.importorskip("igraph")
    a = _adata_with_neighbors()
    out = LD.fast_leiden(a, flavor="igraph", n_iterations=2, directed=False,
                         random_state=0)
    assert out is None
    labels = a.obs["leiden"]
    # Two well-separated blobs: the clustering must never merge a point from
    # blob A with a point from blob B (each blob may further sub-split at
    # resolution=1.0, but the two halves stay in disjoint cluster sets).
    assert labels.nunique() >= 2
    first = set(labels.values[:20])
    second = set(labels.values[20:])
    assert first.isdisjoint(second)
    # params recorded.
    assert a.uns["leiden"]["params"]["random_state"] == 0


def test_fast_leiden_copy_returns_new():
    sc = pytest.importorskip("scanpy")
    pytest.importorskip("igraph")
    a = _adata_with_neighbors()
    out = LD.fast_leiden(a, flavor="igraph", n_iterations=2, directed=False,
                         random_state=0, copy=True)
    assert out is not None and out is not a
    assert "leiden" not in a.obs.columns
    assert "leiden" in out.obs.columns
