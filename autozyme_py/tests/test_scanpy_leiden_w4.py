"""Wave-4 ``leiden`` coverage — the last reachable pure-python lines in
``_leiden.py`` that waves 1-3 left uncovered.

Wave-3 documented the scBLAS native path (``_load_scblas_leiden`` body /
``_scblas_upper_edges`` with a loaded lib / line 259) as "not reachable because
the scBLAS shared library is not built/shipped in autozyme_release". That is
true for the *real* library, but the pure-python plumbing around ctypes is
still drivable:

  * ``_load_scblas_leiden`` (113-156): with ``AUTOZYME_SCBLAS_LEIDEN=1`` and
    NO real lib present, the candidate loop iterates and returns None (113-126,
    156); pointing ``SCBLAS_LIB`` at a real-but-symbol-less dylib exercises the
    ``ctypes.CDLL`` load + the ``except Exception`` fallback (127-128, 154-155);
    a repeat call with a matching load-key returns the cached handle (118-119).
  * ``_scblas_upper_edges`` (168-241): driven against a *fake* lib object
    (monkeypatched ``_load_scblas_leiden``) whose extraction functions compute
    the real upper-triangle edges. Covers the threads==1 and threads>1 paths
    plus the three failure-return guards.
  * ``_build_simple_graph`` line 259 (``sources, targets, weights = extracted``)
    via the same fake-lib path.
  * ``fast_leiden`` Windows / direct-execution branch (line 399) via a
    monkeypatched ``_IS_WINDOWS = True``.
  * the ``except OSError: pass`` on the fork temp-file unlink (330-331).

Still NOT reachable (documented): the fork *child* body (298-316, runs in a
forked process, invisible to the parent tracer) and the real native-symbol
calls inside a genuinely-loaded scBLAS lib.

Env recipe: KMP_DUPLICATE_LIB_OK=TRUE NUMBA_THREADING_LAYER=workqueue
NUMBA_NUM_THREADS=14 OMP_NUM_THREADS=2.
"""
from __future__ import annotations

import ctypes

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


@pytest.fixture(autouse=True)
def _reset_scblas_state(monkeypatch):
    # The module caches load attempts in globals; reset before each test so
    # the load-key logic is exercised cleanly and nothing leaks across tests.
    monkeypatch.setattr(LD, "_SCBLAS_LIB", None, raising=False)
    monkeypatch.setattr(LD, "_SCBLAS_TRIED", False, raising=False)
    monkeypatch.setattr(LD, "_SCBLAS_LOAD_KEY", None, raising=False)
    LD._GRAPH_CACHE.clear()
    yield
    LD._GRAPH_CACHE.clear()


# ==========================================================================
# _load_scblas_leiden — the candidate-loop + load-failure body (113-156)
# ==========================================================================

def test_load_scblas_enabled_no_lib_returns_none(monkeypatch):
    # scBLAS enabled but no real lib anywhere -> the candidate loop runs (the
    # paths don't exist) and the function returns None (lines 113-126, 156).
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN", "1")
    monkeypatch.delenv("SCBLAS_LIB", raising=False)
    monkeypatch.setenv("AUTOZYME_SCBLAS_LIB", "/nonexistent/path/libscblas.dylib")
    assert LD._load_scblas_leiden() is None
    # The attempt is recorded so a repeat short-circuits via the cache.
    assert LD._SCBLAS_TRIED is True


def test_load_scblas_cached_return_on_repeat(monkeypatch):
    # Second call with the SAME load-key returns the cached handle without
    # re-walking the candidate paths (lines 118-119).
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN", "1")
    monkeypatch.delenv("SCBLAS_LIB", raising=False)
    monkeypatch.delenv("AUTOZYME_SCBLAS_LIB", raising=False)
    first = LD._load_scblas_leiden()       # populates _SCBLAS_TRIED / _LOAD_KEY
    sentinel = object()
    monkeypatch.setattr(LD, "_SCBLAS_LIB", sentinel, raising=False)
    second = LD._load_scblas_leiden()      # cache hit -> returns _SCBLAS_LIB
    assert second is sentinel
    assert first is None


def test_load_scblas_real_dylib_missing_symbols_falls_back(monkeypatch, tmp_path):
    # Point SCBLAS_LIB at a real, *on-disk*, loadable dylib that does NOT export
    # the scblas_* symbols. ``path.exists()`` is True (so the candidate loop
    # does not `continue`), ``ctypes.CDLL`` succeeds (line 128), but the first
    # symbol/argtypes assignment raises AttributeError, hitting the
    # `except Exception: _SCBLAS_LIB = None` fallback (lines 154-155) and the
    # final `return None` (156).
    #
    # macOS system libs (libc/libSystem) live in the dyld shared cache and
    # report ``Path.exists() == False``, so they would be skipped by the
    # candidate loop. We compile a trivial empty dylib to a real file instead.
    import shutil
    import subprocess

    cc = shutil.which("cc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler to build a throwaway dylib")
    src = tmp_path / "empty.c"
    src.write_text("int _autozyme_placeholder(void){return 0;}\n")
    dylib = tmp_path / "libnotscblas.dylib"
    proc = subprocess.run(
        [cc, "-shared", "-o", str(dylib), str(src)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not dylib.exists():
        pytest.skip(f"could not build dylib: {proc.stderr}")

    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN", "1")
    monkeypatch.setenv("SCBLAS_LIB", str(dylib))
    monkeypatch.delenv("AUTOZYME_SCBLAS_LIB", raising=False)
    assert LD._load_scblas_leiden() is None
    assert LD._SCBLAS_LIB is None


# ==========================================================================
# _scblas_upper_edges — driven against a fake lib (168-241)
# ==========================================================================

def _fake_lib(parallel=False, *, break_count=False, break_offsets=False,
              break_written=False):
    """A stand-in for the loaded scBLAS lib.

    Its extraction functions compute the *real* upper-triangle edge counts /
    edges from the CSR pointers, so ``_scblas_upper_edges`` produces a valid
    partition unless one of the ``break_*`` flags forces a guard-failure.
    """

    def _read_csr(n, indptr_p, indices_p):
        # Reconstruct numpy views from the ctypes int64/int32 pointers.
        indptr = np.ctypeslib.as_array(indptr_p, shape=(n + 1,))
        nnz = int(indptr[-1])
        indices = np.ctypeslib.as_array(indices_p, shape=(nnz,))
        return indptr, indices

    class _Fn:
        def __init__(self, fn):
            self._fn = fn
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self._fn(*args)

    class _Lib:
        pass

    lib = _Lib()

    def counts(n, indptr_p, indices_p, counts_p):
        indptr, indices = _read_csr(n, indptr_p, indices_p)
        counts_arr = np.ctypeslib.as_array(counts_p, shape=(n,))
        total = 0
        for r in range(n):
            c = 0
            for k in range(int(indptr[r]), int(indptr[r + 1])):
                if int(indices[k]) > r:   # strict upper triangle
                    c += 1
            counts_arr[r] = c
            total += c
        if break_count:
            return -1
        if break_offsets:
            # Return a total that disagrees with the per-row counts written
            # into counts_arr, so offsets[-1] (cumsum of counts) != edge_count.
            return total + 1
        return total

    def counts_parallel(n, indptr_p, indices_p, counts_p, total_p, threads):
        total = counts(n, indptr_p, indices_p, counts_p)
        total_p._obj.value = total
        return 0

    def edges(n, indptr_p, indices_p, data_p, offsets_p, _flag,
              src_p, tgt_p, w_p):
        indptr, indices = _read_csr(n, indptr_p, indices_p)
        offsets = np.ctypeslib.as_array(offsets_p, shape=(n + 1,))
        written = 0
        for r in range(n):
            pos = int(offsets[r])
            for k in range(int(indptr[r]), int(indptr[r + 1])):
                j = int(indices[k])
                if j > r:
                    np.ctypeslib.as_array(src_p, shape=(int(offsets[-1]),))[pos] = r
                    np.ctypeslib.as_array(tgt_p, shape=(int(offsets[-1]),))[pos] = j
                    if w_p is not None:
                        np.ctypeslib.as_array(w_p, shape=(int(offsets[-1]),))[pos] = 1.0
                    pos += 1
                    written += 1
        if break_written:
            return written + 1
        return written

    def edges_parallel(n, indptr_p, indices_p, data_p, offsets_p, _flag,
                       src_p, tgt_p, w_p, total_p, threads):
        written = edges(n, indptr_p, indices_p, data_p, offsets_p, _flag,
                        src_p, tgt_p, w_p)
        total_p._obj.value = written
        return 0

    lib.scblas_leiden_csr_upper_counts_i32 = _Fn(counts)
    lib.scblas_leiden_csr_upper_counts_i32_parallel = _Fn(counts_parallel)
    lib.scblas_leiden_csr_upper_edges_f64 = _Fn(edges)
    lib.scblas_leiden_csr_upper_edges_f64_parallel = _Fn(edges_parallel)
    lib._break_offsets = break_offsets
    return lib


def test_scblas_upper_edges_single_thread_valid(monkeypatch):
    # threads==1 path through the whole counts->edges pipeline (178-185,
    # 200-222, 239-241), returning a valid (sources, targets, weights) tuple.
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_THREADS", "1")
    monkeypatch.setattr(LD, "_load_scblas_leiden", lambda: _fake_lib())
    adj = _block_adjacency()
    out = LD._scblas_upper_edges(adj, include_weights=True)
    assert out is not None
    sources, targets, weights = out
    # 6 strict-upper-triangle edges in the two triangles.
    assert len(sources) == 6
    assert len(targets) == 6
    assert weights is not None and len(weights) == 6
    # Every recorded edge is in the strict upper triangle.
    assert np.all(targets > sources)


def test_scblas_upper_edges_no_weights(monkeypatch):
    # include_weights=False -> data/weights stay None (covers the None branches
    # in the pointer plumbing).
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_THREADS", "1")
    monkeypatch.setattr(LD, "_load_scblas_leiden", lambda: _fake_lib())
    out = LD._scblas_upper_edges(_block_adjacency(), include_weights=False)
    assert out is not None
    _, _, weights = out
    assert weights is None


def test_scblas_upper_edges_parallel_path(monkeypatch):
    # threads>1 path (186-198, 223-238) via the *_parallel entry points.
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_THREADS", "2")
    monkeypatch.setattr(LD, "_load_scblas_leiden", lambda: _fake_lib(parallel=True))
    out = LD._scblas_upper_edges(_block_adjacency(), include_weights=True)
    assert out is not None
    sources, targets, _ = out
    assert len(sources) == 6
    assert np.all(targets > sources)


def test_scblas_upper_edges_negative_count_returns_none(monkeypatch):
    # edge_count < 0 -> early `return None` (lines 197-198).
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_THREADS", "1")
    monkeypatch.setattr(LD, "_load_scblas_leiden",
                        lambda: _fake_lib(break_count=True))
    assert LD._scblas_upper_edges(_block_adjacency()) is None


def test_scblas_upper_edges_written_mismatch_returns_none(monkeypatch):
    # written != edge_count -> `return None` (lines 239-240).
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_THREADS", "1")
    monkeypatch.setattr(LD, "_load_scblas_leiden",
                        lambda: _fake_lib(break_written=True))
    assert LD._scblas_upper_edges(_block_adjacency()) is None


def test_scblas_upper_edges_offsets_mismatch_returns_none(monkeypatch):
    # cumsum(counts)[-1] != edge_count -> `return None` (lines 203-204). The
    # fake reports a total one greater than the per-row counts it wrote.
    monkeypatch.setenv("AUTOZYME_SCBLAS_LEIDEN_THREADS", "1")
    monkeypatch.setattr(LD, "_load_scblas_leiden",
                        lambda: _fake_lib(break_offsets=True))
    assert LD._scblas_upper_edges(_block_adjacency()) is None


def test_scblas_upper_edges_dense_adjacency_returns_none(monkeypatch):
    # `not sparse.issparse(adjacency)` short-circuit (line 165) even with a
    # loaded lib.
    monkeypatch.setattr(LD, "_load_scblas_leiden", lambda: _fake_lib())
    dense = np.eye(4, dtype=np.float64)
    assert LD._scblas_upper_edges(dense) is None


def test_scblas_upper_edges_lib_none_returns_none(monkeypatch):
    # lib is None -> line 165 short-circuit.
    monkeypatch.setattr(LD, "_load_scblas_leiden", lambda: None)
    assert LD._scblas_upper_edges(_block_adjacency()) is None


# ==========================================================================
# _build_simple_graph — the `extracted` (scBLAS) branch (line 259)
# ==========================================================================

def test_build_simple_graph_uses_extracted_edges(monkeypatch):
    pytest.importorskip("igraph")
    monkeypatch.delenv("AUTOZYME_SCBLAS_LEIDEN_CACHE", raising=False)
    # Make _scblas_upper_edges return a concrete edge list so _build_simple_graph
    # takes the `else: sources, targets, weights = extracted` branch (line 259).
    sources = np.array([0, 1], dtype=np.int32)
    targets = np.array([1, 2], dtype=np.int32)
    weights = np.array([3.0, 4.0], dtype=np.float64)
    monkeypatch.setattr(LD, "_scblas_upper_edges",
                        lambda adj, include_weights=True: (sources, targets, weights))
    adj = _block_adjacency()
    g = LD._build_simple_graph(adj, use_weights=True)
    assert g.vcount() == 6
    assert g.ecount() == 2
    # Weights came straight from the extracted array (NOT doubled like the
    # scipy-triu fallback).
    assert sorted(g.es["weight"]) == [3.0, 4.0]


def test_build_simple_graph_extracted_no_weights(monkeypatch):
    pytest.importorskip("igraph")
    monkeypatch.delenv("AUTOZYME_SCBLAS_LEIDEN_CACHE", raising=False)
    sources = np.array([0], dtype=np.int32)
    targets = np.array([1], dtype=np.int32)
    monkeypatch.setattr(LD, "_scblas_upper_edges",
                        lambda adj, include_weights=True: (sources, targets, None))
    g = LD._build_simple_graph(_block_adjacency(), use_weights=False)
    assert g.ecount() == 1
    assert "weight" not in g.es.attributes()


# ==========================================================================
# fast_leiden — the Windows / direct-execution branch (line 399)
# ==========================================================================

def test_fast_leiden_windows_branch_runs_direct(monkeypatch):
    # Force the `_IS_WINDOWS` branch so fast_leiden calls _run_leiden_direct
    # (line 399) instead of forking. Drive a real tiny neighbors graph through
    # the whole wrapper and assert a valid two-cluster partition is written.
    sc = pytest.importorskip("scanpy")
    ad = pytest.importorskip("anndata")
    pytest.importorskip("igraph")

    monkeypatch.setattr(LD, "_IS_WINDOWS", True)

    rng = np.random.default_rng(0)
    a_pts = rng.normal(0, 0.2, size=(15, 5))
    b_pts = rng.normal(9, 0.2, size=(15, 5))
    X = np.vstack([a_pts, b_pts]).astype(np.float32)
    a = ad.AnnData(X)
    import autozyme
    with autozyme.disabled():
        sc.pp.neighbors(a, n_neighbors=8, use_rep="X")

    out = LD.fast_leiden(a, flavor="igraph", n_iterations=2, directed=False,
                         random_state=0)
    assert out is None
    labels = a.obs["leiden"]
    assert labels.nunique() >= 2
    # The two well-separated blobs never share a cluster.
    assert set(labels.values[:15]).isdisjoint(set(labels.values[15:]))


# ==========================================================================
# _run_leiden_in_fork — the unlink OSError swallow (lines 330-331)
# ==========================================================================

@pytest.mark.skipif(LD._IS_WINDOWS, reason="fork path is Unix-only")
def test_fork_unlink_oserror_is_swallowed(monkeypatch):
    # If os.unlink on the shm temp file raises OSError, the finally-block
    # swallows it (lines 330-331) and the (successful) membership is still
    # returned. We let the real fork run, but make the parent's unlink raise.
    pytest.importorskip("igraph")
    g = LD._build_simple_graph(_block_adjacency(), use_weights=True)

    real_unlink = LD.os.unlink
    state = {"raised": False}

    def flaky_unlink(path):
        # Raise once (the parent's cleanup), then delegate so the file is
        # actually removed and nothing leaks.
        if not state["raised"]:
            state["raised"] = True
            try:
                real_unlink(path)
            except OSError:
                pass
            raise OSError("simulated unlink failure")
        return real_unlink(path)

    monkeypatch.setattr(LD.os, "unlink", flaky_unlink)
    membership = LD._run_leiden_in_fork(
        g, g.vcount(), random_state=0,
        clustering_args={
            "objective_function": "modularity",
            "n_iterations": 2,
            "weights": "weight",
            "resolution": 1.0,
        },
    )
    assert state["raised"] is True
    assert membership.shape == (6,)
    assert len(set(membership.tolist())) == 2
