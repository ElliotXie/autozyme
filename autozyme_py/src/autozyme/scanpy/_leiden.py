"""Fast Leiden: simple graph from upper triangle, no Python overhead.

Vendored from scanpy-turbo/_turbo/leiden.py. ``zyme=False`` looks up the
upstream original via the autozyme dispatcher's ``__autozyme_original__``
attribute.

Builds an undirected simple graph from the upper triangle of the adjacency
matrix with 2x weights (half the edges, same effective strength). On Unix,
runs ``community_leiden`` in a forked child process so tracemalloc hooks
don't slow down the C-level main loop. On Windows, runs direct.
"""

from __future__ import annotations

import ctypes
import gc
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

_IS_WINDOWS = platform.system() == "Windows"
_SCBLAS_LIB = None
_SCBLAS_TRIED = False
_SCBLAS_LOAD_KEY = None
_GRAPH_CACHE = {}


def _orig_leiden():
    """Fetch upstream sc.tl.leiden via the autozyme dispatcher attribute."""
    import scanpy as sc
    return getattr(sc.tl.leiden, "__autozyme_original__", sc.tl.leiden)


def _scblas_enabled():
    value = os.environ.get("AUTOZYME_SCBLAS_LEIDEN", "0").strip().lower()
    return value in {"1", "true", "on", "yes"}


def _scblas_threads():
    raw = os.environ.get("AUTOZYME_SCBLAS_LEIDEN_THREADS", "1")
    try:
        return int(raw)
    except ValueError:
        return 1


def _graph_cache_enabled():
    value = os.environ.get("AUTOZYME_SCBLAS_LEIDEN_CACHE", "0").strip().lower()
    return value in {"1", "true", "on", "yes"}


def _graph_cache_limit():
    raw = os.environ.get("AUTOZYME_SCBLAS_LEIDEN_CACHE_SIZE", "4")
    try:
        value = int(raw)
    except ValueError:
        return 4
    return max(1, value)


def _graph_cache_key(adjacency, use_weights):
    if not _graph_cache_enabled():
        return None
    try:
        shape = (int(adjacency.shape[0]), int(adjacency.shape[1]))
        nnz = int(adjacency.nnz)
    except Exception:
        return None
    return (
        id(adjacency),
        shape,
        nnz,
        bool(use_weights),
        os.environ.get("AUTOZYME_SCBLAS_LEIDEN"),
        os.environ.get("AUTOZYME_SCBLAS_LEIDEN_THREADS"),
        os.environ.get("SCBLAS_LIB"),
        os.environ.get("AUTOZYME_SCBLAS_LIB"),
    )


def _scblas_candidate_paths():
    env = os.environ.get("SCBLAS_LIB") or os.environ.get("AUTOZYME_SCBLAS_LIB")
    if env:
        yield Path(env)

    here = Path(__file__).resolve()
    for parent in here.parents:
        lib_dir = parent / "lib_scblas" / "build"
        if lib_dir.exists():
            for name in (
                "libscblas.0.1.0.dylib",
                "libscblas.dylib",
                "libscblas.so",
                "scblas.dll",
            ):
                yield lib_dir / name


def _load_scblas_leiden():
    """Load optional scBLAS Leiden extraction symbols.

    Returns None on any failure so the Scanpy patch can fall back to its
    previous SciPy-triu path without changing user-visible behavior.
    """
    global _SCBLAS_LIB, _SCBLAS_TRIED, _SCBLAS_LOAD_KEY

    if not _scblas_enabled():
        return None
    load_key = (
        os.environ.get("SCBLAS_LIB"),
        os.environ.get("AUTOZYME_SCBLAS_LIB"),
        os.environ.get("AUTOZYME_SCBLAS_LEIDEN"),
    )
    if _SCBLAS_TRIED and _SCBLAS_LOAD_KEY == load_key:
        return _SCBLAS_LIB

    _SCBLAS_TRIED = True
    _SCBLAS_LOAD_KEY = load_key
    _SCBLAS_LIB = None
    for path in _scblas_candidate_paths():
        if not path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
            c_int = ctypes.c_int
            c_i64_p = ctypes.POINTER(ctypes.c_int64)
            c_i32_p = ctypes.POINTER(ctypes.c_int32)
            c_f64_p = ctypes.POINTER(ctypes.c_double)

            lib.scblas_leiden_csr_upper_counts_i32.argtypes = [
                c_int, c_i64_p, c_i32_p, c_i64_p,
            ]
            lib.scblas_leiden_csr_upper_counts_i32.restype = ctypes.c_int64
            lib.scblas_leiden_csr_upper_counts_i32_parallel.argtypes = [
                c_int, c_i64_p, c_i32_p, c_i64_p, c_i64_p, c_int,
            ]
            lib.scblas_leiden_csr_upper_counts_i32_parallel.restype = c_int
            lib.scblas_leiden_csr_upper_edges_f64.argtypes = [
                c_int, c_i64_p, c_i32_p, c_f64_p, c_i64_p, c_int,
                c_i32_p, c_i32_p, c_f64_p,
            ]
            lib.scblas_leiden_csr_upper_edges_f64.restype = ctypes.c_int64
            lib.scblas_leiden_csr_upper_edges_f64_parallel.argtypes = [
                c_int, c_i64_p, c_i32_p, c_f64_p, c_i64_p, c_int,
                c_i32_p, c_i32_p, c_f64_p, c_i64_p, c_int,
            ]
            lib.scblas_leiden_csr_upper_edges_f64_parallel.restype = c_int
            _SCBLAS_LIB = lib
            return lib
        except Exception:
            _SCBLAS_LIB = None
    return None


def _ptr(arr, ctype):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


def _scblas_upper_edges(adjacency, include_weights=True):
    lib = _load_scblas_leiden()
    if lib is None or not sparse.issparse(adjacency):
        return None

    csr = adjacency.tocsr()
    csr.sort_indices()
    n = int(csr.shape[0])
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
    data = np.ascontiguousarray(csr.data, dtype=np.float64) if include_weights else None

    counts = np.zeros(n, dtype=np.int64)
    threads = _scblas_threads()
    if threads == 1:
        edge_count = int(
            lib.scblas_leiden_csr_upper_counts_i32(
                n,
                _ptr(indptr, ctypes.c_int64),
                _ptr(indices, ctypes.c_int32),
                _ptr(counts, ctypes.c_int64),
            )
        )
    else:
        total = ctypes.c_int64(-1)
        lib.scblas_leiden_csr_upper_counts_i32_parallel(
            n,
            _ptr(indptr, ctypes.c_int64),
            _ptr(indices, ctypes.c_int32),
            _ptr(counts, ctypes.c_int64),
            ctypes.byref(total),
            threads,
        )
        edge_count = int(total.value)
    if edge_count < 0:
        return None

    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    if int(offsets[-1]) != edge_count:
        return None

    sources = np.empty(edge_count, dtype=np.int32)
    targets = np.empty(edge_count, dtype=np.int32)
    weights = np.empty(edge_count, dtype=np.float64) if include_weights else None
    if threads == 1:
        written = int(
            lib.scblas_leiden_csr_upper_edges_f64(
                n,
                _ptr(indptr, ctypes.c_int64),
                _ptr(indices, ctypes.c_int32),
                _ptr(data, ctypes.c_double) if data is not None else None,
                _ptr(offsets, ctypes.c_int64),
                1,
                _ptr(sources, ctypes.c_int32),
                _ptr(targets, ctypes.c_int32),
                _ptr(weights, ctypes.c_double) if weights is not None else None,
            )
        )
    else:
        total = ctypes.c_int64(-1)
        lib.scblas_leiden_csr_upper_edges_f64_parallel(
            n,
            _ptr(indptr, ctypes.c_int64),
            _ptr(indices, ctypes.c_int32),
            _ptr(data, ctypes.c_double) if data is not None else None,
            _ptr(offsets, ctypes.c_int64),
            1,
            _ptr(sources, ctypes.c_int32),
            _ptr(targets, ctypes.c_int32),
            _ptr(weights, ctypes.c_double) if weights is not None else None,
            ctypes.byref(total),
            threads,
        )
        written = int(total.value)
    if written != edge_count:
        return None
    return sources, targets, weights


def _build_simple_graph(adjacency, use_weights=True):
    """Build simple igraph Graph from symmetric sparse adjacency."""
    import igraph as ig

    cache_key = _graph_cache_key(adjacency, use_weights)
    if cache_key is not None and cache_key in _GRAPH_CACHE:
        return _GRAPH_CACHE[cache_key]

    extracted = _scblas_upper_edges(adjacency, include_weights=use_weights)
    if extracted is None:
        upper = sparse.triu(adjacency, k=1).tocoo()
        sources = upper.row
        targets = upper.col
        weights = upper.data * 2 if use_weights else None
    else:
        sources, targets, weights = extracted

    g = ig.Graph(
        n=adjacency.shape[0],
        edges=list(zip(sources.tolist(), targets.tolist())),
        directed=False,
    )
    if use_weights and weights is not None:
        g.es["weight"] = weights.tolist()
    if cache_key is not None:
        while len(_GRAPH_CACHE) >= _graph_cache_limit():
            _GRAPH_CACHE.pop(next(iter(_GRAPH_CACHE)))
        _GRAPH_CACHE[cache_key] = g
    return g


def _run_leiden_direct(g, random_state, clustering_args):
    """Run Leiden directly in the current process."""
    from scanpy._utils.random import set_igraph_random_state
    with set_igraph_random_state(random_state):
        part = g.community_leiden(**clustering_args)
    return np.array(part.membership, dtype=np.int32)


def _run_leiden_in_fork(g, n_nodes, random_state, clustering_args):
    """Run community_leiden in a forked child process (Unix only)."""
    import mmap
    import tempfile
    import tracemalloc

    from scanpy._utils.random import set_igraph_random_state

    shm = tempfile.NamedTemporaryFile(delete=False)
    shm_path = shm.name
    shm.write(b"\x00" * (n_nodes * 4 + 4))
    shm.close()

    pid = os.fork()
    if pid == 0:
        try:
            tracemalloc.stop()
        except RuntimeError:
            pass
        try:
            with set_igraph_random_state(random_state):
                part = g.community_leiden(**clustering_args)
            membership = np.array(part.membership, dtype=np.int32)
            with open(shm_path, "r+b") as f:
                mm = mmap.mmap(f.fileno(), 0)
                mm[4 : 4 + n_nodes * 4] = membership.tobytes()
                mm[0:4] = b"DONE"
                mm.close()
            os._exit(0)
        except BaseException:
            # Don't write the DONE marker on any failure (raise, segfault
            # via os._exit, or KeyboardInterrupt). The parent's marker check
            # is the only signal that "all bytes are valid".
            os._exit(1)
    else:
        _, status = os.waitpid(pid, 0)
        try:
            with open(shm_path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                marker = bytes(mm[0:4])
                membership = np.frombuffer(
                    mm[4 : 4 + n_nodes * 4], dtype=np.int32,
                ).copy()
                mm.close()
        finally:
            try:
                os.unlink(shm_path)
            except OSError:
                pass
        if marker != b"DONE":
            # Child died before flushing membership. Returning the
            # pre-initialized zeros would silently put every node in cluster
            # 0; raise instead so the caller sees the failure.
            raise RuntimeError(
                f"leiden fork child exited (status={status}) without writing "
                f"completion marker; partition is unavailable"
            )
        return membership


def fast_leiden(
    adata, resolution=1.0, *, key_added="leiden", adjacency=None,
    use_weights=True, n_iterations=-1, random_state=0,
    neighbors_key=None, obsp=None, flavor="igraph",
    directed=False, copy=False, restrict_to=None,
    partition_type=None, zyme=True, **clustering_args,
):
    if not zyme:
        return _orig_leiden()(
            adata, resolution=resolution, key_added=key_added,
            adjacency=adjacency, use_weights=use_weights,
            n_iterations=n_iterations, random_state=random_state,
            neighbors_key=neighbors_key, obsp=obsp, flavor=flavor,
            directed=directed, copy=copy, restrict_to=restrict_to,
            partition_type=partition_type, **clustering_args,
        )

    from natsort import natsorted
    from scanpy import _utils as sc_utils

    adata_out = adata.copy() if copy else adata

    # Honor `flavor` exactly like baseline scanpy does. The igraph C
    # implementation and the leidenalg Python library are different
    # algorithms that produce noticeably different partitions on complex
    # graphs (ARI ~0.82 between the two on a 117k-cell / 41-cluster atlas).
    # n_iterations > 2 would be silently capped to 2 below; upstream igraph runs
    # the requested count, so defer to it rather than under-converge. n_iterations
    # < 0 (the "auto" default) and <= 2 stay on the fast path unchanged.
    if (flavor == "leidenalg" or restrict_to is not None
            or partition_type is not None or directed or n_iterations > 2):
        return _orig_leiden()(
            adata, resolution=resolution, key_added=key_added,
            adjacency=adjacency, use_weights=use_weights,
            n_iterations=n_iterations, random_state=random_state,
            neighbors_key=neighbors_key, obsp=obsp, flavor=flavor,
            directed=directed, copy=copy, restrict_to=restrict_to,
            partition_type=partition_type, **clustering_args,
        )

    if adjacency is None:
        adjacency = sc_utils._choose_graph(adata_out, obsp, neighbors_key)
    g = _build_simple_graph(adjacency, use_weights=use_weights)

    # Cap iterations at 2 (matches scanpy's recommended igraph default).
    effective_n_iterations = 2 if n_iterations < 0 else min(n_iterations, 2)
    clustering_args["n_iterations"] = effective_n_iterations
    if use_weights:
        clustering_args["weights"] = "weight"
    if resolution is not None:
        clustering_args["resolution"] = resolution
    clustering_args.setdefault("objective_function", "modularity")

    gc.disable()

    if _IS_WINDOWS:
        membership = _run_leiden_direct(g, random_state, clustering_args)
    else:
        membership = _run_leiden_in_fork(g, g.vcount(), random_state, clustering_args)

    gc.enable()

    adata_out.obs[key_added] = pd.Categorical(
        values=membership.astype("U"),
        categories=natsorted(map(str, np.unique(membership))),
    )
    adata_out.uns[key_added] = {}
    adata_out.uns[key_added]["params"] = dict(
        resolution=resolution, random_state=random_state, n_iterations=n_iterations,
    )
    return adata_out if copy else None
