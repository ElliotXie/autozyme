"""Fast Scanpy neighbors through vendored kNN + UMAP graph-prep kernels.

Supported fast path is intentionally narrow and mirrors the full e2e candidate:
large-data PCA embeddings, ``method='umap'``, ``knn=True``, Euclidean metric.
Unsupported states and missing native libraries fall back to upstream Scanpy.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp

_NATIVE = None
_NATIVE_TRIED = False
_NATIVE_LOAD_KEY = None


def _orig_neighbors():
    import scanpy as sc
    return getattr(sc.pp.neighbors, "__autozyme_original__", sc.pp.neighbors)


def _env_enabled() -> bool:
    value = os.environ.get("AUTOZYME_SCBLAS_NEIGHBORS", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _n_threads() -> int:
    for var in (
        "AUTOZYME_SCBLAS_NEIGHBORS_THREADS",
        "ZYME_THREADS",
        "AUTOZYME_THREADS",
        "OMP_NUM_THREADS",
    ):
        raw = os.environ.get(var, "")
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return os.cpu_count() or 1


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def _load_native_neighbors():
    global _NATIVE, _NATIVE_TRIED, _NATIVE_LOAD_KEY

    if not _env_enabled():
        return None
    load_key = (os.environ.get("AUTOZYME_SCBLAS_NEIGHBORS"),)
    if _NATIVE_TRIED and _NATIVE_LOAD_KEY == load_key:
        return _NATIVE

    _NATIVE_TRIED = True
    _NATIVE_LOAD_KEY = load_key
    _NATIVE = None
    try:
        from autozyme import _native_scanpy as native
    except Exception:
        return None
    _NATIVE = native
    return native


def _load_scblas_neighbors():
    """Compatibility alias for older smoke/debug snippets."""
    return _load_native_neighbors()
    return None


def _choose_x(adata, use_rep, n_pcs):
    if use_rep not in (None, "X_pca"):
        return None
    if "X_pca" not in adata.obsm:
        return None
    from scanpy.tools._utils import _choose_representation

    x = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
    if sp.issparse(x):
        x = x.toarray()
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] > 200:
        return None
    return np.ascontiguousarray(x, dtype=np.float32)


def _sort_knn_rows(idx: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(dist, axis=1, kind="stable")
    rows = np.arange(idx.shape[0])[:, None]
    return idx[rows, order], dist[rows, order]


def _native_knn(native, x, n_neighbors, seed, threads):
    n, d = x.shape
    k = n_neighbors - 1
    idx = np.empty((n, k), dtype=np.int32)
    dist2 = np.empty((n, k), dtype=np.float32)
    rc = int(
        native.knn_descent_f32(
            x,
            idx,
            dist2,
            int(n),
            int(d),
            int(k),
            _int_env("AUTOZYME_SCBLAS_NEIGHBORS_TREES", 24),
            _int_env("AUTOZYME_SCBLAS_NEIGHBORS_ITERS", 25),
            int(seed),
            _int_env("AUTOZYME_SCBLAS_NEIGHBORS_LEAF_SIZE", 30),
            int(threads),
        )
    )
    if rc != 0:
        return None
    dist = np.sqrt(np.maximum(dist2, np.float32(0.0))).astype(np.float32)
    idx, dist = _sort_knn_rows(idx, dist)
    self_idx = np.arange(n, dtype=np.int32)[:, None]
    dense_idx = np.ascontiguousarray(np.concatenate([self_idx, idx], axis=1), dtype=np.int32)
    dense_dist = np.ascontiguousarray(
        np.concatenate([np.zeros((n, 1), dtype=np.float32), dist], axis=1),
        dtype=np.float32,
    )
    return dense_idx, dense_dist


def _sparse_distances(knn_idx, knn_dist):
    n, k = knn_idx.shape
    rows = np.repeat(np.arange(n, dtype=np.int32), k - 1)
    cols = np.ascontiguousarray(knn_idx[:, 1:].reshape(-1), dtype=np.int32)
    vals = np.ascontiguousarray(knn_dist[:, 1:].reshape(-1), dtype=np.float32)
    mat = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    mat.sort_indices()
    return mat


def _native_connectivities(native, knn_idx, knn_dist, threads):
    idx = np.ascontiguousarray(knn_idx, dtype=np.int32)
    dists = np.ascontiguousarray(knn_dist, dtype=np.float32)
    n_samples, n_neighbors = dists.shape
    flat_n = n_samples * n_neighbors
    sym_rows = np.zeros(flat_n * 2, dtype=np.int32)
    sym_cols = np.zeros(flat_n * 2, dtype=np.int32)
    sym_vals = np.zeros(flat_n * 2, dtype=np.float32)

    native.umap_graph_f32(
        idx,
        dists,
        sym_rows,
        sym_cols,
        sym_vals,
        int(n_samples),
        int(n_neighbors),
    )

    graph = sp.coo_matrix((sym_vals, (sym_rows, sym_cols)), shape=(n_samples, n_samples), dtype=np.float32)
    graph.eliminate_zeros()
    graph = graph.tocsr()
    graph.sort_indices()
    return graph


def _write_neighbors(
    adata,
    distances,
    connectivities,
    n_neighbors,
    random_state,
    key_added,
    use_rep,
    n_pcs,
):
    if key_added is None:
        neighbors_key = "neighbors"
        distances_key = "distances"
        connectivities_key = "connectivities"
    else:
        neighbors_key = key_added
        distances_key = f"{key_added}_distances"
        connectivities_key = f"{key_added}_connectivities"

    adata.obsp[distances_key] = distances
    adata.obsp[connectivities_key] = connectivities
    params = {
        "n_neighbors": int(n_neighbors),
        "method": "umap",
        "random_state": int(random_state),
        "metric": "euclidean",
    }
    if use_rep is not None:
        params["use_rep"] = use_rep
    elif n_pcs is not None:
        params["n_pcs"] = int(n_pcs)

    adata.uns[neighbors_key] = {
        "connectivities_key": connectivities_key,
        "distances_key": distances_key,
        "params": params,
    }


def fast_neighbors(
    adata,
    n_neighbors=15,
    n_pcs=None,
    *posargs,
    use_rep=None,
    knn=True,
    method="umap",
    transformer=None,
    metric="euclidean",
    metric_kwds=None,
    random_state=0,
    key_added=None,
    copy=False,
    zyme=True,
):
    original = _orig_neighbors()
    fallback_kwargs = dict(
        use_rep=use_rep,
        knn=knn,
        method=method,
        transformer=transformer,
        metric=metric,
        metric_kwds={} if metric_kwds is None else metric_kwds,
        random_state=random_state,
        key_added=key_added,
        copy=copy,
    )
    if posargs:
        return original(adata, n_neighbors, n_pcs, *posargs, **fallback_kwargs)

    def fallback():
        return original(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, **fallback_kwargs)

    if (
        not zyme
        or not knn
        or method != "umap"
        or transformer is not None
        or metric != "euclidean"
        or (metric_kwds not in (None, {}))
        or not isinstance(random_state, (int, np.integer))
        or n_neighbors < 2
        or n_neighbors >= adata.n_obs
    ):
        return fallback()

    native = _load_native_neighbors()
    if native is None:
        return fallback()

    x = _choose_x(adata, use_rep, n_pcs)
    if x is None:
        return fallback()

    target = adata.copy() if copy else adata
    try:
        threads = _n_threads()
        knn_result = _native_knn(native, x, int(n_neighbors), int(random_state), threads)
        if knn_result is None:
            return fallback()
        knn_idx, knn_dist = knn_result
        distances = _sparse_distances(knn_idx, knn_dist)
        connectivities = _native_connectivities(native, knn_idx, knn_dist, threads)
        _write_neighbors(
            target,
            distances,
            connectivities,
            int(n_neighbors),
            int(random_state),
            key_added,
            use_rep,
            n_pcs,
        )
    except Exception:
        return fallback()

    return target if copy else None
