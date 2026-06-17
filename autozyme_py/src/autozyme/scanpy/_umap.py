"""Fast Scanpy UMAP layout through vendored native kernels.

This patch accelerates the ``sc.tl.umap`` layout call by temporarily replacing
umap-learn's ``simplicial_set_embedding`` with the deterministic native layout
optimizer vendored into autozyme for the duration of one upstream Scanpy call.

The native extension is optional. If it is unavailable, or if the call requests
an unsupported UMAP mode, execution falls back to upstream Scanpy without
changing user-visible behavior.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any

import numpy as np

from autozyme._threads import auto_threads

_NATIVE = None
_NATIVE_TRIED = False
_NATIVE_LOAD_KEY = None


class _FallbackToOriginal(Exception):
    pass


def _orig_umap():
    import scanpy as sc
    return getattr(sc.tl.umap, "__autozyme_original__", sc.tl.umap)


def _env_enabled() -> bool:
    value = os.environ.get("AUTOZYME_SCBLAS_UMAP", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _n_threads() -> int:
    for var in (
        "AUTOZYME_SCBLAS_UMAP_THREADS",
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
    # No explicit knob set: defer to the unified resolver (scale-to-hardware,
    # capped at 16) rather than raw cpu_count().
    return auto_threads(default=None)


def _load_native_umap():
    global _NATIVE, _NATIVE_TRIED, _NATIVE_LOAD_KEY

    if not _env_enabled():
        return None
    load_key = (os.environ.get("AUTOZYME_SCBLAS_UMAP"),)
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


def _load_scblas_umap():
    """Compatibility alias for older smoke/debug snippets."""
    return _load_native_umap()


class _NativeSimplicialSetEmbedding:
    def __init__(self, native, threads: int):
        self.native = native
        self.threads = int(threads)
        self.last_call: dict[str, Any] | None = None

    def __call__(
        self,
        data,
        graph,
        n_components,
        initial_alpha,
        a,
        b,
        gamma,
        negative_sample_rate,
        n_epochs,
        init,
        random_state,
        metric,
        metric_kwds,
        densmap,
        densmap_kwds,
        output_dens,
        output_metric=None,
        output_metric_kwds=None,
        euclidean_output=True,
        parallel=False,
        verbose=False,
        tqdm_kwds=None,
    ):
        if densmap or output_dens or not euclidean_output:
            raise _FallbackToOriginal
        if isinstance(n_epochs, list):
            raise _FallbackToOriginal

        import scipy.sparse
        from sklearn.decomposition import PCA, TruncatedSVD
        from sklearn.neighbors import KDTree
        import umap.umap_ as uu

        t_total = time.perf_counter()
        graph = graph.tocoo(copy=True)
        graph.sum_duplicates()
        default_epochs = 500 if graph.shape[0] <= 10000 else 200
        n_epochs = default_epochs if n_epochs is None else int(n_epochs)

        if graph.nnz > 0:
            denom = n_epochs if n_epochs > 10 else default_epochs
            graph.data[graph.data < (graph.data.max() / float(denom))] = 0.0
            graph.eliminate_zeros()

        if graph.nnz <= 0:
            raise _FallbackToOriginal

        embedding = _initial_embedding(
            data=data,
            graph=graph,
            n_components=int(n_components),
            init=init,
            random_state=random_state,
            metric=metric,
            metric_kwds=metric_kwds,
            pca_cls=PCA,
            svd_cls=TruncatedSVD,
            kdtree_cls=KDTree,
            uu=uu,
            sparse_mod=scipy.sparse,
        )
        embedding = _rescale_embedding(embedding)

        head = np.ascontiguousarray(graph.row, dtype=np.int32)
        tail = np.ascontiguousarray(graph.col, dtype=np.int32)
        epochs_per_sample = np.ascontiguousarray(
            uu.make_epochs_per_sample(graph.data, n_epochs).astype(np.float32)
        )
        emb = np.ascontiguousarray(embedding.copy(), dtype=np.float32)

        try:
            seed = int(random_state.randint(0, np.iinfo(np.uint32).max))
        except Exception:
            seed = 0

        t_layout = time.perf_counter()
        rc = int(
            self.native.umap_layout_euclidean_f32(
                emb,
                head,
                tail,
                epochs_per_sample,
                int(emb.shape[0]),
                int(emb.shape[1]),
                int(n_epochs),
                float(a),
                float(b),
                float(gamma),
                float(initial_alpha),
                float(negative_sample_rate),
                int(seed),
                int(self.threads),
            )
        )
        self.last_call = {
            "total_s": time.perf_counter() - t_total,
            "layout_s": time.perf_counter() - t_layout,
            "return_code": rc,
            "n_edges": int(head.size),
            "n_epochs": int(n_epochs),
            "threads": int(self.threads),
            "parallel": bool(rc == 0),
        }
        return emb, {}


def _initial_embedding(
    *,
    data,
    graph,
    n_components,
    init,
    random_state,
    metric,
    metric_kwds,
    pca_cls,
    svd_cls,
    kdtree_cls,
    uu,
    sparse_mod,
):
    if isinstance(init, str) and init == "random":
        return random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)

    if isinstance(init, str) and init == "pca":
        pca = svd_cls(n_components=n_components, random_state=random_state)
        if not sparse_mod.issparse(data):
            pca = pca_cls(n_components=n_components, random_state=random_state)
        embedding = pca.fit_transform(data).astype(np.float32)
        return uu.noisy_scale_coords(
            embedding, random_state, max_coord=10, noise=0.0001
        )

    if isinstance(init, str) and init == "spectral":
        embedding = uu.spectral_layout(
            data, graph, n_components, random_state,
            metric=metric, metric_kwds=metric_kwds,
        )
        return uu.noisy_scale_coords(
            embedding, random_state, max_coord=10, noise=0.0001
        )

    if isinstance(init, str) and init == "tswspectral":
        embedding = uu.tswspectral_layout(
            data, graph, n_components, random_state,
            metric=metric, metric_kwds=metric_kwds,
        )
        return uu.noisy_scale_coords(
            embedding, random_state, max_coord=10, noise=0.0001
        )

    init_data = np.array(init)
    if len(init_data.shape) != 2:
        raise _FallbackToOriginal
    if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
        tree = kdtree_cls(init_data)
        dist, _ = tree.query(init_data, k=2)
        nndist = np.mean(dist[:, 1])
        return init_data + random_state.normal(
            scale=0.001 * nndist, size=init_data.shape
        ).astype(np.float32)
    return init_data


def _rescale_embedding(embedding):
    embedding = np.asarray(embedding, dtype=np.float32)
    lo = np.min(embedding, 0)
    span = np.max(embedding, 0) - lo
    span[span == 0.0] = 1.0
    return (10.0 * (embedding - lo) / span).astype(np.float32, order="C")


@contextmanager
def _patched_umap_layout(wrapper):
    import umap.umap_ as uu

    original = uu.simplicial_set_embedding
    uu.simplicial_set_embedding = wrapper
    try:
        yield
    finally:
        uu.simplicial_set_embedding = original


def fast_umap(
    adata,
    *,
    min_dist=0.5,
    spread=1.0,
    n_components=2,
    maxiter=None,
    alpha=1.0,
    gamma=1.0,
    negative_sample_rate=5,
    init_pos="spectral",
    random_state=0,
    a=None,
    b=None,
    method="umap",
    key_added=None,
    neighbors_key="neighbors",
    copy=False,
    zyme=True,
):
    original = _orig_umap()
    call = dict(
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        maxiter=maxiter,
        alpha=alpha,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        init_pos=init_pos,
        random_state=random_state,
        a=a,
        b=b,
        method=method,
        key_added=key_added,
        neighbors_key=neighbors_key,
        copy=copy,
    )
    if not zyme or method != "umap":
        return original(adata, **call)

    native = _load_native_umap()
    if native is None:
        return original(adata, **call)

    wrapper = _NativeSimplicialSetEmbedding(native, _n_threads())
    try:
        with _patched_umap_layout(wrapper):
            return original(adata, **call)
    except Exception:
        # Any failure in the native layout path -- the explicit
        # _FallbackToOriginal signal or an unexpected error from the C kernel --
        # reverts to the stock uwot/scanpy layout instead of surfacing the error.
        # _patched_umap_layout's finally has already restored the original layout
        # function, so this re-run is the unmodified baseline.
        return original(adata, **call)
