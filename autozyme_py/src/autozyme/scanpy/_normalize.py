"""Fast normalize_total + log1p — parallel numba CSR kernels.

Vendored from scanpy-turbo/_turbo/normalize.py, with API-correct semantics:
``fast_normalize_total`` performs library-size normalization only (linear scale);
``fast_log1p`` applies log1p in a separate pass.

Architecture:

  ``_row_sums``            per-row sum (parallel prange).
  ``_scale_only``          in-place row scaling using precomputed sums.
  ``_fused_normalize_only`` combined sum + scale on the explicit-target path.
  ``_log1p_inplace``       parallel log1p on CSR ``.data``.

The standard tutorial pair ``normalize_total → log1p`` therefore touches
``.data`` twice (scale, then log). The old fused scale+log1p-in-normalize
design saved one pass but violated the ``normalize_total`` API when log1p
was not called.
"""

from __future__ import annotations

import os
import warnings

import numba as nb
import numpy as np
from scipy import sparse

from autozyme._threads import auto_threads, safe_set_num_threads

_INT32_MAX = 2**31 - 1
_DTYPE_WARNED = False


@nb.njit(
    "void(float32[::1], int32[::1], float32[::1])",
    parallel=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def _row_sums(data, indptr, sums):
    """Per-row sum into ``sums`` (length n_rows). Parallel over rows."""
    n_rows = indptr.shape[0] - 1
    for i in nb.prange(n_rows):
        rs = indptr[i]
        re = indptr[i + 1]
        s = np.float32(0.0)
        for j in range(rs, re):
            s += data[j]
        sums[i] = s


@nb.njit(
    "void(float32[::1], int32[::1], float32[::1], float32)",
    parallel=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def _scale_only(data, indptr, sums, target_sum):
    """In-place row scaling to ``target_sum`` using precomputed per-row sums."""
    n_rows = indptr.shape[0] - 1
    for i in nb.prange(n_rows):
        s = sums[i]
        if s == np.float32(0.0):
            continue
        rs = indptr[i]
        re = indptr[i + 1]
        scale = np.float32(target_sum) / s
        for j in range(rs, re):
            data[j] *= scale


@nb.njit(
    "void(float32[::1], int32[::1], float32)",
    parallel=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def _fused_normalize_only(data, indptr, target_sum):
    """Combined per-row sum + scale (no log1p). Explicit-target path."""
    n_rows = indptr.shape[0] - 1
    for i in nb.prange(n_rows):
        rs = indptr[i]
        re = indptr[i + 1]
        s = np.float32(0.0)
        for j in range(rs, re):
            s += data[j]
        if s == np.float32(0.0):
            continue
        scale = np.float32(target_sum) / s
        for j in range(rs, re):
            data[j] *= scale


@nb.njit(
    "void(float32[::1])",
    parallel=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def _log1p_inplace(data):
    """Parallel in-place log1p on CSR nonzero values."""
    for j in nb.prange(data.shape[0]):
        data[j] = np.log1p(data[j])


# Warmup so the first user call doesn't pay JIT cost.
_wd = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
_wp = np.array([0, 2, 4], dtype=np.int32)
_ws = np.zeros(2, dtype=np.float32)
_row_sums(_wd, _wp, _ws)
_scale_only(_wd.copy(), _wp, _ws, np.float32(1e4))
_fused_normalize_only(_wd.copy(), _wp, np.float32(1e4))
_log1p_inplace(_wd.copy())
del _wd, _wp, _ws


def _n_threads():
    """Resolve thread count at call time so ``set_threads()`` takes effect.

    Honors attest's per-run thread budget (``ZYME_THREADS`` /
    ``AUTOZYME_THREADS`` / ``OMP_NUM_THREADS``) before falling back to the
    legacy ``SCANPY_TURBO_THREADS`` knob and finally the hardware default.
    Without this, the patched path silently runs full cpu_count() under
    ``zyme attest --threads N``, making the T=1/T=4/T=8 columns unfair.
    Malformed values fall through to the next source.
    """
    for var in ("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
                "SCANPY_TURBO_THREADS"):
        raw = os.environ.get(var, "")
        if not raw:
            continue
        try:
            n = int(raw)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    # No explicit knob set: defer to the unified resolver (scale-to-hardware,
    # capped at 16) rather than raw cpu_count(). scanpy's numba family scales
    # past 4 threads (finalized t4 is 2-4x slower than best), so default=None.
    return auto_threads(default=None)


def _orig(name: str):
    """Fetch upstream original via the autozyme dispatcher attribute."""
    import scanpy as sc
    fn = getattr(sc.pp, name)
    return getattr(fn, "__autozyme_original__", fn)


def _maybe_warn_about_dtype(X):
    """Emit one RuntimeWarning per session when the wrapper is forced to cast."""
    global _DTYPE_WARNED
    if _DTYPE_WARNED:
        return
    needs_cast = X.dtype != np.float32
    if not needs_cast and sparse.issparse(X):
        if X.indptr.dtype != np.int32 or X.indices.dtype != np.int32:
            needs_cast = X.nnz <= _INT32_MAX
    if not needs_cast:
        return
    indptr_dt = X.indptr.dtype if sparse.issparse(X) else None
    warnings.warn(
        f"autozyme.scanpy: input AnnData uses non-canonical dtypes "
        f"(data={X.dtype}"
        + (f", indptr={indptr_dt}" if indptr_dt is not None else "")
        + "). Casting once to float32 + int32; on large data this can take "
        "several hundred ms and the cost is charged to this function. "
        "Call autozyme.scanpy.zyme_prepare(adata) once after loading to do "
        "the cast upfront and time individual steps cleanly.",
        RuntimeWarning,
        stacklevel=3,
    )
    _DTYPE_WARNED = True


def _ensure_csr_float32(X):
    """Coerce ``X`` to float32 CSR with int32 indices when possible."""
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    elif not isinstance(X, sparse.csr_matrix):
        X = X.tocsr()
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    if X.nnz <= _INT32_MAX:
        if X.indptr.dtype != np.int32:
            X.indptr = X.indptr.astype(np.int32)
        if X.indices.dtype != np.int32:
            X.indices = X.indices.astype(np.int32)
    return X


def fast_normalize_total(adata, target_sum=None, *, copy=False, inplace=True,
                         zyme=True, **kwargs):
    """Parallel ``normalize_total`` — library-size scaling only (no log1p).

    Honors scanpy's return contract: ``copy=True`` returns the modified
    AnnData; the default (``inplace=True, copy=False``) mutates and returns
    ``None``. ``inplace=False`` (returns a dict) is delegated to upstream.
    Pass ``zyme=False`` to fall back to upstream ``sc.pp.normalize_total``.
    """
    if not zyme or not inplace or kwargs:
        return _orig("normalize_total")(
            adata, target_sum=target_sum, copy=copy, inplace=inplace, **kwargs
        )

    if copy:
        adata = adata.copy()

    X = adata.X
    _maybe_warn_about_dtype(X)
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
        adata.X = X
    elif not isinstance(X, sparse.csr_matrix):
        X = X.tocsr()
        adata.X = X

    if X.nnz > _INT32_MAX:
        # Upstream handles the int64-overflow branch. We've already copied
        # adata above when copy=True, so call upstream in-place and return
        # the local handle to preserve the contract.
        _orig("normalize_total")(
            adata, target_sum=target_sum, copy=False, inplace=True, **kwargs
        )
        return adata if copy else None

    X = _ensure_csr_float32(X)
    adata.X = X

    safe_set_num_threads(max(1, _n_threads()))

    if target_sum is None:
        n_rows = X.shape[0]
        sums = np.empty(n_rows, dtype=np.float32)
        _row_sums(X.data, X.indptr, sums)
        nz = sums[sums > 0]
        target_sum_v = float(np.median(nz)) if nz.size else 1e4
        _scale_only(X.data, X.indptr, sums, np.float32(target_sum_v))
    else:
        _fused_normalize_only(X.data, X.indptr, np.float32(target_sum))

    return adata if copy else None


def fast_log1p(adata, *, copy=False, zyme=True, **kwargs):
    """Parallel in-place ``log1p`` on sparse CSR ``.data``.

    Honors scanpy's return contract: ``copy=True`` returns the modified
    AnnData; the default mutates and returns ``None``.
    Pass ``zyme=False`` to fall back to upstream ``sc.pp.log1p``.
    """
    if not zyme or kwargs:
        return _orig("log1p")(adata, copy=copy, **kwargs)

    if copy:
        adata = adata.copy()

    X = adata.X
    if sparse.issparse(X):
        if not isinstance(X, sparse.csr_matrix):
            X = X.tocsr()
            adata.X = X
        if X.dtype != np.float32:
            X = X.astype(np.float32)
            adata.X = X
        safe_set_num_threads(max(1, _n_threads()))
        _log1p_inplace(X.data)
    else:
        np.log1p(X, out=X)

    # Match upstream's output contract: scanpy's log1p records this so downstream
    # code (and a second log1p call) can detect the data is already log-scaled.
    # The fast path only runs for the default base=None call (any kwarg falls
    # back above), so the recorded base is always None here.
    adata.uns["log1p"] = {"base": None}

    return adata if copy else None
