"""One-shot dtype coercion for the autozyme scanpy plugin.

The accelerated kernels expect ``float32`` ``.X`` with ``int32`` CSR
``indptr``/``indices``. AnnData objects from CZI Cell Census, Seurat→Scanpy
conversion, and other producers often arrive as ``float64`` with ``int64``
indices, which forces a per-call cast at the entrance of every fast
function — and that cast time gets attributed to whichever fast function
ran first (typically ``normalize_total``), making it look slow.

Calling ``zyme_prepare(adata)`` once after loading data does the cast
upfront so per-function timings reflect kernel work, not memory-bandwidth-
bound copies that would have happened anyway.

Vendored from scanpy-turbo/_turbo/_prepare.py (the canonical implementation).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

_INT32_MAX = 2**31 - 1


def zyme_prepare(adata, *, layers=True, copy=False):
    """Coerce ``adata`` to the dtypes the fast kernels expect.

    Casts ``.X`` to ``float32``; for sparse ``.X``, also casts ``indptr``
    and ``indices`` to ``int32`` when ``nnz`` fits. Optionally extends to
    ``adata.layers``.

    Parameters
    ----------
    adata
        AnnData object. Modified in place unless ``copy=True``.
    layers
        If True (default), also coerce every entry in ``adata.layers``.
    copy
        If True, return a converted deep copy instead of mutating ``adata``.

    Returns
    -------
    AnnData if ``copy=True``, else None.

    Examples
    --------
    >>> import autozyme, scanpy as sc
    >>> autozyme.activate("scanpy")
    >>> from autozyme.scanpy import zyme_prepare
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> zyme_prepare(adata)              # one-time coerce
    >>> sc.pp.normalize_total(adata)     # kernel-only timing
    """
    if copy:
        adata = adata.copy()

    _coerce_x(adata)
    if layers and adata.layers:
        for key in list(adata.layers.keys()):
            _coerce_layer(adata, key)

    if copy:
        return adata
    return None


def _coerce_x(adata) -> None:
    X = adata.X
    new_X = _coerce_matrix(X)
    if new_X is not X:
        adata.X = new_X


def _coerce_layer(adata, key: str) -> None:
    X = adata.layers[key]
    new_X = _coerce_matrix(X)
    if new_X is not X:
        adata.layers[key] = new_X


def _coerce_matrix(X):
    """Return X coerced to canonical fast-path dtypes (or X if already canonical)."""
    if sparse.issparse(X):
        if not isinstance(X, sparse.csr_matrix):
            X = X.tocsr()
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        # Index dtype only fits int32 when nnz < 2^31. Above that, keep
        # int64 — the wrapper will fall back to upstream scanpy.
        if X.nnz <= _INT32_MAX:
            if X.indptr.dtype != np.int32:
                X.indptr = X.indptr.astype(np.int32)
            if X.indices.dtype != np.int32:
                X.indices = X.indices.astype(np.int32)
        return X
    if X.dtype != np.float32:
        return X.astype(np.float32)
    return X
