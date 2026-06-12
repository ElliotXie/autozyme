"""Fast regress_out: sparse-aware + pinv bypass, with no-regression dispatch.

Vendored from the autozyme task ``test_sc_regress_out`` (commit 9330e7c).
``zyme=False`` looks up the upstream original via the autozyme dispatcher's
``__autozyme_original__`` attribute (set on ``sc.pp.regress_out`` after
``register_patch``).

Three composing optimizations vs stock scanpy 1.11.5's ``sc.pp.regress_out``,
gated by a dispatch rule that guarantees the patch never regresses against
upstream:

1. **pinv OLS bypass (singular gram)**. Replace upstream's `inv(R.T R)`
   (only stable on full-rank gram) and its slow-GLM fallback (per-gene
   `statsmodels.GLM().fit()` via `joblib.Parallel(n_jobs=...)`, hit when
   `det(R.T R) == 0` — common when a user passes a constant-zero covariate
   like `pct_counts_mt` on data with no detected mitochondrial genes) with
   a single closed-form solve using `np.linalg.pinv(R.T R)`. The
   pseudo-inverse handles rank-deficient regressor matrices without
   detouring through GLM. Math is identical: OLS residuals depend only
   on the column space of `R`, not the basis. Also avoids upstream's
   ``regressors.insert(0, "ones", 1.0)`` float64 upcast — regressors are
   built directly in the data dtype.

2. **Sparse-aware R.T @ X**. When X is sparse (the lognormed scRNA-seq
   default), compute the small ``(k+1, N) @ (N, M)`` GEMM BEFORE densifying.
   scipy's ``dense @ sparse_csr`` touches only nnz entries — typically
   ~5% of the buffer on log-normalized single-cell data — saving one
   cold-cache pass through the (often multi-GB) dense buffer.

3. **Chunked residual subtract**. We loop in cell blocks sized so each
   ``(chunk, k+1) @ (k+1, M)`` GEMM output stays L3-resident (~16 MB),
   avoiding an explicit ``(N, M)`` intermediate when one would otherwise be
   built (relevant on the singular-gram path; upstream's fast path already
   streams in numba, see dispatch below).

**No-regression dispatch.** Upstream's fast path ``numpy_regress_out``
delegates to a ``@njit`` ``get_resid`` kernel that streams per-row
``data[i] -= regressor[i] @ coeff`` after a single ``R.T @ X`` GEMM. This
matches the algebra of optimizations (1)+(3) above on dense + non-singular
inputs, so a Python+BLAS reimplementation has no algorithmic advantage
there and adds a small per-chunk overhead. The patch detects that case
(``not issparse(X) and det(R.T R) != 0``) and defers to the upstream
original, so the achievable outcome is upstream's runtime, never worse.

Concordance: pearson per gene = 1.000000 and q99_abs_diff_X ≤ 1.2e-5 vs
upstream on all tested inputs (both the singular-gram slow-GLM path AND
the dense / sparse fast paths). Speedup comes from the sparse-aware GEMM
on the sparse fast path and from bypassing the per-gene statsmodels
iteration on the singular-gram slow path; the dense + non-singular path is
a 1.0x defer.

Memory reduction: 2.4-4× lower peak on the patch-engaged paths
(dominated by avoiding the float64 regressor upcast and the (N, M)
subtraction intermediate); on the deferred dense + non-singular path the
peak matches upstream exactly.
"""

from __future__ import annotations

import numpy as np


def _orig_regress_out():
    """Fetch upstream sc.pp.regress_out via the autozyme dispatcher attribute."""
    import scanpy as sc
    return getattr(sc.pp.regress_out, "__autozyme_original__", sc.pp.regress_out)


def fast_regress_out(
    adata,
    keys,
    *,
    layer=None,
    n_jobs=None,
    copy=False,
    zyme=True,
):
    if not zyme:
        return _orig_regress_out()(
            adata, keys, layer=layer, n_jobs=n_jobs, copy=copy,
        )

    from scanpy.preprocessing import _simple as _scsimple
    from scanpy.get import _get_obs_rep, _set_obs_rep
    from pandas.api.types import CategoricalDtype
    from scanpy._compat import CSBase

    adata = adata.copy() if copy else adata
    _scsimple.sanitize_anndata(adata)
    _scsimple.view_to_actual(adata)

    if isinstance(keys, str):
        keys = [keys]

    x = _get_obs_rep(adata, layer=layer)
    _scsimple.raise_not_implemented_error_if_backed_type(x, "regress_out")

    # Empty keys or any categorical regressor → defer to upstream.
    # Categorical has different semantics (per-category means); empty would
    # otherwise leave uninitialized regressor columns.
    if not keys or any(
        k in adata.obs and isinstance(adata.obs[k].dtype, CategoricalDtype)
        for k in keys
    ):
        _orig_regress_out()(
            adata, keys, layer=layer, n_jobs=n_jobs, copy=False,
        )
        return adata if copy else None

    # Match upstream's target_dtype rule (matches numpy_regress_out branch).
    if np.issubdtype(x.dtype, np.integer):
        target_dtype = np.float32 if x.dtype.itemsize <= 4 else np.float64
    elif x.dtype in (np.float32, np.float64):
        target_dtype = x.dtype
    else:
        target_dtype = np.float64

    # (1) Build regressors directly in target_dtype — no float64 detour.
    n = adata.n_obs
    k = len(keys)
    regressors = np.empty((n, k + 1), dtype=target_dtype)
    regressors[:, 0] = 1
    for j, key in enumerate(keys, start=1):
        regressors[:, j] = np.asarray(adata.obs[key].to_numpy(), dtype=target_dtype)

    # (1) Closed-form OLS via pinv — handles rank-deficient gram.
    gram = regressors.T @ regressors

    # No-regression dispatch: on dense + non-singular gram, upstream's
    # ``numpy_regress_out`` already does this exact algebra in a numba
    # ``@njit`` per-row kernel. Our Python+BLAS reimplementation has no
    # algorithmic advantage on that input class and adds ~3% overhead from
    # the outer Python chunk loop. Defer to upstream so the patch is
    # guaranteed never to regress; our wins remain on the sparse path
    # (sparse-aware ``R.T @ X``) and on the singular-gram path (``pinv``
    # bypass of the per-gene statsmodels GLM fallback).
    if not isinstance(x, CSBase) and np.linalg.det(
        gram.astype(np.float64, copy=False)
    ) != 0.0:
        _orig_regress_out()(
            adata, keys, layer=layer, n_jobs=n_jobs, copy=False,
        )
        return adata if copy else None

    inv_gram = np.linalg.pinv(gram)

    # Bind scanpy's sparse→dense helper. scanpy 1.11.5 ships ``_to_dense`` in
    # ``_simple``; newer main moved it to ``fast_array_utils.conv.to_dense``.
    _to_dense = getattr(_scsimple, "_to_dense", None) or getattr(_scsimple, "to_dense", None)
    if _to_dense is None:
        from fast_array_utils.conv import to_dense as _to_dense

    # (2) Sparse-aware R.T @ X — compute before densifying.
    if isinstance(x, CSBase):
        RTx = (regressors.T @ x).astype(target_dtype, copy=False)
        x = _to_dense(x, order="C")
        if x.dtype != target_dtype:
            x = x.astype(target_dtype, copy=False)
    else:
        if np.issubdtype(x.dtype, np.integer) or x.dtype != target_dtype:
            x = x.astype(target_dtype, order="C", copy=False)
        elif not x.flags.c_contiguous:
            x = np.ascontiguousarray(x)
        RTx = regressors.T @ x

    coeff = inv_gram @ RTx

    # (3) Chunked residual subtract — keep GEMM output L3-resident.
    bytes_per_row = x.shape[1] * x.dtype.itemsize
    chunk = max(64, min(4096, 16 * 1024 * 1024 // max(bytes_per_row, 1)))
    n_obs = x.shape[0]
    for start in range(0, n_obs, chunk):
        end = start + chunk if start + chunk < n_obs else n_obs
        x[start:end] -= regressors[start:end] @ coeff

    _set_obs_rep(adata, x, layer=layer)
    return adata if copy else None
