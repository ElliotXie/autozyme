"""Fast scale: numba-accelerated fused scale + clip.

Vendored from scanpy-turbo/_turbo/scale.py. ``zyme=False`` looks up the
upstream original via the autozyme dispatcher's ``__autozyme_original__``
attribute (set on ``sc.pp.scale`` after register_patch).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _orig_scale():
    """Fetch upstream sc.pp.scale via the autozyme dispatcher attribute."""
    import scanpy as sc
    return getattr(sc.pp.scale, "__autozyme_original__", sc.pp.scale)


if HAS_NUMBA:
    @numba.njit(cache=True)
    def _accumulate_stats(indices, data, n_vars):
        sums = np.zeros(n_vars, dtype=np.float64)
        sumsq = np.zeros(n_vars, dtype=np.float64)
        for i in range(data.shape[0]):
            col = indices[i]
            value = data[i]
            sums[col] += value
            sumsq[col] += value * value
        return sums, sumsq

    @numba.njit(parallel=True, cache=True, fastmath=True)
    def _fused_scale_clip(x, mean, inv_std, max_value):
        # 2026-05-21 fix: restore symmetric clip to [-max_value, +max_value].
        # Previous `_fused_scale_clip_upper` only applied the upper cap, which
        # diverged from `sc.pp.scale`'s documented two-sided clip. A `zyme
        # validate iterate` audit (Cat 4 — fixture-only behavior) flagged that
        # benchmark data never exercised the negative tail (post-scale values
        # stayed in [-3, +3]) so the asymmetry was invisible to concordance
        # gates, but real data with rare-low or bimodal genes / doublets /
        # batch outliers gets unbounded negative values, breaking downstream
        # PCA / UMAP / clustering. The negative branch costs one extra register
        # compare per element and does not affect JIT vectorization.
        n_rows, n_cols = x.shape
        for i in numba.prange(n_rows):
            row = x[i]
            for j in range(n_cols):
                value = (row[j] - mean[j]) * inv_std[j]
                if value > max_value:
                    value = max_value
                elif value < -max_value:
                    value = -max_value
                row[j] = value

    # Back-compat alias for any external callers that imported the old name.
    _fused_scale_clip_upper = _fused_scale_clip

    # Warmup JIT
    _w_x = np.zeros((1, 1), dtype=np.float32)
    _w_v = np.ones(1, dtype=np.float32)
    _accumulate_stats(np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), 1)
    _fused_scale_clip(_w_x, _w_v, _w_v, np.float32(10.0))
    del _w_x, _w_v


def fast_scale(
    data,
    *,
    zero_center=True,
    max_value=None,
    copy=False,
    layer=None,
    obsm=None,
    mask_obs=None,
    zyme=True,
):
    if not zyme:
        return _orig_scale()(
            data, zero_center=zero_center, max_value=max_value, copy=copy,
            layer=layer, obsm=obsm, mask_obs=mask_obs,
        )

    from anndata import AnnData

    if (
        not HAS_NUMBA
        or not isinstance(data, AnnData)
        or not zero_center
        or layer is not None
        or obsm is not None
        or mask_obs is not None
    ):
        return _orig_scale()(
            data, zero_center=zero_center, max_value=max_value, copy=copy,
            layer=layer, obsm=obsm, mask_obs=mask_obs,
        )

    adata = data.copy() if copy else data
    x = adata.X
    if not sparse.isspmatrix_csr(x):
        return _orig_scale()(
            data, zero_center=zero_center, max_value=max_value, copy=copy,
            layer=layer, obsm=obsm, mask_obs=mask_obs,
        )

    n_obs = x.shape[0]
    sums, sumsq = _accumulate_stats(x.indices, x.data, x.shape[1])
    mean = sums / n_obs
    var = sumsq / n_obs - mean * mean
    if n_obs > 1:
        var *= n_obs / (n_obs - 1)
    std = np.sqrt(var, dtype=np.float64)
    std[std == 0] = 1.0

    if x.dtype != np.float32:
        x = x.astype(np.float32)
    scaled = x.toarray()
    _fused_scale_clip_upper(
        scaled,
        mean.astype(np.float32, copy=False),
        np.reciprocal(std, dtype=np.float32),
        np.float32(max_value if max_value is not None else np.inf),
    )

    adata.X = scaled
    adata.var["mean"] = mean
    adata.var["var"] = var
    adata.var["std"] = std
    return adata if copy else None
