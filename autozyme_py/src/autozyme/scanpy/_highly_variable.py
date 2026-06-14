"""Fast highly_variable_genes kernels.

Vendored from scanpy-turbo/_turbo/highly_variable.py. ``zyme=False`` looks
up the upstream original via the autozyme dispatcher's
``__autozyme_original__`` attribute.

Single ``prange`` over CSR rows accumulates per-column ``sum`` and
``sum-of-squares`` simultaneously, then computes variance in closed form
without a second pass for flavor='seurat'. A separate narrow fast path covers
batch-aware flavor='seurat_v3'/'seurat_v3_paper' on CSR raw counts.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

try:
    import numba
    from numba import prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _orig_hvg():
    """Fetch upstream highly_variable_genes via the autozyme dispatcher."""
    import scanpy as sc
    return getattr(sc.pp.highly_variable_genes, "__autozyme_original__",
                   sc.pp.highly_variable_genes)


if HAS_NUMBA:
    @numba.njit(cache=True,
                fastmath={'afn', 'reassoc', 'contract', 'arcp'},
                boundscheck=False, parallel=True)
    def _hvg_one_pass(data, indices, indptr, n_cols, n_rows, n_bins, n_threads):
        """Single-pass per-column sum + sum-of-squares; closed-form variance."""

        # Size per-thread buffers by the live pool, not the n_threads arg:
        # get_thread_id() can range up to get_num_threads(), so a caller
        # passing a smaller n_threads would otherwise index out of bounds.
        # The extra rows stay all-zero, so the reduction below is unchanged.
        n_buf = numba.get_num_threads()
        local_sum = np.zeros((n_buf, n_cols), dtype=np.float64)
        local_sumsq = np.zeros((n_buf, n_cols), dtype=np.float64)

        for row in prange(n_rows):
            tid = numba.get_thread_id()
            for i in range(indptr[row], indptr[row + 1]):
                j = indices[i]
                x = np.float64(np.expm1(data[i]))
                local_sum[tid, j] += x
                local_sumsq[tid, j] += x * x

        inv_n = 1.0 / n_rows
        inv_n1 = 1.0 / (n_rows - 1.0)
        mean = np.empty(n_cols, dtype=np.float64)
        dispersion = np.empty(n_cols, dtype=np.float64)
        mean_log = np.empty(n_cols, dtype=np.float64)
        ml_min = np.inf
        ml_max = -np.inf

        for j in range(n_cols):
            s = 0.0
            ss = 0.0
            for t in range(n_buf):
                s += local_sum[t, j]
                ss += local_sumsq[t, j]
            m = s * inv_n
            if m == 0.0:
                m = 1e-12
            mean[j] = m
            ccs = ss - n_rows * m * m
            if ccs < 0.0:
                ccs = 0.0  # guard catastrophic cancellation
            v = ccs * inv_n1
            d = v / m
            if d <= 0.0:
                dispersion[j] = np.nan
            else:
                dispersion[j] = np.log(d)
            ml = np.log1p(m)
            mean_log[j] = ml
            if ml < ml_min:
                ml_min = ml
            if ml > ml_max:
                ml_max = ml

        bin_width = (ml_max - ml_min) / n_bins
        if bin_width == 0.0:
            bin_width = 1.0
        inv_bw = 1.0 / bin_width

        bin_count = np.zeros(n_bins, dtype=np.int64)
        bin_sum = np.zeros(n_bins, dtype=np.float64)
        bin_sum_sq = np.zeros(n_bins, dtype=np.float64)
        bin_indices = np.empty(n_cols, dtype=np.int64)

        for j in range(n_cols):
            b = int((mean_log[j] - ml_min) * inv_bw)
            if b < 0:
                b = 0
            elif b >= n_bins:
                b = n_bins - 1
            bin_indices[j] = b
            d = dispersion[j]
            if not np.isnan(d):
                bin_count[b] += 1
                bin_sum[b] += d
                bin_sum_sq[b] += d * d

        bin_avg = np.zeros(n_bins, dtype=np.float64)
        bin_dev = np.zeros(n_bins, dtype=np.float64)
        for b in range(n_bins):
            c = bin_count[b]
            if c == 0:
                bin_dev[b] = 1.0
                continue
            avg = bin_sum[b] / c
            if c == 1:
                bin_avg[b] = 0.0
                bin_dev[b] = avg if avg != 0.0 else 1.0
            else:
                dev = np.sqrt((bin_sum_sq[b] - bin_sum[b] ** 2 / c) / (c - 1))
                if dev == 0.0 or np.isnan(dev):
                    dev = avg if avg != 0.0 else 1.0
                bin_avg[b] = avg
                bin_dev[b] = dev

        disp_norm = np.empty(n_cols, dtype=np.float64)
        for j in range(n_cols):
            b = bin_indices[j]
            disp_norm[j] = (dispersion[j] - bin_avg[b]) / bin_dev[b]

        return mean, dispersion, disp_norm

    # Warmup with non-degenerate data so the first user call doesn't pay JIT.
    _wd = np.array([0.5, 1.0, 0.3, 0.8, 1.5, 0.2], dtype=np.float32)
    _wi = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
    _wp = np.array([0, 3, 6], dtype=np.int32)
    _hvg_one_pass(_wd, _wi, _wp, 3, 2, 20, 1)
    del _wd, _wi, _wp

    @numba.njit(cache=True, boundscheck=False, parallel=True)
    def _v3_batch_mean_var(
        data, indices, indptr, row_to_batch,
        n_cols, n_batches, batch_sizes, n_threads, do_check,
    ):
        """Compute per-batch/overall mean/variance and optional count-data check."""
        n_rows = indptr.shape[0] - 1
        # Size per-thread buffers by the live pool, not the n_threads arg:
        # get_thread_id() can range up to get_num_threads(), so a smaller
        # n_threads would index out of bounds. Extra rows stay all-zero.
        n_buf = numba.get_num_threads()
        local_sum = np.zeros((n_buf, n_batches, n_cols), dtype=np.float64)
        local_sumsq = np.zeros((n_buf, n_batches, n_cols), dtype=np.float64)
        local_bad_values = np.zeros(n_buf, dtype=np.uint8)

        for row in prange(n_rows):
            tid = numba.get_thread_id()
            b = row_to_batch[row]
            for p in range(indptr[row], indptr[row + 1]):
                col = indices[p]
                val = np.float64(data[p])
                if do_check and local_bad_values[tid] == 0:
                    if np.signbit(val) or val != np.floor(val):
                        local_bad_values[tid] = 1
                local_sum[tid, b, col] += val
                local_sumsq[tid, b, col] += val * val

        col_sum = np.zeros((n_batches, n_cols), dtype=np.float64)
        col_sumsq = np.zeros((n_batches, n_cols), dtype=np.float64)
        has_bad_values = False
        for tid in range(n_buf):
            if local_bad_values[tid] != 0:
                has_bad_values = True
            for b in range(n_batches):
                for col in range(n_cols):
                    col_sum[b, col] += local_sum[tid, b, col]
                    col_sumsq[b, col] += local_sumsq[tid, b, col]

        means = np.empty((n_batches, n_cols), dtype=np.float64)
        variances = np.empty((n_batches, n_cols), dtype=np.float64)
        overall_sum = np.zeros(n_cols, dtype=np.float64)
        overall_sumsq = np.zeros(n_cols, dtype=np.float64)

        for b in range(n_batches):
            n = np.float64(batch_sizes[b])
            scale = n / (n - 1.0)
            for col in range(n_cols):
                s = col_sum[b, col]
                ss = col_sumsq[b, col]
                m = s / n
                v = (ss / n - m * m) * scale
                if v < 0.0:
                    v = 0.0
                means[b, col] = m
                variances[b, col] = v
                overall_sum[col] += s
                overall_sumsq[col] += ss

        overall_mean = np.empty(n_cols, dtype=np.float64)
        overall_var = np.empty(n_cols, dtype=np.float64)
        n_total = np.float64(n_rows)
        overall_scale = n_total / (n_total - 1.0)
        for col in range(n_cols):
            m = overall_sum[col] / n_total
            v = (overall_sumsq[col] / n_total - m * m) * overall_scale
            if v < 0.0:
                v = 0.0
            overall_mean[col] = m
            overall_var[col] = v

        return means, variances, overall_mean, overall_var, has_bad_values

    @numba.njit(cache=True, boundscheck=False, parallel=True)
    def _v3_batch_clip_sum(
        data, indices, indptr, row_to_batch,
        n_cols, n_batches, clip_vals, n_threads,
    ):
        """Compute clipped sums and squared sums for all batches."""
        n_rows = indptr.shape[0] - 1
        # Size per-thread buffers by the live pool, not the n_threads arg:
        # get_thread_id() can range up to get_num_threads(), so a smaller
        # n_threads would index out of bounds. Extra rows stay all-zero.
        n_buf = numba.get_num_threads()
        local_sum = np.zeros((n_buf, n_batches, n_cols), dtype=np.float64)
        local_sumsq = np.zeros((n_buf, n_batches, n_cols), dtype=np.float64)

        for row in prange(n_rows):
            tid = numba.get_thread_id()
            b = row_to_batch[row]
            for p in range(indptr[row], indptr[row + 1]):
                col = indices[p]
                val = np.float64(data[p])
                cap = clip_vals[b, col]
                if val > cap:
                    val = cap
                local_sum[tid, b, col] += val
                local_sumsq[tid, b, col] += val * val

        out_sum = np.zeros((n_batches, n_cols), dtype=np.float64)
        out_sumsq = np.zeros((n_batches, n_cols), dtype=np.float64)
        for tid in range(n_buf):
            for b in range(n_batches):
                for col in range(n_cols):
                    out_sum[b, col] += local_sum[tid, b, col]
                    out_sumsq[b, col] += local_sumsq[tid, b, col]
        return out_sum, out_sumsq

    _v3_wd = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    _v3_wi = np.array([0, 1, 0, 1], dtype=np.int32)
    _v3_wp = np.array([0, 2, 4], dtype=np.int32)
    _v3_r2b = np.array([0, 0], dtype=np.int32)
    _v3_bs = np.array([2], dtype=np.int64)
    _v3_mean, _v3_var, _v3_om, _v3_ov, _v3_bad = _v3_batch_mean_var(
        _v3_wd, _v3_wi, _v3_wp, _v3_r2b, 2, 1, _v3_bs, 1, True,
    )
    _v3_clip = np.array([[10.0, 10.0]], dtype=np.float64)
    _v3_batch_clip_sum(_v3_wd, _v3_wi, _v3_wp, _v3_r2b, 2, 1, _v3_clip, 1)
    del _v3_wd, _v3_wi, _v3_wp, _v3_r2b, _v3_bs, _v3_mean, _v3_var, _v3_om, _v3_ov, _v3_bad, _v3_clip


def _fast_hvg_seurat(adata, *, n_top_genes=None, n_bins=20,
                     layer=None, subset=False, inplace=True,
                     min_disp=0.5, max_disp=np.inf,
                     min_mean=0.0125, max_mean=3,
                     batch_key=None, **kwargs):
    """Optimized HVG for flavor='seurat'."""
    import pandas as pd

    x = adata.layers[layer] if layer else adata.X

    if sp.issparse(x):
        if not sp.isspmatrix_csr(x):
            x = x.tocsr()
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        n_threads = numba.get_num_threads()
        mean, dispersion, disp_norm = _hvg_one_pass(
            x.data, x.indices, x.indptr,
            x.shape[1], x.shape[0], n_bins, n_threads,
        )
    else:
        # Dense input: rare; fall back to upstream so we don't reimplement it.
        return _orig_hvg()(
            adata, layer=layer, n_top_genes=n_top_genes, n_bins=n_bins,
            min_disp=min_disp, max_disp=max_disp,
            min_mean=min_mean, max_mean=max_mean,
            flavor="seurat", subset=subset, inplace=inplace,
            batch_key=batch_key, **kwargs,
        )

    n_genes = len(mean)
    if n_top_genes is not None:
        disp_norm_clean = np.nan_to_num(disp_norm, nan=-np.inf)
        if n_top_genes < n_genes:
            idx = np.argpartition(disp_norm_clean, -n_top_genes)[-n_top_genes:]
            cutoff = disp_norm_clean[idx].min()
        else:
            cutoff = -np.inf
        highly_variable = disp_norm_clean >= cutoff
    else:
        disp_norm_clean = np.nan_to_num(disp_norm)
        mean_log = np.log1p(mean)
        highly_variable = (
            (mean_log > min_mean) & (mean_log < max_mean)
            & (disp_norm_clean > min_disp) & (disp_norm_clean < max_disp)
        )

    # Stock scanpy stores log1p(mean) for flavor="seurat".
    means_stored = np.log1p(mean)

    if inplace:
        adata.uns["hvg"] = {"flavor": "seurat"}
        adata.var["highly_variable"] = highly_variable
        adata.var["means"] = means_stored
        adata.var["dispersions"] = dispersion
        adata.var["dispersions_norm"] = disp_norm.astype(np.float32)
        if subset:
            adata._inplace_subset_var(highly_variable)
        return None
    df = pd.DataFrame({
        "highly_variable": highly_variable,
        "means": means_stored,
        "dispersions": dispersion,
        "dispersions_norm": disp_norm,
    }, index=adata.var_names)
    if subset:
        df = df.loc[df["highly_variable"]]
    return df


def _fast_hvg_seurat_v3_batch(
    adata,
    *,
    flavor="seurat_v3_paper",
    layer=None,
    n_top_genes=2000,
    batch_key=None,
    check_values=True,
    span=0.3,
    subset=False,
    inplace=True,
    **kwargs,
):
    """Optimized batch-aware HVG for seurat_v3(_paper) on CSR raw counts."""
    if (
        not HAS_NUMBA
        or batch_key is None
        or flavor not in {"seurat_v3", "seurat_v3_paper"}
        or n_top_genes is None
        or subset
        or not inplace
        or kwargs
    ):
        return _orig_hvg()(
            adata,
            flavor=flavor,
            layer=layer,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            check_values=check_values,
            span=span,
            subset=subset,
            inplace=inplace,
            **kwargs,
        )

    from pandas.api.types import CategoricalDtype
    try:
        from skmisc.loess import loess
    except ImportError:
        return _orig_hvg()(
            adata,
            flavor=flavor,
            layer=layer,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            check_values=check_values,
            span=span,
            subset=subset,
            inplace=inplace,
        )

    x = adata.layers[layer] if layer else adata.X
    if not sp.isspmatrix_csr(x):
        return _orig_hvg()(
            adata,
            flavor=flavor,
            layer=layer,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            check_values=check_values,
            span=span,
            subset=subset,
            inplace=inplace,
        )
    if x.shape[0] < 2 or x.shape[1] == 0:
        return _orig_hvg()(
            adata,
            flavor=flavor,
            layer=layer,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            check_values=check_values,
            span=span,
            subset=subset,
            inplace=inplace,
        )
    batch = adata.obs[batch_key]
    if not isinstance(batch.dtype, CategoricalDtype):
        batch = batch.astype("category")
    codes = batch.cat.codes.to_numpy(dtype=np.int32, copy=True)
    if np.any(codes < 0):
        return _orig_hvg()(
            adata,
            flavor=flavor,
            layer=layer,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            check_values=check_values,
            span=span,
            subset=subset,
            inplace=inplace,
        )

    n_batches = len(batch.cat.categories)
    batch_sizes = np.bincount(codes, minlength=n_batches).astype(np.int64)
    if n_batches < 1 or np.any(batch_sizes < 2):
        return _orig_hvg()(
            adata,
            flavor=flavor,
            layer=layer,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            check_values=check_values,
            span=span,
            subset=subset,
            inplace=inplace,
        )

    n_threads = numba.get_num_threads()
    n_genes = x.shape[1]
    means, variances, overall_mean, overall_var, has_bad_values = _v3_batch_mean_var(
        x.data,
        x.indices,
        x.indptr,
        codes,
        n_genes,
        n_batches,
        batch_sizes,
        n_threads,
        bool(check_values),
    )
    if check_values and has_bad_values:
        warnings.warn(
            f"`{flavor=!r}` expects raw count data, but non-integers were found.",
            UserWarning,
            stacklevel=3,
        )

    clip_vals = np.empty((n_batches, n_genes), dtype=np.float64)
    reg_stds = np.empty((n_batches, n_genes), dtype=np.float64)
    for b in range(n_batches):
        mean = means[b]
        var = variances[b]
        not_const = var > 0
        if int(np.sum(not_const)) < 2:
            return _orig_hvg()(
                adata,
                flavor=flavor,
                layer=layer,
                n_top_genes=n_top_genes,
                batch_key=batch_key,
                check_values=check_values,
                span=span,
                subset=subset,
                inplace=inplace,
            )
        fitted = np.zeros(n_genes, dtype=np.float64)
        model = loess(
            np.log10(mean[not_const]),
            np.log10(var[not_const]),
            span=span,
            degree=2,
        )
        model.fit()
        fitted[not_const] = model.outputs.fitted_values
        reg_std = np.sqrt(10 ** fitted)
        reg_stds[b] = reg_std
        clip_vals[b] = reg_std * np.sqrt(float(batch_sizes[b])) + mean

    clip_sums, clip_sumsq = _v3_batch_clip_sum(
        x.data,
        x.indices,
        x.indptr,
        codes,
        n_genes,
        n_batches,
        clip_vals,
        n_threads,
    )

    norm_gene_vars = np.empty((n_batches, n_genes), dtype=np.float64)
    for b in range(n_batches):
        n = float(batch_sizes[b])
        mean = means[b]
        reg_std = reg_stds[b]
        norm_gene_vars[b] = (
            (n * np.square(mean)) + clip_sumsq[b] - 2.0 * clip_sums[b] * mean
        ) / ((n - 1.0) * np.square(reg_std))

    ranked = np.empty((n_batches, n_genes), dtype=np.float32)
    ranks = np.arange(n_genes, dtype=np.float32)
    for b in range(n_batches):
        order = np.argsort(-norm_gene_vars[b])
        ranked[b, order] = ranks

    num_batches_high_var = np.sum(ranked < int(n_top_genes), axis=0).astype(np.int32)
    ranked[ranked >= int(n_top_genes)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_ranked = np.nanmedian(ranked, axis=0)
    variances_norm = np.mean(norm_gene_vars, axis=0)

    max_rank = np.finfo(np.float32).max
    rank_key = np.nan_to_num(median_ranked, nan=max_rank).astype(np.float32)
    if flavor == "seurat_v3":
        sorted_idx = np.lexsort((-num_batches_high_var, rank_key))
    else:
        sorted_idx = np.lexsort((rank_key, -num_batches_high_var))

    highly_variable = np.zeros(n_genes, dtype=bool)
    highly_variable[sorted_idx[: int(n_top_genes)]] = True

    adata.uns["hvg"] = {"flavor": flavor}
    adata.var["highly_variable"] = highly_variable
    adata.var["highly_variable_rank"] = median_ranked
    adata.var["means"] = overall_mean
    adata.var["variances"] = overall_var
    adata.var["variances_norm"] = variances_norm.astype("float64", copy=False)
    adata.var["highly_variable_nbatches"] = num_batches_high_var
    return None


def _patched_hvg(adata, *, flavor="seurat", zyme=True, **kwargs):
    if not zyme:
        return _orig_hvg()(adata, flavor=flavor, **kwargs)
    if flavor in {"seurat_v3", "seurat_v3_paper"} and kwargs.get("batch_key") is not None:
        return _fast_hvg_seurat_v3_batch(adata, flavor=flavor, **kwargs)
    if kwargs.get("batch_key") is not None:
        return _orig_hvg()(adata, flavor=flavor, **kwargs)
    if flavor == "seurat" and HAS_NUMBA:
        return _fast_hvg_seurat(adata, **kwargs)
    return _orig_hvg()(adata, flavor=flavor, **kwargs)
