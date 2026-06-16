"""Fast rank_genes_groups: fused numba kernels for Wilcoxon rank-sum test.

Vendored from scanpy-turbo/_turbo/rank_genes.py — top-level replacement only.

The upstream scanpy-turbo also patched ``_RankGenes._basic_stats``,
``.wilcoxon``, ``.compute_statistics`` for fallback paths (tie_correct,
reference group, dense input). Those class patches populated
``self._results`` (legacy API), but scanpy 1.11.5's
``rank_genes_groups`` consumes ``self.stats`` (DataFrame, new API). The
mismatch crashes the fallback path. Class patches removed from this
vendor — fallback delegates to vanilla scanpy (correct, unaccelerated).

``zyme=False`` (or being inside ``with autozyme.disabled():``) routes
through the dispatcher's stored upstream original.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _orig_rank_genes():
    """Fetch upstream sc.tl.rank_genes_groups via dispatcher attribute."""
    import scanpy as sc
    return getattr(sc.tl.rank_genes_groups, "__autozyme_original__",
                   sc.tl.rank_genes_groups)


if HAS_NUMBA:

    # ===== Numba kernels =====

    @njit(parallel=True)
    def _fused_stats_rank_sums_csc(indptr, indices, data, n_cells,
                                    group_int, group_sizes, n_groups,
                                    group_sums, group_sq_sums, group_nnz, rank_sums):
        """Fused: compute group stats + rank sums in single CSC pass."""
        n_cols = rank_sums.shape[1]
        for j in prange(n_cols):
            col_offset = indptr[j]
            nnz_j = indptr[j + 1] - col_offset
            n_zeros = n_cells - nnz_j
            zero_rank = (n_zeros + 1.0) / 2.0

            g_nnz = np.zeros(n_groups, dtype=np.int64)
            g_sum = np.zeros(n_groups, dtype=np.float64)
            g_sq_sum = np.zeros(n_groups, dtype=np.float64)
            g_rank_sum = np.zeros(n_groups, dtype=np.float64)

            if nnz_j > 0:
                vals = np.empty(nnz_j, dtype=np.float64)
                row_idx = np.empty(nnz_j, dtype=np.int64)
                for k in range(nnz_j):
                    vals[k] = data[col_offset + k]
                    row_idx[k] = indices[col_offset + k]
                    g = group_int[row_idx[k]]
                    if g >= 0:
                        g_nnz[g] += 1
                        g_sum[g] += vals[k]
                        g_sq_sum[g] += vals[k] * vals[k]

                sort_order = np.argsort(vals)

                i = 0
                while i < nnz_j:
                    j_end = i + 1
                    while j_end < nnz_j and vals[sort_order[j_end]] == vals[sort_order[i]]:
                        j_end += 1
                    avg_rank = n_zeros + (i + 1.0 + j_end) / 2.0
                    for k in range(i, j_end):
                        g = group_int[row_idx[sort_order[k]]]
                        if g >= 0:
                            g_rank_sum[g] += avg_rank
                    i = j_end

            for g in range(n_groups):
                group_sums[g, j] = g_sum[g]
                group_sq_sums[g, j] = g_sq_sum[g]
                group_nnz[g, j] = g_nnz[g]
                rank_sums[g, j] = g_rank_sum[g] + (group_sizes[g] - g_nnz[g]) * zero_rank

    @njit(parallel=True)
    def _fused_all_csc(indptr, indices, data, n_cells,
                       group_int, group_sizes, n_groups,
                       scores, pvals, g_sum_out, total_sum_out):
        """Fused: sums + rank sums + z-scores + p-values in single CSC pass."""
        n_cols = scores.shape[1]
        sqrt2_inv = 1.0 / math.sqrt(2.0)
        n_p1 = float(n_cells + 1)
        half_n_p1 = n_p1 / 2.0
        twelve_inv = 1.0 / 12.0

        z_std_inv = np.empty(n_groups, dtype=np.float64)
        z_mean_rank = np.empty(n_groups, dtype=np.float64)
        for g in range(n_groups):
            n_g = float(group_sizes[g])
            n_rest = float(n_cells) - n_g
            z_std_inv[g] = 1.0 / math.sqrt(n_g * n_rest * n_p1 * twelve_inv)
            z_mean_rank[g] = n_g * half_n_p1

        for j in prange(n_cols):
            col_offset = indptr[j]
            nnz_j = indptr[j + 1] - col_offset
            n_zeros = n_cells - nnz_j
            zero_rank = (n_zeros + 1.0) / 2.0

            g_nnz = np.zeros(n_groups, dtype=np.int64)
            g_sum = np.zeros(n_groups, dtype=np.float64)
            g_rank_sum = np.zeros(n_groups, dtype=np.float64)

            if nnz_j > 0:
                i = 0
                while i < nnz_j:
                    j_end = i + 1
                    while j_end < nnz_j and data[col_offset + j_end] == data[col_offset + i]:
                        j_end += 1
                    avg_rank = n_zeros + (i + 1.0 + j_end) / 2.0
                    v = np.float64(data[col_offset + i])
                    for k in range(i, j_end):
                        g = group_int[indices[col_offset + k]]
                        if g >= 0:
                            g_nnz[g] += 1
                            g_sum[g] += v
                            g_rank_sum[g] += avg_rank
                    i = j_end

            col_total = 0.0
            for g in range(n_groups):
                g_sum_out[g, j] = g_sum[g]
                col_total += g_sum[g]
                rs = g_rank_sum[g] + (group_sizes[g] - g_nnz[g]) * zero_rank
                z = (rs - z_mean_rank[g]) * z_std_inv[g]
                if z != z:
                    z = 0.0
                scores[g, j] = z
                pvals[g, j] = math.erfc(abs(z) * sqrt2_inv)
            total_sum_out[j] = col_total

    @njit(parallel=True)
    def _bh_correct_2d(pvals, pvals_adj, n_groups, n_genes):
        """Parallel BH correction across groups."""
        for g in prange(n_groups):
            order = np.argsort(pvals[g])
            running_min = 1.0
            for i in range(n_genes - 1, -1, -1):
                adjusted = pvals[g, order[i]] * n_genes / (i + 1.0)
                if adjusted < running_min:
                    running_min = adjusted
                if running_min > 1.0:
                    running_min = 1.0
                pvals_adj[g, order[i]] = running_min

    @njit
    def _dual_sort(vals, idx, n):
        """Sort vals[0:n] and idx[0:n] together by vals, in-place."""
        if n <= 1:
            return
        stack = np.empty(80, dtype=np.int64)
        top = 0
        stack[top] = 0; top += 1
        stack[top] = n; top += 1
        while top > 0:
            top -= 1; hi = stack[top]
            top -= 1; lo = stack[top]
            if hi - lo <= 16:
                for i in range(lo + 1, hi):
                    kv = vals[i]; ki = idx[i]
                    j = i - 1
                    while j >= lo and vals[j] > kv:
                        vals[j + 1] = vals[j]; idx[j + 1] = idx[j]; j -= 1
                    vals[j + 1] = kv; idx[j + 1] = ki
                continue
            mid = (lo + hi) >> 1
            if vals[lo] > vals[mid]:
                vals[lo], vals[mid] = vals[mid], vals[lo]; idx[lo], idx[mid] = idx[mid], idx[lo]
            if vals[lo] > vals[hi - 1]:
                vals[lo], vals[hi - 1] = vals[hi - 1], vals[lo]; idx[lo], idx[hi - 1] = idx[hi - 1], idx[lo]
            if vals[mid] > vals[hi - 1]:
                vals[mid], vals[hi - 1] = vals[hi - 1], vals[mid]; idx[mid], idx[hi - 1] = idx[hi - 1], idx[mid]
            pivot = vals[mid]
            vals[mid], vals[hi - 2] = vals[hi - 2], vals[mid]; idx[mid], idx[hi - 2] = idx[hi - 2], idx[mid]
            i_pos = lo; j_pos = hi - 2
            while True:
                i_pos += 1
                while vals[i_pos] < pivot: i_pos += 1
                j_pos -= 1
                while vals[j_pos] > pivot: j_pos -= 1
                if i_pos >= j_pos: break
                vals[i_pos], vals[j_pos] = vals[j_pos], vals[i_pos]; idx[i_pos], idx[j_pos] = idx[j_pos], idx[i_pos]
            vals[i_pos], vals[hi - 2] = vals[hi - 2], vals[i_pos]; idx[i_pos], idx[hi - 2] = idx[hi - 2], idx[i_pos]
            if i_pos - lo > hi - i_pos - 1:
                stack[top] = lo; top += 1; stack[top] = i_pos; top += 1
                stack[top] = i_pos + 1; top += 1; stack[top] = hi; top += 1
            else:
                stack[top] = i_pos + 1; top += 1; stack[top] = hi; top += 1
                stack[top] = lo; top += 1; stack[top] = i_pos; top += 1

    @njit
    def _partial_dual_sort(vals, idx, n, k):
        """Partial sort: put smallest k elements in sorted order at vals[0:k]."""
        if k >= n:
            _dual_sort(vals, idx, n)
            return
        if k <= 0 or n <= 1:
            return
        stack = np.empty(80, dtype=np.int64)
        top = 0
        stack[top] = 0; top += 1
        stack[top] = n; top += 1
        while top > 0:
            top -= 1; hi = stack[top]
            top -= 1; lo = stack[top]
            if hi - lo <= 16:
                for i in range(lo + 1, hi):
                    kv = vals[i]; ki = idx[i]
                    j = i - 1
                    while j >= lo and vals[j] > kv:
                        vals[j + 1] = vals[j]; idx[j + 1] = idx[j]; j -= 1
                    vals[j + 1] = kv; idx[j + 1] = ki
                continue
            mid = (lo + hi) >> 1
            if vals[lo] > vals[mid]:
                vals[lo], vals[mid] = vals[mid], vals[lo]; idx[lo], idx[mid] = idx[mid], idx[lo]
            if vals[lo] > vals[hi - 1]:
                vals[lo], vals[hi - 1] = vals[hi - 1], vals[lo]; idx[lo], idx[hi - 1] = idx[hi - 1], idx[lo]
            if vals[mid] > vals[hi - 1]:
                vals[mid], vals[hi - 1] = vals[hi - 1], vals[mid]; idx[mid], idx[hi - 1] = idx[hi - 1], idx[mid]
            pivot = vals[mid]
            vals[mid], vals[hi - 2] = vals[hi - 2], vals[mid]; idx[mid], idx[hi - 2] = idx[hi - 2], idx[mid]
            i_pos = lo; j_pos = hi - 2
            while True:
                i_pos += 1
                while vals[i_pos] < pivot: i_pos += 1
                j_pos -= 1
                while vals[j_pos] > pivot: j_pos -= 1
                if i_pos >= j_pos: break
                vals[i_pos], vals[j_pos] = vals[j_pos], vals[i_pos]; idx[i_pos], idx[j_pos] = idx[j_pos], idx[i_pos]
            vals[i_pos], vals[hi - 2] = vals[hi - 2], vals[i_pos]; idx[i_pos], idx[hi - 2] = idx[hi - 2], idx[i_pos]
            if i_pos > lo and lo < k:
                stack[top] = lo; top += 1; stack[top] = i_pos; top += 1
            if i_pos + 1 < hi and i_pos + 1 < k:
                stack[top] = i_pos + 1; top += 1; stack[top] = hi; top += 1

    @njit(parallel=True)
    def _batch_top_n(scores, n_top, result):
        """Find top-n indices per row, sorted by descending score."""
        n_groups = scores.shape[0]
        n_genes = scores.shape[1]
        for g in prange(n_groups):
            vals = np.empty(n_genes, dtype=np.float64)
            idx_arr = np.empty(n_genes, dtype=np.int64)
            for i in range(n_genes):
                vals[i] = -scores[g, i]
                idx_arr[i] = i
            _partial_dual_sort(vals, idx_arr, n_genes, n_top)
            for i in range(n_top):
                result[g, i] = idx_arr[i]

    @njit(parallel=True)
    def _bh_gather(pvals, all_scores, top_idx,
                   out_scores, out_pvals, out_pvals_adj,
                   n_groups, n_genes, n_top):
        """Fused BH correction + top-N gather."""
        for g in prange(n_groups):
            pv = pvals[g].copy()
            order = np.empty(n_genes, dtype=np.int64)
            for i in range(n_genes):
                order[i] = i
            _dual_sort(pv, order, n_genes)
            adj = np.empty(n_genes, dtype=np.float64)
            running_min = 1.0
            for i in range(n_genes - 1, -1, -1):
                adjusted = pv[i] * n_genes / (i + 1.0)
                if adjusted < running_min:
                    running_min = adjusted
                if running_min > 1.0:
                    running_min = 1.0
                adj[order[i]] = running_min
            for i in range(n_top):
                idx = top_idx[g, i]
                out_scores[i, g] = all_scores[g, idx]
                out_pvals[i, g] = pvals[g, idx]
                out_pvals_adj[i, g] = adj[idx]

    @njit(parallel=True)
    def _presort_csc_columns(indptr, indices, data):
        """Pre-sort all CSC columns by value."""
        n_cols = len(indptr) - 1
        for j in prange(n_cols):
            start = indptr[j]
            end = indptr[j + 1]
            nnz = end - start
            if nnz > 1:
                _dual_sort(data[start:end], indices[start:end], nnz)

    @njit(parallel=True)
    def _compute_topn_logfc(g_sum_all, total_sum, top_idx, out_logfc,
                             inv_n_g, inv_n_rest, log_base_factor,
                             n_groups, n_top):
        """Compute logfc only for top-N genes per group (52x fewer transcendental calls)."""
        for g in prange(n_groups):
            for i in range(n_top):
                gene = top_idx[g, i]
                mean_g = g_sum_all[g, gene] * inv_n_g[g]
                mean_rest = (total_sum[gene] - g_sum_all[g, gene]) * inv_n_rest[g]
                if log_base_factor == 1.0:
                    expm1_mean = math.expm1(mean_g)
                    expm1_rest = math.expm1(mean_rest)
                else:
                    expm1_mean = math.expm1(mean_g * log_base_factor)
                    expm1_rest = math.expm1(mean_rest * log_base_factor)
                out_logfc[i, g] = math.log2((expm1_mean + 1e-9) / (expm1_rest + 1e-9))

    @njit(parallel=True)
    def _sparse_rankdata_csc(indptr, indices, data, n_cells, ranks):
        """Rank cells from CSC sparse data — fallback for tie_correct=True."""
        n_cols = ranks.shape[1]
        for j in prange(n_cols):
            col_offset = indptr[j]
            nnz_j = indptr[j + 1] - col_offset
            n_zeros = n_cells - nnz_j
            zero_rank = (n_zeros + 1.0) / 2.0
            for i in range(n_cells):
                ranks[i, j] = zero_rank
            if nnz_j == 0:
                continue
            vals = np.empty(nnz_j, dtype=np.float64)
            row_idx = np.empty(nnz_j, dtype=np.int64)
            for k in range(nnz_j):
                vals[k] = data[col_offset + k]
                row_idx[k] = indices[col_offset + k]
            sort_order = np.argsort(vals)
            i = 0
            while i < nnz_j:
                j_end = i + 1
                while j_end < nnz_j and vals[sort_order[j_end]] == vals[sort_order[i]]:
                    j_end += 1
                avg_rank = n_zeros + (i + 1.0 + j_end) / 2.0
                for k in range(i, j_end):
                    ranks[row_idx[sort_order[k]], j] = avg_rank
                i = j_end

    # ===== Warmup all JIT functions =====
    def _warmup():
        _w1 = np.zeros((1, 1), dtype=np.float64)
        _w2 = np.zeros((1, 1), dtype=np.float64)
        _w3 = np.zeros((1, 1), dtype=np.int64)
        _w4 = np.zeros((1, 1), dtype=np.float64)
        _fused_stats_rank_sums_csc(
            np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int64),
            np.array([1.0], dtype=np.float64), 2,
            np.array([0, 0], dtype=np.int64), np.array([2], dtype=np.int64), 1,
            _w1, _w2, _w3, _w4,
        )
        _w8 = np.zeros((1, 1), dtype=np.float64)
        _w9 = np.zeros((1, 1), dtype=np.float64)
        _w10 = np.zeros((1, 1), dtype=np.float64)
        _w11 = np.zeros(1, dtype=np.float64)
        _fused_all_csc(
            np.array([0, 1], dtype=np.int32), np.array([0], dtype=np.int32),
            np.array([1.0], dtype=np.float32), 4,
            np.array([0, 0, -1, -1], dtype=np.int64), np.array([2], dtype=np.int64), 1,
            _w8, _w9, _w10, _w11,
        )
        _w5 = np.zeros((1, 2), dtype=np.float64)
        _w6 = np.zeros((1, 2), dtype=np.float64)
        _bh_correct_2d(_w5, _w6, 1, 2)
        _dual_sort(np.array([2.0, 1.0], dtype=np.float64), np.array([0, 1], dtype=np.int64), 2)
        _dual_sort(np.array([2.0, 1.0], dtype=np.float32), np.array([0, 1], dtype=np.int32), 2)
        _partial_dual_sort(np.array([3.0, 1.0, 2.0], dtype=np.float64), np.array([0, 1, 2], dtype=np.int64), 3, 2)
        _wt = np.empty((1, 2), dtype=np.int64)
        _batch_top_n(np.array([[3.0, 1.0, 2.0]], dtype=np.float64), 2, _wt)
        _wb1 = np.zeros((2, 2), dtype=np.float64)
        _wb2 = np.zeros((2, 1), dtype=np.float64)
        _bh_gather(_wb1, _wb1.copy(), np.zeros((2, 1), dtype=np.int64),
                   _wb2, _wb2.copy(), _wb2.copy(), 2, 2, 1)
        _presort_csc_columns(
            np.array([0, 1], dtype=np.int32), np.array([0], dtype=np.int32),
            np.array([1.0], dtype=np.float32))
        _wt_gsum = np.zeros((2, 2), dtype=np.float64)
        _wt_tsum = np.zeros(2, dtype=np.float64)
        _wt_tidx = np.zeros((2, 1), dtype=np.int64)
        _wt_out = np.zeros((1, 2), dtype=np.float64)
        _compute_topn_logfc(_wt_gsum, _wt_tsum, _wt_tidx, _wt_out,
                            np.ones(2, dtype=np.float64), np.ones(2, dtype=np.float64),
                            1.0, 2, 1)
        _w7 = np.empty((2, 1), dtype=np.float64)
        _sparse_rankdata_csc(
            np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int64),
            np.array([1.0], dtype=np.float64), 2, _w7,
        )

    _warmup()


# ===== Top-level function replacement =====

def _fast_rank_genes_groups(
    adata, groupby, *, groups="all", reference="rest",
    n_genes=None, method="wilcoxon", corr_method="benjamini-hochberg",
    tie_correct=False, pts=False, key_added="rank_genes_groups",
    copy=False, use_raw=None, layer=None, rankby_abs=False,
    mask_var=None, zyme=True, **kwds,
):
    """Fast rank_genes_groups — fused numba kernels for wilcoxon.

    Uses fully fused fast path for wilcoxon + rest + sparse + no tie_correct.
    All other cases (tie_correct=True, reference!=rest, dense input) fall
    through to vanilla scanpy — correct but unaccelerated. Pass
    ``zyme=False`` to force a fully vanilla call stack.
    """
    from scanpy._compat import CSBase

    # zyme=False: enter autozyme.disabled() so the class-method dispatchers
    # also fall through. This gives a fully vanilla call stack mid-session,
    # without manually saving/restoring _RankGenes._basic_stats etc.
    if not zyme:
        from autozyme import disabled
        with disabled():
            return _orig_rank_genes()(
                adata, groupby, groups=groups, reference=reference,
                n_genes=n_genes, method=method, corr_method=corr_method,
                tie_correct=tie_correct, pts=pts, key_added=key_added,
                copy=copy, use_raw=use_raw, layer=layer, rankby_abs=rankby_abs,
                mask_var=mask_var, **kwds,
            )

    # For non-fast-path cases, use original (which benefits from patched class methods)
    if not HAS_NUMBA or method != "wilcoxon" or reference != "rest" or tie_correct:
        return _orig_rank_genes()(
            adata, groupby, groups=groups, reference=reference,
            n_genes=n_genes, method=method, corr_method=corr_method,
            tie_correct=tie_correct, pts=pts, key_added=key_added,
            copy=copy, use_raw=use_raw, layer=layer, rankby_abs=rankby_abs,
            mask_var=mask_var, **kwds,
        )

    if use_raw is None:
        use_raw = adata.raw is not None
    if "only_positive" in kwds:
        rankby_abs = not kwds.pop("only_positive")

    adata = adata.copy() if copy else adata

    # Mirror stock scanpy: coerce str/object groupby to categorical. The fused
    # path below reads groupby via cat.codes/cat.categories and so assumes it is
    # already categorical; stock does this coercion via sanitize_anndata before
    # reading groupby, and this vendored fast path dropped that step (so a str
    # groupby crashed with "Can only use .cat accessor with a 'category' dtype").
    # sanitize_anndata is a no-op on already-categorical columns (zero measured
    # cost on the benchmark path) and uses the same natsort category order as
    # stock, keeping the output group order bit-identical. Empty categories are
    # left as-is to match stock, which raises on <2-sample groups, not drops them.
    from scanpy._utils import sanitize_anndata
    sanitize_anndata(adata)

    if key_added is None:
        key_added = "rank_genes_groups"
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = dict(
        groupby=groupby, reference=reference, method=method,
        use_raw=use_raw, layer=layer, corr_method=corr_method,
    )

    # Get X
    adata_comp = adata
    if layer is not None:
        X = adata_comp.layers[layer]
    else:
        if use_raw and adata.raw is not None:
            adata_comp = adata.raw
        X = adata_comp.X
    var_names = adata_comp.var_names

    if mask_var is not None:
        from scanpy.get import _check_mask
        mask = _check_mask(adata, mask_var, "var")
        X = X[:, mask]
        var_names = var_names[mask]

    # Build group mappings from categorical codes
    cat = adata.obs[groupby]
    all_categories = cat.cat.categories
    codes = cat.cat.codes.values
    n_obs = len(codes)

    if groups == "all":
        groups_order = all_categories
        n_groups = len(all_categories)
        group_int = codes.astype(np.int64)
        group_sizes = np.bincount(codes[codes >= 0], minlength=n_groups).astype(np.int64)
    else:
        if isinstance(groups, (str, int)):
            raise ValueError("Specify a sequence of groups")
        groups_list = [str(g) if isinstance(g, int) else g for g in groups]
        groups_ids = [np.where(all_categories == name)[0][0] for name in groups_list]
        groups_order = all_categories[groups_ids]
        n_groups = len(groups_ids)
        group_int = np.full(n_obs, -1, dtype=np.int64)
        group_sizes = np.empty(n_groups, dtype=np.int64)
        for i, gid in enumerate(groups_ids):
            mask_g = codes == gid
            group_int[mask_g] = i
            group_sizes[i] = mask_g.sum()

    n_cells, n_genes_total = X.shape

    # Fast path: sparse + rest comparison
    if isinstance(X, CSBase):
        X_csc = X.tocsc()
        _presort_csc_columns(X_csc.indptr, X_csc.indices, X_csc.data)

        all_scores = np.empty((n_groups, n_genes_total), dtype=np.float64)
        all_pvals = np.empty((n_groups, n_genes_total), dtype=np.float64)
        g_sum_all = np.empty((n_groups, n_genes_total), dtype=np.float64)
        total_sum = np.empty(n_genes_total, dtype=np.float64)

        _fused_all_csc(X_csc.indptr, X_csc.indices, X_csc.data, n_cells,
                       group_int, group_sizes, n_groups,
                       all_scores, all_pvals, g_sum_all, total_sum)

        log_base = adata.uns.get("log1p", {}).get("base")
        log_base_factor = np.log(log_base) if log_base is not None else 1.0

        groups_names = [str(groups_order[gi]) for gi in range(n_groups)]
        n_out = n_genes if n_genes is not None else n_genes_total
        names_dt = np.dtype([(name, "O") for name in groups_names])
        f32_dt = np.dtype([(name, "<f4") for name in groups_names])
        f64_dt = np.dtype([(name, "<f8") for name in groups_names])
        var_names_np = np.asarray(var_names)

        inv_n_g = 1.0 / group_sizes.astype(np.float64)
        inv_n_rest = 1.0 / (n_cells - group_sizes).astype(np.float64)

        if n_genes is not None and n_genes < n_genes_total:
            scores_for_sort = np.abs(all_scores) if rankby_abs else all_scores
            top_idx = np.empty((n_groups, n_genes), dtype=np.int64)
            _batch_top_n(scores_for_sort, n_genes, top_idx)

            out_s = np.empty((n_genes, n_groups), dtype=np.float64)
            out_p = np.empty((n_genes, n_groups), dtype=np.float64)
            out_pa = np.empty((n_genes, n_groups), dtype=np.float64)

            if corr_method == "benjamini-hochberg":
                _bh_gather(all_pvals, all_scores, top_idx,
                           out_s, out_p, out_pa, n_groups, n_genes_total, n_genes)
            elif corr_method == "bonferroni":
                all_pvals_adj = np.minimum(all_pvals * n_genes_total, 1.0)
                row_idx = np.arange(n_groups)[:, None]
                np.copyto(out_s, np.ascontiguousarray(all_scores[row_idx, top_idx].T))
                np.copyto(out_p, np.ascontiguousarray(all_pvals[row_idx, top_idx].T))
                np.copyto(out_pa, np.ascontiguousarray(all_pvals_adj[row_idx, top_idx].T))
            else:
                row_idx = np.arange(n_groups)[:, None]
                np.copyto(out_s, np.ascontiguousarray(all_scores[row_idx, top_idx].T))
                np.copyto(out_p, np.ascontiguousarray(all_pvals[row_idx, top_idx].T))
                out_pa = out_p.copy()

            out_l = np.empty((n_genes, n_groups), dtype=np.float64)
            _compute_topn_logfc(
                g_sum_all, total_sum, top_idx, out_l,
                inv_n_g, inv_n_rest, log_base_factor,
                n_groups, n_genes,
            )

            scores_arr = out_s.astype(np.float32).ravel().view(f32_dt).view(np.recarray)
            pvals_arr = out_p.ravel().view(f64_dt).view(np.recarray)
            pvals_adj_arr = out_pa.ravel().view(f64_dt).view(np.recarray)
            logfc_arr = out_l.astype(np.float32).ravel().view(f32_dt).view(np.recarray)

            all_names = var_names_np[top_idx.ravel()].reshape(n_groups, n_genes)
            names_flat = np.ascontiguousarray(all_names.T)
            try:
                names_arr = names_flat.view(names_dt).reshape(n_genes).view(np.recarray)
            except (ValueError, TypeError):
                names_arr = np.recarray(n_genes, dtype=names_dt)
                for i in range(n_groups):
                    names_arr[groups_names[i]] = all_names[i]
        else:
            mean_g = g_sum_all * inv_n_g[:, None]
            mean_rest = total_sum[None, :] - g_sum_all
            mean_rest *= inv_n_rest[:, None]
            if log_base_factor != 1.0:
                mean_g *= log_base_factor
                mean_rest *= log_base_factor
            np.expm1(mean_g, out=mean_g)
            mean_g += 1e-9
            np.expm1(mean_rest, out=mean_rest)
            mean_rest += 1e-9
            mean_g /= mean_rest
            np.log2(mean_g, out=mean_g)
            all_logfc = mean_g
            del mean_rest

            if corr_method == "benjamini-hochberg":
                all_pvals_adj = np.empty_like(all_pvals)
                _bh_correct_2d(all_pvals, all_pvals_adj, n_groups, n_genes_total)
            elif corr_method == "bonferroni":
                all_pvals_adj = np.minimum(all_pvals * n_genes_total, 1.0)
            else:
                all_pvals_adj = all_pvals.copy()

            n_out = n_genes_total
            names_arr = np.recarray(n_out, dtype=names_dt)
            scores_arr = np.recarray(n_out, dtype=f32_dt)
            pvals_arr = np.recarray(n_out, dtype=f64_dt)
            pvals_adj_arr = np.recarray(n_out, dtype=f64_dt)
            logfc_arr = np.recarray(n_out, dtype=f32_dt)
            sort_kind = "stable"
            for i in range(n_groups):
                gn = groups_names[i]
                # Sort by descending score (matches vanilla scanpy's convention:
                # adata.uns['rank_genes_groups']['names'][group][:N] returns top-N).
                # Without this sort, the recarray stored names in var_names order
                # and downstream slicing returned random low-expression genes.
                # Honor rankby_abs here exactly as the n_genes (top-N) branch does
                # above; otherwise a full-output call with rankby_abs=True silently
                # ranked by signed score instead of |score|, diverging from vanilla.
                sort_key = np.abs(all_scores[i]) if rankby_abs else all_scores[i]
                sort_idx = np.argsort(-sort_key, kind=sort_kind)
                names_arr[gn] = var_names_np[sort_idx]
                scores_arr[gn] = all_scores[i][sort_idx]
                pvals_arr[gn] = all_pvals[i][sort_idx]
                pvals_adj_arr[gn] = all_pvals_adj[i][sort_idx]
                logfc_arr[gn] = all_logfc[i][sort_idx]

        adata.uns[key_added]["names"] = names_arr
        adata.uns[key_added]["scores"] = scores_arr
        adata.uns[key_added]["pvals"] = pvals_arr
        adata.uns[key_added]["pvals_adj"] = pvals_adj_arr
        adata.uns[key_added]["logfoldchanges"] = logfc_arr

        if pts:
            masks = np.zeros((n_groups, n_obs), dtype=bool)
            masks[group_int[group_int >= 0], np.arange(n_obs)[group_int >= 0]] = True
            X_bool = (X != 0).astype(np.float64)
            gs = group_sizes[:, None].astype(np.float64)
            n_rest_arr = (n_cells - group_sizes)[:, None].astype(np.float64)
            group_nnz = np.zeros((n_groups, n_genes_total), dtype=np.float64)
            for gi in range(n_groups):
                group_nnz[gi] = X_bool[masks[gi]].sum(axis=0).A1 if isinstance(X, CSBase) else X_bool[masks[gi]].sum(axis=0)
            adata.uns[key_added]["pts"] = pd.DataFrame(
                (group_nnz / gs).T, index=var_names, columns=groups_names)
            overall_nnz = group_nnz.sum(axis=0, keepdims=True)
            adata.uns[key_added]["pts_rest"] = pd.DataFrame(
                ((overall_nnz - group_nnz) / n_rest_arr).T, index=var_names, columns=groups_names)

        return adata if copy else None
    else:
        # Dense fallback — use original (with patched class methods active)
        return _orig_rank_genes()(
            adata, groupby, groups=groups, reference=reference,
            n_genes=n_genes, method=method, corr_method=corr_method,
            tie_correct=tie_correct, pts=pts, key_added=key_added,
            copy=copy, use_raw=use_raw, layer=layer, rankby_abs=rankby_abs,
            mask_var=mask_var, **kwds,
        )
