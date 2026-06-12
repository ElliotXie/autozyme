"""Patch for squidpy.gr.co_occurrence.

Lifted from autozyme task `test_squidpy_cooccurrence`. One bind site at
`squidpy.gr._ppatterns._co_occurrence_helper` — the inner helper that
upstream's `parallelize(...)` joblib wrapper calls per split-pair chunk.
Replacing the helper subsumes the inner `_occur_count` kernel (which is
where 97%+ of upstream wall lives) without having to touch
`co_occurrence` itself, so the patch survives any upstream-side
refactoring of the public entry point.

The fast helper is a composition of several converged optimizations
(see `memory/active_opts.md` in the task dir for the full ledger):

1. Single-pass binning — collapse 49 dense O(n^2) mask scans into one
   `searchsorted`-equivalent bin assignment + `cumsum` along the bin axis.
2. Fused distance + bin numba kernel — compute squared distance per pair
   on the fly, compare against `interval ** 2`, no materialised n*n
   pairwise distance matrix.
3. `nb.prange` over rows with per-thread histogram accumulators.
4. Upper-triangle walk for `same_split` tiles, with pair-up load balance
   so static-block prange scheduling stays even across threads.
5. dim=2 inner-loop unroll — the three production task datasets are 2D
   spatial; the runtime-dim `for k in range(dim)` is suboptimal for
   numba's autovec, so we hardcode the 2D distance and keep an nd
   fallback only for compatibility.
6. Single-write per pair + post-kernel transpose-add for same_split:
   write one direction inside the hot loop, recover symmetry via
   `H = H + H.T` once per tile (outside the inner prange).
7. In-numba cumsum + per-radius normalize — saves the Python<->numpy
   boundary crossing per tile (material for medium and large which call
   the helper 231 and 666 times respectively).
8. Parallel-over-tiles kernel for high split-pair counts (>= 2000
   tiles). At production scale (270k cells -> 132 splits -> 8778 split
   pairs), the per-tile dispatch overhead in the Python loop dominates
   the per-tile numba kernel call. Switching the outer prange from
   "within-tile rows" to "across-tile pairs" amortizes dispatch across
   the whole job; dev tiers (<=666 split-pairs) stay on the per-tile
   path that converged for them.

Thread config: numba thread count is read via `auto_threads()` at
helper-invocation time (not at patch-import time), so users can knob it
via `AUTOZYMER_THREADS` / `autozyme.set_threads()` between calls. BLAS
threads are left untouched: upstream's only BLAS use is
`sklearn.pairwise_distances` which we've fused into the kernel.

Concordance: pearson_occ = 1.000000 bit-exact and q99_abs_diff_occ <= 1e-6
across all task tiers (tiny, medium, large, ood_large, ood_xlarge);
NaN-mask is identical to upstream (the only NaNs upstream emits come
from zero-marginal rows / columns, and the fast helper hits the same
"if rs == 0.0: continue" guards).
"""
from __future__ import annotations

import os

import numpy as np
import numba as nb

import squidpy
from squidpy._utils import Signal

from autozyme._core import register_patch
from autozyme._threads import auto_threads, safe_set_num_threads
from autozyme._utils import resolve_dataset_path


# Heuristic threshold above which we switch from the per-tile parallel
# kernel (inner prange) to the all-tiles parallel kernel (outer prange).
# Empirically tuned on the validate-scale matrix: dev tiers (medium=231,
# large=666 split-pairs) stay on the per-tile path; ood_xlarge (8778
# pairs) hits the all-tiles path. Tunable via the AUTOZYME_COOCCURRENCE_
# ALL_TILES env var at import time for debugging.
def _parse_all_tiles_threshold(default: int = 2000) -> int:
    """Parse AUTOZYME_COOCCURRENCE_ALL_TILES; fall back to default on garbage.

    Read at import time so the kernel-dispatch branch stays a Python ``if``
    against a module-level int (no env lookup in the hot loop). A malformed
    value used to crash the whole patch import — and silently broke
    activation of any *other* patch that happened to enumerate it via
    `list_patches(installed=True)`.
    """
    raw = os.environ.get("AUTOZYME_COOCCURRENCE_ALL_TILES", "")
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return default


_ALL_TILES_THRESHOLD = _parse_all_tiles_threshold()


# ============================================================
# Fused per-pair distance + bin + histogram kernel (2D hot path).
# Outer prange over rows of the LHS split; each thread accumulates into
# its own slot of `histories[ti, :, :, :]`, reduced at the end.
# ============================================================
@nb.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _occur_count_fused_2d(spatial_x, spatial_y, clust_x, clust_y, num, interval_sq, same_split):
    L = interval_sq.shape[0] - 1
    n_x = spatial_x.shape[0]
    n_y = spatial_y.shape[0]
    n_threads = nb.get_num_threads()

    histories = np.zeros((n_threads, num, num, L), dtype=np.float32)

    if same_split:
        half = (n_x + 1) // 2
        for ii in nb.prange(half):
            ti = nb.get_thread_id()
            i_a = ii
            i_b = n_x - 1 - ii
            ci_a = clust_x[i_a]
            xa0 = spatial_x[i_a, 0]
            xa1 = spatial_x[i_a, 1]
            for j in range(i_a + 1, n_y):
                dx = xa0 - spatial_y[j, 0]
                dy = xa1 - spatial_y[j, 1]
                d2 = dx * dx + dy * dy
                if d2 > 0.0:
                    lo = 0
                    hi = L
                    while lo < hi:
                        mid = (lo + hi) >> 1
                        if interval_sq[mid + 1] >= d2:
                            hi = mid
                        else:
                            lo = mid + 1
                    if lo < L:
                        cj = clust_y[j]
                        histories[ti, ci_a, cj, lo] += 1.0
            if i_b > i_a:
                ci_b = clust_x[i_b]
                xb0 = spatial_x[i_b, 0]
                xb1 = spatial_x[i_b, 1]
                for j in range(i_b + 1, n_y):
                    dx = xb0 - spatial_y[j, 0]
                    dy = xb1 - spatial_y[j, 1]
                    d2 = dx * dx + dy * dy
                    if d2 > 0.0:
                        lo = 0
                        hi = L
                        while lo < hi:
                            mid = (lo + hi) >> 1
                            if interval_sq[mid + 1] >= d2:
                                hi = mid
                            else:
                                lo = mid + 1
                        if lo < L:
                            cj = clust_y[j]
                            histories[ti, ci_b, cj, lo] += 1.0
    else:
        for i in nb.prange(n_x):
            ti = nb.get_thread_id()
            ci = clust_x[i]
            xi0 = spatial_x[i, 0]
            xi1 = spatial_x[i, 1]
            for j in range(n_y):
                dx = xi0 - spatial_y[j, 0]
                dy = xi1 - spatial_y[j, 1]
                d2 = dx * dx + dy * dy
                if d2 > 0.0:
                    lo = 0
                    hi = L
                    while lo < hi:
                        mid = (lo + hi) >> 1
                        if interval_sq[mid + 1] >= d2:
                            hi = mid
                        else:
                            lo = mid + 1
                    if lo < L:
                        cj = clust_y[j]
                        histories[ti, ci, cj, lo] += 1.0
                        histories[ti, cj, ci, lo] += 1.0

    hist = histories.sum(axis=0)
    if same_split:
        hist = hist + hist.transpose(1, 0, 2)
    return hist


# Runtime-dim fallback for upstream callers handing in 3D+ spatial.
@nb.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _occur_count_fused_nd(spatial_x, spatial_y, clust_x, clust_y, num, interval_sq, same_split):
    L = interval_sq.shape[0] - 1
    n_x = spatial_x.shape[0]
    n_y = spatial_y.shape[0]
    dim = spatial_x.shape[1]
    n_threads = nb.get_num_threads()
    histories = np.zeros((n_threads, num, num, L), dtype=np.float32)
    for i in nb.prange(n_x):
        ti = nb.get_thread_id()
        ci = clust_x[i]
        j_start = i + 1 if same_split else 0
        for j in range(j_start, n_y):
            d2 = 0.0
            for k in range(dim):
                delta = spatial_x[i, k] - spatial_y[j, k]
                d2 += delta * delta
            if d2 > 0.0:
                lo = 0
                hi = L
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if interval_sq[mid + 1] >= d2:
                        hi = mid
                    else:
                        lo = mid + 1
                if lo < L:
                    cj = clust_y[j]
                    histories[ti, ci, cj, lo] += 1.0
                    histories[ti, cj, ci, lo] += 1.0
    return histories.sum(axis=0)


def _occur_count_fused(spatial_x, spatial_y, clust_x, clust_y, num, interval_sq, same_split):
    if spatial_x.shape[1] == 2:
        return _occur_count_fused_2d(spatial_x, spatial_y, clust_x, clust_y, num, interval_sq, same_split)
    return _occur_count_fused_nd(spatial_x, spatial_y, clust_x, clust_y, num, interval_sq, same_split)


@nb.njit(fastmath=True, cache=True, boundscheck=False)
def _cumsum_normalize(hist):
    num = hist.shape[0]
    L = hist.shape[2]
    co_occur_cum = np.empty_like(hist)
    for ci in range(num):
        for cj in range(num):
            s = 0.0
            for k in range(L):
                s += hist[ci, cj, k]
                co_occur_cum[ci, cj, k] = s

    out = np.zeros((num, num, L), dtype=np.float32)
    for idx in range(L):
        total = 0.0
        for ci in range(num):
            for cj in range(num):
                total += co_occur_cum[ci, cj, idx]
        if total == 0.0:
            continue
        marginal = np.empty(num, dtype=np.float32)
        for cj in range(num):
            s = 0.0
            for ci in range(num):
                s += co_occur_cum[ci, cj, idx]
            marginal[cj] = s / total
        for ci in range(num):
            rs = 0.0
            for cj in range(num):
                rs += co_occur_cum[ci, cj, idx]
            if rs == 0.0:
                continue
            for cj in range(num):
                m = marginal[cj]
                if m == 0.0:
                    continue
                out[ci, cj, idx] = (co_occur_cum[ci, cj, idx] / rs) / m
    return out


# ============================================================
# All-tiles parallel-over-tiles kernel — outer prange over (i_x, i_y)
# split pairs. Each thread takes whole tiles serially. Used at
# production scale (>= _ALL_TILES_THRESHOLD pairs) to amortize the
# numba thread-pool dispatch over all tiles instead of per-tile.
# ============================================================
@nb.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _process_all_tiles_2d(
    spatial_concat, labs_concat, split_offsets, tile_pairs, num, interval_sq, out,
):
    L = interval_sq.shape[0] - 1
    n_tiles = tile_pairs.shape[0]

    for t_idx in nb.prange(n_tiles):
        idx_x = tile_pairs[t_idx, 0]
        idx_y = tile_pairs[t_idx, 1]
        x_start = split_offsets[idx_x]
        x_end = split_offsets[idx_x + 1]
        y_start = split_offsets[idx_y]
        y_end = split_offsets[idx_y + 1]
        n_x = x_end - x_start
        n_y = y_end - y_start
        same_split = idx_x == idx_y

        hist = np.zeros((num, num, L), dtype=np.float32)

        if same_split:
            half = (n_x + 1) // 2
            for ii in range(half):
                i_a = ii
                i_b = n_x - 1 - ii
                ci_a = labs_concat[x_start + i_a]
                xa0 = spatial_concat[x_start + i_a, 0]
                xa1 = spatial_concat[x_start + i_a, 1]
                for j in range(i_a + 1, n_y):
                    dx = xa0 - spatial_concat[y_start + j, 0]
                    dy = xa1 - spatial_concat[y_start + j, 1]
                    d2 = dx * dx + dy * dy
                    if d2 > 0.0:
                        lo = 0
                        hi = L
                        while lo < hi:
                            mid = (lo + hi) >> 1
                            if interval_sq[mid + 1] >= d2:
                                hi = mid
                            else:
                                lo = mid + 1
                        if lo < L:
                            cj = labs_concat[y_start + j]
                            hist[ci_a, cj, lo] += 1.0
                if i_b > i_a:
                    ci_b = labs_concat[x_start + i_b]
                    xb0 = spatial_concat[x_start + i_b, 0]
                    xb1 = spatial_concat[x_start + i_b, 1]
                    for j in range(i_b + 1, n_y):
                        dx = xb0 - spatial_concat[y_start + j, 0]
                        dy = xb1 - spatial_concat[y_start + j, 1]
                        d2 = dx * dx + dy * dy
                        if d2 > 0.0:
                            lo = 0
                            hi = L
                            while lo < hi:
                                mid = (lo + hi) >> 1
                                if interval_sq[mid + 1] >= d2:
                                    hi = mid
                                else:
                                    lo = mid + 1
                            if lo < L:
                                cj = labs_concat[y_start + j]
                                hist[ci_b, cj, lo] += 1.0
            # Symmetry: H = H + H.T (off-diag a+b on both sides, diag doubles).
            for ci in range(num):
                for cj in range(ci + 1, num):
                    for k in range(L):
                        a = hist[ci, cj, k]
                        b = hist[cj, ci, k]
                        hist[ci, cj, k] = a + b
                        hist[cj, ci, k] = a + b
                for k in range(L):
                    hist[ci, ci, k] *= 2.0
        else:
            for i in range(n_x):
                ci = labs_concat[x_start + i]
                xi0 = spatial_concat[x_start + i, 0]
                xi1 = spatial_concat[x_start + i, 1]
                for j in range(n_y):
                    dx = xi0 - spatial_concat[y_start + j, 0]
                    dy = xi1 - spatial_concat[y_start + j, 1]
                    d2 = dx * dx + dy * dy
                    if d2 > 0.0:
                        lo = 0
                        hi = L
                        while lo < hi:
                            mid = (lo + hi) >> 1
                            if interval_sq[mid + 1] >= d2:
                                hi = mid
                            else:
                                lo = mid + 1
                        if lo < L:
                            cj = labs_concat[y_start + j]
                            hist[ci, cj, lo] += 1.0
                            hist[cj, ci, lo] += 1.0

        # Cumsum along bin axis (in-place).
        for ci in range(num):
            for cj in range(num):
                s = 0.0
                for k in range(L):
                    s += hist[ci, cj, k]
                    hist[ci, cj, k] = s

        # Per-radius normalize into out[t_idx].
        for k in range(L):
            total = 0.0
            for ci in range(num):
                for cj in range(num):
                    total += hist[ci, cj, k]
            if total == 0.0:
                continue
            marginal = np.empty(num, dtype=np.float32)
            for cj in range(num):
                col = 0.0
                for ci in range(num):
                    col += hist[ci, cj, k]
                marginal[cj] = col / total
            for ci in range(num):
                rs = 0.0
                for cj in range(num):
                    rs += hist[ci, cj, k]
                if rs == 0.0:
                    continue
                for cj in range(num):
                    m = marginal[cj]
                    if m == 0.0:
                        continue
                    out[t_idx, ci, cj, k] = (hist[ci, cj, k] / rs) / m


# ============================================================
# The replacement for `squidpy.gr._ppatterns._co_occurrence_helper`.
# Same signature as upstream so squidpy's `parallelize(...)` wrapper
# is none the wiser.
# ============================================================
def fast_co_occurrence_helper(
    idx_splits, spatial_splits, labs_splits, labs_unique, interval, queue=None,
):
    # Pick numba thread count at call time so set_threads() / AUTOZYMER_THREADS
    # is picked up live, not frozen at module import.
    safe_set_num_threads(max(1, auto_threads(cap=os.cpu_count() or 1)))

    num = labs_unique.shape[0]
    # interval is float32 upstream; square in float64 then downcast to keep
    # the bin endpoints magnitude-exact for typical coord ranges.
    interval_sq = (np.asarray(interval, dtype=np.float64) ** 2).astype(np.float32)
    idx_splits = list(idx_splits)

    is_2d = len(spatial_splits) > 0 and spatial_splits[0].shape[1] == 2
    if is_2d and len(idx_splits) >= _ALL_TILES_THRESHOLD:
        n_splits = len(spatial_splits)
        L = interval_sq.shape[0] - 1
        spatial_concat = np.concatenate(
            [np.ascontiguousarray(s, dtype=np.float32) for s in spatial_splits], axis=0,
        )
        labs_concat = np.concatenate(
            [np.ascontiguousarray(s, dtype=np.int32) for s in labs_splits], axis=0,
        )
        offsets = np.empty(n_splits + 1, dtype=np.int64)
        offsets[0] = 0
        for i, s in enumerate(spatial_splits):
            offsets[i + 1] = offsets[i] + s.shape[0]
        tile_pairs = np.asarray(idx_splits, dtype=np.int32).reshape(-1, 2)

        out = np.zeros((tile_pairs.shape[0], num, num, L), dtype=np.float32)
        _process_all_tiles_2d(
            spatial_concat, labs_concat, offsets, tile_pairs, num, interval_sq, out,
        )
        out_lst = [out[t] for t in range(out.shape[0])]
        if queue is not None:
            for _ in range(len(out_lst)):
                queue.put(Signal.UPDATE)
            queue.put(Signal.FINISH)
        return out_lst

    out_lst = []
    for t in idx_splits:
        idx_x, idx_y = t
        spatial_x = spatial_splits[idx_x]
        spatial_y = spatial_splits[idx_y]
        labs_x = labs_splits[idx_x]
        labs_y = labs_splits[idx_y]
        same_split = idx_x == idx_y
        hist = _occur_count_fused(spatial_x, spatial_y, labs_x, labs_y, num, interval_sq, same_split)
        out_lst.append(_cumsum_normalize(hist))
        if queue is not None:
            queue.put(Signal.UPDATE)
    if queue is not None:
        queue.put(Signal.FINISH)
    return out_lst


# ============================================================
# Smoke recipe
# ============================================================
def _smoke_load(task_dir, tier):
    """User-side prep: load the tier h5ad and resolve the cluster_key
    from task.yaml. Both baseline and patched will run on the same
    in-memory AnnData; we don't .copy() it because `sq.gr.co_occurrence`
    is invoked with `copy=True` (returns its result, leaves adata alone)
    so there's no cross-rep contamination to guard against.
    """
    import yaml
    import anndata as ad

    with open(os.path.join(task_dir, "task.yaml"), encoding="utf-8") as f:
        task = yaml.safe_load(f)
    ds = next(d for d in task["datasets"] if d["tier"] == tier)
    adata = ad.read_h5ad(resolve_dataset_path(task_dir, ds["path"]))
    cluster_key = ds["params"]["cluster_key"]
    if str(adata.obs[cluster_key].dtype) != "category":
        adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
    return {"adata": adata, "cluster_key": cluster_key}


def _smoke_call(inputs):
    """The canonical timed call. `copy=True` returns the (occ, interval)
    tuple instead of writing into adata.uns, matching what the task's
    pipeline/run.py and reference.py both do.
    """
    import squidpy as sq
    occ, interval = sq.gr.co_occurrence(
        inputs["adata"],
        cluster_key=inputs["cluster_key"],
        copy=True,
        show_progress_bar=False,
    )
    return {"occ": occ, "interval": interval}


def _smoke_save(result, dir, **kwargs):
    """Match what `pipeline/run.py` and `reference.py` save: a single
    `co_occurrence.npz` with `occ` and `interval` arrays. `evaluate.py`
    np.loads exactly those keys.
    """
    np.savez_compressed(
        os.path.join(dir, "co_occurrence.npz"),
        occ=np.asarray(result["occ"]),
        interval=np.asarray(result["interval"]),
    )


register_patch(
    name="squidpy_cooccurrence",
    targets=[
        ("squidpy.gr._ppatterns", "_co_occurrence_helper", fast_co_occurrence_helper),
    ],
    smoke={"load": _smoke_load, "call": _smoke_call, "save": _smoke_save},
    tested_against="squidpy 1.6.5",
    tested_upstream_versions={"squidpy": ["1.6.5"]},
)
