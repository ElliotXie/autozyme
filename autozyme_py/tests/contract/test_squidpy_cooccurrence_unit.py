"""Unit tests for the pure numba/numpy kernels in autozyme.squidpy_cooccurrence.

The existing contract test (test_squidpy_cooccurrence.py) drives the patch end
to end through squidpy. This file exercises the self-contained kernels directly
on small fixed inputs, checking them against an independent numpy reference:

  - _parse_all_tiles_threshold  (env parsing)
  - _occur_count_fused_2d / _nd / _occur_count_fused  (fused distance+bin histogram)
  - _cumsum_normalize  (per-radius co-occurrence normalisation)
  - _process_all_tiles_2d  (all-tiles parallel kernel: hist + cumsum + normalise)

squidpy itself only needs to import for the module to load; none of the kernels
below call into squidpy at runtime.
"""
from __future__ import annotations

import os

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("numba")
# The module imports squidpy at top level; skip cleanly if it is absent.
pytest.importorskip("squidpy")

from autozyme import squidpy_cooccurrence as cooc


# --------------------------------------------------------------------------
# numpy reference for one tile's histogram (bin assignment by interval edges)
# --------------------------------------------------------------------------
def _ref_hist(spatial_x, spatial_y, clust_x, clust_y, num, interval_sq, same_split):
    L = interval_sq.shape[0] - 1
    hist = np.zeros((num, num, L), dtype=np.float64)
    n_x = spatial_x.shape[0]
    n_y = spatial_y.shape[0]
    for i in range(n_x):
        j0 = i + 1 if same_split else 0
        for j in range(j0, n_y):
            d2 = float(np.sum((spatial_x[i] - spatial_y[j]) ** 2))
            if d2 <= 0.0:
                continue
            # bin = first index lo in [0, L) with interval_sq[lo+1] >= d2
            lo = int(np.searchsorted(interval_sq[1:], d2, side="left"))
            if lo < L:
                hist[clust_x[i], clust_y[j], lo] += 1.0
                if not same_split:
                    hist[clust_y[j], clust_x[i], lo] += 1.0
    if same_split:
        hist = hist + hist.transpose(1, 0, 2)
    return hist


def _ref_cumsum_normalize(hist):
    num, _, L = hist.shape
    cum = np.cumsum(hist, axis=2)
    out = np.zeros_like(hist, dtype=np.float64)
    for idx in range(L):
        total = cum[:, :, idx].sum()
        if total == 0.0:
            continue
        marginal = cum[:, :, idx].sum(axis=0) / total  # per cj
        for ci in range(num):
            rs = cum[ci, :, idx].sum()
            if rs == 0.0:
                continue
            for cj in range(num):
                m = marginal[cj]
                if m == 0.0:
                    continue
                out[ci, cj, idx] = (cum[ci, cj, idx] / rs) / m
    return out


def _fixture(seed=0, n=12, num=3, same_split=False):
    rng = np.random.default_rng(seed)
    spatial = rng.random((n, 2)).astype(np.float32) * 5.0
    clust = rng.integers(0, num, size=n).astype(np.int32)
    interval = np.linspace(0.0, 8.0, 6).astype(np.float64)
    interval_sq = (interval ** 2).astype(np.float32)
    return spatial, clust, num, interval_sq


# --------------------------------------------------------------------------
# _parse_all_tiles_threshold
# --------------------------------------------------------------------------
def test_parse_threshold_default_when_unset(monkeypatch):
    monkeypatch.delenv("AUTOZYME_COOCCURRENCE_ALL_TILES", raising=False)
    assert cooc._parse_all_tiles_threshold(default=2000) == 2000
    assert cooc._parse_all_tiles_threshold(default=7) == 7


def test_parse_threshold_valid_value(monkeypatch):
    monkeypatch.setenv("AUTOZYME_COOCCURRENCE_ALL_TILES", "512")
    assert cooc._parse_all_tiles_threshold() == 512


def test_parse_threshold_garbage_falls_back(monkeypatch):
    for bad in ("abc", "0", "-5", ""):
        monkeypatch.setenv("AUTOZYME_COOCCURRENCE_ALL_TILES", bad)
        assert cooc._parse_all_tiles_threshold(default=99) == 99


# --------------------------------------------------------------------------
# _occur_count_fused_2d / _nd vs numpy reference
# --------------------------------------------------------------------------
@pytest.mark.parametrize("same_split", [False, True])
def test_occur_count_fused_2d_matches_reference(same_split):
    spatial, clust, num, interval_sq = _fixture(seed=1, n=14, num=3)
    got = cooc._occur_count_fused_2d(
        spatial, spatial, clust, clust, num, interval_sq, same_split
    )
    ref = _ref_hist(spatial, spatial, clust, clust, num, interval_sq, same_split)
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64), ref, rtol=0, atol=0)


@pytest.mark.parametrize("same_split", [False, True])
def test_occur_count_fused_2d_cross_split(same_split):
    # Distinct LHS / RHS splits: only same_split=False is a meaningful cross
    # tile, but the kernel must still run for both flags.
    sx, cx, num, iv = _fixture(seed=2, n=9, num=4)
    sy, cy, _, _ = _fixture(seed=3, n=11, num=4)
    if same_split:
        sy, cy = sx, cx  # same_split path requires identical x/y
    got = cooc._occur_count_fused_2d(sx, sy, cx, cy, num, iv, same_split)
    ref = _ref_hist(sx, sy, cx, cy, num, iv, same_split)
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64), ref)


def test_occur_count_fused_nd_matches_2d_dispatch():
    # 3D coords route through the nd kernel; verify against the reference.
    rng = np.random.default_rng(5)
    n, num = 10, 3
    spatial = (rng.random((n, 3)) * 4.0).astype(np.float32)
    clust = rng.integers(0, num, size=n).astype(np.int32)
    iv = (np.linspace(0, 6, 5) ** 2).astype(np.float32)
    got = cooc._occur_count_fused(spatial, spatial, clust, clust, num, iv, False)
    ref = _ref_hist(spatial, spatial, clust, clust, num, iv, False)
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64), ref)


def test_occur_count_fused_dispatch_picks_2d():
    # _occur_count_fused dispatches on shape[1] == 2.
    spatial, clust, num, iv = _fixture(seed=7, n=8, num=2)
    a = cooc._occur_count_fused(spatial, spatial, clust, clust, num, iv, False)
    b = cooc._occur_count_fused_2d(spatial, spatial, clust, clust, num, iv, False)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_occur_count_total_pairs_conserved():
    # For a cross-split (no self pairs), the histogram should count every
    # in-range ordered pair exactly twice (i->j and j->i).
    sx, cx, num, iv = _fixture(seed=9, n=8, num=3)
    sy, cy, _, _ = _fixture(seed=10, n=7, num=3)
    hist = np.asarray(
        cooc._occur_count_fused_2d(sx, sy, cx, cy, num, iv, False), dtype=np.float64
    )
    # Manual count of in-range pairs.
    L = iv.shape[0] - 1
    n_pairs = 0
    for i in range(sx.shape[0]):
        for j in range(sy.shape[0]):
            d2 = float(np.sum((sx[i] - sy[j]) ** 2))
            if d2 > 0.0 and int(np.searchsorted(iv[1:], d2, side="left")) < L:
                n_pairs += 1
    assert hist.sum() == pytest.approx(2.0 * n_pairs)


# --------------------------------------------------------------------------
# _cumsum_normalize
# --------------------------------------------------------------------------
def test_cumsum_normalize_matches_reference():
    spatial, clust, num, iv = _fixture(seed=11, n=16, num=3)
    hist = np.asarray(
        cooc._occur_count_fused_2d(spatial, spatial, clust, clust, num, iv, False),
        dtype=np.float32,
    )
    got = np.asarray(cooc._cumsum_normalize(hist), dtype=np.float64)
    ref = _ref_cumsum_normalize(hist.astype(np.float64))
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_cumsum_normalize_zero_hist_is_zero():
    hist = np.zeros((3, 3, 4), dtype=np.float32)
    out = np.asarray(cooc._cumsum_normalize(hist))
    assert np.all(out == 0.0)


# --------------------------------------------------------------------------
# _process_all_tiles_2d  (the production-scale all-tiles kernel)
# --------------------------------------------------------------------------
def test_process_all_tiles_matches_per_tile_path():
    rng = np.random.default_rng(21)
    num = 3
    # Two splits.
    s0 = (rng.random((8, 2)) * 5).astype(np.float32)
    s1 = (rng.random((9, 2)) * 5).astype(np.float32)
    c0 = rng.integers(0, num, size=8).astype(np.int32)
    c1 = rng.integers(0, num, size=9).astype(np.int32)
    spatial_splits = [s0, s1]
    labs_splits = [c0, c1]
    iv = (np.linspace(0, 7, 6) ** 2).astype(np.float32)
    L = iv.shape[0] - 1

    spatial_concat = np.concatenate(spatial_splits, axis=0).astype(np.float32)
    labs_concat = np.concatenate(labs_splits, axis=0).astype(np.int32)
    offsets = np.array([0, 8, 17], dtype=np.int64)
    # All tile pairs (same and cross).
    tile_pairs = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int32)
    out = np.zeros((tile_pairs.shape[0], num, num, L), dtype=np.float32)
    cooc._process_all_tiles_2d(
        spatial_concat, labs_concat, offsets, tile_pairs, num, iv, out
    )

    # Reference: per-tile hist -> cumsum-normalise.
    for t, (ix, iy) in enumerate(tile_pairs):
        sx, sy = spatial_splits[ix], spatial_splits[iy]
        cx, cy = labs_splits[ix], labs_splits[iy]
        same = bool(ix == iy)
        ref_hist = _ref_hist(sx, sy, cx, cy, num, iv, same)
        ref = _ref_cumsum_normalize(ref_hist)
        np.testing.assert_allclose(
            np.asarray(out[t], dtype=np.float64), ref, rtol=1e-5, atol=1e-6,
            err_msg=f"tile {t} ({ix},{iy}) mismatch",
        )
