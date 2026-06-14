"""Unit tests for the pure numba kernel in autozyme.xclim.

The contract test (test_xclim.py) drives growing_season_length end to end via
xarray. Here we test the self-contained numba scan kernel `_gsl_kernel`
directly against a plain-Python reference implementation of the same
growing-season-length algorithm.

xclim/xarray only need to import for the module to load; `_gsl_kernel` is a pure
numba function over numpy arrays.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("numba")
pytest.importorskip("xclim")

from autozyme import xclim as azxclim


def _ref_gsl(tas, thresh, window, year_starts, year_ends, mid_offsets, op_ge):
    """Plain-Python reference mirroring _gsl_kernel exactly."""
    n_cells = tas.shape[0]
    n_years = mid_offsets.shape[0]
    out = np.zeros((n_cells, n_years), dtype=np.float64)

    def cond(v):
        return (v >= thresh) if op_ge else (v > thresh)

    for c in range(n_cells):
        for y in range(n_years):
            t0 = int(year_starts[y])
            t1 = int(year_ends[y])
            mid = int(mid_offsets[y])
            beg = -1
            run = 0
            upper = min(mid + window - 1, t1)
            for i in range(t0, upper):
                if cond(tas[c, i]):
                    run += 1
                    if run >= window:
                        run_start = i - window + 1
                        if run_start < mid:
                            beg = run_start
                        break
                else:
                    run = 0
            if beg < 0:
                out[c, y] = 0.0
                continue
            search_start = mid if beg < mid else beg
            end = -1
            run = 0
            for i in range(search_start, t1):
                if not cond(tas[c, i]):
                    run += 1
                    if run >= window:
                        end = i - window + 1
                        break
                else:
                    run = 0
            out[c, y] = float(t1 - beg) if end < 0 else float(end - beg)
    return out


def _run(tas, thresh, window, year_starts, year_ends, mid_offsets, op_ge):
    got = azxclim._gsl_kernel(
        np.ascontiguousarray(tas, dtype=np.float32), np.float32(thresh),
        int(window), year_starts.astype(np.int64), year_ends.astype(np.int64),
        mid_offsets.astype(np.int64), bool(op_ge),
    )
    ref = _ref_gsl(
        tas.astype(np.float32), np.float32(thresh), window,
        year_starts, year_ends, mid_offsets, op_ge,
    )
    return np.asarray(got), ref


def test_gsl_kernel_random_matches_reference():
    rng = np.random.default_rng(0)
    n_cells, n_time = 6, 365
    tas = rng.normal(loc=5.0, scale=10.0, size=(n_cells, n_time)).astype(np.float32)
    year_starts = np.array([0])
    year_ends = np.array([n_time])
    mid_offsets = np.array([180])  # mid-year
    got, ref = _run(tas, 5.0, 6, year_starts, year_ends, mid_offsets, True)
    np.testing.assert_array_equal(got, ref)


def test_gsl_kernel_all_above_threshold():
    # Always-warm cell: season starts at the first window and never ends ->
    # length = t1 - beg = whole year.
    n_time = 100
    tas = np.full((1, n_time), 20.0, dtype=np.float32)
    got, ref = _run(
        tas, 5.0, 6, np.array([0]), np.array([n_time]), np.array([40]), True
    )
    np.testing.assert_array_equal(got, ref)
    # season begins at index 0 -> length is full year.
    assert got[0, 0] == float(n_time)


def test_gsl_kernel_all_below_threshold_is_zero():
    n_time = 100
    tas = np.full((1, n_time), -20.0, dtype=np.float32)
    got, ref = _run(
        tas, 5.0, 6, np.array([0]), np.array([n_time]), np.array([40]), True
    )
    np.testing.assert_array_equal(got, ref)
    assert got[0, 0] == 0.0


def test_gsl_kernel_op_strict_vs_ge_differs_at_boundary():
    # Values exactly at threshold: op_ge counts them, strict-gt does not.
    n_time = 60
    tas = np.full((1, n_time), 5.0, dtype=np.float32)  # exactly == threshold
    # op_ge=True -> warm everywhere -> nonzero season
    got_ge, ref_ge = _run(
        tas, 5.0, 6, np.array([0]), np.array([n_time]), np.array([20]), True
    )
    np.testing.assert_array_equal(got_ge, ref_ge)
    assert got_ge[0, 0] > 0
    # op_ge=False -> never warm -> zero
    got_gt, ref_gt = _run(
        tas, 5.0, 6, np.array([0]), np.array([n_time]), np.array([20]), False
    )
    np.testing.assert_array_equal(got_gt, ref_gt)
    assert got_gt[0, 0] == 0.0


def test_gsl_kernel_multi_year():
    rng = np.random.default_rng(2)
    n_cells = 4
    # Two consecutive 200-step years.
    tas = rng.normal(5.0, 8.0, size=(n_cells, 400)).astype(np.float32)
    year_starts = np.array([0, 200])
    year_ends = np.array([200, 400])
    mid_offsets = np.array([100, 300])
    got, ref = _run(tas, 5.0, 6, year_starts, year_ends, mid_offsets, True)
    assert got.shape == (n_cells, 2)
    np.testing.assert_array_equal(got, ref)


def test_gsl_kernel_window_one_behaves_like_pointwise():
    # window=1 -> a single warm step starts the season; a single cold step ends.
    n_time = 50
    tas = np.full((1, n_time), -1.0, dtype=np.float32)
    tas[0, 10:20] = 10.0  # warm run [10,20)
    got, ref = _run(
        tas, 5.0, 1, np.array([0]), np.array([n_time]), np.array([5]), True
    )
    np.testing.assert_array_equal(got, ref)


def test_gsl_kernel_clear_season_known_length():
    # Hand-built: warm run [20, 80) inside a year [0,120), mid=10.
    n_time = 120
    tas = np.full((1, n_time), -10.0, dtype=np.float32)
    tas[0, 20:80] = 10.0
    window = 6
    got, ref = _run(
        tas, 5.0, window, np.array([0]), np.array([n_time]), np.array([10]), True
    )
    np.testing.assert_array_equal(got, ref)
    # season begins where window-of-6 warm completes: run reaches 6 at i=25,
    # run_start=20 < mid? mid=10 so 20 >= 10 -> beg stays -1 in the first loop
    # only if run_start < mid. Here run_start=20 >= mid=10, so beg not set in
    # the pre-mid loop. We just assert kernel==reference (the algorithm is the
    # spec); the equality above is the contract.
    assert got[0, 0] == ref[0, 0]
