"""End-to-end / wrapper-line tests for autozyme.astropy_boxleastsquares.

Wave-1 (`test_astropy_boxleastsquares_unit.py`) tested `_thread_count` and
`_bls_chunk_edges` directly. The existing `test_astropy_boxleastsquares.py`
drives `BLS.power` on small grids — but a small grid keeps `fast_bls_fast` on
its single-worker short-circuit (`n_periods // 16384 == 0`), so the
ThreadPoolExecutor concat path never runs.

This file covers the COVERAGE-VISIBLE multi-worker dispatch in `fast_bls_fast`:
the chunk-edge partition, the ThreadPoolExecutor `executor.map`, and the
per-field `np.concatenate` reassembly — by lowering `_MIN_PERIODS_PER_WORKER`
so a modest grid splits across workers, and asserting parity vs the
single-worker (and vanilla) result.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("astropy")
from astropy.timeseries import BoxLeastSquares

import autozyme
from autozyme import astropy_boxleastsquares as azbls


@pytest.fixture
def bls_signal():
    rng = np.random.default_rng(0)
    n = 1500
    t = np.linspace(0, 80, n)
    P_true = 1.5
    phase = (t % P_true) / P_true
    transit = (phase > 0.45) & (phase < 0.5)
    y = 1.0 - 0.05 * transit.astype(float) + rng.normal(scale=0.005, size=n)
    dy = np.full(n, 0.005)
    return t, y, dy


def test_fast_bls_multiworker_matches_single_worker(bls_signal, monkeypatch):
    """Lowering _MIN_PERIODS_PER_WORKER + forcing several workers exercises the
    chunked ThreadPoolExecutor concat path; result must equal the single-worker
    path bit-for-bit (same upstream kernel, just sliced)."""
    t, y, dy = bls_signal
    periods = np.linspace(0.5, 5.0, 400)
    durations = np.array([0.05, 0.1])

    autozyme.activate("astropy_boxleastsquares")

    # Multi-worker: tiny per-worker minimum + >=2 workers.
    monkeypatch.setattr(azbls, "_MIN_PERIODS_PER_WORKER", 50)
    monkeypatch.setenv("ZYME_THREADS", "4")
    multi = BoxLeastSquares(t, y, dy).power(periods, durations).power

    # Single-worker short-circuit (the unpatched upstream kernel).
    monkeypatch.setattr(azbls, "_MIN_PERIODS_PER_WORKER", 10_000_000)
    single = BoxLeastSquares(t, y, dy).power(periods, durations).power

    assert len(multi) == len(periods)
    np.testing.assert_allclose(multi, single, rtol=1e-10, atol=1e-12)


def test_fast_bls_multiworker_matches_vanilla(bls_signal, monkeypatch):
    """The chunked multi-worker result matches a fully-disabled (vanilla)
    BLS.power on the same grid."""
    t, y, dy = bls_signal
    periods = np.linspace(0.6, 4.0, 300)
    durations = np.array([0.08])

    autozyme.activate("astropy_boxleastsquares")
    monkeypatch.setattr(azbls, "_MIN_PERIODS_PER_WORKER", 40)
    monkeypatch.setenv("ZYME_THREADS", "3")
    fast = BoxLeastSquares(t, y, dy).power(periods, durations).power

    with autozyme.disabled():
        ref = BoxLeastSquares(t, y, dy).power(periods, durations).power

    np.testing.assert_allclose(fast, ref, rtol=1e-9, atol=1e-11)


def test_fast_bls_all_result_fields_concatenated(bls_signal, monkeypatch):
    """Every BLSResults field (depth, duration, transit_time, ...) survives the
    per-field np.concatenate reassembly with the right length."""
    t, y, dy = bls_signal
    periods = np.linspace(0.5, 3.0, 256)
    durations = np.array([0.05, 0.1])

    autozyme.activate("astropy_boxleastsquares")
    monkeypatch.setattr(azbls, "_MIN_PERIODS_PER_WORKER", 32)
    monkeypatch.setenv("ZYME_THREADS", "4")
    res = BoxLeastSquares(t, y, dy).power(periods, durations)
    for attr in ("period", "power", "depth", "duration", "transit_time",
                 "depth_err", "depth_snr", "log_likelihood"):
        assert len(getattr(res, attr)) == len(periods), f"{attr} length mismatch"


def test_activate_restore_lifecycle():
    """activate binds bls_fast; deactivate restores the original."""
    from astropy.timeseries.periodograms.bls import methods as bls_methods

    autozyme.deactivate("astropy_boxleastsquares")
    original = bls_methods.bls_fast
    assert autozyme.activate("astropy_boxleastsquares") is True
    assert bls_methods.bls_fast is not original
    assert getattr(bls_methods.bls_fast, "__autozyme_fast__", None) is azbls.fast_bls_fast
    autozyme.deactivate("astropy_boxleastsquares")
    assert bls_methods.bls_fast is original
