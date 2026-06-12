"""Contract tests for the astropy.timeseries.BoxLeastSquares patch.

Patched surface (1 target):
  - astropy.timeseries.periodograms.bls.methods.bls_fast
    (the inner periodogram engine)

User-facing entry: ``BoxLeastSquares(t, y, dy).power(periods, durations)``.
The patched bls_fast accelerates the inner loop; vanilla and patched
must produce numerically equivalent BLSResults.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
astropy = pytest.importorskip("astropy")
from astropy.timeseries import BoxLeastSquares


@pytest.fixture
def bls_signal():
    """Synthetic transit signal: 100-day baseline, 1.5-day period, 5%
    transit depth, 4h transit duration. Enough signal that BLS recovers
    the period."""
    rng = np.random.default_rng(0)
    n = 2000
    t = np.linspace(0, 100, n)  # days
    P_true = 1.5
    depth = 0.05
    dur = 4.0 / 24.0  # 4 hours in days
    phase = (t % P_true) / P_true
    transit = (phase > 0.45) & (phase < 0.45 + dur / P_true)
    y = 1.0 - depth * transit.astype(float)
    y += rng.normal(scale=0.005, size=n)
    dy = np.full(n, 0.005)
    return t, y, dy


def test_bls_power_returns_blsresults(bls_signal):
    """BLS.power() returns BLSResults with period/power/duration arrays."""
    import autozyme
    autozyme.activate("astropy_boxleastsquares")

    t, y, dy = bls_signal
    bls = BoxLeastSquares(t, y, dy)
    periods = np.linspace(0.5, 5.0, 200)
    durations = np.array([0.1, 0.2])
    result = bls.power(periods, durations)

    for attr in ("period", "power", "duration", "depth"):
        assert hasattr(result, attr), f"BLSResults missing .{attr}"
    assert len(result.power) == len(periods)


def test_bls_power_zyme_false_matches_vanilla(bls_signal):
    """Patched power array matches vanilla within float tolerance."""
    import autozyme
    autozyme.activate("astropy_boxleastsquares")

    t, y, dy = bls_signal
    periods = np.linspace(0.5, 5.0, 200)
    durations = np.array([0.1, 0.2])

    bls = BoxLeastSquares(t, y, dy)
    fast = bls.power(periods, durations).power
    with autozyme.disabled():
        bls_v = BoxLeastSquares(t, y, dy)
        ref = bls_v.power(periods, durations).power

    np.testing.assert_allclose(fast, ref, rtol=1e-5, atol=1e-8,
                               err_msg="BLS power drift patched vs vanilla")


def test_bls_recovers_known_period(bls_signal):
    """End-to-end: BLS power peaks at the true 1.5-day injected period."""
    import autozyme
    autozyme.activate("astropy_boxleastsquares")

    t, y, dy = bls_signal
    periods = np.linspace(0.5, 5.0, 500)
    durations = np.array([0.1, 0.2])
    result = BoxLeastSquares(t, y, dy).power(periods, durations)
    peak_period = result.period[np.argmax(result.power)]
    # Within ~5% of true 1.5 days.
    assert abs(peak_period - 1.5) / 1.5 < 0.05, (
        f"BLS peak at {peak_period:.3f}, expected ~1.5"
    )
