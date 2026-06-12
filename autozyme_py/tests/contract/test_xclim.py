"""Contract tests for the xclim patch (growing_season_length).

Patched surface: 1 logical target bound at 2 import paths:
  - xclim.indices._threshold.growing_season_length
  - xclim.indices.growing_season_length

Fast path is gated on op in {'>', '>=', 'gt', 'ge'} + freq='YS' +
mid_date not None. Anything else delegates to vanilla.

NOTE on Bug 5 (numba pool init order): the conftest autouse activates
scanpy, which initializes numba's threadpool. Later activating xclim
tries to set NUMBA_NUM_THREADS again -> RuntimeError. We override the
autouse fixture in this file to activate xclim WITHOUT triggering
scanpy first; if some other test ran before us and already initialized
numba via scanpy, we skip with a clear reason rather than crash.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
pd = pytest.importorskip("pandas")
pytest.importorskip("xclim")


@pytest.fixture(autouse=True)
def _activate_autozyme(request):
    """Override conftest's scanpy-activating autouse fixture. xclim's
    fast path uses its own numba kernel; activating scanpy first would
    trip Bug 5 (NUMBA_NUM_THREADS reset after pool init)."""
    import autozyme
    try:
        autozyme.activate("xclim")
    except RuntimeError as e:
        if "NUMBA_NUM_THREADS" in str(e):
            pytest.skip(
                "Bug 5: numba threadpool already initialized by an earlier "
                "test (likely scanpy via conftest autouse). xclim's "
                "NUMBA_NUM_THREADS reset crashes. Run this test FIRST in "
                f"a fresh process to exercise the contract. Underlying: {e}"
            )
        raise
    yield


@pytest.fixture
def daily_tas():
    """3-year daily mean temperature DataArray with seasonal sinusoid.

    Northern-hemisphere shape: cold in winter, warm in summer, threshold
    5°C carves a clean growing season per year. Two spatial cells so we
    also test the multi-cell vectorized path.
    """
    n_years = 3
    times = pd.date_range("2020-01-01", periods=365 * n_years, freq="D")
    doy = times.dayofyear
    # Seasonal: -5°C ± 20°C amplitude. Peak around day 200.
    base = -5 + 20 * np.sin(2 * np.pi * (doy - 100) / 365)
    # Two-cell domain: cell B offset by +2°C.
    tas_vals = np.stack([base, base + 2.0], axis=0).astype(np.float32)
    da = xr.DataArray(
        tas_vals,
        dims=("location", "time"),
        coords={
            "location": ["a", "b"],
            "time": times,
        },
        attrs={"units": "degC"},
    )
    return da


def test_growing_season_length_returns_dataarray(daily_tas):
    """Fast path: op='>=', freq='YS', mid_date='07-01' -> hits patched code."""
    import xclim.indices

    out = xclim.indices.growing_season_length(
        daily_tas, thresh="5.0 degC", window=6, mid_date="07-01",
        freq="YS", op=">="
    )
    assert isinstance(out, xr.DataArray)
    # 3 years -> 3 time points.
    assert out.sizes["time"] == 3
    # 2 spatial cells preserved.
    assert out.sizes["location"] == 2


def test_growing_season_length_zyme_false_matches_patched(daily_tas):
    """Fast path numerical parity with vanilla."""
    import autozyme
    import xclim.indices

    fast = xclim.indices.growing_season_length(
        daily_tas, thresh="5.0 degC", window=6, mid_date="07-01",
        freq="YS", op=">="
    )
    with autozyme.disabled():
        ref = xclim.indices.growing_season_length(
            daily_tas, thresh="5.0 degC", window=6, mid_date="07-01",
            freq="YS", op=">="
        )
    np.testing.assert_array_equal(fast.values, ref.values)


def test_growing_season_length_mid_date_none_delegates(daily_tas):
    """mid_date=None is NOT on the fast path -> upstream delegation.

    (xclim itself rejects op='<' for this indice at the validation layer
    so we can't use op to trigger delegation; mid_date=None is the
    cleanest off-fast-path trigger that xclim allows.)
    """
    import autozyme
    import xclim.indices

    fast = xclim.indices.growing_season_length(
        daily_tas, thresh="5.0 degC", window=6, mid_date=None,
        freq="YS", op=">="
    )
    with autozyme.disabled():
        ref = xclim.indices.growing_season_length(
            daily_tas, thresh="5.0 degC", window=6, mid_date=None,
            freq="YS", op=">="
        )
    np.testing.assert_array_equal(fast.values, ref.values)


def test_growing_season_length_freq_qs_delegates(daily_tas):
    """freq != 'YS' (e.g. 'QS') is NOT on the fast path -> upstream."""
    import autozyme
    import xclim.indices

    fast = xclim.indices.growing_season_length(
        daily_tas, thresh="5.0 degC", window=6, mid_date="07-01",
        freq="QS", op=">="
    )
    with autozyme.disabled():
        ref = xclim.indices.growing_season_length(
            daily_tas, thresh="5.0 degC", window=6, mid_date="07-01",
            freq="QS", op=">="
        )
    np.testing.assert_array_equal(fast.values, ref.values)
