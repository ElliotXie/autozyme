"""Contract tests for the sarsen patch (Sentinel-1 SAR terrain correction).

Full terrain correction needs Sentinel-1 product + DEM fixtures. These
contracts cover representative pure-function hot paths on tiny xarray inputs:
DEM tensor construction and slant-range to ground-range conversion.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
pytest.importorskip("sarsen")
pytest.importorskip("xarray_sentinel")


@pytest.fixture(autouse=True)
def _activate_autozyme(request):
    """Override conftest's scanpy-activating autouse for the numba thread pool."""
    import autozyme

    try:
        autozyme.activate("sarsen")
    except RuntimeError as e:
        if "NUMBA_NUM_THREADS" in str(e):
            pytest.skip(f"numba thread pool already initialized: {e}")
        raise
    yield
    autozyme.deactivate_all()


def test_convert_to_dem_3d_matches_upstream_values():
    """Patched DEM tensor has the same plane values and coordinates."""
    import autozyme
    import sarsen.scene

    dem = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        dims=("y", "x"),
        coords={"y": [10.0, 20.0], "x": [30.0, 40.0]},
    )
    fast = sarsen.scene.convert_to_dem_3d(dem)
    with autozyme.disabled():
        ref = sarsen.scene.convert_to_dem_3d(dem)

    assert fast.dims == ref.dims
    assert list(fast.coords["axis"].values) == list(ref.coords["axis"].values)
    np.testing.assert_allclose(fast.values, ref.values, rtol=1e-7, atol=1e-7)


def test_slant_range_time_to_ground_range_matches_upstream():
    """Patched polynomial interpolation matches xarray-sentinel upstream."""
    import autozyme
    import xarray_sentinel

    source_time = np.array(
        ["2020-01-01T00:00:00", "2020-01-01T00:00:10"],
        dtype="datetime64[ns]",
    )
    degree = np.array([0, 1, 2])
    coordinate_conversion = xr.Dataset({
        "sr0": xr.DataArray(
            [100.0, 120.0],
            dims=("azimuth_time",),
            coords={"azimuth_time": source_time},
        ),
        "srgrCoefficients": xr.DataArray(
            [[10.0, 1.5, 0.01], [11.0, 1.4, 0.02]],
            dims=("azimuth_time", "degree"),
            coords={"azimuth_time": source_time, "degree": degree},
        ),
    })
    azimuth_time = xr.DataArray(
        np.array(
            [
                ["2020-01-01T00:00:00", "2020-01-01T00:00:10"],
                ["2020-01-01T00:00:10", "2020-01-01T00:00:00"],
            ],
            dtype="datetime64[ns]",
        ),
        dims=("y", "x"),
    )
    slant_range_time = xr.DataArray(
        np.array([[1.0e-6, 1.1e-6], [1.2e-6, 1.3e-6]]),
        dims=("y", "x"),
    )

    fast = xarray_sentinel.slant_range_time_to_ground_range(
        azimuth_time, slant_range_time, coordinate_conversion
    )
    with autozyme.disabled():
        ref = xarray_sentinel.slant_range_time_to_ground_range(
            azimuth_time, slant_range_time, coordinate_conversion
        )
    np.testing.assert_allclose(fast.values, ref.values, rtol=1e-12, atol=1e-12)
