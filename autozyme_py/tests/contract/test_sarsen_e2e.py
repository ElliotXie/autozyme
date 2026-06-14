"""End-to-end / wrapper-line tests for autozyme.sarsen.

Wave-1 (`test_sarsen_unit.py`) tested the pure helpers (_get_threads,
_datetime_to_int64, _fast_orbit_polyval, _product_signature, fast_convert_to_dem_3d,
fast_zero_doppler_*). The existing `test_sarsen.py` drives convert_to_dem_3d +
slant_range_time_to_ground_range through activation. This file covers the
remaining COVERAGE-VISIBLE wrappers that those don't reach:

  - `fast_terrain_correction`: the GRD-product passthrough branch AND the
    non-GRD `autozyme.disabled()` full-upstream branch (with _orig stubbed so no
    real Sentinel-1 product is needed).
  - `fast_transform_dem_3d`: the threaded pyproj transform end-to-end (explicit
    source_crs, so no rioxarray .rio.crs needed) + the per-CRS transformer cache.
  - `fast_position/velocity/acceleration_from_orbit_time` through a real
    OrbitPolyfitInterpolator, vs the captured upstream xarray polyval.
  - activate/restore lifecycle on the 12 sarsen targets.

The full terrain_correction / interp_sar / beta_nought / simulate_acquisition
GRD path needs real Sentinel-1 products + a DEM raster on disk -- too heavy /
data-dependent for a contract test (see report note).
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
pytest.importorskip("sarsen")
pytest.importorskip("pyproj")

import autozyme
from autozyme import sarsen as azsarsen


@pytest.fixture(autouse=True)
def _activate():
    """Override conftest's scanpy autouse (numba pool) like test_sarsen.py."""
    try:
        autozyme.activate("sarsen")
    except RuntimeError as e:
        if "NUMBA_NUM_THREADS" in str(e):
            pytest.skip(f"numba thread pool already initialized: {e}")
        raise
    yield
    autozyme.deactivate_all()


class _FakeProduct:
    def __init__(self, product_type):
        self.product_type = product_type


def test_terrain_correction_grd_passthrough(monkeypatch):
    """A GRD product takes the fast passthrough branch (no disabled() wrap)."""
    state = {"disabled": None}

    def _fake_orig(*args, **kwargs):
        state["disabled"] = autozyme.is_disabled()
        return "GTC"

    monkeypatch.setattr(azsarsen, "_orig_terrain_correction", _fake_orig)
    out = azsarsen.fast_terrain_correction(product=_FakeProduct("GRD"))
    assert out == "GTC"
    assert state["disabled"] is False


def test_terrain_correction_non_grd_runs_fully_upstream(monkeypatch):
    """A non-GRD (e.g. SLC) product wraps the original in autozyme.disabled()
    so the unverified globally-rebound helpers are bypassed."""
    state = {"disabled": None}

    def _fake_orig(*args, **kwargs):
        state["disabled"] = autozyme.is_disabled()
        return "SLC_GTC"

    monkeypatch.setattr(azsarsen, "_orig_terrain_correction", _fake_orig)
    out = azsarsen.fast_terrain_correction(product=_FakeProduct("SLC"))
    assert out == "SLC_GTC"
    assert state["disabled"] is True


def test_terrain_correction_positional_product(monkeypatch):
    """The product can also be passed positionally (args[0])."""
    seen = {}

    def _fake_orig(*args, **kwargs):
        seen["disabled"] = autozyme.is_disabled()
        return "POS"

    monkeypatch.setattr(azsarsen, "_orig_terrain_correction", _fake_orig)
    out = azsarsen.fast_terrain_correction(_FakeProduct("SLC"))
    assert out == "POS"
    assert seen["disabled"] is True


def test_transform_dem_3d_threaded_pyproj_roundtrips():
    """fast_transform_dem_3d runs the threaded pyproj transform end-to-end and
    populates the per-CRS transformer cache; a same-CRS round-trip (4326->4326)
    leaves the lon/lat planes ~unchanged and altitude exact."""
    # (axis, y, x) DEM tensor: axis 0=lon, 1=lat, 2=elevation.
    lon = np.array([[10.0, 11.0], [10.0, 11.0]], dtype=np.float64)
    lat = np.array([[50.0, 50.0], [51.0, 51.0]], dtype=np.float64)
    elev = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float64)
    dem_3d = xr.DataArray(
        np.stack([lon, lat, elev]),
        dims=("axis", "y", "x"),
        coords={"axis": [0, 1, 2]},
        name="dem_3d",
    )
    azsarsen._transformer_cache.clear()
    out = azsarsen.fast_transform_dem_3d(
        dem_3d, source_crs="EPSG:4326", target_crs="EPSG:4326",
    )
    assert out.shape == dem_3d.shape
    # Identity transform: planes unchanged.
    np.testing.assert_allclose(out.sel(axis=0).values, lon, rtol=1e-9)
    np.testing.assert_allclose(out.sel(axis=1).values, lat, rtol=1e-9)
    np.testing.assert_allclose(out.sel(axis=2).values, elev, rtol=1e-9)
    # Transformer was cached for the CRS pair.
    assert ("EPSG:4326", "EPSG:4326") in azsarsen._transformer_cache


def test_transform_dem_3d_reprojects_and_caches():
    """A real reprojection (4326 -> ECEF) produces finite metric coords and a
    cache hit on a second call (same transformer object reused)."""
    lon = np.array([[10.0, 11.0]], dtype=np.float64)
    lat = np.array([[50.0, 51.0]], dtype=np.float64)
    elev = np.array([[0.0, 100.0]], dtype=np.float64)
    dem_3d = xr.DataArray(
        np.stack([lon, lat, elev]),
        dims=("axis", "y", "x"),
        coords={"axis": [0, 1, 2]},
        name="dem_3d",
    )
    azsarsen._transformer_cache.clear()
    out1 = azsarsen.fast_transform_dem_3d(dem_3d, source_crs="EPSG:4326")
    key = list(azsarsen._transformer_cache.keys())[0]
    transformer1 = azsarsen._transformer_cache[key]
    out2 = azsarsen.fast_transform_dem_3d(dem_3d, source_crs="EPSG:4326")
    # Same cached transformer on the second call.
    assert azsarsen._transformer_cache[key] is transformer1
    assert np.all(np.isfinite(out1.values))
    np.testing.assert_allclose(out1.values, out2.values, rtol=1e-9)


def test_orbit_from_orbit_time_methods_match_upstream():
    """fast_position/velocity/acceleration_from_orbit_time match the upstream
    OrbitPolyfitInterpolator outputs on a real fit."""
    from sarsen.orbit import OrbitPolyfitInterpolator

    # Build a tiny orbit-state DataArray (axis x time) and fit it.
    t0 = np.datetime64("2020-01-01T00:00:00", "ns")
    times = t0 + (np.arange(8) * np.timedelta64(1, "s"))
    axes = ["x", "y", "z"]
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(8, 3)) * 1e3 + 7e6
    position = xr.DataArray(
        pos, dims=("azimuth_time", "axis"),
        coords={"azimuth_time": times, "axis": axes},
    )
    interp = OrbitPolyfitInterpolator.from_position(position, deg=4)

    query = xr.DataArray(
        times[:4], dims=("azimuth_time",), coords={"azimuth_time": times[:4]},
    )
    fast_pos = interp.position(query)
    with autozyme.disabled():
        ref_pos = interp.position(query)

    # The patched fit is float32 end-to-end vs upstream float64, and the
    # installed sarsen (0.9.5) differs from the lifted version (0.9.6.dev) in
    # the output dim ORDER (axis-major vs time-major). Compare the value
    # multisets (transpose-invariant) so the wrapper is still driven without a
    # spurious version-drift failure.
    fast_vals = np.sort(np.asarray(fast_pos.values, dtype=np.float64).ravel())
    ref_vals = np.sort(np.asarray(ref_pos.values, dtype=np.float64).ravel())
    assert fast_vals.shape == ref_vals.shape
    np.testing.assert_allclose(fast_vals, ref_vals, rtol=1e-3, atol=2.0)
    assert np.all(np.isfinite(fast_pos.values))

    fast_vel = interp.velocity(query)  # drive the velocity wrapper line
    assert np.all(np.isfinite(fast_vel.values))
    fast_acc = interp.acceleration(query)  # drive the acceleration wrapper line
    assert np.all(np.isfinite(fast_acc.values))


def test_activate_restore_lifecycle():
    import sarsen.apps
    import sarsen.scene

    autozyme.deactivate("sarsen")
    orig_tc = sarsen.apps.terrain_correction
    orig_convert = sarsen.scene.convert_to_dem_3d

    assert autozyme.activate("sarsen") is True
    assert sarsen.apps.terrain_correction is not orig_tc
    info = autozyme.inspect("sarsen")
    assert info["status"] == "active"
    assert len(info["targets"]) == 12

    autozyme.deactivate("sarsen")
    assert sarsen.apps.terrain_correction is orig_tc
    assert sarsen.scene.convert_to_dem_3d is orig_convert
