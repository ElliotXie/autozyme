"""Wave-3 heavy-path tests for autozyme.sarsen.

Wave-1 (`test_sarsen_unit.py`) tested the pure helpers; wave-2
(`test_sarsen_e2e.py`) covered fast_terrain_correction's scope guard,
fast_transform_dem_3d, and the orbit-fit wrappers. The full
terrain_correction / interp_sar / simulate_acquisition GRD path needs a real
Sentinel-1 product + DEM raster on disk and is genuinely too heavy / data
dependent for a contract test, so it was skipped.

This file drives the heavier INTERNAL data-plane kernels that DON'T need a real
SAR product, using synthetic xarray inputs, asserting parity vs independent
numpy / scipy references:

  - `fast_zero_doppler_plane_distance_velocity`: the Newton inner kernel
    (concurrent position+velocity polyval via the 2-worker pool, the
    out-buffer (dem - position) subtract, the einsum dot product). Parity vs a
    direct numpy einsum reference.
  - `fast_ground_range_interp_sar`: the method="nearest" scipy
    RegularGridInterpolator + chunked-thread dispatch -> bit-exact vs a direct
    single-shot RGI; AND the non-"nearest" fall-through to upstream.
  - `_fast_orbit_polyval` with NON-contiguous degrees (a gap, e.g. [0, 3]):
    drives the inner `current_degree - degree` multi-multiply loop that a
    dense-degree fit never hits.
  - `_apply_writable_azimuth_wrap` / `_apply_win_dtype_safe_wrap`: the
    workability shims (writable copy; datetime64 us->ns cast) + their
    idempotence guards.
  - `fast_beta_nought`: the already-cached-value early return.
  - `_sarsen_xarray_options`: the use_bottleneck=False scope.

The full GRD terrain_correction pipeline (simulate_acquisition / the timed
beta_nought compute / the smoke recipe) remains out of reach without real
Sentinel-1 data; see report.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
pytest.importorskip("sarsen")
pytest.importorskip("scipy")

import autozyme
from autozyme import sarsen as azsarsen


@pytest.fixture(autouse=True)
def _activate():
    """Mirror the e2e file's guard: tolerate a pre-initialized numba pool."""
    try:
        autozyme.activate("sarsen")
    except RuntimeError as e:
        if "NUMBA_NUM_THREADS" in str(e):
            pytest.skip(f"numba thread pool already initialized: {e}")
        raise
    yield
    autozyme.deactivate_all()


# --------------------------------------------------------------------------
# fast_zero_doppler_plane_distance_velocity (Newton inner kernel)
# --------------------------------------------------------------------------
def _vec_da(arr, ny, nx):
    return xr.DataArray(
        arr, dims=("axis", "y", "x"),
        coords={"axis": [0, 1, 2], "y": np.arange(ny), "x": np.arange(nx)},
    )


def test_zero_doppler_plane_distance_velocity_matches_einsum():
    ny, nx = 3, 4
    rng = np.random.default_rng(0)
    dem = rng.standard_normal((3, ny, nx))
    pos = rng.standard_normal((3, ny, nx))
    vel = rng.standard_normal((3, ny, nx))
    dem_ecef = _vec_da(dem, ny, nx)
    orbit = SimpleNamespace(
        position_from_orbit_time=lambda t: _vec_da(pos, ny, nx),
        velocity_from_orbit_time=lambda t: _vec_da(vel, ny, nx),
    )
    plane, (dem_distance, velocity) = (
        azsarsen.fast_zero_doppler_plane_distance_velocity(
            dem_ecef, orbit, orbit_time=None
        )
    )
    # plane = sum_a (dem - pos) * vel ; dem_distance = dem - pos.
    ref_dist = dem - pos
    ref_plane = np.einsum("ayx,ayx->yx", ref_dist, vel)
    np.testing.assert_allclose(dem_distance.values, ref_dist, rtol=1e-12)
    np.testing.assert_allclose(plane.values, ref_plane, rtol=1e-12)
    # The velocity is threaded back out unchanged for the _prime kernel.
    np.testing.assert_allclose(velocity.values, vel, rtol=1e-12)


def test_zero_doppler_prime_consumes_velocity_payload():
    """The plane + (dem_distance, velocity) payload feeds the _prime kernel; the
    two stages compose into the Newton derivative used by backward_geocode."""
    ny, nx = 3, 4
    rng = np.random.default_rng(1)
    dem = rng.standard_normal((3, ny, nx))
    pos = rng.standard_normal((3, ny, nx))
    vel = rng.standard_normal((3, ny, nx))
    acc = rng.standard_normal((3, ny, nx))
    dem_ecef = _vec_da(dem, ny, nx)
    orbit = SimpleNamespace(
        position_from_orbit_time=lambda t: _vec_da(pos, ny, nx),
        velocity_from_orbit_time=lambda t: _vec_da(vel, ny, nx),
        acceleration_from_orbit_time=lambda t: _vec_da(acc, ny, nx),
    )
    _, payload = azsarsen.fast_zero_doppler_plane_distance_velocity(
        dem_ecef, orbit, orbit_time=None
    )
    prime = azsarsen.fast_zero_doppler_plane_distance_velocity_prime(
        orbit, orbit_time=None, payload=payload
    )
    ref = (np.einsum("ayx,ayx->yx", dem - pos, acc)
           - np.einsum("ayx,ayx->yx", vel, vel))
    np.testing.assert_allclose(prime.values, ref, rtol=1e-10)


# --------------------------------------------------------------------------
# fast_ground_range_interp_sar (scipy RGI nearest + fall-through)
# --------------------------------------------------------------------------
def _interp_inputs(seed=0):
    rng = np.random.default_rng(seed)
    n_az, n_gr = 5, 6
    az_coord = (np.datetime64("2021-01-01T00:00:00", "ns")
                + np.arange(n_az) * np.timedelta64(1, "s"))
    gr_coord = np.linspace(0.0, 1000.0, n_gr)
    vals = rng.standard_normal((n_az, n_gr))
    data = xr.DataArray(
        vals, dims=("azimuth_time", "ground_range"),
        coords={"azimuth_time": az_coord, "ground_range": gr_coord},
    )
    ty, tx = 4, 3
    tgt_az = (np.datetime64("2021-01-01T00:00:01", "ns")
              + np.zeros((ty, tx), dtype="timedelta64[ns]"))
    target_az = xr.DataArray(tgt_az, dims=("y", "x"))
    ground = xr.DataArray(np.full((ty, tx), 500.0), dims=("y", "x"))
    return data, target_az, ground, az_coord, gr_coord, vals, tgt_az


def test_ground_range_interp_sar_nearest_matches_scipy():
    from scipy.interpolate import RegularGridInterpolator

    data, target_az, ground, az_coord, gr_coord, vals, tgt_az = _interp_inputs()

    class _GRProduct:
        def slant_range_time_to_ground_range(self, az, srt):
            return ground

    out = azsarsen.fast_ground_range_interp_sar(
        _GRProduct(), data, target_az, ground_range=ground, method="nearest"
    )
    azc = azsarsen._datetime_to_int64(az_coord).astype(np.float64)
    rgi = RegularGridInterpolator(
        (azc, gr_coord), vals, method="nearest",
        bounds_error=False, fill_value=np.nan,
    )
    ta = azsarsen._datetime_to_int64(tgt_az).ravel().astype(np.float64)
    tg = ground.values.ravel()
    ref = rgi((ta, tg)).reshape(out.shape)
    np.testing.assert_allclose(out.values, ref, equal_nan=True, rtol=1e-12)
    assert out.shape == target_az.shape


def test_ground_range_interp_sar_non_nearest_falls_through(monkeypatch):
    data, target_az, ground, *_ = _interp_inputs()
    seen = {}

    def _fake_orig(self, data, az, slant_range_time=None, method="nearest",
                   ground_range=None):
        seen["method"] = method
        return "UPSTREAM"

    monkeypatch.setattr(
        azsarsen, "_orig_ground_range_interp_sar", _fake_orig
    )
    out = azsarsen.fast_ground_range_interp_sar(
        object(), data, target_az, method="linear", ground_range=ground
    )
    assert out == "UPSTREAM"
    assert seen["method"] == "linear"


def test_ground_range_interp_sar_computes_ground_range_when_absent():
    """When ground_range is None, the helper derives it via the product's
    slant_range_time_to_ground_range (the assert + compute path)."""
    from scipy.interpolate import RegularGridInterpolator

    data, target_az, ground, az_coord, gr_coord, vals, tgt_az = _interp_inputs(2)

    class _GRProduct:
        called = False

        def slant_range_time_to_ground_range(self, az, srt):
            type(self).called = True
            return ground

    prod = _GRProduct()
    srt = xr.DataArray(np.full(target_az.shape, 1e-3), dims=("y", "x"))
    out = azsarsen.fast_ground_range_interp_sar(
        prod, data, target_az, slant_range_time=srt, method="nearest"
    )
    assert _GRProduct.called is True
    assert out.shape == target_az.shape
    assert np.all(np.isfinite(out.values))


# --------------------------------------------------------------------------
# _fast_orbit_polyval — non-contiguous degrees (gap)
# --------------------------------------------------------------------------
def test_fast_orbit_polyval_gap_degrees():
    """Degrees [0, 3] (a gap) drive the `current_degree - degree` repeated-
    multiply loop a dense fit never enters; result = c0 + c3 * t^3 per axis."""
    coeff = np.array([[1.0, 2.0], [0.5, -1.0]], dtype=np.float32)  # (deg=2, axis=2)
    coefficients = xr.DataArray(
        coeff, dims=("degree", "axis"), coords={"degree": [0, 3], "axis": [0, 1]}
    )
    t = np.linspace(-1.0, 1.0, 9).astype(np.float32)
    orbit_time = xr.DataArray(t, dims=("orbit_time",))
    out = azsarsen._fast_orbit_polyval(orbit_time, coefficients, "position")
    for a in range(2):
        ref = coeff[0, a] + coeff[1, a] * (t.astype(np.float64) ** 3)
        np.testing.assert_allclose(
            out.values[a].astype(np.float64), ref, rtol=1e-3, atol=1e-4
        )


# --------------------------------------------------------------------------
# fast_slant_range_time_to_ground_range — the non-Horner (gap-degree) branch
# --------------------------------------------------------------------------
def test_slant_range_non_horner_branch_matches_upstream():
    """Non-contiguous srgr degrees ([0, 2, 4]) make use_horner False, driving
    the `else` np.power accumulation branch (lines 355-361); parity vs upstream.
    The existing test_sarsen.py only covers the contiguous (Horner) branch."""
    import xarray_sentinel

    source_time = np.array(
        ["2020-01-01T00:00:00", "2020-01-01T00:00:10"], dtype="datetime64[ns]"
    )
    degree = np.array([0, 2, 4])  # gaps -> use_horner False
    cc = xr.Dataset({
        "sr0": xr.DataArray(
            [100.0, 120.0], dims=("azimuth_time",),
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
            [["2020-01-01T00:00:00", "2020-01-01T00:00:10"],
             ["2020-01-01T00:00:10", "2020-01-01T00:00:00"]],
            dtype="datetime64[ns]",
        ),
        dims=("y", "x"),
    )
    slant_range_time = xr.DataArray(
        np.array([[1.0e-6, 1.1e-6], [1.2e-6, 1.3e-6]]), dims=("y", "x")
    )
    fast = xarray_sentinel.slant_range_time_to_ground_range(
        azimuth_time, slant_range_time, cc
    )
    with autozyme.disabled():
        ref = xarray_sentinel.slant_range_time_to_ground_range(
            azimuth_time, slant_range_time, cc
        )
    np.testing.assert_allclose(
        fast.values, ref.values, rtol=1e-10, atol=1e-10, equal_nan=True
    )


# --------------------------------------------------------------------------
# Workability wraps (writable copy + dtype-safe cast) + idempotence
# --------------------------------------------------------------------------
def test_writable_azimuth_wrap_returns_writable_copy():
    import xarray_sentinel.sentinel1 as xs1

    saved = xs1.make_azimuth_time

    def _readonly_make(*a, **k):
        arr = np.arange(5)
        arr.setflags(write=False)
        return arr

    xs1.make_azimuth_time = _readonly_make
    try:
        azsarsen._apply_writable_azimuth_wrap()
        assert getattr(xs1.make_azimuth_time, "__autozyme_writable_wrap__", False)
        out = xs1.make_azimuth_time()
        assert out.flags.writeable is True
        # Idempotent: re-applying does not double-wrap.
        wrapped = xs1.make_azimuth_time
        azsarsen._apply_writable_azimuth_wrap()
        assert xs1.make_azimuth_time is wrapped
    finally:
        xs1.make_azimuth_time = saved


def test_win_dtype_safe_wrap_casts_mismatched_azimuth_dtype():
    import xarray_sentinel
    import xarray_sentinel.sentinel1 as xs1

    saved = xarray_sentinel.slant_range_time_to_ground_range
    saved_xs1 = getattr(xs1, "slant_range_time_to_ground_range", None)
    seen = {}

    def _fake_slant(azimuth_time, slant_range_time, coordinate_conversion):
        seen["dtype"] = azimuth_time.dtype
        return "OK"

    xarray_sentinel.slant_range_time_to_ground_range = _fake_slant
    try:
        azsarsen._apply_win_dtype_safe_wrap()
        wrapped = xarray_sentinel.slant_range_time_to_ground_range
        assert getattr(wrapped, "__autozyme_dtype_safe_wrap__", False)
        az = xr.DataArray(np.array(["2021-01-01"], dtype="datetime64[us]"))
        cc = SimpleNamespace(
            azimuth_time=xr.DataArray(
                np.array(["2021-01-01"], dtype="datetime64[ns]")
            )
        )
        assert wrapped(az, None, cc) == "OK"
        # The mismatched us azimuth_time was cast to the cc's ns dtype.
        assert seen["dtype"] == np.dtype("datetime64[ns]")
        # Idempotent.
        azsarsen._apply_win_dtype_safe_wrap()
        assert xarray_sentinel.slant_range_time_to_ground_range is wrapped
    finally:
        xarray_sentinel.slant_range_time_to_ground_range = saved
        if saved_xs1 is not None:
            xs1.slant_range_time_to_ground_range = saved_xs1


# --------------------------------------------------------------------------
# fast_beta_nought cached-value early return + xarray options scope
# --------------------------------------------------------------------------
def test_beta_nought_returns_cached_value_without_recompute():
    azsarsen._beta_nought_cache.clear()

    class _Product:
        pass

    prod = _Product()
    entry = azsarsen._beta_nought_cache_entry(prod)
    entry["value"] = "BN_CACHED"  # pre-populate; compute must be skipped.
    assert azsarsen.fast_beta_nought(prod) == "BN_CACHED"


def test_sarsen_xarray_options_disables_bottleneck():
    with azsarsen._sarsen_xarray_options():
        assert xr.get_options()["use_bottleneck"] is False
