"""Unit tests for the pure numpy/xarray helpers in autozyme.sarsen.

The contract test (test_sarsen.py) drives terrain_correction end to end. Here we
test the self-contained helpers directly:

  - _get_threads               ZYME_THREADS parsing
  - _datetime_to_int64         zero-copy datetime64->int64 view
  - _fast_orbit_polyval        fused Horner polyval, vs np.polyval
  - _product_signature         identity-tuple builder
  - beta_nought cache bookkeeping (_new_/_drop_/_beta_nought_cache_entry)
  - fast_convert_to_dem_3d / fast_zero_doppler_*  xarray numerics

sarsen must import for the module to load.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
pytest.importorskip("sarsen")

from autozyme import sarsen as azsarsen


# --------------------------------------------------------------------------
# _get_threads
# --------------------------------------------------------------------------
def test_get_threads_default_when_unset(monkeypatch):
    monkeypatch.delenv("ZYME_THREADS", raising=False)
    assert azsarsen._get_threads(default=7) == 7


def test_get_threads_reads_env(monkeypatch):
    monkeypatch.setenv("ZYME_THREADS", "5")
    assert azsarsen._get_threads(default=1) == 5


def test_get_threads_garbage_falls_back(monkeypatch):
    monkeypatch.setenv("ZYME_THREADS", "not-a-number")
    assert azsarsen._get_threads(default=3) == 3
    monkeypatch.setenv("ZYME_THREADS", "")
    assert azsarsen._get_threads(default=4) == 4


# --------------------------------------------------------------------------
# _datetime_to_int64
# --------------------------------------------------------------------------
def test_datetime_to_int64_zero_copy_view():
    times = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T00:00:01"], dtype="datetime64[ns]"
    )
    out = azsarsen._datetime_to_int64(times)
    assert out.dtype == np.int64
    # ns since epoch: second value is first + 1e9 ns.
    assert out[1] - out[0] == 1_000_000_000


def test_datetime_to_int64_casts_other_units():
    times = np.array(["2024-06-01", "2024-06-02"], dtype="datetime64[D]")
    out = azsarsen._datetime_to_int64(times)
    assert out.dtype == np.int64
    # one day in ns
    assert out[1] - out[0] == 24 * 3600 * 1_000_000_000


# --------------------------------------------------------------------------
# _fast_orbit_polyval
# --------------------------------------------------------------------------
def test_fast_orbit_polyval_matches_numpy_polyval():
    # coefficients indexed by `degree`, with a trailing value dimension.
    rng = np.random.default_rng(0)
    n_axis = 3
    max_deg = 4
    coeff = rng.standard_normal((max_deg + 1, n_axis)).astype(np.float32)
    degrees = np.arange(max_deg + 1)
    coefficients = xr.DataArray(
        coeff, dims=("degree", "axis"),
        coords={"degree": degrees, "axis": np.arange(n_axis)},
    )
    t = np.linspace(-1.0, 1.0, 11).astype(np.float32)
    orbit_time = xr.DataArray(t, dims=("orbit_time",))
    out = azsarsen._fast_orbit_polyval(orbit_time, coefficients, "position")
    assert out.dims == ("axis", "orbit_time")
    # np.polyval expects highest-degree-first; coeff[degree=d] multiplies t^d.
    for a in range(n_axis):
        poly_hi_first = coeff[::-1, a].astype(np.float64)
        ref = np.polyval(poly_hi_first, t.astype(np.float64))
        np.testing.assert_allclose(
            np.asarray(out.values[a], dtype=np.float64), ref, rtol=1e-4, atol=1e-5
        )


def test_fast_orbit_polyval_constant_coeff():
    # Degree-0 only -> constant output for every t.
    coeff = np.array([[2.0, -3.0]], dtype=np.float32)  # (degree=1, axis=2)
    coefficients = xr.DataArray(
        coeff, dims=("degree", "axis"),
        coords={"degree": [0], "axis": [0, 1]},
    )
    t = np.linspace(0, 5, 7).astype(np.float32)
    out = azsarsen._fast_orbit_polyval(
        xr.DataArray(t, dims=("orbit_time",)), coefficients, "velocity"
    )
    np.testing.assert_allclose(out.values[0], 2.0, rtol=1e-5)
    np.testing.assert_allclose(out.values[1], -3.0, rtol=1e-5)


# --------------------------------------------------------------------------
# _product_signature
# --------------------------------------------------------------------------
def test_product_signature_stable_and_distinct():
    p1 = SimpleNamespace(
        kwargs={"a": 1}, product_urlpath="/x", measurement_group="IW/VV",
        measurement_chunks=None,
    )
    p2 = SimpleNamespace(
        kwargs={"a": 1}, product_urlpath="/x", measurement_group="IW/VV",
        measurement_chunks=None,
    )
    p3 = SimpleNamespace(
        kwargs={"a": 2}, product_urlpath="/x", measurement_group="IW/VV",
        measurement_chunks=None,
    )
    assert azsarsen._product_signature(p1) == azsarsen._product_signature(p2)
    assert azsarsen._product_signature(p1) != azsarsen._product_signature(p3)


# --------------------------------------------------------------------------
# beta_nought cache bookkeeping
# --------------------------------------------------------------------------
class _Product:
    """Weakref-able product stand-in."""


def test_beta_nought_cache_entry_creates_and_reuses():
    azsarsen._beta_nought_cache.clear()
    prod = _Product()
    e1 = azsarsen._beta_nought_cache_entry(prod)
    e2 = azsarsen._beta_nought_cache_entry(prod)
    # Same live product -> same cache entry.
    assert e1 is e2
    assert e1["ref"]() is prod
    assert e1["value"] is None and "lock" in e1


def test_beta_nought_cache_drop_when_collected():
    azsarsen._beta_nought_cache.clear()
    prod = _Product()
    key = id(prod)
    azsarsen._beta_nought_cache_entry(prod)
    assert key in azsarsen._beta_nought_cache
    # Dropping a still-live entry is a no-op (ref() is not None).
    azsarsen._drop_beta_nought_cache_entry(key)
    assert key in azsarsen._beta_nought_cache


def test_beta_nought_cache_distinct_products():
    azsarsen._beta_nought_cache.clear()
    a, b = _Product(), _Product()
    ea = azsarsen._beta_nought_cache_entry(a)
    eb = azsarsen._beta_nought_cache_entry(b)
    assert ea is not eb


# --------------------------------------------------------------------------
# fast_convert_to_dem_3d
# --------------------------------------------------------------------------
def test_fast_convert_to_dem_3d_planes():
    ny, nx = 4, 5
    x = np.linspace(0, 40, nx)
    y = np.linspace(0, 30, ny)
    dem = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx)
    raster = xr.DataArray(
        dem, dims=("y", "x"), coords={"x": x, "y": y}, attrs={"foo": "bar"}
    )
    out = azsarsen.fast_convert_to_dem_3d(raster)
    assert out.shape == (3, ny, nx)
    # plane 0 = x broadcast, plane 1 = y broadcast, plane 2 = dem.
    np.testing.assert_allclose(out.values[0], np.broadcast_to(x, (ny, nx)).astype(np.float32))
    np.testing.assert_allclose(out.values[1], np.broadcast_to(y[:, None], (ny, nx)).astype(np.float32))
    np.testing.assert_allclose(out.values[2], dem.astype(np.float32))


# --------------------------------------------------------------------------
# fast_zero_doppler_plane_distance_velocity_prime  (pure einsum math)
# --------------------------------------------------------------------------
def test_zero_doppler_prime_einsum():
    ny, nx = 3, 4
    rng = np.random.default_rng(1)
    dist = rng.standard_normal((3, ny, nx))
    vel = rng.standard_normal((3, ny, nx))
    acc = rng.standard_normal((3, ny, nx))
    dem_distance = xr.DataArray(dist, dims=("axis", "y", "x"))
    velocity = xr.DataArray(vel, dims=("axis", "y", "x"))
    acceleration = xr.DataArray(acc, dims=("axis", "y", "x"))

    orbit = SimpleNamespace(
        acceleration_from_orbit_time=lambda t: acceleration
    )
    out = azsarsen.fast_zero_doppler_plane_distance_velocity_prime(
        orbit, orbit_time=None, payload=(dem_distance, velocity),
    )
    # plane' = sum_a dist*acc - sum_a vel*vel  (over the axis dimension)
    ref = np.einsum("ayx,ayx->yx", dist, acc) - np.einsum("ayx,ayx->yx", vel, vel)
    np.testing.assert_allclose(out.values, ref, rtol=1e-12)
