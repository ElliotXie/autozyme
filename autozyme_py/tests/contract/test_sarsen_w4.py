"""Wave-4 reachable geometry + beta_nought-cache tests for autozyme.sarsen.

Waves 1-3 covered the pure helpers, the Newton inner kernels, the RGI interp
path + fall-through, the orbit polyval (gap degrees), the workability shims, and
`fast_beta_nought`'s already-cached early return. This file reaches the
remaining COVERAGE-VISIBLE Python branches:

  - `_fast_orbit_polyval` with a non-zero LOWEST degree (degrees [1, 3]): drives
    the trailing `for _ in range(current_degree): result *= t` multiply (src line
    406) that the wave-3 [0, 3] case skipped (last degree there was 0).
  - `fast_transform_dem_3d` with `source_crs=None` -> derive it from
    `dem_3d.rio.crs` (src line 189).
  - `fast_convert_to_dem_3d` carrying a `spatial_ref` coord (src line 275).
  - the beta_nought cache DIRECTORY machinery (src 581-638), which wave-3 only
    touched via the early return:
      * a non-weakref-able product -> the `weakref.ref` TypeError fallback to a
        content signature (src 593-594),
      * `_product_signature` repr-exception fallback (src 610-611),
      * the signature-match cached-return path (src 630-631),
      * the stale-entry (dead weakref) replacement path (src 636-638).

NOT reachable here (documented): `_compute_beta_nought_for` (src 642-646) and
`fast_beta_nought`'s lock+compute body (src 653-656) call
`xarray_sentinel.calibrate_intensity` on a real Sentinel-1 product measurement
-- they need a real .SAFE product. The smoke recipe (`_smoke_load`/`_smoke_call`/
`_smoke_save`, src 779-893) runs the full GRD `apps.terrain_correction` against a
real Sentinel-1 SAFE directory + a real DEM raster (large on-disk data, multi-
second-to-minutes) -- genuinely infeasible for a self-contained contract test,
as wave-3 already noted. The threaded pyproj / numba kernels are invisible to
coverage.py.
"""
from __future__ import annotations

import threading

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
pytest.importorskip("sarsen")
pytest.importorskip("scipy")
pytest.importorskip("rioxarray")  # for rio.write_crs / rio.crs

import autozyme
from autozyme import sarsen as azsarsen


@pytest.fixture(autouse=True)
def _activate():
    try:
        autozyme.activate("sarsen")
    except RuntimeError as e:
        if "NUMBA_NUM_THREADS" in str(e):
            pytest.skip(f"numba thread pool already initialized: {e}")
        raise
    yield
    autozyme.deactivate_all()


# --------------------------------------------------------------------------
# _fast_orbit_polyval: trailing multiply when lowest degree > 0
# --------------------------------------------------------------------------
def test_fast_orbit_polyval_nonzero_lowest_degree():
    """Degrees [1, 3] (lowest degree 1, not 0) drive the trailing
    `for _ in range(current_degree): result *= t` multiply at src line 406 that
    a [0, 3] fit never enters. result = c1*t + c3*t^3 per axis."""
    coeff = np.array([[1.0, 2.0], [0.5, -1.0]], dtype=np.float32)  # (deg, axis)
    coefficients = xr.DataArray(
        coeff, dims=("degree", "axis"), coords={"degree": [1, 3], "axis": [0, 1]}
    )
    t = np.linspace(-1.0, 1.0, 7).astype(np.float32)
    orbit_time = xr.DataArray(t, dims=("orbit_time",))
    out = azsarsen._fast_orbit_polyval(orbit_time, coefficients, "position")
    td = t.astype(np.float64)
    for a in range(2):
        ref = coeff[0, a] * td + coeff[1, a] * (td ** 3)
        np.testing.assert_allclose(
            out.values[a].astype(np.float64), ref, rtol=1e-3, atol=1e-4
        )


# --------------------------------------------------------------------------
# fast_transform_dem_3d: source_crs=None -> dem_3d.rio.crs
# --------------------------------------------------------------------------
def test_transform_dem_3d_source_crs_from_rio():
    """source_crs=None makes the helper read the CRS off `dem_3d.rio.crs`
    (src line 189), then run the threaded pyproj transform."""
    ny, nx = 3, 4
    da = xr.DataArray(
        np.random.default_rng(0).standard_normal((3, ny, nx)),
        dims=("axis", "y", "x"),
        coords={
            "axis": [0, 1, 2],
            "y": np.arange(ny, dtype="float64"),
            "x": np.arange(nx, dtype="float64"),
        },
    ).rio.write_crs("EPSG:4326")
    out = azsarsen.fast_transform_dem_3d(da, source_crs=None)
    assert out.shape == (3, ny, nx)
    assert np.all(np.isfinite(out.values))


# --------------------------------------------------------------------------
# fast_convert_to_dem_3d: spatial_ref coord carried through
# --------------------------------------------------------------------------
def test_convert_to_dem_3d_carries_spatial_ref():
    """A dem_raster with a `spatial_ref` coord makes the builder propagate it to
    the output coords (src line 275)."""
    ny, nx = 3, 4
    dem = xr.DataArray(
        np.arange(ny * nx, dtype="float64").reshape(ny, nx),
        dims=("y", "x"),
        coords={
            "y": np.arange(ny, dtype="float64"),
            "x": np.arange(nx, dtype="float64"),
            "spatial_ref": 0,
        },
        name="dem",
    )
    out = azsarsen.fast_convert_to_dem_3d(dem)
    assert out.shape == (3, ny, nx)
    assert "spatial_ref" in out.coords


# --------------------------------------------------------------------------
# beta_nought cache directory machinery
# --------------------------------------------------------------------------
def test_beta_nought_cache_non_weakrefable_uses_signature():
    """A non-weakref-able product makes `_new_beta_nought_cache_entry` fall back
    to a content signature (the `weakref.ref` TypeError branch, src 593-594), and
    a second lookup with the same signature returns the SAME cached entry
    (the signature-match path, src 630-631)."""
    azsarsen._beta_nought_cache.clear()

    class _NoWeakref:
        __slots__ = ()  # no __weakref__ slot -> weakref.ref(...) raises TypeError

    prod = _NoWeakref()
    entry = azsarsen._beta_nought_cache_entry(prod)
    assert entry["ref"] is None
    assert entry["signature"] is not None
    # Same product -> signature match -> same entry returned (not re-created).
    again = azsarsen._beta_nought_cache_entry(prod)
    assert again is entry


def test_product_signature_repr_exception_fallback():
    """`_product_signature` falls back to a plain repr when sorting the kwargs
    items raises (src lines 610-611). The signature is still a 6-tuple."""

    class _Unsortable:
        def items(self):
            raise RuntimeError("boom")

    class _BadKwargs:
        kwargs = _Unsortable()

    sig = azsarsen._product_signature(_BadKwargs())
    assert isinstance(sig, tuple)
    assert len(sig) == 6


def test_beta_nought_cache_stale_entry_replaced():
    """A cache slot whose weakref is dead (ref() is None) and whose product is
    no longer alive is REPLACED with a fresh entry bound to the live product
    (the stale-replacement path, src 636-638)."""
    azsarsen._beta_nought_cache.clear()

    class _WeakP:
        pass

    prod = _WeakP()
    key = id(prod)
    # Plant a stale entry under prod's id: a dead weakref, no signature.
    azsarsen._beta_nought_cache[key] = {
        "value": None,
        "lock": threading.Lock(),
        "ref": (lambda: None),  # ref() -> None: dead
        "signature": None,
    }
    fresh = azsarsen._beta_nought_cache_entry(prod)
    assert fresh["ref"] is not None
    assert fresh["ref"]() is prod
