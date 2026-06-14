"""Wave-4 tests for autozyme.xclim: the mid-date-absent fallback + smoke recipe.

KERNEL CEILING: the heavy `_gsl_kernel` numba @njit body (src lines 52-99) is
INVISIBLE to coverage.py; wave-1 (`test_xclim_unit.py`) drives it directly. The
import-time TF/TBB guard (src 29-37) only fires when `tensorflow` is already in
`sys.modules` at xclim-submodule import -- an import-ordering side effect not
reachable from a normal test (xclim is imported standalone here). Everything
else reachable is covered by waves 1-2 (fast path, zyme=False, mid_date=None,
freq!=YS, unsupported-op delegations).

This file covers the two remaining COVERAGE-VISIBLE Python lines neither hit:

  - the mid-date-PRESENT-IN-SCHEMA-but-ABSENT-IN-A-YEAR fallback (src line 139):
    `mid_date="02-29"` on a `noleap` calendar is a syntactically-supported
    mid_date the fast path accepts, but `02-29` never occurs in any noleap year,
    so the per-year mid_offset lookup finds nothing -> delegate to upstream. (The
    existing tests only hit `mid_date=None`, which short-circuits earlier.) We
    assert the fallback result matches the captured upstream original.
  - the smoke recipe: `_smoke_load` (open the NetCDF, load the `tas` var),
    `_smoke_call` (the public-alias `growing_season_length`), `_smoke_save`
    (gsl.npy), via a synthetic NetCDF written with xarray's own writer.
"""
from __future__ import annotations

import os
import tempfile
import warnings

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
pytest.importorskip("xclim")
pytest.importorskip("h5netcdf")  # NetCDF backend for the smoke round-trip

import autozyme
from autozyme import xclim as azxclim

_PARAMS = dict(thresh="5.0 degC", window=6, mid_date="07-01", freq="YS", op=">=")


@pytest.fixture(autouse=True)
def _activate():
    warnings.filterwarnings("ignore")
    autozyme.activate("xclim")
    yield
    autozyme.deactivate_all()


def test_mid_date_absent_in_noleap_year_delegates():
    """`mid_date="02-29"` on a noleap calendar is a supported mid_date string, so
    the fast path is entered, but 02-29 never occurs -> the per-year mid_offset
    lookup falls back to upstream (src line 139). Result matches upstream."""
    times = xr.date_range("1990-01-01", periods=730, freq="D", calendar="noleap")
    tas = xr.DataArray(
        (np.random.default_rng(0).random((2, 730)).astype(np.float32) * 20 + 273.15),
        dims=("lat", "time"),
        coords={"lat": [0, 1], "time": times},
        attrs={"units": "K"},
        name="tas",
    )
    out = azxclim.fast_growing_season_length(
        tas, thresh="5.0 degC", window=6, mid_date="02-29", freq="YS", op=">="
    )
    ref = azxclim._orig_growing_season_length(
        tas, thresh="5.0 degC", window=6, mid_date="02-29", freq="YS", op=">="
    )
    np.testing.assert_allclose(
        np.asarray(out.values), np.asarray(ref.values), equal_nan=True,
        rtol=1e-6, atol=1e-6,
    )


@pytest.fixture
def smoke_task_dir():
    import yaml

    td = tempfile.mkdtemp(prefix="autozyme_xclim_w4_")
    os.makedirs(os.path.join(td, "data"), exist_ok=True)
    times = xr.date_range("1990-01-01", periods=1095, freq="D", calendar="noleap")
    tas = xr.DataArray(
        (np.random.default_rng(1).random((3, 1095)).astype(np.float64) * 25 + 270),
        dims=("lat", "time"),
        coords={"lat": [0, 1, 2], "time": times},
        attrs={"units": "K"},
        name="tas",
    )
    tas.to_dataset(name="tas").to_netcdf(
        os.path.join(td, "data", "clim.nc"), engine="h5netcdf"
    )
    with open(os.path.join(td, "task.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"datasets": [{"tier": "small", "path": "data/clim.nc"}]}, f
        )
    return {"dir": td, "n_lat": 3, "n_years": 3}


def test_smoke_load_call_save_roundtrip(smoke_task_dir):
    """`_smoke_load` opens the NetCDF + loads `tas`, `_smoke_call` runs
    `growing_season_length` via the public alias (the active patched version),
    `_smoke_save` writes gsl.npy."""
    inputs = azxclim._smoke_load(smoke_task_dir["dir"], "small")
    assert "tas" in inputs
    assert inputs["tas"].shape == (smoke_task_dir["n_lat"], 1095)

    out = azxclim._smoke_call(inputs)
    assert out.shape == (smoke_task_dir["n_lat"], smoke_task_dir["n_years"])

    out_dir = tempfile.mkdtemp(prefix="autozyme_xclim_w4_out_")
    azxclim._smoke_save(out, out_dir)
    arr = np.load(os.path.join(out_dir, "gsl.npy"))
    assert arr.shape == (smoke_task_dir["n_lat"], smoke_task_dir["n_years"])
    assert np.all(np.isfinite(arr))
    # growing-season length is bounded by the days-per-noleap-year.
    assert np.all((arr >= 0) & (arr <= 365))
