"""Patch for xclim.indices.growing_season_length.

Lifted from autozyme task `test_xclim`. Replaces upstream's xarray.apply_ufunc
chain (~50% wall time in dispatch on tiny tier) with a single numba-compiled
per-cell scan over the boolean threshold mask.

Two registered targets — the same fast function, bound at both the canonical
implementation site and the public package alias, since users import the
function from either path:
  - `xclim.indices._threshold.growing_season_length`
  - `xclim.indices.growing_season_length`
"""
from __future__ import annotations

import os
import sys

# numba's parallel TBB layer can deadlock with TensorFlow's threading layer
# on first parallel JIT compile if both are loaded in the same process. If
# TF is already in sys.modules (e.g. autozyme.sccoda was activated earlier),
# fall back to single-thread numba — the prange kernel still gets the JIT
# speedup, just without parallelism. Set BEFORE numba is imported.
#
# Bug 5 guard (2026-05-28): if numba's threadpool is ALREADY initialized by
# an earlier patch (scanpy / squidpy_cooccurrence), changing the env var
# now triggers numba's config-time mismatch check ("currently have X,
# trying to set Y" RuntimeError). The TF/TBB deadlock concern is moot once
# the pool is running anyway — skip the env-var set in that case.
if "tensorflow" in sys.modules:
    _pool_already_init = False
    try:
        from numba.np.ufunc import parallel as _nb_parallel  # noqa: PLC0415
        _pool_already_init = bool(_nb_parallel._is_initialized)
    except ImportError:
        pass
    if not _pool_already_init:
        os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import xarray as xr
from numba import njit, prange

import xclim
import xclim.indices
import xclim.indices._threshold  # ensure module loaded before patching
from xclim.core.units import convert_units_to

from autozyme._core import register_patch
from autozyme._utils import resolve_dataset_path


@njit(cache=True, parallel=True, boundscheck=False, fastmath=True)
def _gsl_kernel(tas, thresh, window, year_starts, year_ends, mid_offsets, op_ge):
    n_cells, _ = tas.shape
    n_years = mid_offsets.shape[0]
    out = np.zeros((n_cells, n_years), dtype=np.float64)
    for c in prange(n_cells):
        for y in range(n_years):
            t0 = year_starts[y]
            t1 = year_ends[y]
            mid = mid_offsets[y]
            beg = -1
            run = 0
            upper = mid + window - 1
            if upper > t1:
                upper = t1
            for i in range(t0, upper):
                v = tas[c, i]
                cond = (v >= thresh) if op_ge else (v > thresh)
                if cond:
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
                v = tas[c, i]
                cond = (v >= thresh) if op_ge else (v > thresh)
                if not cond:
                    run += 1
                    if run >= window:
                        end = i - window + 1
                        break
                else:
                    run = 0
            if end < 0:
                out[c, y] = float(t1 - beg)
            else:
                out[c, y] = float(end - beg)
    return out


def fast_growing_season_length(tas, thresh="5.0 degC", window=6,
                               mid_date="07-01", freq="YS", op=">=",
                               zyme=True, **kwargs):
    if not zyme or op not in (">", ">=", "gt", "ge") or mid_date is None or freq != "YS":
        return _orig_growing_season_length(
            tas, thresh=thresh, window=window, mid_date=mid_date,
            freq=freq, op=op, **kwargs,
        )
    op_ge = op in (">=", "ge")
    thresh_val = float(convert_units_to(thresh, tas, context="infer"))

    time_dim = "time"
    other_dims = [d for d in tas.dims if d != time_dim]
    arr = tas.transpose(*other_dims, time_dim).values
    spatial_shape = arr.shape[:-1]
    n_time = arr.shape[-1]
    arr_flat = np.ascontiguousarray(arr.reshape(-1, n_time), dtype=np.float32)

    times = tas.indexes[time_dim]
    years = np.asarray(times.year)
    year_starts = np.concatenate(([0], np.flatnonzero(np.diff(years)) + 1)).astype(np.int64)
    year_ends = np.concatenate((year_starts[1:], [n_time])).astype(np.int64)
    n_years = year_starts.size

    mm, dd = (int(x) for x in mid_date.split("-"))
    months = np.asarray(times.month)
    days = np.asarray(times.day)
    mid_match = np.flatnonzero((months == mm) & (days == dd))
    mid_offsets = np.empty(n_years, dtype=np.int64)
    for y in range(n_years):
        in_year = mid_match[(mid_match >= year_starts[y]) & (mid_match < year_ends[y])]
        if not in_year.size:
            # mid_date absent in this year's calendar (e.g. '02-29' on a noleap
            # year): the fast path would silently force length 0, whereas
            # upstream delegates. Out of scope — fall back. The default
            # mid_date '07-01' is present every year, so this never fires
            # in-scope (zero cost on the supported path).
            return _orig_growing_season_length(
                tas, thresh=thresh, window=window, mid_date=mid_date,
                freq=freq, op=op, **kwargs,
            )
        mid_offsets[y] = in_year[0]

    res_flat = _gsl_kernel(arr_flat, np.float32(thresh_val), int(window),
                           year_starts, year_ends, mid_offsets, op_ge)
    res = res_flat.reshape(*spatial_shape, n_years)

    out_time = times[year_starts]
    coords = {d: tas.coords[d] for d in other_dims if d in tas.coords}
    coords[time_dim] = out_time
    return xr.DataArray(
        res, dims=tuple(other_dims) + (time_dim,), coords=coords,
        attrs={"long_name": "Length of the season.",
               "description": "Number of steps of the original series in the season, between 'start' and 'end'.",
               "units": "d"},
        name=tas.name or "growing_season_length",
    )


# Capture upstream original at submodule import — used by zyme=False fallback
# AND the unsupported-args fallback (op=='!=' etc.). Capture happens before
# register_patch, so it's the un-patched version.
_orig_growing_season_length = xclim.indices._threshold.growing_season_length


_PARAMS = dict(thresh="5.0 degC", window=6, mid_date="07-01", freq="YS", op=">=")

# Warm the numba kernel at submodule import (which happens lazily, only on
# the first autozyme.activate("xclim") — so process is uncontaminated).
_warm_array = xr.DataArray(
    np.zeros((2, 2, 730), dtype=np.float32),
    dims=("lat", "lon", "time"),
    coords={"time": xr.date_range("1990-01-01", periods=730, freq="D", calendar="noleap")},
    attrs={"units": "K"}, name="tas",
)
fast_growing_season_length(_warm_array, **_PARAMS)
del _warm_array


# ---------- smoke recipe ----------

def _smoke_load(task_dir, tier):
    import yaml
    with open(os.path.join(task_dir, "task.yaml"), encoding="utf-8") as f:
        task = yaml.safe_load(f)
    ds = next(d for d in task["datasets"] if d["tier"] == tier)
    nc_path = resolve_dataset_path(task_dir, ds["path"])
    dataset = xr.open_dataset(nc_path)
    return {"tas": dataset["tas"].load()}


def _smoke_call(inputs):
    # Use the public alias so namespace lookup picks up the active version.
    from xclim.indices import growing_season_length
    out = growing_season_length(inputs["tas"], **_PARAMS)
    return out.compute() if hasattr(out, "compute") else out


def _smoke_save(out, dir, **kwargs):
    np.save(os.path.join(dir, "gsl.npy"), np.asarray(out.values))


register_patch(
    name="xclim",
    targets=[
        ("xclim.indices._threshold", "growing_season_length", fast_growing_season_length),
        ("xclim.indices",            "growing_season_length", fast_growing_season_length),
    ],
    smoke={"load": _smoke_load, "call": _smoke_call, "save": _smoke_save},
    tested_against="xclim 0.60.0",
    tested_upstream_versions={"xclim": ["0.60.0"]},
)
