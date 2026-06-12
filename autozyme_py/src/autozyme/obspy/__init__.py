"""Patch for obspy Stream.filter (bandpass).

Lifted from autozyme task `test_obspy`. Three coordinated patches:
  - obspy.core.trace._get_function_from_entry_point — lru_cache on the
    entry-point lookup that obspy hits per-trace per-filter call.
  - obspy.signal.filter.bandpass — cached SOS coefficients (the
    iirfilter call dominates at small N).
  - obspy.core.stream.Stream.filter (class method) — ThreadPoolExecutor
    over per-trace filtering for parallelism.
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from obspy.core.util import base as _obspy_base
from scipy.signal import iirfilter, sosfilt

from autozyme._core import register_patch
from autozyme._utils import resolve_dataset_path

# Capture upstream originals BEFORE register_patch runs, so they're frozen.
_orig_entrypoint = _obspy_base._get_function_from_entry_point

@lru_cache(maxsize=32)
def fast_get_function_from_entry_point(group, type):
    return _orig_entrypoint(group, type)


@lru_cache(maxsize=32)
def _cached_iir_sos(corners, freqs, df, rp, rs, btype, ftype):
    fe = 0.5 * df
    normalized_freqs = [f / fe for f in freqs]
    if len(normalized_freqs) == 1:
        normalized_freqs = normalized_freqs[0]
    return iirfilter(corners, normalized_freqs, rp=rp, rs=rs, btype=btype,
                     ftype=ftype, output="sos")


def fast_bandpass(data, freqmin, freqmax, df, corners=4, zerophase=False,
                  rp=None, rs=None, ftype="butter", axis=-1):
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    if high - 1.0 > -1e-6:
        msg = ("Selected high corner frequency ({}) of bandpass is at or "
               "above Nyquist ({}). Applying a high-pass instead.").format(
            freqmax, fe)
        warnings.warn(msg)
        from obspy.signal.filter import highpass
        return highpass(data, freq=freqmin, df=df, corners=corners,
                        ftype=ftype, zerophase=zerophase)
    if low > 1:
        raise ValueError("Selected low corner frequency is above Nyquist.")

    sos = _cached_iir_sos(corners, (freqmin, freqmax), df, rp, rs, "band", ftype)
    if zerophase:
        firstpass = np.flip(sosfilt(sos, data, axis=axis), axis=axis)
        return np.flip(sosfilt(sos, firstpass, axis=axis), axis=axis)
    return sosfilt(sos, data, axis=axis)


def fast_stream_filter(self, type, *args, **options):
    # obspy's `buffered_load_entry_point` caches function references the FIRST
    # time it resolves an entry point. If a baseline call (with our patch
    # restored) ran before us, the cache holds upstream `bandpass` — so our
    # subsequently-setattr'd `fast_bandpass` would be invisible to Trace.filter.
    # Clear the cache so the next entry-point load re-reads the module attr.
    try:
        from obspy.core.util.misc import _ENTRY_POINT_CACHE
        _ENTRY_POINT_CACHE.clear()
    except Exception:
        pass

    from autozyme._threads import auto_threads
    worker_cap = auto_threads(cap=24)
    max_workers = min(len(self), worker_cap)

    if max_workers <= 1 or len(self) <= 1:
        for tr in self:
            tr.filter(type, *args, **options)
        return self

    def apply_filter(tr):
        tr.filter(type, *args, **options)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(apply_filter, self))
    return self


# ---------- smoke recipe ----------

def _smoke_load(task_dir, tier):
    import json
    import yaml
    from obspy import Stream, Trace, UTCDateTime

    with open(os.path.join(task_dir, "task.yaml"), encoding="utf-8") as f:
        task = yaml.safe_load(f)
    ds = next(d for d in task["datasets"] if d["tier"] == tier)
    manifest_path = resolve_dataset_path(task_dir, ds["path"])
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    rng = np.random.default_rng(manifest["seed"])
    n_traces = manifest["n_traces"]
    n_samples = manifest["n_samples"]
    sr = manifest["sampling_rate"]
    t0 = UTCDateTime(2024, 1, 1)
    traces = []
    for i in range(n_traces):
        data = rng.standard_normal(n_samples).astype("float32")
        tr = Trace(
            data=data,
            header={"sampling_rate": sr, "starttime": t0,
                    "network": "XX", "station": f"S{i:03d}", "channel": "BHZ"},
        )
        traces.append(tr)
    return {
        "stream": Stream(traces),
        "filter_kwargs": manifest["filter_kwargs"],
        "n_repeats": manifest.get("n_repeats", 1),
        "subsample_stride": manifest["subsample_stride"],
    }


def _smoke_call(inputs):
    fk = inputs["filter_kwargs"]
    st = inputs["stream"].copy()  # don't mutate input
    for _ in range(inputs["n_repeats"]):
        st.filter(
            fk["type"],
            freqmin=fk["freqmin"], freqmax=fk["freqmax"],
            corners=fk["corners"], zerophase=fk["zerophase"],
        )
    return {"stream": st, "stride": inputs["subsample_stride"]}


def _smoke_save(result, dir, **kwargs):
    stride = result["stride"]
    out = np.stack([tr.data[::stride] for tr in result["stream"]], axis=0)
    np.save(os.path.join(dir, "filtered_subsample.npy"), out)


register_patch(
    name="obspy",
    targets=[
        ("obspy.core.trace", "_get_function_from_entry_point",
         fast_get_function_from_entry_point),
        ("obspy.signal.filter", "bandpass", fast_bandpass),
        ("obspy.core.stream.Stream", "filter", fast_stream_filter),
    ],
    smoke={"load": _smoke_load, "call": _smoke_call, "save": _smoke_save},
    tested_against="obspy 1.5.0",
    tested_upstream_versions={"obspy": ["1.5.0"]},
)
