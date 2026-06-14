"""Wave-4 wrapper/smoke-line tests for autozyme.obspy.

Wave-1 (`_unit`) covered `fast_bandpass` numerics + `_cached_iir_sos` + the
entry-point lru_cache; wave-2 (`_e2e`) covered `fast_stream_filter`'s parallel
and serial branches. The COVERAGE-VISIBLE lines still missing are:

  - line 57: the ``low > 1`` ValueError inside ``fast_bandpass`` (only
    reachable when freqmin is above Nyquist but freqmax is NOT, i.e. an
    inverted-band call — wave-1's "low above Nyquist" test used freqmin>freqmax
    with freqmax ALSO above Nyquist, so it returned the highpass at line 54
    instead).
  - lines 75-76: the ``except Exception`` guard around the
    ``_ENTRY_POINT_CACHE.clear()`` in ``fast_stream_filter``.
  - the smoke recipe ``_smoke_load`` / ``_smoke_call`` / ``_smoke_save``
    (lines 98-146), driven against a tiny synthetic JSON manifest in a tmpdir.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

obspy = pytest.importorskip("obspy")
yaml = pytest.importorskip("yaml")

import autozyme
from autozyme import obspy as azobspy


# --------------------------------------------------------------------------
# line 57: low > 1 (freqmin above Nyquist) but high <= Nyquist -> ValueError
# --------------------------------------------------------------------------
def test_fast_bandpass_low_above_nyquist_but_high_below_raises():
    """An inverted band (freqmin > freqmax) where freqmin is above Nyquist but
    freqmax is below it skips the highpass branch (high-1 <= -1e-6) and trips
    the explicit ``low > 1`` ValueError at line 57."""
    data = np.zeros(128)
    df = 100.0  # Nyquist = 50
    # freqmax=40 < Nyquist -> high=0.8, high-1 = -0.2 <= -1e-6 -> no highpass.
    # freqmin=60 > Nyquist -> low = 1.2 > 1 -> ValueError.
    with pytest.raises(ValueError, match="low corner frequency is above Nyquist"):
        azobspy.fast_bandpass(data, freqmin=60.0, freqmax=40.0, df=df, corners=4)


# --------------------------------------------------------------------------
# lines 75-76: _ENTRY_POINT_CACHE.clear() except-guard
# --------------------------------------------------------------------------
def test_stream_filter_cache_clear_guard_swallows_failure(monkeypatch):
    """If the _ENTRY_POINT_CACHE .clear() raises, fast_stream_filter must still
    filter the stream (the try/except at 72-76 must swallow it). We make .clear
    raise rather than deleting the cache object, so the per-trace dispatch the
    filtering itself relies on stays intact."""
    from obspy import Stream, Trace, UTCDateTime
    import obspy.core.util.misc as misc

    autozyme.activate("obspy")

    # Make the cache's clear() blow up so the except branch (lines 75-76) runs,
    # without removing the cache (which obspy's per-trace dispatch still reads).
    if hasattr(misc, "_ENTRY_POINT_CACHE"):
        class _BoomDict(dict):
            def clear(self):
                raise RuntimeError("boom")
        monkeypatch.setattr(misc, "_ENTRY_POINT_CACHE",
                            _BoomDict(misc._ENTRY_POINT_CACHE), raising=False)

    rng = np.random.default_rng(0)
    t0 = UTCDateTime(2024, 1, 1)
    traces = [
        Trace(data=rng.standard_normal(256).astype("float64"),
              header={"sampling_rate": 100.0, "starttime": t0,
                      "network": "XX", "station": f"S{i}", "channel": "BHZ"})
        for i in range(3)
    ]
    st = Stream(traces)
    out = st.filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4)
    assert out is st
    assert all(tr.stats.npts == 256 for tr in st)


# --------------------------------------------------------------------------
# smoke recipe: _smoke_load / _smoke_call / _smoke_save  (lines 98-146)
# --------------------------------------------------------------------------
def _make_task_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = {
        "tier": "small",
        "n_traces": 4,
        "n_samples": 2048,
        "sampling_rate": 100.0,
        "seed": 42,
        "n_repeats": 2,
        "filter_kwargs": {
            "type": "bandpass",
            "freqmin": 1.0,
            "freqmax": 10.0,
            "corners": 4,
            "zerophase": True,
        },
        "subsample_stride": 8,
    }
    (data_dir / "small.json").write_text(__import__("json").dumps(manifest),
                                         encoding="utf-8")
    task = {"datasets": [{"tier": "small", "name": "synth",
                          "path": "./data/small.json"}]}
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    return str(tmp_path)


def test_smoke_load_builds_stream_and_kwargs(tmp_path):
    """_smoke_load reads the manifest, synthesizes a seeded Stream, and returns
    the filter kwargs / repeats / stride (covers lines 98-128)."""
    from obspy import Stream

    task_dir = _make_task_dir(tmp_path)
    out = azobspy._smoke_load(task_dir, "small")
    assert isinstance(out["stream"], Stream)
    assert len(out["stream"]) == 4
    assert out["filter_kwargs"]["type"] == "bandpass"
    assert out["n_repeats"] == 2
    assert out["subsample_stride"] == 8
    # Seeded: re-loading produces identical trace data.
    out2 = azobspy._smoke_load(task_dir, "small")
    np.testing.assert_array_equal(out["stream"][0].data, out2["stream"][0].data)


def test_smoke_call_filters_without_mutating_input(tmp_path):
    """_smoke_call copies the stream and applies the filter n_repeats times,
    leaving the input stream untouched (covers lines 132-140)."""
    task_dir = _make_task_dir(tmp_path)
    inputs = azobspy._smoke_load(task_dir, "small")
    pristine = inputs["stream"][0].data.copy()
    result = azobspy._smoke_call(inputs)
    assert len(result["stream"]) == 4
    assert result["stride"] == 8
    # The input stream (inputs["stream"]) was NOT mutated — call() copies.
    np.testing.assert_array_equal(inputs["stream"][0].data, pristine)


def test_smoke_save_writes_subsampled_npy(tmp_path):
    """_smoke_save stacks the stride-subsampled traces into filtered_subsample.npy
    (covers lines 144-146)."""
    task_dir = _make_task_dir(tmp_path)
    inputs = azobspy._smoke_load(task_dir, "small")
    result = azobspy._smoke_call(inputs)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    azobspy._smoke_save(result, str(out_dir))
    arr = np.load(out_dir / "filtered_subsample.npy")
    n_samples = 2048
    expected_cols = len(range(0, n_samples, 8))
    assert arr.shape == (4, expected_cols)


def test_smoke_call_matches_unpatched_baseline(tmp_path):
    """The patched smoke filter result matches a vanilla (zyme=disabled)
    Stream.filter on the same synthetic stream within FP tolerance."""
    task_dir = _make_task_dir(tmp_path)
    autozyme.activate("obspy")
    inputs = azobspy._smoke_load(task_dir, "small")
    patched = azobspy._smoke_call(inputs)
    with autozyme.disabled():
        ref_inputs = azobspy._smoke_load(task_dir, "small")
        ref = azobspy._smoke_call(ref_inputs)
    for tr_p, tr_r in zip(patched["stream"], ref["stream"]):
        np.testing.assert_allclose(tr_p.data, tr_r.data, rtol=1e-6, atol=1e-9)
