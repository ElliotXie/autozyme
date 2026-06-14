"""End-to-end / wrapper-line tests for autozyme.obspy.

Wave-1 (`test_obspy_unit.py`) tested `fast_bandpass`, `_cached_iir_sos`, and the
entry-point cache directly. The existing `test_obspy.py` drives `bandpass`. This
file covers the remaining COVERAGE-VISIBLE wrapper that neither hits:

  - `fast_stream_filter`: the multi-trace ThreadPoolExecutor branch AND the
    single-trace serial branch, plus the _ENTRY_POINT_CACHE clear.
  - End-to-end parity of a patched Stream.filter vs vanilla.
  - activate/restore lifecycle across all three obspy bind sites.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
obspy = pytest.importorskip("obspy")

import autozyme
from autozyme import obspy as azobspy


def _make_stream(n_traces, n_samples=512, sr=100.0):
    from obspy import Stream, Trace, UTCDateTime

    rng = np.random.default_rng(0)
    t0 = UTCDateTime(2024, 1, 1)
    traces = []
    for i in range(n_traces):
        data = rng.standard_normal(n_samples).astype("float64")
        tr = Trace(
            data=data,
            header={"sampling_rate": sr, "starttime": t0,
                    "network": "XX", "station": f"S{i:03d}", "channel": "BHZ"},
        )
        traces.append(tr)
    return Stream(traces)


def test_stream_filter_multi_trace_threadpool_matches_vanilla():
    """fast_stream_filter's ThreadPoolExecutor path (len(stream) > 1) returns
    the same filtered traces as the vanilla serial Stream.filter."""
    autozyme.activate("obspy")
    st_fast = _make_stream(6)
    st_van = st_fast.copy()

    out = st_fast.filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4,
                         zerophase=False)
    assert out is st_fast  # Stream.filter returns self

    with autozyme.disabled():
        st_van.filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4,
                      zerophase=False)

    for tr_f, tr_v in zip(st_fast, st_van):
        np.testing.assert_allclose(tr_f.data, tr_v.data, rtol=1e-6, atol=1e-9)


def test_stream_filter_single_trace_serial_branch():
    """A 1-trace stream takes the serial (max_workers<=1) branch."""
    autozyme.activate("obspy")
    st = _make_stream(1)
    n0 = st[0].stats.npts
    out = st.filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4)
    assert out is st
    assert st[0].stats.npts == n0  # filtering preserves length


def test_stream_filter_forces_threads_one_serial(monkeypatch):
    """With ZYME_THREADS=1, even a multi-trace stream takes the serial path
    (max_workers<=1) — still produces correct output."""
    monkeypatch.setenv("ZYME_THREADS", "1")
    autozyme.activate("obspy")
    st_fast = _make_stream(4)
    st_van = st_fast.copy()
    st_fast.filter("bandpass", freqmin=2.0, freqmax=8.0, corners=4)
    with autozyme.disabled():
        st_van.filter("bandpass", freqmin=2.0, freqmax=8.0, corners=4)
    for tr_f, tr_v in zip(st_fast, st_van):
        np.testing.assert_allclose(tr_f.data, tr_v.data, rtol=1e-6, atol=1e-9)


def test_activate_restore_all_three_targets():
    """activate binds all three obspy targets; deactivate restores them."""
    import obspy.core.trace as trace_mod
    import obspy.signal.filter as filt_mod
    from obspy.core.stream import Stream

    autozyme.deactivate("obspy")
    orig_ep = trace_mod._get_function_from_entry_point
    orig_bandpass = filt_mod.bandpass
    orig_stream_filter = Stream.filter

    assert autozyme.activate("obspy") is True
    assert filt_mod.bandpass is not orig_bandpass
    assert Stream.filter is not orig_stream_filter
    assert getattr(filt_mod.bandpass, "__autozyme_fast__", None) is azobspy.fast_bandpass

    info = autozyme.inspect("obspy")
    assert info["status"] == "active"
    assert len(info["targets"]) == 3

    autozyme.deactivate("obspy")
    assert filt_mod.bandpass is orig_bandpass
    assert Stream.filter is orig_stream_filter
    assert trace_mod._get_function_from_entry_point is orig_ep


def test_trace_filter_uses_patched_bandpass_after_activate():
    """A single Trace.filter('bandpass', ...) routes through the patched
    bandpass and matches vanilla — confirms the _ENTRY_POINT_CACHE clear in
    fast_stream_filter doesn't break per-trace dispatch."""
    autozyme.activate("obspy")
    st = _make_stream(2)
    tr_fast = st[0].copy()
    tr_van = st[0].copy()
    tr_fast.filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4)
    with autozyme.disabled():
        tr_van.filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4)
    np.testing.assert_allclose(tr_fast.data, tr_van.data, rtol=1e-6, atol=1e-9)
