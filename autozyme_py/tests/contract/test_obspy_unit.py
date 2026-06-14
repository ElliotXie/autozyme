"""Unit tests for the pure scipy-signal helpers in autozyme.obspy.

The contract test (test_obspy.py) drives Stream.filter end to end. Here we test
the self-contained filtering helpers directly against scipy references:

  - _cached_iir_sos       cached SOS coefficient design, vs scipy.signal.iirfilter
  - fast_bandpass         bandpass filtering, vs scipy.signal.sosfilt
  - fast_get_function_from_entry_point   lru_cache wrapper

obspy must import for the module to load; these helpers use only scipy.signal.
"""
from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")
sig = pytest.importorskip("scipy.signal")
pytest.importorskip("obspy")

from autozyme import obspy as azobspy


# --------------------------------------------------------------------------
# _cached_iir_sos
# --------------------------------------------------------------------------
def test_cached_iir_sos_matches_scipy_iirfilter():
    corners, df = 4, 100.0
    freqs = (5.0, 20.0)
    sos = azobspy._cached_iir_sos(corners, freqs, df, None, None, "band", "butter")
    fe = 0.5 * df
    ref = sig.iirfilter(corners, [f / fe for f in freqs], rp=None, rs=None,
                        btype="band", ftype="butter", output="sos")
    np.testing.assert_allclose(sos, ref, rtol=1e-12, atol=1e-14)


def test_cached_iir_sos_is_memoized():
    a = azobspy._cached_iir_sos(4, (5.0, 20.0), 100.0, None, None, "band", "butter")
    b = azobspy._cached_iir_sos(4, (5.0, 20.0), 100.0, None, None, "band", "butter")
    # lru_cache returns the identical array object on a cache hit.
    assert a is b


def test_cached_iir_sos_single_freq_highpass():
    # A single normalized frequency exercises the len==1 unwrap branch.
    sos = azobspy._cached_iir_sos(2, (10.0,), 100.0, None, None, "highpass", "butter")
    ref = sig.iirfilter(2, 10.0 / 50.0, btype="highpass", ftype="butter", output="sos")
    np.testing.assert_allclose(sos, ref, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------
# fast_bandpass
# --------------------------------------------------------------------------
def test_fast_bandpass_matches_scipy_sosfilt():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(2000)
    df = 100.0
    out = azobspy.fast_bandpass(data, 5.0, 20.0, df, corners=4, zerophase=False)
    sos = sig.iirfilter(4, [5.0 / 50.0, 20.0 / 50.0], btype="band",
                        ftype="butter", output="sos")
    ref = sig.sosfilt(sos, data)
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-12)


def test_fast_bandpass_zerophase_is_forward_backward():
    rng = np.random.default_rng(1)
    data = rng.standard_normal(1500)
    df = 100.0
    out = azobspy.fast_bandpass(data, 5.0, 20.0, df, corners=4, zerophase=True)
    sos = sig.iirfilter(4, [5.0 / 50.0, 20.0 / 50.0], btype="band",
                        ftype="butter", output="sos")
    firstpass = np.flip(sig.sosfilt(sos, data))
    ref = np.flip(sig.sosfilt(sos, firstpass))
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-11)


def test_fast_bandpass_high_above_nyquist_warns_and_highpasses():
    rng = np.random.default_rng(2)
    data = rng.standard_normal(500)
    df = 100.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = azobspy.fast_bandpass(data, 5.0, 60.0, df, corners=4)  # 60 > Nyquist=50
        assert any("Nyquist" in str(wi.message) for wi in w)
    # Result equals an obspy highpass at freqmin.
    from obspy.signal.filter import highpass
    ref = highpass(data, freq=5.0, df=df, corners=4, ftype="butter", zerophase=False)
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-11)


def test_fast_bandpass_low_above_nyquist_raises():
    data = np.zeros(100)
    with pytest.raises(ValueError):
        # freqmin above Nyquist (low > 1) -> ValueError.
        azobspy.fast_bandpass(data, 60.0, 80.0, df=100.0)


# --------------------------------------------------------------------------
# fast_get_function_from_entry_point (lru_cache wrapper)
# --------------------------------------------------------------------------
def test_get_function_from_entry_point_caches():
    # bandpass is a registered obspy filter entry point; the wrapper memoizes
    # the resolved callable.
    f1 = azobspy.fast_get_function_from_entry_point("filter", "bandpass")
    f2 = azobspy.fast_get_function_from_entry_point("filter", "bandpass")
    assert f1 is f2
    assert callable(f1)
