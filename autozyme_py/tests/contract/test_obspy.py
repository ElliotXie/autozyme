"""Contract tests for the obspy patch (seismic signal processing).

Patched surface (3 targets):
  - obspy.core.trace._get_function_from_entry_point (cache helper)
  - obspy.signal.filter.bandpass    (the heavy lifter; cached IIR SOS)
  - obspy.core.stream.Stream.filter (method binding that uses bandpass)

User-facing entry: ``trace.filter('bandpass', freqmin=, freqmax=)``
or call bandpass() directly. Contract pins:
  - bandpass(data, freqmin, freqmax, df) returns filtered array of
    same shape; zyme=False matches patched output.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
obspy = pytest.importorskip("obspy")


@pytest.fixture
def seismic_signal():
    """5-second synthetic trace at 100 Hz: 5 Hz tone + 50 Hz noise."""
    df = 100.0  # sampling rate (Hz)
    duration = 5.0
    n = int(df * duration)
    t = np.arange(n) / df
    signal = np.sin(2 * np.pi * 5 * t)      # 5 Hz tone we want to keep
    noise = 0.5 * np.sin(2 * np.pi * 50 * t)  # 50 Hz noise to filter out
    rng = np.random.default_rng(0)
    return (signal + noise + 0.05 * rng.standard_normal(n)).astype(np.float64), df


def test_bandpass_returns_same_shape_array(seismic_signal):
    """bandpass() returns a float array of the same shape as input."""
    import autozyme
    autozyme.activate("obspy")
    from obspy.signal.filter import bandpass

    data, df = seismic_signal
    out = bandpass(data, freqmin=1.0, freqmax=10.0, df=df, corners=4)
    assert isinstance(out, np.ndarray)
    assert out.shape == data.shape


def test_bandpass_zyme_false_matches_vanilla(seismic_signal):
    """Patched bandpass matches vanilla within float tolerance."""
    import autozyme
    autozyme.activate("obspy")
    from obspy.signal.filter import bandpass

    data, df = seismic_signal
    fast = bandpass(data, freqmin=1.0, freqmax=10.0, df=df, corners=4)
    with autozyme.disabled():
        ref = bandpass(data, freqmin=1.0, freqmax=10.0, df=df, corners=4)
    np.testing.assert_allclose(fast, ref, rtol=1e-6, atol=1e-9,
                               err_msg="bandpass drift patched vs vanilla")


def test_bandpass_attenuates_above_passband(seismic_signal):
    """End-to-end: 50 Hz noise should be heavily attenuated by 1-10 Hz BP."""
    import autozyme
    autozyme.activate("obspy")
    from obspy.signal.filter import bandpass

    data, df = seismic_signal
    out = bandpass(data, freqmin=1.0, freqmax=10.0, df=df, corners=4)
    # FFT to check 50 Hz bin is suppressed.
    fft_in = np.abs(np.fft.rfft(data))
    fft_out = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(data), 1.0 / df)
    # Bin nearest 50 Hz.
    idx_50 = np.argmin(np.abs(freqs - 50))
    # Bin nearest 5 Hz (passband, should survive).
    idx_5 = np.argmin(np.abs(freqs - 5))
    # 4th-order Butterworth at 5x cutoff gives ~5x attenuation here, not
    # textbook -56 dB -- the noise leaks into adjacent bins. 0.3 ratio is
    # a sanity check, not a filter-design spec.
    assert fft_out[idx_50] < 0.3 * fft_in[idx_50], "50 Hz not attenuated"
    # Passband bin should be roughly preserved (within ~20%).
    assert fft_out[idx_5] > 0.7 * fft_in[idx_5], "5 Hz attenuated unexpectedly"


def test_bandpass_zerophase_returns_same_shape(seismic_signal):
    """zerophase=True path returns same-shape output (uses forward+reverse)."""
    import autozyme
    autozyme.activate("obspy")
    from obspy.signal.filter import bandpass

    data, df = seismic_signal
    out = bandpass(data, freqmin=1.0, freqmax=10.0, df=df, corners=4,
                   zerophase=True)
    assert out.shape == data.shape
