"""Tests for autozyme.auto_threads — sensible thread count picker."""
from __future__ import annotations

import os

import pytest

import autozyme
from autozyme._threads import auto_threads


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Each test gets a clean env + module option state."""
    monkeypatch.delenv("AUTOZYMER_THREADS", raising=False)
    # Clear the module-level option set by set_threads()
    import autozyme._threads as _t
    saved = _t._AUTOZYME_THREADS_OPTION
    _t._AUTOZYME_THREADS_OPTION = None
    yield
    _t._AUTOZYME_THREADS_OPTION = saved


def test_env_var_wins(monkeypatch):
    monkeypatch.setenv("AUTOZYMER_THREADS", "4")
    assert auto_threads() == 4
    assert auto_threads(cap=2) == 4   # env wins over cap
    assert auto_threads(cap=100) == 4


def test_option_wins_over_cap():
    autozyme.set_threads(5)
    try:
        assert auto_threads() == 5
        assert auto_threads(cap=2) == 5
    finally:
        # set_threads also writes env vars; clean up
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
            os.environ.pop(v, None)


def test_env_wins_over_option(monkeypatch):
    monkeypatch.setenv("AUTOZYMER_THREADS", "7")
    autozyme.set_threads(3)
    try:
        assert auto_threads() == 7
    finally:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
            os.environ.pop(v, None)


def test_cap_applied_no_override():
    n = auto_threads(cap=2)
    assert 1 <= n <= 2


def test_default_path_no_override_no_cap():
    n = auto_threads()
    assert isinstance(n, int)
    assert n >= 1
    assert n <= 16   # hard ceiling


def test_invalid_env_falls_through(monkeypatch):
    monkeypatch.setenv("AUTOZYMER_THREADS", "not-a-number")
    autozyme.set_threads(3)
    try:
        # invalid env → falls through to option
        assert auto_threads() == 3
    finally:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
            os.environ.pop(v, None)


def test_zero_env_falls_through(monkeypatch):
    monkeypatch.setenv("AUTOZYMER_THREADS", "0")
    autozyme.set_threads(4)
    try:
        assert auto_threads() == 4
    finally:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
            os.environ.pop(v, None)


def test_invalid_cap_raises():
    """Non-positive / non-int cap is a programming error, not a soft hint —
    mirrors ``set_threads``'s ValueError contract so callers can't silently
    burst past their intended ceiling by passing a stringified env var."""
    import pytest
    with pytest.raises(ValueError, match="cap must be >= 1"):
        auto_threads(cap=0)
    with pytest.raises(ValueError, match="cap must be >= 1"):
        auto_threads(cap=-5)
    with pytest.raises(ValueError, match="cap must be int or None"):
        auto_threads(cap="bad")


def test_default_capped_at_16():
    # Even on a 64-core machine, default path tops out at 16.
    n = auto_threads(cap=1000)
    assert n <= 16


def test_env_bypasses_16_ceiling(monkeypatch):
    # CI thread sweeps need to be able to test counts > 16.
    monkeypatch.setenv("AUTOZYMER_THREADS", "32")
    assert auto_threads() == 32


def _clear_thread_env(monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
              "AUTOZYMER_THREADS"):
        monkeypatch.delenv(v, raising=False)


def test_conservative_default_is_four(monkeypatch):
    # No env / option override, plenty of cores -> conservative floor of 4.
    _clear_thread_env(monkeypatch)
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    assert auto_threads() == 4
    assert auto_threads(cap=8) == 4     # cap >= 4 doesn't lower the floor
    assert auto_threads(cap=2) == 2     # cap < 4 still bites


def test_default_clipped_to_core_count(monkeypatch):
    # The floor of 4 never exceeds the machine's core count.
    _clear_thread_env(monkeypatch)
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert auto_threads() == 2


def test_default_none_scales_to_hardware(monkeypatch):
    _clear_thread_env(monkeypatch)
    monkeypatch.setattr(os, "cpu_count", lambda: 12)
    assert auto_threads(default=None) == 11          # cpu_count - 1
    assert auto_threads(default=None, cap=8) == 8    # cap still bites
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert auto_threads(default=None) == 16          # 16 ceiling holds


def test_env_wins_over_default_none(monkeypatch):
    # Benchmark pinning (env) must override the scaling opt-out too.
    _clear_thread_env(monkeypatch)
    monkeypatch.setenv("ZYME_THREADS", "3")
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    assert auto_threads(default=None) == 3


def test_invalid_default_raises(monkeypatch):
    _clear_thread_env(monkeypatch)
    with pytest.raises(ValueError, match="default must be >= 1"):
        auto_threads(default=0)
    with pytest.raises(ValueError, match="default must be int or None"):
        auto_threads(default="bad")


def test_exported_at_top_level():
    # auto_threads is part of the public API
    assert hasattr(autozyme, "auto_threads")
    assert autozyme.auto_threads is auto_threads
