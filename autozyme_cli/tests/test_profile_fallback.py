"""Unit tests for backends.resolve() — fallback / refusal logic.

These don't spawn any subprocess; they monkeypatch the package-probe
helpers and assert resolve() does the right thing on the matrix:
  - cpu     → always succeeds
  - full    → falls back to cpu when scalene/profvis missing (with warning)
  - mem     → refuses (raises) when memray missing
  - native  → refuses (raises) when sample(1) unavailable / not macOS
"""
from __future__ import annotations

import pytest

from zyme.commands.profile import backends


# --------------------------------------------------------------------------
# cpu — always succeeds
# --------------------------------------------------------------------------

def test_cpu_no_probe_needed_py():
    eff, warn = backends.resolve("cpu", "py", executor=None)
    assert eff == "cpu"
    assert warn is None


def test_cpu_no_probe_needed_R():
    eff, warn = backends.resolve("cpu", "R", executor=None)
    assert eff == "cpu"
    assert warn is None


# --------------------------------------------------------------------------
# full — fallback to cpu when scalene / profvis missing
# --------------------------------------------------------------------------

def test_full_py_falls_back_when_scalene_missing(monkeypatch):
    monkeypatch.setattr(backends, "_probe_python_pkg", lambda pkg, ex: False)
    eff, warn = backends.resolve("full", "py", executor=None)
    assert eff == "cpu", "must fall back to cpu when scalene unavailable"
    assert warn is not None
    assert "scalene" in warn.lower()
    assert "pip install scalene" in warn  # install hint must be present


def test_full_py_keeps_full_when_scalene_present(monkeypatch):
    monkeypatch.setattr(backends, "_probe_python_pkg", lambda pkg, ex: True)
    eff, warn = backends.resolve("full", "py", executor=None)
    assert eff == "full"
    assert warn is None


def test_full_R_falls_back_when_profvis_missing(monkeypatch):
    monkeypatch.setattr(backends, "_probe_r_pkg", lambda pkg, ex: False)
    eff, warn = backends.resolve("full", "R", executor=None)
    assert eff == "cpu", "must fall back to cpu when profvis missing"
    assert warn is not None
    assert "profvis" in warn.lower()
    assert "install.packages" in warn


def test_full_R_keeps_full_when_profvis_present(monkeypatch):
    monkeypatch.setattr(backends, "_probe_r_pkg", lambda pkg, ex: True)
    eff, warn = backends.resolve("full", "R", executor=None)
    assert eff == "full"
    assert warn is None


# --------------------------------------------------------------------------
# mem — REFUSES (raises) instead of falling back. Different signal axis.
# --------------------------------------------------------------------------

def test_mem_py_refuses_when_memray_missing(monkeypatch):
    monkeypatch.setattr(backends, "_probe_python_pkg", lambda pkg, ex: False)
    with pytest.raises(RuntimeError, match=r"backend=mem requires memray"):
        backends.resolve("mem", "py", executor=None)


def test_mem_py_succeeds_when_memray_present(monkeypatch):
    monkeypatch.setattr(backends, "_probe_python_pkg", lambda pkg, ex: True)
    eff, warn = backends.resolve("mem", "py", executor=None)
    assert eff == "mem"
    assert warn is None


def test_mem_R_no_probe_needed():
    """R mem uses stdlib Rprof+memory, no optional dep — should succeed
    without any probing."""
    eff, warn = backends.resolve("mem", "R", executor=None)
    assert eff == "mem"
    assert warn is None


# --------------------------------------------------------------------------
# native — REFUSES on non-macOS or when sample(1) missing.
# --------------------------------------------------------------------------

def test_native_refuses_when_unsupported(monkeypatch):
    """Force native.is_supported to fail; expect RuntimeError."""
    from zyme.commands.profile import native
    monkeypatch.setattr(native, "is_supported",
                        lambda: (False, "test: not supported"))
    with pytest.raises(RuntimeError, match=r"backend=native unavailable"):
        backends.resolve("native", "py", executor=None)


def test_native_succeeds_when_supported(monkeypatch):
    from zyme.commands.profile import native
    monkeypatch.setattr(native, "is_supported", lambda: (True, ""))
    eff, warn = backends.resolve("native", "py", executor=None)
    assert eff == "native"
    assert warn is None


# --------------------------------------------------------------------------
# Unknown backend → ValueError (not silently fallback)
# --------------------------------------------------------------------------

def test_unknown_backend_raises():
    with pytest.raises(ValueError, match=r"unknown backend"):
        backends.resolve("rumpelstiltskin", "py", executor=None)
