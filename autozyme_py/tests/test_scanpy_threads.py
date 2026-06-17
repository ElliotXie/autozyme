"""Scanpy private thread resolvers delegate to the unified ``auto_threads``.

The three scanpy ``_n_threads()`` resolvers (_normalize, _neighbors, _umap)
used to fall back to raw ``os.cpu_count()``. They now defer their no-knob
fallback to ``auto_threads(default=None)`` (scale-to-hardware, capped at 16) so
there is a single default-thread story. Each keeps its own patch-specific knob
(SCANPY_TURBO_THREADS / AUTOZYME_SCBLAS_*_THREADS) ahead of the fallback, and
the harness env vars still win (so benchmark pinning is untouched).
"""
from __future__ import annotations

import os

import pytest

from autozyme._threads import auto_threads
from autozyme.scanpy import _neighbors, _normalize, _umap

# (module, its highest-priority patch-specific knob)
_RESOLVERS = [
    (_normalize, "SCANPY_TURBO_THREADS"),
    (_neighbors, "AUTOZYME_SCBLAS_NEIGHBORS_THREADS"),
    (_umap, "AUTOZYME_SCBLAS_UMAP_THREADS"),
]

_ALL_THREAD_ENV = (
    "ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS", "AUTOZYMER_THREADS",
    "SCANPY_TURBO_THREADS",
    "AUTOZYME_SCBLAS_NEIGHBORS_THREADS", "AUTOZYME_SCBLAS_UMAP_THREADS",
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Clean env + module option so the fallback path is exercised."""
    for var in _ALL_THREAD_ENV:
        monkeypatch.delenv(var, raising=False)
    import autozyme._threads as _t
    saved = _t._AUTOZYME_THREADS_OPTION
    _t._AUTOZYME_THREADS_OPTION = None
    yield
    _t._AUTOZYME_THREADS_OPTION = saved


@pytest.mark.parametrize("mod,_knob", _RESOLVERS)
def test_fallback_delegates_to_auto_threads(mod, _knob, monkeypatch):
    # Big box, no override: must match auto_threads(default=None) and be capped
    # at 16 -- i.e. NOT the old raw os.cpu_count() (64).
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert mod._n_threads() == auto_threads(default=None) == 16


@pytest.mark.parametrize("mod,_knob", _RESOLVERS)
def test_fallback_scales_on_small_box(mod, _knob, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert mod._n_threads() == auto_threads(default=None) == 7  # cpu_count - 1


@pytest.mark.parametrize("mod,knob", _RESOLVERS)
def test_patch_specific_knob_wins(mod, knob, monkeypatch):
    # Each resolver's own knob still takes precedence over the fallback.
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setenv(knob, "7")
    assert mod._n_threads() == 7


@pytest.mark.parametrize("mod,_knob", _RESOLVERS)
def test_harness_pin_wins(mod, _knob, monkeypatch):
    # Benchmark-safety: the attest harness pins ZYME_THREADS; it must still win
    # so the convergence cannot perturb a measured sweep.
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setenv("ZYME_THREADS", "1")
    assert mod._n_threads() == 1
