"""Back-compat tests: the OLD profile path (`ZYME_PROFILE=1` env var, no
explicit backend) must still work.

Pre-`zyme profile` subcommand, profiling was triggered by setting
ZYME_PROFILE=1 and calling `zyme run`; the helper's `with_profile()` block
in pipeline/run.py would activate cProfile / Rprof. Adding the new
backend dispatch must not break that path: ZYME_PROFILE=1 with no
ZYME_PROFILE_BACKEND set must default to cpu (cProfile / Rprof).

We exercise this at the helper level (no subprocess), since that's where
the dispatch lives. Spawning a full pipeline would conflate this test
with task-runtime issues.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def env_profile_only(monkeypatch):
    """Set ZYME_PROFILE=1 with no BACKEND override (the old path)."""
    monkeypatch.setenv("ZYME_PROFILE", "1")
    monkeypatch.delenv("ZYME_PROFILE_BACKEND", raising=False)
    monkeypatch.delenv("ZYME_SCALENE_ACTIVE", raising=False)
    monkeypatch.delenv("ZYME_MEMRAY_ACTIVE", raising=False)


def test_with_profile_defaults_to_cprofile(env_profile_only, tmp_path, monkeypatch):
    """With ZYME_PROFILE=1 and no BACKEND set, with_profile() must run
    cProfile and write profile.out — the old behavior."""
    monkeypatch.chdir(tmp_path)
    from zyme.helpers import with_profile

    # Sanity: do real CPU work so cProfile has something to capture.
    with with_profile():
        s = 0
        for i in range(200_000):
            s += i * i

    profile_out = tmp_path / "profile.out"
    assert profile_out.exists(), \
        f"with_profile() must produce profile.out at cwd in default cpu mode " \
        f"(found: {list(tmp_path.iterdir())})"
    assert profile_out.stat().st_size > 0, "profile.out must not be empty"


def test_with_profile_unset_is_noop(monkeypatch, tmp_path):
    """When ZYME_PROFILE is unset, with_profile() must be a no-op
    (no profile.out written, zero overhead)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ZYME_PROFILE", raising=False)
    from zyme.helpers import with_profile

    with with_profile():
        sum(range(10_000))

    assert not (tmp_path / "profile.out").exists(), \
        "no profile.out should be written when ZYME_PROFILE is unset"


def test_with_profile_explicit_cpu_backend(monkeypatch, tmp_path):
    """ZYME_PROFILE=1 + ZYME_PROFILE_BACKEND=cpu should match the unset-
    backend path (cProfile)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZYME_PROFILE", "1")
    monkeypatch.setenv("ZYME_PROFILE_BACKEND", "cpu")
    from zyme.helpers import with_profile

    with with_profile():
        sum(i * i for i in range(200_000))

    assert (tmp_path / "profile.out").exists()


def test_with_profile_unknown_backend_falls_back_to_cpu(
        monkeypatch, tmp_path, capsys):
    """An unknown ZYME_PROFILE_BACKEND should warn and fall back to cpu —
    not crash. Defensive: if a stale env var leaks through, profiling
    should still produce something rather than silently break the run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZYME_PROFILE", "1")
    monkeypatch.setenv("ZYME_PROFILE_BACKEND", "rumpelstiltskin")
    from zyme.helpers import with_profile

    with with_profile():
        sum(range(100_000))

    captured = capsys.readouterr()
    assert "unknown" in captured.err.lower() or "rumpelstiltskin" in captured.err
    assert (tmp_path / "profile.out").exists(), \
        "fallback to cpu should still produce profile.out"
