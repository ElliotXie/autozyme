"""Unit tests for zyme.commands.profile.backends (probe helpers + pipeline-arg
builders) and zyme.commands.profile.archive (run-dir prep, JSON/log writers,
latest symlink, segment sanitizing).

backends.resolve() itself is already covered by test_profile_fallback.py, so
here we target the un-covered helpers: _probe_python_pkg / _probe_r_pkg /
_resolve_python_bin (monkeypatched subprocess boundary), and the
memray_pipeline_args / scalene_pipeline_args formatters.

archive.py is currently only exercised through subprocess archive tests; the
pure functions get direct unit coverage here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from zyme.commands.profile import backends, archive


# ===========================================================================
# backends.py — probe helpers
# ===========================================================================

class _Result:
    def __init__(self, returncode):
        self.returncode = returncode


def test_probe_python_pkg_importable(monkeypatch):
    monkeypatch.setattr(backends, "_resolve_python_bin", lambda ex: "/fake/python")
    monkeypatch.setattr(backends.subprocess, "run",
                        lambda c, **k: _Result(0))
    assert backends._probe_python_pkg("memray", None) is True


def test_probe_python_pkg_not_importable(monkeypatch):
    monkeypatch.setattr(backends, "_resolve_python_bin", lambda ex: "/fake/python")
    monkeypatch.setattr(backends.subprocess, "run",
                        lambda c, **k: _Result(1))
    assert backends._probe_python_pkg("nopkg", None) is False


def test_probe_python_pkg_no_interpreter(monkeypatch):
    monkeypatch.setattr(backends, "_resolve_python_bin", lambda ex: None)
    assert backends._probe_python_pkg("memray", None) is False


def test_probe_python_pkg_subprocess_error(monkeypatch):
    monkeypatch.setattr(backends, "_resolve_python_bin", lambda ex: "/fake/python")

    def boom(cmd, **kw):
        raise FileNotFoundError("python")

    monkeypatch.setattr(backends.subprocess, "run", boom)
    assert backends._probe_python_pkg("memray", None) is False


def test_probe_r_pkg_installed(monkeypatch):
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _Result(0)

    monkeypatch.setattr(backends.subprocess, "run", fake_run)
    assert backends._probe_r_pkg("profvis", {"rscript": "/opt/Rscript"}) is True
    assert captured["cmd"][0] == "/opt/Rscript"
    assert "profvis" in captured["cmd"][2]


def test_probe_r_pkg_missing(monkeypatch):
    monkeypatch.setattr(backends.subprocess, "run", lambda c, **k: _Result(1))
    assert backends._probe_r_pkg("profvis", None) is False


def test_probe_r_pkg_default_rscript(monkeypatch):
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _Result(0)

    monkeypatch.setattr(backends.subprocess, "run", fake_run)
    backends._probe_r_pkg("profvis", None)
    assert captured["cmd"][0] == "Rscript"


def test_probe_r_pkg_subprocess_error(monkeypatch):
    def boom(cmd, **kw):
        raise OSError("no R")
    monkeypatch.setattr(backends.subprocess, "run", boom)
    assert backends._probe_r_pkg("profvis", None) is False


# ===========================================================================
# backends.py — _resolve_python_bin
# ===========================================================================

def test_resolve_python_bin_no_executor_uses_sys_executable():
    assert backends._resolve_python_bin(None) == sys.executable
    assert backends._resolve_python_bin({}) == sys.executable


def test_resolve_python_bin_executor_python(monkeypatch):
    import zyme.runner as runner
    monkeypatch.setattr(runner, "_resolve_python", lambda name: f"/envs/{name}/bin/python")
    assert backends._resolve_python_bin({"python": "myenv"}) == "/envs/myenv/bin/python"


def test_resolve_python_bin_resolution_failure(monkeypatch):
    import zyme.runner as runner

    def boom(name):
        raise RuntimeError("env not found")

    monkeypatch.setattr(runner, "_resolve_python", boom)
    assert backends._resolve_python_bin({"python": "ghost"}) is None


# ===========================================================================
# backends.py — pipeline-arg builders
# ===========================================================================

def test_memray_pipeline_args(tmp_path):
    out = tmp_path / "memray.bin"
    args = backends.memray_pipeline_args(out)
    assert args == ["-m", "memray", "run", "-o", str(out), "-f", "--native"]
    # --native is essential (captures C frames) — must be present.
    assert "--native" in args


def test_scalene_pipeline_args(tmp_path):
    out = tmp_path / "scalene.json"
    args = backends.scalene_pipeline_args(out)
    assert args[:3] == ["-m", "scalene", "run"]
    assert "-o" in args and str(out) in args
    assert "--memory" in args
    assert "--profile-system-libraries" in args
    # the '---' separator must terminate the scalene flags.
    assert args[-1] == "---"


# ===========================================================================
# archive.py — _safe_seg
# ===========================================================================

def test_safe_seg_sanitizes_and_truncates():
    assert archive._safe_seg("cpu") == "cpu"
    assert archive._safe_seg("ood/large weird:tier") == "ood-large-weird-tier"
    # empty -> "x" placeholder (only the empty-string case triggers `or "x"`).
    assert archive._safe_seg("") == "x"
    # all-disallowed collapses to a single "-" (truthy, so no placeholder).
    assert archive._safe_seg("***") == "-"
    # truncation at 40 chars.
    assert len(archive._safe_seg("a" * 100)) == 40


def test_safe_seg_keeps_allowed_chars():
    assert archive._safe_seg("v1.2_beta-3") == "v1.2_beta-3"


# ===========================================================================
# archive.py — prepare_run_dir
# ===========================================================================

def test_prepare_run_dir_archive_creates_timestamped(tmp_path):
    run_dir, rel = archive.prepare_run_dir(
        tmp_path, backend="cpu", tier="tiny", archive=True)
    assert run_dir.exists() and run_dir.is_dir()
    assert run_dir.parent.name == "profile_history"
    # name convention: <ts>_<backend>_<tier>
    assert "cpu" in run_dir.name
    assert "tiny" in run_dir.name
    assert rel.startswith("profile_history/")
    assert rel == run_dir.relative_to(tmp_path).as_posix()


def test_prepare_run_dir_collision_appends_suffix(tmp_path, monkeypatch):
    # Freeze the timestamp so two prepare calls collide.
    import zyme.commands.profile.archive as arch

    class _FrozenDT:
        @staticmethod
        def now(tz=None):
            import datetime as _d
            return _d.datetime(2026, 1, 1, 0, 0, 0, tzinfo=_d.timezone.utc)

    monkeypatch.setattr(arch, "datetime", _FrozenDT)
    d1, _ = archive.prepare_run_dir(tmp_path, backend="cpu", tier="tiny")
    d2, _ = archive.prepare_run_dir(tmp_path, backend="cpu", tier="tiny")
    assert d1 != d2
    # second dir gets a -2 suffix.
    assert d2.name.endswith("-2")


def test_prepare_run_dir_no_archive_uses_current(tmp_path):
    run_dir, rel = archive.prepare_run_dir(
        tmp_path, backend="cpu", tier="tiny", archive=False)
    assert run_dir.name == "current"
    assert rel == "profile_history/current"


def test_prepare_run_dir_no_archive_overwrites_current(tmp_path):
    d1, _ = archive.prepare_run_dir(tmp_path, backend="cpu", tier="tiny", archive=False)
    (d1 / "stale.txt").write_text("old")
    d2, _ = archive.prepare_run_dir(tmp_path, backend="cpu", tier="tiny", archive=False)
    # current/ is wiped and recreated -> stale file gone.
    assert d1 == d2
    assert not (d2 / "stale.txt").exists()


# ===========================================================================
# archive.py — write_profile_json / write_run_log
# ===========================================================================

def test_write_profile_json_roundtrips(tmp_path):
    run_dir, _ = archive.prepare_run_dir(tmp_path, backend="cpu", tier="tiny")
    data = {"backend": "cpu", "hotspots": [], "obj": object()}
    out = archive.write_profile_json(run_dir, data)
    assert out == run_dir / "profile.json"
    loaded = json.loads(out.read_text())
    assert loaded["backend"] == "cpu"
    # non-serializable values fall back to str() via default=str.
    assert isinstance(loaded["obj"], str)


def test_write_run_log(tmp_path):
    run_dir, _ = archive.prepare_run_dir(tmp_path, backend="cpu", tier="tiny")
    out = archive.write_run_log(run_dir, "log contents\nline2\n")
    assert out == run_dir / "run.log"
    assert out.read_text() == "log contents\nline2\n"


# ===========================================================================
# archive.py — update_latest_symlink
# ===========================================================================

def test_update_latest_symlink_points_at_run(tmp_path):
    hist = tmp_path / "profile_history"
    hist.mkdir()
    run_dir = hist / "20260101_cpu_tiny"
    run_dir.mkdir()
    archive.update_latest_symlink(hist, run_dir)
    link = hist / "latest"
    assert link.is_symlink()
    # relative target so the link survives moves.
    assert Path(link.readlink() if hasattr(link, "readlink") else
                __import__("os").readlink(link)).name == run_dir.name
    # resolves to the run dir.
    assert link.resolve() == run_dir.resolve()


def test_update_latest_symlink_replaces_existing(tmp_path):
    hist = tmp_path / "profile_history"
    hist.mkdir()
    r1 = hist / "run1"
    r2 = hist / "run2"
    r1.mkdir()
    r2.mkdir()
    archive.update_latest_symlink(hist, r1)
    archive.update_latest_symlink(hist, r2)
    assert (hist / "latest").resolve() == r2.resolve()


def test_update_latest_symlink_silent_on_oserror(tmp_path, monkeypatch):
    hist = tmp_path / "profile_history"
    hist.mkdir()
    run_dir = hist / "run1"
    run_dir.mkdir()

    # Force symlink_to to raise; the helper must swallow it (best-effort).
    orig_symlink_to = Path.symlink_to

    def boom(self, target, **kw):
        raise OSError("symlinks unsupported")

    monkeypatch.setattr(Path, "symlink_to", boom)
    # should not raise.
    archive.update_latest_symlink(hist, run_dir)
