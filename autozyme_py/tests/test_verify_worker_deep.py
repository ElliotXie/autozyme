"""Deep tests for autozyme._verify_worker error/entry paths.

test_framework_misc_unit.py (wave 1) covers arg-parsing, _peak_rss_mb type,
and main() happy path (baseline + patched). This file pushes the remaining
gap: the error branches inside main() (unregistered patch, no smoke recipe),
the psutil fallback in _peak_rss_mb, and the ``python -m autozyme._verify_worker``
module entry point (the ``if __name__ == "__main__"`` guard) via a real
subprocess.

All use the stdlib-only synthetic ``_test_json`` patch where an upstream is
needed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import autozyme
from autozyme import _verify_worker as W
from autozyme._core import _REGISTRY, _import_submodule


@pytest.fixture(autouse=True)
def _ensure_test_patch_registered():
    _import_submodule("_test_json")
    yield
    if _REGISTRY.get("_test_json") and _REGISTRY["_test_json"].injected:
        autozyme.deactivate("_test_json")


# ==========================================================================
# main() error branches
# ==========================================================================
def test_main_unregistered_patch_raises(tmp_path):
    """A patch name that imports but doesn't exist anywhere -> ImportError
    from _import_submodule (line 82 region)."""
    out_dir = tmp_path / "out"
    with pytest.raises((ImportError, RuntimeError)):
        W.main(["--patch", "definitely_not_a_real_patch_zzz",
                "--task-dir", str(tmp_path), "--tier", "tiny",
                "--output-dir", str(out_dir)])


def test_main_patch_without_smoke_raises(tmp_path, monkeypatch):
    """A registered patch whose smoke is None -> RuntimeError 'no smoke
    recipe' (line 89)."""
    # Register a smoke-less patch on an unclaimed json attr.
    autozyme.register_patch("worker_nosmoke",
                            [("json", "loads", lambda s: s)])
    try:
        out_dir = tmp_path / "out"
        with pytest.raises(RuntimeError, match="no smoke recipe"):
            W.main(["--patch", "worker_nosmoke",
                    "--task-dir", str(tmp_path), "--tier", "tiny",
                    "--output-dir", str(out_dir)])
    finally:
        _REGISTRY.pop("worker_nosmoke", None)


def test_main_returns_zero_and_writes_json(tmp_path, capsys):
    """Baseline run prints exactly one JSON line with elapsed_sec + peak_mb."""
    out_dir = tmp_path / "out"
    rc = W.main(["--patch", "_test_json", "--task-dir", str(tmp_path),
                 "--tier", "tiny", "--output-dir", str(out_dir)])
    assert rc == 0
    last = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()][-1]
    payload = json.loads(last)
    assert set(payload) >= {"elapsed_sec", "peak_mb"}
    assert payload["elapsed_sec"] >= 0


# ==========================================================================
# _peak_rss_mb — value is sane on this POSIX host
# ==========================================================================
def test_peak_rss_mb_positive_on_posix():
    v = W._peak_rss_mb()
    # mac/linux: resource.getrusage path -> a positive float
    assert v is None or (isinstance(v, float) and v > 0)


def test_peak_rss_mb_psutil_fallback(monkeypatch):
    """Force the resource import to fail so the psutil fallback (50-55) runs.

    If psutil is present it returns a float; if not, None. Either way no
    raise — that's the contract."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "resource":
            raise ImportError("forced: no resource module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    v = W._peak_rss_mb()
    assert v is None or isinstance(v, float)


# ==========================================================================
# `python -m autozyme._verify_worker` entry point (line 132 + main end-to-end)
# ==========================================================================
def test_module_entrypoint_subprocess(tmp_path):
    """Run the worker exactly as _verify._run_worker does: as a fresh-Python
    `-m` subprocess. Exercises the __main__ guard and the full main() path."""
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "autozyme._verify_worker",
         "--patch", "_test_json", "--task-dir", str(tmp_path),
         "--tier", "tiny", "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    last = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
    payload = json.loads(last)
    assert "elapsed_sec" in payload
    assert (out_dir / "output.txt").read_text().strip() == "6"


def test_module_entrypoint_subprocess_activate(tmp_path):
    """--activate path through the subprocess entry point."""
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "autozyme._verify_worker",
         "--patch", "_test_json", "--task-dir", str(tmp_path),
         "--tier", "tiny", "--output-dir", str(out_dir), "--activate"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    last = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
    assert "elapsed_sec" in json.loads(last)


def test_module_entrypoint_subprocess_bad_patch_nonzero(tmp_path):
    """An unregistered patch makes the worker subprocess exit non-zero with a
    traceback on stderr (the path _run_worker turns into a RuntimeError)."""
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "autozyme._verify_worker",
         "--patch", "no_such_patch_zzz", "--task-dir", str(tmp_path),
         "--tier", "tiny", "--output-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "no_such_patch_zzz" in proc.stderr or "could not load" in proc.stderr
