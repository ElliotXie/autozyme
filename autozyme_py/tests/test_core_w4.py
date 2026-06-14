"""Wave-4 tests for _verify subprocess-orchestration branches reachable in-process.

_core.py is already at 100% across the existing core suites, so this file
targets the remaining REACHABLE branches of the _verify subprocess-orchestration
layer that a real worker spawn does NOT deterministically hit:

  * _run_worker stdout JSON-readback fallbacks — non-zero exit raise (170-176),
    empty-stdout raise (182-183), JSONDecodeError-skip then no-elapsed_sec
    raise (187-188, 195-198), and a valid trailing JSON line after a stray
    non-JSON print (184-194) — all driven via a faked subprocess.Popen so the
    parse logic is exercised without a multi-process timing dependency,
  * _run_evaluate's Rscript command-build branch for an evaluate.R file (87).

These deliberately avoid claiming ``json::dumps`` (the _test_json target) so
the module stays conflict-free with the _test_json-based files and with
test_core.py's own synthetic patches. No optional upstream is needed.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from autozyme import _verify as V


# --------------------------------------------------------------------------
# A minimal fake Popen that satisfies _run_worker's protocol:
#   - .stdout / .stderr are file-like with read()/readline()/close()
#   - .wait() sets nothing extra (returncode pre-set)
# --------------------------------------------------------------------------
class _FakeStream:
    def __init__(self, text: str):
        self._lines = text.splitlines(keepends=True)
        self._idx = 0
        self._text = text
        self.closed = False

    def read(self) -> str:
        return self._text

    def readline(self) -> str:
        if self._idx >= len(self._lines):
            return ""
        line = self._lines[self._idx]
        self._idx += 1
        return line

    def close(self) -> None:
        self.closed = True


class _FakePopen:
    def __init__(self, stdout_text: str, stderr_text: str, returncode: int):
        self.stdout = _FakeStream(stdout_text)
        self.stderr = _FakeStream(stderr_text)
        self.returncode = returncode

    def wait(self) -> int:
        return self.returncode


def _patch_popen(monkeypatch, stdout_text="", stderr_text="", returncode=0):
    def fake_popen(cmd, **kwargs):
        return _FakePopen(stdout_text, stderr_text, returncode)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)


def _call_run_worker(verbose=False):
    return V._run_worker("dummy", "/tmp/whatever", "tiny", "/tmp/out",
                         activate=False, verbose=verbose)


# ==========================================================================
# _run_worker — JSON readback happy path (valid trailing line after noise)
# ==========================================================================
def test_run_worker_parses_trailing_json_after_noise(monkeypatch):
    """A stray non-JSON banner line precedes the JSON payload; the parser walks
    from the bottom, skips the banner, and returns (elapsed, peak)."""
    _patch_popen(
        monkeypatch,
        stdout_text='[some upstream banner]\n{"elapsed_sec": 1.5, "peak_mb": 42.0}\n',
        returncode=0,
    )
    elapsed, peak = _call_run_worker()
    assert elapsed == 1.5
    assert peak == 42.0


def test_run_worker_peak_none_when_absent(monkeypatch):
    """peak_mb missing -> returned as None (190-194 None branch)."""
    _patch_popen(monkeypatch,
                 stdout_text='{"elapsed_sec": 0.25}\n', returncode=0)
    elapsed, peak = _call_run_worker()
    assert elapsed == 0.25
    assert peak is None


# ==========================================================================
# _run_worker — error / fallback raises
# ==========================================================================
def test_run_worker_nonzero_exit_raises(monkeypatch):
    """returncode != 0 -> RuntimeError quoting stdout + stderr (170-176)."""
    _patch_popen(monkeypatch, stdout_text="partial\n",
                 stderr_text="traceback here\n", returncode=2)
    with pytest.raises(RuntimeError, match="exited 2"):
        _call_run_worker()


def test_run_worker_empty_stdout_raises(monkeypatch):
    """No non-empty stdout lines -> 'printed no stdout' RuntimeError (182-183)."""
    _patch_popen(monkeypatch, stdout_text="   \n\n", returncode=0)
    with pytest.raises(RuntimeError, match="printed no stdout"):
        _call_run_worker()


def test_run_worker_no_json_with_elapsed_raises(monkeypatch):
    """Stdout lines exist but none decode to JSON with an elapsed_sec key:
    the JSONDecodeError-continue (187-188) runs for every line, then the
    final 'no JSON line with elapsed_sec' RuntimeError fires (195-198)."""
    _patch_popen(
        monkeypatch,
        stdout_text='not json at all\n{"other": 1}\nstill not json\n',
        returncode=0,
    )
    with pytest.raises(RuntimeError, match="no JSON line with 'elapsed_sec'"):
        _call_run_worker()


def test_run_worker_verbose_tees_stderr(monkeypatch, capsys):
    """verbose=True mirrors the worker's stderr to our stderr via the tee
    thread (144-146)."""
    _patch_popen(
        monkeypatch,
        stdout_text='{"elapsed_sec": 0.1, "peak_mb": 1.0}\n',
        stderr_text="[autozyme] worker progress line\n",
        returncode=0,
    )
    elapsed, _ = _call_run_worker(verbose=True)
    assert elapsed == 0.1
    err = capsys.readouterr().err
    assert "worker progress line" in err


# ==========================================================================
# _run_evaluate — Rscript command-build branch for an evaluate.R file (87)
# ==========================================================================
def test_run_evaluate_builds_rscript_cmd_for_dot_r(tmp_path, monkeypatch):
    """An evaluate.R (no evaluate.py) takes the else-branch that builds an
    ['Rscript', dest] command. We intercept subprocess.run to assert the
    command shape without needing R installed."""
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "evaluate.R").write_text('cat("output_match: 1.0\\n")\n')
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    ref_dir = temp_dir / "reference_output_tiny"
    ref_dir.mkdir()

    captured = {}

    class _Ret:
        returncode = 0
        stdout = "output_match: 1.0\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Ret()

    monkeypatch.setattr(subprocess, "run", fake_run)
    lines = V._run_evaluate(str(task_dir), str(temp_dir), str(ref_dir), "tiny")
    assert captured["cmd"][0] == "Rscript"
    assert captured["cmd"][1].endswith("evaluate.R")
    assert any("output_match" in ln for ln in lines)


def test_run_evaluate_rscript_nonzero_raises(tmp_path, monkeypatch):
    """The Rscript branch still reports a non-zero exit as RuntimeError."""
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "evaluate.R").write_text('stop("boom")\n')
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    ref_dir = temp_dir / "reference_output_tiny"
    ref_dir.mkdir()

    class _Ret:
        returncode = 1
        stdout = ""
        stderr = "Error: boom\n"

    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: _Ret())
    with pytest.raises(RuntimeError, match="evaluate exited 1"):
        V._run_evaluate(str(task_dir), str(temp_dir), str(ref_dir), "tiny")
