"""Wave-3 mop-up for zyme.dispatch.master.

test_master_deep.py covered the pure helpers / renderers / discovery; this file
fills the remaining REACHABLE in-process branches around (but NOT including) the
daemon double-fork:

  - _resume_prompt message assembly
  - _reflection_output_paths + _write_reflection_metadata YAML emit
  - _reflect_resume_message
  - _terminate (already-dead, graceful-poll)
  - _read_stream_with_stalls driven against a REAL os.pipe-backed proc stub
    (line splitting, EOF drain, partial-line flush)
  - _results_round_snapshot branch matrix (header variants, phase filter,
    pending vs terminal, sub-round labels, missing file)
  - _refresh_results_round_state + _max_rounds_reached
  - _run_resume_message with the subprocess.Popen boundary monkeypatched
    (no-session error, OSError-on-launch, a full stubbed stream pass)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from zyme.dispatch import master
from zyme.dispatch.master import (
    _max_rounds_reached,
    _read_stream_with_stalls,
    _reflect_resume_message,
    _reflection_output_paths,
    _refresh_results_round_state,
    _results_round_snapshot,
    _resume_prompt,
    _run_resume_message,
    _terminate,
    _write_reflection_metadata,
)
from zyme.dispatch.state import (
    ensure_dispatch_dirs,
    state_path,
    write_state_atomic,
)


_HDR = (
    "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
    "metrics_json\thypothesis\tdescription\tphase\n"
)


def _write_results(task_dir: Path, body: str):
    (task_dir / "results.tsv").write_text(_HDR + body, encoding="utf-8")


# ===========================================================================
# _resume_prompt
# ===========================================================================

def test_resume_prompt_includes_state():
    msg = _resume_prompt("prompts/2_iterate.md", 30, {
        "results_rounds": 4, "last_results_round": 4, "last_results_status": "keep"})
    assert "prompts/2_iterate.md" in msg
    assert "4 completed round(s)" in msg
    assert "last=4:keep" in msg
    # encouragement tier for 4 rounds.
    assert "you can do this" in msg


def test_resume_prompt_zero_rounds():
    msg = _resume_prompt("p.md", None, {})
    assert msg.startswith("Continue.")
    assert "0 completed round(s)" in msg


# ===========================================================================
# reflection helpers
# ===========================================================================

def test_reflection_output_paths_layout(tmp_path):
    out = _reflection_output_paths(tmp_path, "iteration", "test/foo")
    assert out["prompt_feedback"].parent == tmp_path / "prompt_reflect_feedback"
    assert out["zyme_cli_feedback"].parent == tmp_path / "zyme_cli_feedback"
    assert out["metadata"].parent == tmp_path / "metadata"
    # task name sanitized in the stem.
    assert "test_foo" in out["metadata"].name
    assert out["metadata"].suffix == ".yaml"


def test_write_reflection_metadata_yaml(tmp_path):
    meta_path = tmp_path / "m.yaml"
    state = {"agent": "claude", "model": "opus", "effort": "high",
             "prompt": "prompts/2_iterate.md", "max_rounds": 30,
             "force_mode": True}
    task = {"name": "task_a", "task_dir": "/x/task_a", "actual_model": "opus-4",
            "results_rounds": 5, "last_results_round": 5,
            "last_results_status": "keep", "agent_session_id": "sid-1"}
    _write_reflection_metadata(
        meta_path, state, task, "prompts/reflect.md", "iteration",
        prompt_feedback_path=tmp_path / "pf.md",
        zyme_cli_feedback_path=tmp_path / "cf.md",
        reflection_root=tmp_path)
    text = meta_path.read_text(encoding="utf-8")
    assert 'task: "task_a"' in text
    assert 'category: "iteration"' in text
    assert "force_mode: true" in text
    assert "max_rounds: 30" in text
    assert 'agent_session_id: "sid-1"' in text


def test_reflect_resume_message(tmp_path):
    msg = _reflect_resume_message(
        prompt="prompts/reflect.md",
        state={"prompt": "prompts/2_iterate.md"},
        task={"name": "task_a"},
        prompt_feedback_path=tmp_path / "pf.md",
        zyme_cli_feedback_path=tmp_path / "cf.md",
        metadata_path=tmp_path / "m.yaml",
        category="iteration")
    assert "prompts/reflect.md" in msg
    assert "task_a" in msg
    assert "iteration" in msg
    assert str(tmp_path / "pf.md") in msg
    assert "Do not commit or push" in msg


# ===========================================================================
# _terminate
# ===========================================================================

class _PollProc:
    """Minimal Popen-like double for _terminate."""
    def __init__(self, *, poll_seq):
        self._poll_seq = list(poll_seq)
        self.terminated = False
        self.killed = False

    def poll(self):
        if len(self._poll_seq) > 1:
            return self._poll_seq.pop(0)
        return self._poll_seq[0]

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_terminate_already_dead_noop():
    p = _PollProc(poll_seq=[0])  # already exited
    _terminate(p)
    assert not p.terminated and not p.killed


def test_terminate_graceful_exit():
    # First poll None (alive) -> terminate(); next poll 0 inside the wait loop.
    p = _PollProc(poll_seq=[None, 0])
    _terminate(p)
    assert p.terminated
    assert not p.killed


# ===========================================================================
# _read_stream_with_stalls — real os.pipe-backed proc
# ===========================================================================

class _PipeProc:
    """Popen-like wrapper around a read end of an os.pipe()."""
    def __init__(self, payload: bytes, *, rc=0):
        r, w = os.pipe()
        os.write(w, payload)
        os.close(w)
        self._rc = rc
        self.stdout = os.fdopen(r, "r")
        self._waited = False

    def wait(self):
        self._waited = True
        return self._rc

    def poll(self):
        # The reader hits EOF via os.read returning "" before it polls.
        return self._rc if self._waited else None


@pytest.mark.skipif(os.name == "nt", reason="POSIX select() path only")
def test_read_stream_splits_lines_and_exits():
    proc = _PipeProc(b'{"a":1}\n{"b":2}\npartial-no-newline')
    events = list(_read_stream_with_stalls(proc, stall_threshold_s=999))
    kinds = [k for k, _ in events]
    assert "line" in kinds
    lines = [payload for k, payload in events if k == "line"]
    assert '{"a":1}' in lines
    assert '{"b":2}' in lines
    # the trailing partial line is flushed before exit.
    assert "partial-no-newline" in lines
    assert events[-1][0] == "exit"


@pytest.mark.skipif(os.name == "nt", reason="POSIX select() path only")
def test_read_stream_no_stdout_exits():
    class _NoOut:
        stdout = None

        def wait(self):
            return 7
    events = list(_read_stream_with_stalls(_NoOut(), stall_threshold_s=1))
    assert events == [("exit", 7)]


# ===========================================================================
# _results_round_snapshot — branch matrix
# ===========================================================================

def test_snapshot_missing_file(tmp_path):
    out = _results_round_snapshot(tmp_path)
    assert out == {"completed_rounds": 0, "pending_rounds": 0,
                   "last_round": None, "last_status": None}


def test_snapshot_header_only(tmp_path):
    (tmp_path / "results.tsv").write_text(_HDR, encoding="utf-8")
    out = _results_round_snapshot(tmp_path)
    assert out["completed_rounds"] == 0


def test_snapshot_counts_terminal_and_pending(tmp_path):
    _write_results(tmp_path,
                   "0\tup\ta\t10\t0\t5\tbaseline\t{}\t\t\toptimize\n"
                   "1\tc1\ta\t8\t20\t5\tkeep\t{}\t\t\toptimize\n"
                   "2\tc2\ta\t9\t10\t5\tpending\t{}\t\t\toptimize\n")
    out = _results_round_snapshot(tmp_path)
    assert out["completed_rounds"] == 1  # round 1 keep
    assert out["pending_rounds"] == 1    # round 2 pending
    assert out["last_round"] == 2
    assert out["last_status"] == "pending"


def test_snapshot_phase_filter_excludes_scaling(tmp_path):
    _write_results(tmp_path,
                   "1\tc1\ta\t8\t20\t5\tkeep\t{}\t\t\tscaling\n")
    out = _results_round_snapshot(tmp_path, phase="optimize")
    assert out["completed_rounds"] == 0
    # phase="all" disables the filter.
    out_all = _results_round_snapshot(tmp_path, phase="all")
    assert out_all["completed_rounds"] == 1


def test_snapshot_skips_non_integer_rounds(tmp_path):
    # sub-round labels like "1.1" and round 0 are skipped.
    _write_results(tmp_path,
                   "0\tup\ta\t10\t0\t5\tbaseline\t{}\t\t\toptimize\n"
                   "1.1\tc1\ta\t8\t20\t5\trerun\t{}\t\t\toptimize\n"
                   "1\tc1\ta\t8\t20\t5\tdiscard\t{}\t\t\toptimize\n")
    out = _results_round_snapshot(tmp_path)
    assert out["completed_rounds"] == 1  # only round 1 discard
    assert out["last_round"] == 1


def test_snapshot_missing_phase_col_defaults_optimize(tmp_path):
    # No phase column -> rows treated as optimize phase.
    (tmp_path / "results.tsv").write_text(
        "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\n"
        "1\tc1\ta\t8\t20\t5\tkeep\n", encoding="utf-8")
    out = _results_round_snapshot(tmp_path, phase="optimize")
    assert out["completed_rounds"] == 1


# ===========================================================================
# _refresh_results_round_state + _max_rounds_reached
# ===========================================================================

def test_refresh_results_round_state(tmp_path):
    _write_results(tmp_path,
                   "1\tc1\ta\t8\t20\t5\tkeep\t{}\t\t\toptimize\n"
                   "2\tc2\ta\t7\t30\t5\tkeep\t{}\t\t\toptimize\n")
    task = {"task_dir": str(tmp_path), "round": 0}
    snap = _refresh_results_round_state(task)
    assert snap["completed_rounds"] == 2
    assert task["results_rounds"] == 2
    assert task["round"] == 2  # max(0, 2)
    assert task["last_results_status"] == "keep"


def test_max_rounds_reached_false_when_unset(tmp_path):
    task = {"task_dir": str(tmp_path)}
    assert _max_rounds_reached(task, None) is False
    assert _max_rounds_reached(task, 0) is False


def test_max_rounds_reached_true(tmp_path):
    _write_results(tmp_path,
                   "1\tc1\ta\t8\t20\t5\tkeep\t{}\t\t\toptimize\n"
                   "2\tc2\ta\t7\t30\t5\tkeep\t{}\t\t\toptimize\n")
    task = {"task_dir": str(tmp_path)}
    assert _max_rounds_reached(task, 2) is True
    assert _max_rounds_reached(task, 5) is False


# ===========================================================================
# _run_resume_message — subprocess boundary monkeypatched
# ===========================================================================

def _resume_state(tmp_path, *, session="sid-1"):
    ensure_dispatch_dirs(tmp_path)
    state = {
        "agent": "claude",
        "prompt": "prompts/2_iterate.md",
        "model": "opus",
        "effort": "max",
        "stall_threshold_s": 999,
        "queue": [{
            "name": "task_a", "task_dir": str(tmp_path),
            "agent_session_id": session, "status": "running",
        }],
        "current_index": 0,
    }
    write_state_atomic(state_path(tmp_path), state)
    return state


def test_run_resume_message_no_session_raises(tmp_path, monkeypatch):
    state = _resume_state(tmp_path, session=None)
    monkeypatch.setattr(master, "_build_agent_cmd", lambda *a, **k: ["x"])
    with pytest.raises(RuntimeError, match="no agent_session_id"):
        _run_resume_message(tmp_path, state, 0, message="go",
                            event_kind="resume", foreground=False)


def test_run_resume_message_launch_oserror_returns_neg1(tmp_path, monkeypatch):
    state = _resume_state(tmp_path)
    monkeypatch.setattr(master, "_build_agent_cmd", lambda *a, **k: ["x"])

    def boom(*a, **k):
        raise OSError("no binary")
    monkeypatch.setattr(master.subprocess, "Popen", boom)
    rc = _run_resume_message(tmp_path, state, 0, message="go",
                             event_kind="resume", foreground=False)
    assert rc == -1


def test_run_resume_message_full_stream(tmp_path, monkeypatch):
    state = _resume_state(tmp_path)
    monkeypatch.setattr(master, "_build_agent_cmd", lambda *a, **k: ["x"])

    class FakeProc:
        def __init__(self):
            self.pid = 4242
            self.returncode = 0

        def wait(self):
            return 0
    monkeypatch.setattr(master.subprocess, "Popen", lambda *a, **k: FakeProc())
    # Stub the stream to emit one parsed line then exit.
    monkeypatch.setattr(
        master, "_read_stream_with_stalls",
        lambda proc, st: iter([("line", '{"k":1}'), ("exit", 0)]))
    monkeypatch.setattr(
        master, "parse_agent_line",
        lambda payload, agent="claude": [
            {"kind": "claude_text", "snippet": "hi"}])
    rc = _run_resume_message(tmp_path, state, 0, message="go",
                             event_kind="dispatch_resume", foreground=True)
    # Clean exit returns the wait() rc; agent_pid is reset to None afterwards.
    assert rc == 0
    assert state["queue"][0]["agent_pid"] is None
    # The parsed event was applied to the task state.
    assert state["queue"][0].get("last_event_kind") == "claude_text"
