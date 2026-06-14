"""Wave-3 mop-up for zyme.commands.iterate.

Wave-2 (tests/commands/test_iterate_cmd.py) covered the pure helpers and a
single launch that reaches max-rounds. This file drives the rest of the
auto-resume state machine with the subprocess/agent boundary monkeypatched:

  - the RESUME path (launch_index>0 builds a _resume_prompt + message)
  - the no-session-id-on-resume fallback ("Starting fresh")
  - the agent-error (rc != 0) early return
  - the no-progress accumulation -> give-up transition
  - the stall path (stall event with max-rounds-during-stall + plain stall)
  - the OSError-on-launch failure branch
  - the _SHOULD_STOP interrupt mid-stream

Everything below subprocess.Popen / _read_stream_with_stalls / _build_agent_cmd
is stubbed; the real loop logic runs.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands import iterate as it


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class FakeProc:
    def __init__(self, rc=0):
        self.stdout = iter([])
        self.returncode = rc
        self.terminated = False

    def wait(self):
        return self.returncode


def _task(tmp_path: Path) -> Path:
    (tmp_path / "task.yaml").write_text("target_function: foo\n")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "2_iterate.md").write_text(
        "# Iterate\n## Role\nbody content\n")
    return tmp_path


def _args(td, **kw):
    base = dict(task_dir=str(td), prompt="prompts/2_iterate.md",
                max_rounds=5, no_progress_limit=2, stall_threshold=900,
                model="m", effort="high")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _reset_stop(monkeypatch):
    monkeypatch.setattr(it, "_SHOULD_STOP", False)
    # Never let _install_signal_handlers actually rebind real signals during
    # the loop tests (the autouse fixture keeps the suite isolated).
    monkeypatch.setattr(it, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(it, "_build_agent_cmd", lambda *a, **k: ["claude", "-p"])
    monkeypatch.setattr(it, "_terminate", lambda proc: None)
    monkeypatch.setattr(it, "parse_agent_line",
                        lambda payload, agent="claude": [])
    yield


def _counter(seq):
    """Return a _count_rounds replacement yielding snapshots from `seq`,
    repeating the last entry once exhausted."""
    state = {"i": 0}

    def fake(_d):
        i = min(state["i"], len(seq) - 1)
        state["i"] += 1
        done = seq[i]
        return {"completed_rounds": done, "last_round": str(done),
                "last_status": "keep"}
    return fake


# --------------------------------------------------------------------------
# launch failure
# --------------------------------------------------------------------------

def test_launch_oserror_exits(tmp_path, monkeypatch, capsys):
    td = _task(tmp_path)
    monkeypatch.setattr(it, "_count_rounds", _counter([0]))

    def boom(*a, **k):
        raise OSError("no such binary")
    monkeypatch.setattr(it.subprocess, "Popen", boom)
    with pytest.raises(SystemExit):
        it.cmd_iterate(_args(td))
    assert "failed to launch claude" in capsys.readouterr().err


# --------------------------------------------------------------------------
# agent error (rc != 0)
# --------------------------------------------------------------------------

def test_agent_nonzero_rc_returns(tmp_path, monkeypatch, capsys):
    td = _task(tmp_path)
    monkeypatch.setattr(it, "_count_rounds", _counter([0, 0, 0]))
    monkeypatch.setattr(it.subprocess, "Popen", lambda *a, **k: FakeProc(rc=1))
    monkeypatch.setattr(it, "_read_stream_with_stalls",
                        lambda proc, st: iter([("exit", None)]))
    it.cmd_iterate(_args(td))
    err = capsys.readouterr().err
    assert "claude exited with rc=1" in err
    assert "agent error" in err


# --------------------------------------------------------------------------
# no-progress give-up
# --------------------------------------------------------------------------

def test_no_progress_gives_up_after_limit(tmp_path, monkeypatch, capsys):
    td = _task(tmp_path)
    # Always 0 completed rounds, clean exits -> each relaunch adds a
    # no-progress strike; with limit=2 it gives up on the 2nd.
    monkeypatch.setattr(it, "_count_rounds", lambda _d: {
        "completed_rounds": 0, "last_round": None, "last_status": None})
    monkeypatch.setattr(it.subprocess, "Popen", lambda *a, **k: FakeProc(rc=0))
    monkeypatch.setattr(it, "_read_stream_with_stalls",
                        lambda proc, st: iter([("exit", None)]))
    it.cmd_iterate(_args(td, max_rounds=5, no_progress_limit=2))
    err = capsys.readouterr().err
    assert "agent exited with no new rounds (1/2)" in err
    assert "agent exited with no new rounds (2/2)" in err
    assert "giving up: 2 consecutive exits" in err


# --------------------------------------------------------------------------
# RESUME path: first launch makes progress but stays below max, then resumes.
# --------------------------------------------------------------------------

def test_resume_path_builds_resume_prompt(tmp_path, monkeypatch, capsys):
    td = _task(tmp_path)
    # Snapshots: pre-loop=1, after launch0=2 (progress, below max=5),
    # then resume launch1 reaches 5.
    monkeypatch.setattr(it, "_count_rounds", _counter([1, 2, 2, 5, 5]))

    procs = [FakeProc(rc=0), FakeProc(rc=0)]
    pop_calls = {"n": 0}

    def fake_popen(*a, **k):
        p = procs[min(pop_calls["n"], len(procs) - 1)]
        pop_calls["n"] += 1
        return p
    monkeypatch.setattr(it.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(it.subprocess, "Popen", fake_popen)

    # First stream sees a session id (so resume has a session to resume),
    # second stream just exits.
    streams = iter([
        iter([("line", '{"session": "sess-xyz"}'), ("exit", None)]),
        iter([("exit", None)]),
    ])
    monkeypatch.setattr(it, "_read_stream_with_stalls",
                        lambda proc, st: next(streams))
    monkeypatch.setattr(
        it, "parse_agent_line",
        lambda payload, agent="claude": (
            [{"kind": "agent_session", "session_id": "sess-xyz"}]
            if "session" in payload else []))

    resume_seen = {}
    real_resume = it._resume_prompt

    def spy_resume(prompt, max_rounds, task_state):
        resume_seen["called"] = True
        resume_seen["task_state"] = task_state
        return real_resume(prompt, max_rounds, task_state)
    monkeypatch.setattr(it, "_resume_prompt", spy_resume)

    it.cmd_iterate(_args(td, max_rounds=5, no_progress_limit=3))
    err = capsys.readouterr().err
    assert "launching claude" in err
    assert "resuming session sess-xyz" in err
    assert resume_seen.get("called")
    # The resume task-state carried the prior completed count.
    assert resume_seen["task_state"]["results_rounds"] == 2


# --------------------------------------------------------------------------
# resume requested but no session id -> "Starting fresh" fallback.
# --------------------------------------------------------------------------

def test_resume_without_session_starts_fresh(tmp_path, monkeypatch, capsys):
    td = _task(tmp_path)
    # progress on launch0 (0 -> 1) but no session captured, then launch1
    # resumes-but-cannot (fresh) and reaches max.
    monkeypatch.setattr(it, "_count_rounds", _counter([0, 1, 5, 5]))
    monkeypatch.setattr(it.subprocess, "Popen", lambda *a, **k: FakeProc(rc=0))
    # No session-id event in any stream; a fresh empty-exit stream per launch.
    monkeypatch.setattr(it, "_read_stream_with_stalls",
                        lambda proc, st: iter([("exit", None)]))
    it.cmd_iterate(_args(td, max_rounds=5, no_progress_limit=3))
    err = capsys.readouterr().err
    assert "cannot resume" in err
    assert "Starting fresh" in err


# --------------------------------------------------------------------------
# stall events
# --------------------------------------------------------------------------

def test_stall_then_max_rounds_during_stall(tmp_path, monkeypatch, capsys):
    td = _task(tmp_path)
    # pre-loop=0; first _count_rounds inside stall handler reports max.
    monkeypatch.setattr(it, "_count_rounds", _counter([0, 5, 5]))
    monkeypatch.setattr(it.subprocess, "Popen", lambda *a, **k: FakeProc(rc=0))
    monkeypatch.setattr(it, "_read_stream_with_stalls",
                        lambda proc, st: iter([("stall", "930"), ("exit", None)]))
    it.cmd_iterate(_args(td, max_rounds=5))
    err = capsys.readouterr().err
    assert "stall detected: 930s" in err
    assert "max rounds reached (during stall)" in err


def test_stall_below_max_continues_to_exit(tmp_path, monkeypatch, capsys):
    td = _task(tmp_path)
    # Stall fires but rounds still below max, then the agent exits cleanly with
    # progress reaching max so the loop ends after this single launch.
    monkeypatch.setattr(it, "_count_rounds", _counter([0, 2, 5, 5]))
    monkeypatch.setattr(it.subprocess, "Popen", lambda *a, **k: FakeProc(rc=0))
    monkeypatch.setattr(it, "_read_stream_with_stalls",
                        lambda proc, st: iter([("stall", "950"), ("exit", None)]))
    it.cmd_iterate(_args(td, max_rounds=5))
    err = capsys.readouterr().err
    assert "stall detected: 950s" in err
    assert "done: 5/5" in err


# --------------------------------------------------------------------------
# _SHOULD_STOP interrupt mid-stream
# --------------------------------------------------------------------------

def test_should_stop_interrupts_mid_stream(tmp_path, monkeypatch, capsys):
    td = _task(tmp_path)
    monkeypatch.setattr(it, "_count_rounds", _counter([0, 0, 0]))
    monkeypatch.setattr(it.subprocess, "Popen", lambda *a, **k: FakeProc(rc=0))

    # The loop enters with _SHOULD_STOP False; the stream generator flips it
    # True right before yielding the first line so the in-stream guard fires,
    # terminates the process, and the post-wait path returns "interrupted".
    def stream(proc, st):
        it._SHOULD_STOP = True
        yield ("line", "{}")
        yield ("exit", None)
    monkeypatch.setattr(it, "_read_stream_with_stalls", stream)
    it.cmd_iterate(_args(td, max_rounds=5))
    assert "interrupted" in capsys.readouterr().err


# --------------------------------------------------------------------------
# zyme_* event printing
# --------------------------------------------------------------------------

def test_loop_ends_via_stop_after_full_iteration(tmp_path, monkeypatch, capsys):
    # The agent exits cleanly with progress (so no error / no-progress return),
    # below max-rounds. The stop flag is flipped on the post-wait snapshot
    # (after the line-218 interrupted guard already passed), so the iteration
    # completes and the next `while not _SHOULD_STOP` check ends the loop,
    # emitting the trailing "stopped" status line.
    td = _task(tmp_path)
    monkeypatch.setattr(it.subprocess, "Popen", lambda *a, **k: FakeProc(rc=0))
    calls = {"n": 0}

    def counter(_d):
        calls["n"] += 1
        if calls["n"] >= 2:
            it._SHOULD_STOP = True
        return {"completed_rounds": 0 if calls["n"] == 1 else 1,
                "last_round": "1", "last_status": "keep"}
    monkeypatch.setattr(it, "_count_rounds", counter)
    monkeypatch.setattr(it, "_read_stream_with_stalls",
                        lambda proc, st: iter([("exit", None)]))
    it.cmd_iterate(_args(td, max_rounds=5, no_progress_limit=3))
    assert "stopped" in capsys.readouterr().err


def test_zyme_event_printed(tmp_path, monkeypatch, capsys):
    td = _task(tmp_path)
    monkeypatch.setattr(it, "_count_rounds", _counter([0, 5, 5]))
    monkeypatch.setattr(it.subprocess, "Popen", lambda *a, **k: FakeProc(rc=0))
    monkeypatch.setattr(it, "_read_stream_with_stalls",
                        lambda proc, st: iter([("line", "x"), ("exit", None)]))
    monkeypatch.setattr(
        it, "parse_agent_line",
        lambda payload, agent="claude": [
            {"kind": "zyme_accept", "description": "kept the win"}])
    it.cmd_iterate(_args(td, max_rounds=5))
    err = capsys.readouterr().err
    assert "[zyme_accept] kept the win" in err
