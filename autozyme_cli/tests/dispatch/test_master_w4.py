"""Wave-4 coverage for zyme.dispatch.master — the reachable non-daemon
helper / renderer / stream / stop branches left by test_master_deep.py,
test_master_rounds.py, test_master_w3.py.

Targets:
  - find_claude_binary / find_codex_binary / find_cursor_binary: env-override
    not-executable, PATH hit, ~/.local/bin fallback, not-found RuntimeError
  - _read_stream_with_stalls POSIX stall branch (select returns empty), and
    the os.read OSError-on-decode branch, via monkeypatched select/os.read
  - _read_stream_with_stalls_threaded (the Windows pipe reader) driven
    directly: line + eof, no-stdout, and the stall path
  - stop_dispatch live-master path: SIGTERM + clean exit, + timeout, +
    ProcessLookupError on kill, monkeypatched so no real processes are touched
  - stream_task_logs follow=True path (monkeypatched follow_events)
  - render_status running-duration + stalled + reflect annotations

No real subprocesses or agents are spawned; all process boundaries are
monkeypatched. The genuine daemon double-fork (_spawn_daemon, lines 296-344)
and the foreground master loop's launch path remain subprocess-only and are
documented as unreachable in unit tests.
"""
from __future__ import annotations

import os
import queue as _queue
import signal
from pathlib import Path

import pytest

import zyme.dispatch.master as master
from zyme.dispatch.master import (
    _default_reflection_root,
    _read_stream_with_stalls,
    _read_stream_with_stalls_threaded,
    _results_round_snapshot,
    _terminate,
    find_claude_binary,
    find_codex_binary,
    find_cursor_binary,
    render_status,
    resume_dispatch_task,
    start_dispatch,
    stop_dispatch,
    stream_task_logs,
)
from zyme.dispatch.state import ensure_dispatch_dirs, pid_path, state_path, write_pid


# ===========================================================================
# Binary finders — env not-executable / PATH / fallback / not-found
# ===========================================================================

class TestFindClaudeBinary:
    def test_env_override_not_executable_raises(self, tmp_path, monkeypatch):
        bogus = tmp_path / "not_exec"
        bogus.write_text("")  # exists but not chmod +x
        monkeypatch.setenv("ZYME_CLAUDE_BIN", str(bogus))
        with pytest.raises(RuntimeError, match="not executable"):
            find_claude_binary()

    def test_path_hit(self, monkeypatch):
        monkeypatch.delenv("ZYME_CLAUDE_BIN", raising=False)
        monkeypatch.setattr(master.shutil, "which", lambda n: "/usr/bin/claude")
        assert find_claude_binary() == "/usr/bin/claude"

    def test_local_bin_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ZYME_CLAUDE_BIN", raising=False)
        monkeypatch.setattr(master.shutil, "which", lambda n: None)
        home = tmp_path / "home"
        fake = home / ".local" / "bin" / "claude"
        fake.parent.mkdir(parents=True)
        fake.write_text("#!/bin/sh\n")
        os.chmod(fake, 0o755)
        monkeypatch.setattr(master.Path, "home", staticmethod(lambda: home))
        assert find_claude_binary() == str(fake)

    def test_not_found_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ZYME_CLAUDE_BIN", raising=False)
        monkeypatch.setattr(master.shutil, "which", lambda n: None)
        monkeypatch.setattr(master.Path, "home", staticmethod(lambda: tmp_path / "empty"))
        with pytest.raises(RuntimeError, match="claude binary not found"):
            find_claude_binary()


class TestFindCodexBinary:
    def test_env_override_not_executable(self, tmp_path, monkeypatch):
        bogus = tmp_path / "x"
        bogus.write_text("")
        monkeypatch.setenv("ZYME_CODEX_BIN", str(bogus))
        with pytest.raises(RuntimeError, match="not executable"):
            find_codex_binary()

    def test_path_hit(self, monkeypatch):
        monkeypatch.delenv("ZYME_CODEX_BIN", raising=False)
        monkeypatch.setattr(master.shutil, "which", lambda n: "/usr/bin/codex")
        assert find_codex_binary() == "/usr/bin/codex"

    def test_not_found(self, monkeypatch):
        monkeypatch.delenv("ZYME_CODEX_BIN", raising=False)
        monkeypatch.setattr(master.shutil, "which", lambda n: None)
        with pytest.raises(RuntimeError, match="codex binary not found"):
            find_codex_binary()


class TestFindCursorBinary:
    def test_env_override_not_executable(self, tmp_path, monkeypatch):
        bogus = tmp_path / "x"
        bogus.write_text("")
        monkeypatch.setenv("ZYME_CURSOR_AGENT_BIN", str(bogus))
        with pytest.raises(RuntimeError, match="not executable"):
            find_cursor_binary()

    def test_path_hit(self, monkeypatch):
        monkeypatch.delenv("ZYME_CURSOR_AGENT_BIN", raising=False)
        monkeypatch.setattr(master.shutil, "which", lambda n: "/usr/bin/cursor-agent")
        assert find_cursor_binary() == "/usr/bin/cursor-agent"

    def test_local_bin_fallback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ZYME_CURSOR_AGENT_BIN", raising=False)
        monkeypatch.setattr(master.shutil, "which", lambda n: None)
        home = tmp_path / "home"
        fake = home / ".local" / "bin" / "cursor-agent"
        fake.parent.mkdir(parents=True)
        fake.write_text("#!/bin/sh\n")
        os.chmod(fake, 0o755)
        monkeypatch.setattr(master.Path, "home", staticmethod(lambda: home))
        assert find_cursor_binary() == str(fake)

    def test_not_found(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ZYME_CURSOR_AGENT_BIN", raising=False)
        monkeypatch.setattr(master.shutil, "which", lambda n: None)
        monkeypatch.setattr(master.Path, "home", staticmethod(lambda: tmp_path / "e"))
        with pytest.raises(RuntimeError, match="cursor-agent binary not found"):
            find_cursor_binary()


# ===========================================================================
# _read_stream_with_stalls — POSIX stall + drain branches
# ===========================================================================

class _FakeProc:
    """Popen-like with a controllable poll() sequence + drain bytes."""
    def __init__(self, *, poll_seq, drain=b"", rc=0):
        self._poll = list(poll_seq)
        self._rc = rc
        r, w = os.pipe()
        if drain:
            os.write(w, drain)
        os.close(w)
        self.stdout = os.fdopen(r, "r")

    def poll(self):
        return self._poll.pop(0) if self._poll else self._rc

    def wait(self):
        return self._rc


@pytest.mark.skipif(os.name == "nt", reason="POSIX select() path only")
def test_read_stream_stall_then_exit(monkeypatch):
    """select() returns empty -> elapsed >= threshold -> ('stall', ...),
    then poll() returns rc -> drain remaining + ('exit', rc)."""
    proc = _FakeProc(poll_seq=[None, 0], drain=b"tail-line\n", rc=0)

    # Make select always report "no data" so we go down the else/stall branch.
    monkeypatch.setattr(master.select, "select", lambda r, w, x, t: ([], [], []))

    # Make the monotonic clock jump past the stall threshold on the 2nd call.
    times = iter([0.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(master.time, "monotonic", lambda: next(times, 100.0))

    events = list(_read_stream_with_stalls(proc, stall_threshold_s=10))
    kinds = [k for k, _ in events]
    assert "stall" in kinds
    assert events[-1][0] == "exit"
    # The drained tail line was emitted.
    assert any(k == "line" and "tail-line" in str(p) for k, p in events)


@pytest.mark.skipif(os.name == "nt", reason="POSIX select() path only")
def test_read_stream_read_oserror_treated_as_eof(monkeypatch):
    """os.read raising OSError -> chunk='' -> EOF path -> ('exit', rc)."""
    r, w = os.pipe()
    os.write(w, b"ignored")
    os.close(w)

    class P:
        stdout = os.fdopen(r, "r")

        def wait(self):
            return 3

    monkeypatch.setattr(master.select, "select", lambda r, w, x, t: ([P.stdout.fileno()], [], []))

    def _boom(fd, n):
        raise OSError("read failed")
    monkeypatch.setattr(master.os, "read", _boom)

    events = list(_read_stream_with_stalls(P(), stall_threshold_s=999))
    assert events[-1] == ("exit", 3)


# ===========================================================================
# _read_stream_with_stalls_threaded (the Windows pipe reader) — direct
# ===========================================================================

class _ThreadProc:
    def __init__(self, lines, rc=0):
        # stdout iterable yields raw lines (with newline), like a text pipe.
        self.stdout = iter(lines)
        self._rc = rc

    def wait(self):
        return self._rc


def test_threaded_reader_lines_then_eof():
    proc = _ThreadProc(['{"a":1}\n', '\n', '  \n', '{"b":2}\n'], rc=0)
    events = list(_read_stream_with_stalls_threaded(proc, stall_threshold_s=999))
    lines = [p for k, p in events if k == "line"]
    assert '{"a":1}' in lines
    assert '{"b":2}' in lines
    # blank/whitespace lines are not emitted.
    assert "" not in lines
    assert events[-1][0] == "exit"


def test_threaded_reader_no_stdout():
    class P:
        stdout = None

        def wait(self):
            return 5
    events = list(_read_stream_with_stalls_threaded(P(), stall_threshold_s=1))
    assert events == [("exit", 5)]


def test_threaded_reader_stall_branch(monkeypatch):
    """The queue.get times out (Empty) before any line arrives and elapsed
    exceeds the stall threshold -> ('stall', ...). Then a line + eof arrive."""
    # Build a queue that raises Empty once, then yields a line and eof.
    real_queue = master.queue.Queue

    class _ScriptedQueue:
        def __init__(self):
            self._script = [
                ("__empty__", None),
                ("line", '{"x":1}'),
                ("eof", None),
            ]

        def put(self, item):
            pass  # the real reader thread also puts; ignore to stay scripted

        def get(self, timeout=None):
            kind, payload = self._script.pop(0)
            if kind == "__empty__":
                raise _queue.Empty()
            return (kind, payload)

    monkeypatch.setattr(master.queue, "Queue", lambda: _ScriptedQueue())

    # Force the reader thread to be a no-op (our scripted queue drives events).
    class _NoThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass
    monkeypatch.setattr(master.threading, "Thread", _NoThread)

    # Make elapsed exceed the stall threshold on the Empty branch.
    times = iter([0.0, 1000.0, 1000.0, 1000.0])
    monkeypatch.setattr(master.time, "monotonic", lambda: next(times, 1000.0))

    class P:
        stdout = iter([])

        def wait(self):
            return 0

    events = list(_read_stream_with_stalls_threaded(P(), stall_threshold_s=10))
    kinds = [k for k, _ in events]
    assert "stall" in kinds
    assert ("line", '{"x":1}') in events
    assert events[-1][0] == "exit"

    # restore (defensive; monkeypatch undoes anyway)
    master.queue.Queue = real_queue


# ===========================================================================
# stop_dispatch — live master path (monkeypatched, no real processes)
# ===========================================================================

class TestStopDispatchLive:
    def _setup(self, tmp_path, *, agent_pid=None):
        ensure_dispatch_dirs(tmp_path)
        write_pid(pid_path(tmp_path), 999_001)
        queue = [{"name": "t", "agent_pid": agent_pid}]
        from zyme.dispatch.state import write_state_atomic
        write_state_atomic(state_path(tmp_path), {"queue": queue})

    def test_clean_exit_after_sigterm(self, tmp_path, monkeypatch):
        self._setup(tmp_path, agent_pid=999_002)
        sent = []

        def fake_kill(pid, sig):
            sent.append((pid, sig))

        # Master alive until SIGTERM, then dies on next pid_alive check.
        alive_calls = {"master": 0}

        def fake_alive(pid):
            if pid == 999_001:
                alive_calls["master"] += 1
                # alive on the collection pass + first while check is not used;
                # report alive once, then dead.
                return alive_calls["master"] <= 1
            if pid == 999_002:
                return True  # agent pid collected
            return False

        monkeypatch.setattr(master.os, "kill", fake_kill)
        monkeypatch.setattr(master, "pid_alive", fake_alive)
        out = stop_dispatch(tmp_path, timeout=2.0)
        assert (999_001, signal.SIGTERM) in sent
        assert out["master_stopped"] is True
        assert 999_002 in out["agent_pids"]

    def test_process_lookup_on_kill_marks_stopped(self, tmp_path, monkeypatch):
        self._setup(tmp_path)

        def fake_kill(pid, sig):
            raise ProcessLookupError("gone")

        monkeypatch.setattr(master.os, "kill", fake_kill)
        monkeypatch.setattr(master, "pid_alive", lambda pid: pid == 999_001)
        out = stop_dispatch(tmp_path, timeout=1.0)
        assert out["master_stopped"] is True

    def test_timeout_kills_agent_pids(self, tmp_path, monkeypatch):
        self._setup(tmp_path, agent_pid=999_003)
        sent = []

        def fake_kill(pid, sig):
            sent.append(pid)
        # Master never dies -> hits the timeout branch + agent-pid kill loop.
        monkeypatch.setattr(master.os, "kill", fake_kill)
        monkeypatch.setattr(master, "pid_alive", lambda pid: True)
        slept = {"n": 0}

        def fake_sleep(s):
            slept["n"] += 1
        monkeypatch.setattr(master.time, "sleep", fake_sleep)
        # First while check inside the deadline (enters loop body + sleep),
        # then the clock jumps past the deadline.
        seq = iter([0.0, 0.1, 100.0, 100.0])
        monkeypatch.setattr(master.time, "monotonic", lambda: next(seq, 100.0))
        out = stop_dispatch(tmp_path, timeout=1.0)
        assert "did not exit" in out.get("error", "")
        assert slept["n"] >= 1  # the wait-loop sleep ran
        # Agent pid was force-killed in the last-resort loop.
        assert 999_003 in sent

    def test_timeout_agent_kill_process_lookup_swallowed(self, tmp_path, monkeypatch):
        self._setup(tmp_path, agent_pid=999_004)

        def fake_kill(pid, sig):
            if pid == 999_004:
                raise ProcessLookupError("agent already gone")
            # master SIGTERM succeeds

        monkeypatch.setattr(master.os, "kill", fake_kill)
        monkeypatch.setattr(master, "pid_alive", lambda pid: True)
        monkeypatch.setattr(master.time, "sleep", lambda s: None)
        seq = iter([0.0, 100.0, 100.0])
        monkeypatch.setattr(master.time, "monotonic", lambda: next(seq, 100.0))
        out = stop_dispatch(tmp_path, timeout=1.0)  # must not raise
        assert "did not exit" in out.get("error", "")


# ===========================================================================
# stream_task_logs follow=True
# ===========================================================================

def test_stream_task_logs_follow(monkeypatch, tmp_path):
    ensure_dispatch_dirs(tmp_path)
    fake_events = [
        {"ts": "2026-01-01T00:00:01Z", "kind": "claude_text", "snippet": "hi"},
        {"ts": "2026-01-01T00:00:02Z", "kind": "claude_done",
         "num_turns": 3, "total_cost_usd": 0.5},
    ]
    monkeypatch.setattr(master, "follow_events", lambda p: iter(fake_events))
    out = list(stream_task_logs(tmp_path, "task_a", follow=True))
    assert any("hi" in line for line in out)
    assert any("done" in line for line in out)


def test_stream_task_logs_skips_blank_lines(tmp_path):
    ensure_dispatch_dirs(tmp_path)
    from zyme.dispatch.state import task_events_path
    p = task_events_path(tmp_path, "task_a")
    p.write_text(
        '{"ts":"2026-01-01T00:00:00Z","kind":"claude_text","snippet":"ok"}\n'
        '\n'                # blank line -> continue (line 1671)
        '   \n'            # whitespace-only -> continue
    )
    out = list(stream_task_logs(tmp_path, "task_a", follow=False, n=None))
    assert len(out) == 1
    assert "ok" in out[0]


def test_stream_task_logs_read_oserror(monkeypatch, tmp_path):
    """non-follow path where read_text raises OSError -> yields nothing."""
    ensure_dispatch_dirs(tmp_path)
    from zyme.dispatch.state import task_events_path
    p = task_events_path(tmp_path, "task_a")
    p.write_text('{"ts":"x","kind":"claude_text","snippet":"y"}\n')

    original = Path.read_text

    def boom(self, *a, **k):
        if self.name.endswith("events.ndjson"):
            raise OSError("read fail")
        return original(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", boom)
    out = list(stream_task_logs(tmp_path, "task_a", follow=False))
    assert out == []


# ===========================================================================
# render_status — running duration + stalled + reflect annotations
# ===========================================================================

# ===========================================================================
# _terminate kill + ProcessLookupError branches
# ===========================================================================

class _KillProc:
    def __init__(self, *, poll_seq, terminate_exc=None):
        self._poll = list(poll_seq)
        self._terminate_exc = terminate_exc
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._poll.pop(0) if len(self._poll) > 1 else self._poll[0]

    def terminate(self):
        self.terminated = True
        if self._terminate_exc:
            raise self._terminate_exc

    def kill(self):
        self.killed = True


def test_terminate_kills_after_timeout(monkeypatch):
    # Stays alive through the whole wait loop -> kill() is called.
    proc = _KillProc(poll_seq=[None])  # always alive
    monkeypatch.setattr(master.time, "sleep", lambda s: None)
    seq = iter([0.0, 100.0, 100.0])  # deadline immediately past
    monkeypatch.setattr(master.time, "monotonic", lambda: next(seq, 100.0))
    _terminate(proc)
    assert proc.terminated and proc.killed


def test_terminate_terminate_process_lookup(monkeypatch):
    # terminate() raises ProcessLookupError -> early return, no kill.
    proc = _KillProc(poll_seq=[None], terminate_exc=ProcessLookupError())
    _terminate(proc)
    assert proc.terminated and not proc.killed


def test_terminate_kill_process_lookup(monkeypatch):
    # Survives wait, kill() raises ProcessLookupError -> swallowed.
    class P(_KillProc):
        def kill(self):
            self.killed = True
            raise ProcessLookupError()
    proc = P(poll_seq=[None])
    monkeypatch.setattr(master.time, "sleep", lambda s: None)
    seq = iter([0.0, 100.0, 100.0])
    monkeypatch.setattr(master.time, "monotonic", lambda: next(seq, 100.0))
    _terminate(proc)  # must not raise
    assert proc.killed


# ===========================================================================
# _default_reflection_root
# ===========================================================================

def test_default_reflection_root_ends_in_reflections():
    root = _default_reflection_root()
    assert root.name == "reflections"


# ===========================================================================
# _read_stream_with_stalls — nt dispatch + POSIX drain OSError
# ===========================================================================

def test_read_stream_dispatches_to_threaded_on_nt(monkeypatch):
    # os.name == 'nt' -> delegates to the threaded reader.
    monkeypatch.setattr(master.os, "name", "nt")

    class P:
        stdout = iter(['{"a":1}\n'])

        def wait(self):
            return 0
    events = list(_read_stream_with_stalls(P(), stall_threshold_s=999))
    assert events[-1][0] == "exit"
    assert ("line", '{"a":1}') in events


@pytest.mark.skipif(os.name == "nt", reason="POSIX path only")
def test_read_stream_drain_flushes_partial_tail(monkeypatch):
    """select-empty -> poll() returns rc -> drain yields a partial tail line
    (no trailing newline) which is flushed via the final buf.strip() (1281-2)."""
    r, w = os.pipe()  # keep a valid fd for fileno(); we patch read()

    class _Stdout:
        def fileno(self):
            return r

        def read(self):
            return "tail-without-newline"

    class P:
        stdout = _Stdout()

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(master.select, "select", lambda r2, w2, x2, t: ([], [], []))
    seq = iter([0.0, 0.0, 0.0])
    monkeypatch.setattr(master.time, "monotonic", lambda: next(seq, 0.0))
    events = list(_read_stream_with_stalls(P(), stall_threshold_s=999))
    os.close(r)
    os.close(w)
    lines = [p for k, p in events if k == "line"]
    assert "tail-without-newline" in lines
    assert events[-1][0] == "exit"


@pytest.mark.skipif(os.name == "nt", reason="POSIX path only")
def test_read_stream_drain_oserror(monkeypatch):
    """After EOF select-empty path, proc.stdout.read() raises OSError so the
    drain is skipped, then the buffered tail is flushed + exit."""
    r, w = os.pipe()
    os.close(w)  # immediate EOF on read

    class _Stdout:
        def __init__(self, fd):
            self._fd = fd

        def fileno(self):
            return self._fd

        def read(self):
            raise OSError("drain failed")

    class P:
        stdout = _Stdout(r)

        def __init__(self):
            self._polls = iter([0])

        def poll(self):
            return next(self._polls, 0)

        def wait(self):
            return 0

    monkeypatch.setattr(master.select, "select", lambda r2, w2, x2, t: ([], [], []))
    seq = iter([0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(master.time, "monotonic", lambda: next(seq, 0.0))
    events = list(_read_stream_with_stalls(P(), stall_threshold_s=999))
    os.close(r)
    assert events[-1][0] == "exit"


# ===========================================================================
# _results_round_snapshot — OSError + missing round/status columns
# ===========================================================================

def test_snapshot_oserror_returns_default(tmp_path, monkeypatch):
    p = tmp_path / "results.tsv"
    p.write_text("round\tstatus\n1\tkeep\n")

    original = Path.read_text

    def boom(self, *a, **k):
        if self.name == "results.tsv":
            raise OSError("read fail")
        return original(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", boom)
    out = _results_round_snapshot(tmp_path)
    assert out == {"completed_rounds": 0, "pending_rounds": 0,
                   "last_round": None, "last_status": None}


def test_snapshot_missing_round_and_status_columns(tmp_path):
    # Header without 'round' -> round_idx falls back to 0; without 'status'
    # -> status_idx falls back to 6. A 7-col row places status at index 6.
    p = tmp_path / "results.tsv"
    p.write_text(
        "a\tb\tc\td\te\tf\tg\n"
        "1\tx\ty\tz\tp\tq\tkeep\n"
    )
    out = _results_round_snapshot(tmp_path, phase="all")
    assert out["completed_rounds"] == 1
    assert out["last_status"] == "keep"


def test_snapshot_blank_line_skipped(tmp_path):
    # A blank body line is skipped (line 1404 `if not line.strip(): continue`).
    p = tmp_path / "results.tsv"
    p.write_text(
        "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tphase\n"
        "\n"                       # blank line -> continue
        "1\tc\td\t8\t0\t5\tkeep\t{}\t\t\toptimize\n"
    )
    out = _results_round_snapshot(tmp_path, phase="all")
    assert out["completed_rounds"] == 1


def test_snapshot_short_row_below_round_idx(tmp_path):
    # A row too short to reach round_idx is skipped (line 1407).
    short_dir = tmp_path / "sd"
    short_dir.mkdir()
    (short_dir / "results.tsv").write_text(
        "x\tround\tstatus\n"
        "onlyonecell\n"            # round_idx=1 >= len(parts)=1 -> skip
        "row\t1\tkeep\n"
    )
    out = _results_round_snapshot(short_dir, phase="all")
    assert out["completed_rounds"] == 1


# ===========================================================================
# start_dispatch already-running + resume_dispatch_task prompt-only
# ===========================================================================

def test_start_dispatch_refuses_when_prior_alive(tmp_path, monkeypatch):
    ensure_dispatch_dirs(tmp_path)
    write_pid(pid_path(tmp_path), 12345)
    monkeypatch.setattr(master, "pid_alive", lambda pid: True)
    with pytest.raises(RuntimeError, match="already running"):
        start_dispatch(
            workspace=tmp_path,
            tasks=[{"name": "t", "task_dir": str(tmp_path)}],
            prompt="prompts/x.md",
        )


def test_resume_dispatch_task_prompt_only_builds_message(tmp_path, monkeypatch):
    # prompt given but no message -> message becomes "read and follow <prompt>".
    from zyme.dispatch.state import write_state_atomic
    ensure_dispatch_dirs(tmp_path)
    state = {
        "queue": [{"name": "a", "task_dir": str(tmp_path),
                   "agent_session_id": "sid"}],
        "current_index": 0,
        "prompt": "prompts/2_iterate.md",
    }
    write_state_atomic(state_path(tmp_path), state)

    captured = {}

    def fake_resume(workspace, st, i, *, message, event_kind, foreground, event_fields=None):
        captured["message"] = message
        return 0
    monkeypatch.setattr(master, "_run_resume_message", fake_resume)
    rc = resume_dispatch_task(workspace=tmp_path, task_name="a",
                              prompt="prompts/extra.md", foreground=False)
    assert rc == 0
    assert captured["message"] == "read and follow prompts/extra.md"


# ===========================================================================
# stop_dispatch dead-master unlink OSError is swallowed
# ===========================================================================

def test_stop_dispatch_dead_master_unlink_oserror(tmp_path, monkeypatch):
    ensure_dispatch_dirs(tmp_path)
    write_pid(pid_path(tmp_path), 999_009)
    # Master reported dead.
    monkeypatch.setattr(master, "pid_alive", lambda pid: False)

    original_unlink = Path.unlink

    def boom(self, *a, **k):
        if self.name == "master.pid":
            raise OSError("cannot unlink")
        return original_unlink(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", boom)
    out = stop_dispatch(tmp_path)  # must not raise
    assert out["master_stopped"] is True


def test_render_status_running_stalled_reflect(tmp_path):
    from zyme.dispatch.state import write_state_atomic
    ensure_dispatch_dirs(tmp_path)
    state = {
        "workspace": str(tmp_path),
        "prompt": "prompts/2_iterate.md",
        "agent": "claude",
        "model": "opus",
        "max_rounds": 30,
        "force_mode": True,
        "reflect": True,
        "reflect_prompt": "prompts/5_reflect.md",
        "master_pid": 2_000_000_000,  # dead -> 'gone'
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": None,
        "eta_s": 125,
        "queue": [
            {
                "name": "running_task", "status": "running",
                "started_at": "2026-01-01T00:00:00Z",
                "stalled": True, "reflect_status": "running",
                "round": 4, "accepts": 2, "rejects": 1,
                "last_event_at": "2026-01-01T00:00:05Z",
                "last_event_kind": "claude_text",
                "duration_s": None,
            },
            {
                "name": "done_task", "status": "done", "duration_s": 90,
                "round": 7, "accepts": 5, "rejects": 0,
            },
        ],
    }
    write_state_atomic(state_path(tmp_path), state)
    out = render_status(tmp_path)
    assert "max rounds: 30" in out
    assert "force mode: on" in out
    assert "reflect" in out
    assert "gone" in out
    assert "eta" in out
    assert "STALLED" in out
    assert "reflect=running" in out
    assert "running_task" in out and "done_task" in out
