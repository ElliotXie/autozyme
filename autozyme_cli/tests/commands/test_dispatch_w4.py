"""Wave-4 mop-up for zyme.commands.dispatch.

Wave-2 (test_dispatch_cmd.py) covered the resolver helpers, dry-run, and the
common error gates. This file fills the remaining REACHABLE in-process
branches that wave-2 skipped — all of them stop short of forking a daemon by
monkeypatching the `zyme.dispatch[.state|.resources|.master]` boundary:

  - cmd_dispatch: agent="auto" detection, find_agent_binary RuntimeError,
    per-agent model defaults, --reflect prompt validation, non-auto
    --disk-floor parse (+ ValueError), and the non-dry-run start_dispatch path
    (detach banner).
  - cmd_dispatch_resume: --prompt file-existence check + the real
    resume_dispatch_task call (rc gate).
  - cmd_dispatch_logs: KeyboardInterrupt during streaming is swallowed.
  - cmd_dispatch_wait: verbose finished banner, timeout exit(124), verbose
    progress line.

No process is started; tmp_path scoped; deterministic.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from zyme.commands import dispatch as d


# --------------------------------------------------------------------------
# cmd_dispatch — agent resolution, model defaults, reflect, disk-floor, start
# --------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    t = tmp_path / "test_a"
    (t / "prompts").mkdir(parents=True)
    (t / "task.yaml").write_text("x")
    (t / "prompts" / "2_iterate.md").write_text("p")
    (t / "prompts" / "reflect.md").write_text("r")
    return tmp_path


def _dispatch_args(workspace, **over):
    base = dict(
        workspace=str(workspace), tasks=["test_a"],
        prompt="prompts/2_iterate.md", agent="claude", model=None,
        effort="high", max_rounds=10, force_mode=False, reflect=False,
        reflect_prompt=None, ram_floor="8G", disk_floor="auto",
        detach=False, dry_run=True, stall_threshold=900,
        reflection_root=None, reflect_category=None)
    base.update(over)
    return SimpleNamespace(**base)


def _patch_boundary(monkeypatch, *, prior_pid=None, alive=False,
                    parse=None, start=None, detect=None):
    import zyme.dispatch as zd
    import zyme.dispatch.state as zds
    import zyme.dispatch.resources as zdr
    import zyme.dispatch.master as zdm
    monkeypatch.setattr(zd, "find_agent_binary",
                        lambda agent: "/bin/" + agent, raising=False)
    monkeypatch.setattr(zdr, "parse_size_gb",
                        parse or (lambda s: None if s == "auto" else 8.0))
    monkeypatch.setattr(zds, "read_pid", lambda p: prior_pid)
    monkeypatch.setattr(zds, "pid_alive", lambda pid: alive)
    monkeypatch.setattr(zds, "pid_path", lambda ws: ws / "pid")
    monkeypatch.setattr(zds, "state_path", lambda ws: ws / "state.json")
    monkeypatch.setattr(zds, "master_log_path", lambda ws: ws / "log")
    monkeypatch.setattr(zds, "ensure_dispatch_dirs", lambda ws: None)
    if start is not None:
        monkeypatch.setattr(zd, "start_dispatch", start, raising=False)
    if detect is not None:
        monkeypatch.setattr(zdm, "detect_agent_binary", detect, raising=False)


def test_auto_agent_detected(workspace, monkeypatch, capsys):
    _patch_boundary(monkeypatch, detect=lambda: ("codex", "/bin/codex"))
    args = _dispatch_args(workspace, agent="auto")
    d.cmd_dispatch(args)
    # Detection rewrites args.agent in place; dry-run echoes it.
    assert args.agent == "codex"
    assert "agent     : codex" in capsys.readouterr().out


def test_find_agent_binary_runtime_error_dies(workspace, monkeypatch):
    import zyme.dispatch as zd
    import zyme.dispatch.state as zds
    monkeypatch.setattr(zds, "read_pid", lambda p: None)
    monkeypatch.setattr(zds, "pid_alive", lambda pid: False)
    monkeypatch.setattr(zds, "pid_path", lambda ws: ws / "pid")

    def boom(agent):
        raise RuntimeError("agent binary not found")
    monkeypatch.setattr(zd, "find_agent_binary", boom, raising=False)
    with pytest.raises(SystemExit):
        d.cmd_dispatch(_dispatch_args(workspace))


def test_claude_default_model(workspace, monkeypatch, capsys):
    _patch_boundary(monkeypatch)
    args = _dispatch_args(workspace, agent="claude", model=None)
    d.cmd_dispatch(args)
    assert args.model == "claude-opus-4-7[1m]"


def test_codex_default_model(workspace, monkeypatch, capsys):
    _patch_boundary(monkeypatch)
    args = _dispatch_args(workspace, agent="codex", model=None)
    d.cmd_dispatch(args)
    assert args.model  # set to DEFAULT_CODEX_MODEL
    capsys.readouterr()


def test_cursor_default_model(workspace, monkeypatch, capsys):
    _patch_boundary(monkeypatch)
    args = _dispatch_args(workspace, agent="cursor", model=None)
    d.cmd_dispatch(args)
    assert args.model  # set to DEFAULT_CURSOR_MODEL
    capsys.readouterr()


def test_reflect_validates_prompt_present(workspace, monkeypatch, capsys):
    _patch_boundary(monkeypatch)
    args = _dispatch_args(workspace, reflect=True,
                          reflect_prompt="prompts/reflect.md")
    d.cmd_dispatch(args)
    out = capsys.readouterr().out
    assert "reflect   : True" in out


def test_reflect_missing_prompt_dies(workspace, monkeypatch):
    _patch_boundary(monkeypatch)
    args = _dispatch_args(workspace, reflect=True,
                          reflect_prompt="prompts/does_not_exist.md")
    with pytest.raises(SystemExit):
        d.cmd_dispatch(args)


def test_disk_floor_explicit_number(workspace, monkeypatch, capsys):
    _patch_boundary(monkeypatch, parse=lambda s: 16.0)
    args = _dispatch_args(workspace, disk_floor="16G")
    d.cmd_dispatch(args)
    assert "disk floor: 16.0 GB" in capsys.readouterr().out


def test_disk_floor_bad_value_dies(workspace, monkeypatch):
    def parse(s):
        if s == "garbage":
            raise ValueError("bad size")
        return None
    _patch_boundary(monkeypatch, parse=parse)
    args = _dispatch_args(workspace, disk_floor="garbage")
    with pytest.raises(SystemExit):
        d.cmd_dispatch(args)


def test_non_dry_run_starts_and_prints_detach_banner(workspace, monkeypatch,
                                                     capsys):
    captured = {}

    def fake_start(**kw):
        captured.update(kw)
        return 9999
    _patch_boundary(monkeypatch, start=fake_start)
    args = _dispatch_args(workspace, dry_run=False, detach=True, model="m")
    d.cmd_dispatch(args)
    out = capsys.readouterr().out
    assert "dispatch started (PID 9999, daemonized)" in out
    assert captured["agent"] == "claude"
    assert captured["detach"] is True


def test_non_dry_run_start_runtime_error_dies(workspace, monkeypatch):
    def fake_start(**kw):
        raise RuntimeError("resource gate failed")
    _patch_boundary(monkeypatch, start=fake_start)
    args = _dispatch_args(workspace, dry_run=False, detach=False, model="m")
    with pytest.raises(SystemExit):
        d.cmd_dispatch(args)


# --------------------------------------------------------------------------
# cmd_dispatch_resume — prompt file check + real resume call
# --------------------------------------------------------------------------

def _patch_resume_state(monkeypatch, state, *, alive=False, resume=None):
    import zyme.dispatch as zd
    import zyme.dispatch.state as zds
    monkeypatch.setattr(zds, "read_pid", lambda p: 1 if alive else None)
    monkeypatch.setattr(zds, "pid_alive", lambda pid: alive)
    monkeypatch.setattr(zds, "pid_path", lambda ws: ws / "pid")
    monkeypatch.setattr(zds, "state_path", lambda ws: ws / "s.json")
    monkeypatch.setattr(zds, "read_state", lambda p: state)
    if resume is not None:
        monkeypatch.setattr(zd, "resume_dispatch_task", resume, raising=False)


def test_resume_prompt_file_missing_dies(tmp_path, monkeypatch):
    task = tmp_path / "test_a"
    task.mkdir()
    state = {"agent": "claude", "model": "m", "queue": [
        {"name": "test_a", "task_dir": str(task), "agent_session_id": "s1"}]}
    _patch_resume_state(monkeypatch, state)
    args = SimpleNamespace(workspace=str(tmp_path), task="test_a",
                           prompt="prompts/missing.md", message=None,
                           dry_run=True, stall_threshold=900)
    with pytest.raises(SystemExit):
        d.cmd_dispatch_resume(args)


def test_resume_prompt_file_present_dry_run(tmp_path, monkeypatch, capsys):
    task = tmp_path / "test_a"
    (task / "prompts").mkdir(parents=True)
    (task / "prompts" / "p.md").write_text("x")
    state = {"agent": "claude", "model": "m", "queue": [
        {"name": "test_a", "task_dir": str(task), "agent_session_id": "s1"}]}
    _patch_resume_state(monkeypatch, state)
    args = SimpleNamespace(workspace=str(tmp_path), task="test_a",
                           prompt="prompts/p.md", message=None,
                           dry_run=True, stall_threshold=900)
    d.cmd_dispatch_resume(args)
    out = capsys.readouterr().out
    # When no --message, dry-run synthesizes "read and follow <prompt>".
    assert "read and follow prompts/p.md" in out


def test_resume_invokes_real_resume(tmp_path, monkeypatch):
    task = tmp_path / "test_a"
    task.mkdir()
    state = {"agent": "claude", "model": "m", "queue": [
        {"name": "test_a", "task_dir": str(task), "agent_session_id": "s1"}]}
    captured = {}

    def fake_resume(**kw):
        captured.update(kw)
        return 0
    _patch_resume_state(monkeypatch, state, resume=fake_resume)
    args = SimpleNamespace(workspace=str(tmp_path), task="test_a",
                           prompt=None, message="continue", dry_run=False,
                           stall_threshold=900)
    d.cmd_dispatch_resume(args)
    assert captured["task_name"] == "test_a"
    assert captured["message"] == "continue"


def test_resume_nonzero_rc_dies(tmp_path, monkeypatch):
    task = tmp_path / "test_a"
    task.mkdir()
    state = {"agent": "claude", "model": "m", "queue": [
        {"name": "test_a", "task_dir": str(task), "agent_session_id": "s1"}]}
    _patch_resume_state(monkeypatch, state, resume=lambda **k: 7)
    args = SimpleNamespace(workspace=str(tmp_path), task="test_a",
                           prompt=None, message="x", dry_run=False,
                           stall_threshold=900)
    with pytest.raises(SystemExit):
        d.cmd_dispatch_resume(args)


# --------------------------------------------------------------------------
# cmd_dispatch_logs — KeyboardInterrupt is swallowed
# --------------------------------------------------------------------------

def test_logs_keyboard_interrupt_swallowed(tmp_path, monkeypatch):
    import zyme.dispatch as zd
    import zyme.dispatch.state as zds
    monkeypatch.setattr(zds, "state_path", lambda ws: ws / "s")
    monkeypatch.setattr(zds, "read_state",
                        lambda p: {"queue": [{"name": "test_a"}]})
    monkeypatch.setattr(zds, "task_events_path",
                        lambda ws, name: ws / f"{name}.events")

    def boom(ws, task, follow, n):
        yield "first line"
        raise KeyboardInterrupt
    monkeypatch.setattr(zd, "stream_task_logs", boom, raising=False)
    args = SimpleNamespace(workspace=str(tmp_path), task="test_a",
                           follow=True, last_n=10, full=False)
    # Should return cleanly (Ctrl-C), not propagate.
    d.cmd_dispatch_logs(args)


def test_logs_full_passes_none_n(tmp_path, monkeypatch, capsys):
    import zyme.dispatch as zd
    import zyme.dispatch.state as zds
    monkeypatch.setattr(zds, "state_path", lambda ws: ws / "s")
    monkeypatch.setattr(zds, "read_state",
                        lambda p: {"queue": [{"name": "test_a"}]})
    monkeypatch.setattr(zds, "task_events_path",
                        lambda ws, name: ws / f"{name}.events")
    captured = {}

    def streamer(ws, task, follow, n):
        captured["n"] = n
        return iter(["L"])
    monkeypatch.setattr(zd, "stream_task_logs", streamer, raising=False)
    args = SimpleNamespace(workspace=str(tmp_path), task="test_a",
                           follow=False, last_n=10, full=True)
    d.cmd_dispatch_logs(args)
    assert captured["n"] is None  # --full overrides last_n


# --------------------------------------------------------------------------
# cmd_dispatch_wait — verbose finished, timeout, verbose progress
# --------------------------------------------------------------------------

def test_wait_verbose_finished_banner(tmp_path, monkeypatch, capsys):
    import zyme.dispatch.state as zds
    monkeypatch.setattr(zds, "state_path", lambda ws: ws / "s")
    monkeypatch.setattr(zds, "pid_path", lambda ws: ws / "pid")
    monkeypatch.setattr(zds, "read_state",
                        lambda p: {"finished_at": "2026", "queue": []})
    monkeypatch.setattr(zds, "read_pid", lambda p: None)
    monkeypatch.setattr(zds, "pid_alive", lambda pid: False)
    args = SimpleNamespace(workspace=str(tmp_path), poll=1, timeout=None,
                           verbose=True)
    with pytest.raises(SystemExit) as ei:
        d.cmd_dispatch_wait(args)
    assert ei.value.code == 0
    assert "dispatch finished after" in capsys.readouterr().out


def test_wait_timeout_exits_124(tmp_path, monkeypatch, capsys):
    import time as _t
    import zyme.dispatch.state as zds
    monkeypatch.setattr(zds, "state_path", lambda ws: ws / "s")
    monkeypatch.setattr(zds, "pid_path", lambda ws: ws / "pid")
    monkeypatch.setattr(zds, "read_state", lambda p: {"queue": []})
    # Master stays alive so we fall through to the timeout check.
    monkeypatch.setattr(zds, "read_pid", lambda p: 42)
    monkeypatch.setattr(zds, "pid_alive", lambda pid: True)
    # First monotonic() = start; second (in loop) is far past timeout.
    import time as _time
    seq = iter([0.0, 100.0, 200.0, 300.0])
    monkeypatch.setattr(_time, "monotonic", lambda: next(seq))
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    args = SimpleNamespace(workspace=str(tmp_path), poll=1, timeout=10,
                           verbose=False)
    with pytest.raises(SystemExit) as ei:
        d.cmd_dispatch_wait(args)
    assert ei.value.code == 124
    assert "timeout after 10s" in capsys.readouterr().out


def test_wait_verbose_progress_then_finish(tmp_path, monkeypatch, capsys):
    import zyme.dispatch.state as zds
    monkeypatch.setattr(zds, "state_path", lambda ws: ws / "s")
    monkeypatch.setattr(zds, "pid_path", lambda ws: ws / "pid")

    # First loop: alive -> verbose progress line, then sleep.
    # Second loop: dead + finished_at -> exit 0.
    states = iter([
        {"queue": [{"name": "test_a", "status": "running"},
                   {"name": "test_b", "finished_at": "x"}]},  # initial read
        {"queue": [{"name": "test_a", "status": "running"},
                   {"name": "test_b", "finished_at": "x"}]},  # verbose read
        {"finished_at": "2026", "queue": []},                 # after pid dead
    ])
    monkeypatch.setattr(zds, "read_state", lambda p: next(states))
    pids = iter([123, None])
    monkeypatch.setattr(zds, "read_pid", lambda p: next(pids))
    monkeypatch.setattr(zds, "pid_alive", lambda pid: pid is not None)
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    args = SimpleNamespace(workspace=str(tmp_path), poll=1, timeout=None,
                           verbose=True)
    with pytest.raises(SystemExit) as ei:
        d.cmd_dispatch_wait(args)
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "tasks done" in out
    assert "running: test_a" in out
