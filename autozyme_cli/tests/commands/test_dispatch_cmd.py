"""Unit tests for zyme.commands.dispatch — CLI wrappers around the dispatch daemon.

The daemon itself (zyme.dispatch.*) is not started here. We cover the pure
resolver helpers (workspace, task-path lookup across 3 modes, prompt-file
validation) and drive each cmd_dispatch* entrypoint over a monkeypatched
zyme.dispatch boundary (state/pid/start/stop/render stubs), focusing on the
dry-run / error-gate / report-printing branches that don't fork a process.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands import dispatch as d


# --------------------------------------------------------------------------
# _resolve_dispatch_workspace
# --------------------------------------------------------------------------

class TestResolveWorkspace:
    def test_explicit(self, tmp_path):
        args = SimpleNamespace(workspace=str(tmp_path))
        assert d._resolve_dispatch_workspace(args) == tmp_path.resolve()

    def test_cwd_fallback(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        args = SimpleNamespace(workspace=None)
        assert d._resolve_dispatch_workspace(args) == tmp_path.resolve()

    def test_missing_dies(self, tmp_path):
        args = SimpleNamespace(workspace=str(tmp_path / "nope"))
        with pytest.raises(SystemExit):
            d._resolve_dispatch_workspace(args)


# --------------------------------------------------------------------------
# _resolve_task_paths
# --------------------------------------------------------------------------

class TestResolveTaskPaths:
    def test_bare_name(self, tmp_path):
        t = tmp_path / "test_a"
        t.mkdir()
        (t / "task.yaml").write_text("x")
        out = d._resolve_task_paths(tmp_path, ["test_a"])
        assert out[0]["name"] == "test_a"
        assert Path(out[0]["task_dir"]) == t.resolve()

    def test_test_prefix_legacy(self, tmp_path):
        t = tmp_path / "test_foo"
        t.mkdir()
        (t / "task.yaml").write_text("x")
        # "foo" should resolve to test_foo via the legacy prefix
        out = d._resolve_task_paths(tmp_path, ["foo"])
        assert out[0]["name"] == "test_foo"

    def test_explicit_path(self, tmp_path):
        t = tmp_path / "sub" / "mytask"
        t.mkdir(parents=True)
        (t / "task.yaml").write_text("x")
        out = d._resolve_task_paths(tmp_path, [str(t)])
        assert Path(out[0]["task_dir"]) == t.resolve()

    def test_unresolvable_dies(self, tmp_path):
        with pytest.raises(SystemExit):
            d._resolve_task_paths(tmp_path, ["does_not_exist"])


# --------------------------------------------------------------------------
# _validate_prompt_files
# --------------------------------------------------------------------------

class TestValidatePromptFiles:
    def test_all_present(self, tmp_path):
        t = tmp_path / "test_a"
        (t / "prompts").mkdir(parents=True)
        (t / "prompts" / "2_iterate.md").write_text("p")
        tasks = [{"name": "test_a", "task_dir": str(t)}]
        d._validate_prompt_files("prompts/2_iterate.md", tasks)  # no raise

    def test_missing_dies(self, tmp_path):
        t = tmp_path / "test_a"
        t.mkdir()
        tasks = [{"name": "test_a", "task_dir": str(t)}]
        with pytest.raises(SystemExit):
            d._validate_prompt_files("prompts/2_iterate.md", tasks)


# --------------------------------------------------------------------------
# cmd_dispatch (dry-run + gates)
# --------------------------------------------------------------------------

class TestCmdDispatch:
    @pytest.fixture
    def workspace(self, tmp_path):
        t = tmp_path / "test_a"
        (t / "prompts").mkdir(parents=True)
        (t / "task.yaml").write_text("x")
        (t / "prompts" / "2_iterate.md").write_text("p")
        return tmp_path

    def _patch_dispatch(self, monkeypatch, *, prior_pid=None, alive=False):
        """Stub the lazy `from zyme.dispatch import ...` symbols."""
        import zyme.dispatch as zd
        import zyme.dispatch.state as zds
        import zyme.dispatch.resources as zdr
        monkeypatch.setattr(zd, "find_agent_binary", lambda agent: "/bin/" + agent,
                            raising=False)
        monkeypatch.setattr(zdr, "parse_size_gb", lambda s: None if s == "auto" else 8.0)
        monkeypatch.setattr(zds, "read_pid", lambda p: prior_pid)
        monkeypatch.setattr(zds, "pid_alive", lambda pid: alive)
        monkeypatch.setattr(zds, "pid_path", lambda ws: ws / "pid")
        monkeypatch.setattr(zds, "state_path", lambda ws: ws / "state.json")
        monkeypatch.setattr(zds, "master_log_path", lambda ws: ws / "log")
        monkeypatch.setattr(zds, "ensure_dispatch_dirs", lambda ws: None)

    def test_dry_run(self, workspace, monkeypatch, capsys):
        self._patch_dispatch(monkeypatch)
        args = SimpleNamespace(
            workspace=str(workspace), tasks=["test_a"],
            prompt="prompts/2_iterate.md", agent="claude", model="m",
            effort="high", max_rounds=10, force_mode=False, reflect=False,
            reflect_prompt=None, ram_floor="8G", disk_floor="auto",
            detach=False, dry_run=True, stall_threshold=900,
            reflection_root=None, reflect_category=None)
        d.cmd_dispatch(args)
        out = capsys.readouterr().out
        assert "queue" in out
        assert "test_a" in out

    def test_refuses_when_master_running(self, workspace, monkeypatch):
        self._patch_dispatch(monkeypatch, prior_pid=4242, alive=True)
        args = SimpleNamespace(
            workspace=str(workspace), tasks=["test_a"],
            prompt="prompts/2_iterate.md", agent="claude", model="m",
            effort="high", max_rounds=10, force_mode=False, reflect=False,
            reflect_prompt=None, ram_floor="8G", disk_floor="auto",
            detach=False, dry_run=True, stall_threshold=900,
            reflection_root=None, reflect_category=None)
        with pytest.raises(SystemExit):
            d.cmd_dispatch(args)

    def test_bad_ram_floor_dies(self, workspace, monkeypatch):
        import zyme.dispatch as zd
        import zyme.dispatch.state as zds
        import zyme.dispatch.resources as zdr
        monkeypatch.setattr(zd, "find_agent_binary", lambda a: "/bin/x", raising=False)
        monkeypatch.setattr(zds, "read_pid", lambda p: None)
        monkeypatch.setattr(zds, "pid_alive", lambda pid: False)
        monkeypatch.setattr(zds, "pid_path", lambda ws: ws / "pid")

        def bad_parse(s):
            raise ValueError("bad size")
        monkeypatch.setattr(zdr, "parse_size_gb", bad_parse)
        args = SimpleNamespace(
            workspace=str(workspace), tasks=["test_a"],
            prompt="prompts/2_iterate.md", agent="claude", model="m",
            effort="high", max_rounds=10, force_mode=False, reflect=False,
            reflect_prompt=None, ram_floor="garbage", disk_floor="auto",
            detach=False, dry_run=True, stall_threshold=900,
            reflection_root=None, reflect_category=None)
        with pytest.raises(SystemExit):
            d.cmd_dispatch(args)


# --------------------------------------------------------------------------
# status / usage / prices wrappers
# --------------------------------------------------------------------------

class TestStatusUsagePrices:
    def test_status(self, tmp_path, monkeypatch, capsys):
        import zyme.dispatch as zd
        monkeypatch.setattr(zd, "render_status", lambda ws: "STATUS OK", raising=False)
        d.cmd_dispatch_status(SimpleNamespace(workspace=str(tmp_path)))
        assert "STATUS OK" in capsys.readouterr().out

    def test_usage_text(self, tmp_path, monkeypatch, capsys):
        import zyme.dispatch as zd
        monkeypatch.setattr(zd, "collect_usage",
                            lambda ws, **k: {"total": 1}, raising=False)
        monkeypatch.setattr(zd, "render_usage", lambda s: "USAGE TABLE", raising=False)
        args = SimpleNamespace(workspace=str(tmp_path), token_budget=None,
                               budget_basis=None, price_model=None,
                               json_output=False)
        d.cmd_dispatch_usage(args)
        assert "USAGE TABLE" in capsys.readouterr().out

    def test_usage_json(self, tmp_path, monkeypatch, capsys):
        import zyme.dispatch as zd
        monkeypatch.setattr(zd, "collect_usage",
                            lambda ws, **k: {"total": 1}, raising=False)
        args = SimpleNamespace(workspace=str(tmp_path), token_budget=None,
                               budget_basis=None, price_model=None,
                               json_output=True)
        d.cmd_dispatch_usage(args)
        assert '"total": 1' in capsys.readouterr().out

    def test_prices_text(self, tmp_path, monkeypatch, capsys):
        import zyme.dispatch as zd
        monkeypatch.setattr(zd, "render_price_table", lambda: "PRICES", raising=False)
        d.cmd_dispatch_prices(SimpleNamespace(json_output=False))
        assert "PRICES" in capsys.readouterr().out

    def test_prices_json(self, monkeypatch, capsys):
        import zyme.dispatch as zd
        monkeypatch.setattr(zd, "list_prices", lambda: {"m": 1.0}, raising=False)
        d.cmd_dispatch_prices(SimpleNamespace(json_output=True))
        assert '"m"' in capsys.readouterr().out


# --------------------------------------------------------------------------
# stop
# --------------------------------------------------------------------------

class TestStop:
    def test_clean_stop(self, tmp_path, monkeypatch, capsys):
        import zyme.dispatch as zd
        monkeypatch.setattr(zd, "stop_dispatch", lambda ws: {
            "master_pid": 100, "master_stopped": True, "agent_pids": [101]},
            raising=False)
        d.cmd_dispatch_stop(SimpleNamespace(workspace=str(tmp_path)))
        out = capsys.readouterr().out
        assert "master stopped: True" in out
        assert "agent pids" in out

    def test_error_when_not_stopped(self, tmp_path, monkeypatch, capsys):
        import zyme.dispatch as zd
        monkeypatch.setattr(zd, "stop_dispatch", lambda ws: {
            "master_pid": None, "master_stopped": False,
            "error": "no master found"}, raising=False)
        d.cmd_dispatch_stop(SimpleNamespace(workspace=str(tmp_path)))
        err = capsys.readouterr().err
        assert "no master found" in err

    def test_soft_error_after_stop(self, tmp_path, monkeypatch, capsys):
        import zyme.dispatch as zd
        monkeypatch.setattr(zd, "stop_dispatch", lambda ws: {
            "master_pid": 5, "master_stopped": True,
            "error": "pid file was stale"}, raising=False)
        d.cmd_dispatch_stop(SimpleNamespace(workspace=str(tmp_path)))
        out = capsys.readouterr().out
        assert "pid file was stale" in out


# --------------------------------------------------------------------------
# resume (dry-run + error gates)
# --------------------------------------------------------------------------

class TestResume:
    def _patch_state(self, monkeypatch, state, *, alive=False):
        import zyme.dispatch.state as zds
        monkeypatch.setattr(zds, "read_pid", lambda p: 1 if alive else None)
        monkeypatch.setattr(zds, "pid_alive", lambda pid: alive)
        monkeypatch.setattr(zds, "pid_path", lambda ws: ws / "pid")
        monkeypatch.setattr(zds, "state_path", lambda ws: ws / "s.json")
        monkeypatch.setattr(zds, "read_state", lambda p: state)

    def test_no_state_dies(self, tmp_path, monkeypatch):
        self._patch_state(monkeypatch, None)
        args = SimpleNamespace(workspace=str(tmp_path), task=None, prompt=None,
                               message=None, dry_run=True, stall_threshold=900)
        with pytest.raises(SystemExit):
            d.cmd_dispatch_resume(args)

    def test_still_running_dies(self, tmp_path, monkeypatch):
        self._patch_state(monkeypatch, {"queue": []}, alive=True)
        args = SimpleNamespace(workspace=str(tmp_path), task=None, prompt=None,
                               message=None, dry_run=True, stall_threshold=900)
        with pytest.raises(SystemExit):
            d.cmd_dispatch_resume(args)

    def test_single_task_dry_run(self, tmp_path, monkeypatch, capsys):
        state = {
            "agent": "claude", "model": "m",
            "queue": [{"name": "test_a", "task_dir": str(tmp_path / "test_a"),
                       "agent_session_id": "sess1"}],
        }
        self._patch_state(monkeypatch, state)
        args = SimpleNamespace(workspace=str(tmp_path), task=None,
                               prompt=None, message="hi", dry_run=True,
                               stall_threshold=900)
        d.cmd_dispatch_resume(args)
        out = capsys.readouterr().out
        assert "test_a" in out
        assert "sess1" in out

    def test_multi_task_requires_name(self, tmp_path, monkeypatch):
        state = {"agent": "claude", "model": "m", "queue": [
            {"name": "test_a", "task_dir": "/a", "agent_session_id": "s1"},
            {"name": "test_b", "task_dir": "/b", "agent_session_id": "s2"}]}
        self._patch_state(monkeypatch, state)
        args = SimpleNamespace(workspace=str(tmp_path), task=None, prompt=None,
                               message=None, dry_run=True, stall_threshold=900)
        with pytest.raises(SystemExit):
            d.cmd_dispatch_resume(args)

    def test_unknown_task_dies(self, tmp_path, monkeypatch):
        state = {"agent": "claude", "model": "m", "queue": [
            {"name": "test_a", "task_dir": "/a", "agent_session_id": "s1"}]}
        self._patch_state(monkeypatch, state)
        args = SimpleNamespace(workspace=str(tmp_path), task="ghost",
                               prompt=None, message=None, dry_run=True,
                               stall_threshold=900)
        with pytest.raises(SystemExit):
            d.cmd_dispatch_resume(args)

    def test_no_session_id_dies(self, tmp_path, monkeypatch):
        state = {"agent": "claude", "model": "m", "queue": [
            {"name": "test_a", "task_dir": "/a"}]}  # no agent_session_id
        self._patch_state(monkeypatch, state)
        args = SimpleNamespace(workspace=str(tmp_path), task="test_a",
                               prompt=None, message=None, dry_run=True,
                               stall_threshold=900)
        with pytest.raises(SystemExit):
            d.cmd_dispatch_resume(args)


# --------------------------------------------------------------------------
# wait
# --------------------------------------------------------------------------

class TestWait:
    def test_no_state_dies(self, tmp_path, monkeypatch):
        import zyme.dispatch.state as zds
        monkeypatch.setattr(zds, "read_state", lambda p: None)
        monkeypatch.setattr(zds, "state_path", lambda ws: ws / "s")
        monkeypatch.setattr(zds, "pid_path", lambda ws: ws / "pid")
        args = SimpleNamespace(workspace=str(tmp_path), poll=1, timeout=None,
                               verbose=False)
        with pytest.raises(SystemExit):
            d.cmd_dispatch_wait(args)

    def test_already_finished_exits_zero(self, tmp_path, monkeypatch):
        import zyme.dispatch.state as zds
        monkeypatch.setattr(zds, "state_path", lambda ws: ws / "s")
        monkeypatch.setattr(zds, "pid_path", lambda ws: ws / "pid")
        monkeypatch.setattr(zds, "read_state",
                            lambda p: {"finished_at": "2026", "queue": []})
        monkeypatch.setattr(zds, "read_pid", lambda p: None)
        monkeypatch.setattr(zds, "pid_alive", lambda pid: False)
        args = SimpleNamespace(workspace=str(tmp_path), poll=1, timeout=None,
                               verbose=False)
        with pytest.raises(SystemExit) as ei:
            d.cmd_dispatch_wait(args)
        assert ei.value.code == 0

    def test_crashed_exits_two(self, tmp_path, monkeypatch, capsys):
        import zyme.dispatch.state as zds
        monkeypatch.setattr(zds, "state_path", lambda ws: ws / "s")
        monkeypatch.setattr(zds, "pid_path", lambda ws: ws / "pid")
        # state has no finished_at -> crash
        monkeypatch.setattr(zds, "read_state", lambda p: {"queue": []})
        monkeypatch.setattr(zds, "read_pid", lambda p: None)
        monkeypatch.setattr(zds, "pid_alive", lambda pid: False)
        args = SimpleNamespace(workspace=str(tmp_path), poll=1, timeout=None,
                               verbose=False)
        with pytest.raises(SystemExit) as ei:
            d.cmd_dispatch_wait(args)
        assert ei.value.code == 2


# --------------------------------------------------------------------------
# logs
# --------------------------------------------------------------------------

class TestLogs:
    def test_streams_lines(self, tmp_path, monkeypatch, capsys):
        import zyme.dispatch as zd
        import zyme.dispatch.state as zds
        monkeypatch.setattr(zds, "state_path", lambda ws: ws / "s")
        monkeypatch.setattr(zds, "read_state",
                            lambda p: {"queue": [{"name": "test_a"}]})
        monkeypatch.setattr(zds, "task_events_path",
                            lambda ws, name: ws / f"{name}.events")
        monkeypatch.setattr(zd, "stream_task_logs",
                            lambda ws, task, follow, n: iter(["line1", "line2"]),
                            raising=False)
        args = SimpleNamespace(workspace=str(tmp_path), task="test_a",
                               follow=False, last_n=10, full=False)
        d.cmd_dispatch_logs(args)
        out = capsys.readouterr().out
        assert "line1" in out and "line2" in out

    def test_unknown_task_no_events_dies(self, tmp_path, monkeypatch):
        import zyme.dispatch.state as zds
        monkeypatch.setattr(zds, "state_path", lambda ws: ws / "s")
        monkeypatch.setattr(zds, "read_state",
                            lambda p: {"queue": [{"name": "test_a"}]})
        monkeypatch.setattr(zds, "task_events_path",
                            lambda ws, name: ws / f"{name}.events")
        args = SimpleNamespace(workspace=str(tmp_path), task="ghost",
                               follow=False, last_n=10, full=False)
        with pytest.raises(SystemExit):
            d.cmd_dispatch_logs(args)
