"""Deep coverage tests for zyme.dispatch.master.

Targets the pure helper / renderer / discovery lines not exercised by
tests/dispatch/test_master_rounds.py (which drives the integration path via
cursor stubs):
  - find_*_binary discovery (env override + missing) and detect/find dispatch
  - _resolve_disk_floor / _project_eta / _codex_reasoning_effort /
    _agent_model_mismatch / _resume_encouragement / _yaml_scalar
  - _reflection_file_stem / _available_reflection_path
  - render_status / _status_counts / _fmt_duration / _truncate /
    _running_duration_s
  - _print_summary_line / _format_log_line for every event kind
  - stop_dispatch (no-pidfile + dead-master) / stream_task_logs (missing/tail)
  - _state_task_index resolution + resume_dispatch_task validation errors
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from zyme.dispatch import master
from zyme.dispatch.master import (
    DEFAULT_AGENT,
    _agent_model_mismatch,
    _available_reflection_path,
    _build_agent_cmd,
    _codex_reasoning_effort,
    _fmt_duration,
    _format_log_line,
    _print_summary_line,
    _project_eta,
    _reflection_file_stem,
    _resolve_disk_floor,
    _resume_encouragement,
    _running_duration_s,
    _state_task_index,
    _status_counts,
    _truncate,
    _yaml_scalar,
    detect_agent_binary,
    find_agent_binary,
    find_claude_binary,
    find_codex_binary,
    find_cursor_binary,
    render_status,
    resume_dispatch_task,
    stop_dispatch,
    stream_task_logs,
)
from zyme.dispatch.state import (
    ensure_dispatch_dirs,
    state_path,
    task_events_path,
    write_state_atomic,
)


# --------------------------------------------------------------------------
# Binary discovery
# --------------------------------------------------------------------------

class TestBinaryDiscovery:
    def test_claude_env_override(self, monkeypatch):
        monkeypatch.setenv("ZYME_CLAUDE_BIN", sys.executable)
        assert find_claude_binary() == sys.executable

    def test_claude_env_override_not_executable(self, tmp_path, monkeypatch):
        bad = tmp_path / "not_exec"
        bad.write_text("")  # exists but not +x
        monkeypatch.setenv("ZYME_CLAUDE_BIN", str(bad))
        with pytest.raises(RuntimeError, match="not executable"):
            find_claude_binary()

    def test_codex_env_override(self, monkeypatch):
        monkeypatch.setenv("ZYME_CODEX_BIN", sys.executable)
        assert find_codex_binary() == sys.executable

    def test_cursor_env_override(self, monkeypatch):
        monkeypatch.setenv("ZYME_CURSOR_AGENT_BIN", sys.executable)
        assert find_cursor_binary() == sys.executable

    def test_codex_missing_raises(self, monkeypatch):
        monkeypatch.delenv("ZYME_CODEX_BIN", raising=False)
        monkeypatch.setattr(master.shutil, "which", lambda _n: None)
        with pytest.raises(RuntimeError, match="codex binary not found"):
            find_codex_binary()

    def test_find_agent_binary_dispatch(self, monkeypatch):
        monkeypatch.setenv("ZYME_CLAUDE_BIN", sys.executable)
        monkeypatch.setenv("ZYME_CODEX_BIN", sys.executable)
        monkeypatch.setenv("ZYME_CURSOR_AGENT_BIN", sys.executable)
        assert find_agent_binary("claude") == sys.executable
        assert find_agent_binary("codex") == sys.executable
        assert find_agent_binary("cursor") == sys.executable

    def test_find_agent_binary_unsupported(self):
        with pytest.raises(RuntimeError, match="unsupported dispatch agent"):
            find_agent_binary("gemini")

    def test_detect_agent_binary_prefers_claude(self, monkeypatch):
        monkeypatch.setenv("ZYME_CLAUDE_BIN", sys.executable)
        name, path = detect_agent_binary()
        assert name == "claude"
        assert path == sys.executable

    def test_detect_agent_binary_none_found(self, monkeypatch):
        # Force every finder to fail so the aggregate raises.
        def _raise():
            raise RuntimeError("not found")
        monkeypatch.setattr(master, "find_claude_binary", _raise)
        monkeypatch.setattr(master, "find_codex_binary", _raise)
        monkeypatch.setattr(master, "find_cursor_binary", _raise)
        with pytest.raises(RuntimeError, match="no agent binary found"):
            detect_agent_binary()

    def test_find_agent_binary_auto(self, monkeypatch):
        monkeypatch.setenv("ZYME_CLAUDE_BIN", sys.executable)
        assert find_agent_binary("auto") == sys.executable


# --------------------------------------------------------------------------
# _build_agent_cmd: claude (no-resume) + codex effort variations
# --------------------------------------------------------------------------

class TestBuildAgentCmd:
    def test_claude_fresh_includes_stream_json(self, monkeypatch):
        monkeypatch.setenv("ZYME_CLAUDE_BIN", sys.executable)
        cmd = _build_agent_cmd("claude", "prompts/2_iterate.md", "opus", "max")
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "--effort" in cmd
        assert cmd[cmd.index("--effort") + 1] == "max"

    def test_codex_effort_max_maps_high(self, monkeypatch):
        monkeypatch.setenv("ZYME_CODEX_BIN", sys.executable)
        cmd = _build_agent_cmd("codex", "p.md", "gpt-5.5", "max")
        assert 'model_reasoning_effort="high"' in cmd

    def test_cursor_uses_default_model_when_none(self, monkeypatch):
        monkeypatch.setenv("ZYME_CURSOR_AGENT_BIN", sys.executable)
        cmd = _build_agent_cmd("cursor", "p.md", None, "max")
        assert "--model" in cmd

    def test_unsupported_agent_raises(self):
        with pytest.raises(RuntimeError, match="unsupported dispatch agent"):
            _build_agent_cmd("bogus", "p.md", "m", "e")


# --------------------------------------------------------------------------
# small pure helpers
# --------------------------------------------------------------------------

class TestCodexReasoningEffort:
    def test_max_high(self):
        assert _codex_reasoning_effort("max") == "high"

    def test_passthrough_known(self):
        assert _codex_reasoning_effort("xhigh") == "xhigh"
        assert _codex_reasoning_effort("low") == "low"

    def test_none_and_unknown(self):
        assert _codex_reasoning_effort(None) is None
        assert _codex_reasoning_effort("turbo") is None


class TestAgentModelMismatch:
    def test_missing_args_none(self):
        assert _agent_model_mismatch("claude", None, "x") is None
        assert _agent_model_mismatch("claude", "x", None) is None

    def test_cursor_informational_none(self):
        assert _agent_model_mismatch("cursor", "composer-2", "Composer 2 Fast") is None

    def test_claude_returns_none_by_design(self):
        # Current policy: never hard-fail on model naming for any agent.
        assert _agent_model_mismatch("claude", "opus", "sonnet") is None


class TestResumeEncouragement:
    @pytest.mark.parametrize("completed,fragment", [
        (0, "Continue."),
        (3, "you can do this"),
        (7, "small measured wins"),
        (15, "stronger"),
        (25, "advancing science"),
        (50, "deep work"),
    ])
    def test_tiers(self, completed, fragment):
        assert fragment in _resume_encouragement(completed)


class TestYamlScalar:
    def test_none(self):
        assert _yaml_scalar(None) == "null"

    def test_bool(self):
        assert _yaml_scalar(True) == "true"
        assert _yaml_scalar(False) == "false"

    def test_numbers(self):
        assert _yaml_scalar(5) == "5"
        assert _yaml_scalar(1.5) == "1.5"

    def test_string_quoted(self):
        assert _yaml_scalar("hello") == '"hello"'


class TestResolveDiskFloor:
    def test_explicit_override(self):
        assert _resolve_disk_floor({"disk_floor_gb": 12.5}, {"task_dir": "/x"}) == 12.5

    def test_fallback_when_no_estimate(self, monkeypatch):
        monkeypatch.setattr(master, "estimate_disk_need_gb", lambda _p: 0)
        out = _resolve_disk_floor({"disk_floor_gb": None}, {"task_dir": "/x"})
        assert out == master.DEFAULT_DISK_FLOOR_FALLBACK_GB

    def test_estimate_clamped_to_min(self, monkeypatch):
        monkeypatch.setattr(master, "estimate_disk_need_gb", lambda _p: 0.5)
        out = _resolve_disk_floor({"disk_floor_gb": None}, {"task_dir": "/x"})
        assert out == master.DEFAULT_DISK_FLOOR_MIN_GB

    def test_estimate_used_when_above_min(self, monkeypatch):
        monkeypatch.setattr(master, "estimate_disk_need_gb", lambda _p: 9.0)
        out = _resolve_disk_floor({"disk_floor_gb": None}, {"task_dir": "/x"})
        assert out == 9.0


class TestProjectEta:
    def test_no_durations_none(self):
        assert _project_eta({"queue": []}, []) is None

    def test_zero_pending(self):
        state = {"queue": [{"status": "done"}]}
        assert _project_eta(state, [10.0]) == 0

    def test_projects_mean_times_pending(self):
        state = {"queue": [{"status": "pending"}, {"status": "pending"}]}
        assert _project_eta(state, [10.0, 20.0]) == 30  # 2 pending * mean(15)


class TestFmtDuration:
    def test_none(self):
        assert _fmt_duration(None) == ""

    def test_seconds(self):
        assert _fmt_duration(45) == "45s"

    def test_minutes(self):
        assert _fmt_duration(125) == "2m05s"

    def test_hours(self):
        assert _fmt_duration(3725) == "1h02m"


class TestTruncate:
    def test_short_passthrough(self):
        assert _truncate("abc", 5) == "abc"

    def test_long_ellipsized(self):
        out = _truncate("abcdefgh", 5)
        assert len(out) == 5
        assert out.endswith("…")


class TestRunningDurationS:
    def test_no_started(self):
        assert _running_duration_s({}) is None

    def test_bad_format(self):
        assert _running_duration_s({"started_at": "not-a-date"}) is None

    def test_valid_iso(self):
        out = _running_duration_s({"started_at": "2020-01-01T00:00:00Z"})
        assert isinstance(out, int) and out > 0


# --------------------------------------------------------------------------
# reflection path helpers
# --------------------------------------------------------------------------

class TestReflectionFileStem:
    def test_sanitizes_non_alnum(self):
        assert _reflection_file_stem("iter", "test/foo bar") == "iter_test_foo_bar"

    def test_keeps_dash_underscore(self):
        assert _reflection_file_stem("cat-1", "task_a") == "cat-1_task_a"


class TestAvailableReflectionPath:
    def test_base_when_free(self, tmp_path: Path):
        out = _available_reflection_path(tmp_path, "stem", ".md")
        assert out == tmp_path / "stem.md"

    def test_dated_when_base_taken(self, tmp_path: Path):
        (tmp_path / "stem.md").write_text("x")
        out = _available_reflection_path(tmp_path, "stem", ".md")
        assert out != tmp_path / "stem.md"
        assert "stem__" in out.name

    def test_numbered_when_dated_taken(self, tmp_path: Path):
        import time
        date = time.strftime("%Y-%m-%d")
        (tmp_path / "stem.md").write_text("x")
        (tmp_path / f"stem__{date}.md").write_text("x")
        out = _available_reflection_path(tmp_path, "stem", ".md")
        assert out.name.endswith("_2.md")


# --------------------------------------------------------------------------
# render_status / _status_counts
# --------------------------------------------------------------------------

def _mk_state(tmp_path: Path, **extra) -> dict:
    state = {
        "workspace": str(tmp_path),
        "prompt": "prompts/2_iterate.md",
        "agent": "claude",
        "master_pid": 2_000_000_000,  # dead pid -> 'gone'
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": None,
        "queue": [
            {"name": "task_a", "status": "done", "duration_s": 120,
             "round": 3, "accepts": 2, "rejects": 1,
             "last_event_at": "2026-01-01T00:05:00Z",
             "last_event_kind": "claude_done", "stalled": False},
            {"name": "task_b", "status": "running",
             "started_at": "2026-01-01T00:06:00Z",
             "stalled": True, "reflect_status": "running"},
        ],
    }
    state.update(extra)
    return state


class TestRenderStatus:
    def test_no_state(self, tmp_path: Path):
        out = render_status(tmp_path)
        assert "no dispatch state" in out

    def test_renders_full_snapshot(self, tmp_path: Path):
        ensure_dispatch_dirs(tmp_path)
        state = _mk_state(tmp_path, max_rounds=30, force_mode=True,
                          reflect=True, eta_s=240)
        write_state_atomic(state_path(tmp_path), state)
        out = render_status(tmp_path)
        assert "workspace" in out
        assert "max rounds: 30" in out
        assert "force mode: on" in out
        assert "reflect" in out
        assert "STALLED" in out
        assert "task_a" in out and "task_b" in out
        assert "eta" in out

    def test_status_counts(self, tmp_path: Path):
        state = _mk_state(tmp_path)
        counts = _status_counts(state)
        assert counts["done"] == 1
        assert counts["running"] == 1


# --------------------------------------------------------------------------
# _print_summary_line + _format_log_line — every event kind
# --------------------------------------------------------------------------

_EVENTS = [
    {"kind": "claude_text", "snippet": "hi", "ts": "2026-01-01T00:00:01Z"},
    {"kind": "claude_tool_use", "tool": "Bash", "summary": "ls",
     "ts": "2026-01-01T00:00:02Z"},
    {"kind": "claude_tool_result", "ok": True, "summary": "done",
     "ts": "2026-01-01T00:00:03Z"},
    {"kind": "claude_tool_result", "ok": False, "summary": "boom",
     "ts": "2026-01-01T00:00:03Z"},
    {"kind": "claude_done", "num_turns": 5, "total_cost_usd": 0.5,
     "ts": "2026-01-01T00:00:04Z"},
    {"kind": "codex_event", "event": "x", "summary": "s",
     "ts": "2026-01-01T00:00:05Z"},
    {"kind": "cursor_text", "snippet": "c", "ts": "2026-01-01T00:00:06Z"},
    {"kind": "zyme_run", "hypothesis": "try this", "ts": "2026-01-01T00:00:07Z"},
    {"kind": "zyme_accept", "description": "kept", "ts": "2026-01-01T00:00:08Z"},
    {"kind": "unknown_kind", "ts": "2026-01-01T00:00:09Z"},
]


class TestPrinters:
    @pytest.mark.parametrize("ev", _EVENTS)
    def test_print_summary_line_does_not_raise(self, ev, capsys):
        _print_summary_line("task_a", ev)
        # Output goes to stderr; just confirm something was emitted.
        assert capsys.readouterr().err.strip()

    @pytest.mark.parametrize("ev", _EVENTS)
    def test_format_log_line_covers_kind(self, ev):
        line = _format_log_line("task_a", ev)
        assert isinstance(line, str) and line


# --------------------------------------------------------------------------
# stop_dispatch
# --------------------------------------------------------------------------

class TestStopDispatch:
    def test_no_pid_file(self, tmp_path: Path):
        out = stop_dispatch(tmp_path)
        assert out["error"] == "no master pid file"

    def test_dead_master_cleans_pidfile(self, tmp_path: Path):
        ensure_dispatch_dirs(tmp_path)
        from zyme.dispatch.state import pid_path, write_pid
        write_pid(pid_path(tmp_path), 2_000_000_000)  # dead pid
        out = stop_dispatch(tmp_path)
        assert out["master_stopped"] is True
        assert out["error"] == "master not running"
        assert not pid_path(tmp_path).is_file()


# --------------------------------------------------------------------------
# stream_task_logs
# --------------------------------------------------------------------------

class TestStreamTaskLogs:
    def test_missing_file_yields_nothing(self, tmp_path: Path):
        ensure_dispatch_dirs(tmp_path)
        out = list(stream_task_logs(tmp_path, "ghost", follow=False))
        assert out == []

    def test_tails_n_events(self, tmp_path: Path):
        ensure_dispatch_dirs(tmp_path)
        p = task_events_path(tmp_path, "task_a")
        lines = [
            json.dumps({"ts": f"2026-01-01T00:00:0{i}Z", "kind": "claude_text",
                        "snippet": f"s{i}"})
            for i in range(5)
        ]
        p.write_text("\n".join(lines) + "\n")
        out = list(stream_task_logs(tmp_path, "task_a", follow=False, n=2))
        assert len(out) == 2

    def test_skips_corrupt_lines(self, tmp_path: Path):
        ensure_dispatch_dirs(tmp_path)
        p = task_events_path(tmp_path, "task_a")
        p.write_text(
            '{"ts":"2026-01-01T00:00:00Z","kind":"claude_text","snippet":"ok"}\n'
            'garbage\n'
        )
        out = list(stream_task_logs(tmp_path, "task_a", follow=False, n=None))
        assert len(out) == 1


# --------------------------------------------------------------------------
# _state_task_index + resume_dispatch_task validation
# --------------------------------------------------------------------------

class TestStateTaskIndex:
    def test_empty_queue_raises(self):
        with pytest.raises(RuntimeError, match="no tasks"):
            _state_task_index({"queue": []}, None)

    def test_single_task_no_name(self):
        state = {"queue": [{"name": "only", "task_dir": "/x/only"}]}
        assert _state_task_index(state, None) == 0

    def test_by_name(self):
        state = {"queue": [{"name": "a", "task_dir": "/x/a"},
                           {"name": "b", "task_dir": "/x/b"}]}
        assert _state_task_index(state, "b") == 1

    def test_by_task_dir_basename(self):
        state = {"queue": [{"name": "long_name", "task_dir": "/x/test_foo"}]}
        assert _state_task_index(state, "test_foo") == 0

    def test_unknown_name_raises(self):
        state = {"queue": [{"name": "a", "task_dir": "/x/a"}]}
        with pytest.raises(RuntimeError, match="not found"):
            _state_task_index(state, "ghost")

    def test_multiple_no_name_uses_current_index(self):
        state = {"queue": [{"name": "a", "task_dir": "/x/a"},
                           {"name": "b", "task_dir": "/x/b"}],
                 "current_index": 1}
        assert _state_task_index(state, None) == 1

    def test_multiple_no_name_no_current_raises(self):
        state = {"queue": [{"name": "a", "task_dir": "/x/a"},
                           {"name": "b", "task_dir": "/x/b"}]}
        with pytest.raises(RuntimeError, match="task name is required"):
            _state_task_index(state, None)


class TestResumeDispatchValidation:
    def test_no_state_raises(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="no dispatch state"):
            resume_dispatch_task(workspace=tmp_path, task_name="x")

    def test_no_message_no_prompt_raises(self, tmp_path: Path):
        ensure_dispatch_dirs(tmp_path)
        state = {
            "queue": [{"name": "a", "task_dir": str(tmp_path),
                       "agent_session_id": "sid"}],
            "current_index": 0,
        }
        write_state_atomic(state_path(tmp_path), state)
        with pytest.raises(RuntimeError, match="requires either prompt or message"):
            resume_dispatch_task(workspace=tmp_path, task_name="a")
