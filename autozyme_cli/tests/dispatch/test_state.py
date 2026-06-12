"""Tests for zyme.dispatch.state — atomic state writes, event log, PID file."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from zyme.dispatch.state import (
    append_event,
    ensure_dispatch_dirs,
    events_path,
    follow_events,
    make_event,
    make_initial_state,
    master_log_path,
    pid_alive,
    pid_path,
    read_pid,
    read_state,
    state_path,
    tail_events,
    task_err_path,
    task_events_path,
    task_index,
    task_log_dir,
    task_out_path,
    utcnow_iso,
    workspace_dispatch_dir,
    write_pid,
    write_state_atomic,
)


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------

class TestPathHelpers:
    def test_workspace_dispatch_dir(self, tmp_path: Path):
        assert workspace_dispatch_dir(tmp_path) == tmp_path / ".zyme_dispatch"

    def test_state_path(self, tmp_path: Path):
        assert state_path(tmp_path) == tmp_path / ".zyme_dispatch" / "state.json"

    def test_events_path(self, tmp_path: Path):
        assert events_path(tmp_path) == tmp_path / ".zyme_dispatch" / "events.ndjson"

    def test_pid_path(self, tmp_path: Path):
        assert pid_path(tmp_path) == tmp_path / ".zyme_dispatch" / "master.pid"

    def test_master_log_path(self, tmp_path: Path):
        assert master_log_path(tmp_path).name == "master.log"

    def test_per_task_paths(self, tmp_path: Path):
        assert task_out_path(tmp_path, "abc").name == "abc.out"
        assert task_err_path(tmp_path, "abc").name == "abc.err"
        assert task_events_path(tmp_path, "abc").name == "abc.events.ndjson"

    def test_ensure_dispatch_dirs_creates_dispatch_and_logs(self, tmp_path: Path):
        ensure_dispatch_dirs(tmp_path)
        assert workspace_dispatch_dir(tmp_path).is_dir()
        assert task_log_dir(tmp_path).is_dir()


# --------------------------------------------------------------------------
# utcnow_iso — ISO-format Z-suffixed timestamp
# --------------------------------------------------------------------------

class TestUtcnowIso:
    def test_format(self):
        ts = utcnow_iso()
        # Looks like 2026-05-08T18:34:21.123456+00:00 OR 2026-05-08T18:34:21Z.
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ts)


# --------------------------------------------------------------------------
# write_state_atomic / read_state
# --------------------------------------------------------------------------

class TestStateIo:
    def test_write_then_read(self, tmp_path: Path):
        path = tmp_path / "state.json"
        state = {"schema_version": 1, "queue": [{"name": "a"}]}
        write_state_atomic(path, state)
        assert read_state(path) == state

    def test_read_missing_returns_none(self, tmp_path: Path):
        assert read_state(tmp_path / "absent.json") is None

    def test_read_corrupt_returns_none(self, tmp_path: Path):
        path = tmp_path / "state.json"
        path.write_text("{not valid json")
        assert read_state(path) is None

    def test_atomic_via_tmp_then_rename(self, tmp_path: Path):
        # The implementation writes <name>.tmp then os.replace's. After a
        # successful write, the tmp file should not linger.
        path = tmp_path / "state.json"
        write_state_atomic(path, {"k": 1})
        assert path.exists()
        assert not (tmp_path / "state.json.tmp").exists()

    def test_write_creates_parent_dir(self, tmp_path: Path):
        path = tmp_path / "deep" / "deeper" / "state.json"
        write_state_atomic(path, {"x": 1})
        assert path.is_file()


# --------------------------------------------------------------------------
# make_initial_state / task_index
# --------------------------------------------------------------------------

class TestMakeInitialState:
    def test_basic_shape(self, tmp_path: Path):
        s = make_initial_state(
            workspace=tmp_path,
            tasks=[{"name": "a", "task_dir": str(tmp_path / "a")},
                   {"name": "b", "task_dir": str(tmp_path / "b")}],
            prompt="prompts/x.md", model="opus", effort="med",
            ram_floor_gb=8.0, disk_floor_gb=10.0, detached=False,
        )
        assert s["schema_version"] == 1
        assert s["finished_at"] is None
        assert s["current_index"] == 0
        assert len(s["queue"]) == 2
        # Queue entries enriched with runtime fields.
        first = s["queue"][0]
        assert first["status"] == "pending"
        assert first["rc"] is None
        assert first["accepts"] == 0
        assert first["stalled"] is False

    def test_task_dir_resolved(self, tmp_path: Path):
        # Even relative paths get resolved.
        s = make_initial_state(
            workspace=tmp_path,
            tasks=[{"name": "a", "task_dir": "a/b/../c"}],
            prompt="x", model="m", effort="e",
            ram_floor_gb=1, disk_floor_gb=1, detached=False,
        )
        assert Path(s["queue"][0]["task_dir"]).is_absolute()


class TestTaskIndex:
    def test_finds_existing(self):
        state = {"queue": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
        assert task_index(state, "b") == 1

    def test_raises_when_missing(self):
        state = {"queue": [{"name": "a"}]}
        with pytest.raises(KeyError, match="not in queue"):
            task_index(state, "ghost")


# --------------------------------------------------------------------------
# Event log
# --------------------------------------------------------------------------

class TestEventLog:
    def test_make_event_includes_ts_and_kind(self):
        e = make_event("test_kind", foo=1, bar="x")
        assert e["kind"] == "test_kind"
        assert "ts" in e
        assert e["foo"] == 1
        assert e["bar"] == "x"

    def test_append_event_writes_ndjson_line(self, tmp_path: Path):
        path = tmp_path / "events.ndjson"
        append_event(path, {"ts": "now", "kind": "k1", "x": 1})
        append_event(path, {"ts": "later", "kind": "k2"})
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["x"] == 1
        assert json.loads(lines[1])["kind"] == "k2"

    def test_append_creates_parent_dir(self, tmp_path: Path):
        path = tmp_path / "deep" / "events.ndjson"
        append_event(path, {"ts": "t", "kind": "k"})
        assert path.is_file()

    def test_append_swallows_oserror(self, tmp_path: Path, monkeypatch):
        # Force open() to raise OSError; append_event must not propagate.
        path = tmp_path / "events.ndjson"

        original_open = open

        def _raising_open(*a, **kw):
            if str(a[0]).endswith("events.ndjson"):
                raise OSError("simulated disk full")
            return original_open(*a, **kw)

        monkeypatch.setattr("builtins.open", _raising_open)
        # Should NOT raise.
        append_event(path, {"ts": "t", "kind": "k"})


class TestTailEvents:
    def test_empty_when_missing(self, tmp_path: Path):
        assert tail_events(tmp_path / "absent.ndjson") == []

    def test_returns_last_n(self, tmp_path: Path):
        path = tmp_path / "events.ndjson"
        for i in range(5):
            append_event(path, {"ts": f"t{i}", "kind": f"k{i}"})
        out = tail_events(path, n=3)
        assert [e["kind"] for e in out] == ["k2", "k3", "k4"]

    def test_n_larger_than_lines(self, tmp_path: Path):
        path = tmp_path / "events.ndjson"
        append_event(path, {"ts": "t", "kind": "only"})
        out = tail_events(path, n=10)
        assert len(out) == 1

    def test_skips_corrupt_lines(self, tmp_path: Path):
        path = tmp_path / "events.ndjson"
        path.write_text(
            '{"ts":"a","kind":"k1"}\n'
            'not json\n'
            '{"ts":"b","kind":"k2"}\n'
        )
        out = tail_events(path, n=10)
        assert [e["kind"] for e in out] == ["k1", "k2"]


# --------------------------------------------------------------------------
# PID file
# --------------------------------------------------------------------------

class TestPidFile:
    def test_write_then_read(self, tmp_path: Path):
        path = tmp_path / "master.pid"
        write_pid(path, 12345)
        assert read_pid(path) == 12345

    def test_read_missing(self, tmp_path: Path):
        assert read_pid(tmp_path / "absent.pid") is None

    def test_read_corrupt(self, tmp_path: Path):
        path = tmp_path / "master.pid"
        path.write_text("not a number")
        assert read_pid(path) is None

    def test_pid_alive_for_self(self):
        # Our own process is alive.
        assert pid_alive(os.getpid()) is True

    def test_pid_alive_for_pid_zero(self):
        # PID 0 is special and most OSes treat signals to it as the process
        # group. Whatever the impl returns, it must not raise.
        result = pid_alive(0)
        assert isinstance(result, bool)
