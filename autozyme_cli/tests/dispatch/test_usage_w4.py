"""Wave-4 coverage for zyme.dispatch.usage — the reachable gaps left by
test_usage.py + test_pricing_w3.py.

Targets the small-helper + edge branches:
  - collect_usage budget_basis validation (line 49)
  - blank task-name skip (line 57)
  - raw-log-wins-over-event-log skip (line 78)
  - "no model price match" + "no task usage" notes / render branches
  - _read_raw_log system line + non-result skip
  - _read_event_log agent_model + non-claude_done skip
  - _consume_terminal_event usage-not-dict early return + cost_usd fallback
  - _iter_json_objects: non-dict skip, JSONDecodeError skip, OSError return
  - _find_session_id list/nested recursion
  - _attach_estimated_cost: no-estimate early return
  - numeric/format helpers: _as_int / _as_float / _fmt_int / _fmt_duration_ms
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from zyme.dispatch.state import ensure_dispatch_dirs, state_path, task_out_path, task_events_path
from zyme.dispatch.usage import (
    collect_usage,
    render_usage,
    _as_float,
    _as_int,
    _attach_estimated_cost,
    _consume_terminal_event,
    _find_session_id,
    _fmt_cost,
    _fmt_duration_ms,
    _fmt_int,
    _iter_json_objects,
    _read_event_log,
    _read_raw_log,
    _task_bucket,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_state(tmp_path: Path, state: dict) -> None:
    ensure_dispatch_dirs(tmp_path)
    state_path(tmp_path).write_text(json.dumps(state), encoding="utf-8")


# --------------------------------------------------------------------------
# collect_usage — top-level validation + edge notes
# --------------------------------------------------------------------------

class TestCollectUsageEdges:
    def test_invalid_budget_basis_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="budget_basis"):
            collect_usage(tmp_path, token_budget=10, budget_basis="bogus")

    def test_blank_task_name_skipped(self, tmp_path: Path):
        # A queue entry with an empty name must be skipped (line 57).
        _write_state(tmp_path, {
            "agent": "claude",
            "model": "claude-opus-4-7",
            "queue": [
                {"name": "   ", "status": "done"},
                {"name": "real_task", "status": "done"},
            ],
        })
        summary = collect_usage(tmp_path)
        assert "real_task" in summary["tasks"]
        # The blank-named one didn't become a task.
        assert all(name.strip() for name in summary["tasks"])

    def test_no_tasks_renders_placeholder(self, tmp_path: Path):
        _write_state(tmp_path, {"agent": "claude", "model": "m", "queue": []})
        summary = collect_usage(tmp_path)
        assert summary["tasks"] == {}
        rendered = render_usage(summary)
        assert "(no task usage found)" in rendered

    def test_unpriced_note_when_no_model_match(self, tmp_path: Path):
        # A done event with usage but an unknown model -> estimated cost None,
        # so the "No model price match" note fires (lines 127-135).
        _write_state(tmp_path, {
            "agent": "claude",
            "model": "totally-unknown-model-xyz",
            "queue": [{"name": "task_u", "status": "done"}],
        })
        _write_jsonl(task_events_path(tmp_path, "task_u"), [
            {
                "kind": "claude_done",
                "duration_ms": 100,
                "model": "totally-unknown-model-xyz",
                "usage": {"inputTokens": 5, "outputTokens": 2,
                          "cacheReadTokens": 0, "cacheWriteTokens": 0},
            },
        ])
        summary = collect_usage(tmp_path)
        task = summary["tasks"]["task_u"]
        assert task["requests"] == 1
        assert task["estimated_cost_usd"] is None
        assert any("No model price match" in n for n in summary["notes"])

    def test_raw_log_wins_over_event_log(self, tmp_path: Path):
        # When a raw .out log already produced result events, the matching
        # .events.ndjson is NOT double-counted (line 78 `continue`).
        _write_state(tmp_path, {
            "agent": "cursor",
            "model": "composer-2",
            "queue": [{"name": "task_r", "status": "done"}],
        })
        _write_jsonl(task_out_path(tmp_path, "task_r"), [
            {"type": "result", "duration_ms": 100,
             "usage": {"inputTokens": 10, "outputTokens": 1,
                       "cacheReadTokens": 0, "cacheWriteTokens": 0}},
        ])
        _write_jsonl(task_events_path(tmp_path, "task_r"), [
            {"kind": "claude_done", "duration_ms": 999,
             "usage": {"inputTokens": 9999, "outputTokens": 9999,
                       "cacheReadTokens": 0, "cacheWriteTokens": 0}},
        ])
        summary = collect_usage(tmp_path)
        task = summary["tasks"]["task_r"]
        # Only the raw-log result counted, not the event-log done.
        assert task["requests"] == 1
        assert task["input_tokens"] == 10


# --------------------------------------------------------------------------
# _read_raw_log / _read_event_log line-type branches
# --------------------------------------------------------------------------

class TestReadLogs:
    def test_raw_log_system_line_records_model_and_session(self, tmp_path: Path):
        path = tmp_path / "x.out"
        _write_jsonl(path, [
            {"type": "system", "model": "Composer 2", "session_id": "sid-9"},
            {"type": "other", "ignored": True},  # non-result, non-system skip
            {"type": "result", "duration_ms": 50,
             "usage": {"inputTokens": 3, "outputTokens": 1,
                       "cacheReadTokens": 0, "cacheWriteTokens": 0}},
        ])
        bucket = _task_bucket("x")
        _read_raw_log(path, bucket)
        assert "Composer 2" in bucket["models"]
        assert "sid-9" in bucket["sessions"]
        assert bucket["raw_result_events"] == 1

    def test_event_log_agent_model_and_non_done_skip(self, tmp_path: Path):
        path = tmp_path / "x.events.ndjson"
        _write_jsonl(path, [
            {"kind": "agent_model", "model": "claude-x", "session_id": "sid-2"},
            {"kind": "claude_text", "snippet": "ignored"},  # non-done skip
            {"kind": "claude_done", "duration_ms": 10, "total_cost_usd": 0.1,
             "usage": {"inputTokens": 1, "outputTokens": 1,
                       "cacheReadTokens": 0, "cacheWriteTokens": 0}},
        ])
        bucket = _task_bucket("x")
        _read_event_log(path, bucket)
        assert "claude-x" in bucket["models"]
        assert "sid-2" in bucket["sessions"]
        assert bucket["requests"] == 1


# --------------------------------------------------------------------------
# _consume_terminal_event — usage-not-dict + cost_usd fallback key
# --------------------------------------------------------------------------

class TestConsumeTerminalEvent:
    def test_usage_not_dict_returns_without_tokens(self):
        bucket = _task_bucket("x")
        _consume_terminal_event({"usage": "not-a-dict"}, bucket)
        assert bucket["requests"] == 1
        assert bucket["usage_events"] == 0
        assert bucket["input_tokens"] == 0

    def test_missing_usage_key(self):
        bucket = _task_bucket("x")
        _consume_terminal_event({"duration_ms": 5}, bucket)
        assert bucket["requests"] == 1
        assert bucket["usage_events"] == 0

    def test_cost_usd_fallback_key(self):
        # No total_cost_usd, but a cost_usd alias is honored.
        bucket = _task_bucket("x")
        _consume_terminal_event({"cost_usd": 0.5, "duration_ms": 1}, bucket)
        assert bucket["cost_usd"] == 0.5


# --------------------------------------------------------------------------
# _iter_json_objects — non-dict skip, decode error skip, OSError return
# --------------------------------------------------------------------------

class TestIterJsonObjects:
    def test_skips_non_dict_and_corrupt(self, tmp_path: Path):
        path = tmp_path / "log"
        path.write_text(
            '{"a":1}\n'
            '\n'                  # blank line skipped
            '[1,2,3]\n'           # list (non-dict) skipped
            'not json\n'          # decode error skipped
            '{"b":2}\n'
        )
        objs = list(_iter_json_objects(path))
        assert objs == [{"a": 1}, {"b": 2}]

    def test_oserror_returns_empty(self, tmp_path: Path, monkeypatch):
        path = tmp_path / "log"
        path.write_text('{"a":1}\n')

        import zyme.dispatch.usage as usage_mod
        original_open = open

        def _boom(*a, **kw):
            if a and str(a[0]).endswith("log"):
                raise OSError("cannot open")
            return original_open(*a, **kw)

        monkeypatch.setattr(usage_mod, "open", _boom, raising=False)
        # The function uses builtin open; patch builtins instead.
        monkeypatch.setattr("builtins.open", _boom)
        assert list(_iter_json_objects(path)) == []


# --------------------------------------------------------------------------
# _find_session_id — nested dict + list recursion
# --------------------------------------------------------------------------

class TestFindSessionId:
    def test_direct_key(self):
        assert _find_session_id({"session_id": "abc"}) == "abc"

    def test_camelcase_alias(self):
        assert _find_session_id({"conversationId": "conv-7"}) == "conv-7"

    def test_nested_dict_recursion(self):
        assert _find_session_id({"meta": {"deep": {"thread_id": "t-9"}}}) == "t-9"

    def test_list_recursion(self):
        assert _find_session_id([{"x": 1}, {"sessionId": "s-2"}]) == "s-2"

    def test_no_id_returns_empty(self):
        assert _find_session_id({"nothing": "here", "n": 1}) == ""

    def test_scalar_returns_empty(self):
        assert _find_session_id(42) == ""


# --------------------------------------------------------------------------
# _attach_estimated_cost — no estimate early return
# --------------------------------------------------------------------------

class TestAttachEstimatedCost:
    def test_no_candidates_leaves_cost_none(self):
        task = _task_bucket("x")
        _attach_estimated_cost(task, candidates=[])
        assert task["estimated_cost_usd"] is None
        assert task["pricing_model_id"] is None

    def test_unknown_model_leaves_cost_none(self):
        task = _task_bucket("x")
        task["input_tokens"] = 100
        _attach_estimated_cost(task, candidates=["nonexistent-model-zzz"])
        assert task["estimated_cost_usd"] is None


# --------------------------------------------------------------------------
# numeric + format helpers
# --------------------------------------------------------------------------

class TestNumericHelpers:
    def test_as_int_bool_and_none(self):
        assert _as_int(True) is None
        assert _as_int(None) is None

    def test_as_int_unparseable(self):
        assert _as_int("not-a-number") is None
        assert _as_int(7) == 7
        assert _as_int("9") == 9

    def test_as_float_bool_and_none(self):
        assert _as_float(False) is None
        assert _as_float(None) is None

    def test_as_float_unparseable(self):
        assert _as_float("xyz") is None
        assert _as_float("1.5") == 1.5


class TestFormatHelpers:
    def test_fmt_int_thousands(self):
        assert _fmt_int(1234567) == "1,234,567"

    def test_fmt_int_invalid_returns_na(self):
        assert _fmt_int("oops") == "n/a"
        assert _fmt_int(None) == "n/a"

    def test_fmt_duration_zero(self):
        assert _fmt_duration_ms(0) == "0s"
        assert _fmt_duration_ms(-5) == "0s"

    def test_fmt_duration_seconds(self):
        assert _fmt_duration_ms(1500) == "1.5s"

    def test_fmt_duration_minutes(self):
        # 90_000 ms = 90s = 1m30s
        assert _fmt_duration_ms(90_000) == "1m30s"

    def test_fmt_duration_hours(self):
        # 3_900_000 ms = 3900s = 65m = 1h05m
        assert _fmt_duration_ms(3_900_000) == "1h05m"

    def test_fmt_cost_none_and_value(self):
        assert _fmt_cost(None) == "n/a"
        assert _fmt_cost(0.25) == "$0.2500"
        assert _fmt_cost("bad") == "n/a"
