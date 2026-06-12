"""Tests for zyme.dispatch.events — claude stream-json parser + state updater.

The parser sits between claude's `--output-format stream-json` stdout and
the dispatch master's state.json/events.ndjson. It must be tolerant of
malformed lines, schema drift, and unexpected shapes — anything else
crashes the master mid-task. These tests pin the recognized shapes and
the no-raise-on-junk contract.
"""
from __future__ import annotations

import json

import pytest

from zyme.dispatch.events import (
    apply_event_to_task_state,
    parse_claude_line,
    parse_codex_line,
    parse_cursor_line,
    parse_lines,
)


# --------------------------------------------------------------------------
# parse_claude_line — line-level robustness
# --------------------------------------------------------------------------

class TestParseClaudeLineRobustness:
    def test_empty_line(self):
        assert parse_claude_line("") == []
        assert parse_claude_line("   \n") == []

    def test_invalid_json(self):
        assert parse_claude_line("not json at all") == []

    def test_non_dict_json(self):
        assert parse_claude_line("[1,2,3]") == []
        assert parse_claude_line('"a string"') == []

    def test_unknown_type(self):
        assert parse_claude_line('{"type":"system","x":1}') == []

    def test_missing_type_field(self):
        assert parse_claude_line('{"x":1}') == []


class TestParseCodexLineRobustness:
    def test_invalid_json_becomes_text_event(self):
        out = parse_codex_line("plain output")
        assert out[0]["kind"] == "codex_text"
        assert out[0]["snippet"] == "plain output"

    def test_message_like_event_becomes_text(self):
        out = parse_codex_line(json.dumps({
            "type": "assistant_message",
            "message": "Working on it",
        }))
        assert out[0]["kind"] == "claude_text"
        assert out[0]["snippet"] == "Working on it"

    def test_tool_event_extracts_zyme_run(self):
        out = parse_codex_line(json.dumps({
            "type": "tool_call",
            "tool": "shell",
            "input": {"command": "zyme run \"try faster sparse path\""},
        }))
        assert [e["kind"] for e in out] == ["claude_tool_use", "zyme_run"]
        assert out[1]["hypothesis"] == "try faster sparse path"

    def test_exec_command_begin_extracts_zyme_run(self):
        out = parse_codex_line(json.dumps({
            "id": "0",
            "msg": {
                "type": "exec_command_begin",
                "command": ["zsh", "-lc", "zyme run \"try matrix prefilter\""],
            },
        }))
        assert [e["kind"] for e in out] == ["claude_tool_use", "zyme_run"]
        assert out[1]["hypothesis"] == "try matrix prefilter"

    def test_new_item_completed_agent_message_becomes_text(self):
        out = parse_codex_line(json.dumps({
            "type": "item.completed",
            "item": {"id": "item_0", "type": "agent_message", "text": "OK"},
        }))
        assert out[0]["kind"] == "claude_text"
        assert out[0]["snippet"] == "OK"

    def test_new_item_exec_command_extracts_zyme_run(self):
        out = parse_codex_line(json.dumps({
            "type": "item.started",
            "item": {
                "type": "exec_command_begin",
                "command": ["zsh", "-lc", "zyme run \"try direct sums\""],
            },
        }))
        assert [e["kind"] for e in out] == ["claude_tool_use", "zyme_run"]
        assert out[1]["hypothesis"] == "try direct sums"

    def test_done_event_maps_to_done(self):
        out = parse_codex_line(json.dumps({
            "type": "completed",
            "duration_ms": 123,
        }))
        assert out[0]["kind"] == "claude_done"
        assert out[0]["duration_ms"] == 123

    def test_nested_error_event_maps_to_error_done(self):
        out = parse_codex_line(json.dumps({
            "id": "0",
            "msg": {"type": "error", "message": "model requires newer CLI"},
        }))
        assert out[0]["kind"] == "claude_done"
        assert out[0]["is_error"] is True
        assert out[0]["message"] == "model requires newer CLI"


class TestParseCursorLineRobustness:
    def test_plain_output_remains_visible(self):
        out = parse_cursor_line("plain cursor output")
        assert out[0]["kind"] == "cursor_text"
        assert out[0]["snippet"] == "plain cursor output"

    def test_generic_json_event_is_cursor_scoped(self):
        out = parse_cursor_line(json.dumps({
            "type": "session_update",
            "summary": "queued",
        }))
        assert out[0]["kind"] == "cursor_event"
        assert out[0]["event"] == "session_update"

    def test_cursor_stream_json_assistant_text(self):
        out = parse_cursor_line(json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": "OK"}
            ]},
        }))
        assert out[0]["kind"] == "claude_text"
        assert out[0]["snippet"] == "OK"

    def test_cursor_stream_json_result(self):
        out = parse_cursor_line(json.dumps({
            "type": "result",
            "subtype": "success",
            "duration_ms": 2959,
            "is_error": False,
            "session_id": "cursor-session-2",
        }))
        assert out[0]["kind"] == "claude_done"
        assert out[0]["duration_ms"] == 2959
        assert out[0]["session_id"] == "cursor-session-2"

    def test_cursor_system_model_event(self):
        out = parse_cursor_line(json.dumps({
            "type": "system",
            "subtype": "init",
            "model": "Composer 2 Fast",
            "session_id": "cursor-session-1",
        }))
        assert out[0]["kind"] == "agent_model"
        assert out[0]["agent"] == "cursor"
        assert out[0]["model"] == "Composer 2 Fast"
        assert out[0]["session_id"] == "cursor-session-1"


# --------------------------------------------------------------------------
# assistant: text + tool_use blocks
# --------------------------------------------------------------------------

class TestAssistantText:
    def test_simple_text_block(self):
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello world"}]},
        })
        out = parse_claude_line(line)
        assert len(out) == 1
        assert out[0]["kind"] == "claude_text"
        assert out[0]["snippet"] == "Hello world"
        assert out[0]["char_count"] == 11

    def test_empty_text_dropped(self):
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "   "}]},
        })
        assert parse_claude_line(line) == []

    def test_long_text_truncated(self):
        long = "x" * 1000
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": long}]},
        })
        out = parse_claude_line(line)
        assert len(out[0]["snippet"]) <= 240
        assert out[0]["char_count"] == 1000

    def test_newlines_normalized_in_snippet(self):
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "line1\nline2"}]},
        })
        out = parse_claude_line(line)
        # Snippet uses ⏎ instead of literal newlines so it's one display line.
        assert "\n" not in out[0]["snippet"]


class TestAssistantToolUse:
    def test_bash_tool_use(self):
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}
            ]},
        })
        out = parse_claude_line(line)
        assert len(out) == 1
        assert out[0]["kind"] == "claude_tool_use"
        assert out[0]["tool"] == "Bash"
        assert "ls" in out[0]["summary"]

    def test_zyme_run_emits_extra_event(self):
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "name": "Bash",
                 "input": {"command": 'zyme run "vectorized wilcoxon"'}}
            ]},
        })
        out = parse_claude_line(line)
        # Two events: tool_use + zyme_run.
        kinds = [e["kind"] for e in out]
        assert kinds == ["claude_tool_use", "zyme_run"]
        assert out[1]["hypothesis"] == "vectorized wilcoxon"

    def test_zyme_accept_emits_extra_event(self):
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "name": "Bash",
                 "input": {"command": 'zyme accept -m "kept the patch"'}}
            ]},
        })
        out = parse_claude_line(line)
        zev = next(e for e in out if e["kind"] == "zyme_accept")
        assert zev["description"] == "kept the patch"

    def test_zyme_reject_emits_extra_event(self):
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "name": "Bash",
                 "input": {"command": 'zyme reject "concordance failure"'}}
            ]},
        })
        out = parse_claude_line(line)
        zev = next(e for e in out if e["kind"] == "zyme_reject")
        assert zev["description"] == "concordance failure"

    def test_zyme_run_inside_chained_command(self):
        # `cd test_x && zyme run "..."` — the second segment is recognized.
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "name": "Bash",
                 "input": {"command": 'cd test_x && zyme run "tighter loop"'}}
            ]},
        })
        out = parse_claude_line(line)
        kinds = [e["kind"] for e in out]
        assert "zyme_run" in kinds

    def test_non_zyme_bash_no_extra_event(self):
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "name": "Bash",
                 "input": {"command": "git status"}}
            ]},
        })
        out = parse_claude_line(line)
        assert [e["kind"] for e in out] == ["claude_tool_use"]

    def test_read_tool_summary(self):
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "name": "Read",
                 "input": {"file_path": "/path/to/file.py"}}
            ]},
        })
        out = parse_claude_line(line)
        assert out[0]["summary"] == "/path/to/file.py"

    def test_grep_tool_summary(self):
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "name": "Grep",
                 "input": {"pattern": "TODO", "path": "src/"}}
            ]},
        })
        out = parse_claude_line(line)
        assert "TODO" in out[0]["summary"]
        assert "src/" in out[0]["summary"]

    def test_todowrite_tool_summary(self):
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "name": "TodoWrite",
                 "input": {"todos": [{"x": 1}, {"x": 2}, {"x": 3}]}}
            ]},
        })
        out = parse_claude_line(line)
        assert out[0]["summary"] == "3 todo(s)"

    def test_unknown_tool_falls_back_to_keys(self):
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "tool_use", "name": "MysteryTool",
                 "input": {"alpha": 1, "beta": 2}}
            ]},
        })
        out = parse_claude_line(line)
        # Falls back to alphabetical keys.
        assert "alpha" in out[0]["summary"]
        assert "beta" in out[0]["summary"]

    def test_mixed_blocks(self):
        # text + tool_use in same assistant message → both events emitted.
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [
                {"type": "text", "text": "Now running..."},
                {"type": "tool_use", "name": "Bash", "input": {"command": "echo hi"}},
            ]},
        })
        out = parse_claude_line(line)
        assert [e["kind"] for e in out] == ["claude_text", "claude_tool_use"]

    def test_malformed_block_skipped(self):
        line = json.dumps({
            "type": "assistant",
            "message": {"content": ["not a dict", {"type": "text", "text": "ok"}]},
        })
        out = parse_claude_line(line)
        assert [e["kind"] for e in out] == ["claude_text"]


# --------------------------------------------------------------------------
# user: tool_result blocks
# --------------------------------------------------------------------------

class TestUserToolResult:
    def test_string_content(self):
        line = json.dumps({
            "type": "user",
            "message": {"content": [
                {"type": "tool_result", "content": "stdout from cmd"},
            ]},
        })
        out = parse_claude_line(line)
        assert out[0]["kind"] == "claude_tool_result"
        assert out[0]["ok"] is True
        assert "stdout from cmd" in out[0]["summary"]

    def test_error_flag(self):
        line = json.dumps({
            "type": "user",
            "message": {"content": [
                {"type": "tool_result", "is_error": True, "content": "boom"},
            ]},
        })
        out = parse_claude_line(line)
        assert out[0]["ok"] is False

    def test_block_content_flattened(self):
        # tool_result.content can be [{type:text, text:...}, ...]
        line = json.dumps({
            "type": "user",
            "message": {"content": [
                {"type": "tool_result", "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ]},
            ]},
        })
        out = parse_claude_line(line)
        # Joined, then truncated.
        assert "first" in out[0]["summary"]
        assert "second" in out[0]["summary"]

    def test_non_tool_result_block_skipped(self):
        line = json.dumps({
            "type": "user",
            "message": {"content": [{"type": "something_else"}]},
        })
        assert parse_claude_line(line) == []


# --------------------------------------------------------------------------
# result: terminal event
# --------------------------------------------------------------------------

class TestResultEvent:
    def test_basic(self):
        line = json.dumps({
            "type": "result",
            "num_turns": 12,
            "duration_ms": 45000,
            "duration_api_ms": 43000,
            "total_cost_usd": 0.42,
            "request_id": "req-1",
            "usage": {"inputTokens": 10, "outputTokens": 4},
            "is_error": False,
            "subtype": "success",
        })
        out = parse_claude_line(line)
        assert len(out) == 1
        assert out[0]["kind"] == "claude_done"
        assert out[0]["num_turns"] == 12
        assert out[0]["duration_ms"] == 45000
        assert out[0]["duration_api_ms"] == 43000
        assert out[0]["total_cost_usd"] == 0.42
        assert out[0]["request_id"] == "req-1"
        assert out[0]["usage"] == {"inputTokens": 10, "outputTokens": 4}
        assert out[0]["is_error"] is False

    def test_missing_fields_become_none(self):
        line = json.dumps({"type": "result"})
        out = parse_claude_line(line)
        assert out[0]["kind"] == "claude_done"
        assert out[0]["num_turns"] is None
        assert out[0]["is_error"] is False  # bool-coerced from None


# --------------------------------------------------------------------------
# parse_lines (iterable convenience)
# --------------------------------------------------------------------------

class TestParseLines:
    def test_aggregates(self):
        lines = [
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "first"}]}}),
            "",
            json.dumps({"type": "result", "num_turns": 1}),
            "junk",
        ]
        out = parse_lines(lines)
        kinds = [e["kind"] for e in out]
        assert kinds == ["claude_text", "claude_done"]


# --------------------------------------------------------------------------
# apply_event_to_task_state
# --------------------------------------------------------------------------

class TestApplyEventToTaskState:
    def _empty_state(self):
        return {
            "n_events": 0, "accepts": 0, "rejects": 0,
            "stalled": True, "round": None,
        }

    def test_increments_n_events_and_clears_stall(self):
        state = self._empty_state()
        apply_event_to_task_state(state, {"ts": "t1", "kind": "claude_text"})
        assert state["n_events"] == 1
        assert state["last_event_at"] == "t1"
        assert state["last_event_kind"] == "claude_text"
        assert state["stalled"] is False

    def test_zyme_accept_increments_accepts_and_round(self):
        state = self._empty_state()
        apply_event_to_task_state(state, {"ts": "t", "kind": "zyme_accept"})
        assert state["accepts"] == 1
        assert state["round"] == 1

    def test_zyme_reject_increments_rejects_and_round(self):
        state = self._empty_state()
        apply_event_to_task_state(state, {"ts": "t", "kind": "zyme_reject"})
        assert state["rejects"] == 1
        assert state["round"] == 1

    def test_round_is_sum(self):
        state = self._empty_state()
        for kind in ("zyme_accept", "zyme_reject", "zyme_accept"):
            apply_event_to_task_state(state, {"ts": "t", "kind": kind})
        assert state["accepts"] == 2
        assert state["rejects"] == 1
        assert state["round"] == 3

    def test_handles_none_initial_counters(self):
        # When state was just constructed and counters aren't set yet.
        state = {}
        apply_event_to_task_state(state, {"ts": "t", "kind": "zyme_accept"})
        assert state["accepts"] == 1
        assert state["round"] == 1

    def test_records_agent_session_id(self):
        state = self._empty_state()
        apply_event_to_task_state(state, {
            "ts": "t",
            "kind": "agent_model",
            "session_id": "abc-123",
        })
        assert state["agent_session_id"] == "abc-123"
