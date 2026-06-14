"""Supplemental unit tests for zyme.cost — gaps left by tests/test_cost.py.

The existing tests/test_cost.py covers the happy paths (cmd_to_phase, parse_ts,
active_minutes, collect_session sum/dedup/window, codex cumulative-delta, and the
compute_task_cost integration). This file targets the UNCOVERED branches:

  * _pick_model empty / _resolve_price_name family fallbacks + passthrough
  * resolve_transcript with no CC root
  * collect_session: malformed JSON lines, <synthetic> model, non-assistant rows,
    missing message dict, empty-usage skip
  * collect_codex_rollout: unparseable, no-cwd, malformed lines
  * _codex_windowed_usage empty
  * _collect_captured: malformed lines, multi-record aggregation, unknown agent
  * parse_stream_usage: cost_usd / duration / empty / no-results-no-codex /
    session-id fallback search
  * _codex_pricing_tokens edge math
  * compute_task_cost: model_override path, no-price warning

It does NOT touch tests/test_cost.py and reuses no fixtures from it (the
task_dir fixture comes from the shared conftest.py).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from zyme import cost as costmod
from zyme.cost import (
    _codex_pricing_tokens,
    _codex_windowed_usage,
    _collect_captured,
    _empty_tokens,
    _pick_model,
    _resolve_price_name,
    append_agent_usage,
    collect_codex_rollout,
    collect_session,
    compute_task_cost,
    parse_stream_usage,
    parse_ts,
    resolve_transcript,
)


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolate_agent_roots(tmp_path, monkeypatch):
    """Same isolation the existing suite uses: never read real ~/.claude/~/.codex."""
    monkeypatch.setattr(costmod, "CC_TRANSCRIPT_ROOT", tmp_path / "_no_claude")
    monkeypatch.setattr(costmod, "CODEX_SESSIONS_ROOT", tmp_path / "_no_codex")


def _write_audit(task_dir: Path, rows: list[dict]) -> None:
    (task_dir / ".zyme" / "audit.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")


# --------------------------------------------------------------------------
# _pick_model / _resolve_price_name
# --------------------------------------------------------------------------
class TestPickModel:
    def test_empty_returns_none(self):
        assert _pick_model({}) is None

    def test_most_frequent_wins(self):
        assert _pick_model({"a": 1, "b": 5, "c": 2}) == "b"


class TestResolvePriceName:
    def test_none_returns_none(self):
        assert _resolve_price_name(None) is None

    def test_known_model_passthrough(self):
        # an alias that find_model_price accepts is returned unchanged
        assert _resolve_price_name("claude-opus-4-8") == "claude-opus-4-8"

    def test_opus_family_fallback(self):
        # B8 fix: an unlisted opus version maps to the family's *-latest price
        # ROW id (with the `anthropic:` prefix) so find_model_price matches it.
        assert _resolve_price_name("claude-opus-4-99-future") == "anthropic:claude-opus-4-latest"

    def test_sonnet_family_fallback(self):
        assert _resolve_price_name("claude-sonnet-9-x") == "anthropic:claude-sonnet-4-latest"

    def test_haiku_family_fallback(self):
        assert _resolve_price_name("claude-haiku-9-x") == "anthropic:claude-haiku-4.5"

    def test_family_fallback_actually_resolves_to_a_price(self):
        # B8 regression: the fallback must yield a real, priced row (previously
        # the bare `claude-*-latest` string was not a key/alias, so this was None).
        from zyme.dispatch.pricing import find_model_price
        for unlisted in ("claude-opus-4-99-future", "claude-sonnet-9-x", "claude-haiku-9-x"):
            resolved = _resolve_price_name(unlisted)
            assert find_model_price(resolved) is not None, unlisted

    def test_unknown_model_passthrough(self):
        # not a claude family; pricing will return None for it downstream
        assert _resolve_price_name("some-random-model") == "some-random-model"


# --------------------------------------------------------------------------
# resolve_transcript with no CC root
# --------------------------------------------------------------------------
def test_resolve_transcript_no_root(monkeypatch, tmp_path):
    monkeypatch.setattr(costmod, "CC_TRANSCRIPT_ROOT", tmp_path / "absent")
    assert resolve_transcript("anything") is None


def test_resolve_transcript_finds_match(monkeypatch, tmp_path):
    root = tmp_path / "proj"
    (root / "encdir").mkdir(parents=True)
    f = root / "encdir" / "sess123.jsonl"
    f.write_text("{}\n")
    monkeypatch.setattr(costmod, "CC_TRANSCRIPT_ROOT", root)
    assert resolve_transcript("sess123") == f
    assert resolve_transcript("missing") is None


# --------------------------------------------------------------------------
# collect_session edge cases
# --------------------------------------------------------------------------
class TestCollectSessionEdges:
    def test_malformed_lines_skipped(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text(
            "not json at all\n"
            "\n"
            + json.dumps({
                "type": "assistant", "timestamp": "2026-05-10T00:00:00Z",
                "requestId": "r", "message": {"id": "m",
                    "model": "claude-opus-4-8",
                    "usage": {"input_tokens": 5, "output_tokens": 7}},
            }) + "\n"
        )
        d = collect_session(p)
        assert d["input_tokens"] == 5 and d["output_tokens"] == 7

    def test_synthetic_model_skipped(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text(json.dumps({
            "type": "assistant", "timestamp": "2026-05-10T00:00:00Z",
            "message": {"id": "m", "model": "<synthetic>",
                        "usage": {"output_tokens": 999}},
        }) + "\n")
        d = collect_session(p)
        assert d["total_tokens"] == 0

    def test_non_assistant_rows_only_timestamped(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text(
            json.dumps({"type": "user", "timestamp": "2026-05-10T00:00:00Z"}) + "\n"
            + json.dumps({"type": "summary", "timestamp": "2026-05-10T00:01:00Z"}) + "\n"
        )
        d = collect_session(p)
        assert d["total_tokens"] == 0
        assert len(d["events"]) == 2  # timestamps still collected

    def test_message_not_dict_skipped(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text(json.dumps({
            "type": "assistant", "timestamp": "2026-05-10T00:00:00Z",
            "message": "oops not a dict",
        }) + "\n")
        d = collect_session(p)
        assert d["total_tokens"] == 0

    def test_empty_usage_skipped(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text(json.dumps({
            "type": "assistant", "timestamp": "2026-05-10T00:00:00Z",
            "message": {"id": "m", "model": "claude-opus-4-8", "usage": {}},
        }) + "\n")
        d = collect_session(p)
        assert d["total_tokens"] == 0
        assert d["n_assistant"] == 0


# --------------------------------------------------------------------------
# collect_codex_rollout edge cases
# --------------------------------------------------------------------------
class TestCollectCodexRollout:
    def test_missing_file_returns_none(self, tmp_path):
        assert collect_codex_rollout(tmp_path / "absent.jsonl") is None

    def test_no_cwd_returns_none(self, tmp_path):
        p = tmp_path / "r.jsonl"
        # token_count event but never a session_meta/turn_context cwd
        p.write_text(json.dumps({
            "timestamp": "2026-05-10T00:00:00Z", "type": "event_msg",
            "payload": {"type": "token_count",
                        "info": {"total_token_usage": {"input_tokens": 5}}},
        }) + "\n")
        assert collect_codex_rollout(p) is None

    def test_malformed_lines_skipped(self, tmp_path):
        p = tmp_path / "r.jsonl"
        p.write_text(
            "garbage\n"
            "   \n"
            + json.dumps({"timestamp": "2026-05-10T00:00:00Z",
                          "type": "session_meta", "payload": {"cwd": "/work"}}) + "\n"
            + json.dumps({"timestamp": "2026-05-10T00:01:00Z",
                          "type": "turn_context",
                          "payload": {"model": "gpt-5.5"}}) + "\n"
        )
        d = collect_codex_rollout(p)
        assert d is not None
        assert d["cwd"] == "/work"
        assert d["model_counts"] == {"gpt-5.5": 1}

    def test_token_events_sorted(self, tmp_path):
        p = tmp_path / "r.jsonl"
        p.write_text(
            json.dumps({"timestamp": "2026-05-10T00:00:00Z", "type": "session_meta",
                        "payload": {"cwd": "/w"}}) + "\n"
            + json.dumps({"timestamp": "2026-05-10T00:05:00Z", "type": "event_msg",
                          "payload": {"type": "token_count",
                                      "info": {"total_token_usage": {"input_tokens": 9}}}}) + "\n"
            + json.dumps({"timestamp": "2026-05-10T00:02:00Z", "type": "event_msg",
                          "payload": {"type": "token_count",
                                      "info": {"total_token_usage": {"input_tokens": 3}}}}) + "\n"
        )
        d = collect_codex_rollout(p)
        ts = [t for t, _ in d["token_events"]]
        assert ts == sorted(ts)


# --------------------------------------------------------------------------
# _codex_windowed_usage / _codex_pricing_tokens
# --------------------------------------------------------------------------
class TestCodexMath:
    def test_windowed_empty_is_zero(self):
        assert _codex_windowed_usage([], None, None) == _empty_tokens()

    def test_pricing_tokens_subtracts_cached_from_input(self):
        out = _codex_pricing_tokens(
            {"input_tokens": 1000, "cached_input_tokens": 300, "output_tokens": 50})
        assert out == {"input_tokens": 700, "cache_read_tokens": 300,
                       "output_tokens": 50, "cache_write_tokens": 0}

    def test_pricing_tokens_none_input(self):
        assert _codex_pricing_tokens(None) == {
            "input_tokens": 0, "cache_read_tokens": 0,
            "output_tokens": 0, "cache_write_tokens": 0}

    def test_pricing_tokens_cached_exceeds_input_floored(self):
        # defensive max(.,0): cached > input shouldn't go negative
        out = _codex_pricing_tokens(
            {"input_tokens": 100, "cached_input_tokens": 500, "output_tokens": 0})
        assert out["input_tokens"] == 0
        assert out["cache_read_tokens"] == 500


# --------------------------------------------------------------------------
# _collect_captured edge cases
# --------------------------------------------------------------------------
class TestCollectCaptured:
    def test_missing_file_empty(self, task_dir):
        # no agent_usage.jsonl
        assert _collect_captured(task_dir) == []

    def test_malformed_lines_skipped_and_aggregated(self, task_dir):
        p = task_dir / ".zyme" / "agent_usage.jsonl"
        p.write_text(
            "not json\n"
            "\n"
            + json.dumps({"ts": "2026-06-06T00:00:00+00:00", "agent": "cursor",
                          "model": "Composer 2", "input_tokens": 100,
                          "output_tokens": 50}) + "\n"
            + json.dumps({"ts": "2026-06-06T00:05:00+00:00", "agent": "cursor",
                          "model": "Composer 2", "input_tokens": 10,
                          "output_tokens": 5}) + "\n"
        )
        out = _collect_captured(task_dir)
        assert len(out) == 1
        a = out[0]
        assert a["agent"] == "cursor"
        assert a["input_tokens"] == 110 and a["output_tokens"] == 55
        assert a["n_records"] == 2
        assert a["n_scoped"] == 2 and a["n_shared"] == 0
        assert a["total_tokens"] == 165
        assert a["model_counts"]["Composer 2"] == 2

    def test_unknown_agent_label(self, task_dir):
        p = task_dir / ".zyme" / "agent_usage.jsonl"
        p.write_text(json.dumps(
            {"ts": "2026-06-06T00:00:00+00:00", "input_tokens": 1}) + "\n")
        out = _collect_captured(task_dir)
        assert out[0]["agent"] == "unknown"

    def test_two_agents_separated(self, task_dir):
        p = task_dir / ".zyme" / "agent_usage.jsonl"
        p.write_text(
            json.dumps({"ts": "2026-06-06T00:00:00+00:00", "agent": "cursor",
                        "input_tokens": 1}) + "\n"
            + json.dumps({"ts": "2026-06-06T00:01:00+00:00", "agent": "claude",
                          "input_tokens": 2}) + "\n"
        )
        out = {a["agent"]: a for a in _collect_captured(task_dir)}
        assert set(out) == {"cursor", "claude"}
        assert out["cursor"]["input_tokens"] == 1
        assert out["claude"]["input_tokens"] == 2


# --------------------------------------------------------------------------
# parse_stream_usage gaps
# --------------------------------------------------------------------------
class TestParseStreamUsage:
    def test_empty_input(self):
        rec = parse_stream_usage([], agent="cursor")
        assert rec["agent"] == "cursor"
        # no total_tokens key is emitted; all four token fields are zero
        assert rec["input_tokens"] == 0 and rec["output_tokens"] == 0
        assert rec["cache_read_tokens"] == 0 and rec["cache_write_tokens"] == 0
        assert rec["n_results"] == 0
        assert rec["model"] is None
        assert rec["session_id"] is None
        assert rec["cost_usd"] is None
        assert rec["duration_ms"] is None

    def test_malformed_lines_skipped(self):
        rec = parse_stream_usage(
            ["garbage", "", json.dumps({"type": "result",
                "usage": {"inputTokens": 4, "outputTokens": 2}})],
            agent="cursor")
        assert rec["input_tokens"] == 4 and rec["output_tokens"] == 2

    def test_cost_usd_summed(self):
        rec = parse_stream_usage([
            json.dumps({"type": "result", "total_cost_usd": 0.5, "duration_ms": 100,
                        "usage": {"inputTokens": 1}}),
            json.dumps({"type": "result", "total_cost_usd": 0.25, "duration_ms": 200,
                        "usage": {"inputTokens": 1}}),
        ], agent="claude")
        assert rec["cost_usd"] == pytest.approx(0.75)
        assert rec["duration_ms"] == 300
        assert rec["n_results"] == 2

    def test_cost_usd_fallback_key(self):
        rec = parse_stream_usage([
            json.dumps({"type": "result", "cost_usd": 1.0,
                        "usage": {"inputTokens": 1}}),
        ], agent="claude")
        assert rec["cost_usd"] == pytest.approx(1.0)

    def test_cost_usd_bad_value_ignored(self):
        rec = parse_stream_usage([
            json.dumps({"type": "result", "total_cost_usd": "notanumber",
                        "usage": {"inputTokens": 1}}),
        ], agent="claude")
        assert rec["cost_usd"] is None

    def test_no_results_no_codex_zero(self):
        rec = parse_stream_usage([
            json.dumps({"type": "system", "model": "Composer 2"}),
        ], agent="cursor")
        assert rec["n_results"] == 0
        assert rec["input_tokens"] == 0
        assert rec["model"] == "Composer 2"

    def test_session_id_deep_search_fallback(self):
        # no top-level/session_meta id, but a nested conversation_id exists
        rec = parse_stream_usage([
            json.dumps({"type": "result", "usage": {"inputTokens": 1},
                        "meta": {"nested": {"conversation_id": "deep-id"}}}),
        ], agent="cursor")
        assert rec["session_id"] == "deep-id"

    def test_session_id_inside_list(self):
        # _find_session_id must recurse into list values too
        rec = parse_stream_usage([
            json.dumps({"type": "result", "usage": {"inputTokens": 1},
                        "items": [{"x": 1}, {"thread_id": "list-id"}]}),
        ], agent="cursor")
        assert rec["session_id"] == "list-id"

    def test_codex_fallback_when_no_results(self):
        # no result rows, but codex token_count events -> codex pricing path
        rec = parse_stream_usage([
            json.dumps({"type": "turn_context", "payload": {"model": "gpt-5.5"}}),
            json.dumps({"type": "event_msg", "payload": {"type": "token_count",
                "info": {"total_token_usage": {"input_tokens": 200,
                                               "cached_input_tokens": 50,
                                               "output_tokens": 10}}}}),
        ], agent="codex")
        assert rec["n_results"] == 1
        assert rec["input_tokens"] == 150  # 200 - 50 cached
        assert rec["cache_read_tokens"] == 50
        assert rec["output_tokens"] == 10

    def test_non_dict_json_line_skipped(self):
        rec = parse_stream_usage([
            json.dumps([1, 2, 3]),  # valid json, but not a dict
            json.dumps({"type": "result", "usage": {"inputTokens": 9}}),
        ], agent="cursor")
        assert rec["input_tokens"] == 9


# --------------------------------------------------------------------------
# compute_task_cost: model_override + no-price warning
# --------------------------------------------------------------------------
class TestComputeTaskCostExtras:
    def test_model_override_applied(self, task_dir, monkeypatch):
        proj_root = task_dir / "_cc"
        enc = costmod._encode_cwd(task_dir.resolve())
        proj = proj_root / enc
        proj.mkdir(parents=True)
        sid = "11111111-1111-1111-1111-111111111111"
        # transcript reports opus, but we override pricing to sonnet-latest
        (proj / f"{sid}.jsonl").write_text(json.dumps({
            "type": "assistant", "timestamp": "2026-05-10T00:05:00Z",
            "requestId": "a", "message": {"id": "1", "model": "claude-opus-4-8",
                "usage": {"input_tokens": 0, "output_tokens": 1_000_000}},
        }) + "\n")
        monkeypatch.setattr(costmod, "CC_TRANSCRIPT_ROOT", proj_root)
        _write_audit(task_dir, [
            {"ts": "2026-05-10T00:00:00+00:00", "cmd": "run",
             "duration_s": 1.0, "cc_session": sid},
        ])
        # use a real alias the price table recognizes (the bare *-latest id is
        # NOT itself an alias, so model_override must be a listed alias).
        rep = compute_task_cost(task_dir, model_override="claude-sonnet-4-5")
        a = rep["agents"][0]
        assert a["dominant_model"] == "claude-sonnet-4-5"
        assert a["cost"]["model_price_id"] == "anthropic:claude-sonnet-4-latest"
        # sonnet output @ $15/Mtok on 1M output = $15.00 (not opus's $25)
        assert a["cost"]["total_usd"] == pytest.approx(15.0)

    def test_no_price_for_model_warning(self, task_dir, monkeypatch):
        proj_root = task_dir / "_cc"
        enc = costmod._encode_cwd(task_dir.resolve())
        proj = proj_root / enc
        proj.mkdir(parents=True)
        sid = "22222222-2222-2222-2222-222222222222"
        (proj / f"{sid}.jsonl").write_text(json.dumps({
            "type": "assistant", "timestamp": "2026-05-10T00:05:00Z",
            "requestId": "a", "message": {"id": "1", "model": "mystery-model-x",
                "usage": {"input_tokens": 100, "output_tokens": 100}},
        }) + "\n")
        monkeypatch.setattr(costmod, "CC_TRANSCRIPT_ROOT", proj_root)
        _write_audit(task_dir, [
            {"ts": "2026-05-10T00:00:00+00:00", "cmd": "run",
             "duration_s": 1.0, "cc_session": sid},
        ])
        rep = compute_task_cost(task_dir)
        a = rep["agents"][0]
        assert a["cost"] is None
        assert any("no price entry" in w for w in rep["warnings"])

    def test_missing_transcript_warning(self, task_dir, monkeypatch):
        # audit references a cc_session whose transcript file doesn't exist
        proj_root = task_dir / "_cc"
        proj_root.mkdir()
        monkeypatch.setattr(costmod, "CC_TRANSCRIPT_ROOT", proj_root)
        _write_audit(task_dir, [
            {"ts": "2026-05-10T00:00:00+00:00", "cmd": "run",
             "duration_s": 1.0, "cc_session": "ghost-session"},
        ])
        rep = compute_task_cost(task_dir)
        # no agents counted; a "not found" warning is emitted
        assert rep["agents"] == []
        assert any("not found" in w for w in rep["warnings"])
