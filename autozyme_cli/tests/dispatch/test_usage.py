from __future__ import annotations

import json
from pathlib import Path

from zyme.dispatch.state import ensure_dispatch_dirs, state_path, task_out_path, task_events_path
from zyme.dispatch.usage import collect_usage, render_usage


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_collect_usage_reads_cursor_raw_result_tokens(tmp_path: Path):
    ensure_dispatch_dirs(tmp_path)
    state_path(tmp_path).write_text(json.dumps({
        "agent": "cursor",
        "model": "composer-2",
        "effort": "max",
        "max_rounds": 30,
        "force_mode": True,
        "queue": [{"name": "task_a", "status": "running", "task_dir": "/tmp/task_a"}],
    }), encoding="utf-8")
    _write_jsonl(task_out_path(tmp_path, "task_a"), [
        {
            "type": "system",
            "model": "Composer 2 Fast",
            "session_id": "session-1",
        },
        {
            "type": "result",
            "duration_ms": 1000,
            "duration_api_ms": 900,
            "session_id": "session-1",
            "request_id": "request-1",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
                "cacheReadTokens": 100,
                "cacheWriteTokens": 2,
            },
        },
    ])

    summary = collect_usage(tmp_path, token_budget=200, budget_basis="total")

    task = summary["tasks"]["task_a"]
    assert task["requests"] == 1
    assert task["input_tokens"] == 10
    assert task["output_tokens"] == 5
    assert task["cache_read_tokens"] == 100
    assert task["cache_write_tokens"] == 2
    assert task["non_cache_tokens"] == 17
    assert task["total_tokens"] == 117
    assert task["duration_api_ms"] == 900
    assert task["sessions"] == ["session-1"]
    assert task["models"] == ["Composer 2 Fast"]
    assert task["pricing_model_id"] == "cursor:composer-2-fast"
    assert round(task["estimated_cost_usd"], 7) == 0.0002055
    assert summary["totals"]["total_tokens"] == 117
    assert round(summary["totals"]["estimated_cost_usd"], 7) == 0.0002055
    assert summary["budget"]["used_pct"] == 58.5

    rendered = render_usage(summary)
    assert "task_a" in rendered
    assert "$0.0002" in rendered
    assert "117 / 200" in rendered
    assert "Cursor emits token telemetry" in rendered


def test_collect_usage_falls_back_to_parsed_done_events(tmp_path: Path):
    ensure_dispatch_dirs(tmp_path)
    state_path(tmp_path).write_text(json.dumps({
        "agent": "claude",
        "model": "claude-opus-4-7",
        "queue": [{"name": "task_b", "status": "done"}],
    }), encoding="utf-8")
    _write_jsonl(task_events_path(tmp_path, "task_b"), [
        {
            "kind": "agent_model",
            "model": "claude-opus-4-7",
            "session_id": "session-2",
        },
        {
            "kind": "claude_done",
            "duration_ms": 2500,
            "total_cost_usd": 0.25,
            "session_id": "session-2",
            "usage": {
                "inputTokens": 20,
                "outputTokens": 7,
                "cacheReadTokens": 0,
                "cacheWriteTokens": 3,
            },
        },
    ])

    summary = collect_usage(tmp_path, token_budget=30, budget_basis="non-cache")

    task = summary["tasks"]["task_b"]
    assert task["requests"] == 1
    assert task["cost_usd"] == 0.25
    assert task["duration_ms"] == 2500
    assert task["non_cache_tokens"] == 30
    assert task["pricing_model_id"] == "anthropic:claude-opus-4-latest"
    assert summary["totals"]["cost_usd"] == 0.25
    assert summary["totals"]["estimated_cost_usd"] == 0.00029375
    assert summary["budget"]["used_pct"] == 100.0


def test_collect_usage_can_override_price_model(tmp_path: Path):
    ensure_dispatch_dirs(tmp_path)
    state_path(tmp_path).write_text(json.dumps({
        "agent": "codex",
        "model": None,
        "queue": [{"name": "task_c", "status": "done"}],
    }), encoding="utf-8")
    _write_jsonl(task_out_path(tmp_path, "task_c"), [
        {
            "type": "result",
            "duration_ms": 1000,
            "usage": {
                "inputTokens": 1_000_000,
                "outputTokens": 100_000,
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
            },
        },
    ])

    summary = collect_usage(tmp_path, price_model="gpt-5.3-codex")

    task = summary["tasks"]["task_c"]
    assert task["pricing_model_id"] == "openai:gpt-5.3-codex"
    assert task["estimated_cost_usd"] == 3.15
