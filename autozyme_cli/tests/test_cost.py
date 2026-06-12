"""Tests for zyme.cost — per-task time/token/cost accounting (multi-agent).

Covers pure helpers (cmd_to_phase, parse_ts, active_minutes, collect_session
windowing, codex cumulative-delta) and the compute_task_cost integration over
synthetic audit.jsonl + synthetic Claude transcripts + synthetic Codex rollouts
(CC_TRANSCRIPT_ROOT / CODEX_SESSIONS_ROOT monkeypatched to tmp).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from zyme import cost as costmod
from zyme.cost import (
    active_minutes,
    append_agent_usage,
    cmd_to_phase,
    collect_session,
    compute_task_cost,
    parse_stream_usage,
    parse_ts,
    _codex_windowed_usage,
)


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolate_agent_roots(tmp_path, monkeypatch):
    """Point both agent-log roots at empty dirs so tests never read the real
    ~/.claude or ~/.codex. Individual tests override as needed."""
    monkeypatch.setattr(costmod, "CC_TRANSCRIPT_ROOT", tmp_path / "_no_claude")
    monkeypatch.setattr(costmod, "CODEX_SESSIONS_ROOT", tmp_path / "_no_codex")


# --------------------------------------------------------------------------
# cmd_to_phase / time helpers
# --------------------------------------------------------------------------
class TestCmdToPhase:
    def test_known_and_multiword(self):
        assert cmd_to_phase("init") == "init"
        assert cmd_to_phase("run") == "iterate"
        assert cmd_to_phase("baseline reference") == "iterate"
        assert cmd_to_phase("attest") == "validate"
        assert cmd_to_phase("publish-speedups") == "package"

    def test_unknown_is_meta(self):
        assert cmd_to_phase("status") == "meta"
        assert cmd_to_phase("") == "meta"
        assert cmd_to_phase(None) == "meta"


class TestTimeHelpers:
    def test_parse_ts(self):
        assert parse_ts("2026-05-10T07:28:01+00:00") is not None
        assert parse_ts("2026-05-10T07:28:01Z") is not None
        assert parse_ts("") is None
        assert parse_ts("nope") is None

    def test_active_minutes_excludes_long_gaps(self):
        base = _dt("2026-05-10T00:00:00")
        evs = [
            base,
            _dt("2026-05-10T00:02:00"),
            _dt("2026-05-10T00:05:00"),
            _dt("2026-05-10T01:05:00"),  # +60m idle -> excluded
            _dt("2026-05-10T01:06:00"),
        ]
        active, raw = active_minutes(evs, gap_min=10)
        assert active == pytest.approx(6.0)
        assert raw == pytest.approx(66.0)

    def test_active_minutes_degenerate(self):
        assert active_minutes([], 10) == (0.0, 0.0)
        assert active_minutes([_dt("2026-05-10T00:00:00")], 10) == (0.0, 0.0)


# --------------------------------------------------------------------------
# collect_session (Claude) — sum, dedup, window
# --------------------------------------------------------------------------
def _write_cc_transcript(path: Path, msgs: list[dict]) -> None:
    lines = []
    for m in msgs:
        lines.append(json.dumps({
            "type": "assistant",
            "timestamp": m["ts"],
            "requestId": m.get("rid", ""),
            "message": {
                "id": m.get("mid", ""),
                "model": m.get("model", "claude-opus-4-8"),
                "usage": {
                    "input_tokens": m.get("in", 0),
                    "output_tokens": m.get("out", 0),
                    "cache_read_input_tokens": m.get("cr", 0),
                    "cache_creation_input_tokens": m.get("cw", 0),
                },
            },
        }))
    path.write_text("\n".join(lines) + "\n")


class TestCollectSession:
    def test_sums_and_dedups(self, tmp_path: Path):
        p = tmp_path / "s.jsonl"
        _write_cc_transcript(p, [
            {"ts": "2026-05-10T00:00:00Z", "in": 10, "out": 20, "cr": 100, "cw": 5, "rid": "a", "mid": "1"},
            {"ts": "2026-05-10T00:01:00Z", "in": 10, "out": 20, "cr": 100, "cw": 5, "rid": "a", "mid": "1"},  # dup
            {"ts": "2026-05-10T00:02:00Z", "in": 1, "out": 2, "cr": 3, "cw": 4, "rid": "b", "mid": "2"},
        ])
        d = collect_session(p)
        assert (d["input_tokens"], d["output_tokens"], d["cache_read_tokens"], d["cache_write_tokens"]) == (11, 22, 103, 9)
        assert d["total_tokens"] == 11 + 22 + 103 + 9
        assert d["model_counts"]["claude-opus-4-8"] == 2

    def test_window_filters(self, tmp_path: Path):
        p = tmp_path / "s.jsonl"
        _write_cc_transcript(p, [
            {"ts": "2026-05-10T00:00:00Z", "out": 100, "rid": "a", "mid": "1"},  # before lo
            {"ts": "2026-05-10T01:00:00Z", "out": 200, "rid": "b", "mid": "2"},  # in window
            {"ts": "2026-05-10T05:00:00Z", "out": 400, "rid": "c", "mid": "3"},  # after hi
        ])
        d = collect_session(p, lo=_dt("2026-05-10T00:30:00"), hi=_dt("2026-05-10T02:00:00"))
        assert d["output_tokens"] == 200


# --------------------------------------------------------------------------
# codex cumulative-delta windowing
# --------------------------------------------------------------------------
class TestCodexWindow:
    def _ev(self, ts, inp, cached, out):
        return (_dt(ts), {"input_tokens": inp, "cached_input_tokens": cached, "output_tokens": out})

    def test_full_session_takes_final_cumulative(self):
        evs = [
            self._ev("2026-05-10T00:00:00", 100, 50, 10),
            self._ev("2026-05-10T00:10:00", 500, 300, 40),   # final cumulative
        ]
        u = _codex_windowed_usage(evs, None, None)
        # non-cached input = 500-300, cache_read = 300, output = 40
        assert u == {"input_tokens": 200, "cache_read_tokens": 300,
                     "output_tokens": 40, "cache_write_tokens": 0}

    def test_window_is_delta_between_bounds(self):
        evs = [
            self._ev("2026-05-10T00:00:00", 100, 0, 10),   # before window
            self._ev("2026-05-10T01:00:00", 300, 0, 30),   # in window
            self._ev("2026-05-10T02:00:00", 800, 0, 80),   # in window (latest <= hi)
            self._ev("2026-05-10T09:00:00", 999, 0, 99),   # after window
        ]
        u = _codex_windowed_usage(evs, lo=_dt("2026-05-10T00:30:00"), hi=_dt("2026-05-10T02:30:00"))
        # delta = cumulative@2:00 (800/80) - cumulative just before 0:30 (100/10)
        assert u["input_tokens"] == 700
        assert u["output_tokens"] == 70


# --------------------------------------------------------------------------
# compute_task_cost — integration
# --------------------------------------------------------------------------
def _write_audit(task_dir: Path, rows: list[dict]) -> None:
    (task_dir / ".zyme" / "audit.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")


def _write_codex_rollout(root: Path, cwd: str, model: str,
                         token_events: list[tuple[str, dict]]) -> None:
    """token_events: [(ts, total_token_usage dict)] cumulative."""
    d = root / "2026" / "05" / "10"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"rollout-2026-05-10T00-00-00-{abs(hash(cwd + model + str(token_events))) % 10**9}.jsonl"
    lines = [
        json.dumps({"timestamp": "2026-05-10T00:00:00.000Z", "type": "session_meta",
                    "payload": {"cwd": cwd}}),
        json.dumps({"timestamp": "2026-05-10T00:00:01.000Z", "type": "turn_context",
                    "payload": {"cwd": cwd, "model": model}}),
    ]
    for ts, tu in token_events:
        lines.append(json.dumps({"timestamp": ts, "type": "event_msg",
                                 "payload": {"type": "token_count",
                                             "info": {"total_token_usage": tu}}}))
    f.write_text("\n".join(lines) + "\n")


class TestComputeTaskCost:
    def test_time_only_no_sessions(self, task_dir: Path):
        _write_audit(task_dir, [
            {"ts": "2026-05-10T00:00:00+00:00", "cmd": "init", "duration_s": 5.0},
            {"ts": "2026-05-10T00:10:00+00:00", "cmd": "run", "duration_s": 120.0},
            {"ts": "2026-05-10T00:30:00+00:00", "cmd": "attest", "duration_s": 60.0},
        ])
        rep = compute_task_cost(task_dir)
        t = rep["time"]
        assert t["n_invocations"] == 3
        assert t["calendar_span_min"] == pytest.approx(30.0)
        assert t["cli_wall_min"] == pytest.approx((5 + 120 + 60) / 60)
        assert t["by_phase"]["iterate"]["cli_wall_s"] == pytest.approx(120.0)
        assert t["by_phase"]["validate"]["n_calls"] == 1
        assert rep["agents"] == []
        assert rep["cost"] is None
        assert rep["tokens"]["total_tokens"] == 0
        assert any("Cursor" in w for w in rep["warnings"])

    def test_empty_audit(self, task_dir: Path):
        rep = compute_task_cost(task_dir)
        assert rep["time"]["n_invocations"] == 0
        assert rep["cost"] is None
        assert rep["agents"] == []

    def test_claude_scoped_tokens_and_cost(self, task_dir: Path, monkeypatch):
        proj_root = task_dir / "_cc"
        enc = costmod._encode_cwd(task_dir.resolve())
        proj = proj_root / enc
        proj.mkdir(parents=True)
        sid = "deadbeef-0000-0000-0000-000000000000"
        _write_cc_transcript(proj / f"{sid}.jsonl", [
            {"ts": "2026-05-10T00:05:00Z", "model": "claude-opus-4-8",
             "in": 0, "out": 1_000_000, "cr": 0, "cw": 0, "rid": "a", "mid": "1"},
        ])
        monkeypatch.setattr(costmod, "CC_TRANSCRIPT_ROOT", proj_root)
        _write_audit(task_dir, [
            {"ts": "2026-05-10T00:00:00+00:00", "cmd": "run", "duration_s": 1.0, "cc_session": sid},
        ])
        rep = compute_task_cost(task_dir)
        assert [a["agent"] for a in rep["agents"]] == ["claude"]
        cl = rep["agents"][0]
        assert cl["n_scoped"] == 1 and cl["n_shared"] == 0
        assert cl["output_tokens"] == 1_000_000
        # 1M output @ $25/Mtok (opus 4.x) = $25.00
        assert cl["cost"]["total_usd"] == pytest.approx(25.0)
        assert rep["cost"]["total_usd"] == pytest.approx(25.0)
        assert rep["cost"]["by_agent"] == {"claude": pytest.approx(25.0)}

    def test_claude_shared_is_windowed(self, task_dir: Path, monkeypatch):
        proj_root = task_dir / "_cc"
        other = proj_root / "-Users-someone-workspace"
        other.mkdir(parents=True)
        sid = "cafef00d-1111-1111-1111-111111111111"
        _write_cc_transcript(other / f"{sid}.jsonl", [
            {"ts": "2026-05-10T00:00:00Z", "out": 500, "rid": "x", "mid": "0"},   # before
            {"ts": "2026-05-10T02:00:00Z", "out": 700, "rid": "y", "mid": "1"},   # near audit ts
            {"ts": "2026-05-10T09:00:00Z", "out": 900, "rid": "z", "mid": "2"},   # after
        ])
        monkeypatch.setattr(costmod, "CC_TRANSCRIPT_ROOT", proj_root)
        _write_audit(task_dir, [
            {"ts": "2026-05-10T02:00:00+00:00", "cmd": "run", "duration_s": 1.0, "cc_session": sid},
        ])
        rep = compute_task_cost(task_dir, gap_min=10)
        cl = rep["agents"][0]
        assert cl["n_shared"] == 1
        assert cl["output_tokens"] == 700
        assert any("shared" in w for w in rep["warnings"])

    def test_codex_scoped(self, task_dir: Path, monkeypatch):
        root = task_dir / "_codex"
        monkeypatch.setattr(costmod, "CODEX_SESSIONS_ROOT", root)
        _write_codex_rollout(
            root, cwd=str(task_dir.resolve()), model="gpt-5.5",
            token_events=[
                ("2026-05-10T00:00:30Z", {"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 10}),
                ("2026-05-10T00:05:00Z", {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 0}),
            ],
        )
        _write_audit(task_dir, [
            {"ts": "2026-05-10T00:00:00+00:00", "cmd": "run", "duration_s": 1.0},
        ])
        rep = compute_task_cost(task_dir)
        assert [a["agent"] for a in rep["agents"]] == ["codex"]
        cx = rep["agents"][0]
        assert cx["n_scoped"] == 1
        assert cx["input_tokens"] == 1_000_000  # final cumulative, non-cached
        assert cx["dominant_model"] == "gpt-5.5"
        # 1M input @ $5/Mtok (gpt-5.5) = $5.00
        assert cx["cost"]["total_usd"] == pytest.approx(5.0)

    def test_captured_cursor(self, task_dir: Path):
        rec = parse_stream_usage([
            json.dumps({"type": "system", "model": "Composer 2", "session_id": "s1"}),
            json.dumps({"type": "result", "duration_ms": 1000, "session_id": "s1",
                        "usage": {"inputTokens": 100, "outputTokens": 200,
                                  "cacheReadTokens": 300, "cacheWriteTokens": 0}}),
        ], agent="cursor")
        assert rec["model"] == "Composer 2"
        assert rec["input_tokens"] == 100 and rec["output_tokens"] == 200
        assert rec["cache_read_tokens"] == 300 and rec["n_results"] == 1
        append_agent_usage(task_dir, rec)
        _write_audit(task_dir, [
            {"ts": "2026-06-06T00:00:00+00:00", "cmd": "run", "duration_s": 1.0},
        ])
        rep = compute_task_cost(task_dir)
        assert [a["agent"] for a in rep["agents"]] == ["cursor"]
        cx = rep["agents"][0]
        assert cx["source"] == "captured" and cx["n_records"] == 1
        assert cx["cost"]["model_price_id"] == "cursor:composer-2"
        assert rep["cost"]["total_usd"] > 0

    def test_parse_stream_codex_cumulative(self):
        # Codex token_count is cumulative — take the final one.
        rec = parse_stream_usage([
            json.dumps({"type": "session_meta", "payload": {"id": "cx1", "cwd": "/x"}}),
            json.dumps({"type": "turn_context", "payload": {"model": "gpt-5.5"}}),
            json.dumps({"type": "event_msg", "payload": {"type": "token_count",
                "info": {"total_token_usage": {"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 5}}}}),
            json.dumps({"type": "event_msg", "payload": {"type": "token_count",
                "info": {"total_token_usage": {"input_tokens": 900, "cached_input_tokens": 200, "output_tokens": 50}}}}),
        ], agent="codex")
        assert rec["model"] == "gpt-5.5" and rec["session_id"] == "cx1"
        # final cumulative: non-cached input 900-200, cache_read 200, output 50
        assert rec["input_tokens"] == 700
        assert rec["cache_read_tokens"] == 200
        assert rec["output_tokens"] == 50

    def test_captured_overrides_scraper(self, task_dir: Path, monkeypatch):
        # A codex rollout (cwd==task) AND a captured codex record both exist;
        # the captured record must win and the rollout must be skipped.
        root = task_dir / "_codex"
        monkeypatch.setattr(costmod, "CODEX_SESSIONS_ROOT", root)
        _write_codex_rollout(
            root, cwd=str(task_dir.resolve()), model="gpt-5.5",
            token_events=[("2026-05-10T00:05:00Z",
                           {"input_tokens": 50_000_000, "cached_input_tokens": 0, "output_tokens": 0})],
        )
        append_agent_usage(task_dir, parse_stream_usage([
            json.dumps({"type": "turn_context", "payload": {"model": "gpt-5.5"}}),
            json.dumps({"type": "event_msg", "payload": {"type": "token_count",
                "info": {"total_token_usage": {"input_tokens": 7, "cached_input_tokens": 0, "output_tokens": 0}}}}),
        ], agent="codex"))
        _write_audit(task_dir, [{"ts": "2026-05-10T00:00:00+00:00", "cmd": "run", "duration_s": 1.0}])
        rep = compute_task_cost(task_dir)
        assert [a["agent"] for a in rep["agents"]] == ["codex"]
        assert rep["agents"][0]["source"] == "captured"
        assert rep["agents"][0]["input_tokens"] == 7   # captured, NOT the 50M rollout

    def test_codex_ancestor_cwd_excluded_with_note(self, task_dir: Path, monkeypatch):
        root = task_dir / "_codex"
        monkeypatch.setattr(costmod, "CODEX_SESSIONS_ROOT", root)
        # cwd is the PARENT of task_dir -> ambiguous, must be excluded.
        _write_codex_rollout(
            root, cwd=str(task_dir.resolve().parent), model="gpt-5.5",
            token_events=[
                ("2026-05-10T00:01:00Z", {"input_tokens": 9_999_999, "cached_input_tokens": 0, "output_tokens": 0}),
            ],
        )
        _write_audit(task_dir, [
            {"ts": "2026-05-10T00:00:00+00:00", "cmd": "run", "duration_s": 1.0},
            {"ts": "2026-05-10T00:02:00+00:00", "cmd": "run", "duration_s": 1.0},
        ])
        rep = compute_task_cost(task_dir)
        assert rep["agents"] == []          # nothing attributable
        assert rep["tokens"]["total_tokens"] == 0
        assert any("ancestor dir" in w for w in rep["warnings"])
