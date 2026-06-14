"""Unit tests for zyme.commands.cost — the `zyme cost` COMMAND module.

This is the command wrapper + formatters, NOT zyme.cost (the accounting library,
covered by tests/test_cost.py and tests/test_cost_unit.py). No overlap.

Covered: the pure formatters (_fmt_min / _fmt_int / _fmt_usd), the per-agent
block renderer (_print_agent) across source labels + captured vs session shapes,
and the cmd_cost / cmd_cost_capture entrypoints driven over a stubbed
compute_task_cost / parse_stream_usage / append_agent_usage boundary.
"""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands import cost as costcmd


# --------------------------------------------------------------------------
# formatters
# --------------------------------------------------------------------------

class TestFmtMin:
    @pytest.mark.parametrize("m,expect", [
        (0.0, "0.0m"),
        (3.25, "3.2m"),
        (59.9, "59.9m"),
        (60.0, "1h00m"),
        (104.0, "1h44m"),
        (125.0, "2h05m"),
    ])
    def test_values(self, m, expect):
        assert costcmd._fmt_min(m) == expect


class TestFmtInt:
    def test_thousands_separator(self):
        assert costcmd._fmt_int(1234567) == "1,234,567"

    def test_zero(self):
        assert costcmd._fmt_int(0) == "0"


class TestFmtUsd:
    def test_none(self):
        assert costcmd._fmt_usd(None) == "n/a"

    def test_value(self):
        assert costcmd._fmt_usd(1234.5) == "$1,234.50"

    def test_small(self):
        assert costcmd._fmt_usd(0.07) == "$0.07"


# --------------------------------------------------------------------------
# _print_agent
# --------------------------------------------------------------------------

class TestPrintAgent:
    def _agent(self, **kw):
        base = {
            "agent": "claude", "source": "claude_transcript",
            "n_sessions": 2, "n_scoped": 1, "n_shared": 1,
            "dominant_model": "claude-opus", "input_tokens": 1000,
            "output_tokens": 500, "cache_read_tokens": 200,
            "cache_write_tokens": 100, "total_tokens": 1800,
            "cost": {"total_usd": 1.23, "model_price_id": "opus-price"},
        }
        base.update(kw)
        return base

    def test_session_source(self, capsys):
        costcmd._print_agent(self._agent())
        out = capsys.readouterr().out
        assert "[claude]" in out
        assert "transcripts" in out  # source label
        assert "2 session(s)" in out
        assert "1,800" in out  # total tokens formatted
        assert "$1.23" in out

    def test_captured_source(self, capsys):
        a = self._agent(source="captured", n_records=4)
        costcmd._print_agent(a)
        out = capsys.readouterr().out
        assert "captured" in out
        assert "4 captured run(s)" in out

    def test_no_cost(self, capsys):
        a = self._agent(cost=None)
        costcmd._print_agent(a)
        out = capsys.readouterr().out
        assert "cost         : n/a" in out

    def test_codex_rollout_label(self, capsys):
        a = self._agent(agent="codex", source="codex_rollout")
        costcmd._print_agent(a)
        assert "rollouts" in capsys.readouterr().out

    def test_unknown_source_passthrough(self, capsys):
        a = self._agent(source="mystery")
        costcmd._print_agent(a)
        assert "mystery" in capsys.readouterr().out


# --------------------------------------------------------------------------
# cmd_cost
# --------------------------------------------------------------------------

def _report(**kw):
    base = {
        "task": "test_x",
        "time": {
            "n_invocations": 3, "calendar_span_min": 120.0, "first_ts": "T0",
            "last_ts": "T1", "active_min": 45.0, "cli_wall_min": 10.0,
            "by_phase": {"init": {"n_calls": 1, "cli_wall_s": 60},
                         "iterate": {"n_calls": 2, "cli_wall_s": 120}},
        },
        "agents": [],
        "cost": None,
        "warnings": [],
    }
    base.update(kw)
    return base


class TestCmdCost:
    def _args(self, td, **kw):
        base = dict(task_dir=str(td), gap_min=5.0, model=None, json=False)
        base.update(kw)
        return SimpleNamespace(**base)

    def _task(self, tmp_path):
        (tmp_path / "task.yaml").write_text("target_function: foo\n")
        return tmp_path

    def test_json_output(self, tmp_path, monkeypatch, capsys):
        td = self._task(tmp_path)
        monkeypatch.setattr(costcmd, "compute_task_cost",
                            lambda td, gap_min, model_override: _report())
        costcmd.cmd_cost(self._args(td, json=True))
        out = capsys.readouterr().out
        assert '"task": "test_x"' in out

    def test_empty_report_message(self, tmp_path, monkeypatch, capsys):
        td = self._task(tmp_path)
        empty = _report(time={**_report()["time"], "n_invocations": 0},
                        agents=[])
        monkeypatch.setattr(costcmd, "compute_task_cost",
                            lambda td, gap_min, model_override: empty)
        costcmd.cmd_cost(self._args(td))
        out = capsys.readouterr().out
        assert "nothing to account" in out

    def test_time_and_phase_breakdown(self, tmp_path, monkeypatch, capsys):
        td = self._task(tmp_path)
        monkeypatch.setattr(costcmd, "compute_task_cost",
                            lambda td, gap_min, model_override: _report())
        costcmd.cmd_cost(self._args(td))
        out = capsys.readouterr().out
        assert "TIME" in out
        assert "calendar span" in out
        assert "init" in out and "iterate" in out
        assert "no Claude/Codex sessions matched" in out

    def test_agents_and_cost_block(self, tmp_path, monkeypatch, capsys):
        td = self._task(tmp_path)
        agent = {
            "agent": "claude", "source": "captured", "n_records": 2,
            "n_sessions": 0, "dominant_model": "opus", "input_tokens": 1,
            "output_tokens": 1, "cache_read_tokens": 0, "cache_write_tokens": 0,
            "total_tokens": 2, "cost": {"total_usd": 0.5, "model_price_id": "p"},
        }
        rep = _report(agents=[agent], cost={
            "total_usd": 0.5, "by_agent": {"claude": 0.5}})
        monkeypatch.setattr(costcmd, "compute_task_cost",
                            lambda td, gap_min, model_override: rep)
        costcmd.cmd_cost(self._args(td))
        out = capsys.readouterr().out
        assert "TOKENS" in out
        assert "COST (estimated)" in out
        assert "TOTAL" in out

    def test_multi_agent_cost_breakdown_and_warnings(self, tmp_path, monkeypatch, capsys):
        td = self._task(tmp_path)
        def agent(name):
            return {"agent": name, "source": "captured", "n_records": 1,
                    "n_sessions": 0, "dominant_model": "m", "input_tokens": 1,
                    "output_tokens": 1, "cache_read_tokens": 0,
                    "cache_write_tokens": 0, "total_tokens": 2,
                    "cost": {"total_usd": 0.1, "model_price_id": "p"}}
        rep = _report(
            agents=[agent("claude"), agent("codex")],
            cost={"total_usd": 0.2,
                  "by_agent": {"claude": 0.1, "codex": 0.1}},
            warnings=["cursor tokens unavailable"])
        monkeypatch.setattr(costcmd, "compute_task_cost",
                            lambda td, gap_min, model_override: rep)
        costcmd.cmd_cost(self._args(td))
        out = capsys.readouterr().out
        assert "claude  :" in out
        assert "codex   :" in out
        assert "NOTES" in out
        assert "cursor tokens unavailable" in out


# --------------------------------------------------------------------------
# cmd_cost_capture
# --------------------------------------------------------------------------

class TestCmdCostCapture:
    def _task(self, tmp_path):
        (tmp_path / "task.yaml").write_text("target_function: foo\n")
        return tmp_path

    def _record(self, **kw):
        base = {"input_tokens": 100, "output_tokens": 50,
                "cache_read_tokens": 0, "cache_write_tokens": 0,
                "n_results": 1, "model": "claude-opus"}
        base.update(kw)
        return base

    def test_from_file(self, tmp_path, monkeypatch, capsys):
        td = self._task(tmp_path)
        stream = tmp_path / "run.jsonl"
        stream.write_text('{"usage": {}}\n')
        monkeypatch.setattr(costcmd, "parse_stream_usage",
                            lambda lines, agent: self._record())
        monkeypatch.setattr(costcmd, "append_agent_usage",
                            lambda td, rec: td / ".zyme" / "agent_usage.jsonl")
        args = SimpleNamespace(task_dir=str(td), agent="claude",
                               from_file=str(stream), quiet=True)
        costcmd.cmd_cost_capture(args)
        out = capsys.readouterr().out
        assert "cost-capture[claude]" in out
        assert "150 tokens" in out

    def test_zero_record_message(self, tmp_path, monkeypatch, capsys):
        td = self._task(tmp_path)
        stream = tmp_path / "empty.jsonl"
        stream.write_text("\n")
        zero = self._record(input_tokens=0, output_tokens=0, n_results=0)
        monkeypatch.setattr(costcmd, "parse_stream_usage",
                            lambda lines, agent: zero)
        monkeypatch.setattr(costcmd, "append_agent_usage",
                            lambda td, rec: Path("/tmp/x"))
        args = SimpleNamespace(task_dir=str(td), agent="cursor",
                               from_file=str(stream), quiet=True)
        costcmd.cmd_cost_capture(args)
        out = capsys.readouterr().out
        assert "no usage found" in out

    def test_stdin_stream_with_passthrough(self, tmp_path, monkeypatch, capsys):
        td = self._task(tmp_path)
        monkeypatch.setattr("sys.stdin", io.StringIO("line1\nline2\n"))
        seen = {}

        def fake_parse(gen, agent):
            # consume the generator to trigger passthrough
            lines = list(gen)
            seen["lines"] = lines
            return self._record()
        monkeypatch.setattr(costcmd, "parse_stream_usage", fake_parse)
        monkeypatch.setattr(costcmd, "append_agent_usage",
                            lambda td, rec: Path("/tmp/x"))
        args = SimpleNamespace(task_dir=str(td), agent="codex",
                               from_file=None, quiet=False)
        costcmd.cmd_cost_capture(args)
        out = capsys.readouterr().out
        # passthrough echoed stdin to stdout
        assert "line1" in out
        assert seen["lines"] == ["line1\n", "line2\n"]
