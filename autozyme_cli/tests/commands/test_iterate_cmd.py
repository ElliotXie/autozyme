"""Unit tests for zyme.commands.iterate — the single-task auto-resume babysitter.

The agent-launch loop forks claude via subprocess; that's not exercised here.
We cover the pure helpers (prompt-header stripping, round snapshot wrapper,
status printing, signal-handler install) and the cmd_iterate early-exit gates
(prompt-not-found, already-at-max-rounds) that return before any subprocess.
"""
from __future__ import annotations

import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands import iterate as it


# --------------------------------------------------------------------------
# _strip_prompt_header
# --------------------------------------------------------------------------

class TestStripPromptHeader:
    def test_strips_title_and_role(self):
        text = ("# Iterate Prompt\n"
                "## Role\n"
                "You are an optimizer.\n"
                "Do the work.\n")
        out = it._strip_prompt_header(text)
        assert out.startswith("You are an optimizer.")
        assert "# Iterate Prompt" not in out
        assert "## Role" not in out

    def test_strips_leading_blanks(self):
        text = "\n\n# Title\n\nBody line\n"
        out = it._strip_prompt_header(text)
        assert out.startswith("Body line")

    def test_no_header_unchanged(self):
        text = "Just body content.\nMore.\n"
        out = it._strip_prompt_header(text)
        assert out == text.rstrip("\n") or out.startswith("Just body content.")

    def test_only_header(self):
        text = "# Title\n## Role\n"
        out = it._strip_prompt_header(text)
        assert out == ""


# --------------------------------------------------------------------------
# _count_rounds (wraps _results_round_snapshot)
# --------------------------------------------------------------------------

class TestCountRounds:
    def test_no_results(self, tmp_path):
        snap = it._count_rounds(tmp_path)
        assert snap["completed_rounds"] == 0
        assert snap["last_round"] is None

    def test_counts_keep_rows(self, tmp_path):
        (tmp_path / "results.tsv").write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\t"
            "status\tmetrics_json\thypothesis\tdescription\tphase\n"
            "0\tabc\tds\t10\t0\t100\tbaseline\t{}\tu\t\toptimize\n"
            "1\tdef\tds\t8\t20\t100\tkeep\t{}\th\t\toptimize\n"
            "2\tghi\tds\t9\t10\t100\tdiscard\t{}\th\t\toptimize\n")
        snap = it._count_rounds(tmp_path)
        assert snap["completed_rounds"] >= 1
        assert snap["last_status"] in ("keep", "discard")


# --------------------------------------------------------------------------
# _print_status
# --------------------------------------------------------------------------

class TestPrintStatus:
    def test_renders_line(self, capsys):
        snap = {"completed_rounds": 3, "last_round": "3", "last_status": "keep"}
        it._print_status("done", snap, 50)
        err = capsys.readouterr().err
        assert "done: 3/50 rounds" in err
        assert "last=3:keep" in err


# --------------------------------------------------------------------------
# _install_signal_handlers
# --------------------------------------------------------------------------

class TestInstallSignalHandlers:
    def test_installs_and_restores(self):
        prev_term = signal.getsignal(signal.SIGTERM)
        prev_int = signal.getsignal(signal.SIGINT)
        try:
            it._install_signal_handlers()
            # handlers now point at the module's _handler closure
            assert signal.getsignal(signal.SIGTERM) is not prev_term
            assert callable(signal.getsignal(signal.SIGTERM))
        finally:
            signal.signal(signal.SIGTERM, prev_term)
            signal.signal(signal.SIGINT, prev_int)

    def test_handler_sets_should_stop(self):
        prev_term = signal.getsignal(signal.SIGTERM)
        prev_int = signal.getsignal(signal.SIGINT)
        prev_flag = it._SHOULD_STOP
        try:
            it._SHOULD_STOP = False
            it._install_signal_handlers()
            handler = signal.getsignal(signal.SIGTERM)
            handler(signal.SIGTERM, None)  # invoke directly
            assert it._SHOULD_STOP is True
        finally:
            it._SHOULD_STOP = prev_flag
            signal.signal(signal.SIGTERM, prev_term)
            signal.signal(signal.SIGINT, prev_int)


# --------------------------------------------------------------------------
# cmd_iterate early-exit gates
# --------------------------------------------------------------------------

class TestCmdIterateGates:
    def _task(self, tmp_path, *, prompt=True, rounds=0):
        (tmp_path / "task.yaml").write_text("target_function: foo\n")
        if prompt:
            (tmp_path / "prompts").mkdir()
            (tmp_path / "prompts" / "2_iterate.md").write_text(
                "# Iterate\n## Role\nbody\n")
        if rounds:
            lines = ["round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\t"
                     "status\tmetrics_json\thypothesis\tdescription\tphase"]
            for r in range(1, rounds + 1):
                lines.append(f"{r}\tc\tds\t8\t20\t100\tkeep\t{{}}\th\t\toptimize")
            (tmp_path / "results.tsv").write_text("\n".join(lines) + "\n")
        return tmp_path

    def _args(self, td, **kw):
        base = dict(task_dir=str(td), prompt="prompts/2_iterate.md",
                    max_rounds=50, no_progress_limit=3, stall_threshold=900,
                    model=None, effort=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_prompt_not_found_exits(self, tmp_path):
        td = self._task(tmp_path, prompt=False)
        with pytest.raises(SystemExit):
            it.cmd_iterate(self._args(td))

    def test_already_at_max_rounds_returns(self, tmp_path, capsys, monkeypatch):
        td = self._task(tmp_path, rounds=5)
        # Force completed_rounds >= max via a stubbed snapshot.
        monkeypatch.setattr(it, "_count_rounds", lambda d: {
            "completed_rounds": 50, "last_round": "50", "last_status": "keep"})
        # Ensure the subprocess loop is never reached: if it tries, fail loudly.
        monkeypatch.setattr(it.subprocess, "Popen",
                            lambda *a, **k: pytest.fail("should not launch"))
        it.cmd_iterate(self._args(td, max_rounds=50))
        err = capsys.readouterr().err
        assert "already at 50/50 rounds" in err


class TestCmdIterateLoop:
    """Drive one launch iteration with the subprocess boundary fully stubbed."""

    def _task(self, tmp_path):
        (tmp_path / "task.yaml").write_text("target_function: foo\n")
        (tmp_path / "prompts").mkdir()
        (tmp_path / "prompts" / "2_iterate.md").write_text(
            "# Iterate\n## Role\nbody content\n")
        return tmp_path

    def _args(self, td, **kw):
        base = dict(task_dir=str(td), prompt="prompts/2_iterate.md",
                    max_rounds=1, no_progress_limit=3, stall_threshold=900,
                    model="m", effort="high")
        base.update(kw)
        return SimpleNamespace(**base)

    def test_single_launch_reaches_max_rounds(self, tmp_path, monkeypatch, capsys):
        td = self._task(tmp_path)

        class FakeProc:
            def __init__(self):
                self.stdout = iter([])
                self.returncode = 0
            def wait(self):
                return 0
        monkeypatch.setattr(it.subprocess, "Popen", lambda *a, **k: FakeProc())
        monkeypatch.setattr(it, "_build_agent_cmd",
                            lambda *a, **k: ["claude", "-p"])
        monkeypatch.setattr(it, "_terminate", lambda proc: None)

        # Stream: a session-id event line, then a zyme_run line, then exit.
        def fake_stream(proc, stall):
            yield ("line", '{"session": "sess1"}')
            yield ("line", '{"run": 1}')
            yield ("exit", None)
        monkeypatch.setattr(it, "_read_stream_with_stalls", fake_stream)

        # parse_agent_line returns our synthetic events for the two lines.
        def fake_parse(payload, agent="claude"):
            if "session" in payload:
                return [{"kind": "agent_session", "session_id": "sess1"}]
            return [{"kind": "zyme_run", "hypothesis": "try X"}]
        monkeypatch.setattr(it, "parse_agent_line", fake_parse)

        # Round snapshot: starts at 0, then after the first event hits max=1.
        calls = {"n": 0}

        def fake_count(d):
            calls["n"] += 1
            # First call (pre-loop) is 0; once inside, report 1 (>= max).
            done = 0 if calls["n"] == 1 else 1
            return {"completed_rounds": done, "last_round": str(done),
                    "last_status": "keep"}
        monkeypatch.setattr(it, "_count_rounds", fake_count)
        # Reset the module stop flag.
        monkeypatch.setattr(it, "_SHOULD_STOP", False)

        it.cmd_iterate(self._args(td, max_rounds=1))
        err = capsys.readouterr().err
        assert "launching claude" in err
        assert "max rounds reached" in err or "done" in err
