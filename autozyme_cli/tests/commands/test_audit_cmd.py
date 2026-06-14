"""Unit tests for zyme.commands.audit — the `zyme audit` log-reader COMMAND.

This is the command module (commands/audit.py: _render_cc_tool + cmd_audit),
NOT zyme.audit (the append-only ledger library covered by tests/test_audit.py).
No overlap.

Covered: _render_cc_tool across every tool kind + subagent suffix, and cmd_audit
in both table + --json modes over a real .zyme/audit.jsonl, plus the empty-log
message.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands import audit as auditcmd


# --------------------------------------------------------------------------
# _render_cc_tool
# --------------------------------------------------------------------------

class TestRenderCcTool:
    def test_read_with_range(self):
        out = auditcmd._render_cc_tool(
            {"tool": "Read", "path": "/a/b.py", "offset": 10, "limit": 5})
        assert out == "Read /a/b.py [10:+5]"

    def test_read_no_range(self):
        out = auditcmd._render_cc_tool({"tool": "Read", "path": "/a/b.py"})
        assert out == "Read /a/b.py"

    def test_edit_with_replace_all(self):
        out = auditcmd._render_cc_tool({
            "tool": "Edit", "path": "/x", "replace_all": True,
            "old_chars": 3, "new_chars": 7})
        assert "Edit all /x" in out
        assert "(-3/+7 chars)" in out

    def test_edit_without_replace_all(self):
        out = auditcmd._render_cc_tool({
            "tool": "Edit", "path": "/x", "old_chars": 1, "new_chars": 2})
        assert out.startswith("Edit /x")

    def test_write(self):
        out = auditcmd._render_cc_tool({"tool": "Write", "path": "/x", "bytes": 42})
        assert out == "Write /x (42 bytes)"

    def test_bash(self):
        out = auditcmd._render_cc_tool({"tool": "Bash", "cmd": "ls -la"})
        assert out == "Bash $ ls -la"

    def test_grep_with_path(self):
        out = auditcmd._render_cc_tool({"tool": "Grep", "pattern": "foo", "path": "/src"})
        assert out == "Grep foo in /src"

    def test_glob_no_path(self):
        out = auditcmd._render_cc_tool({"tool": "Glob", "pattern": "*.py"})
        assert out == "Glob *.py"

    def test_agent(self):
        out = auditcmd._render_cc_tool({"tool": "Agent", "agent": "Explore", "desc": "go"})
        assert out == "Agent[Explore] go"

    def test_webfetch(self):
        out = auditcmd._render_cc_tool({"tool": "WebFetch", "url": "http://x"})
        assert out == "WebFetch http://x"

    def test_websearch(self):
        out = auditcmd._render_cc_tool({"tool": "WebSearch", "query": "q"})
        assert out == "WebSearch q"

    def test_generic_tool(self):
        out = auditcmd._render_cc_tool({"tool": "Mystery", "input": "blob"})
        assert out == "Mystery blob"

    def test_missing_tool_name(self):
        out = auditcmd._render_cc_tool({})
        assert out.startswith("? ")

    def test_subagent_suffix(self):
        out = auditcmd._render_cc_tool(
            {"tool": "Read", "path": "/a", "subagent": True})
        assert out.endswith(" (subagent)")


# --------------------------------------------------------------------------
# cmd_audit
# --------------------------------------------------------------------------

class TestCmdAudit:
    def _task_with_log(self, tmp_path, rows):
        (tmp_path / "task.yaml").write_text("target_function: foo\n")
        z = tmp_path / ".zyme"
        z.mkdir()
        with (z / "audit.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return tmp_path

    def _args(self, td, **kw):
        base = dict(task_dir=str(td), last=None, json=False)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_empty_log_message(self, tmp_path, capsys):
        (tmp_path / "task.yaml").write_text("target_function: foo\n")
        (tmp_path / ".zyme").mkdir()
        auditcmd.cmd_audit(self._args(tmp_path))
        out = capsys.readouterr().out
        assert "no audit entries" in out

    def test_json_mode(self, tmp_path, capsys):
        rows = [{"ts": "2026-01-01T00:00:00Z", "cmd": "run",
                 "duration_s": 1.5, "exit_code": 0, "argv": ["zyme", "run"]}]
        td = self._task_with_log(tmp_path, rows)
        auditcmd.cmd_audit(self._args(td, json=True))
        out = capsys.readouterr().out
        assert '"cmd": "run"' in out

    def test_table_mode_full(self, tmp_path, capsys):
        rows = [{
            "ts": "2026-01-01T12:00:00Z", "cmd": "run", "duration_s": 2.5,
            "exit_code": 0,
            "argv": ["zyme", "run", "--tier", "tiny"],
            "outputs": {"created": ["results.tsv"], "modified": ["task.yaml"]},
            "error": None,
            "cc_tools": [
                {"tool": "Read", "path": "/a.py"},
                {"tool": "Bash", "cmd": "ls"},
            ],
        }]
        td = self._task_with_log(tmp_path, rows)
        auditcmd.cmd_audit(self._args(td))
        out = capsys.readouterr().out
        assert "run" in out
        assert "created=results.tsv" in out
        assert "modified=task.yaml" in out
        assert "cc: Read /a.py" in out
        assert "1 entry" in out

    def test_table_truncates_long_argv(self, tmp_path, capsys):
        long_arg = "x" * 80
        rows = [{"ts": "2026-01-01T00:00:00Z", "cmd": "run", "duration_s": 1.0,
                 "exit_code": 0, "argv": ["zyme", "run", long_arg]}]
        td = self._task_with_log(tmp_path, rows)
        auditcmd.cmd_audit(self._args(td))
        out = capsys.readouterr().out
        assert "..." in out

    def test_table_with_error(self, tmp_path, capsys):
        rows = [{"ts": "bad-ts", "cmd": "run", "duration_s": 1.0,
                 "exit_code": -1, "argv": ["zyme", "run"],
                 "error": "ValueError: boom"}]
        td = self._task_with_log(tmp_path, rows)
        auditcmd.cmd_audit(self._args(td))
        out = capsys.readouterr().out
        assert "! ValueError: boom" in out

    def test_last_n_limits(self, tmp_path, capsys):
        rows = [{"ts": f"2026-01-0{i}T00:00:00Z", "cmd": "run",
                 "duration_s": 1.0, "exit_code": 0, "argv": ["zyme", "run"]}
                for i in range(1, 4)]
        td = self._task_with_log(tmp_path, rows)
        auditcmd.cmd_audit(self._args(td, last=2))
        out = capsys.readouterr().out
        assert "2 entries" in out

    def test_plural_entries_label(self, tmp_path, capsys):
        rows = [{"ts": "2026-01-01T00:00:00Z", "cmd": "run", "duration_s": 1.0,
                 "exit_code": 0, "argv": ["zyme", "run"]} for _ in range(3)]
        td = self._task_with_log(tmp_path, rows)
        auditcmd.cmd_audit(self._args(td))
        out = capsys.readouterr().out
        assert "3 entries" in out
