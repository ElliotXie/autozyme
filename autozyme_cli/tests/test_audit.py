"""In-process unit tests for zyme.audit.

Audit is append-only JSONL bookkeeping plus a Claude-Code transcript scraper.
Everything here is pure filesystem / string / JSON work — no subprocess, no
network, no real agent. The CC-scraping path is exercised by writing a
synthetic transcript and pointing env vars at it.

Covers:
  - _resolve_task_dir / _is_task_dir
  - _snapshot / _diff_snapshots
  - _short_exc
  - _StderrTee (tee, ring buffer, tail, isatty passthrough)
  - _truncate / _encode_cwd / _is_cc_session
  - _summarize_tool_call (Read/Write/Edit/Bash/Grep/Agent/WebFetch/WebSearch/
    generic/skip)
  - _last_audit_ts / read_audit
  - _collect_cc_tools (window filter, cwd filter, sidechain)
  - _find_session_transcript (session-id glob + most-recent fallback)
  - AuditContext end-to-end (success / die() / exception, output deltas,
    stderr_tail, workspace-level skip, init-creates-task.yaml)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme import audit


# --------------------------------------------------------------------------
# task-dir resolution
# --------------------------------------------------------------------------

class TestResolveTaskDir:
    def test_explicit_task_dir(self, tmp_path):
        args = SimpleNamespace(task_dir=str(tmp_path))
        assert audit._resolve_task_dir(args) == tmp_path.resolve()

    def test_falls_back_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        args = SimpleNamespace(task_dir=None)
        assert audit._resolve_task_dir(args) == tmp_path.resolve()

    def test_no_attr_uses_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert audit._resolve_task_dir(SimpleNamespace()) == tmp_path.resolve()

    def test_is_task_dir(self, tmp_path):
        assert audit._is_task_dir(tmp_path) is False
        (tmp_path / "task.yaml").write_text("x: 1\n")
        assert audit._is_task_dir(tmp_path) is True


# --------------------------------------------------------------------------
# snapshot / diff
# --------------------------------------------------------------------------

class TestSnapshot:
    def test_missing_files_are_none(self, tmp_path):
        snap = audit._snapshot(tmp_path)
        assert all(v is None for v in snap.values())
        assert "results.tsv" in snap

    def test_existing_file_recorded(self, tmp_path):
        (tmp_path / "results.tsv").write_text("hi")
        snap = audit._snapshot(tmp_path)
        assert snap["results.tsv"] is not None
        mtime, size = snap["results.tsv"]
        assert size == 2

    def test_diff_created(self):
        before = {"results.tsv": None}
        after = {"results.tsv": (1.0, 10)}
        assert audit._diff_snapshots(before, after) == {"created": ["results.tsv"]}

    def test_diff_deleted(self):
        before = {"results.tsv": (1.0, 10)}
        after = {"results.tsv": None}
        assert audit._diff_snapshots(before, after) == {"deleted": ["results.tsv"]}

    def test_diff_modified(self):
        before = {"results.tsv": (1.0, 10)}
        after = {"results.tsv": (2.0, 12)}
        assert audit._diff_snapshots(before, after) == {"modified": ["results.tsv"]}

    def test_diff_unchanged_empty(self):
        snap = {"results.tsv": (1.0, 10)}
        assert audit._diff_snapshots(snap, snap) == {}

    def test_diff_multiple_buckets(self):
        before = {"results.tsv": None, "verify.tsv": (1.0, 5), "task.yaml": (1.0, 5)}
        after = {"results.tsv": (1.0, 3), "verify.tsv": None, "task.yaml": (2.0, 9)}
        out = audit._diff_snapshots(before, after)
        assert out == {
            "created": ["results.tsv"],
            "deleted": ["verify.tsv"],
            "modified": ["task.yaml"],
        }


# --------------------------------------------------------------------------
# _short_exc
# --------------------------------------------------------------------------

class TestShortExc:
    def test_class_and_message(self):
        assert audit._short_exc(ValueError("boom")) == "ValueError: boom"

    def test_empty_message_class_only(self):
        assert audit._short_exc(RuntimeError("")) == "RuntimeError"

    def test_first_line_only(self):
        assert audit._short_exc(ValueError("line1\nline2")) == "ValueError: line1"

    def test_truncates_long(self):
        out = audit._short_exc(ValueError("x" * 500))
        assert len(out) == 200
        assert out.endswith("...")


# --------------------------------------------------------------------------
# _StderrTee
# --------------------------------------------------------------------------

class TestStderrTee:
    def test_writes_pass_through(self):
        import io
        buf = io.StringIO()
        tee = audit._StderrTee(buf, max_lines=5)
        tee.write("hello\n")
        assert buf.getvalue() == "hello\n"

    def test_returns_byte_count(self):
        import io
        tee = audit._StderrTee(io.StringIO())
        assert tee.write("abc") == 3

    def test_tail_ring_buffer(self):
        import io
        tee = audit._StderrTee(io.StringIO(), max_lines=2)
        tee.write("a\nb\nc\n")
        assert tee.tail() == ["b", "c"]

    def test_tail_includes_partial(self):
        import io
        tee = audit._StderrTee(io.StringIO(), max_lines=5)
        tee.write("line1\npartial")
        assert tee.tail() == ["line1", "partial"]

    def test_non_str_coerced(self):
        import io
        buf = io.StringIO()
        tee = audit._StderrTee(buf)
        tee.write(123)
        assert buf.getvalue() == "123"

    def test_isatty_passthrough(self):
        import io
        tee = audit._StderrTee(io.StringIO())
        # StringIO.isatty() returns False
        assert tee.isatty() is False

    def test_flush_delegates(self):
        import io
        buf = io.StringIO()
        tee = audit._StderrTee(buf)
        tee.flush()  # must not raise

    def test_getattr_delegates_to_original(self):
        class Fake:
            custom_attr = "delegated"
            def write(self, s):
                return len(s)
            def flush(self):
                pass
        tee = audit._StderrTee(Fake())
        assert tee.custom_attr == "delegated"

    def test_fileno_delegates(self):
        # real sys.__stderr__ exposes a fileno; StringIO does not, so use the
        # actual stderr stream to exercise the passthrough.
        tee = audit._StderrTee(sys.__stderr__)
        assert isinstance(tee.fileno(), int)


# --------------------------------------------------------------------------
# misc string helpers
# --------------------------------------------------------------------------

class TestStringHelpers:
    def test_truncate_short(self):
        assert audit._truncate("abc", 10) == "abc"

    def test_truncate_long(self):
        assert audit._truncate("abcdefghij", 5) == "ab..."

    def test_encode_cwd(self):
        assert audit._encode_cwd(Path("/Users/me/my_project")) == \
            "-Users-me-my-project"

    def test_encode_cwd_nested_underscores(self):
        assert audit._encode_cwd(Path("/a/sub_field/test_x")) == \
            "-a-sub-field-test-x"


class TestIsCcSession:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("CLAUDECODE", raising=False)
        monkeypatch.delenv("ZYME_AUDIT_NO_CC", raising=False)
        assert audit._is_cc_session() is False

    def test_on_with_claudecode(self, monkeypatch):
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.delenv("ZYME_AUDIT_NO_CC", raising=False)
        assert audit._is_cc_session() is True

    def test_disabled_override(self, monkeypatch):
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("ZYME_AUDIT_NO_CC", "1")
        assert audit._is_cc_session() is False


# --------------------------------------------------------------------------
# _summarize_tool_call
# --------------------------------------------------------------------------

class TestSummarizeToolCall:
    def _tu(self, name, inp):
        return {"name": name, "input": inp}

    def test_skipped_tools(self):
        assert audit._summarize_tool_call(
            self._tu("TodoWrite", {}), "T", False) is None
        assert audit._summarize_tool_call(
            self._tu("ToolSearch", {}), "T", False) is None

    def test_read(self):
        out = audit._summarize_tool_call(
            self._tu("Read", {"file_path": "/a/b.py", "offset": 10, "limit": 5}),
            "TS", False)
        assert out == {"ts": "TS", "tool": "Read", "path": "/a/b.py",
                       "offset": 10, "limit": 5}

    def test_write_bytes(self):
        out = audit._summarize_tool_call(
            self._tu("Write", {"file_path": "/x", "content": "hello"}), "TS", False)
        assert out["bytes"] == 5
        assert out["path"] == "/x"

    def test_edit_char_counts(self):
        out = audit._summarize_tool_call(
            self._tu("Edit", {"file_path": "/x", "old_string": "ab",
                              "new_string": "cdef", "replace_all": True}),
            "TS", False)
        assert out["old_chars"] == 2
        assert out["new_chars"] == 4
        assert out["replace_all"] is True

    def test_bash_first_line_and_desc(self):
        out = audit._summarize_tool_call(
            self._tu("Bash", {"command": "ls -la\nrm x", "description": "list"}),
            "TS", False)
        assert out["cmd"] == "ls -la"
        assert out["desc"] == "list"

    def test_grep(self):
        out = audit._summarize_tool_call(
            self._tu("Grep", {"pattern": "foo", "path": "/src"}), "TS", False)
        assert out["pattern"] == "foo"
        assert out["path"] == "/src"

    def test_agent(self):
        out = audit._summarize_tool_call(
            self._tu("Agent", {"subagent_type": "Explore", "description": "go"}),
            "TS", False)
        assert out["agent"] == "Explore"
        assert out["desc"] == "go"

    def test_agent_default_subagent(self):
        out = audit._summarize_tool_call(self._tu("Agent", {}), "TS", False)
        assert out["agent"] == "general-purpose"

    def test_webfetch_websearch(self):
        wf = audit._summarize_tool_call(
            self._tu("WebFetch", {"url": "http://x"}), "TS", False)
        assert wf["url"] == "http://x"
        ws = audit._summarize_tool_call(
            self._tu("WebSearch", {"query": "q"}), "TS", False)
        assert ws["query"] == "q"

    def test_generic_tool_input_json(self):
        out = audit._summarize_tool_call(
            self._tu("MysteryTool", {"a": 1}), "TS", False)
        assert out["tool"] == "MysteryTool"
        assert json.loads(out["input"]) == {"a": 1}

    def test_sidechain_marks_subagent(self):
        out = audit._summarize_tool_call(
            self._tu("Read", {"file_path": "/a"}), "TS", True)
        assert out["subagent"] is True

    def test_missing_name(self):
        out = audit._summarize_tool_call({}, "TS", False)
        assert out["tool"] == "?"

    def test_non_dict_input_tolerated(self):
        out = audit._summarize_tool_call(
            {"name": "Read", "input": "notadict"}, "TS", False)
        assert out["path"] == ""


# --------------------------------------------------------------------------
# _last_audit_ts / read_audit
# --------------------------------------------------------------------------

class TestAuditLog:
    def _write_rows(self, task_dir, rows):
        d = task_dir / ".zyme"
        d.mkdir(exist_ok=True)
        with (d / "audit.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    def test_last_audit_ts_none_when_absent(self, tmp_path):
        assert audit._last_audit_ts(tmp_path) is None

    def test_last_audit_ts_reads_last_row(self, tmp_path):
        self._write_rows(tmp_path, [{"ts": "A"}, {"ts": "B"}, {"ts": "C"}])
        assert audit._last_audit_ts(tmp_path) == "C"

    def test_last_audit_ts_empty_file(self, tmp_path):
        d = tmp_path / ".zyme"
        d.mkdir()
        (d / "audit.jsonl").write_text("")
        assert audit._last_audit_ts(tmp_path) is None

    def test_read_audit_absent(self, tmp_path):
        assert audit.read_audit(tmp_path) == []

    def test_read_audit_rows(self, tmp_path):
        self._write_rows(tmp_path, [{"ts": "A"}, {"ts": "B"}])
        rows = audit.read_audit(tmp_path)
        assert [r["ts"] for r in rows] == ["A", "B"]

    def test_read_audit_last_n(self, tmp_path):
        self._write_rows(tmp_path, [{"ts": "A"}, {"ts": "B"}, {"ts": "C"}])
        rows = audit.read_audit(tmp_path, last_n=2)
        assert [r["ts"] for r in rows] == ["B", "C"]

    def test_read_audit_skips_bad_lines(self, tmp_path):
        d = tmp_path / ".zyme"
        d.mkdir()
        (d / "audit.jsonl").write_text('{"ts":"A"}\nGARBAGE\n{"ts":"B"}\n')
        rows = audit.read_audit(tmp_path)
        assert [r["ts"] for r in rows] == ["A", "B"]


# --------------------------------------------------------------------------
# _collect_cc_tools
# --------------------------------------------------------------------------

class TestCollectCcTools:
    def _transcript(self, tmp_path, events):
        p = tmp_path / "session.jsonl"
        with p.open("w") as fh:
            for e in events:
                fh.write(json.dumps(e) + "\n")
        return p

    def _evt(self, ts, tool, cwd, sidechain=False):
        return {
            "timestamp": ts,
            "cwd": cwd,
            "isSidechain": sidechain,
            "message": {"content": [
                {"type": "tool_use", "name": tool, "input": {"file_path": "/x"}},
            ]},
        }

    def test_window_filter(self, tmp_path):
        td = str(tmp_path)
        tx = self._transcript(tmp_path, [
            self._evt("2026-01-01T00:00:00+00:00", "Read", td),   # before
            self._evt("2026-01-01T00:00:30+00:00", "Write", td),  # in
            self._evt("2026-01-01T00:01:30+00:00", "Edit", td),   # after
        ])
        out = audit._collect_cc_tools(
            tx, tmp_path,
            since_ts="2026-01-01T00:00:10+00:00",
            until_ts="2026-01-01T00:01:00+00:00")
        assert [t["tool"] for t in out] == ["Write"]

    def test_z_suffix_normalized(self, tmp_path):
        td = str(tmp_path)
        tx = self._transcript(tmp_path, [
            self._evt("2026-01-01T00:00:30Z", "Read", td),
        ])
        out = audit._collect_cc_tools(
            tx, tmp_path,
            since_ts="2026-01-01T00:00:00Z",
            until_ts="2026-01-01T00:01:00Z")
        assert len(out) == 1

    def test_cwd_filter_excludes_other_task(self, tmp_path):
        tx = self._transcript(tmp_path, [
            self._evt("2026-01-01T00:00:30+00:00", "Read", "/some/other/task"),
        ])
        out = audit._collect_cc_tools(
            tx, tmp_path, since_ts=None, until_ts="2026-01-01T01:00:00+00:00")
        assert out == []

    def test_no_since_takes_all_before_until(self, tmp_path):
        td = str(tmp_path)
        tx = self._transcript(tmp_path, [
            self._evt("2026-01-01T00:00:01+00:00", "Read", td),
            self._evt("2026-01-01T00:00:02+00:00", "Write", td),
        ])
        out = audit._collect_cc_tools(
            tx, tmp_path, since_ts=None, until_ts="2026-01-01T01:00:00+00:00")
        assert len(out) == 2

    def test_sidechain_flag(self, tmp_path):
        td = str(tmp_path)
        tx = self._transcript(tmp_path, [
            self._evt("2026-01-01T00:00:30+00:00", "Read", td, sidechain=True),
        ])
        out = audit._collect_cc_tools(
            tx, tmp_path, since_ts=None, until_ts="2026-01-01T01:00:00+00:00")
        assert out[0]["subagent"] is True

    def test_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "session.jsonl"
        p.write_text('GARBAGE\n{"no_timestamp": 1}\n')
        out = audit._collect_cc_tools(
            p, tmp_path, since_ts=None, until_ts="2026-01-01T01:00:00+00:00")
        assert out == []

    def test_missing_file_returns_empty(self, tmp_path):
        out = audit._collect_cc_tools(
            tmp_path / "nope.jsonl", tmp_path, since_ts=None,
            until_ts="2026-01-01T01:00:00+00:00")
        assert out == []

    def test_non_dict_message_skipped(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:30+00:00",
            "cwd": str(tmp_path),
            "message": "not-a-dict",
        }) + "\n")
        out = audit._collect_cc_tools(
            p, tmp_path, since_ts=None, until_ts="2026-01-01T01:00:00+00:00")
        assert out == []

    def test_content_not_a_list_skipped(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:30+00:00",
            "cwd": str(tmp_path),
            "message": {"content": "text-not-list"},
        }) + "\n")
        out = audit._collect_cc_tools(
            p, tmp_path, since_ts=None, until_ts="2026-01-01T01:00:00+00:00")
        assert out == []

    def test_non_tooluse_content_block_skipped(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:30+00:00",
            "cwd": str(tmp_path),
            "message": {"content": [
                {"type": "text", "text": "hi"},
                "not-a-dict",
            ]},
        }) + "\n")
        out = audit._collect_cc_tools(
            p, tmp_path, since_ts=None, until_ts="2026-01-01T01:00:00+00:00")
        assert out == []

    def test_skipped_tool_filtered_in_window(self, tmp_path):
        # TodoWrite is in-window but _summarize returns None -> not collected.
        p = tmp_path / "s.jsonl"
        p.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:30+00:00",
            "cwd": str(tmp_path),
            "message": {"content": [
                {"type": "tool_use", "name": "TodoWrite", "input": {}},
            ]},
        }) + "\n")
        out = audit._collect_cc_tools(
            p, tmp_path, since_ts=None, until_ts="2026-01-01T01:00:00+00:00")
        assert out == []


# --------------------------------------------------------------------------
# _find_session_transcript
# --------------------------------------------------------------------------

class TestFindSessionTranscript:
    def test_none_when_root_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(audit, "CC_TRANSCRIPT_ROOT", tmp_path / "nope")
        assert audit._find_session_transcript(tmp_path) is None

    def test_session_id_glob(self, tmp_path, monkeypatch):
        root = tmp_path / "projects"
        proj = root / "-Users-x"
        proj.mkdir(parents=True)
        (proj / "abc123.jsonl").write_text("{}")
        monkeypatch.setattr(audit, "CC_TRANSCRIPT_ROOT", root)
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "abc123")
        assert audit._find_session_transcript(tmp_path) == proj / "abc123.jsonl"

    def test_fallback_most_recent(self, tmp_path, monkeypatch):
        root = tmp_path / "projects"
        proj = root / "-p"
        proj.mkdir(parents=True)
        old = proj / "old.jsonl"
        new = proj / "new.jsonl"
        old.write_text("{}")
        new.write_text("{}")
        # make `new` strictly newer
        import os
        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))
        monkeypatch.setattr(audit, "CC_TRANSCRIPT_ROOT", root)
        monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
        assert audit._find_session_transcript(tmp_path) == new

    def test_no_candidates_returns_none(self, tmp_path, monkeypatch):
        root = tmp_path / "projects"
        (root / "-p").mkdir(parents=True)
        monkeypatch.setattr(audit, "CC_TRANSCRIPT_ROOT", root)
        monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
        assert audit._find_session_transcript(tmp_path) is None


# --------------------------------------------------------------------------
# AuditContext end-to-end
# --------------------------------------------------------------------------

class TestAuditContext:
    def _make_task(self, tmp_path):
        (tmp_path / "task.yaml").write_text("target_function: foo\n")
        return tmp_path

    def _last_record(self, task_dir):
        rows = audit.read_audit(task_dir)
        return rows[-1] if rows else None

    def test_success_writes_row(self, tmp_path, monkeypatch):
        td = self._make_task(tmp_path)
        monkeypatch.delenv("CLAUDECODE", raising=False)
        args = SimpleNamespace(task_dir=str(td))
        with audit.AuditContext(args, ["zyme", "status"], "status"):
            pass
        rec = self._last_record(td)
        assert rec["cmd"] == "status"
        assert rec["exit_code"] == 0
        assert rec["argv"] == ["zyme", "status"]
        assert "error" not in rec
        # stderr restored
        assert sys.stderr is not None

    def test_records_output_delta(self, tmp_path, monkeypatch):
        td = self._make_task(tmp_path)
        monkeypatch.delenv("CLAUDECODE", raising=False)
        args = SimpleNamespace(task_dir=str(td))
        with audit.AuditContext(args, ["zyme", "run"], "run"):
            (td / "results.tsv").write_text("new file")
        rec = self._last_record(td)
        assert rec["outputs"]["created"] == ["results.tsv"]

    def test_systemexit_records_exit_code(self, tmp_path, monkeypatch):
        td = self._make_task(tmp_path)
        monkeypatch.delenv("CLAUDECODE", raising=False)
        args = SimpleNamespace(task_dir=str(td))
        with pytest.raises(SystemExit):
            with audit.AuditContext(args, ["zyme", "run"], "run"):
                sys.stderr.write("fatal: something broke\n")
                raise SystemExit(2)
        rec = self._last_record(td)
        assert rec["exit_code"] == 2
        assert "error" not in rec  # die() carries no payload
        assert "fatal: something broke" in rec["stderr_tail"]

    def test_uncaught_exception_records_minus_one(self, tmp_path, monkeypatch):
        td = self._make_task(tmp_path)
        monkeypatch.delenv("CLAUDECODE", raising=False)
        args = SimpleNamespace(task_dir=str(td))
        with pytest.raises(ValueError):
            with audit.AuditContext(args, ["zyme", "run"], "run"):
                raise ValueError("kaboom")
        rec = self._last_record(td)
        assert rec["exit_code"] == -1
        assert rec["error"] == "ValueError: kaboom"

    def test_workspace_level_skips(self, tmp_path, monkeypatch):
        # no task.yaml -> not a task dir -> no audit row written
        monkeypatch.delenv("CLAUDECODE", raising=False)
        args = SimpleNamespace(task_dir=str(tmp_path))
        with audit.AuditContext(args, ["zyme", "scan"], "scan"):
            pass
        assert not (tmp_path / ".zyme" / "audit.jsonl").exists()

    def test_init_creates_task_yaml_midrun(self, tmp_path, monkeypatch):
        # Entered as non-task; task.yaml appears during the command (init).
        monkeypatch.delenv("CLAUDECODE", raising=False)
        args = SimpleNamespace(task_dir=str(tmp_path))
        with audit.AuditContext(args, ["zyme", "init"], "init"):
            (tmp_path / "task.yaml").write_text("target_function: foo\n")
        rec = self._last_record(tmp_path)
        assert rec is not None
        assert rec["cmd"] == "init"

    def test_systemexit_none_code_records_one(self, tmp_path, monkeypatch):
        # SystemExit() has code=None; the handler only int()s int codes,
        # else records 1. (code=0 from a clean die() would record 0; a bare
        # raise SystemExit() is treated as a non-clean exit.)
        td = self._make_task(tmp_path)
        monkeypatch.delenv("CLAUDECODE", raising=False)
        args = SimpleNamespace(task_dir=str(td))
        with pytest.raises(SystemExit):
            with audit.AuditContext(args, ["zyme", "run"], "run"):
                raise SystemExit()  # code=None
        rec = self._last_record(td)
        assert rec["exit_code"] == 1

    def test_systemexit_zero_code_records_zero(self, tmp_path, monkeypatch):
        td = self._make_task(tmp_path)
        monkeypatch.delenv("CLAUDECODE", raising=False)
        args = SimpleNamespace(task_dir=str(td))
        with pytest.raises(SystemExit):
            with audit.AuditContext(args, ["zyme", "run"], "run"):
                raise SystemExit(0)
        rec = self._last_record(td)
        assert rec["exit_code"] == 0

    def test_cc_session_embeds_cc_tools(self, tmp_path, monkeypatch):
        # End-to-end the CLAUDECODE branch: a transcript with a tool_use event
        # inside the audit window should land in the row's cc_tools.
        td = self._make_task(tmp_path)
        # Synthetic CC transcript discoverable via session-id glob.
        root = tmp_path / "cc_projects"
        proj = root / "-proj"
        proj.mkdir(parents=True)
        tx = proj / "sess42.jsonl"
        # Event timestamp must fall AFTER the audit row's start_iso. We can't
        # know start_iso ahead of time, so make the event "now-ish" relative
        # to a since_ts of None by writing it with a far-future timestamp and
        # a since lower bound from a prior row in the past.
        # Window is (since_ts, until_ts) where until = this row's start_iso
        # (captured at __init__, ~now). The event must fall strictly between
        # the prior row (since) and now (until).
        prior_ts = "2020-01-01T00:00:00+00:00"
        evt_ts = "2021-01-01T00:00:00+00:00"  # after prior, before "now"
        tx.write_text(json.dumps({
            "timestamp": evt_ts,
            "cwd": str(td),
            "message": {"content": [
                {"type": "tool_use", "name": "Read",
                 "input": {"file_path": "/some/file.py"}},
            ]},
        }) + "\n")
        # Seed a prior audit row so last_audit_ts = prior_ts (window lower bound).
        (td / ".zyme").mkdir(exist_ok=True)
        (td / ".zyme" / "audit.jsonl").write_text(
            json.dumps({"ts": prior_ts, "cmd": "init"}) + "\n")

        monkeypatch.setattr(audit, "CC_TRANSCRIPT_ROOT", root)
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.delenv("ZYME_AUDIT_NO_CC", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "sess42")

        args = SimpleNamespace(task_dir=str(td))
        with audit.AuditContext(args, ["zyme", "status"], "status"):
            pass
        rec = self._last_record(td)
        assert rec["cc_session"] == "sess42"
        assert any(t["tool"] == "Read" for t in rec["cc_tools"])

    def test_exit_swallows_audit_write_failure(self, tmp_path, monkeypatch, capsys):
        # If _write raises, __exit__ must not mask the real error; it emits a
        # one-line breadcrumb and returns False.
        td = self._make_task(tmp_path)
        monkeypatch.delenv("CLAUDECODE", raising=False)

        def boom(*a, **k):
            raise RuntimeError("disk full")
        monkeypatch.setattr(audit.AuditContext, "_write", boom)

        args = SimpleNamespace(task_dir=str(td))
        ctx = audit.AuditContext(args, ["zyme", "status"], "status")
        with ctx:
            pass
        captured = capsys.readouterr()
        assert "failed to record this invocation" in captured.err
