"""Unit tests for zyme.commands.validate — the adversarial-audit driver.

The real command spawns an agent subprocess; that path is not exercised here.
We cover everything pure: precondition gates, prompt/report path resolution,
the markdown-report -> findings parser, file:line splitting, one-line flattening,
TSV escaping + writing, assistant-text extraction across backend schemas, the
local mirror copy, and the user-facing summary.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from zyme.commands import validate as v


# --------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------

def _make_init_ready(task_dir: Path):
    (task_dir / "task.yaml").write_text("target_function: foo\n")
    (task_dir / "reference_outputs").mkdir()
    (task_dir / "reference_outputs" / "tiny").mkdir()
    (task_dir / "reference_outputs" / "tiny" / "out.pkl").write_text("x")
    (task_dir / "evaluate.py").write_text("print('ok')")


class TestPreconditions:
    def test_no_task_yaml(self, tmp_path):
        with pytest.raises(SystemExit):
            v._check_preconditions("init", tmp_path)

    def test_no_reference_outputs(self, tmp_path):
        (tmp_path / "task.yaml").write_text("x: 1\n")
        (tmp_path / "evaluate.py").write_text("p")
        with pytest.raises(SystemExit):
            v._check_preconditions("init", tmp_path)

    def test_no_evaluate(self, tmp_path):
        (tmp_path / "task.yaml").write_text("x: 1\n")
        (tmp_path / "reference_outputs").mkdir()
        (tmp_path / "reference_outputs" / "t").mkdir()
        (tmp_path / "reference_outputs" / "t" / "o").write_text("x")
        with pytest.raises(SystemExit):
            v._check_preconditions("init", tmp_path)

    def test_init_passes(self, tmp_path):
        _make_init_ready(tmp_path)
        v._check_preconditions("init", tmp_path)  # must not raise

    def test_flat_reference_layout_ok(self, tmp_path):
        (tmp_path / "task.yaml").write_text("x: 1\n")
        (tmp_path / "evaluate.R").write_text("p")
        (tmp_path / "reference_output_tiny").mkdir()
        v._check_preconditions("init", tmp_path)

    def test_pipeline_evaluate_ok(self, tmp_path):
        (tmp_path / "task.yaml").write_text("x: 1\n")
        (tmp_path / "reference_outputs").mkdir()
        (tmp_path / "reference_outputs" / "t").mkdir()
        (tmp_path / "reference_outputs" / "t" / "o").write_text("x")
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "evaluate.py").write_text("p")
        v._check_preconditions("init", tmp_path)

    def test_iterate_needs_results(self, tmp_path):
        _make_init_ready(tmp_path)
        with pytest.raises(SystemExit):
            v._check_preconditions("iterate", tmp_path)

    def test_iterate_needs_data_row(self, tmp_path):
        _make_init_ready(tmp_path)
        (tmp_path / "results.tsv").write_text("round\tstatus\n")  # header only
        with pytest.raises(SystemExit):
            v._check_preconditions("iterate", tmp_path)

    def test_iterate_needs_best_ref(self, tmp_path):
        _make_init_ready(tmp_path)
        (tmp_path / "results.tsv").write_text("round\tstatus\n1\tkeep\n")
        with pytest.raises(SystemExit):
            v._check_preconditions("iterate", tmp_path)

    def test_iterate_passes(self, tmp_path):
        _make_init_ready(tmp_path)
        (tmp_path / "results.tsv").write_text("round\tstatus\n1\tkeep\n")
        (tmp_path / ".zyme").mkdir()
        (tmp_path / ".zyme" / "best.ref").write_text("abc")
        v._check_preconditions("iterate", tmp_path)


class TestResultsHasDataRow:
    def test_header_only_false(self, tmp_path):
        p = tmp_path / "r.tsv"
        p.write_text("round\tstatus\n")
        assert v._results_has_data_row(p) is False

    def test_with_data_true(self, tmp_path):
        p = tmp_path / "r.tsv"
        p.write_text("round\tstatus\n1\tkeep\n")
        assert v._results_has_data_row(p) is True

    def test_blank_rows_only_false(self, tmp_path):
        p = tmp_path / "r.tsv"
        p.write_text("round\tstatus\n\n   \n")
        assert v._results_has_data_row(p) is False

    def test_missing_file_false(self, tmp_path):
        assert v._results_has_data_row(tmp_path / "nope.tsv") is False


# --------------------------------------------------------------------------
# prompt + report path resolution
# --------------------------------------------------------------------------

class TestResolvePromptPath:
    def test_init_prompt_exists(self):
        # The real validate_init.md ships in the package — resolve it.
        p = v._resolve_prompt_path("init")
        assert p.name == "validate_init.md"
        assert p.is_file()

    def test_iterate_prompt_exists(self):
        p = v._resolve_prompt_path("iterate")
        assert p.name == "validate_iterate.md"
        assert p.is_file()


class TestResolveReportPath:
    def test_out_override(self, tmp_path):
        out = tmp_path / "custom.md"
        got = v._resolve_report_path("init", tmp_path, str(out), None)
        assert got == out.resolve()

    def test_default_v1_when_no_existing(self, tmp_path):
        got = v._resolve_report_path("init", tmp_path, None, None)
        assert got.parent.name == "v1"
        assert got.name.startswith("validate_init_")
        assert got.name.endswith(".md")

    def test_explicit_version(self, tmp_path):
        got = v._resolve_report_path("init", tmp_path, None, "3")
        assert got.parent.name == "v3"

    def test_v_prefixed_version(self, tmp_path):
        got = v._resolve_report_path("iterate", tmp_path, None, "v2")
        assert got.parent.name == "v2"

    def test_bad_version_dies(self, tmp_path):
        with pytest.raises(SystemExit):
            v._resolve_report_path("init", tmp_path, None, "notanum")


# --------------------------------------------------------------------------
# findings parser
# --------------------------------------------------------------------------

_SAMPLE_REPORT = """\
# Validator report

### Finding 1 — LIKELY_HACK — Cat C - timer hoist
**File:line evidence**: `pipeline/run.py:218-265`
**Mechanism**: The timer is hoisted above the loop so only one
iteration is measured.

### Finding 2 — FAIL — Cat A wrong output
**File evidence**: pipeline/run.R
**Mechanism**: output shape mismatch.

### Finding 3 — WEAK — speculative
no fields here
"""


class TestParseFindings:
    def test_three_findings(self):
        fs = v._parse_findings(_SAMPLE_REPORT)
        assert len(fs) == 3
        assert fs[0]["finding_id"] == 1
        assert fs[0]["severity"] == "LIKELY_HACK"
        assert fs[0]["category"] == "Cat C - timer hoist"
        assert fs[0]["file_ref"] == "pipeline/run.py"
        assert fs[0]["line_ref"] == "218-265"
        assert "timer is hoisted" in fs[0]["mechanism_summary"]

    def test_severity_uppercased(self):
        fs = v._parse_findings(_SAMPLE_REPORT)
        assert fs[1]["severity"] == "FAIL"
        assert fs[1]["file_ref"] == "pipeline/run.R"
        assert fs[1]["line_ref"] == ""

    def test_finding_without_fields(self):
        fs = v._parse_findings(_SAMPLE_REPORT)
        assert fs[2]["file_ref"] == ""
        assert fs[2]["mechanism_summary"] == ""

    def test_empty_report(self):
        assert v._parse_findings("no findings here") == []

    def test_lowercase_severity_is_kept_and_normalized(self):
        # B18 fix: the header regex accepts case-insensitive severities and
        # _parse_findings normalizes them to upper-case, so a finding whose
        # severity is written lowercase (e.g. "likely_hack") is no longer
        # silently dropped -- it is kept with severity normalized to uppercase
        # (so "fail" and "FAIL" behave identically downstream).
        report = (
            "### Finding 1 — likely_hack — lowercase cat\n"
            "**Mechanism**: kept.\n"
        )
        findings = v._parse_findings(report)
        assert len(findings) == 1
        assert findings[0]["severity"] == "LIKELY_HACK"
        assert findings[0]["category"] == "lowercase cat"


class TestSplitFileLine:
    def test_with_line_range(self):
        assert v._split_file_line("pipeline/run.py:218-265") == ("pipeline/run.py", "218-265")

    def test_single_line(self):
        assert v._split_file_line("a.py:42") == ("a.py", "42")

    def test_no_colon(self):
        assert v._split_file_line("a.py") == ("a.py", "")

    def test_colon_but_non_numeric(self):
        # e.g. a URL or namespace colon -> not split
        assert v._split_file_line("pkg::fn") == ("pkg::fn", "")

    def test_empty(self):
        assert v._split_file_line("") == ("", "")

    def test_strips_backticks(self):
        assert v._split_file_line("`a.py:5`") == ("a.py", "5")


class TestFlattenOneLine:
    def test_collapses_whitespace(self):
        assert v._flatten_one_line("a\n  b\t c") == "a b c"

    def test_truncates(self):
        out = v._flatten_one_line("x" * 500, max_chars=10)
        assert len(out) == 10
        assert out.endswith("…")

    def test_empty(self):
        assert v._flatten_one_line("") == ""


class TestMatchOrEmpty:
    def test_hit(self):
        pat = re.compile(r"foo=(\d+)")
        assert v._match_or_empty(pat, "foo=42") == "42"

    def test_miss(self):
        pat = re.compile(r"foo=(\d+)")
        assert v._match_or_empty(pat, "nothing") == ""


# --------------------------------------------------------------------------
# TSV writer + escaping
# --------------------------------------------------------------------------

class TestTsvSafe:
    def test_replaces_separators(self):
        assert v._tsv_safe("a\tb\nc\rd") == "a b c d"

    def test_none_to_empty(self):
        assert v._tsv_safe(None) == ""


class TestSafeRelative:
    def test_relative(self, tmp_path):
        base = tmp_path
        p = tmp_path / "sub" / "f.md"
        p.parent.mkdir()
        p.write_text("x")
        assert v._safe_relative(p, base) == str(Path("sub") / "f.md")

    def test_not_under_base_absolute(self, tmp_path):
        p = tmp_path / "f.md"
        p.write_text("x")
        other = tmp_path.parent / "elsewhere"
        out = v._safe_relative(p, other)
        assert out == str(p.resolve())


class TestAppendTsv:
    def test_new_file_writes_header(self, tmp_path):
        tsv = tmp_path / "validate.tsv"
        report = tmp_path / "report.md"
        report.write_text("x")
        findings = [{
            "finding_id": 1, "severity": "FAIL", "category": "Cat A",
            "file_ref": "run.py", "line_ref": "5",
            "mechanism_summary": "broke\tit",
        }]
        v._append_tsv(tsv_path=tsv, phase="init", agent="claude",
                      model=None, findings=findings, report_path=report)
        lines = tsv.read_text().splitlines()
        assert lines[0].split("\t") == v._TSV_HEADER
        assert lines[1].split("\t")[1] == "init"
        assert lines[1].split("\t")[3] == "(default)"  # model default
        # tab in mechanism replaced
        assert "broke it" in lines[1]

    def test_appends_without_dup_header(self, tmp_path):
        tsv = tmp_path / "validate.tsv"
        report = tmp_path / "report.md"
        report.write_text("x")
        f = [{"finding_id": 1, "severity": "WEAK", "category": "c",
              "file_ref": "", "line_ref": "", "mechanism_summary": ""}]
        v._append_tsv(tsv_path=tsv, phase="init", agent="codex",
                      model="gpt", findings=f, report_path=report)
        v._append_tsv(tsv_path=tsv, phase="iterate", agent="codex",
                      model="gpt", findings=f, report_path=report)
        lines = tsv.read_text().splitlines()
        assert lines.count("\t".join(v._TSV_HEADER)) == 1
        assert len(lines) == 3  # header + 2 rows

    def test_no_findings_writes_only_header(self, tmp_path):
        tsv = tmp_path / "validate.tsv"
        report = tmp_path / "report.md"
        report.write_text("x")
        v._append_tsv(tsv_path=tsv, phase="init", agent="claude",
                      model=None, findings=[], report_path=report)
        lines = tsv.read_text().splitlines()
        assert len(lines) == 1


# --------------------------------------------------------------------------
# assistant-text extraction (multi-backend stream-json)
# --------------------------------------------------------------------------

class TestExtractAssistantText:
    def test_blank(self):
        assert v._extract_assistant_text("   \n", "claude") == ""

    def test_non_json(self):
        assert v._extract_assistant_text("not json", "claude") == ""

    def test_non_dict_json(self):
        assert v._extract_assistant_text("[1,2,3]", "claude") == ""

    def test_claude_assistant_text(self):
        line = '{"type":"assistant","message":{"content":[{"type":"text","text":"hello"}]}}'
        assert v._extract_assistant_text(line, "claude") == "hello"

    def test_claude_multiple_blocks(self):
        line = ('{"type":"assistant","message":{"content":'
                '[{"type":"text","text":"a"},{"type":"text","text":"b"}]}}')
        assert v._extract_assistant_text(line, "claude") == "a\nb"

    def test_codex_msg_wrapper_string(self):
        line = '{"msg":{"type":"agent_message","message":"codex says hi"}}'
        assert v._extract_assistant_text(line, "codex") == "codex says hi"

    def test_codex_item_wrapper_text(self):
        line = '{"item":{"type":"agent_message","text":"item text"}}'
        assert v._extract_assistant_text(line, "codex") == "item text"

    def test_codex_content_list(self):
        line = ('{"msg":{"type":"assistant_message","content":'
                '[{"text":"x"},"y"]}}')
        assert v._extract_assistant_text(line, "codex") == "x\ny"

    def test_no_assistant_event(self):
        line = '{"type":"tool_use","name":"Write"}'
        assert v._extract_assistant_text(line, "claude") == ""


# --------------------------------------------------------------------------
# local mirror
# --------------------------------------------------------------------------

class TestMirrorReportLocally:
    def test_copies_into_zyme_validate(self, tmp_path):
        fw = tmp_path / "fw" / "validation" / "test_x" / "v2"
        fw.mkdir(parents=True)
        report = fw / "validate_init_TS.md"
        report.write_text("report body")
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        dest = v._mirror_report_locally(report, task_dir)
        assert dest.is_file()
        assert dest.read_text() == "report body"
        # version dir preserved
        assert dest.parent.name == "v2"
        assert ".zyme" in dest.parts

    def test_no_version_dir(self, tmp_path):
        report = tmp_path / "somewhere" / "validate_init_TS.md"
        report.parent.mkdir()
        report.write_text("body")
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        dest = v._mirror_report_locally(report, task_dir)
        assert dest.parent.name == "validate"


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

class TestPrintSummary:
    def test_counts_by_severity(self, tmp_path, capsys):
        findings = [
            {"severity": "FAIL"}, {"severity": "FAIL"},
            {"severity": "WEAK"}, {"severity": "MYSTERY"},
        ]
        v._print_summary("init", tmp_path / "test_t", "claude", findings,
                         tmp_path / "r.md", local_mirror=tmp_path / "m.md")
        out = capsys.readouterr().out
        assert "findings: 4" in out
        assert "FAIL: 2" in out
        assert "WEAK: 1" in out
        assert "off-schema" in out  # MYSTERY
        assert "mirror:" in out

    def test_no_mirror(self, tmp_path, capsys):
        v._print_summary("iterate", tmp_path / "t", "codex", [],
                         tmp_path / "r.md", local_mirror=None)
        out = capsys.readouterr().out
        assert "findings: 0" in out
        assert "mirror:" not in out


# --------------------------------------------------------------------------
# _run_validate end-to-end with the agent-invocation boundary stubbed
# --------------------------------------------------------------------------

class TestRunValidateEndToEnd:
    def test_full_init_run(self, tmp_path, monkeypatch, capsys):
        _make_init_ready(tmp_path)

        # Stub the agent invocation: it must "write" a report to report_path.
        report_body = (
            "# report\n\n"
            "### Finding 1 — LIKELY_HACK — Cat C\n"
            "**File:line evidence**: `pipeline/run.py:10-20`\n"
            "**Mechanism**: hoisted.\n"
        )

        def fake_invoke(*, prompt_path, report_path, task_dir, agent, model, effort):
            report_path.write_text(report_body)
        monkeypatch.setattr(v, "_invoke_validator", fake_invoke)

        from types import SimpleNamespace
        args = SimpleNamespace(
            task_dir=str(tmp_path),
            out=str(tmp_path / "report.md"),
            version=None,
            agent="claude",
            model=None,
            effort="high",
        )
        rc = v._run_validate("init", args)
        assert rc == 0
        # validate.tsv was written with the parsed finding
        tsv = (tmp_path / "validate.tsv").read_text()
        assert "LIKELY_HACK" in tsv
        # mirror exists
        assert (tmp_path / ".zyme" / "validate").exists()
        out = capsys.readouterr().out
        assert "complete" in out

    def test_run_dies_if_report_missing(self, tmp_path, monkeypatch):
        _make_init_ready(tmp_path)

        def fake_invoke(**kw):
            pass  # never writes report
        monkeypatch.setattr(v, "_invoke_validator", fake_invoke)

        from types import SimpleNamespace
        args = SimpleNamespace(
            task_dir=str(tmp_path), out=str(tmp_path / "r.md"),
            version=None, agent="claude", model=None, effort="high")
        with pytest.raises(SystemExit):
            v._run_validate("init", args)

    def test_auto_resolves_agent(self, tmp_path, monkeypatch, capsys):
        _make_init_ready(tmp_path)

        def fake_invoke(*, prompt_path, report_path, **kw):
            report_path.write_text("# empty report, no findings\n")
        monkeypatch.setattr(v, "_invoke_validator", fake_invoke)
        # agent="auto" triggers detect_agent_binary in the dispatch module.
        import zyme.dispatch.master as master
        monkeypatch.setattr(master, "detect_agent_binary",
                            lambda: ("codex", "/bin/codex"))

        from types import SimpleNamespace
        args = SimpleNamespace(
            task_dir=str(tmp_path), out=str(tmp_path / "r.md"),
            version=None, agent="auto", model=None, effort="high")
        rc = v._run_validate("init", args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "launching codex" in out


# --------------------------------------------------------------------------
# cmd_validate_init / iterate thin wrappers
# --------------------------------------------------------------------------

class TestCmdWrappers:
    def test_cmd_validate_iterate_delegates(self, monkeypatch):
        seen = {}

        def fake_run(phase, args):
            seen["phase"] = phase
            return 0
        monkeypatch.setattr(v, "_run_validate", fake_run)
        from types import SimpleNamespace
        assert v.cmd_validate_iterate(SimpleNamespace()) == 0
        assert seen["phase"] == "iterate"

    def test_cmd_validate_init_no_register(self, monkeypatch):
        monkeypatch.setattr(v, "_run_validate", lambda phase, args: 0)
        from types import SimpleNamespace
        # register_template absent -> no auto-register
        assert v.cmd_validate_init(SimpleNamespace(register_template=None)) == 0

    def test_cmd_validate_init_with_register(self, monkeypatch):
        monkeypatch.setattr(v, "_run_validate", lambda phase, args: 0)
        called = {}
        monkeypatch.setattr(v, "_auto_register_iterate_template",
                            lambda args: called.setdefault("registered", True))
        from types import SimpleNamespace
        rc = v.cmd_validate_init(SimpleNamespace(register_template="tpl"))
        assert rc == 0
        assert called.get("registered") is True

    def test_cmd_validate_init_rc_nonzero_skips_register(self, monkeypatch):
        monkeypatch.setattr(v, "_run_validate", lambda phase, args: 1)
        called = {}
        monkeypatch.setattr(v, "_auto_register_iterate_template",
                            lambda args: called.setdefault("registered", True))
        from types import SimpleNamespace
        rc = v.cmd_validate_init(SimpleNamespace(register_template="tpl"))
        assert rc == 1
        assert "registered" not in called


class TestAutoRegisterIterateTemplate:
    def test_calls_bench_register(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "task.yaml").write_text("target_function: foo\n")
        import zyme.commands.bench as bench
        seen = {}

        def fake_register(reg_args):
            seen["name"] = reg_args.name
            seen["stage"] = reg_args.stage
        monkeypatch.setattr(bench, "cmd_bench_register_template", fake_register)

        from types import SimpleNamespace
        args = SimpleNamespace(task_dir=str(tmp_path),
                               register_template="mytpl", register_force=False)
        v._auto_register_iterate_template(args)
        assert seen["name"] == "mytpl"
        assert seen["stage"] == "iterate"

    def test_register_failure_is_nonfatal(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "task.yaml").write_text("target_function: foo\n")
        import zyme.commands.bench as bench

        def boom(reg_args):
            raise SystemExit(1)
        monkeypatch.setattr(bench, "cmd_bench_register_template", boom)

        from types import SimpleNamespace
        args = SimpleNamespace(task_dir=str(tmp_path),
                               register_template="mytpl", register_force=False)
        # Must NOT propagate the SystemExit — registration failure is a warning.
        v._auto_register_iterate_template(args)
        err = capsys.readouterr().err
        assert "warn" in err.lower()
