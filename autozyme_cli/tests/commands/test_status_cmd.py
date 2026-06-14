"""Unit tests for zyme.commands.status — the `zyme status` snapshot command.

Two layers:

  1. Pure helpers (no I/O beyond a tmp task dir): phase-cap lookup, audit
     verdict rollup, validate.tsv parsing, the audit print section.

  2. cmd_status end-to-end, driven against a crafted temp task dir. It is
     read-only and degrades gracefully when not a git repo (`git ... check=
     False` returns ""), so we can drive it without a real repo and assert on
     the rendered one-screen text via capsys. Where a specific HEAD/clean
     state matters we monkeypatch `status.git`.

All output assertions scope to the rendered text — no subprocess, no network.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import zyme.commands.status as status
from zyme.commands.status import (
    _AUDIT_SEV_ORDER,
    _PHASE_CAPS,
    _audit_verdict,
    _phase_cap,
    _print_audit_section,
    _read_validate_tsv,
    cmd_status,
)


def _args(task_dir: Path, *, last: int = 5, phase: str = "optimize") -> SimpleNamespace:
    return SimpleNamespace(task_dir=str(task_dir), last=last, phase=phase)


# --------------------------------------------------------------------------
# _phase_cap
# --------------------------------------------------------------------------

class TestPhaseCap:
    def test_known_phases(self):
        assert _phase_cap("optimize") == 100
        assert _phase_cap("validate") == 30
        assert _phase_cap("memory") == 50
        assert _phase_cap("all") == 100

    def test_unknown_phase_falls_back_to_100(self):
        assert _phase_cap("nonsense") == 100
        assert _phase_cap("") == 100

    def test_caps_table_matches(self):
        assert _PHASE_CAPS == {"optimize": 100, "validate": 30, "memory": 50, "all": 100}


# --------------------------------------------------------------------------
# _audit_verdict — highest-severity-first rollup
# --------------------------------------------------------------------------

class TestAuditVerdict:
    def test_empty_is_pass(self):
        assert _audit_verdict({}) == "PASS"
        assert _audit_verdict({"FAIL": 0, "WEAK": 0}) == "PASS"

    def test_first_severity_in_order_wins(self):
        # FAIL outranks everything.
        assert _audit_verdict({"WEAK": 3, "FAIL": 1}) == "FAIL"

    def test_likely_hack_when_no_fail(self):
        assert _audit_verdict({"LIKELY_HACK": 2, "WEAK": 1}) == "LIKELY_HACK"

    def test_weak_only(self):
        assert _audit_verdict({"WEAK": 5}) == "WEAK"

    def test_severity_order_constant(self):
        assert _AUDIT_SEV_ORDER == (
            "FAIL", "LIKELY_HACK", "LIKELY_HACK_INVITED", "WEAK",
        )


# --------------------------------------------------------------------------
# _read_validate_tsv
# --------------------------------------------------------------------------

_VALIDATE_HEADER = (
    "phase\ttimestamp\tvalidator_agent\tvalidator_model\tseverity\treport_path"
)


def _write_validate(task_dir: Path, *rows: str) -> None:
    (task_dir / "validate.tsv").write_text(
        _VALIDATE_HEADER + "\n" + "\n".join(rows) + ("\n" if rows else "")
    )


class TestReadValidateTsv:
    def test_missing_file_returns_empty(self, task_dir):
        assert _read_validate_tsv(task_dir) == {}

    def test_bad_header_returns_empty(self, task_dir):
        (task_dir / "validate.tsv").write_text("not\tthe\tright\theader\n")
        assert _read_validate_tsv(task_dir) == {}

    def test_single_invocation_tallies_severity(self, task_dir):
        _write_validate(
            task_dir,
            "iterate\t2026-01-01T00:00:00Z\tclaude\topus\tWEAK\trep.md",
            "iterate\t2026-01-01T00:00:00Z\tclaude\topus\tFAIL\trep.md",
        )
        out = _read_validate_tsv(task_dir)
        assert set(out) == {"iterate"}
        a = out["iterate"]
        assert a["sev_counts"] == {"WEAK": 1, "FAIL": 1}
        assert a["agent"] == "claude"
        assert a["model"] == "opus"
        assert a["report_path"] == "rep.md"

    def test_keeps_only_latest_invocation_per_phase(self, task_dir):
        # Two invocations of the same phase; only the newest timestamp's rows
        # should be tallied.
        _write_validate(
            task_dir,
            "iterate\t2026-01-01T00:00:00Z\tclaude\topus\tFAIL\told.md",
            "iterate\t2026-02-02T00:00:00Z\tclaude\topus\tWEAK\tnew.md",
        )
        out = _read_validate_tsv(task_dir)
        assert out["iterate"]["timestamp"] == "2026-02-02T00:00:00Z"
        assert out["iterate"]["sev_counts"] == {"WEAK": 1}
        assert out["iterate"]["report_path"] == "new.md"

    def test_two_phases_both_returned(self, task_dir):
        _write_validate(
            task_dir,
            "init\t2026-01-01T00:00:00Z\tclaude\topus\tPASS\ti.md",
            "iterate\t2026-01-02T00:00:00Z\tcursor\tcomposer\tWEAK\tit.md",
        )
        out = _read_validate_tsv(task_dir)
        assert set(out) == {"init", "iterate"}

    def test_short_rows_skipped(self, task_dir):
        # A row with fewer columns than the header is dropped.
        (task_dir / "validate.tsv").write_text(
            _VALIDATE_HEADER + "\n"
            "iterate\t2026-01-01T00:00:00Z\tclaude\topus\tWEAK\trep.md\n"
            "iterate\ttoo\tshort\n"
        )
        out = _read_validate_tsv(task_dir)
        assert out["iterate"]["sev_counts"] == {"WEAK": 1}


# --------------------------------------------------------------------------
# _print_audit_section
# --------------------------------------------------------------------------

class TestPrintAuditSection:
    def test_not_run_when_no_validate_tsv(self, task_dir, capsys):
        _print_audit_section(task_dir)
        out = capsys.readouterr().out
        assert "Audit: not run" in out

    def test_single_audit_block(self, task_dir, capsys):
        _write_validate(
            task_dir,
            "iterate\t2026-01-01T00:00:00Z\tclaude\topus\tFAIL\trep.md",
        )
        _print_audit_section(task_dir)
        out = capsys.readouterr().out
        assert "Audit (iterate): verdict FAIL" in out
        assert "findings 1" in out
        assert "report: rep.md" in out
        assert "claude/opus" in out

    def test_both_phases_table(self, task_dir, capsys):
        _write_validate(
            task_dir,
            "init\t2026-01-01T00:00:00Z\tclaude\topus\tPASS\ti.md",
            "iterate\t2026-01-02T00:00:00Z\tcursor\tcomposer\tWEAK\tit.md",
        )
        _print_audit_section(task_dir)
        out = capsys.readouterr().out
        assert "Audit:" in out
        assert "init" in out
        assert "iterate" in out
        assert "WEAK" in out


# --------------------------------------------------------------------------
# cmd_status — end to end on a crafted temp task dir
# --------------------------------------------------------------------------

class TestCmdStatusEndToEnd:
    def test_no_results_tsv_prints_no_rounds(self, task_dir, capsys):
        # task_dir fixture has task.yaml + .zyme/, but no results.tsv.
        cmd_status(_args(task_dir))
        out = capsys.readouterr().out
        assert f"Task: {task_dir.name}" in out
        assert "no results.tsv yet" in out

    def test_full_snapshot_with_results(self, task_dir_with_results_v0, capsys):
        cmd_status(_args(task_dir_with_results_v0))
        out = capsys.readouterr().out
        assert "Decision rounds:" in out
        # v0 fixture has 1 keep, 1 discard.
        assert "keeps=1" in out
        assert "discards=1" in out
        assert "Patch stack" in out
        assert "Last" in out and "decision" in out

    def test_best_ref_short_sha_shown(self, task_dir_with_results_v0, capsys):
        (task_dir_with_results_v0 / ".zyme" / "best.ref").write_text(
            "def5678901234567890\n"
        )
        cmd_status(_args(task_dir_with_results_v0))
        out = capsys.readouterr().out
        assert "Best: def5678" in out

    def test_no_best_ref_shows_dash(self, task_dir_with_results_v0, capsys):
        cmd_status(_args(task_dir_with_results_v0))
        out = capsys.readouterr().out
        assert "Best: —" in out
        assert "(no best yet)" in out

    def test_per_tier_table_speedup(self, task_dir_with_results_v0, capsys):
        # baseline 10s, best keep 8s → +20%.
        (task_dir_with_results_v0 / ".zyme" / "best.ref").write_text(
            "def5678901234\n"
        )
        cmd_status(_args(task_dir_with_results_v0))
        out = capsys.readouterr().out
        assert "Per tier:" in out
        assert "10.000s" in out  # baseline
        assert "+20.0%" in out

    def test_last_n_limit_respected(self, task_dir_with_results_v0, capsys):
        cmd_status(_args(task_dir_with_results_v0, last=1))
        out = capsys.readouterr().out
        # v0 has 2 decision rows (keep + discard); --last 1 shows only one.
        assert "Last 1 decision" in out

    def test_phase_all_widens_narrative(self, task_dir_with_results_v0, capsys):
        cmd_status(_args(task_dir_with_results_v0, phase="all"))
        out = capsys.readouterr().out
        assert "Decision rounds (all phases):" in out

    def test_validate_phase_label(self, task_dir_with_results_v2, capsys):
        cmd_status(_args(task_dir_with_results_v2, phase="validate"))
        out = capsys.readouterr().out
        assert "Scale-fix rounds:" in out

    def test_memory_phase_label(self, task_dir_with_results_v0, capsys):
        cmd_status(_args(task_dir_with_results_v0, phase="memory"))
        out = capsys.readouterr().out
        assert "Memory rounds:" in out

    def test_head_state_from_monkeypatched_git(self, task_dir_with_results_v0,
                                               capsys, monkeypatch):
        # Simulate a real repo with a dirty (modified) tree.
        def fake_git(*args, cwd=None, check=True, **kw):
            if args[:2] == ("rev-parse", "HEAD"):
                return "abcdef1234567890"
            if args[:1] == ("status",) and "--porcelain" in args:
                return " M pipeline/run.py"
            return ""
        monkeypatch.setattr(status, "git", fake_git)
        cmd_status(_args(task_dir_with_results_v0))
        out = capsys.readouterr().out
        assert "HEAD: abcdef1" in out
        assert "DIRTY (modified)" in out

    def test_head_untracked_only_is_clean(self, task_dir_with_results_v0,
                                          capsys, monkeypatch):
        def fake_git(*args, cwd=None, check=True, **kw):
            if args[:2] == ("rev-parse", "HEAD"):
                return "abcdef1234567890"
            if args[:1] == ("status",) and "--porcelain" in args:
                return "?? newfile.txt"
            return ""
        monkeypatch.setattr(status, "git", fake_git)
        cmd_status(_args(task_dir_with_results_v0))
        out = capsys.readouterr().out
        assert "clean (untracked-only)" in out

    def test_head_modified_and_untracked(self, task_dir_with_results_v0,
                                         capsys, monkeypatch):
        def fake_git(*args, cwd=None, check=True, **kw):
            if args[:2] == ("rev-parse", "HEAD"):
                return "abcdef1234567890"
            if args[:1] == ("status",) and "--porcelain" in args:
                return " M a.py\n?? b.txt"
            return ""
        monkeypatch.setattr(status, "git", fake_git)
        cmd_status(_args(task_dir_with_results_v0))
        out = capsys.readouterr().out
        assert "DIRTY (modified+untracked)" in out

    def test_validate_phase_verify_matrix(self, task_dir_full, capsys):
        # validate-phase view reads verify.tsv for the per-tier / verify matrix.
        (task_dir_full / "results.tsv").write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\tthread\n"
            "0\tabc1234\ttiny_a\t10.0\t0.0\t512.0\tbaseline\t{}\tup\t\toptimize\t1\n"
            "1\tdef5678\ttiny_a\t7.0\t30.0\t490.0\tkeep\t{}\t[scale-fix] S\t\tvalidate\t1\n"
        )
        (task_dir_full / "verify.tsv").write_text(
            "phase\tcommit\ttier\tthread\tspeed_sec\tstatus\n"
            "validate\tdef5678\ttiny\t1\t7.0\tpass\n"
            "validate\tdef5678\ttiny\t1\t7.2\tpass\n"
        )
        cmd_status(_args(task_dir_full, phase="validate"))
        out = capsys.readouterr().out
        assert "Scale-fix rounds:" in out
        assert "Verify matrix" in out
        assert "pass/total" in out

    def test_phase3_callout_when_validate_rounds_present(self, task_dir_full,
                                                         capsys):
        # results.tsv with an optimize keep + a validate decision row.
        (task_dir_full / "results.tsv").write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\tthread\n"
            "0\tabc1234\ttiny_a\t10.0\t0.0\t512.0\tbaseline\t{}\tup\t\toptimize\t1\n"
            "1\tdef5678\ttiny_a\t8.0\t20.0\t500.0\tkeep\t{}\tH\t\toptimize\t1\n"
            "2\tghi9012\ttiny_a\t7.0\t30.0\t490.0\tkeep\t{}\t[scale-fix] S\t\tvalidate\t1\n"
        )
        cmd_status(_args(task_dir_full, phase="optimize"))
        out = capsys.readouterr().out
        assert "Phase 3 (validate-scaling):" in out
