"""Wave-3 mop-up for zyme.commands.run.

Wave-2 (tests/commands/test_run_cmd.py) covered the big cmd_run/accept/reject/
rollback entry points end-to-end on a real throwaway git repo. This file fills
the remaining REACHABLE in-process helper / decision branches:

  - _unstage_generated_pipeline_artifacts (real staged-then-reset on a git repo)
  - _row_thread_for_estimate parsing edge cases
  - _current_head_speed_estimate HEAD-median branch + baseline-ratio projection
  - _task_definition_changes / _commit_files prefix-stripping
  - _write_setup_audit JSONL append
  - _audit_commit_files native-source carve-out (setup vs fix-loop scope)
  - _maybe_memory_regression_warning thresholds (fires / silent / thread-join)

Only git is "real"; no pipeline subprocess is launched.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import zyme.commands.run as run_cmd
from zyme.commands.run import (
    _audit_commit_files,
    _commit_files,
    _current_head_speed_estimate,
    _maybe_memory_regression_warning,
    _row_thread_for_estimate,
    _task_definition_changes,
    _unstage_generated_pipeline_artifacts,
    _write_setup_audit,
)


_TASK_YAML = (
    "target_repo: stub\n"
    "target_function: foo\n"
    "datasets:\n"
    "  - {tier: tiny, name: tiny_a, path: data/t.h5ad}\n"
    "  - {tier: medium, name: medium_a, path: data/m.h5ad}\n"
    "metrics:\n"
    "  - {name: speedup, comparator: gte, threshold: 1.0}\n"
)

_HDR = (
    "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
    "metrics_json\thypothesis\tdescription\tphase\tthread\n"
)


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, text=True)


def _make_git_task(tmp_path: Path) -> Path:
    d = tmp_path / "task"
    d.mkdir()
    (d / "task.yaml").write_text(_TASK_YAML)
    (d / "pipeline").mkdir()
    (d / "pipeline" / "run.py").write_text("print('orig')\n")
    (d / ".zyme").mkdir()
    _git("init", "-q", cwd=d)
    _git("config", "user.email", "t@t", cwd=d)
    _git("config", "user.name", "t", cwd=d)
    _git("add", "-A", cwd=d)
    _git("commit", "-q", "-m", "init", cwd=d)
    return d


def _head(d):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=d,
                          capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------------
# _unstage_generated_pipeline_artifacts
# ---------------------------------------------------------------------------

def test_unstage_drops_only_generated(tmp_path):
    d = _make_git_task(tmp_path)
    # Stage an edit to run.py (NOT generated) + a generated metrics.json.
    (d / "pipeline" / "run.py").write_text("print('v2')\n")
    (d / "pipeline" / "metrics.json").write_text("{}")
    _git("add", "-A", cwd=d)
    dropped = _unstage_generated_pipeline_artifacts(d)
    assert dropped == ["pipeline/metrics.json"]
    # run.py stays staged.
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=d,
        capture_output=True, text=True).stdout.split()
    assert "pipeline/run.py" in staged
    assert "pipeline/metrics.json" not in staged


def test_unstage_noop_when_nothing_generated(tmp_path):
    d = _make_git_task(tmp_path)
    (d / "pipeline" / "run.py").write_text("print('v2')\n")
    _git("add", "-A", cwd=d)
    assert _unstage_generated_pipeline_artifacts(d) == []


# ---------------------------------------------------------------------------
# _row_thread_for_estimate
# ---------------------------------------------------------------------------

def test_row_thread_default_when_no_col():
    assert _row_thread_for_estimate(["a", "b"], {"speed": 0}) == run_cmd.LEGACY_THREAD


def test_row_thread_default_when_idx_out_of_range():
    assert _row_thread_for_estimate(["a"], {"thread": 5}) == run_cmd.LEGACY_THREAD


def test_row_thread_parses_int():
    assert _row_thread_for_estimate(["x", "4"], {"thread": 1}) == 4


def test_row_thread_bad_value_falls_back():
    assert _row_thread_for_estimate(["x", "nope"], {"thread": 1}) == \
        run_cmd.LEGACY_THREAD


# ---------------------------------------------------------------------------
# _current_head_speed_estimate: HEAD-median path
# ---------------------------------------------------------------------------

def test_head_speed_estimate_uses_head_median(tmp_path):
    d = _make_git_task(tmp_path)
    head7 = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=d,
                           capture_output=True, text=True).stdout.strip()
    # Two pending rows at HEAD for tiny_a, thread 1 -> median.
    (d / "results.tsv").write_text(
        _HDR
        + "0\tabc1234\ttiny_a\t10.0\t0.0\t512.0\tbaseline\t{}\tu\t\toptimize\t1\n"
        + f"1\t{head7}\ttiny_a\t6.0\t40.0\t500\tpending\t{{}}\th\t\toptimize\t1\n"
        + f"1\t{head7}\ttiny_a\t8.0\t20.0\t500\trerun\t{{}}\th\t\toptimize\t1\n"
    )
    speed, src = _current_head_speed_estimate(d, "tiny_a", thread=1)
    assert speed == pytest.approx(7.0)  # median(6, 8)
    assert "current HEAD median, n=2" == src


def test_head_speed_estimate_baseline_ratio_projection(tmp_path):
    # medium_a has only a baseline row; tiny_a has baseline 10 + best 5 (=2x).
    # The projection returns baseline_medium / 2.0.
    d = _make_git_task(tmp_path)
    (d / ".zyme" / "best.ref").write_text("def5678901234\n")
    (d / "results.tsv").write_text(
        _HDR
        + "0\tabc1234\ttiny_a\t10.0\t0.0\t512.0\tbaseline\t{}\tu\t\toptimize\t1\n"
        + "1\tdef5678\ttiny_a\t5.0\t50.0\t500\tkeep\t{}\th\t\toptimize\t1\n"
        + "0\tabc1234\tmedium_a\t40.0\t0.0\t900.0\tbaseline\t{}\tu\t\toptimize\t1\n"
    )
    speed, src = _current_head_speed_estimate(d, "medium_a", thread=1)
    assert speed == pytest.approx(20.0)  # 40 / 2.0
    assert "median speedup" in src


# ---------------------------------------------------------------------------
# _task_definition_changes / _commit_files
# ---------------------------------------------------------------------------

def test_task_definition_changes_detects_yaml_edit(tmp_path):
    d = _make_git_task(tmp_path)
    (d / "task.yaml").write_text(_TASK_YAML + "# tweak\n")
    changed = _task_definition_changes(d)
    assert "task.yaml" in changed


def test_commit_files_repo_root_returns_repo_relative(tmp_path):
    d = _make_git_task(tmp_path)
    (d / "pipeline" / "run.py").write_text("print('v2')\n")
    _git("add", "-A", cwd=d)
    _git("commit", "-q", "-m", "edit", cwd=d)
    files = _commit_files(d, _head(d))
    # Task IS the repo root -> task_rel == "." -> repo-relative paths returned.
    assert "pipeline/run.py" in files


def test_commit_files_strips_task_prefix_in_shared_repo(tmp_path):
    # Shared repo: <root>/.git, task lives at <root>/test_fix/sc_leiden/.
    root = tmp_path / "shared"
    root.mkdir()
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "t@t", cwd=root)
    _git("config", "user.name", "t", cwd=root)
    task = root / "test_fix" / "sc_leiden"
    (task / "pipeline").mkdir(parents=True)
    (task / "pipeline" / "run.py").write_text("print('x')\n")
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "init", cwd=root)
    (task / "pipeline" / "run.py").write_text("print('y')\n")
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "edit", cwd=root)
    files = _commit_files(task, _head(root))
    # The task prefix is stripped so the path compares against the allowlist.
    assert "pipeline/run.py" in files
    assert "test_fix/sc_leiden/pipeline/run.py" not in files


# ---------------------------------------------------------------------------
# _write_setup_audit
# ---------------------------------------------------------------------------

def test_write_setup_audit_appends_jsonl(tmp_path):
    d = _make_git_task(tmp_path)
    _write_setup_audit(d, "deadbeef1234", "wire threads",
                       ["task.yaml", "setup/prep.py"])
    log = d / ".zyme" / "setup_audit.log"
    assert log.exists()
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["commit"] == "deadbee"  # commit_sha[:7]
    assert rec["message"] == "wire threads"
    assert rec["files"] == ["task.yaml", "setup/prep.py"]


# ---------------------------------------------------------------------------
# _audit_commit_files — native-source + setup-scope branches
# ---------------------------------------------------------------------------

def _commit_file(d, rel, body="x\n"):
    p = d / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    _git("add", "-A", cwd=d)
    _git("commit", "-q", "-m", f"touch {rel}", cwd=d)
    return _head(d)


def test_audit_fix_loop_allows_native_sibling(tmp_path, capsys):
    d = _make_git_task(tmp_path)
    sha = _commit_file(d, "pipeline/kernel.cpp")
    _audit_commit_files(d, sha, "fix-loop", "optimize", "vectorize via sourceCpp")
    # .cpp sibling in pipeline/ is allowed -> no violation logged.
    assert "[audit]" not in capsys.readouterr().out
    assert not (d / ".zyme" / "edit_audit.log").exists()


def test_audit_fix_loop_flags_nested_native(tmp_path, capsys):
    d = _make_git_task(tmp_path)
    # A .cpp NOT directly under pipeline/ (2 slashes) is not the carve-out.
    sha = _commit_file(d, "pipeline/sub/kernel.cpp")
    _audit_commit_files(d, sha, "fix-loop", "optimize", "h")
    out = capsys.readouterr().out
    assert "[audit]" in out
    assert "pipeline/sub/kernel.cpp" in out


def test_audit_setup_scope_allows_reference_and_setup(tmp_path, capsys):
    d = _make_git_task(tmp_path)
    sha = _commit_file(d, "setup/prep.py")  # under setup/ prefix -> allowed
    _audit_commit_files(d, sha, "setup", "optimize", "h")
    assert "[audit]" not in capsys.readouterr().out


def test_audit_setup_scope_flags_outside(tmp_path, capsys):
    d = _make_git_task(tmp_path)
    sha = _commit_file(d, "weird_helper.py")  # not in setup allowlist
    _audit_commit_files(d, sha, "setup", "optimize", "h")
    out = capsys.readouterr().out
    assert "[audit]" in out
    assert "weird_helper.py" in out


def test_audit_unknown_kind_returns_early(tmp_path, capsys):
    d = _make_git_task(tmp_path)
    sha = _commit_file(d, "anything.py")
    _audit_commit_files(d, sha, "other", "optimize", "h")
    assert capsys.readouterr().out == ""
    assert not (d / ".zyme" / "edit_audit.log").exists()


# ---------------------------------------------------------------------------
# _maybe_memory_regression_warning
# ---------------------------------------------------------------------------

def test_memory_warning_no_file(tmp_path):
    # Missing results.tsv -> silent.
    _maybe_memory_regression_warning(tmp_path, tmp_path / "nope.tsv")


def test_memory_warning_fires_on_regression(tmp_path, capsys):
    p = tmp_path / "results.tsv"
    p.write_text(
        _HDR
        + "0\tabc\ttiny_a\t10\t0\t100.0\tbaseline\t{}\tu\t\toptimize\t1\n"
        + "1\tdef\ttiny_a\t8\t20\t160.0\tkeep\t{}\th\t\toptimize\t1\n"
    )
    _maybe_memory_regression_warning(tmp_path, p)
    out = capsys.readouterr().out
    assert "[memory] WARNING" in out
    assert "60%" in out  # 160 vs 100 = +60%


def test_memory_warning_silent_below_threshold(tmp_path, capsys):
    p = tmp_path / "results.tsv"
    p.write_text(
        _HDR
        + "0\tabc\ttiny_a\t10\t0\t100.0\tbaseline\t{}\tu\t\toptimize\t1\n"
        + "1\tdef\ttiny_a\t8\t20\t105.0\tkeep\t{}\th\t\toptimize\t1\n"  # +5%
    )
    _maybe_memory_regression_warning(tmp_path, p)
    assert "[memory]" not in capsys.readouterr().out


def test_memory_warning_silent_without_keep(tmp_path, capsys):
    p = tmp_path / "results.tsv"
    p.write_text(_HDR
                 + "0\tabc\ttiny_a\t10\t0\t100.0\tbaseline\t{}\tu\t\toptimize\t1\n")
    _maybe_memory_regression_warning(tmp_path, p)
    assert "[memory]" not in capsys.readouterr().out


def test_memory_warning_thread_join_skips_serial_baseline(tmp_path, capsys):
    # keep at thread=8 but baseline only recorded at thread=1 -> no match,
    # so no comparison and no warning.
    p = tmp_path / "results.tsv"
    p.write_text(
        _HDR
        + "0\tabc\ttiny_a\t10\t0\t100.0\tbaseline\t{}\tu\t\toptimize\t1\n"
        + "1\tdef\ttiny_a\t8\t20\t500.0\tkeep\t{}\th\t\toptimize\t8\n"
    )
    _maybe_memory_regression_warning(tmp_path, p)
    assert "[memory]" not in capsys.readouterr().out


def test_memory_warning_silent_when_keep_peak_missing(tmp_path, capsys):
    p = tmp_path / "results.tsv"
    p.write_text(
        _HDR
        + "0\tabc\ttiny_a\t10\t0\t100.0\tbaseline\t{}\tu\t\toptimize\t1\n"
        + "1\tdef\ttiny_a\t8\t20\t0\tkeep\t{}\th\t\toptimize\t1\n"  # peak 0
    )
    _maybe_memory_regression_warning(tmp_path, p)
    assert "[memory]" not in capsys.readouterr().out
