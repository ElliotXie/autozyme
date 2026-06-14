"""Wave-4 mop-up for zyme.commands.run — the optimization-loop commands.

Wave-2 (test_run_cmd.py) and wave-3 (test_run_w3.py) covered the helpers + the
core cmd_run/accept/reject/rollback happy paths over a real throwaway git repo.
This file fills the remaining REACHABLE decision branches those skipped:

  cmd_run:
    - --setup advancing best.ref (prior-best banner vs initialized banner),
    - task-definition-changed-outside-setup die,
    - version-drift WARN,
    - resolve_tiers ValueError (bad --dataset),
    - --phase validate auto-[scale-fix] prefix,
    - hoist-audit: exempt (task.yaml::hoist_exempt), bypass (--bypass-hoist),
      and blocked (sys.exit 2),
    - --rerun cost banner with an unknown estimate.
  cmd_accept:
    - HEAD-does-not-match-pending die,
    - description-omitted hint,
    - best_state cpu_sec / concordance lines via a metrics_json payload.
  cmd_reject:
    - cross-task sibling-commit refusal in a shared repo,
    - first-attempt (no best.ref) message,
    - crash auto --keep-tree default.
  cmd_rollback:
    - prior-keep flow (re-point best.ref + reset),
    - multi-keep dry-run banner.

Only runner.run_task and (where noted) module-level scan helpers are stubbed;
git is real.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import zyme.commands.run as run_cmd
from zyme.commands.run import (
    _maybe_reprofile_hint,
    _regenerate_best_state,
    cmd_accept,
    cmd_dryrun,
    cmd_reject,
    cmd_rollback,
    cmd_run,
)


_HDR = (
    "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
    "metrics_json\thypothesis\tdescription\tphase\tthread\n"
)


_TASK_YAML = (
    "target_repo: stub\n"
    "target_function: foo\n"
    "datasets:\n"
    "  - {tier: tiny, name: tiny_a, path: data/t.h5ad}\n"
    "metrics:\n"
    "  - {name: speedup, comparator: gte, threshold: 1.0}\n"
)

_BASELINE_RESULTS = (
    "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
    "metrics_json\thypothesis\tdescription\tphase\tthread\n"
    "0\tabc1234\ttiny_a\t10.0\t0.0\t512.0\tbaseline\t{}\tupstream\t\toptimize\t1\n"
)


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, text=True)


def _make_git_task(tmp_path: Path, *, with_baseline=True, name="task") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "task.yaml").write_text(_TASK_YAML)
    (d / "pipeline").mkdir()
    (d / "pipeline" / "run.py").write_text("print('orig')\n")
    (d / ".zyme").mkdir()
    # Real tasks gitignore runtime state so `git reset --hard` (reject/rollback)
    # doesn't revert results.tsv / .zyme/ / artifacts/.
    (d / ".gitignore").write_text(
        "results.tsv\n.zyme/\nartifacts/\nmemory/\n")
    if with_baseline:
        (d / "results.tsv").write_text(_BASELINE_RESULTS)
    _git("init", "-q", cwd=d)
    _git("config", "user.email", "t@t", cwd=d)
    _git("config", "user.name", "t", cwd=d)
    _git("add", "-A", cwd=d)
    _git("commit", "-q", "-m", "init", cwd=d)
    return d


def _ok_log(speed=5.0, peak=400.0, cpu=4.5, metrics_line="") -> str:
    body = (
        f"speed_sec:        {speed:.3f}\n"
        f"peak_mb:          {peak:.1f}\n"
        f"status:           ok\n"
        f"cpu_sec: {cpu}\n"
    )
    if metrics_line:
        body += metrics_line + "\n"
    return body


def _run_args(task_dir: Path, hypothesis="vectorize", **over) -> SimpleNamespace:
    base = dict(
        task_dir=str(task_dir), setup=None, rerun=False, hypothesis=hypothesis,
        dataset=None, extra_tiers=None, n_reps=None, phase="optimize",
        thread=None, yes=False, bypass_hoist=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _head(d):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=d,
                          capture_output=True, text=True).stdout.strip()


_round_seq = [0]


def _do_round(d, monkeypatch, log=None):
    # Unique pipeline body each call so there's always something to commit.
    _round_seq[0] += 1
    (d / "pipeline" / "run.py").write_text(
        f"print('faster v{_round_seq[0]}')\n")
    monkeypatch.setattr(run_cmd, "run_task",
                        lambda *a, **k: (log if log is not None else _ok_log()))
    cmd_run(_run_args(d, hypothesis=f"vectorize {_round_seq[0]}"))


# ===========================================================================
# cmd_run — --setup best.ref advance
# ===========================================================================

def test_setup_initializes_best_ref(tmp_path, capsys):
    d = _make_git_task(tmp_path)
    # No best.ref yet -> "initialized" banner (line 515).
    (d / "task.yaml").write_text(_TASK_YAML + "# tweak\n")
    cmd_run(_run_args(d, hypothesis=None, setup="wire threads"))
    out = capsys.readouterr().out
    assert "setup commit:" in out
    assert "best.ref initialized" in out
    assert (d / ".zyme" / "best.ref").exists()


def test_setup_advances_existing_best_ref(tmp_path, capsys):
    d = _make_git_task(tmp_path)
    (d / ".zyme" / "best.ref").write_text(_head(d))
    (d / "task.yaml").write_text(_TASK_YAML + "# more\n")
    cmd_run(_run_args(d, hypothesis=None, setup="adjust tiers"))
    out = capsys.readouterr().out
    assert "best.ref advanced:" in out


def test_setup_warns_when_evaluator_changed(tmp_path, capsys):
    d = _make_git_task(tmp_path)
    # Edit evaluate.py (a truth-surface file) so the WARN branch fires.
    (d / "evaluate.py").write_text("# evaluator v1\n")
    cmd_run(_run_args(d, hypothesis=None, setup="repair evaluator"))
    out = capsys.readouterr().out
    assert "truth surface changed" in out


# ===========================================================================
# cmd_run — task-definition-changed-outside-setup die
# ===========================================================================

def test_task_definition_changed_outside_setup_dies(tmp_path, capsys):
    d = _make_git_task(tmp_path)
    # Dirty task.yaml without --setup -> die (lines 519-526).
    (d / "task.yaml").write_text(_TASK_YAML + "# illegal edit\n")
    with pytest.raises(SystemExit):
        cmd_run(_run_args(d, hypothesis="vectorize"))
    assert "task-definition files changed outside --setup" in capsys.readouterr().err


# ===========================================================================
# cmd_run — version drift WARN
# ===========================================================================

def test_version_drift_warns(tmp_path, monkeypatch, capsys):
    d = _make_git_task(tmp_path)
    (d / "pipeline" / "run.py").write_text("print('v2')\n")
    monkeypatch.setattr(run_cmd, "check_upstream_version_drift", lambda td: {
        "agrees": False, "source": "DESCRIPTION", "package": "mgcv",
        "upstream_version": "1.9", "installed_version": "1.8",
    })
    monkeypatch.setattr(run_cmd, "run_task", lambda *a, **k: _ok_log())
    cmd_run(_run_args(d, hypothesis="vectorize"))
    out = capsys.readouterr().out
    assert "version drift detected" in out


# ===========================================================================
# cmd_run — bad --dataset (resolve_tiers ValueError)
# ===========================================================================

def test_unknown_dataset_dies(tmp_path, capsys):
    d = _make_git_task(tmp_path)
    (d / "pipeline" / "run.py").write_text("print('v2')\n")
    with pytest.raises(SystemExit):
        cmd_run(_run_args(d, hypothesis="vectorize", dataset="no_such_tier"))


# ===========================================================================
# cmd_run — --phase validate auto-prefix
# ===========================================================================

def test_validate_phase_auto_prefixes_scale_fix(tmp_path, monkeypatch, capsys):
    d = _make_git_task(tmp_path)
    (d / "pipeline" / "run.py").write_text("print('v2')\n")
    monkeypatch.setattr(run_cmd, "run_task", lambda *a, **k: _ok_log())
    cmd_run(_run_args(d, hypothesis="fix medium", phase="validate"))
    out = capsys.readouterr().out
    assert "auto-prefixed `[scale-fix]`" in out
    # commit message carries the prefix
    msg = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=d,
                         capture_output=True, text=True).stdout
    assert "[scale-fix]" in msg


# ===========================================================================
# cmd_run — hoist-audit branches
# ===========================================================================

def test_hoist_exempt_logs_and_continues(tmp_path, monkeypatch, capsys):
    d = _make_git_task(tmp_path)
    (d / "pipeline" / "run.py").write_text("print('v2')\n")
    monkeypatch.setattr(run_cmd, "scan_pipeline_hoist",
                        lambda p: ["upstream call at L3"])
    monkeypatch.setattr(run_cmd, "read_hoist_exempt", lambda td: "vetted manually")
    logged = {}
    monkeypatch.setattr(run_cmd, "append_hoist_log",
                        lambda *a, **kw: logged.update(kw))
    monkeypatch.setattr(run_cmd, "run_task", lambda *a, **k: _ok_log())
    cmd_run(_run_args(d, hypothesis="vectorize"))
    out = capsys.readouterr().out
    assert "skipped (task.yaml::hoist_exempt" in out
    assert logged["outcome"] == "exempted"


def test_hoist_bypass_warns_and_continues(tmp_path, monkeypatch, capsys):
    d = _make_git_task(tmp_path)
    (d / "pipeline" / "run.py").write_text("print('v2')\n")
    monkeypatch.setattr(run_cmd, "scan_pipeline_hoist", lambda p: ["v"])
    monkeypatch.setattr(run_cmd, "read_hoist_exempt", lambda td: None)
    monkeypatch.setattr(run_cmd, "format_violations_message",
                        lambda p, v, hypothesis: "VIOLATION MSG")
    logged = {}
    monkeypatch.setattr(run_cmd, "append_hoist_log",
                        lambda *a, **kw: logged.update(kw))
    monkeypatch.setattr(run_cmd, "run_task", lambda *a, **k: _ok_log())
    cmd_run(_run_args(d, hypothesis="vectorize", bypass_hoist="known prewarm"))
    out = capsys.readouterr().out
    assert "BYPASSED for this round" in out
    assert logged["outcome"] == "bypassed"


def test_hoist_blocked_exits_two(tmp_path, monkeypatch, capsys):
    d = _make_git_task(tmp_path)
    (d / "pipeline" / "run.py").write_text("print('v2')\n")
    monkeypatch.setattr(run_cmd, "scan_pipeline_hoist", lambda p: ["v"])
    monkeypatch.setattr(run_cmd, "read_hoist_exempt", lambda td: None)
    monkeypatch.setattr(run_cmd, "format_violations_message",
                        lambda p, v, hypothesis: "VIOLATION MSG")
    logged = {}
    monkeypatch.setattr(run_cmd, "append_hoist_log",
                        lambda *a, **kw: logged.update(kw))
    with pytest.raises(SystemExit) as ei:
        cmd_run(_run_args(d, hypothesis="vectorize"))
    assert ei.value.code == 2
    assert logged["outcome"] == "blocked"


# ===========================================================================
# cmd_run — --rerun cost banner with unknown estimate
# ===========================================================================

def test_rerun_banner_unknown_estimate(tmp_path, monkeypatch, capsys):
    # No baseline + no head measurements -> estimate unknown branch (606-607).
    # The banner renders before any measurement; the rerun itself may then
    # exit (no data file for the stubbed pipeline), which is fine — we only
    # care that the unknown-estimate branch printed.
    d = _make_git_task(tmp_path, with_baseline=False)
    monkeypatch.setattr(run_cmd, "run_task", lambda *a, **k: _ok_log())
    try:
        cmd_run(_run_args(d, hypothesis=None, rerun=True, n_reps=2))
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "estimate unknown" in out


# ===========================================================================
# cmd_accept — HEAD mismatch + cpu_sec/concordance best_state
# ===========================================================================

def test_accept_head_mismatch_dies(tmp_path, monkeypatch, capsys):
    d = _make_git_task(tmp_path)
    _do_round(d, monkeypatch)
    capsys.readouterr()
    # Advance HEAD with an unrelated commit so it no longer matches the pending
    # row's recorded commit -> the safety die (lines 1119-1129).
    (d / "sibling.txt").write_text("x")
    _git("add", "-A", cwd=d)
    _git("commit", "-q", "-m", "sibling slipped in", cwd=d)
    with pytest.raises(SystemExit):
        cmd_accept(SimpleNamespace(task_dir=str(d), description="x",
                                   dismiss_housekeeping=False))
    assert "does not match the pending round" in capsys.readouterr().err


def test_accept_no_description_hint(tmp_path, monkeypatch, capsys):
    d = _make_git_task(tmp_path)
    _do_round(d, monkeypatch)
    capsys.readouterr()
    cmd_accept(SimpleNamespace(task_dir=str(d), description=None,
                               dismiss_housekeeping=False))
    out = capsys.readouterr().out
    assert "description omitted" in out


def test_accept_best_state_has_cpu_and_concordance(tmp_path, monkeypatch, capsys):
    d = _make_git_task(tmp_path)
    # Runner log emits cpu_sec + a numeric concordance metric (jaccard). Both
    # become metrics in results.tsv, so best_state.md renders the CPU-time +
    # Concordance lines (1384-1389).
    log = _ok_log(metrics_line="jaccard: 0.97")
    _do_round(d, monkeypatch, log=log)
    capsys.readouterr()
    cmd_accept(SimpleNamespace(task_dir=str(d), description="keep it",
                               dismiss_housekeeping=False))
    capsys.readouterr()
    text = (d / "memory" / "best_state.md").read_text()
    assert "CPU time:" in text
    assert "Concordance:" in text
    assert "jaccard=0.97" in text


# ===========================================================================
# cmd_reject — cross-task refusal, first-attempt, crash keep-tree
# ===========================================================================

def test_reject_refuses_cross_task_commits(tmp_path, monkeypatch, capsys):
    # Shared repo: <root>/.git, task at <root>/test_fix/sc_a/. A sibling commit
    # touches files OUTSIDE the task between best and HEAD -> reject refuses.
    root = tmp_path / "shared"
    root.mkdir()
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "t@t", cwd=root)
    _git("config", "user.name", "t", cwd=root)
    task = root / "test_fix" / "sc_a"
    (task / "pipeline").mkdir(parents=True)
    (task / "pipeline" / "run.py").write_text("print('orig')\n")
    (task / ".zyme").mkdir()
    (task / "task.yaml").write_text(_TASK_YAML)
    (task / "results.tsv").write_text(
        _BASELINE_RESULTS
        + "1\tdef5678\ttiny_a\t5\t50\t400\tpending\th\t\toptimize\t1\n")
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "init", cwd=root)
    best_sha = _head(root)
    (task / ".zyme" / "best.ref").write_text(best_sha)
    # Sibling commit OUTSIDE the task.
    (root / "other_task" / "x.py").parent.mkdir(parents=True)
    (root / "other_task" / "x.py").write_text("y\n")
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "sibling task work", cwd=root)
    # A commit in the task too (so there's something to reject), no uncommitted.
    (task / "pipeline" / "run.py").write_text("print('attempt')\n")
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "attempt", cwd=root)
    with pytest.raises(SystemExit):
        cmd_reject(SimpleNamespace(task_dir=str(task), description="x",
                                   keep_tree=False, force=False))
    assert "OUTSIDE" in capsys.readouterr().err


def test_reject_first_attempt_no_best_ref(tmp_path, monkeypatch, capsys):
    d = _make_git_task(tmp_path)
    _do_round(d, monkeypatch)
    capsys.readouterr()
    # No best.ref -> the "first attempt, no prior best" message (1492-1497).
    cmd_reject(SimpleNamespace(task_dir=str(d), description="bad",
                               keep_tree=True, force=False))
    out = capsys.readouterr().out
    assert "no `best.ref` yet" in out


def test_reject_crash_defaults_keep_tree(tmp_path, monkeypatch, capsys):
    d = _make_git_task(tmp_path)
    (d / ".zyme" / "best.ref").write_text(_head(d))
    # A crash round leaves an uncommitted fix-in-progress; reject auto
    # --keep-tree (lines 1434-1440) instead of refusing.
    _do_round(d, monkeypatch, log=_ok_log())  # commit a baseline keep first
    # Now produce a crash pending row by hand-editing results.tsv last status.
    lines = (d / "results.tsv").read_text().splitlines()
    parts = lines[-1].split("\t")
    parts[6] = "crash"
    lines[-1] = "\t".join(parts)
    (d / "results.tsv").write_text("\n".join(lines) + "\n")
    # Leave an uncommitted edit (the "fix in progress").
    (d / "pipeline" / "run.py").write_text("print('fixing')\n")
    cmd_reject(SimpleNamespace(task_dir=str(d), description="crashed",
                               keep_tree=False, force=False))
    out = capsys.readouterr().out
    assert "defaulting to --keep-tree" in out


# ===========================================================================
# cmd_rollback — prior-keep flow + multi-keep dry-run
# ===========================================================================

def _accept_round(d, monkeypatch, capsys):
    _do_round(d, monkeypatch)
    capsys.readouterr()
    cmd_accept(SimpleNamespace(task_dir=str(d), description="k",
                               dismiss_housekeeping=False))
    capsys.readouterr()


def test_rollback_prior_keep_repoints_best(tmp_path, monkeypatch, capsys):
    d = _make_git_task(tmp_path)
    _accept_round(d, monkeypatch, capsys)
    first_keep_sha = _head(d)
    _accept_round(d, monkeypatch, capsys)
    # Two keeps now; rollback demotes the latest and re-points best.ref to the
    # prior keep + resets HEAD there (lines 1645-1655).
    cmd_rollback(SimpleNamespace(task_dir=str(d), description="regression",
                                 dry_run=False))
    out = capsys.readouterr().out
    assert "best now points at" in out
    assert _head(d) == first_keep_sha
    # latest keep row flipped to rollback
    statuses = [ln.split("\t")[6]
                for ln in (d / "results.tsv").read_text().splitlines()[1:]]
    assert "rollback" in statuses


def test_rollback_dry_run_multi_keep_banner(tmp_path, monkeypatch, capsys):
    d = _make_git_task(tmp_path)
    _accept_round(d, monkeypatch, capsys)
    _accept_round(d, monkeypatch, capsys)
    before = (d / "results.tsv").read_text()
    cmd_rollback(SimpleNamespace(task_dir=str(d), description="check",
                                 dry_run=True))
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "repoint best.ref" in out
    assert (d / "results.tsv").read_text() == before  # untouched


def test_rollback_no_rows_dies(tmp_path, capsys):
    d = _make_git_task(tmp_path, with_baseline=False)
    (d / "results.tsv").write_text(_HDR)  # header only, no data rows
    with pytest.raises(SystemExit):
        cmd_rollback(SimpleNamespace(task_dir=str(d), description="",
                                     dry_run=False))
    assert "no rows" in capsys.readouterr().err


# ===========================================================================
# cmd_dryrun — stderr passthrough
# ===========================================================================

def test_dryrun_passes_stderr(tmp_path, monkeypatch, capsys):
    d = _make_git_task(tmp_path)
    monkeypatch.setattr(run_cmd, "dryrun_task",
                        lambda td, dataset_entry: (0, "OUT\n", "ERR\n", 1.5))
    cmd_dryrun(SimpleNamespace(task_dir=str(d), dataset=None))
    captured = capsys.readouterr()
    assert "OUT" in captured.out
    assert "ERR" in captured.err


def test_dryrun_bad_dataset_dies(tmp_path, capsys):
    d = _make_git_task(tmp_path)
    with pytest.raises(SystemExit):
        cmd_dryrun(SimpleNamespace(task_dir=str(d), dataset="ghost_tier"))


# ===========================================================================
# pure helper: _maybe_reprofile_hint fires on a 20% step crossing
# ===========================================================================

def test_reprofile_hint_fires_on_step(tmp_path, capsys):
    p = tmp_path / "results.tsv"
    # Two keeps; baseline 10s, latest 6s -> 40% cumulative -> crosses the 20%
    # step boundary from the default 0 -> hint prints + writes the marker file.
    (tmp_path / ".zyme").mkdir()
    p.write_text(
        _HDR
        + "0\tabc\ttiny_a\t10\t0\t100\tbaseline\t{}\tu\t\toptimize\t1\n"
        + "1\tdef\ttiny_a\t9\t10\t100\tkeep\t{}\th\tfirst\toptimize\t1\n"
        + "2\tghi\ttiny_a\t6\t40\t100\tkeep\t{}\th\tsecond\toptimize\t1\n"
    )
    _maybe_reprofile_hint(tmp_path, p)
    out = capsys.readouterr().out
    assert "cumulative speedup is now 40%" in out
    assert (tmp_path / ".zyme" / "last_reprofile_hint_pct").exists()


def test_reprofile_hint_silent_below_step(tmp_path, capsys):
    p = tmp_path / "results.tsv"
    (tmp_path / ".zyme").mkdir()
    # Pre-seed a high prior hint so the +20% step isn't crossed again.
    (tmp_path / ".zyme" / "last_reprofile_hint_pct").write_text("40.0")
    p.write_text(
        _HDR
        + "0\tabc\ttiny_a\t10\t0\t100\tbaseline\t{}\tu\t\toptimize\t1\n"
        + "1\tdef\ttiny_a\t9\t10\t100\tkeep\t{}\th\ta\toptimize\t1\n"
        + "2\tghi\ttiny_a\t8\t20\t100\tkeep\t{}\th\tb\toptimize\t1\n"  # 20% < 40+20
    )
    _maybe_reprofile_hint(tmp_path, p)
    assert "[hint]" not in capsys.readouterr().out


# ===========================================================================
# pure helper: _regenerate_best_state no-baseline-for-tier branch
# ===========================================================================

def test_best_state_no_baseline_for_tier(tmp_path):
    p = tmp_path / "results.tsv"
    # A keep at tier 'medium_a' with NO baseline row for that tier -> the
    # "(no baseline recorded for this tier)" branch (line 1379).
    p.write_text(
        _HDR
        + "0\tabc\ttiny_a\t10\t0\t100\tbaseline\t{}\tu\t\toptimize\t1\n"
        + "1\tdef\tmedium_a\t4\t60\t200\tkeep\t{}\th\tnice\toptimize\t1\n"
    )
    out = _regenerate_best_state(tmp_path)
    text = Path(out).read_text()
    assert "no baseline recorded for this tier" in text
