"""Unit tests for zyme.commands.run — the optimization-loop commands
(run / dryrun / accept / reject / rollback) and their pure helpers.

Two layers:

  1. Pure helpers (no subprocess): generated-artifact classification, duration
     formatting, crash-message extraction, pending-row / last-status lookups,
     median-speedup-ratio + head-speed estimation, best_state regeneration,
     reprofile-hint stepping, commit-file scope audit.

  2. The cmd_* entry points, driven against a REAL throwaway git repo (so the
     git-presence guard and commit/reset machinery run for real) with the only
     genuine subprocess boundary — `runner.run_task` / `runner.dryrun_task` —
     monkeypatched to return a canned runner-log string. This exercises the
     surrounding orchestration (commit, results.tsv row, artifacts snapshot,
     budget banner, accept/reject/rollback state transitions) deterministically.

No real pipeline subprocess, no network. Everything on tmp_path.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import zyme.commands.run as run_cmd
from zyme.commands.run import (
    _current_head_speed_estimate,
    _extract_crash_msg_from_log,
    _find_pending_row,
    _format_duration,
    _is_generated_pipeline_artifact,
    _last_data_row_status,
    _maybe_reprofile_hint,
    _median_speedup_ratio,
    _pending_rounds,
    _regenerate_best_state,
    _status_paths,
    cmd_accept,
    cmd_dryrun,
    cmd_reject,
    cmd_rollback,
    cmd_run,
)


# --------------------------------------------------------------------------
# A real (tiny) git task repo + a canned runner-log.
# --------------------------------------------------------------------------

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


def _make_git_task(tmp_path: Path, *, with_baseline=True) -> Path:
    """A throwaway git repo that looks like a zyme task."""
    d = tmp_path / "task"
    d.mkdir()
    (d / "task.yaml").write_text(_TASK_YAML)
    (d / "pipeline").mkdir()
    (d / "pipeline" / "run.py").write_text("print('orig')\n")
    (d / ".zyme").mkdir()
    if with_baseline:
        (d / "results.tsv").write_text(_BASELINE_RESULTS)
    _git("init", "-q", cwd=d)
    _git("config", "user.email", "t@t", cwd=d)
    _git("config", "user.name", "t", cwd=d)
    _git("add", "-A", cwd=d)
    _git("commit", "-q", "-m", "init", cwd=d)
    return d


def _ok_log(speed=5.0, peak=400.0, cpu=4.5) -> str:
    return (
        f"speed_sec:        {speed:.3f}\n"
        f"peak_mb:          {peak:.1f}\n"
        f"status:           ok\n"
        f"cpu_sec: {cpu}\n"
    )


def _crash_log(msg="ValueError: boom") -> str:
    return (
        f"CRASH: pipeline exited 1\n"
        f"crash_msg: {msg}\n"
        f"speed_sec:        0.000\n"
        f"peak_mb:          0.0\n"
        f"status:           crash\n"
    )


def _run_args(task_dir: Path, hypothesis="vectorize", **over) -> SimpleNamespace:
    base = dict(
        task_dir=str(task_dir), setup=None, rerun=False, hypothesis=hypothesis,
        dataset=None, extra_tiers=None, n_reps=None, phase="optimize",
        thread=None, yes=False, bypass_hoist=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------
# _is_generated_pipeline_artifact
# --------------------------------------------------------------------------

class TestIsGeneratedPipelineArtifact:
    @pytest.mark.parametrize("rel,expected", [
        ("pipeline/output_tiny/result.csv", True),
        ("pipeline/output/x.parquet", True),
        ("pipeline/metrics.json", True),
        ("pipeline/profile.json", True),
        ("pipeline/Rprof.out", True),
        ("pipeline/native_sample_3.txt", True),
        ("pipeline/result.h5ad", True),
        ("pipeline/result.rds", True),
        ("pipeline/run.py", False),
        ("pipeline/helper.R", False),
        ("task.yaml", False),
        ("evaluate.py", False),
        ("pipeline/sub/dir/run.py", False),
    ])
    def test_classification(self, rel, expected):
        assert _is_generated_pipeline_artifact(rel) is expected

    def test_empty_path(self):
        assert _is_generated_pipeline_artifact("") is False


# --------------------------------------------------------------------------
# _format_duration
# --------------------------------------------------------------------------

class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(5.0) == "5.0s"

    def test_minutes(self):
        assert _format_duration(90.0) == "1.5 min"

    def test_hours(self):
        assert _format_duration(7200.0) == "2.0 h"


# --------------------------------------------------------------------------
# _extract_crash_msg_from_log
# --------------------------------------------------------------------------

class TestExtractCrashMsgFromLog:
    def test_present(self):
        log = "noise\ncrash_msg: ValueError: bad input\nmore\n"
        assert _extract_crash_msg_from_log(log) == "ValueError: bad input"

    def test_absent(self):
        assert _extract_crash_msg_from_log("no crash here\n") == ""


# --------------------------------------------------------------------------
# _pending_rounds / _find_pending_row / _last_data_row_status
# --------------------------------------------------------------------------

class TestPendingHelpers:
    def _write(self, task_dir, *rows):
        p = task_dir / "results.tsv"
        p.write_text(_BASELINE_RESULTS + "".join(rows))
        return p

    def test_pending_rounds_empty_when_no_file(self, tmp_path):
        assert _pending_rounds(tmp_path / "nope.tsv") == []

    def test_pending_rounds_lists_pending(self, task_dir):
        p = self._write(
            task_dir,
            "1\tdef5678\ttiny_a\t8.0\t20.0\t500\tpending\t{}\tH\t\toptimize\t1\n",
            "2\tghi9012\ttiny_a\t7.0\t30.0\t490\tkeep\t{}\tH2\t\toptimize\t1\n",
        )
        assert _pending_rounds(p) == ["1"]

    def test_find_pending_row_returns_latest(self, task_dir):
        p = self._write(
            task_dir,
            "1\tdef5678\ttiny_a\t8.0\t20.0\t500\tpending\t{}\tH\t\toptimize\t1\n",
        )
        row = _find_pending_row(p)
        assert row is not None
        assert row["commit"] == "def5678"

    def test_find_pending_row_none_when_no_pending(self, task_dir):
        p = self._write(
            task_dir,
            "1\tdef5678\ttiny_a\t8.0\t20.0\t500\tkeep\t{}\tH\t\toptimize\t1\n",
        )
        assert _find_pending_row(p) is None

    def test_last_data_row_status(self, task_dir):
        p = self._write(
            task_dir,
            "1\tdef5678\ttiny_a\t8.0\t20.0\t500\tkeep\t{}\tH\t\toptimize\t1\n",
            "2\tghi9012\ttiny_a\t7.0\t30.0\t490\tdiscard\t{}\tH2\t\toptimize\t1\n",
        )
        assert _last_data_row_status(p) == "discard"

    def test_last_data_row_status_none_when_missing(self, tmp_path):
        assert _last_data_row_status(tmp_path / "nope.tsv") is None


# --------------------------------------------------------------------------
# _median_speedup_ratio / _current_head_speed_estimate
# --------------------------------------------------------------------------

class TestMedianSpeedupRatio:
    def test_none_when_no_yaml(self, tmp_path):
        assert _median_speedup_ratio(tmp_path, thread=1) is None

    def test_ratio_from_baseline_and_best(self, task_dir):
        # task_dir fixture uses MINIMAL_TASK_YAML with a single tiny_a dataset.
        (task_dir / "results.tsv").write_text(
            _BASELINE_RESULTS
            + "1\tdef5678\ttiny_a\t5.0\t50.0\t500\tkeep\t{}\tH\t\toptimize\t1\n"
        )
        (task_dir / ".zyme" / "best.ref").write_text("def5678901234\n")
        # baseline 10 / best 5 = 2.0; but exclude_dataset filters tiny_a out.
        r = _median_speedup_ratio(task_dir, thread=1)
        assert r == pytest.approx(2.0)
        assert _median_speedup_ratio(
            task_dir, thread=1, exclude_dataset="tiny_a") is None


class TestCurrentHeadSpeedEstimate:
    def test_unknown_when_nothing(self, task_dir):
        speed, src = _current_head_speed_estimate(task_dir, "tiny_a", thread=1)
        assert speed is None
        assert src == "unknown"

    def test_baseline_fallback(self, task_dir):
        (task_dir / "results.tsv").write_text(_BASELINE_RESULTS)
        speed, src = _current_head_speed_estimate(task_dir, "tiny_a", thread=1)
        assert speed == 10.0
        assert src == "baseline"


# --------------------------------------------------------------------------
# _regenerate_best_state
# --------------------------------------------------------------------------

class TestRegenerateBestState:
    def test_none_without_results(self, task_dir):
        assert _regenerate_best_state(task_dir) is None

    def test_none_without_keeps(self, task_dir):
        (task_dir / "results.tsv").write_text(_BASELINE_RESULTS)
        assert _regenerate_best_state(task_dir) is None

    def test_writes_best_state_md(self, task_dir):
        (task_dir / "results.tsv").write_text(
            _BASELINE_RESULTS
            + "1\tdef5678\ttiny_a\t8.0\t20.0\t500\tkeep\t"
            '{"cpu_sec": 7.5, "pearson": 0.99}\tH\tnice\toptimize\t1\n'
        )
        out = _regenerate_best_state(task_dir)
        assert out is not None and out.exists()
        text = out.read_text()
        assert "Current best state" in text
        assert "20.0% faster" in text
        assert "Patch stack (1 keep" in text
        assert "Concordance" in text  # pearson survives, cpu_sec broken out


# --------------------------------------------------------------------------
# _maybe_reprofile_hint
# --------------------------------------------------------------------------

class TestMaybeReprofileHint:
    def test_no_hint_with_fewer_than_two_keeps(self, task_dir, capsys):
        (task_dir / "results.tsv").write_text(
            _BASELINE_RESULTS
            + "1\tdef5678\ttiny_a\t8.0\t20.0\t500\tkeep\t{}\tH\t\toptimize\t1\n"
        )
        _maybe_reprofile_hint(task_dir, task_dir / "results.tsv")
        assert "[hint]" not in capsys.readouterr().out

    def test_hint_fires_when_crossing_step(self, task_dir, capsys):
        # baseline 10s; two keeps, latest at 5s → 50% cumulative → crosses 20.
        (task_dir / "results.tsv").write_text(
            _BASELINE_RESULTS
            + "1\tdef5678\ttiny_a\t8.0\t20.0\t500\tkeep\t{}\tH\t\toptimize\t1\n"
            + "2\tghi9012\ttiny_a\t5.0\t50.0\t490\tkeep\t{}\tH2\t\toptimize\t1\n"
        )
        _maybe_reprofile_hint(task_dir, task_dir / "results.tsv")
        out = capsys.readouterr().out
        assert "[hint]" in out
        assert "zyme profile" in out
        # the hint threshold file is persisted so it won't refire below 60%.
        assert (task_dir / ".zyme" / "last_reprofile_hint_pct").exists()


# --------------------------------------------------------------------------
# _status_paths — porcelain parsing
# --------------------------------------------------------------------------

class TestStatusPaths:
    def test_parses_changed_paths(self, tmp_path, monkeypatch):
        d = _make_git_task(tmp_path)
        # Edit a tracked file + add an untracked one.
        (d / "task.yaml").write_text(_TASK_YAML + "# edit\n")
        (d / "newfile.txt").write_text("x")
        out = _status_paths(d, ("task.yaml", "newfile.txt"))
        assert "task.yaml" in out


# --------------------------------------------------------------------------
# _cross_task_commits_between — shared-repo sibling-commit detector
# --------------------------------------------------------------------------

class TestCrossTaskCommitsBetween:
    def test_repo_root_task_has_no_siblings(self, tmp_path):
        # When the task IS the git repo root, there can be no sibling tasks,
        # so the helper short-circuits to [].
        d = _make_git_task(tmp_path)
        best = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d,
                            capture_output=True, text=True).stdout.strip()
        # Another commit on top, fully inside the task (= repo root).
        (d / "pipeline" / "run.py").write_text("print('x2')\n")
        _git("add", "-A", cwd=d)
        _git("commit", "-q", "-m", "more", cwd=d)
        assert run_cmd._cross_task_commits_between(d, best) == []

    def test_empty_when_no_intervening_commits(self, tmp_path):
        d = _make_git_task(tmp_path)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d,
                            capture_output=True, text=True).stdout.strip()
        # best == HEAD → nothing between.
        assert run_cmd._cross_task_commits_between(d, head) == []


# --------------------------------------------------------------------------
# cmd_run — git-presence guard
# --------------------------------------------------------------------------

class TestCmdRunGuards:
    def test_non_git_dir_dies(self, task_dir, capsys):
        # task_dir fixture is NOT a git repo.
        with pytest.raises(SystemExit) as ei:
            cmd_run(_run_args(task_dir))
        assert ei.value.code == 1
        assert "not a git repository" in capsys.readouterr().err

    def test_missing_hypothesis_dies(self, tmp_path, capsys):
        d = _make_git_task(tmp_path)
        with pytest.raises(SystemExit):
            cmd_run(_run_args(d, hypothesis=None))
        assert "hypothesis is required" in capsys.readouterr().err

    def test_nothing_to_commit_dies(self, tmp_path, capsys, monkeypatch):
        d = _make_git_task(tmp_path)
        # No pipeline edit → nothing staged.
        monkeypatch.setattr(run_cmd, "run_task", lambda *a, **k: _ok_log())
        with pytest.raises(SystemExit):
            cmd_run(_run_args(d))
        assert "nothing to commit" in capsys.readouterr().err


# --------------------------------------------------------------------------
# cmd_run — happy path with run_task monkeypatched
# --------------------------------------------------------------------------

class TestCmdRunHappyPath:
    def _edit_pipeline(self, d):
        (d / "pipeline" / "run.py").write_text("print('faster')\n")

    def test_fresh_round_writes_pending_row(self, tmp_path, monkeypatch, capsys):
        d = _make_git_task(tmp_path)
        self._edit_pipeline(d)
        monkeypatch.setattr(run_cmd, "run_task", lambda *a, **k: _ok_log())
        cmd_run(_run_args(d, hypothesis="vectorize"))
        out = capsys.readouterr().out
        assert "row(s) logged to results.tsv" in out
        rows = (d / "results.tsv").read_text().splitlines()
        last = rows[-1].split("\t")
        assert last[2] == "tiny_a"  # dataset
        assert last[6] == "pending"  # status
        assert last[8] == "vectorize"  # hypothesis
        # speedup_pct: baseline 10, speed 5 → +50
        assert last[4] == "50.0"

    def test_artifact_snapshot_written(self, tmp_path, monkeypatch, capsys):
        d = _make_git_task(tmp_path)
        self._edit_pipeline(d)
        monkeypatch.setattr(run_cmd, "run_task", lambda *a, **k: _ok_log())
        cmd_run(_run_args(d))
        capsys.readouterr()
        arts = list((d / "artifacts").iterdir())
        assert len(arts) == 1
        # run.log + the copied pipeline file land in the artifact dir.
        names = {p.name for p in arts[0].iterdir()}
        assert "run.log" in names
        assert "run.py" in names

    def test_crash_row_zeroes_payload(self, tmp_path, monkeypatch, capsys):
        d = _make_git_task(tmp_path)
        self._edit_pipeline(d)
        monkeypatch.setattr(run_cmd, "run_task",
                            lambda *a, **k: _crash_log("ValueError: boom"))
        cmd_run(_run_args(d))
        capsys.readouterr()
        last = (d / "results.tsv").read_text().splitlines()[-1].split("\t")
        assert last[6] == "crash"
        assert last[3] == "0.000"   # speed zeroed
        assert last[4] == "0.0"     # speedup zeroed
        assert "boom" in last[9]    # crash msg captured into description

    def test_rerun_does_not_advance_round(self, tmp_path, monkeypatch, capsys):
        d = _make_git_task(tmp_path)
        # First, a real round so a decision exists.
        self._edit_pipeline(d)
        monkeypatch.setattr(run_cmd, "run_task", lambda *a, **k: _ok_log())
        cmd_run(_run_args(d, hypothesis="vectorize"))
        capsys.readouterr()
        # Now a rerun (no hypothesis, no commit).
        cmd_run(_run_args(d, hypothesis=None, rerun=True))
        out = capsys.readouterr().out
        assert "rerun row(s) logged" in out
        last = (d / "results.tsv").read_text().splitlines()[-1].split("\t")
        assert last[6] == "rerun"
        assert "." in last[0]  # round label like "1.1"

    def test_setup_mode_commits_without_results_row(self, tmp_path, monkeypatch,
                                                    capsys):
        d = _make_git_task(tmp_path)
        self._edit_pipeline(d)
        n_rows_before = len((d / "results.tsv").read_text().splitlines())
        cmd_run(_run_args(d, hypothesis=None, setup="wire threads"))
        out = capsys.readouterr().out
        assert "setup commit:" in out
        assert "No pipeline run" in out
        # No new results.tsv row.
        assert len((d / "results.tsv").read_text().splitlines()) == n_rows_before
        # best.ref advanced to the setup commit.
        assert (d / ".zyme" / "best.ref").exists()

    def test_setup_and_rerun_mutually_exclusive(self, tmp_path, capsys):
        d = _make_git_task(tmp_path)
        with pytest.raises(SystemExit):
            cmd_run(_run_args(d, hypothesis=None, setup="x", rerun=True))
        assert "mutually exclusive" in capsys.readouterr().err

    def test_setup_with_positional_hypothesis_dies(self, tmp_path, capsys):
        d = _make_git_task(tmp_path)
        with pytest.raises(SystemExit):
            cmd_run(_run_args(d, hypothesis="h", setup="x"))
        assert "do not also pass a positional hypothesis" in capsys.readouterr().err

    def test_setup_refused_while_pending(self, tmp_path, monkeypatch, capsys):
        d = _make_git_task(tmp_path)
        # Land a pending decision row first.
        self._edit_pipeline(d)
        monkeypatch.setattr(run_cmd, "run_task", lambda *a, **k: _ok_log())
        cmd_run(_run_args(d, hypothesis="vectorize"))
        capsys.readouterr()
        (d / "pipeline" / "run.py").write_text("print('again')\n")
        with pytest.raises(SystemExit):
            cmd_run(_run_args(d, hypothesis=None, setup="wire"))
        assert "pending decision round" in capsys.readouterr().err


# --------------------------------------------------------------------------
# cmd_run — rerun cost banner + safety gate + extra-tiers
# --------------------------------------------------------------------------

_MULTI_TIER_YAML = (
    "target_repo: stub\n"
    "target_function: foo\n"
    "datasets:\n"
    "  - {tier: tiny, name: tiny_a, path: data/t.h5ad}\n"
    "  - {tier: medium, name: medium_a, path: data/m.h5ad}\n"
    "metrics:\n"
    "  - {name: speedup, comparator: gte, threshold: 1.0}\n"
)


class TestCmdRunRerunAndExtraTiers:
    def test_rerun_requires_prior_decision(self, tmp_path, monkeypatch, capsys):
        d = _make_git_task(tmp_path)
        monkeypatch.setattr(run_cmd, "run_task", lambda *a, **k: _ok_log())
        with pytest.raises(SystemExit):
            cmd_run(_run_args(d, hypothesis=None, rerun=True))
        assert "requires a prior decision round" in capsys.readouterr().err

    def test_rerun_with_hypothesis_dies(self, tmp_path, capsys):
        d = _make_git_task(tmp_path)
        with pytest.raises(SystemExit):
            cmd_run(_run_args(d, hypothesis="x", rerun=True))
        assert "do not pass a hypothesis" in capsys.readouterr().err

    def test_n_without_rerun_dies(self, tmp_path, capsys):
        d = _make_git_task(tmp_path)
        with pytest.raises(SystemExit):
            cmd_run(_run_args(d, hypothesis="x", n_reps=3))
        assert "requires --rerun" in capsys.readouterr().err

    def test_rerun_cost_gate_blocks_long_projection(self, tmp_path, monkeypatch,
                                                    capsys):
        # Baseline 1000s × 3 reps → ~50 min projected; the 10-min gate fires.
        d = _make_git_task(tmp_path, with_baseline=False)
        (d / "results.tsv").write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\tthread\n"
            "0\tabc1234\ttiny_a\t1000.0\t0.0\t512.0\tbaseline\t{}\tup\t\toptimize\t1\n"
            "1\tdef5678\ttiny_a\t900.0\t10.0\t500.0\tkeep\t{}\tH\t\toptimize\t1\n"
        )
        monkeypatch.setattr(run_cmd, "run_task", lambda *a, **k: _ok_log())
        with pytest.raises(SystemExit):
            cmd_run(_run_args(d, hypothesis=None, rerun=True, n_reps=3))
        assert "safety gate" in capsys.readouterr().err

    def test_rerun_gate_bypassed_with_yes(self, tmp_path, monkeypatch, capsys):
        d = _make_git_task(tmp_path, with_baseline=False)
        (d / "results.tsv").write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\tthread\n"
            "0\tabc1234\ttiny_a\t1000.0\t0.0\t512.0\tbaseline\t{}\tup\t\toptimize\t1\n"
            "1\tdef5678\ttiny_a\t900.0\t10.0\t500.0\tkeep\t{}\tH\t\toptimize\t1\n"
        )
        monkeypatch.setattr(run_cmd, "run_task", lambda *a, **k: _ok_log())
        # --yes acknowledges the cost; should proceed and write rerun rows.
        cmd_run(_run_args(d, hypothesis=None, rerun=True, n_reps=2, yes=True))
        out = capsys.readouterr().out
        assert "aggregate" in out  # the --n aggregate block prints
        statuses = [ln.split("\t")[6]
                    for ln in (d / "results.tsv").read_text().splitlines()[1:]]
        assert statuses.count("rerun") == 2

    def test_extra_tiers_overlap_with_primary_dies(self, tmp_path, capsys):
        d = _make_git_task(tmp_path)
        (d / "task.yaml").write_text(_MULTI_TIER_YAML)
        # Commit the task-definition change so the task-def guard doesn't fire
        # first; the overlap check is what we want to exercise here.
        _git("add", "-A", cwd=d)
        _git("commit", "-q", "-m", "multi", cwd=d)
        (d / "pipeline" / "run.py").write_text("print('faster')\n")
        with pytest.raises(SystemExit):
            cmd_run(_run_args(d, hypothesis="x", dataset="tiny",
                              extra_tiers="tiny"))
        assert "includes the primary tier" in capsys.readouterr().err

    def test_multi_tier_writes_primary_and_secondary(self, tmp_path, monkeypatch,
                                                     capsys):
        d = _make_git_task(tmp_path)
        (d / "task.yaml").write_text(_MULTI_TIER_YAML)
        (d / "results.tsv").write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\tthread\n"
            "0\tabc1234\ttiny_a\t10.0\t0.0\t512.0\tbaseline\t{}\tup\t\toptimize\t1\n"
            "0\tabc1234\tmedium_a\t20.0\t0.0\t800.0\tbaseline\t{}\tup\t\toptimize\t1\n"
        )
        # commit the new task.yaml + baseline so the tree is clean before edit
        _git("add", "-A", cwd=d)
        _git("commit", "-q", "-m", "multi", cwd=d)
        (d / "pipeline" / "run.py").write_text("print('faster')\n")
        monkeypatch.setattr(run_cmd, "run_task", lambda *a, **k: _ok_log())
        cmd_run(_run_args(d, hypothesis="x", dataset="tiny",
                          extra_tiers="medium"))
        capsys.readouterr()
        rows = [ln.split("\t") for ln
                in (d / "results.tsv").read_text().splitlines()[1:]]
        primary = [r for r in rows if r[2] == "tiny_a" and r[6] == "pending"]
        secondary = [r for r in rows if r[2] == "medium_a" and r[6] == "rerun"]
        assert len(primary) == 1
        assert len(secondary) == 1
        # secondary round label is a `.k` sub-round of the decision round.
        assert "." in secondary[0][0]


# --------------------------------------------------------------------------
# _audit_commit_files — scope violation logging (non-blocking)
# --------------------------------------------------------------------------

class TestAuditCommitFiles:
    def test_fix_loop_violation_logged_and_warned(self, tmp_path, capsys):
        d = _make_git_task(tmp_path)
        # Commit a change that touches a file outside the fix-loop scope.
        (d / "evaluate.py").write_text("print('evil')\n")
        _git("add", "-A", cwd=d)
        _git("commit", "-q", "-m", "touch evaluate", cwd=d)
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d,
                            capture_output=True, text=True).stdout.strip()
        run_cmd._audit_commit_files(d, sha, "fix-loop", "optimize", "hyp")
        out = capsys.readouterr().out
        assert "[audit]" in out
        assert "evaluate.py" in out
        log = (d / ".zyme" / "edit_audit.log")
        assert log.exists()
        assert "evaluate.py" in log.read_text()

    def test_no_violation_no_log(self, tmp_path, capsys):
        d = _make_git_task(tmp_path)
        (d / "pipeline" / "run.py").write_text("print('within scope')\n")
        _git("add", "-A", cwd=d)
        _git("commit", "-q", "-m", "in scope", cwd=d)
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d,
                            capture_output=True, text=True).stdout.strip()
        run_cmd._audit_commit_files(d, sha, "fix-loop", "optimize", "hyp")
        assert "[audit]" not in capsys.readouterr().out
        assert not (d / ".zyme" / "edit_audit.log").exists()


# --------------------------------------------------------------------------
# cmd_dryrun
# --------------------------------------------------------------------------

class TestCmdDryrun:
    def test_dryrun_invokes_dryrun_task(self, tmp_path, monkeypatch, capsys):
        d = _make_git_task(tmp_path)
        seen = {}

        def fake_dryrun(task_dir, dataset_entry=None, **kw):
            seen["tier"] = dataset_entry["tier"] if dataset_entry else None
            return (0, "pipeline ran\n", "", 1.23)

        monkeypatch.setattr(run_cmd, "dryrun_task", fake_dryrun)
        cmd_dryrun(SimpleNamespace(task_dir=str(d), dataset=None))
        out = capsys.readouterr().out
        assert "pipeline ran" in out
        assert "returncode=0" in out
        assert seen["tier"] == "tiny"


# --------------------------------------------------------------------------
# cmd_accept / cmd_reject — driven on a real repo after a real run
# --------------------------------------------------------------------------

def _do_one_round(d, monkeypatch, capsys, log=None):
    (d / "pipeline" / "run.py").write_text("print('faster')\n")
    monkeypatch.setattr(run_cmd, "run_task",
                        lambda *a, **k: (log if log is not None else _ok_log()))
    cmd_run(_run_args(d, hypothesis="vectorize"))
    capsys.readouterr()


class TestCmdAccept:
    def test_no_pending_dies(self, tmp_path, capsys):
        d = _make_git_task(tmp_path)
        with pytest.raises(SystemExit):
            cmd_accept(SimpleNamespace(
                task_dir=str(d), description="x",
                dismiss_housekeeping=False))
        assert "no pending row" in capsys.readouterr().err

    def test_accept_flips_to_keep_and_advances_best(self, tmp_path, monkeypatch,
                                                    capsys):
        d = _make_git_task(tmp_path)
        _do_one_round(d, monkeypatch, capsys)
        cmd_accept(SimpleNamespace(
            task_dir=str(d), description="nice +50%",
            dismiss_housekeeping=False))
        capsys.readouterr()
        last = (d / "results.tsv").read_text().splitlines()[-1].split("\t")
        assert last[6] == "keep"
        assert last[9] == "nice +50%"
        # best.ref now points at HEAD.
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d,
                              capture_output=True, text=True).stdout.strip()
        assert (d / ".zyme" / "best.ref").read_text().strip() == head
        # best_state.md regenerated.
        assert (d / "memory" / "best_state.md").exists()


class TestCmdReject:
    def test_reject_flips_to_discard_and_resets(self, tmp_path, monkeypatch,
                                               capsys):
        d = _make_git_task(tmp_path)
        # Seed a best.ref pointing at the initial commit so reject can reset.
        init_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d,
                                  capture_output=True, text=True).stdout.strip()
        (d / ".zyme" / "best.ref").write_text(init_sha)
        _do_one_round(d, monkeypatch, capsys)
        # After a run the tree is dirty (results.tsv pending row + artifacts).
        # --keep-tree resets HEAD to best with --mixed, preserving the working
        # tree (so the discard flip on results.tsv survives) while moving HEAD.
        cmd_reject(SimpleNamespace(
            task_dir=str(d), description="too slow",
            keep_tree=True, force=False))
        capsys.readouterr()
        # The pending row is flipped to discard and preserved (--mixed).
        statuses = [ln.split("\t")[6]
                    for ln in (d / "results.tsv").read_text().splitlines()[1:]]
        assert "discard" in statuses
        # HEAD reset back to the initial (best) commit.
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d,
                              capture_output=True, text=True).stdout.strip()
        assert head == init_sha

    def test_dirty_tree_without_force_refuses(self, tmp_path, monkeypatch,
                                             capsys):
        # The default `zyme reject` refuses when the working tree is dirty
        # (a run leaves results.tsv / artifacts/ uncommitted) — guards against
        # silently wiping an in-progress fix.
        d = _make_git_task(tmp_path)
        init_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d,
                                  capture_output=True, text=True).stdout.strip()
        (d / ".zyme" / "best.ref").write_text(init_sha)
        _do_one_round(d, monkeypatch, capsys)
        with pytest.raises(SystemExit):
            cmd_reject(SimpleNamespace(
                task_dir=str(d), description="x",
                keep_tree=False, force=False))
        assert "uncommitted changes" in capsys.readouterr().err


# --------------------------------------------------------------------------
# cmd_rollback
# --------------------------------------------------------------------------

class TestCmdRollback:
    def test_no_results_dies(self, task_dir, capsys):
        with pytest.raises(SystemExit):
            cmd_rollback(SimpleNamespace(
                task_dir=str(task_dir), description="", dry_run=False))
        assert "no results.tsv" in capsys.readouterr().err

    def test_no_keep_dies(self, tmp_path, capsys):
        d = _make_git_task(tmp_path)  # only a baseline row, no keep
        with pytest.raises(SystemExit):
            cmd_rollback(SimpleNamespace(
                task_dir=str(d), description="", dry_run=False))
        assert "no `keep` row" in capsys.readouterr().err

    def test_dry_run_no_keep_change(self, tmp_path, monkeypatch, capsys):
        d = _make_git_task(tmp_path)
        _do_one_round(d, monkeypatch, capsys)
        cmd_accept(SimpleNamespace(
            task_dir=str(d), description="k", dismiss_housekeeping=False))
        capsys.readouterr()
        before = (d / "results.tsv").read_text()
        cmd_rollback(SimpleNamespace(
            task_dir=str(d), description="", dry_run=True))
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        assert (d / "results.tsv").read_text() == before  # untouched

    def test_rollback_first_keep_flips_status_removes_best(self, tmp_path,
                                                           monkeypatch, capsys):
        d = _make_git_task(tmp_path)
        _do_one_round(d, monkeypatch, capsys)
        cmd_accept(SimpleNamespace(
            task_dir=str(d), description="k", dismiss_housekeeping=False))
        capsys.readouterr()
        assert (d / ".zyme" / "best.ref").exists()
        cmd_rollback(SimpleNamespace(
            task_dir=str(d), description="regressed at large", dry_run=False))
        capsys.readouterr()
        statuses = [ln.split("\t")[6]
                    for ln in (d / "results.tsv").read_text().splitlines()[1:]]
        assert "rollback" in statuses
        # Only keep was the first → best.ref removed.
        assert not (d / ".zyme" / "best.ref").exists()
        # Annotation appended to description.
        text = (d / "results.tsv").read_text()
        assert "rolled back: regressed at large" in text
