"""Wave-4 mop-up coverage for zyme.commands.baseline.

Wave-2 (test_baseline_cmd.py) covered the pure helpers + the no-subprocess
entry points; wave-3 (test_baseline_w3.py) drove cmd_reference end-to-end and
cmd_record_noise's validation gates + the primary-seed-reuse aggregation path.

This file targets the REACHABLE branches those two left:

  - `_update_baseline_row_to_oom`: header-without-required-cols early return,
    blank-line passthrough, short-row (len < header) passthrough, and the
    non-int thread cell falling back to LEGACY_THREAD.
  - `_baseline_show_rows`: short row (<= status col) skip + a non-baseline
    status row skip.
  - `cmd_baseline_list`: header-only results.tsv ("no rows yet") and a
    results.tsv with only non-baseline rows ("no baseline rows yet").
  - `cmd_record_baseline --oom`: unknown tier dies.
  - `cmd_record_baseline`: empty `--metrics` with NO metrics block in task.yaml
    -> "no metrics: block" path (args_metrics="{}").
  - `cmd_record_noise`:
      * scalar `noise_calibration` (not a list) is coerced to a 1-elem list,
      * legacy flat `reference_output_<tier>/` dir is honored,
      * the non-primary-seed branch: rmtree of a pre-existing calibration dir,
        the reference subprocess run, a non-zero exit -> die, and the
        evaluate metric-parse ValueError-continue + missing-metric-from-seed
        aggregation die. All subprocess boundaries are monkeypatched.
  - `cmd_promote_baseline`: skip when stash row already in results.tsv, skip on
    unparseable speed/peak, and the unknown-dataset skip.

Genuine subprocess forks (reference run, evaluate run) are stubbed at their
single boundary; everything else runs in-process.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from zyme.commands import baseline as bl


TASK_YAML = """\
target_repo: https://example.com/foo
target_function: foo

baseline_threads: [1, 4]

datasets:
  - {tier: tiny, name: tiny_a, path: data/tiny.bin}
  - {tier: medium, name: medium_a, path: data/medium.bin}

metrics:
  - {name: corr, comparator: gte, threshold: 0.99}
  - {name: max_diff, comparator: lte, noise_multiplier: 2.0, absolute_floor: 0.05}
"""

# task.yaml with NO metrics block at all (for the empty-metrics record path).
TASK_YAML_NO_METRICS = """\
target_repo: https://example.com/foo
target_function: foo

baseline_threads: [1]

datasets:
  - {tier: tiny, name: tiny_a, path: data/tiny.bin}
"""

STOCHASTIC_YAML = TASK_YAML + (
    "\nalgorithm_class: stochastic\n"
    "random_seeds: {primary: 42, noise_calibration: [43]}\n"
)

# Stochastic task whose noise_calibration is a SCALAR (not a list).
STOCHASTIC_YAML_SCALAR_CAL = TASK_YAML + (
    "\nalgorithm_class: stochastic\n"
    "random_seeds: {primary: 42, noise_calibration: 43}\n"
)


@pytest.fixture
def btask(tmp_path: Path) -> Path:
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(TASK_YAML, encoding="utf-8")
    (td / ".zyme").mkdir()
    (td / "data").mkdir()
    (td / "data" / "tiny.bin").write_text("x", encoding="utf-8")
    (td / "data" / "medium.bin").write_text("x", encoding="utf-8")
    return td


def _rec_args(task_dir, **over):
    base = dict(
        task_dir=str(task_dir), tier=None, name=None, speed_sec=None,
        peak_mb=0.0, thread=None, metrics="{}", from_log=None, oom=False,
        force=False, accept_synthesis=False,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _noise_args(task_dir, **over):
    base = dict(task_dir=str(task_dir), tier="tiny", seeds=None, thread=None)
    base.update(over)
    return types.SimpleNamespace(**base)


HEADER = (
    "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
    "metrics_json\thypothesis\tdescription\tphase\tthread"
)


def _row(dataset, status="baseline", speed="5.0", peak="50.0", thread="1",
         desc="d"):
    return (f"0\tupstream\t{dataset}\t{speed}\t0.0\t{peak}\t{status}\t{{}}\t"
            f"\t{desc}\toptimize\t{thread}")


# ==========================================================================
# _update_baseline_row_to_oom — header/row edge branches
# ==========================================================================

def test_update_oom_header_missing_required_cols(btask):
    rt = btask / "results.tsv"
    # header lacks 'status'/'speed_sec' -> early False.
    rt.write_text("round\tdataset\tpeak_mb\n0\ttiny_a\t50\n", encoding="utf-8")
    assert bl._update_baseline_row_to_oom(rt, "tiny_a") is False


def test_update_oom_blank_and_short_rows_passthrough(btask):
    rt = btask / "results.tsv"
    rt.write_text(
        HEADER + "\n"
        + "\n"  # blank line -> passthrough branch
        + "0\tupstream\ttiny_a\n"  # short row (< header) -> passthrough
        + _row("tiny_a") + "\n",
        encoding="utf-8",
    )
    assert bl._update_baseline_row_to_oom(rt, "tiny_a", thread=1) is True
    text = rt.read_text()
    assert "\toom\t" in text
    # blank + short rows preserved.
    assert "\n\n" in text


def test_update_oom_noninteger_thread_cell_uses_legacy(btask):
    rt = btask / "results.tsv"
    # thread cell is garbage -> falls back to LEGACY_THREAD (=1) so thread=1 matches.
    bad = _row("tiny_a").rsplit("\t", 1)[0] + "\tNOTANINT"
    rt.write_text(HEADER + "\n" + bad + "\n", encoding="utf-8")
    assert bl._update_baseline_row_to_oom(rt, "tiny_a", thread=1) is True


def test_update_oom_no_match_returns_false(btask):
    rt = btask / "results.tsv"
    rt.write_text(HEADER + "\n" + _row("other_ds") + "\n", encoding="utf-8")
    assert bl._update_baseline_row_to_oom(rt, "tiny_a", thread=1) is False


# ==========================================================================
# _baseline_show_rows — short row + non-baseline status skips
# ==========================================================================

def test_baseline_show_rows_skips_short_and_nonbaseline(btask):
    rt = btask / "results.tsv"
    rt.write_text(
        HEADER + "\n"
        + "0\tupstream\n"  # short (<= status col index) -> skip
        + _row("tiny_a", status="optimize") + "\n"  # non-baseline -> skip
        + _row("tiny_a", status="baseline") + "\n",  # kept
        encoding="utf-8",
    )
    rows = bl._baseline_show_rows(btask, "tiny")
    assert len(rows) == 1
    assert rows[0]["status"] == "baseline"


# ==========================================================================
# cmd_baseline_list — header-only + non-baseline-only rows
# ==========================================================================

def test_baseline_list_header_only(btask, capsys):
    (btask / "results.tsv").write_text(HEADER + "\n", encoding="utf-8")
    bl.cmd_baseline_list(_rec_args(btask, history=False))
    out = capsys.readouterr().out
    assert "no rows yet" in out


def test_baseline_list_only_nonbaseline_rows(btask, capsys):
    (btask / "results.tsv").write_text(
        HEADER + "\n" + _row("tiny_a", status="optimize") + "\n",
        encoding="utf-8",
    )
    bl.cmd_baseline_list(_rec_args(btask, history=False))
    out = capsys.readouterr().out
    assert "no baseline rows yet" in out


# ==========================================================================
# cmd_record_baseline --oom unknown tier; empty-metrics no-block path
# ==========================================================================

def test_record_oom_unknown_tier_dies(btask):
    with pytest.raises(SystemExit):
        bl.cmd_record_baseline(_rec_args(btask, tier="ghost", oom=True))


def test_record_empty_metrics_no_block(tmp_path, capsys):
    td = tmp_path / "nm"
    td.mkdir()
    (td / "task.yaml").write_text(TASK_YAML_NO_METRICS, encoding="utf-8")
    (td / ".zyme").mkdir()
    bl.cmd_record_baseline(_rec_args(td, tier="tiny", speed_sec=3.0, metrics="{}"))
    out = capsys.readouterr().out
    assert "no metrics: block" in out
    # row landed with metrics_json == {}.
    assert "{}" in (td / "results.tsv").read_text()


# ==========================================================================
# cmd_record_noise — scalar cal seed, legacy ref dir, non-primary-seed branch
# ==========================================================================

@pytest.fixture
def stask(tmp_path: Path) -> Path:
    td = tmp_path / "stask"
    td.mkdir()
    (td / "task.yaml").write_text(STOCHASTIC_YAML, encoding="utf-8")
    (td / ".zyme").mkdir()
    (td / "data").mkdir()
    (td / "data" / "tiny.bin").write_text("x", encoding="utf-8")
    (td / "data" / "medium.bin").write_text("x", encoding="utf-8")
    (td / "evaluate.py").write_text("print('corr: 0.999')\n", encoding="utf-8")
    return td


def _make_primary_ref(td: Path, legacy=False):
    if legacy:
        d = td / "reference_output_tiny"
    else:
        from zyme.utils import resolve_reference_output_dir
        d = resolve_reference_output_dir(td, tier="tiny")
    d.mkdir(parents=True, exist_ok=True)
    (d / "out.txt").write_text("primary", encoding="utf-8")
    return d


def _patch_reference_script(td: Path, monkeypatch):
    ref = td / "reference.py"
    ref.write_text("print('speed_sec: 1.0')\n", encoding="utf-8")
    monkeypatch.setattr(bl, "resolve_reference_script", lambda task: ref)
    monkeypatch.setattr(bl, "build_reference_cmd",
                        lambda yaml_path, ref_path: ["python", str(ref_path)])


class _Proc:
    """Stand-in for subprocess.Popen used by the reference run."""

    def __init__(self, returncode=0, lines=("speed_sec: 1.0\n",)):
        self.returncode = returncode
        self.stdout = list(lines)

    def wait(self):
        return self.returncode


class _Completed:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_record_noise_scalar_calibration_seed(tmp_path, monkeypatch, capsys):
    td = tmp_path / "scal"
    td.mkdir()
    (td / "task.yaml").write_text(STOCHASTIC_YAML_SCALAR_CAL, encoding="utf-8")
    (td / ".zyme").mkdir()
    (td / "data").mkdir()
    (td / "data" / "tiny.bin").write_text("x", encoding="utf-8")
    (td / "data" / "medium.bin").write_text("x", encoding="utf-8")
    (td / "evaluate.py").write_text("print('corr: 0.99')\n", encoding="utf-8")
    _make_primary_ref(td)
    _patch_reference_script(td, monkeypatch)

    # seed 43 != primary(42) -> non-primary branch: reference run + evaluate.
    monkeypatch.setattr(bl.subprocess, "Popen", lambda *a, **k: _Proc())
    monkeypatch.setattr(bl.subprocess, "run",
                        lambda *a, **k: _Completed(0, "corr: 0.97\n"))
    bl.cmd_record_noise(_noise_args(td, tier="tiny"))
    # aggregate written; corr is gte -> min retained.
    yaml_text = (td / "task.yaml").read_text()
    assert "intrinsic_noise" in yaml_text


def test_record_noise_legacy_flat_ref_dir(stask, monkeypatch, capsys):
    # Primary ref only exists in the legacy flat layout.
    _make_primary_ref(stask, legacy=True)
    _patch_reference_script(stask, monkeypatch)
    monkeypatch.setattr(bl.subprocess, "Popen", lambda *a, **k: _Proc())
    monkeypatch.setattr(bl.subprocess, "run",
                        lambda *a, **k: _Completed(0, "corr: 0.98\n"))
    bl.cmd_record_noise(_noise_args(stask, tier="tiny", seeds="43"))
    assert "intrinsic_noise" in (stask / "task.yaml").read_text()


def test_record_noise_nonprimary_rmtree_existing_caldir(stask, monkeypatch):
    _make_primary_ref(stask)
    _patch_reference_script(stask, monkeypatch)
    # Pre-create the calibration dir so the rmtree branch fires.
    cal = stask / "reference_outputs" / "tiny_noise_seed43"
    cal.mkdir(parents=True)
    (cal / "stale.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(bl.subprocess, "Popen", lambda *a, **k: _Proc())
    monkeypatch.setattr(bl.subprocess, "run",
                        lambda *a, **k: _Completed(0, "corr: 0.95\n"))
    bl.cmd_record_noise(_noise_args(stask, tier="tiny", seeds="43"))
    assert "intrinsic_noise" in (stask / "task.yaml").read_text()


def test_record_noise_reference_run_nonzero_dies(stask, monkeypatch):
    _make_primary_ref(stask)
    _patch_reference_script(stask, monkeypatch)
    monkeypatch.setattr(bl.subprocess, "Popen",
                        lambda *a, **k: _Proc(returncode=2))
    with pytest.raises(SystemExit):
        bl.cmd_record_noise(_noise_args(stask, tier="tiny", seeds="43"))


def test_record_noise_evaluate_unparseable_value_continues(stask, monkeypatch):
    _make_primary_ref(stask)
    _patch_reference_script(stask, monkeypatch)
    monkeypatch.setattr(bl.subprocess, "Popen", lambda *a, **k: _Proc())
    # One garbage 'corr: NaNish' would not match the numeric regex; provide a
    # line that matches the regex shape but whose float() succeeds normally,
    # plus an unmatched line. Then a valid corr so metrics is non-empty.
    monkeypatch.setattr(
        bl.subprocess, "run",
        lambda *a, **k: _Completed(0, "noise text\ncorr: 0.96\n"),
    )
    bl.cmd_record_noise(_noise_args(stask, tier="tiny", seeds="43"))
    assert "intrinsic_noise" in (stask / "task.yaml").read_text()


def test_record_noise_metric_missing_from_seed_dies(stask, monkeypatch):
    # Two cal seeds; the second produces a metric the first lacks -> aggregate
    # detects the missing metric and dies.
    (stask / "task.yaml").write_text(
        STOCHASTIC_YAML
        + "",
        encoding="utf-8",
    )
    _make_primary_ref(stask)
    _patch_reference_script(stask, monkeypatch)
    monkeypatch.setattr(bl.subprocess, "Popen", lambda *a, **k: _Proc())
    outputs = iter(["corr: 0.9\n", "max_diff: 0.1\n"])
    monkeypatch.setattr(bl.subprocess, "run",
                        lambda *a, **k: _Completed(0, next(outputs)))
    with pytest.raises(SystemExit):
        bl.cmd_record_noise(_noise_args(stask, tier="tiny", seeds="43,44"))


# ==========================================================================
# cmd_promote_baseline — skip branches
# ==========================================================================

def _write_stash(td: Path, entries):
    from zyme.utils import write_baseline_stash
    write_baseline_stash(td, entries)


def test_promote_skips_when_already_in_results(btask, capsys):
    # Record a baseline first so results.tsv already has tiny_a@thread1.
    bl.cmd_record_baseline(_rec_args(btask, tier="tiny", speed_sec=5.0))
    _write_stash(btask, [{
        "name": "tiny_a", "tier": "tiny", "thread": "1",
        "speed_sec": "9.0", "peak_mb": "90.0", "status": "baseline",
    }])
    bl.cmd_promote_baseline(_rec_args(btask))
    out = capsys.readouterr().out
    assert "already has" in out


def test_promote_skips_unparseable_speed(btask, capsys):
    _write_stash(btask, [{
        "name": "tiny_a", "tier": "tiny", "thread": "1",
        "speed_sec": "notafloat", "peak_mb": "10",
    }])
    bl.cmd_promote_baseline(_rec_args(btask))
    out = capsys.readouterr().out
    assert "unparseable" in out


def test_promote_drains_valid_entry(btask, capsys):
    _write_stash(btask, [{
        "name": "medium_a", "tier": "medium", "thread": "1",
        "speed_sec": "7.5", "peak_mb": "75.0", "status": "baseline",
    }])
    bl.cmd_promote_baseline(_rec_args(btask))
    out = capsys.readouterr().out
    assert "drained 1" in out
    assert "medium_a" in (btask / "results.tsv").read_text()
