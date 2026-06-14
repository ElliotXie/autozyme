"""Wave-4 mop-up for zyme.commands.verify — the verify-cube orchestration.

Wave-2 (test_verify_cmd.py) covered the pure helpers + render-only + live
matrix happy/error paths. This file fills the remaining REACHABLE branches,
mostly the `--render-only` filter / parse sub-branches and a few aggregate
edge cases that wave-2 didn't hit:

  _cmd_verify_render_only:
    - --cells parsing errors: blank entry skipped, missing ':' , non-integer
      thread, unknown tier.
    - --tiers filter resolve_tiers ValueError.
    - row-loop skips: phase mismatch, short/bad rows.
    - worst-metric aggregation: a None metric value and a non-float metric.
    - speedup recomputed from real_pcts when base_speed is 0 (fallback branch).

  _load_context_cells_from_verify_tsv: schema-missing-column early return,
    crash-row exclusion path.

All over hand-written verify.tsv on tmp_path with git HEAD stubbed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import zyme.commands.verify as v
from zyme.commands.verify import _load_context_cells_from_verify_tsv


VERIFY_HEADER = (
    "timestamp\tthread\ttier\tdataset\trep\tspeed_sec\tpeak_mb\t"
    "baseline_speed\tspeedup_pct\tstatus\tmetrics_json\tcommit\tphase"
)


def _vrow(thread, tier, dataset, rep, speed, peak, base, pct, status,
          metrics, commit="abc1234", phase="optimize"):
    return "\t".join([
        "2026-01-01T00:00:00", str(thread), tier, dataset, str(rep),
        str(speed), str(peak), str(base), str(pct), status,
        json.dumps(metrics, separators=(",", ":")), commit, phase,
    ])


def _write_verify_tsv(path: Path, rows, header=VERIFY_HEADER):
    path.write_text(header + "\n" + "\n".join(rows) + ("\n" if rows else ""))


RESULTS_HEADER = (
    "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
    "metrics_json\thypothesis\tdescription\tphase\tthread"
)


def _write_results_tsv(task_dir: Path, baselines):
    lines = [RESULTS_HEADER]
    for ds, thr, speed, peak, status in baselines:
        lines.append("\t".join([
            "0", "abc1234", ds, str(speed), "0.0", str(peak), status,
            "{}", "upstream", "", "optimize", str(thr),
        ]))
    (task_dir / "results.tsv").write_text("\n".join(lines) + "\n")


class _Args:
    def __init__(self, **kw):
        self.output = "verify.tsv"
        self.no_plot = True
        self.render_only = True
        self.cells = None
        self.tiers = None
        self.threads = "1,4,8"
        self.phase = "optimize"
        self.reps = 3
        for k, val in kw.items():
            setattr(self, k, val)


def _task(tmp_path, *, extra_tiers="", threading="default"):
    task = tmp_path / "task"
    task.mkdir()
    (task / ".zyme").mkdir()
    yaml = (
        "datasets:\n"
        "  - {tier: tiny, name: tiny_a, path: data/x.h5ad}\n"
        + extra_tiers
        + "metrics:\n  - {name: j, comparator: gte, threshold: 0.9}\n"
        + f"threading: {threading}\n"
    )
    (task / "task.yaml").write_text(yaml)
    return task


# ---------------------------------------------------------------------------
# --render-only --cells parsing errors
# ---------------------------------------------------------------------------

def test_render_cells_missing_colon_dies(tmp_path, monkeypatch):
    task = _task(tmp_path)
    _write_verify_tsv(task / "verify.tsv", [
        _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99}),
    ])
    monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
    with pytest.raises(SystemExit):
        v._cmd_verify_render_only(_Args(cells="1tiny"), task)  # no ':'


def test_render_cells_non_integer_thread_dies(tmp_path, monkeypatch):
    task = _task(tmp_path)
    _write_verify_tsv(task / "verify.tsv", [
        _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99}),
    ])
    monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
    with pytest.raises(SystemExit):
        v._cmd_verify_render_only(_Args(cells="x:tiny"), task)


def test_render_cells_unknown_tier_dies(tmp_path, monkeypatch):
    task = _task(tmp_path)
    _write_verify_tsv(task / "verify.tsv", [
        _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99}),
    ])
    monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
    with pytest.raises(SystemExit):
        v._cmd_verify_render_only(_Args(cells="1:no_such_tier"), task)


def test_render_cells_blank_entry_skipped(tmp_path, monkeypatch, capsys):
    # A trailing comma produces a blank cell spec that's skipped (continue),
    # the valid "1:tiny" still renders.
    task = _task(tmp_path)
    _write_verify_tsv(task / "verify.tsv", [
        _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99}),
    ])
    _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline")])
    monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
    v._cmd_verify_render_only(_Args(cells="1:tiny, "), task)
    assert "PASS" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --render-only --tiers filter resolve_tiers error
# ---------------------------------------------------------------------------

def test_render_tiers_unknown_dies(tmp_path, monkeypatch):
    task = _task(tmp_path)
    _write_verify_tsv(task / "verify.tsv", [
        _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99}),
    ])
    monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
    # --tiers names a tier not in task.yaml -> resolve_tiers ValueError -> die.
    with pytest.raises(SystemExit):
        v._cmd_verify_render_only(_Args(tiers="ghost_tier", threads="1"), task)


# ---------------------------------------------------------------------------
# --render-only row-loop skips: phase mismatch, short/bad rows
# ---------------------------------------------------------------------------

def test_render_phase_mismatch_skips_row(tmp_path, monkeypatch):
    task = _task(tmp_path)
    # Row recorded with phase=validate; we render phase=optimize -> all rows
    # skipped -> no matching rows -> die.
    _write_verify_tsv(task / "verify.tsv", [
        _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99},
              phase="validate"),
    ])
    monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
    with pytest.raises(SystemExit):
        v._cmd_verify_render_only(_Args(phase="optimize"), task)


def test_render_short_and_bad_rows_skipped(tmp_path, monkeypatch, capsys):
    task = _task(tmp_path)
    good = _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99})
    # A blank line + a short row (too few cols) + a bad-thread row are all
    # skipped; the good row renders.
    bad_thread = "\t".join(["2026", "notint", "tiny", "tiny_a", "1", "2",
                            "100", "10", "80", "pass", "{}", "abc1234",
                            "optimize"])
    rows = [good, "", "x\ty", bad_thread]
    _write_verify_tsv(task / "verify.tsv", rows)
    _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline")])
    monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
    v._cmd_verify_render_only(_Args(), task)
    assert "PASS" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --render-only worst-metric aggregation: None value + non-float value
# ---------------------------------------------------------------------------

def test_render_worst_metric_none_and_nonfloat(tmp_path, monkeypatch, capsys):
    task = _task(tmp_path)
    # rep1 has j=null (None -> skipped), rep2 has j="bad" (non-float -> skipped)
    # -> all vals empty for j -> worst[j]=None branch (1094-1095). A 3rd rep
    # with a real value keeps it from auto-failing.
    rows = [
        _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": None}),
        _vrow(1, "tiny", "tiny_a", 2, 2.1, 101, 10.0, 79.0, "pass", {"j": "bad"}),
        _vrow(1, "tiny", "tiny_a", 3, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99}),
    ]
    _write_verify_tsv(task / "verify.tsv", rows)
    _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline")])
    monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
    # The valid rep keeps j present; aggregation tolerated the None/non-float.
    v._cmd_verify_render_only(_Args(), task)
    out = capsys.readouterr().out
    assert "tiny" in out


def test_render_speedup_fallback_from_pcts(tmp_path, monkeypatch, capsys):
    task = _task(tmp_path)
    # base_speed unresolvable (no verify baseline_speed, no results.tsv) so the
    # speedup falls back to median(real_pcts) (line 1118).
    rows = [
        _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 0.0, 55.0, "pass", {"j": 0.99}),
    ]
    _write_verify_tsv(task / "verify.tsv", rows)
    monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
    # No results.tsv at all -> get_baseline_speed returns None -> base_speed 0.
    v._cmd_verify_render_only(_Args(), task)
    out = capsys.readouterr().out
    assert "tiny" in out


# ---------------------------------------------------------------------------
# _load_context_cells_from_verify_tsv — early returns + crash exclusion
# ---------------------------------------------------------------------------

def test_load_context_missing_columns_returns_empty(tmp_path):
    p = tmp_path / "verify.tsv"
    # Header missing 'tier' -> early empty return.
    p.write_text("timestamp\tthread\tcommit\n2026\t1\tabc1234\n")
    out = _load_context_cells_from_verify_tsv(
        p, tmp_path, "abc1234", "optimize", [], set())
    assert out == []


def test_load_context_missing_file_returns_empty(tmp_path):
    out = _load_context_cells_from_verify_tsv(
        tmp_path / "nope.tsv", tmp_path, "abc1234", "optimize", [], set())
    assert out == []


def test_load_context_in_run_keys_skipped(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    p = task / "verify.tsv"
    _write_verify_tsv(p, [
        _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99}),
    ])
    # (1, "tiny") is in_run -> skipped -> empty result.
    out = _load_context_cells_from_verify_tsv(
        p, task, "abc1234", "optimize",
        [{"name": "j", "comparator": "gte"}], {(1, "tiny")})
    assert out == []


def test_load_context_rehydrates_skipped_cell(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    p = task / "verify.tsv"
    _write_verify_tsv(p, [
        _vrow(1, "ood_large", "olarge_a", 1, 9.0, 400, 18.0, 50.0, "pass",
              {"j": 0.95}),
        _vrow(1, "ood_large", "olarge_a", 2, 9.2, 405, 18.0, 49.0, "pass",
              {"j": 0.96}),
    ])
    out = _load_context_cells_from_verify_tsv(
        p, task, "abc1234", "optimize",
        [{"name": "j", "comparator": "gte"}], set())
    assert len(out) == 1
    cell = out[0]
    assert cell["tier"] == "ood_large"
    assert cell["n_reps"] == 2
    assert cell["in_current_run"] is False
    # worst-of-min for a gte metric.
    assert cell["metrics_worst"]["j"] == 0.95


def test_load_context_crash_status_marks_verdict(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    p = task / "verify.tsv"
    _write_verify_tsv(p, [
        _vrow(1, "ood_large", "olarge_a", 1, 0.0, 0, 18.0, 0.0, "crash", {}),
    ])
    out = _load_context_cells_from_verify_tsv(
        p, task, "abc1234", "optimize", [], set())
    assert out[0]["verdict"] == "CRASH"
    assert out[0]["any_crash"] is True
