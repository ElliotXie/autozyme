"""Wave-3 mop-up coverage for zyme.commands.baseline.

Wave-2 (tests/commands/test_baseline_cmd.py) covered the pure helpers and the
no-subprocess entry points. This file targets the REACHABLE in-process
decision/aggregation branches wave-2 left:

  - `_fingerprint_or_die` synthesis-bypass + no-script-yet branches
  - `cmd_reference` end-to-end with the subprocess boundary
    (`subprocess.Popen` + `parse_log`) monkeypatched, exercising the
    single-rep and multi-rep noise-aggregation math (mean/CV/record_tier_noise)
    plus the OOM-detection and missing-speed failure branches
  - `cmd_record_noise` validation gates (tier missing, primary ref missing,
    no evaluate script) and the primary-seed-reuse end-to-end path that runs
    the gte/lte worst-per-metric aggregation with `subprocess.run` stubbed.

The genuine subprocess forks (the reference run itself, the evaluate run for a
non-primary seed) are stubbed at their one boundary; everything else runs.
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

STOCHASTIC_YAML = TASK_YAML + (
    "\nalgorithm_class: stochastic\n"
    "random_seeds: {primary: 42, noise_calibration: [42, 43]}\n"
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


def _ref_args(task_dir, **over):
    base = dict(
        task_dir=str(task_dir), tier="tiny", thread=None, reps=1,
        accept_synthesis=False,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# _fingerprint_or_die
# ---------------------------------------------------------------------------

def test_fingerprint_or_die_no_script_is_noop(btask):
    # No reference.{py,R} present -> nothing to scan, returns cleanly.
    bl._fingerprint_or_die(btask, accept_synthesis=False)


def test_fingerprint_or_die_synthesis_bypass(btask, capsys):
    (btask / "task.yaml").write_text(
        TASK_YAML + "\nsynthesis: hand-built tiny fixture\n", encoding="utf-8")
    (btask / "reference.py").write_text("# would-fail fingerprint\n", encoding="utf-8")
    bl._fingerprint_or_die(btask, accept_synthesis=True)
    assert "check bypassed via --accept-synthesis" in capsys.readouterr().out


def test_fingerprint_or_die_violation_dies(btask, monkeypatch):
    (btask / "reference.py").write_text("# ref\n", encoding="utf-8")

    def boom(_p):
        raise bl.FingerprintViolation("synthetic data smell")
    monkeypatch.setattr(bl, "check_reference_fingerprint", boom)
    with pytest.raises(SystemExit):
        bl._fingerprint_or_die(btask, accept_synthesis=False)


# ---------------------------------------------------------------------------
# cmd_reference — driven end-to-end with the subprocess boundary stubbed.
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, lines, rc=0):
        self.stdout = iter(lines)
        self.returncode = rc

    def wait(self):
        return self.returncode


def _patch_reference_boundary(monkeypatch, *, lines, rc=0,
                              parsed=(5.0, 120.0)):
    """Stub build_reference_cmd, Popen, and parse_log."""
    monkeypatch.setattr(bl, "build_reference_cmd",
                        lambda *_a, **_k: ["true"])
    monkeypatch.setattr(bl.subprocess, "Popen",
                        lambda *a, **k: _FakeProc(lines, rc))
    speed, peak = parsed
    monkeypatch.setattr(bl, "parse_log",
                        lambda log: (speed, peak, None, None))


def test_reference_single_rep_records_baseline(btask, monkeypatch, capsys):
    (btask / "reference.py").write_text("# ref\n", encoding="utf-8")
    _patch_reference_boundary(
        monkeypatch, lines=["speed_sec: 5.0\n", "peak_mb: 120.0\n"],
        parsed=(5.0, 120.0))
    bl.cmd_reference(_ref_args(btask, tier="tiny", reps=1))
    results = (btask / "results.tsv").read_text(encoding="utf-8")
    assert "tiny_a" in results and "baseline" in results
    assert "5.000" in results
    # single-rep path does not emit the noise-calibration banner.
    assert "noise calibration" not in capsys.readouterr().out


def test_reference_multi_rep_aggregates_and_persists_noise(btask, monkeypatch,
                                                           capsys):
    (btask / "reference.py").write_text("# ref\n", encoding="utf-8")
    # parse_log returns increasing speeds across reps; mean should be recorded.
    seq = iter([(4.0, 100.0), (6.0, 140.0), (5.0, 120.0)])
    monkeypatch.setattr(bl, "build_reference_cmd", lambda *_a, **_k: ["true"])
    monkeypatch.setattr(bl.subprocess, "Popen",
                        lambda *a, **k: _FakeProc(["line\n"], 0))
    monkeypatch.setattr(bl, "parse_log",
                        lambda log: (*next(seq), None, None))
    recorded = {}

    def fake_record(**kw):
        recorded.update(kw)
        return {"speed_cv": 0.1633, "n_reps": 3}
    monkeypatch.setattr(bl, "record_tier_noise", fake_record)

    bl.cmd_reference(_ref_args(btask, tier="tiny", reps=3, thread=4))
    out = capsys.readouterr().out
    assert "noise calibration (3 reps)" in out
    assert "calibration saved" in out
    # mean of 4,6,5 = 5.0 recorded as the baseline.
    results = (btask / "results.tsv").read_text(encoding="utf-8")
    row = [ln for ln in results.splitlines() if "tiny_a" in ln][0]
    assert "5.000" in row
    assert row.split("\t")[-1] == "4"  # thread
    # record_tier_noise got all three speed samples.
    assert recorded["speeds"] == [4.0, 6.0, 5.0]
    assert recorded["thread"] == 4


def test_reference_unknown_tier_dies(btask, monkeypatch):
    (btask / "reference.py").write_text("# ref\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        bl.cmd_reference(_ref_args(btask, tier="ghost"))


def test_reference_missing_script_dies(btask):
    with pytest.raises(SystemExit):
        bl.cmd_reference(_ref_args(btask, tier="tiny"))


def test_reference_build_cmd_runtime_error_dies(btask, monkeypatch):
    (btask / "reference.py").write_text("# ref\n", encoding="utf-8")

    def boom(*_a, **_k):
        raise RuntimeError("no interpreter resolved")
    monkeypatch.setattr(bl, "build_reference_cmd", boom)
    with pytest.raises(SystemExit):
        bl.cmd_reference(_ref_args(btask, tier="tiny"))


def test_reference_nonzero_oom_detected_dies(btask, monkeypatch, capsys):
    (btask / "reference.py").write_text("# ref\n", encoding="utf-8")
    _patch_reference_boundary(
        monkeypatch,
        lines=["working...\n", "cannot allocate vector of size 9 Gb\n"],
        rc=137)
    with pytest.raises(SystemExit):
        bl.cmd_reference(_ref_args(btask, tier="tiny", thread=4))
    # OOM-specific advice is surfaced on stderr (die()).
    assert "OOM detected" in capsys.readouterr().err
    # No baseline row recorded.
    assert not (btask / "results.tsv").exists()


def test_reference_nonzero_non_oom_dies(btask, monkeypatch):
    (btask / "reference.py").write_text("# ref\n", encoding="utf-8")
    _patch_reference_boundary(
        monkeypatch, lines=["ValueError: bad shape\n"], rc=1)
    with pytest.raises(SystemExit):
        bl.cmd_reference(_ref_args(btask, tier="tiny"))


def test_reference_no_speed_in_log_dies(btask, monkeypatch):
    (btask / "reference.py").write_text("# ref\n", encoding="utf-8")
    _patch_reference_boundary(
        monkeypatch, lines=["done\n"], parsed=(None, None))
    with pytest.raises(SystemExit):
        bl.cmd_reference(_ref_args(btask, tier="tiny"))


def test_reference_no_peak_treated_as_zero(btask, monkeypatch, capsys):
    (btask / "reference.py").write_text("# ref\n", encoding="utf-8")
    _patch_reference_boundary(
        monkeypatch, lines=["speed_sec: 7.0\n"], parsed=(7.0, None))
    bl.cmd_reference(_ref_args(btask, tier="tiny"))
    assert "no peak_mb in stdout" in capsys.readouterr().out
    results = (btask / "results.tsv").read_text(encoding="utf-8")
    assert "7.000" in results


# ---------------------------------------------------------------------------
# cmd_record_noise — validation gates + primary-seed-reuse aggregation.
# ---------------------------------------------------------------------------

def _noise_args(task_dir, **over):
    base = dict(task_dir=str(task_dir), tier="tiny", thread=None, seeds=None)
    base.update(over)
    return types.SimpleNamespace(**base)


@pytest.fixture
def stask(tmp_path: Path) -> Path:
    td = tmp_path / "stask"
    td.mkdir()
    (td / "task.yaml").write_text(STOCHASTIC_YAML, encoding="utf-8")
    (td / ".zyme").mkdir()
    (td / "data").mkdir()
    (td / "data" / "tiny.bin").write_text("x", encoding="utf-8")
    (td / "data" / "medium.bin").write_text("x", encoding="utf-8")
    (td / "reference.py").write_text("# ref\n", encoding="utf-8")
    return td


def test_record_noise_unknown_tier_dies(stask, monkeypatch):
    monkeypatch.setattr(bl, "build_reference_cmd", lambda *_a, **_k: ["true"])
    with pytest.raises(SystemExit):
        bl.cmd_record_noise(_noise_args(stask, tier="ghost"))


def test_record_noise_primary_ref_missing_dies(stask, monkeypatch):
    monkeypatch.setattr(bl, "build_reference_cmd", lambda *_a, **_k: ["true"])
    # No reference output dir for tiny -> dies before any subprocess.
    with pytest.raises(SystemExit):
        bl.cmd_record_noise(_noise_args(stask, tier="tiny"))


def _make_primary_ref(stask):
    # cmd_record_noise resolves the primary ref dir to the flat legacy form
    # `reference_output_<tier>/` when the task carries no modes block.
    refdir = stask / "reference_output_tiny"
    refdir.mkdir(exist_ok=True)
    (refdir / "out.txt").write_text("ref", encoding="utf-8")
    return refdir


def test_record_noise_no_evaluate_script_dies(stask, monkeypatch):
    monkeypatch.setattr(bl, "build_reference_cmd", lambda *_a, **_k: ["true"])
    _make_primary_ref(stask)
    with pytest.raises(SystemExit):
        bl.cmd_record_noise(_noise_args(stask, tier="tiny"))


def test_record_noise_primary_seed_reuse_aggregates(stask, monkeypatch, capsys):
    """Only the primary seed in cal_seeds -> no reference re-run; evaluate runs
    once and the gte/lte worst-per-metric aggregation is exercised."""
    monkeypatch.setattr(bl, "build_reference_cmd", lambda *_a, **_k: ["true"])
    _make_primary_ref(stask)
    (stask / "evaluate.py").write_text("# eval\n", encoding="utf-8")

    captured = {}

    def fake_run(eval_cmd, env=None, cwd=None, capture_output=True, text=True):
        captured["env"] = env
        return types.SimpleNamespace(
            returncode=0,
            stdout="corr: 0.995\nmax_diff: 0.0010\nirrelevant: 7\n",
            stderr="",
        )
    monkeypatch.setattr(bl.subprocess, "run", fake_run)
    written = {}
    monkeypatch.setattr(
        bl, "write_intrinsic_noise",
        lambda yaml_path, tier, agg: written.update({"tier": tier, "agg": agg}))

    # seeds=42 == primary -> the reuse branch + single seed in aggregation.
    bl.cmd_record_noise(_noise_args(stask, tier="tiny", seeds="42"))
    out = capsys.readouterr().out
    assert "reusing" in out
    assert written["tier"] == "tiny"
    # gte metric -> min; lte metric -> max. Single seed so both equal the value.
    assert written["agg"]["corr"] == pytest.approx(0.995)
    assert written["agg"]["max_diff"] == pytest.approx(0.0010)
    # 'irrelevant' (not in task.yaml metrics) was filtered out.
    assert "irrelevant" not in written["agg"]
    # evaluate got the ZYME_TEST_DIR / ZYME_REFERENCE_DIR wiring.
    assert "ZYME_TEST_DIR" in captured["env"]


def test_record_noise_evaluate_no_parseable_metrics_dies(stask, monkeypatch):
    monkeypatch.setattr(bl, "build_reference_cmd", lambda *_a, **_k: ["true"])
    _make_primary_ref(stask)
    (stask / "evaluate.py").write_text("# eval\n", encoding="utf-8")
    monkeypatch.setattr(
        bl.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(
            returncode=0, stdout="no metrics here\n", stderr=""))
    with pytest.raises(SystemExit):
        bl.cmd_record_noise(_noise_args(stask, tier="tiny", seeds="42"))


def test_record_noise_evaluate_metrics_no_match_dies(stask, monkeypatch):
    monkeypatch.setattr(bl, "build_reference_cmd", lambda *_a, **_k: ["true"])
    _make_primary_ref(stask)
    (stask / "evaluate.py").write_text("# eval\n", encoding="utf-8")
    # parseable, but none of these names are declared in task.yaml.
    monkeypatch.setattr(
        bl.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(
            returncode=0, stdout="other_metric: 0.5\n", stderr=""))
    with pytest.raises(SystemExit):
        bl.cmd_record_noise(_noise_args(stask, tier="tiny", seeds="42"))


def test_record_noise_evaluate_nonzero_dies(stask, monkeypatch):
    monkeypatch.setattr(bl, "build_reference_cmd", lambda *_a, **_k: ["true"])
    _make_primary_ref(stask)
    (stask / "evaluate.py").write_text("# eval\n", encoding="utf-8")
    monkeypatch.setattr(
        bl.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(
            returncode=2, stdout="", stderr="evaluate blew up"))
    with pytest.raises(SystemExit):
        bl.cmd_record_noise(_noise_args(stask, tier="tiny", seeds="42"))
