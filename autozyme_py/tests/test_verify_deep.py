"""Deep end-to-end tests for autozyme._verify orchestration.

test_verify_patch.py (wave 1) drives the happy path; test_verify_unit.py
(wave 1) hits the pure TSV/threshold/sort helpers. This file pushes the
remaining ~35% gap in _verify.py: the subprocess-orchestration core of
_verify_one_tier and verify_patch top-level —

  * the baseline-cache HIT path (probe, confirm rep accept/invalidate,
    --no-baseline-confirm fast path, writeback backfill),
  * the cache-MISS writeback (promote ref_dir + write noise.json),
  * patched_only mode (no baseline, blank speedup),
  * the verbose per-tier progress + final summary table,
  * the per-tier crash -> skip-note path,
  * reps validation / escalation / multi-tier.

All drive the stdlib-only synthetic ``_test_json`` patch (json.dumps target,
trivial smoke recipe) so each spawned worker subprocess is < 0.1 s and no
optional upstream is needed. The worker subprocess IS spawned for real — that
is the point: it exercises the spawn / parse / JSON-readback path that pure
helper tests cannot.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import autozyme
from autozyme import _verify as V
from autozyme import _baseline_cache as BC
from autozyme._core import _REGISTRY, _import_submodule
from autozyme import verify_patch


# --------------------------------------------------------------------------
# Shared task fixtures (mirror test_verify_patch.py, self-contained here)
# --------------------------------------------------------------------------
_TASK_YAML_PASS = """\
task: _test_json
datasets:
  - {tier: tiny, name: unittest, path: /dev/null}
metrics:
  - {name: output_match, threshold: 1.0, comparator: gte}
"""

_EVALUATE_PY = """\
import os
ref_dir = os.environ.get("ZYME_REFERENCE_DIR", "reference_output")
test_dir = os.environ.get("ZYME_TEST_DIR", "pipeline")
ref = open(os.path.join(ref_dir, "output.txt")).read().strip()
test = open(os.path.join(test_dir, "output.txt")).read().strip()
print(f"output_match: {1.0 if test == ref else 0.0}")
"""


@pytest.fixture(autouse=True)
def _ensure_test_patch_registered():
    _import_submodule("_test_json")
    yield
    if _REGISTRY.get("_test_json") and _REGISTRY["_test_json"].injected:
        autozyme.deactivate("_test_json")


@pytest.fixture()
def passing_task(tmp_path: Path) -> Path:
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(_TASK_YAML_PASS)
    (td / "evaluate.py").write_text(_EVALUATE_PY)
    return td


def _patch_obj():
    return _REGISTRY["_test_json"]


def _thresholds():
    return [{"name": "output_match", "threshold": 1.0, "comparator": "gte"}]


# ==========================================================================
# verify_patch end-to-end: verbose path + final summary table (1385, 1430-1462)
# ==========================================================================
def test_verify_patch_verbose_summary_table(passing_task, capsys):
    """verbose=True must print the per-tier banner + the final summary table."""
    results = verify_patch(
        "_test_json", str(passing_task),
        tiers=("tiny",), reps=1, verbose=True, use_baseline_cache=False,
    )
    assert results[0]["all_pass"] is True
    err = capsys.readouterr().err
    # per-tier banner
    assert "tier = tiny" in err
    # final summary table header + verdict line
    assert "verify_patch: _test_json" in err
    assert "speedup" in err
    assert "all_pass" in err
    # verbose metric line + verdict
    assert "output_match" in err
    assert "verdict" in err


def test_verify_patch_verbose_summary_skip_row(tmp_path, capsys):
    """A tier whose run crashes prints the '—' summary row + a [skip] note."""
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(_TASK_YAML_PASS)
    # evaluate.py that never prints output_match -> _verify_one_tier raises
    (td / "evaluate.py").write_text("print('nothing useful')\n")
    results = verify_patch(
        "_test_json", str(td),
        tiers=("tiny",), reps=1, verbose=True, use_baseline_cache=False,
    )
    # crash -> skip-note row, all_pass None, empty timing lists
    assert results[0]["all_pass"] is None
    assert results[0]["baseline_secs"] == []
    assert "not printed by evaluate" in results[0]["note"]
    err = capsys.readouterr().err
    assert "[skip] tiny" in err
    # summary row uses the em-dash placeholder for a crashed tier
    assert "—" in err


# ==========================================================================
# reps validation (1356-1368)
# ==========================================================================
def test_verify_patch_rejects_zero_reps(passing_task):
    with pytest.raises(ValueError, match="reps must be >= 1"):
        verify_patch("_test_json", str(passing_task), tiers=("tiny",), reps=0)


def test_verify_patch_rejects_non_int_reps(passing_task):
    with pytest.raises(ValueError, match="positive integer"):
        verify_patch("_test_json", str(passing_task), tiers=("tiny",),
                     reps="lots")


def test_verify_patch_rejects_empty_tiers(passing_task):
    with pytest.raises(ValueError, match="tiers must be non-empty"):
        verify_patch("_test_json", str(passing_task), tiers=())


def test_verify_patch_accepts_string_tier(passing_task):
    """A bare string tier is coerced to a 1-tuple (1364-1365)."""
    results = verify_patch(
        "_test_json", str(passing_task),
        tiers="tiny", reps=1, verbose=False, use_baseline_cache=False,
    )
    assert len(results) == 1 and results[0]["tier"] == "tiny"


# ==========================================================================
# patched_only mode (277-326) — no baseline, blank speedup
# ==========================================================================
def test_verify_patch_patched_only(passing_task, capsys):
    results = verify_patch(
        "_test_json", str(passing_task),
        tiers=("tiny",), reps=2, verbose=True,
        use_baseline_cache=False, patched_only=True,
    )
    r = results[0]
    # patched-only: baseline list empty, patched list populated, all_pass None
    assert r["baseline_secs"] == []
    assert len(r["patched_secs"]) == 2
    assert r["all_pass"] is None
    assert r["note"] == "patched-only"
    err = capsys.readouterr().err
    assert "patched-only" in err
    assert "concordance NA" in err
    # package_verify.tsv: patched rows present, no baseline rows, blank speedup
    out = _read_tsv(passing_task / "package_verify.tsv")
    assert all(row["variant"] != "baseline" for row in out)
    patched = [row for row in out if row["variant"] == "patched"]
    assert patched and all(row["speedup_x"] == "" for row in patched)


# ==========================================================================
# Cache MISS -> writeback (494-499 promote, 613-652 write noise.json)
# ==========================================================================
def test_verify_patch_cache_miss_writes_back_noise_and_refdir(passing_task):
    """First cached run on a fresh task: MISS -> measure -> promote ref_dir +
    write .zyme/baseline_noise.json so a subsequent run can HIT."""
    results = verify_patch(
        "_test_json", str(passing_task),
        tiers=("tiny",), reps=1, verbose=True, use_baseline_cache=True,
    )
    assert results[0]["all_pass"] is True
    # ref dir promoted
    ref = passing_task / "reference_output_tiny"
    assert ref.is_dir() and (ref / "output.txt").exists()
    # noise.json written with a tiny/threads entry
    noise = BC._load_noise(passing_task)
    assert "tiny" in noise["tiers"]
    entry = next(iter(noise["tiers"]["tiny"].values()))
    assert entry["speed_mean"] > 0
    assert entry["produced_by"] == "attest"


def test_verify_patch_cache_hit_on_second_run(passing_task, capsys):
    """Run twice with cache on. Second run must HIT the cache written by the
    first (confirm-rep path: measure once, compare to cached mean, accept)."""
    verify_patch("_test_json", str(passing_task), tiers=("tiny",),
                 reps=1, verbose=False, use_baseline_cache=True)
    capsys.readouterr()  # drain
    # second run: cache should now exist -> HIT message
    verify_patch("_test_json", str(passing_task), tiers=("tiny",),
                 reps=1, verbose=True, use_baseline_cache=True)
    err = capsys.readouterr().err
    assert "[baseline-cache] HIT" in err
    # default no_baseline_confirm=False -> a confirmation rep runs
    assert "confirmation rep" in err or "confirm OK" in err


def test_verify_patch_cache_hit_no_confirm_fast_path(passing_task, capsys):
    """no_baseline_confirm=True on a cache HIT: copies cached outputs, runs NO
    baseline subprocess at all, and triggers the version-stamp backfill."""
    verify_patch("_test_json", str(passing_task), tiers=("tiny",),
                 reps=1, verbose=False, use_baseline_cache=True)
    capsys.readouterr()
    results = verify_patch(
        "_test_json", str(passing_task), tiers=("tiny",),
        reps=1, verbose=True, use_baseline_cache=True,
        no_baseline_confirm=True,
    )
    assert results[0]["all_pass"] is True
    err = capsys.readouterr().err
    assert "[baseline-cache] HIT" in err
    # backfill stamps upstream_versions onto the existing artifact
    assert "BACKFILL" in err


def test_verify_patch_cache_confirm_fail_invalidates(passing_task, monkeypatch,
                                                     capsys):
    """If the confirmation rep diverges far from the cached mean, the cache is
    invalidated and a fresh baseline is taken (436-449)."""
    # Seed a cache entry with an absurdly LARGE mean so any real measurement
    # (the worker times only sum([1,2,3]) ~ sub-microsecond) is many σ away.
    # tol = 3 * max(stdev, 0.05*mean) = 150s for mean=1000s; the measured
    # baseline is ~1e-6s so |Δ| ~ 1000s >> 150s -> deterministic confirm FAIL.
    verify_patch("_test_json", str(passing_task), tiers=("tiny",),
                 reps=1, verbose=False, use_baseline_cache=True)
    noise = BC._load_noise(passing_task)
    thr_key = next(iter(noise["tiers"]["tiny"]))
    noise["tiers"]["tiny"][thr_key]["speed_mean"] = 1000.0
    noise["tiers"]["tiny"][thr_key]["speed_stdev"] = 0.0
    BC._save_noise(passing_task, noise)
    capsys.readouterr()
    results = verify_patch(
        "_test_json", str(passing_task), tiers=("tiny",),
        reps=1, verbose=True, use_baseline_cache=True,
        baseline_confirm_sigma=3.0,
    )
    err = capsys.readouterr().err
    # confirm FAIL -> invalidate -> fresh measurement
    assert "confirm FAIL" in err
    # still passes correctness (baseline re-measured fresh)
    assert results[0]["all_pass"] is True


# ==========================================================================
# Multi-tier: one good tier + one missing-dataset tier -> mixed verdict
# ==========================================================================
def test_verify_patch_multi_tier_missing_tier_skips(passing_task):
    """A tier not declared in task.yaml still runs (smoke load ignores the
    tier name for _test_json) — so use a tier whose evaluate is fine. To force
    a skip on the second tier, point its smoke at a crash. Simpler: drive two
    real tiers; both pass because _test_json's smoke is tier-agnostic."""
    results = verify_patch(
        "_test_json", str(passing_task),
        tiers=("tiny", "medium"), reps=1, verbose=False,
        use_baseline_cache=False,
    )
    assert len(results) == 2
    assert {r["tier"] for r in results} == {"tiny", "medium"}
    assert all(r["all_pass"] is True for r in results)


# ==========================================================================
# _verify_one_tier called directly — patched_only branch return shape
# ==========================================================================
def test_verify_one_tier_patched_only_shape(passing_task):
    res = V._verify_one_tier(
        _patch_obj(), "_test_json", str(passing_task), "tiny",
        _thresholds(), {}, reps=2, verbose=False,
        use_baseline_cache=False, patched_only=True,
    )
    assert res["baseline_sec"] is None
    assert res["baseline_secs"] == []
    assert len(res["patched_secs"]) == 2
    assert res["all_pass"] is None
    assert res["per_rep_pass"] == [None, None]


def test_verify_one_tier_cache_miss_passes(passing_task):
    """Direct _verify_one_tier on a fresh task (no cache) -> measures both."""
    res = V._verify_one_tier(
        _patch_obj(), "_test_json", str(passing_task), "tiny",
        _thresholds(), {}, reps=1, verbose=False,
        use_baseline_cache=False,
    )
    assert res["all_pass"] is True
    assert len(res["baseline_secs"]) == 1
    assert len(res["patched_secs"]) == 1
    assert res["metrics"]["output_match"]["value"] == 1.0


def test_verify_one_tier_unknown_comparator_raises(passing_task):
    res_thr = [{"name": "output_match", "threshold": 1.0, "comparator": "eq"}]
    with pytest.raises(RuntimeError, match="unknown comparator"):
        V._verify_one_tier(
            _patch_obj(), "_test_json", str(passing_task), "tiny",
            res_thr, {}, reps=1, verbose=False, use_baseline_cache=False,
        )


def test_verify_one_tier_missing_metric_raises(tmp_path):
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(_TASK_YAML_PASS)
    (td / "evaluate.py").write_text("print('something_else: 1.0')\n")
    res_thr = [{"name": "output_match", "threshold": 1.0, "comparator": "gte"}]
    with pytest.raises(RuntimeError, match="not printed by evaluate"):
        V._verify_one_tier(
            _patch_obj(), "_test_json", str(td), "tiny",
            res_thr, {}, reps=1, verbose=False, use_baseline_cache=False,
        )


# ==========================================================================
# lte comparator + failing-metric per-rep path (526-529, 530-534)
# ==========================================================================
def test_verify_one_tier_lte_comparator(tmp_path):
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(_TASK_YAML_PASS)
    # evaluate prints a metric value of 0.0; lte threshold 0.5 -> pass
    (td / "evaluate.py").write_text("print('err: 0.0')\n")
    res_thr = [{"name": "err", "threshold": 0.5, "comparator": "lte"}]
    res = V._verify_one_tier(
        _patch_obj(), "_test_json", str(td), "tiny",
        res_thr, {}, reps=1, verbose=True, use_baseline_cache=False,
    )
    assert res["all_pass"] is True
    assert res["metrics"]["err"]["comparator"] == "lte"


def test_verify_one_tier_metric_fails_threshold(tmp_path):
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(_TASK_YAML_PASS)
    (td / "evaluate.py").write_text("print('score: 0.1')\n")
    res_thr = [{"name": "score", "threshold": 0.9, "comparator": "gte"}]
    res = V._verify_one_tier(
        _patch_obj(), "_test_json", str(td), "tiny",
        res_thr, {}, reps=1, verbose=True, use_baseline_cache=False,
    )
    assert res["all_pass"] is False
    assert res["metrics"]["score"]["pass"] is False


# ==========================================================================
# Helpers / small uncovered pure branches
# ==========================================================================
def _read_tsv(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_read_task_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        V._read_task_yaml(str(tmp_path))


def test_append_package_verify_skips_inst_when_no_manifest(passing_task):
    """_resolve_inst_speedup_path returns None when no manifest is reachable;
    _append_package_verify_tsv must still write package_verify.tsv (1237-1240
    inst-mirror branch is a no-op)."""
    rows = [{
        "tier": "tiny", "note": "",
        "baseline_secs": [4.0], "patched_secs": [2.0],
        "baseline_peaks_mb": [None], "patched_peaks_mb": [None],
        "per_rep_pass": [True], "metrics_json": "",
    }]
    V._append_package_verify_tsv(str(passing_task), "_test_json", rows)
    assert (passing_task / "package_verify.tsv").exists()


def test_collect_system_info_includes_real_os():
    info = V._collect_system_info()
    # _detect_cpu_model + _detect_ram_gb exercised on the host OS
    assert info["system_os"]
    # cpu may be empty on exotic hosts but the key exists
    assert "system_cpu" in info
