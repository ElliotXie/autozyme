"""Unit tests for zyme.commands.verify — the verify-cube orchestration.

verify.py drives the (tier, threads) verify matrix: re-times pipeline/run,
applies pass/scaling-tax verdicts, and persists verify.tsv. Three test layers:

  1. Pure parse/aggregate/decision helpers (no subprocess, no git):
     verify.tsv parsing (_read_existing_verify_reps, _load_context_cells,
     _strip_crash_rows_for_topup, _migrate_verify_tsv_add_phase), the
     scaling-tax grader (_compute_scaling_tax), per-rep status classification
     (_per_rep_verify_status), tolerant float / metrics-json parsing, ram-floor
     / mem-cap resolution, peak_mb extrapolation, the plain-text summary
     fallback, and the probe-cache helpers. These are fully deterministic.

  2. The exclusive-lock helpers (_acquire_verify_lock / _release_verify_lock)
     and the RAM pre-flight guard (_verify_ram_preflight) — small filesystem +
     monkeypatched-resource probes.

  3. The two command entry points (_cmd_verify_render_only end to end from a
     hand-written verify.tsv, and cmd_verify driving the live matrix) with the
     single subprocess boundary (verify.run_task) and git HEAD monkeypatched,
     so the env-assembly / aggregation / verdict / TSV-append logic runs
     without launching anything. die()/sys.exit are caught via SystemExit.

All `die*` helpers call sys.exit; we assert on SystemExit. No real subprocess,
no network; everything is tmp_path-scoped.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import zyme.commands.verify as v
from zyme.commands.verify import (
    _acquire_verify_lock,
    _compute_scaling_tax,
    _dataset_size_mb,
    _estimate_tier_peak_mb,
    _load_context_cells_from_verify_tsv,
    _migrate_verify_tsv_add_phase,
    _parse_metrics_json_field,
    _per_rep_verify_status,
    _read_existing_verify_reps,
    _release_verify_lock,
    _resolve_verify_mem_cap,
    _resolve_verify_ram_floor,
    _strip_crash_rows_for_topup,
    _tolerant_float,
    _verify_probe_cache_has,
    _verify_probe_cache_path,
    _verify_probe_cache_write,
    _verify_ram_preflight,
    _write_verify_summary_txt,
)


# Canonical verify.tsv header (matches the live writer in _cmd_verify_body).
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


# Results.tsv K2 schema (thread column present) so the baseline getters resolve.
RESULTS_HEADER = (
    "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
    "metrics_json\thypothesis\tdescription\tphase\tthread"
)


def _write_results_tsv(task_dir: Path, baselines):
    """baselines: list of (dataset, thread, speed_sec, peak_mb, status)."""
    lines = [RESULTS_HEADER]
    for ds, thr, speed, peak, status in baselines:
        lines.append("\t".join([
            "0", "abc1234", ds, str(speed), "0.0", str(peak), status,
            "{}", "upstream", "", "optimize", str(thr),
        ]))
    (task_dir / "results.tsv").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# _tolerant_float
# --------------------------------------------------------------------------

class TestTolerantFloat:
    @pytest.mark.parametrize("raw,expected", [
        ("3.5", 3.5),
        ("0", 0.0),
        ("-2.25", -2.25),
        ("  7 ", 7.0),
    ])
    def test_real_values(self, raw, expected):
        assert _tolerant_float(raw) == expected

    @pytest.mark.parametrize("raw", ["NA", "n/a", "nan", "none", "null", "", "  "])
    def test_sentinels_become_default(self, raw):
        assert _tolerant_float(raw) == 0.0

    def test_none_becomes_default(self):
        assert _tolerant_float(None) == 0.0

    def test_custom_default(self):
        assert _tolerant_float("NA", default=-1.0) == -1.0

    def test_unparseable_becomes_default(self):
        assert _tolerant_float("twelve") == 0.0

    def test_literal_nan_float_rejected(self):
        # float("nan") != itself -> caught by the v != v guard.
        assert _tolerant_float(float("nan")) == 0.0


# --------------------------------------------------------------------------
# _parse_metrics_json_field
# --------------------------------------------------------------------------

class TestParseMetricsJsonField:
    def test_plain_json(self):
        assert _parse_metrics_json_field('{"jaccard":0.95}') == {"jaccard": 0.95}

    def test_empty_returns_empty(self):
        assert _parse_metrics_json_field("") == {}
        assert _parse_metrics_json_field("   ") == {}

    def test_csv_escaped_doubled_quotes(self):
        raw = '"{""jaccard"":0.95,""x"":1}"'
        assert _parse_metrics_json_field(raw) == {"jaccard": 0.95, "x": 1}

    def test_malformed_returns_empty(self):
        assert _parse_metrics_json_field("{not json") == {}

    def test_non_object_json_normalized_to_empty_dict(self):
        # B16 fix: a valid-but-non-object JSON value (bare scalar / array) is
        # normalized to {} so callers doing `.get(...)` never hit AttributeError.
        assert _parse_metrics_json_field('"plain"') == {}
        assert _parse_metrics_json_field("[1,2,3]") == {}
        assert _parse_metrics_json_field("42") == {}
        # A real object still parses through unchanged.
        assert _parse_metrics_json_field('{"jaccard":0.95}') == {"jaccard": 0.95}


# --------------------------------------------------------------------------
# _read_existing_verify_reps
# --------------------------------------------------------------------------

class TestReadExistingVerifyReps:
    def test_missing_file_returns_empty(self, tmp_path):
        assert _read_existing_verify_reps(tmp_path / "nope.tsv", "abc1234") == {}

    def test_old_schema_no_commit_col_returns_empty(self, tmp_path):
        p = tmp_path / "verify.tsv"
        p.write_text("timestamp\tthread\ttier\n2026\t1\ttiny\n")
        assert _read_existing_verify_reps(p, "abc1234") == {}

    def test_filters_by_commit(self, tmp_path):
        p = tmp_path / "verify.tsv"
        _write_verify_tsv(p, [
            _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass",
                  {"j": 0.99}, commit="abc1234"),
            _vrow(1, "tiny", "tiny_a", 2, 2.1, 101, 10.0, 79.0, "pass",
                  {"j": 0.98}, commit="zzz9999"),
        ])
        out = _read_existing_verify_reps(p, "abc1234")
        assert list(out.keys()) == [(1, "tiny")]
        assert len(out[(1, "tiny")]) == 1
        assert out[(1, "tiny")][0]["rep_idx"] == 1
        assert out[(1, "tiny")][0]["metrics"] == {"j": 0.99}

    def test_crash_rows_filtered_out(self, tmp_path):
        p = tmp_path / "verify.tsv"
        _write_verify_tsv(p, [
            _vrow(1, "tiny", "tiny_a", 1, 0, 0, 10.0, 0.0, "crash", {}),
            _vrow(1, "tiny", "tiny_a", 2, 2.0, 100, 10.0, 80.0, "pass", {"j": 1}),
        ])
        out = _read_existing_verify_reps(p, "abc1234")
        # crash rep is dropped; only the pass rep remains.
        assert len(out[(1, "tiny")]) == 1
        assert out[(1, "tiny")][0]["status"] == "pass"

    def test_oom_rows_kept(self, tmp_path):
        p = tmp_path / "verify.tsv"
        _write_verify_tsv(p, [
            _vrow(1, "ood_large", "olarge", 1, 0, 0, 0, 0.0, "oom", {}),
        ])
        out = _read_existing_verify_reps(p, "abc1234")
        assert out[(1, "ood_large")][0]["status"] == "oom"

    def test_phase_filter(self, tmp_path):
        p = tmp_path / "verify.tsv"
        _write_verify_tsv(p, [
            _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {},
                  phase="optimize"),
            _vrow(1, "tiny", "tiny_a", 2, 2.0, 100, 10.0, 80.0, "pass", {},
                  phase="validate"),
        ])
        out = _read_existing_verify_reps(p, "abc1234", current_phase="validate")
        assert len(out[(1, "tiny")]) == 1

    def test_old_phase_rows_default_to_optimize(self, tmp_path):
        # file has commit but NO phase column -> rows treated as phase=optimize.
        p = tmp_path / "verify.tsv"
        header = (
            "timestamp\tthread\ttier\tdataset\trep\tspeed_sec\tpeak_mb\t"
            "baseline_speed\tspeedup_pct\tstatus\tmetrics_json\tcommit"
        )
        row = "\t".join([
            "2026", "1", "tiny", "tiny_a", "1", "2.0", "100", "10.0", "80.0",
            "pass", "{}", "abc1234",
        ])
        p.write_text(header + "\n" + row + "\n")
        out = _read_existing_verify_reps(p, "abc1234", current_phase="optimize")
        assert (1, "tiny") in out
        out2 = _read_existing_verify_reps(p, "abc1234", current_phase="validate")
        assert out2 == {}

    def test_short_rows_skipped(self, tmp_path):
        p = tmp_path / "verify.tsv"
        p.write_text(VERIFY_HEADER + "\n" + "abc1234\t1\ttiny\n")
        assert _read_existing_verify_reps(p, "abc1234") == {}


# --------------------------------------------------------------------------
# _compute_scaling_tax
# --------------------------------------------------------------------------

def _scell(thread, tier, base, turbo, *, oom=False):
    return {"thread": thread, "tier": tier, "baseline_speed": base,
            "speed_sec_median": turbo, "oom": oom}


class TestComputeScalingTax:
    def test_no_ood_not_applicable(self):
        out = _compute_scaling_tax(
            [_scell(1, "tiny", 10.0, 2.0)], {"hard_fail": 5.0})
        assert out["applicable"] is False
        assert "no OOD-tier cells" in out["reason"]

    def test_no_dev_factors_not_applicable(self):
        # only an ood cell with no valid dev factor.
        out = _compute_scaling_tax(
            [_scell(1, "ood_large", 10.0, 2.0)], {"hard_fail": 5.0})
        assert out["applicable"] is False

    def test_dev_with_zero_baseline_reports_reason(self):
        cells = [_scell(1, "tiny", 0.0, 2.0), _scell(1, "ood_large", 10.0, 2.0)]
        out = _compute_scaling_tax(cells, {"hard_fail": 5.0})
        assert out["applicable"] is False
        assert "no valid speedup factors" in out["reason"]

    def test_pass_verdict_thread_matched(self):
        # dev tiny@1: factor 5x; ood_large@1: factor 4x; tax = 5/4 = 1.25 < 1.5 -> PASS
        cells = [
            _scell(1, "tiny", 10.0, 2.0),
            _scell(1, "ood_large", 40.0, 10.0),
        ]
        out = _compute_scaling_tax(cells, {"hard_fail": 5.0, "ood_large_soft": 1.5})
        assert out["applicable"] is True
        r = out["ood_results"][0]
        assert r["verdict"] == "PASS"
        assert r["reference_kind"] == "thread_matched"
        assert out["hard_fails"] == 0 and out["soft_flags"] == 0

    def test_soft_verdict(self):
        # dev factor 5x; ood factor 2.5x; tax = 2.0 >= 1.5 soft, < 5 hard -> SOFT
        cells = [
            _scell(1, "tiny", 10.0, 2.0),
            _scell(1, "ood_large", 25.0, 10.0),
        ]
        out = _compute_scaling_tax(cells, {"hard_fail": 5.0, "ood_large_soft": 1.5})
        r = out["ood_results"][0]
        assert r["verdict"] == "SOFT"
        assert out["soft_flags"] == 1

    def test_hard_verdict(self):
        # dev factor 10x; ood factor 1x; tax = 10 >= 5 -> HARD
        cells = [
            _scell(1, "tiny", 100.0, 10.0),
            _scell(1, "ood_large", 10.0, 10.0),
        ]
        out = _compute_scaling_tax(cells, {"hard_fail": 5.0, "ood_large_soft": 1.5})
        r = out["ood_results"][0]
        assert r["verdict"] == "HARD"
        assert out["hard_fails"] == 1

    def test_zero_factor_is_hard_zero(self):
        # ood baseline 0 -> factor None -> ZERO verdict, counts as hard.
        cells = [
            _scell(1, "tiny", 10.0, 2.0),
            _scell(1, "ood_large", 0.0, 10.0),
        ]
        out = _compute_scaling_tax(cells, {"hard_fail": 5.0})
        r = out["ood_results"][0]
        assert r["verdict"] == "ZERO"
        assert r["tax"] == float("inf")
        assert out["hard_fails"] == 1

    def test_oom_cells_excluded(self):
        cells = [
            _scell(1, "tiny", 10.0, 2.0),
            _scell(1, "ood_large", 0.0, 0.0, oom=True),
        ]
        out = _compute_scaling_tax(cells, {"hard_fail": 5.0})
        # OOM ood cell is skipped -> no ood cells left -> not applicable.
        assert out["applicable"] is False

    def test_fallback_global_when_no_thread_match(self):
        # dev cell at thread=1, ood cell at thread=8 -> falls back to global geom.
        cells = [
            _scell(1, "tiny", 10.0, 2.0),     # dev factor 5x at thread=1
            _scell(8, "ood_large", 40.0, 10.0),  # ood factor 4x at thread=8
        ]
        out = _compute_scaling_tax(cells, {"hard_fail": 5.0, "ood_large_soft": 1.5})
        r = out["ood_results"][0]
        assert r["reference_kind"] == "fallback_global"

    def test_ood_xlarge_soft_threshold(self):
        cells = [
            _scell(1, "tiny", 10.0, 2.0),       # factor 5x
            _scell(1, "ood_xlarge", 22.5, 10.0),  # factor 2.25x -> tax 2.22 >= 2.0 soft
        ]
        out = _compute_scaling_tax(
            cells, {"hard_fail": 5.0, "ood_xlarge_soft": 2.0})
        r = out["ood_results"][0]
        assert r["threshold_label"] == "ood_xlarge_soft"
        assert r["verdict"] == "SOFT"

    def test_geom_mean_math(self):
        # two dev cells, factors 4x and 9x -> geom = 6.0
        cells = [
            _scell(1, "tiny", 8.0, 2.0),     # 4x
            _scell(1, "medium", 18.0, 2.0),  # 9x
            _scell(1, "ood_large", 12.0, 2.0),
        ]
        out = _compute_scaling_tax(cells, {"hard_fail": 5.0})
        assert out["dev_geom_mean"] == pytest.approx(6.0)


# --------------------------------------------------------------------------
# _per_rep_verify_status
# --------------------------------------------------------------------------

class TestPerRepVerifyStatus:
    def test_crash_always_wins(self):
        assert _per_rep_verify_status(1, 50.0, {}, [], "crash") == "crash"

    def test_thread1_regression_fails(self):
        assert _per_rep_verify_status(1, -5.0, {}, [], "ok") == "fail"

    def test_thread1_nonneg_passes_no_metrics(self):
        assert _per_rep_verify_status(1, 0.0, {}, [], "ok") == "pass"

    def test_thread_gt1_zero_speedup_fails(self):
        assert _per_rep_verify_status(8, 0.0, {}, [], "ok") == "fail"

    def test_thread_gt1_positive_passes(self):
        assert _per_rep_verify_status(8, 10.0, {}, [], "ok") == "pass"

    def test_missing_metric_fails(self):
        spec = [{"name": "j", "comparator": "gte", "threshold": 0.9}]
        assert _per_rep_verify_status(1, 5.0, {}, spec, "ok") == "fail"

    def test_non_numeric_metric_fails(self):
        spec = [{"name": "j", "comparator": "gte", "threshold": 0.9}]
        assert _per_rep_verify_status(1, 5.0, {"j": "oops"}, spec, "ok") == "fail"

    def test_gte_below_threshold_fails(self):
        spec = [{"name": "j", "comparator": "gte", "threshold": 0.9}]
        assert _per_rep_verify_status(1, 5.0, {"j": 0.5}, spec, "ok") == "fail"

    def test_lte_above_threshold_fails(self):
        spec = [{"name": "d", "comparator": "lte", "threshold": 0.05}]
        assert _per_rep_verify_status(1, 5.0, {"d": 0.5}, spec, "ok") == "fail"

    def test_all_metrics_pass(self):
        spec = [
            {"name": "j", "comparator": "gte", "threshold": 0.9},
            {"name": "d", "comparator": "lte", "threshold": 0.05},
        ]
        assert _per_rep_verify_status(
            1, 5.0, {"j": 0.99, "d": 0.01}, spec, "ok") == "pass"

    def test_stochastic_metric_uses_intrinsic_noise(self):
        spec = [{"name": "d", "comparator": "lte", "absolute_floor": 0.05,
                 "noise_multiplier": 2.0}]
        noise = {"medium": {"d": 0.1}}  # eff threshold = max(0.05, 2*0.1)=0.2
        # value 0.15 passes against 0.2 but would fail a fixed 0.05 floor.
        assert _per_rep_verify_status(
            1, 5.0, {"d": 0.15}, spec, "ok", tier="medium",
            intrinsic_noise=noise) == "pass"


# --------------------------------------------------------------------------
# _load_context_cells_from_verify_tsv
# --------------------------------------------------------------------------

class TestLoadContextCells:
    def test_missing_file_empty(self, tmp_path):
        assert _load_context_cells_from_verify_tsv(
            tmp_path / "nope.tsv", tmp_path, "abc1234", "optimize", [], set()) == []

    def test_skips_in_run_keys(self, tmp_path):
        p = tmp_path / "verify.tsv"
        _write_verify_tsv(p, [
            _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.9}),
            _vrow(1, "ood_large", "olarge", 1, 8.0, 400, 40.0, 80.0, "pass", {"j": 0.8}),
        ])
        # tiny is in the current run -> only ood_large becomes a context cell.
        out = _load_context_cells_from_verify_tsv(
            p, tmp_path, "abc1234", "optimize",
            [{"name": "j", "comparator": "gte"}], {(1, "tiny")})
        assert len(out) == 1
        assert out[0]["tier"] == "ood_large"
        assert out[0]["in_current_run"] is False

    def test_aggregates_reps_and_picks_worst_metric(self, tmp_path):
        p = tmp_path / "verify.tsv"
        _write_verify_tsv(p, [
            _vrow(1, "ood_large", "olarge", 1, 8.0, 400, 40.0, 80.0, "pass", {"j": 0.95}),
            _vrow(1, "ood_large", "olarge", 2, 9.0, 410, 40.0, 78.0, "pass", {"j": 0.90}),
        ])
        out = _load_context_cells_from_verify_tsv(
            p, tmp_path, "abc1234", "optimize",
            [{"name": "j", "comparator": "gte"}], set())
        c = out[0]
        assert c["n_reps"] == 2
        # gte metric -> worst is the minimum (0.90).
        assert c["metrics_worst"]["j"] == 0.90
        assert c["speed_sec_median"] == 8.5

    def test_oom_status_sets_verdict(self, tmp_path):
        p = tmp_path / "verify.tsv"
        _write_verify_tsv(p, [
            _vrow(1, "ood_xlarge", "oxl", 1, 0, 0, 0, 0.0, "oom", {}),
        ])
        out = _load_context_cells_from_verify_tsv(
            p, tmp_path, "abc1234", "optimize", [], set())
        assert out[0]["oom"] is True
        assert out[0]["verdict"] == "OOM"

    def test_crash_status_sets_verdict(self, tmp_path):
        p = tmp_path / "verify.tsv"
        _write_verify_tsv(p, [
            _vrow(1, "ood_large", "olarge", 1, 0, 0, 40.0, 0.0, "crash", {}),
        ])
        out = _load_context_cells_from_verify_tsv(
            p, tmp_path, "abc1234", "optimize", [], set())
        assert out[0]["any_crash"] is True
        assert out[0]["verdict"] == "CRASH"

    def test_baseline_falls_back_to_results_tsv(self, tmp_path):
        # verify.tsv carries baseline_speed=NA -> falls back to results.tsv.
        p = tmp_path / "verify.tsv"
        _write_verify_tsv(p, [
            _vrow(1, "ood_large", "olarge", 1, 8.0, 400, "NA", 0.0, "pass", {}),
        ])
        _write_results_tsv(tmp_path, [("olarge", 1, 64.0, 800, "baseline")])
        out = _load_context_cells_from_verify_tsv(
            p, tmp_path, "abc1234", "optimize", [], set())
        assert out[0]["baseline_speed"] == 64.0

    def test_legacy_csv_escaped_and_nonnumeric_metrics(self, tmp_path):
        # metrics_json written as legacy CSV-escaped; one value is non-numeric.
        # Worst-metric aggregation must tolerate both (lte metric -> max).
        p = tmp_path / "verify.tsv"
        header = VERIFY_HEADER
        # build a row by hand with a CSV-escaped metrics_json cell.
        escaped = '"{""d"":0.02,""note"":""skip""}"'
        row = "\t".join([
            "2026", "1", "ood_large", "olarge", "1", "8.0", "400",
            "40.0", "80.0", "pass", escaped, "abc1234", "optimize",
        ])
        p.write_text(header + "\n" + row + "\n")
        out = _load_context_cells_from_verify_tsv(
            p, tmp_path, "abc1234", "optimize",
            [{"name": "d", "comparator": "lte"}], set())
        # numeric 'd' survives; the string 'note' is ignored in worst-metric.
        assert out[0]["metrics_worst"]["d"] == 0.02
        assert out[0]["metrics_worst"].get("note") is None


# --------------------------------------------------------------------------
# _migrate_verify_tsv_add_phase
# --------------------------------------------------------------------------

class TestMigratePhase:
    def test_missing_file(self, tmp_path):
        assert _migrate_verify_tsv_add_phase(tmp_path / "nope.tsv") is False

    def test_already_has_phase(self, tmp_path):
        p = tmp_path / "verify.tsv"
        _write_verify_tsv(p, [_vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0,
                                    "pass", {})])
        assert _migrate_verify_tsv_add_phase(p) is False

    def test_no_commit_col_not_migrated(self, tmp_path):
        p = tmp_path / "verify.tsv"
        p.write_text("timestamp\tthread\ttier\n2026\t1\ttiny\n")
        assert _migrate_verify_tsv_add_phase(p) is False

    def test_migrates_adds_phase_optimize(self, tmp_path):
        p = tmp_path / "verify.tsv"
        header = (
            "timestamp\tthread\ttier\tdataset\trep\tspeed_sec\tpeak_mb\t"
            "baseline_speed\tspeedup_pct\tstatus\tmetrics_json\tcommit"
        )
        row = "\t".join(["2026", "1", "tiny", "tiny_a", "1", "2.0", "100",
                         "10.0", "80.0", "pass", "{}", "abc1234"])
        p.write_text(header + "\n" + row + "\n")
        assert _migrate_verify_tsv_add_phase(p) is True
        lines = p.read_text().splitlines()
        assert lines[0].endswith("\tphase")
        assert lines[1].endswith("\toptimize")


# --------------------------------------------------------------------------
# _strip_crash_rows_for_topup
# --------------------------------------------------------------------------

class TestStripCrashRows:
    def test_missing_file_zero(self, tmp_path):
        assert _strip_crash_rows_for_topup(tmp_path / "nope.tsv", "abc1234") == 0

    def test_strips_only_current_commit_crash(self, tmp_path):
        p = tmp_path / "verify.tsv"
        _write_verify_tsv(p, [
            _vrow(1, "tiny", "tiny_a", 1, 0, 0, 10.0, 0.0, "crash", {},
                  commit="abc1234"),
            _vrow(1, "tiny", "tiny_a", 2, 2.0, 100, 10.0, 80.0, "pass", {},
                  commit="abc1234"),
            _vrow(1, "tiny", "tiny_a", 1, 0, 0, 10.0, 0.0, "crash", {},
                  commit="other99"),
        ])
        n = _strip_crash_rows_for_topup(p, "abc1234")
        assert n == 1
        text = p.read_text()
        # other-commit crash row preserved.
        assert "other99" in text
        # current-commit pass row preserved.
        assert text.count("pass") == 1

    def test_phase_scoped_strip(self, tmp_path):
        p = tmp_path / "verify.tsv"
        _write_verify_tsv(p, [
            _vrow(1, "tiny", "tiny_a", 1, 0, 0, 10.0, 0.0, "crash", {},
                  phase="optimize"),
            _vrow(1, "tiny", "tiny_a", 1, 0, 0, 10.0, 0.0, "crash", {},
                  phase="validate"),
        ])
        n = _strip_crash_rows_for_topup(p, "abc1234", current_phase="validate")
        assert n == 1
        # the optimize-phase crash row is preserved.
        assert "optimize" in p.read_text()


# --------------------------------------------------------------------------
# _dataset_size_mb + _estimate_tier_peak_mb
# --------------------------------------------------------------------------

class TestDatasetSizeMb:
    def test_no_path_zero(self, tmp_path):
        assert _dataset_size_mb(tmp_path, {"path": ""}) == 0.0

    def test_absolute_file(self, tmp_path):
        f = tmp_path / "d.h5ad"
        f.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB
        assert _dataset_size_mb(tmp_path, {"path": str(f)}) == pytest.approx(2.0)

    def test_relative_resolved_against_task_dir(self, tmp_path):
        (tmp_path / "data").mkdir()
        f = tmp_path / "data" / "d.h5ad"
        f.write_bytes(b"x" * (1024 * 1024))  # 1 MB
        assert _dataset_size_mb(tmp_path, {"path": "data/d.h5ad"}) == pytest.approx(1.0)

    def test_directory_sums_files(self, tmp_path):
        d = tmp_path / "ds"
        (d / "sub").mkdir(parents=True)
        (d / "a.bin").write_bytes(b"x" * (1024 * 1024))
        (d / "sub" / "b.bin").write_bytes(b"x" * (1024 * 1024))
        assert _dataset_size_mb(tmp_path, {"path": str(d)}) == pytest.approx(2.0)

    def test_missing_path_zero(self, tmp_path):
        assert _dataset_size_mb(tmp_path, {"path": "nope/missing.h5ad"}) == 0.0


class TestEstimateTierPeakMb:
    def _task(self, tmp_path, baselines):
        _write_results_tsv(tmp_path, baselines)
        return tmp_path

    def test_measured_direct(self, tmp_path):
        task = self._task(tmp_path, [("medium_a", 1, 10.0, 8200, "baseline")])
        peak, src = _estimate_tier_peak_mb(
            task, {"name": "medium_a", "tier": "medium"}, [])
        assert peak == 8200
        assert "measured (8200 MB)" in src

    def test_extrapolated_from_largest(self, tmp_path):
        # ref tier 'large' has a measured peak + a file; target has a file.
        task = tmp_path
        (task / "data").mkdir()
        (task / "data" / "large.h5ad").write_bytes(b"x" * (4 * 1024 * 1024))  # 4 MB
        (task / "data" / "med.h5ad").write_bytes(b"x" * (2 * 1024 * 1024))    # 2 MB
        _write_results_tsv(task, [("large_a", 1, 30.0, 8000, "baseline")])
        all_entries = [
            {"name": "large_a", "tier": "large", "path": "data/large.h5ad"},
            {"name": "med_a", "tier": "medium", "path": "data/med.h5ad"},
        ]
        peak, src = _estimate_tier_peak_mb(task, all_entries[1], all_entries)
        # ratio 2/4 = 0.5 -> 8000*0.5 = 4000
        assert peak == pytest.approx(4000.0)
        assert "extrapolated from large" in src

    def test_file_size_heuristic_when_no_baseline(self, tmp_path):
        task = tmp_path
        (task / "data").mkdir()
        (task / "data" / "x.h5ad").write_bytes(b"x" * (3 * 1024 * 1024))  # 3 MB
        # no baselines at all
        (task / "results.tsv").write_text(RESULTS_HEADER + "\n")
        entry = {"name": "x_a", "tier": "xlarge", "path": "data/x.h5ad"}
        peak, src = _estimate_tier_peak_mb(task, entry, [entry])
        assert peak == pytest.approx(9.0)  # 3 MB * 3.0 multiplier
        assert "from file size" in src

    def test_no_signal_returns_zero(self, tmp_path):
        (tmp_path / "results.tsv").write_text(RESULTS_HEADER + "\n")
        entry = {"name": "x_a", "tier": "xlarge", "path": ""}
        peak, src = _estimate_tier_peak_mb(tmp_path, entry, [entry])
        assert peak == 0.0
        assert src == ""

    def test_extrapolation_ratio_too_large_falls_to_filesize(self, tmp_path):
        task = tmp_path
        (task / "data").mkdir()
        # ref 1 MB, target 100 MB -> ratio 100 > 10 cap -> file-size heuristic.
        (task / "data" / "ref.h5ad").write_bytes(b"x" * (1024 * 1024))
        (task / "data" / "big.h5ad").write_bytes(b"x" * (100 * 1024 * 1024))
        _write_results_tsv(task, [("ref_a", 1, 5.0, 500, "baseline")])
        entries = [
            {"name": "ref_a", "tier": "tiny", "path": "data/ref.h5ad"},
            {"name": "big_a", "tier": "huge", "path": "data/big.h5ad"},
        ]
        peak, src = _estimate_tier_peak_mb(task, entries[1], entries)
        assert "from file size" in src
        assert peak == pytest.approx(300.0)  # 100 MB * 3.0


# --------------------------------------------------------------------------
# _resolve_verify_mem_cap
# --------------------------------------------------------------------------

class TestResolveMemCap:
    def test_explicit_number(self):
        assert _resolve_verify_mem_cap("12.5") == 12.5

    def test_invalid_dies(self):
        with pytest.raises(SystemExit):
            _resolve_verify_mem_cap("notanumber")

    def test_auto_uses_total_ram_fraction(self, monkeypatch):
        import zyme.dispatch.resources as res
        monkeypatch.setattr(res, "total_ram_gb", lambda: 100.0)
        assert _resolve_verify_mem_cap("auto") == pytest.approx(70.0)

    def test_auto_disabled_when_probe_fails(self, monkeypatch, capsys):
        import zyme.dispatch.resources as res
        monkeypatch.setattr(res, "total_ram_gb", lambda: 0.0)
        assert _resolve_verify_mem_cap("auto") == 0.0
        assert "could not probe" in capsys.readouterr().out


# --------------------------------------------------------------------------
# _resolve_verify_ram_floor
# --------------------------------------------------------------------------

class TestResolveRamFloor:
    def test_explicit_number(self, tmp_path):
        floor, notes = _resolve_verify_ram_floor(
            "8.0", [], tmp_path, tmp_path / "task.yaml")
        assert floor == 8.0
        assert notes == []

    def test_invalid_number_dies(self, tmp_path):
        # A non-'auto' spec that float() can't parse -> die (SystemExit).
        with pytest.raises(SystemExit):
            _resolve_verify_ram_floor(
                object(), [], tmp_path, tmp_path / "task.yaml")

    def test_auto_fallback_when_no_signal(self, tmp_path):
        ty = tmp_path / "task.yaml"
        ty.write_text(
            "datasets:\n  - {tier: tiny, name: tiny_a, path: data/missing.h5ad}\n"
        )
        (tmp_path / "results.tsv").write_text(RESULTS_HEADER + "\n")
        cell_pairs = [(1, {"tier": "tiny", "name": "tiny_a"})]
        floor, notes = _resolve_verify_ram_floor("auto", cell_pairs, tmp_path, ty)
        assert floor == v._VERIFY_RAM_FLOOR_FALLBACK_GB
        assert any("no peak_mb signal" in n for n in notes)

    def test_auto_with_measured_peak(self, tmp_path):
        ty = tmp_path / "task.yaml"
        ty.write_text(
            "datasets:\n  - {tier: tiny, name: tiny_a, path: data/x.h5ad}\n"
        )
        _write_results_tsv(tmp_path, [("tiny_a", 1, 5.0, 4096, "baseline")])
        cell_pairs = [(1, {"tier": "tiny", "name": "tiny_a"})]
        floor, notes = _resolve_verify_ram_floor("auto", cell_pairs, tmp_path, ty)
        # 4096 MB * 1.5 / 1024 + 2.0 = 6.0 + 2.0 = 8.0
        assert floor == pytest.approx(8.0)
        assert any("measured" in n for n in notes)


# --------------------------------------------------------------------------
# _verify_ram_preflight
# --------------------------------------------------------------------------

class TestRamPreflight:
    def test_disabled_when_floor_zero(self):
        # No probe import even attempted; returns immediately.
        _verify_ram_preflight(0.0, 1, "tiny", 1, 3)

    def test_passes_when_free_above_floor(self, monkeypatch):
        import zyme.dispatch.resources as res
        monkeypatch.setattr(res, "free_ram_gb", lambda: 32.0)
        _verify_ram_preflight(8.0, 1, "tiny", 1, 3)  # no raise

    def test_aborts_when_free_below_floor(self, monkeypatch):
        import zyme.dispatch.resources as res
        monkeypatch.setattr(res, "free_ram_gb", lambda: 2.0)
        with pytest.raises(SystemExit):
            _verify_ram_preflight(8.0, 4, "medium", 2, 5)


# --------------------------------------------------------------------------
# _write_verify_summary_txt
# --------------------------------------------------------------------------

class TestWriteSummaryTxt:
    def test_table_columns_and_verdicts(self, tmp_path):
        cells = [
            {"thread": 1, "tier": "tiny", "speed_sec_median": 2.5,
             "speedup_pct_median": 80.0, "verdict": "PASS",
             "metrics_worst": {"j": 0.99}},
            {"thread": 8, "tier": "tiny", "speed_sec_median": 1.0,
             "speedup_pct_median": 90.0, "verdict": "FAIL",
             "metrics_worst": {"j": None}},
        ]
        spec = [{"name": "j", "comparator": "gte", "threshold": 0.9}]
        out = tmp_path / "verify.txt"
        _write_verify_summary_txt(cells, spec, 3, out)
        text = out.read_text()
        assert "reps_per_cell: 3" in text
        assert "matplotlib not available" in text
        assert "thread" in text and "verdict" in text
        assert "PASS" in text and "FAIL" in text
        assert "0.9900" in text
        assert "—" in text  # the None metric renders as em-dash


# --------------------------------------------------------------------------
# probe cache helpers
# --------------------------------------------------------------------------

class TestProbeCache:
    def test_cache_path(self, tmp_path):
        assert _verify_probe_cache_path(tmp_path) == tmp_path / ".zyme" / "verify_probe.cache"

    def test_has_false_when_missing(self, tmp_path):
        assert _verify_probe_cache_has(tmp_path, "abc1234", 1, 8) is False

    def test_write_then_has(self, tmp_path):
        _verify_probe_cache_write(tmp_path, "abc1234", 1, 8, "tiny")
        assert _verify_probe_cache_has(tmp_path, "abc1234", 1, 8) is True
        # different thread set -> miss
        assert _verify_probe_cache_has(tmp_path, "abc1234", 1, 4) is False
        # different commit -> miss
        assert _verify_probe_cache_has(tmp_path, "zzz0000", 1, 8) is False

    def test_write_appends(self, tmp_path):
        _verify_probe_cache_write(tmp_path, "abc1234", 1, 8, "tiny")
        _verify_probe_cache_write(tmp_path, "def5678", 1, 4, "medium")
        assert _verify_probe_cache_has(tmp_path, "abc1234", 1, 8) is True
        assert _verify_probe_cache_has(tmp_path, "def5678", 1, 4) is True


# --------------------------------------------------------------------------
# lock helpers
# --------------------------------------------------------------------------

class TestVerifyLock:
    def test_acquire_creates_lock_then_release_removes(self, tmp_path):
        out = tmp_path / "verify.tsv"
        lock = _acquire_verify_lock(out)
        assert lock.exists()
        assert lock.suffix == ".lock"
        contents = lock.read_text()
        assert f"pid={os.getpid()}" in contents
        _release_verify_lock(lock)
        assert not lock.exists()

    def test_release_missing_lock_no_error(self, tmp_path):
        _release_verify_lock(tmp_path / "gone.lock")  # no raise

    def test_held_by_live_process_dies(self, tmp_path):
        out = tmp_path / "verify.tsv"
        lock = out.with_suffix(out.suffix + ".lock")
        # write a lock owned by THIS process (definitely alive) -> die.
        lock.write_text(f"pid={os.getpid()}\nstarted=now\nhost=h\n")
        with pytest.raises(SystemExit):
            _acquire_verify_lock(out)

    def test_stale_lock_reclaimed(self, tmp_path, monkeypatch):
        out = tmp_path / "verify.tsv"
        lock = out.with_suffix(out.suffix + ".lock")
        lock.write_text("pid=999999\nstarted=old\nhost=h\n")

        # Make os.kill(999999, 0) raise ProcessLookupError -> stale.
        real_kill = os.kill

        def fake_kill(pid, sig):
            if pid == 999999:
                raise ProcessLookupError()
            return real_kill(pid, sig)

        monkeypatch.setattr(v.os, "kill", fake_kill)
        new_lock = _acquire_verify_lock(out)
        assert new_lock.exists()
        assert f"pid={os.getpid()}" in new_lock.read_text()
        _release_verify_lock(new_lock)


# --------------------------------------------------------------------------
# _cmd_verify_render_only — entry point, hand-written verify.tsv
# --------------------------------------------------------------------------

class _Args:
    """Lightweight argparse.Namespace stand-in."""
    def __init__(self, **kw):
        self.output = "verify.tsv"
        self.no_plot = True
        self.render_only = True
        self.cells = None
        self.tiers = None
        self.threads = "1,4,8"
        self.phase = "optimize"
        self.reps = 3
        self.skip_probe = True
        self.force_probe = False
        self.probe_tier = None
        self.write_mode = "overwrite"
        self.ram_floor = "0"
        self.mem_cap_gb = "0"
        self.cell_settle_s = 0.0
        self.allow_not_applicable_threads = False
        self.task_dir = None
        for k, val in kw.items():
            setattr(self, k, val)


def _render_task(tmp_path, *, threading="default", metrics=True):
    task = tmp_path / "task"
    task.mkdir()
    (task / ".zyme").mkdir()
    yaml = "datasets:\n  - {tier: tiny, name: tiny_a, path: data/x.h5ad}\n"
    if metrics:
        yaml += "metrics:\n  - {name: j, comparator: gte, threshold: 0.9}\n"
    yaml += f"threading: {threading}\n"
    (task / "task.yaml").write_text(yaml)
    return task


def _render_task_for_plot(tmp_path):
    """Render task whose figure is actually drawn (no_plot=False)."""
    task = tmp_path / "task"
    task.mkdir()
    (task / ".zyme").mkdir()
    (task / "task.yaml").write_text(
        "datasets:\n  - {tier: tiny, name: tiny_a, path: data/x.h5ad}\n"
        "metrics:\n  - {name: j, comparator: gte, threshold: 0.9}\n"
        "threading: default\n"
    )
    return task


class TestCmdVerifyRenderOnly:
    def test_missing_verify_tsv_dies(self, tmp_path, monkeypatch):
        task = _render_task(tmp_path)
        monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
        with pytest.raises(SystemExit):
            v._cmd_verify_render_only(_Args(), task)

    def test_empty_verify_tsv_dies(self, tmp_path, monkeypatch):
        task = _render_task(tmp_path)
        (task / "verify.tsv").write_text("")
        monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
        with pytest.raises(SystemExit):
            v._cmd_verify_render_only(_Args(), task)

    def test_renders_pass_and_exits_zero(self, tmp_path, monkeypatch, capsys):
        task = _render_task(tmp_path)
        _write_verify_tsv(task / "verify.tsv", [
            _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99}),
            _vrow(1, "tiny", "tiny_a", 2, 2.1, 101, 10.0, 79.0, "pass", {"j": 0.98}),
        ])
        _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline")])
        monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
        # no_plot -> no figure; render-only should not raise/exit nonzero.
        v._cmd_verify_render_only(_Args(no_plot=True), task)
        out = capsys.readouterr().out
        assert "1 PASS" in out

    def test_fail_cell_exits_one(self, tmp_path, monkeypatch):
        task = _render_task(tmp_path)
        # metric below threshold -> FAIL -> sys.exit(1)
        _write_verify_tsv(task / "verify.tsv", [
            _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.5}),
        ])
        _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline")])
        monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
        with pytest.raises(SystemExit) as ei:
            v._cmd_verify_render_only(_Args(no_plot=True), task)
        assert ei.value.code == 1

    def test_no_matching_commit_dies(self, tmp_path, monkeypatch):
        task = _render_task(tmp_path)
        _write_verify_tsv(task / "verify.tsv", [
            _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99},
                  commit="other99"),
        ])
        monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
        with pytest.raises(SystemExit):
            v._cmd_verify_render_only(_Args(no_plot=True), task)

    def test_old_schema_no_status_dies(self, tmp_path, monkeypatch):
        task = _render_task(tmp_path)
        (task / "verify.tsv").write_text("timestamp\tthread\ttier\tcommit\n"
                                         "2026\t1\ttiny\tabc1234\n")
        monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
        with pytest.raises(SystemExit):
            v._cmd_verify_render_only(_Args(no_plot=True), task)

    def test_hard_scaling_fail_exits_two(self, tmp_path, monkeypatch):
        task = tmp_path / "task"
        task.mkdir()
        (task / ".zyme").mkdir()
        (task / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: tiny, name: tiny_a, path: data/x.h5ad}\n"
            "  - {tier: ood_large, name: olarge_a, path: data/o.h5ad}\n"
            "metrics:\n  - {name: j, comparator: gte, threshold: 0.9}\n"
            "threading: default\n"
            "scaling_tax_thresholds:\n  ood_large_soft: 1.5\n  hard_fail: 5.0\n"
        )
        # dev tiny factor 10x; ood_large factor 1x -> tax 10 -> HARD -> exit 2.
        _write_verify_tsv(task / "verify.tsv", [
            _vrow(1, "tiny", "tiny_a", 1, 1.0, 100, 10.0, 90.0, "pass", {"j": 0.99}),
            _vrow(1, "ood_large", "olarge_a", 1, 10.0, 400, 10.0, 0.0, "pass", {"j": 0.95}),
        ])
        _write_results_tsv(task, [
            ("tiny_a", 1, 10.0, 200, "baseline"),
            ("olarge_a", 1, 10.0, 800, "baseline"),
        ])
        monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
        with pytest.raises(SystemExit) as ei:
            v._cmd_verify_render_only(_Args(no_plot=True), task)
        assert ei.value.code == 2


# --------------------------------------------------------------------------
# cmd_verify — live matrix with the subprocess boundary monkeypatched
# --------------------------------------------------------------------------

def _live_task(tmp_path, *, tiers_yaml=None, threading="default", metrics=True):
    task = tmp_path / "task"
    task.mkdir()
    (task / ".zyme").mkdir()
    if tiers_yaml is None:
        tiers_yaml = (
            "datasets:\n"
            "  - {tier: tiny, name: tiny_a, path: data/x.h5ad}\n"
        )
    yaml = tiers_yaml
    if metrics:
        yaml += "metrics:\n  - {name: j, comparator: gte, threshold: 0.9}\n"
    yaml += f"threading: {threading}\n"
    (task / "task.yaml").write_text(yaml)
    return task


def _fake_run_log(speed=2.0, peak=100.0, metrics_line="j: 0.99", status_ok=True):
    body = f"speed_sec: {speed}\npeak_mb: {peak}\n{metrics_line}\n"
    if status_ok:
        body += "status:           ok\n"
    else:
        body += "status:           crash\n"
    return body


class TestCmdVerifyLive:
    def _patch_common(self, monkeypatch, run_log_factory):
        # Single subprocess boundary: run_task. git HEAD deterministic.
        monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")

        calls = []

        def fake_run_task(task_dir, *, dataset_entry=None, extra_env=None,
                          mem_cap_gb=None, thread=None, **kw):
            calls.append({"dataset": dataset_entry, "env": dict(extra_env or {}),
                          "thread": thread})
            return run_log_factory(dataset_entry, extra_env, thread)

        monkeypatch.setattr(v, "run_task", fake_run_task)
        return calls

    def test_single_cell_pass_writes_tsv(self, tmp_path, monkeypatch, capsys):
        task = _live_task(tmp_path, threading="not_applicable")
        _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline")])
        self._patch_common(
            monkeypatch,
            lambda ds, env, thr: _fake_run_log(speed=2.0, metrics_line="j: 0.99"),
        )
        args = _Args(render_only=False, threads="1", reps=1, no_plot=True,
                     ram_floor="0", mem_cap_gb="0", cell_settle_s=0.0,
                     skip_probe=True, task_dir=str(task))
        # not_applicable + single cell + all PASS -> exit 0 (no SystemExit).
        v.cmd_verify(args)
        out = capsys.readouterr().out
        assert "1/1 cells PASS" in out
        vtsv = (task / "verify.tsv").read_text()
        assert "tiny\ttiny_a\t1\t" in vtsv
        assert "\tpass\t" in vtsv

    def test_metric_fail_exits_one(self, tmp_path, monkeypatch):
        task = _live_task(tmp_path, threading="not_applicable")
        _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline")])
        self._patch_common(
            monkeypatch,
            lambda ds, env, thr: _fake_run_log(speed=2.0, metrics_line="j: 0.50"),
        )
        args = _Args(render_only=False, threads="1", reps=1, no_plot=True,
                     skip_probe=True, task_dir=str(task))
        with pytest.raises(SystemExit) as ei:
            v.cmd_verify(args)
        assert ei.value.code == 1

    def test_crash_cell_exits_one_and_continues(self, tmp_path, monkeypatch, capsys):
        task = _live_task(tmp_path, threading="not_applicable")
        _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline")])
        self._patch_common(
            monkeypatch,
            lambda ds, env, thr: _fake_run_log(status_ok=False),
        )
        args = _Args(render_only=False, threads="1", reps=1, no_plot=True,
                     skip_probe=True, task_dir=str(task))
        with pytest.raises(SystemExit) as ei:
            v.cmd_verify(args)
        assert ei.value.code == 1
        # verify.tsv has a crash row.
        assert "\tcrash\t" in (task / "verify.tsv").read_text()

    def test_threading_probe_runs_and_passes(self, tmp_path, monkeypatch, capsys):
        # threading default + 2 thread counts -> probe fires; speeds differ >1.5x.
        task = _live_task(tmp_path, threading="default")
        _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline"),
                                  ("tiny_a", 8, 3.0, 200, "baseline")])

        def factory(ds, env, thr):
            # thread=1 slow, thread=8 fast -> ratio ~3.3x passes probe.
            t = int((env or {}).get("ZYME_THREADS", thr or 1))
            return _fake_run_log(speed=(9.0 if t == 1 else 2.0))

        self._patch_common(monkeypatch, factory)
        args = _Args(render_only=False, threads="1,8", reps=1, no_plot=True,
                     skip_probe=False, force_probe=False, probe_tier=None,
                     task_dir=str(task))
        # Both cells pass speedup + metric -> exit 0.
        v.cmd_verify(args)
        out = capsys.readouterr().out
        assert "verify probe] PASS" in out
        assert "2/2 cells PASS" in out

    def test_probe_failure_dies(self, tmp_path, monkeypatch):
        task = _live_task(tmp_path, threading="default")
        _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline"),
                                  ("tiny_a", 8, 3.0, 200, "baseline")])
        # identical speeds across thread axis -> ratio ~1.0 < 1.5 -> probe FAIL.
        self._patch_common(monkeypatch, lambda ds, env, thr: _fake_run_log(speed=5.0))
        args = _Args(render_only=False, threads="1,8", reps=1, no_plot=True,
                     skip_probe=False, force_probe=False, probe_tier=None,
                     task_dir=str(task))
        with pytest.raises(SystemExit):
            v.cmd_verify(args)

    def test_oom_baseline_skips_subprocess(self, tmp_path, monkeypatch, capsys):
        task = _live_task(tmp_path, threading="not_applicable")
        # baseline marked oom -> the tier is unmeasurable; no run_task call.
        _write_results_tsv(task, [("tiny_a", 1, 0.0, 0, "oom")])
        calls = self._patch_common(
            monkeypatch, lambda ds, env, thr: _fake_run_log())
        args = _Args(render_only=False, threads="1", reps=1, no_plot=True,
                     skip_probe=True, task_dir=str(task))
        # OOM cell -> verdict OOM, not a fail -> exit 0.
        v.cmd_verify(args)
        # subprocess never launched for the OOM tier.
        assert calls == []
        assert "\toom\t" in (task / "verify.tsv").read_text()

    def test_invalid_cells_spec_dies(self, tmp_path, monkeypatch):
        task = _live_task(tmp_path, threading="not_applicable")
        self._patch_common(monkeypatch, lambda ds, env, thr: _fake_run_log())
        args = _Args(render_only=False, cells="badnocolon", no_plot=True,
                     skip_probe=True, task_dir=str(task))
        with pytest.raises(SystemExit):
            v.cmd_verify(args)

    def test_unknown_tier_in_cells_dies(self, tmp_path, monkeypatch):
        task = _live_task(tmp_path, threading="not_applicable")
        self._patch_common(monkeypatch, lambda ds, env, thr: _fake_run_log())
        args = _Args(render_only=False, cells="1:nonexistent", no_plot=True,
                     skip_probe=True, task_dir=str(task))
        with pytest.raises(SystemExit):
            v.cmd_verify(args)

    def test_not_applicable_forces_thread1(self, tmp_path, monkeypatch, capsys):
        task = _live_task(tmp_path, threading="not_applicable")
        _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline")])
        calls = self._patch_common(
            monkeypatch, lambda ds, env, thr: _fake_run_log(speed=2.0))
        # request threads 1,4,8 but not_applicable collapses to thread=1.
        args = _Args(render_only=False, threads="1,4,8", reps=1, no_plot=True,
                     skip_probe=True, task_dir=str(task))
        v.cmd_verify(args)
        threads_run = {c["thread"] for c in calls}
        assert threads_run == {1}
        assert "1/1 cells PASS" in capsys.readouterr().out

    def test_topup_reuses_existing_reps(self, tmp_path, monkeypatch, capsys):
        task = _live_task(tmp_path, threading="not_applicable")
        _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline")])
        # Pre-seed verify.tsv with 1 existing pass rep at this commit.
        _write_verify_tsv(task / "verify.tsv", [
            _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99}),
        ])
        calls = self._patch_common(
            monkeypatch, lambda ds, env, thr: _fake_run_log(speed=2.0))
        # reps=2, top-up -> only 1 new rep runs.
        args = _Args(render_only=False, threads="1", reps=2, no_plot=True,
                     skip_probe=True, write_mode="topup", task_dir=str(task))
        v.cmd_verify(args)
        # exactly one new subprocess (rep 2) launched.
        assert len(calls) == 1

    def test_multi_rep_renders_plot_and_cv_table(self, tmp_path, monkeypatch, capsys):
        # 3 reps with varying speed -> CV table printed; plot path exercised.
        task = _live_task(tmp_path, threading="not_applicable")
        _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline")])
        seq = iter([2.0, 2.2, 1.8])

        def factory(ds, env, thr):
            return _fake_run_log(speed=next(seq))

        self._patch_common(monkeypatch, factory)
        args = _Args(render_only=False, threads="1", reps=3, no_plot=False,
                     skip_probe=True, task_dir=str(task))
        v.cmd_verify(args)
        out = capsys.readouterr().out
        assert "| CV |" in out  # CV column appears for reps>1
        # plot files written by the matplotlib branch.
        assert (task / "verify.png").is_file()
        assert (task / "verify.svg").is_file()

    def test_super_linear_multithread_fails(self, tmp_path, monkeypatch):
        # thread=1 factor 2x, thread=8 factor 40x -> ratio 20 > 1.5*8=12 cap
        # -> super-linear FAIL (thread=1 path likely broken).
        task = _live_task(tmp_path, threading="default")
        _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline"),
                                  ("tiny_a", 8, 10.0, 200, "baseline")])

        def factory(ds, env, thr):
            t = int((env or {}).get("ZYME_THREADS", thr or 1))
            # thread=1 -> 5s (factor 2x); thread=8 -> 0.25s (factor 40x).
            return _fake_run_log(speed=(5.0 if t == 1 else 0.25))

        self._patch_common(monkeypatch, factory)
        # skip the probe so we reach the grading directly.
        args = _Args(render_only=False, threads="1,8", reps=1, no_plot=True,
                     skip_probe=True, task_dir=str(task))
        with pytest.raises(SystemExit) as ei:
            v.cmd_verify(args)
        assert ei.value.code == 1

    def test_render_only_with_cells_filter(self, tmp_path, monkeypatch, capsys):
        task = tmp_path / "task"
        task.mkdir()
        (task / ".zyme").mkdir()
        (task / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: tiny, name: tiny_a, path: data/x.h5ad}\n"
            "  - {tier: medium, name: med_a, path: data/m.h5ad}\n"
            "metrics:\n  - {name: j, comparator: gte, threshold: 0.9}\n"
            "threading: default\n"
        )
        _write_verify_tsv(task / "verify.tsv", [
            _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99}),
            _vrow(1, "medium", "med_a", 1, 5.0, 300, 20.0, 75.0, "pass", {"j": 0.95}),
        ])
        _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline"),
                                  ("med_a", 1, 20.0, 600, "baseline")])
        monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
        # --cells filters to just 1:tiny.
        args = _Args(cells="1:tiny", no_plot=True, task_dir=str(task))
        v._cmd_verify_render_only(args, task)
        out = capsys.readouterr().out
        assert "1 cell(s)" in out

    def test_render_only_with_tiers_filter(self, tmp_path, monkeypatch, capsys):
        task = tmp_path / "task"
        task.mkdir()
        (task / ".zyme").mkdir()
        (task / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: tiny, name: tiny_a, path: data/x.h5ad}\n"
            "  - {tier: medium, name: med_a, path: data/m.h5ad}\n"
            "metrics:\n  - {name: j, comparator: gte, threshold: 0.9}\n"
            "threading: default\n"
        )
        _write_verify_tsv(task / "verify.tsv", [
            _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99}),
            _vrow(4, "medium", "med_a", 1, 5.0, 300, 20.0, 75.0, "pass", {"j": 0.95}),
        ])
        _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline"),
                                  ("med_a", 4, 20.0, 600, "baseline")])
        monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
        # --tiers medium + --threads 4 keeps only the medium row.
        args = _Args(tiers="medium", threads="4", no_plot=True, task_dir=str(task))
        v._cmd_verify_render_only(args, task)
        out = capsys.readouterr().out
        assert "1 cell(s)" in out

    def test_missing_baseline_preflight_warning_and_fail(self, tmp_path, monkeypatch, capsys):
        # No results.tsv baseline at all -> pre-flight warns, speedup_pct=0,
        # and at thread=1 the cell still PASSes (>=0 rule) since metric passes.
        task = _live_task(tmp_path, threading="not_applicable")
        # results.tsv with no baseline-class row for tiny_a at thread=1.
        (task / "results.tsv").write_text(RESULTS_HEADER + "\n")
        self._patch_common(
            monkeypatch,
            lambda ds, env, thr: _fake_run_log(speed=2.0, metrics_line="j: 0.99"),
        )
        args = _Args(render_only=False, threads="1", reps=1, no_plot=True,
                     skip_probe=True, task_dir=str(task))
        v.cmd_verify(args)
        out = capsys.readouterr().out
        assert "no baseline" in out  # pre-flight warning fired
        # speedup_pct=0 with thread=1 (>=0) still passes the speedup rule.
        assert "1/1 cells PASS" in out

    def test_oom_cell_already_recorded_not_duplicated(self, tmp_path, monkeypatch, capsys):
        # An OOM row already exists at this commit+phase -> no duplicate row,
        # and no subprocess launched.
        task = _live_task(tmp_path, threading="not_applicable")
        _write_results_tsv(task, [("tiny_a", 1, 0.0, 0, "oom")])
        _write_verify_tsv(task / "verify.tsv", [
            _vrow(1, "tiny", "tiny_a", 1, 0, 0, 0, 0.0, "oom", {}),
        ])
        calls = self._patch_common(
            monkeypatch, lambda ds, env, thr: _fake_run_log())
        args = _Args(render_only=False, threads="1", reps=1, no_plot=True,
                     skip_probe=True, write_mode="append", task_dir=str(task))
        v.cmd_verify(args)
        assert calls == []
        # exactly one oom row (no duplicate appended).
        text = (task / "verify.tsv").read_text()
        assert text.count("\toom\t") == 1

    def test_render_only_renders_matplotlib_figure(self, tmp_path, monkeypatch):
        task = _render_task_for_plot(tmp_path)
        _write_verify_tsv(task / "verify.tsv", [
            _vrow(1, "tiny", "tiny_a", 1, 2.0, 100, 10.0, 80.0, "pass", {"j": 0.99}),
            _vrow(1, "tiny", "tiny_a", 2, 2.1, 100, 10.0, 79.0, "pass", {"j": 0.98}),
        ])
        _write_results_tsv(task, [("tiny_a", 1, 10.0, 200, "baseline")])
        monkeypatch.setattr(v, "git", lambda *a, **k: "abc1234")
        # no_plot=False -> the matplotlib render branch runs and writes files.
        args = _Args(no_plot=False, task_dir=str(task))
        v._cmd_verify_render_only(args, task)
        assert (task / "verify.png").is_file()
