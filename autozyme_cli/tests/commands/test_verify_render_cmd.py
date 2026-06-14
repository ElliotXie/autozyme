"""Unit tests for zyme.commands.verify_render — the matplotlib painting layer.

verify_render.py is pure presentation: it turns already-computed `cells`
dicts + a `scaling_tax` dict (both produced by zyme.commands.verify) into a
six-panel matplotlib figure plus a textual scaling-tax block. Two test layers:

  1. Pure string/number helpers (no matplotlib at all): `_format_scaling_tax`,
     `_tier_color`, `_order_cells_for_plot`, `_cell_xtick_label`, `_fmt_mb`,
     `_fmt_seconds`. These are fully deterministic — assert exact substrings
     and rounding behavior across pass/soft/hard/zero verdicts, fallback
     references, empty/missing fields.

  2. The drawing entry point `_render_verify_matrix` + every individual panel,
     driven through a real (Agg-backend) matplotlib so all the per-panel
     branches (OOM, crash, no-baseline, log-scale, no-metrics, single-rep CV,
     context cells, scaling-tax overlays) execute and the figure files land on
     disk. We assert the output files exist and that panel functions return
     without raising, since the rendered pixels themselves are not the unit
     under test.
"""
from __future__ import annotations

import math

import pytest

import zyme.commands.verify_render as vr


# matplotlib is installed on this machine; force a headless backend so the
# panel-drawing tests never touch a display.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------
# helpers to build realistic cells + scaling-tax dicts
# --------------------------------------------------------------------------

def _cell(thread, tier, *, base=10.0, turbo=2.0, base_mb=1000.0, turbo_mb=600.0,
          metrics=None, verdict="PASS", oom=False, any_crash=False,
          in_run=True, cv=None, speedup_pct=None, n_reps=3):
    """Build a cells[] dict in the exact shape verify.py emits to the renderer."""
    if speedup_pct is None and base > 0 and turbo > 0:
        speedup_pct = (1 - turbo / base) * 100.0
    return {
        "thread": thread, "tier": tier, "dataset": f"{tier}_ds",
        "baseline_speed": base,
        "baseline_peak_mb": base_mb,
        "speed_sec_median": turbo,
        "speed_sec_min": turbo,
        "speed_sec_max": turbo,
        "peak_mb_median": turbo_mb,
        "speedup_pct_median": speedup_pct or 0.0,
        "metrics_worst": metrics if metrics is not None else {},
        "any_crash": any_crash,
        "oom": oom,
        "n_reps": n_reps,
        "in_current_run": in_run,
        "verdict": verdict,
        "reasons": [],
        "cv_pct": cv,
    }


def _tax(*, applicable=True, geom=5.0, by_thread=None, ood=None,
         hard=0, soft=0, dev_cells=None, reason=""):
    if by_thread is None:
        by_thread = {1: 5.0, 8: 6.0}
    return {
        "applicable": applicable,
        "reason": reason,
        "dev_geom_mean": geom,
        "dev_geom_by_thread": by_thread,
        "dev_cells": dev_cells if dev_cells is not None else [(1, "tiny", 5.0)],
        "ood_results": ood if ood is not None else [],
        "hard_fails": hard,
        "soft_flags": soft,
    }


def _ood_result(tier="ood_large", thread=1, *, factor=4.0, tax=1.2,
                verdict="PASS", ref=5.0, ref_kind="thread_matched",
                thr_used=1.5, thr_label="ood_large_soft"):
    return {
        "thread": thread, "tier": tier,
        "factor": factor, "tax": tax,
        "verdict": verdict,
        "threshold_used": thr_used,
        "threshold_label": thr_label,
        "dev_reference": ref,
        "reference_kind": ref_kind,
    }


THRESHOLDS = {"ood_xlarge_soft": 2.0, "ood_large_soft": 1.5, "hard_fail": 5.0}


# --------------------------------------------------------------------------
# _fmt_seconds
# --------------------------------------------------------------------------

class TestFmtSeconds:
    def test_minutes_above_120s(self):
        assert vr._fmt_seconds(180.0) == "3.0m"
        assert vr._fmt_seconds(120.0) == "2.0m"

    def test_whole_seconds_band(self):
        assert vr._fmt_seconds(45.0) == "45s"
        assert vr._fmt_seconds(10.0) == "10s"

    def test_one_decimal_band(self):
        assert vr._fmt_seconds(5.0) == "5.0s"
        assert vr._fmt_seconds(1.0) == "1.0s"

    def test_sub_second_uses_two_sig(self):
        assert vr._fmt_seconds(0.5) == "0.5s"
        assert vr._fmt_seconds(0.012) == "0.012s"


# --------------------------------------------------------------------------
# _fmt_mb
# --------------------------------------------------------------------------

class TestFmtMb:
    def test_gb_at_or_above_1024(self):
        assert vr._fmt_mb(1024.0) == "1.0G"
        assert vr._fmt_mb(2048.0) == "2.0G"
        assert vr._fmt_mb(3276.8) == "3.2G"

    def test_mb_below_1024(self):
        assert vr._fmt_mb(512.0) == "512M"
        assert vr._fmt_mb(1023.0) == "1023M"

    def test_rounds_to_int_mb(self):
        assert vr._fmt_mb(99.4) == "99M"
        assert vr._fmt_mb(99.6) == "100M"


# --------------------------------------------------------------------------
# _cell_xtick_label
# --------------------------------------------------------------------------

class TestCellXtickLabel:
    def test_two_line_tier_thread(self):
        assert vr._cell_xtick_label({"tier": "medium", "thread": 4}) == "medium\n4t"

    def test_ood_tier(self):
        assert vr._cell_xtick_label({"tier": "ood_xlarge", "thread": 8}) == "ood_xlarge\n8t"


# --------------------------------------------------------------------------
# _tier_color
# --------------------------------------------------------------------------

class TestTierColor:
    def test_dev_tier_uses_cool_palette(self):
        c = vr._tier_color("tiny", ["tiny", "medium"], [])
        assert c == vr._TIER_PALETTE_DEV[0]
        c2 = vr._tier_color("medium", ["tiny", "medium"], [])
        assert c2 == vr._TIER_PALETTE_DEV[1]

    def test_ood_tier_uses_warm_palette(self):
        c = vr._tier_color("ood_large", ["tiny"], ["ood_large", "ood_xlarge"])
        assert c == vr._TIER_PALETTE_OOD[0]
        c2 = vr._tier_color("ood_xlarge", ["tiny"], ["ood_large", "ood_xlarge"])
        assert c2 == vr._TIER_PALETTE_OOD[1]

    def test_palette_wraps_modulo(self):
        many = [f"t{i}" for i in range(6)]
        # index 4 wraps to palette[0] for a 4-color palette
        assert vr._tier_color("t4", many, []) == vr._TIER_PALETTE_DEV[0]

    def test_unknown_tier_grey(self):
        assert vr._tier_color("phantom", ["tiny"], ["ood_large"]) == "#888888"

    def test_ood_membership_takes_priority(self):
        # a tier in BOTH lists is colored as OOD (ood check comes first).
        assert vr._tier_color("x", ["x"], ["x"]) == vr._TIER_PALETTE_OOD[0]


# --------------------------------------------------------------------------
# _order_cells_for_plot
# --------------------------------------------------------------------------

class TestOrderCells:
    def test_dev_before_ood_yaml_order_within(self):
        cells = [
            _cell(1, "ood_large"),
            _cell(8, "medium"),
            _cell(1, "medium"),
            _cell(1, "tiny"),
        ]
        ordered, dev, ood, all_tiers, threads = vr._order_cells_for_plot(cells)
        # dev tiers keep first-seen (yaml appearance) order; ood last.
        # First input row is ood_large (ignored for dev order), then medium,
        # then tiny -> dev = [medium, tiny].
        assert dev == ["medium", "tiny"]
        # tier groups: all dev tiers, then ood tiers
        ordered_tiers = [c["tier"] for c in ordered]
        first_ood = ordered_tiers.index("ood_large")
        assert all(not t.startswith("ood_") for t in ordered_tiers[:first_ood])
        assert ood == ["ood_large"]

    def test_threads_ascending_within_tier(self):
        cells = [_cell(8, "tiny"), _cell(1, "tiny"), _cell(4, "tiny")]
        ordered, *_ = vr._order_cells_for_plot(cells)
        assert [c["thread"] for c in ordered] == [1, 4, 8]

    def test_returns_all_threads_sorted(self):
        cells = [_cell(8, "tiny"), _cell(1, "medium"), _cell(4, "tiny")]
        _, _, _, _, threads = vr._order_cells_for_plot(cells)
        assert threads == [1, 4, 8]

    def test_yaml_order_preserved_first_seen(self):
        # medium appears before tiny in the input -> medium ranks first.
        cells = [_cell(1, "medium"), _cell(1, "tiny")]
        ordered, dev, _, _, _ = vr._order_cells_for_plot(cells)
        assert dev == ["medium", "tiny"]
        assert [c["tier"] for c in ordered] == ["medium", "tiny"]


# --------------------------------------------------------------------------
# _format_scaling_tax
# --------------------------------------------------------------------------

class TestFormatScalingTax:
    def test_overall_geom_and_header(self):
        tax = _tax(geom=5.3, by_thread={1: 5.0, 8: 6.0},
                   ood=[_ood_result(verdict="PASS")])
        out = vr._format_scaling_tax(tax, THRESHOLDS)
        assert "=== Scaling tax analysis ===" in out
        assert "overall geometric mean): 5.3×" in out
        assert "per-thread (used for tax): 1t=5.0×, 8t=6.0×" in out

    def test_dev_cells_line(self):
        tax = _tax(dev_cells=[(1, "tiny", 5.0), (8, "medium", 12.0)],
                   ood=[_ood_result()])
        out = vr._format_scaling_tax(tax, THRESHOLDS)
        assert "from: tiny/1t=5×, medium/8t=12×" in out

    def test_pass_verdict_symbol_and_no_rule(self):
        tax = _tax(ood=[_ood_result(tier="ood_large", thread=4, factor=4.5,
                                    tax=1.1, verdict="PASS", ref=5.0)])
        out = vr._format_scaling_tax(tax, THRESHOLDS)
        assert "✓ ood_large" in out
        assert "speedup=    4.5×" in out
        assert "vs 5.0×" in out
        assert "tax=  1.1×" in out
        # PASS rows omit the "(tax > ...)" rule trailer.
        assert "(tax >" not in out.split("PASS")[1].split("\n")[0]

    def test_soft_verdict_symbol_and_rule(self):
        tax = _tax(soft=1, ood=[_ood_result(verdict="SOFT", tax=1.7,
                                            thr_used=1.5,
                                            thr_label="ood_large_soft")])
        out = vr._format_scaling_tax(tax, THRESHOLDS)
        assert "⚠ ood_large" in out
        assert "(tax > 2×, ood_large_soft)" in out  # 1.5 rounds to 2 via :.0f
        assert "1 SOFT FLAG" in out
        assert "SOFT FLAG means" in out

    def test_hard_verdict_symbol_and_footer(self):
        tax = _tax(hard=1, ood=[_ood_result(verdict="HARD", tax=8.0,
                                            thr_used=5.0, thr_label="hard_fail")])
        out = vr._format_scaling_tax(tax, THRESHOLDS)
        assert "✗ ood_large" in out
        assert "1 HARD FAIL" in out
        assert "HARD FAIL means tax > 5× the thread-matched dev speedup" in out
        assert "DISCOVERY:" in out

    def test_zero_factor_renders_infinity(self):
        tax = _tax(hard=1, ood=[_ood_result(verdict="ZERO", factor=None,
                                            tax=float("inf"))])
        out = vr._format_scaling_tax(tax, THRESHOLDS)
        assert "speedup=      0×" in out
        assert "tax=     ∞" in out
        assert "✗ ood_large" in out

    def test_big_factor_no_decimal(self):
        tax = _tax(ood=[_ood_result(factor=120.0, verdict="PASS")])
        out = vr._format_scaling_tax(tax, THRESHOLDS)
        # factors >= 100 print with :.0f (no decimal).
        assert "speedup=    120×" in out

    def test_fallback_global_marker(self):
        tax = _tax(ood=[_ood_result(ref_kind="fallback_global", verdict="PASS")])
        out = vr._format_scaling_tax(tax, THRESHOLDS)
        assert "(global*)" in out
        assert "no dev cell at this thread count" in out

    def test_clean_when_no_fails_or_flags(self):
        tax = _tax(hard=0, soft=0, ood=[_ood_result(verdict="PASS")])
        out = vr._format_scaling_tax(tax, THRESHOLDS)
        assert "all OOD cells within healthy scaling tax." in out
        assert "0 HARD FAIL, 0 SOFT FLAG" in out

    def test_empty_geom_by_thread_skips_per_thread_line(self):
        tax = _tax(by_thread={}, ood=[_ood_result(verdict="PASS")])
        out = vr._format_scaling_tax(tax, THRESHOLDS)
        assert "per-thread (used for tax)" not in out

    def test_missing_dev_reference_omits_ref_str(self):
        r = _ood_result(verdict="PASS")
        r["dev_reference"] = None
        tax = _tax(ood=[r])
        out = vr._format_scaling_tax(tax, THRESHOLDS)
        # No "vs X×" segment when dev_reference is falsy.
        assert "vs " not in out

    def test_out_of_count_in_verdict_line(self):
        tax = _tax(hard=1, soft=1, ood=[
            _ood_result(tier="ood_large", verdict="HARD", tax=9.0),
            _ood_result(tier="ood_xlarge", verdict="SOFT", tax=2.5,
                        thr_used=2.0, thr_label="ood_xlarge_soft"),
        ])
        out = vr._format_scaling_tax(tax, THRESHOLDS)
        assert "(out of 2 OOD cells)" in out


# --------------------------------------------------------------------------
# _render_verify_matrix — drive the full six-panel figure end to end
# --------------------------------------------------------------------------

def _assert_outputs_written(out_base):
    for ext in ("png", "pdf", "svg"):
        p = out_base.parent / f"{out_base.name}.{ext}"
        assert p.is_file(), f"missing {p}"
        assert p.stat().st_size > 0


class TestRenderVerifyMatrix:
    def test_minimal_single_cell_no_metrics(self, tmp_path):
        cells = [_cell(1, "tiny")]
        out_base = tmp_path / "verify"
        vr._render_verify_matrix(cells, "mytask", str(out_base), plt,
                                 metrics_spec=[], n_reps=1)
        _assert_outputs_written(out_base)

    def test_multi_tier_multi_thread_with_metrics(self, tmp_path):
        metrics_spec = [
            {"name": "jaccard", "comparator": "gte", "threshold": 0.9},
            {"name": "max_diff", "comparator": "lte", "threshold": 0.05},
        ]
        cells = [
            _cell(1, "tiny", metrics={"jaccard": 0.99, "max_diff": 0.001}),
            _cell(8, "tiny", metrics={"jaccard": 0.98, "max_diff": 0.002}),
            _cell(1, "medium", metrics={"jaccard": 0.95, "max_diff": 0.01}),
            _cell(1, "ood_large", metrics={"jaccard": 0.80, "max_diff": 0.04},
                  base=100.0, turbo=40.0),
        ]
        tax = _tax(ood=[_ood_result(tier="ood_large", thread=1, factor=2.5,
                                    tax=2.0, verdict="HARD")], hard=1)
        out_base = tmp_path / "verify"
        vr._render_verify_matrix(cells, "mytask", str(out_base), plt,
                                 metrics_spec=metrics_spec, n_reps=3,
                                 scaling_tax=tax)
        _assert_outputs_written(out_base)

    def test_oom_cell_renders(self, tmp_path):
        cells = [
            _cell(1, "tiny", metrics={"jaccard": 0.99}),
            _cell(1, "ood_xlarge", oom=True, base=0.0, turbo=0.0,
                  base_mb=0.0, turbo_mb=0.0, verdict="OOM",
                  metrics={"jaccard": None}),
        ]
        metrics_spec = [{"name": "jaccard", "comparator": "gte", "threshold": 0.9}]
        out_base = tmp_path / "verify"
        vr._render_verify_matrix(cells, "t", str(out_base), plt,
                                 metrics_spec=metrics_spec, n_reps=2)
        _assert_outputs_written(out_base)

    def test_crash_cell_renders(self, tmp_path):
        cells = [
            _cell(1, "tiny", metrics={"jaccard": 0.99}, verdict="PASS"),
            _cell(8, "tiny", any_crash=True, verdict="FAIL",
                  metrics={"jaccard": None}),
        ]
        metrics_spec = [{"name": "jaccard", "comparator": "gte", "threshold": 0.9}]
        out_base = tmp_path / "verify"
        vr._render_verify_matrix(cells, "t", str(out_base), plt,
                                 metrics_spec=metrics_spec, n_reps=3)
        _assert_outputs_written(out_base)

    def test_context_cell_hatched(self, tmp_path):
        cells = [
            _cell(1, "tiny", metrics={"jaccard": 0.99}),
            _cell(1, "ood_large", in_run=False, verdict="—",
                  metrics={"jaccard": 0.85}, base=50.0, turbo=20.0),
        ]
        metrics_spec = [{"name": "jaccard", "comparator": "gte", "threshold": 0.9}]
        out_base = tmp_path / "verify"
        vr._render_verify_matrix(cells, "t", str(out_base), plt,
                                 metrics_spec=metrics_spec, n_reps=1)
        _assert_outputs_written(out_base)

    def test_no_baseline_memory_path(self, tmp_path):
        # baseline_speed and baseline_peak_mb both zero -> "(no baseline)"
        # branches in panels A and C.
        cells = [_cell(1, "tiny", base=0.0, base_mb=0.0,
                       speedup_pct=0.0, metrics={})]
        out_base = tmp_path / "verify"
        vr._render_verify_matrix(cells, "t", str(out_base), plt,
                                 metrics_spec=[], n_reps=1)
        _assert_outputs_written(out_base)

    def test_log_scale_wide_dynamic_range(self, tmp_path):
        # speeds span >20x so panels A/C switch to log scale.
        cells = [
            _cell(1, "tiny", base=1.0, turbo=0.5, base_mb=10.0, turbo_mb=8.0),
            _cell(1, "ood_xlarge", base=500.0, turbo=100.0,
                  base_mb=40000.0, turbo_mb=20000.0),
        ]
        out_base = tmp_path / "verify"
        vr._render_verify_matrix(cells, "t", str(out_base), plt,
                                 metrics_spec=[], n_reps=1)
        _assert_outputs_written(out_base)

    def test_multi_rep_cv_panel(self, tmp_path):
        cells = [
            _cell(1, "tiny", cv=12.0, metrics={"jaccard": 0.99}),
            _cell(8, "tiny", cv=42.0, metrics={"jaccard": 0.97}),  # >30 -> red
            _cell(1, "medium", cv=None, metrics={"jaccard": 0.95}),  # "—"
        ]
        metrics_spec = [{"name": "jaccard", "comparator": "gte", "threshold": 0.9}]
        out_base = tmp_path / "verify"
        vr._render_verify_matrix(cells, "t", str(out_base), plt,
                                 metrics_spec=metrics_spec, n_reps=3)
        _assert_outputs_written(out_base)

    def test_scaling_tax_not_applicable_footer(self, tmp_path):
        cells = [_cell(1, "tiny", metrics={"jaccard": 0.99})]
        metrics_spec = [{"name": "jaccard", "comparator": "gte", "threshold": 0.9}]
        tax = _tax(applicable=False, reason="no OOD-tier cells in matrix")
        out_base = tmp_path / "verify"
        vr._render_verify_matrix(cells, "t", str(out_base), plt,
                                 metrics_spec=metrics_spec, n_reps=1,
                                 scaling_tax=tax)
        _assert_outputs_written(out_base)

    def test_stochastic_metric_label(self, tmp_path):
        # metric with noise_multiplier/absolute_floor instead of threshold
        # exercises the alternate yticks label path in panel D.
        metrics_spec = [{"name": "max_diff", "comparator": "lte",
                         "noise_multiplier": 2.0, "absolute_floor": 0.05}]
        cells = [_cell(1, "tiny", metrics={"max_diff": 0.01})]
        out_base = tmp_path / "verify"
        vr._render_verify_matrix(cells, "t", str(out_base), plt,
                                 metrics_spec=metrics_spec, n_reps=1)
        _assert_outputs_written(out_base)


# --------------------------------------------------------------------------
# Individual panels — direct, with a throwaway axis, for branch coverage
# --------------------------------------------------------------------------

def _fresh_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return fig, ax


class TestPanelsDirect:
    def test_panel_concordance_no_metrics_placeholder(self):
        fig, ax = _fresh_ax()
        try:
            vr._panel_concordance(ax, [_cell(1, "tiny")], [], plt)
        finally:
            plt.close(fig)

    def test_panel_concordance_empty_cells(self):
        fig, ax = _fresh_ax()
        metrics_spec = [{"name": "j", "comparator": "gte", "threshold": 0.9}]
        try:
            vr._panel_concordance(ax, [], metrics_spec, plt)
        finally:
            plt.close(fig)

    def test_panel_concordance_nan_and_missing_values(self):
        fig, ax = _fresh_ax()
        metrics_spec = [{"name": "j", "comparator": "gte", "threshold": 0.9}]
        cells = [
            _cell(1, "tiny", metrics={"j": float("nan")}),
            _cell(8, "tiny", metrics={}),          # missing -> "—"
            _cell(1, "medium", metrics={"j": 0.5}),  # fail (below threshold)
        ]
        try:
            vr._panel_concordance(ax, cells, metrics_spec, plt)
        finally:
            plt.close(fig)

    def test_panel_concordance_threshold_none_skips(self):
        fig, ax = _fresh_ax()
        # absolute_floor None and no threshold -> thresh is None -> skip ramp.
        metrics_spec = [{"name": "j", "comparator": "gte"}]
        cells = [_cell(1, "tiny", metrics={"j": 0.99})]
        try:
            vr._panel_concordance(ax, cells, metrics_spec, plt)
        finally:
            plt.close(fig)

    def test_panel_cv_single_rep_placeholder(self):
        fig, ax = _fresh_ax()
        try:
            vr._panel_cv(ax, [_cell(1, "tiny", cv=None)], 1, plt)
        finally:
            plt.close(fig)

    def test_panel_cv_all_none_values(self):
        fig, ax = _fresh_ax()
        cells = [_cell(1, "tiny", cv=None), _cell(8, "tiny", cv=None)]
        try:
            vr._panel_cv(ax, cells, 3, plt)
        finally:
            plt.close(fig)

    def test_panel_verdict_no_scaling_tax(self):
        fig, ax = _fresh_ax()
        try:
            vr._panel_verdict(ax, [_cell(1, "tiny")], None, plt)
        finally:
            plt.close(fig)

    def test_panel_verdict_all_verdict_kinds(self):
        fig, ax = _fresh_ax()
        cells = [
            _cell(1, "tiny", verdict="PASS"),
            _cell(8, "tiny", verdict="FAIL"),
            _cell(1, "medium", any_crash=True, verdict="FAIL"),
            _cell(1, "ood_large", oom=True, verdict="OOM"),
            _cell(1, "ood_xlarge", in_run=False, verdict="—"),
        ]
        tax = _tax(hard=1, soft=1, ood=[
            _ood_result(tier="ood_large", verdict="HARD", tax=8.0),
            _ood_result(tier="ood_xlarge", verdict="SOFT", tax=2.5,
                        factor=None),
        ])
        try:
            vr._panel_verdict(ax, cells, tax, plt)
        finally:
            plt.close(fig)

    def test_panel_verdict_tax_not_applicable(self):
        fig, ax = _fresh_ax()
        tax = _tax(applicable=False, reason="dev-only matrix")
        try:
            vr._panel_verdict(ax, [_cell(1, "tiny")], tax, plt)
        finally:
            plt.close(fig)

    def test_panel_verdict_single_ood_singular_count(self):
        # Exercises the singular "1 OOD cell" branch (no trailing 's').
        fig, ax = _fresh_ax()
        tax = _tax(ood=[_ood_result(verdict="PASS")])
        try:
            vr._panel_verdict(ax, [_cell(1, "tiny"), _cell(1, "ood_large")],
                              tax, plt)
        finally:
            plt.close(fig)
