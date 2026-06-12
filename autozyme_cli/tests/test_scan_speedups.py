"""Unit tests for scan_speedups coverage rules.

Each test builds minimal TSV fixtures, runs audit_patch, and asserts the
expected issues fire (or don't).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme.scan_speedups import (
    Issue, PatchCoverage,
    audit_patch,
    discover_patch_tsvs,
    has_fail, has_warn_or_fail,
    render_table, render_markdown, render_json_records,
)


HEADER = "\t".join([
    "patch", "package_version", "tier", "threads", "platform", "dataset",
    "variant", "status", "n_reps", "sec_reps", "sec_mean", "sec_median",
    "mem_reps", "mem_mean", "mem_median", "speedup_x_reps", "speedup_x_mean",
    "speedup_x_median", "pass_rate", "metrics_json_median", "fw_versions",
    "ts_first", "ts_last",
])


def _row(**overrides) -> str:
    defaults = dict(
        patch="x", package_version="X 1.0", tier="small", threads="1",
        platform="Windows", dataset="ds", variant="baseline",
        status="ok", n_reps="3",
        sec_reps="1, 1, 1", sec_mean="1", sec_median="1",
        mem_reps="100, 100, 100", mem_mean="100", mem_median="100",
        speedup_x_reps="", speedup_x_mean="", speedup_x_median="",
        pass_rate="1", metrics_json_median="", fw_versions="0.3.0",
        ts_first="2026-06-01", ts_last="2026-06-01",
    )
    defaults.update(overrides)
    cols = HEADER.split("\t")
    return "\t".join(defaults[c] for c in cols)


def _write_tsv(tmp_path: Path, *rows: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "speedups_finalized.tsv"
    p.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return p


def _kinds(cov: PatchCoverage) -> list[str]:
    return [i.kind for i in cov.issues]


# ----------------------------------------------------------------------------
# Rule 1: bad_status
# ----------------------------------------------------------------------------


def test_unknown_status_is_FAIL(tmp_path):
    p = _write_tsv(tmp_path,
        _row(status="weird"),
        _row(variant="patched", status="ok"),
    )
    cov = audit_patch("x", p)
    fail_issues = [i for i in cov.issues if i.kind == "bad_status"]
    assert len(fail_issues) == 1
    assert fail_issues[0].severity == "FAIL"
    assert has_fail([cov])


def test_known_statuses_dont_FAIL(tmp_path):
    # ok/OOM/N/A/partial are all valid; none should fire bad_status.
    rows = [
        _row(tier=t, status=s, n_reps="3", variant="baseline")
        for t, s in [("a", "ok"), ("b", "OOM"), ("c", "N/A"), ("d", "partial")]
    ]
    cov = audit_patch("x", _write_tsv(tmp_path, *rows))
    assert "bad_status" not in _kinds(cov)


# ----------------------------------------------------------------------------
# Rule 2: low_reps
# ----------------------------------------------------------------------------


def test_patched_n_reps_below_3_is_WARN(tmp_path):
    p = _write_tsv(tmp_path,
        _row(variant="baseline", n_reps="3"),
        _row(variant="patched", n_reps="2"),
    )
    cov = audit_patch("x", p)
    low = [i for i in cov.issues if i.kind == "low_reps"]
    assert len(low) == 1
    assert low[0].variant == "patched"
    assert low[0].severity == "WARN"


def test_baseline_n_reps_below_2_is_WARN(tmp_path):
    p = _write_tsv(tmp_path,
        _row(variant="baseline", n_reps="1"),
        _row(variant="patched", n_reps="3"),
    )
    cov = audit_patch("x", p)
    low = [i for i in cov.issues if i.kind == "low_reps"]
    assert len(low) == 1
    assert low[0].variant == "baseline"


def test_baseline_n_reps_eq_2_passes(tmp_path):
    # Default baseline_min_reps=2; baseline n_reps=2 should NOT warn.
    p = _write_tsv(tmp_path,
        _row(variant="baseline", n_reps="2"),
        _row(variant="patched", n_reps="3"),
    )
    cov = audit_patch("x", p)
    assert "low_reps" not in _kinds(cov)


def test_custom_thresholds(tmp_path):
    # patched_min_reps=5, baseline_min_reps=3
    p = _write_tsv(tmp_path,
        _row(variant="baseline", n_reps="2"),  # fails new threshold
        _row(variant="patched", n_reps="4"),   # fails new threshold
    )
    cov = audit_patch("x", p, patched_min_reps=5, baseline_min_reps=3)
    assert len([i for i in cov.issues if i.kind == "low_reps"]) == 2


def test_oom_rows_dont_trigger_low_reps(tmp_path):
    # OOM has n_reps=0 but is a documented gap — low_reps only fires on ok.
    p = _write_tsv(tmp_path,
        _row(variant="baseline", status="OOM", n_reps="0"),
        _row(variant="patched", n_reps="3"),
    )
    cov = audit_patch("x", p)
    assert "low_reps" not in _kinds(cov)


# ----------------------------------------------------------------------------
# Rule 2b/2c: rep variance (sec + mem)
# ----------------------------------------------------------------------------


def test_high_rep_variance_sec_fires(tmp_path):
    p = _write_tsv(tmp_path,
        _row(variant="baseline", n_reps="2",
             sec_reps="1.0, 1.5", sec_mean="1.25", sec_median="1.25"),
        _row(variant="patched", n_reps="2",
             sec_reps="0.5, 0.5", sec_mean="0.5", sec_median="0.5"),
    )
    cov = audit_patch("x", p)
    hv = [i for i in cov.issues if i.kind == "high_rep_variance"]
    assert len(hv) == 1
    assert hv[0].variant == "baseline"
    assert hv[0].severity == "WARN"


def test_high_mem_rep_variance_fires(tmp_path):
    p = _write_tsv(tmp_path,
        _row(variant="baseline", n_reps="2",
             mem_reps="1000, 1300", mem_mean="1150", mem_median="1150"),
        _row(variant="patched", n_reps="2",
             mem_reps="1000, 1000", mem_mean="1000", mem_median="1000"),
    )
    cov = audit_patch("x", p)
    hm = [i for i in cov.issues if i.kind == "high_mem_rep_variance"]
    assert len(hm) == 1
    assert hm[0].variant == "baseline"


def test_high_mem_rep_variance_respects_abs_floor(tmp_path):
    # 30% spread but only 30 MB absolute — below 50 MB floor.
    p = _write_tsv(tmp_path,
        _row(variant="baseline", n_reps="2",
             mem_reps="100, 130", mem_mean="115", mem_median="115"),
        _row(variant="patched", n_reps="3"),
    )
    cov = audit_patch("x", p, mem_variance_abs_mb=50.0)
    assert "high_mem_rep_variance" not in _kinds(cov)


RAW_HEADER = "\t".join([
    "timestamp", "patch_name", "tier", "dataset", "rep_idx", "variant",
    "sec", "peak_mb", "note", "system_os", "system_threads",
])


def _raw_row(**overrides) -> str:
    defaults = dict(
        timestamp="2026-05-27T12:00:00", patch_name="x", tier="ood_large1",
        dataset="ds", rep_idx="1", variant="baseline", sec="1.0",
        peak_mb="10000", note="", system_os="macOS 24.4.0", system_threads="1",
    )
    defaults.update(overrides)
    return "\t".join(defaults[c] for c in RAW_HEADER.split("\t"))


def _write_raw(tmp_path: Path, *rows: str) -> None:
    (tmp_path / "speedups.mac.tsv").write_text(
        RAW_HEADER + "\n" + "\n".join(rows) + "\n"
    )


def test_baseline_era_drift_from_raw_tsv(tmp_path):
    fin = _write_tsv(tmp_path,
        _row(tier="ood_large1", variant="baseline", n_reps="3"),
        _row(tier="ood_large1", variant="patched", n_reps="3"),
    )
    _write_raw(tmp_path,
        _raw_row(variant="baseline", timestamp="2026-05-27T12:00:00",
                 peak_mb="10000", rep_idx="1"),
        _raw_row(variant="baseline", timestamp="2026-06-05T12:00:00",
                 peak_mb="5000", rep_idx="1"),
        _raw_row(variant="patched", timestamp="2026-05-27T12:00:00",
                 peak_mb="10200", rep_idx="1"),
    )
    cov = audit_patch("x", fin)
    drift = [i for i in cov.issues if i.kind == "baseline_era_drift"]
    assert len(drift) == 1
    assert drift[0].variant == "baseline"


def test_mac_only_skips_win_rows_and_platform_asymmetry(tmp_path):
    p = _write_tsv(tmp_path,
        _row(variant="baseline", platform="Windows", n_reps="3"),
        _row(variant="patched", platform="Windows", n_reps="3"),
        _row(tier="large", variant="baseline", platform="macOS", n_reps="3"),
        _row(tier="large", variant="patched", platform="macOS", n_reps="3"),
    )
    cov_all = audit_patch("x", p)
    cov_mac = audit_patch("x", p, platform_filter="mac")
    assert cov_all.n_rows == 4
    assert cov_mac.n_rows == 2
    assert "platform_asymmetry" in _kinds(cov_all)
    assert "platform_asymmetry" not in _kinds(cov_mac)


def test_win_only_filters_finalized_rows(tmp_path):
    p = _write_tsv(tmp_path,
        _row(variant="baseline", platform="Windows", n_reps="2",
             sec_reps="1.0, 1.5", sec_mean="1.25", sec_median="1.25"),
        _row(variant="patched", platform="Windows", n_reps="3"),
        _row(variant="baseline", platform="macOS", n_reps="3"),
        _row(variant="patched", platform="macOS", n_reps="3"),
    )
    cov = audit_patch("x", p, platform_filter="win")
    assert cov.n_rows == 2
    hv = [i for i in cov.issues if i.kind == "high_rep_variance"]
    assert len(hv) == 1
    assert hv[0].platform == "win"


def test_stale_baseline_pairing_from_raw_tsv(tmp_path):
    fin = _write_tsv(tmp_path,
        _row(tier="ood_large1", variant="baseline", n_reps="3"),
        _row(tier="ood_large1", variant="patched", n_reps="3"),
    )
    _write_raw(tmp_path,
        _raw_row(variant="baseline", timestamp="2026-05-27T12:00:00",
                 peak_mb="10000", rep_idx="1"),
        _raw_row(variant="baseline", timestamp="2026-06-05T12:00:00",
                 peak_mb="5000", rep_idx="2"),
        _raw_row(variant="patched", timestamp="2026-05-27T12:00:00",
                 peak_mb="10200", rep_idx="1"),
    )
    cov = audit_patch("x", fin)
    stale = [i for i in cov.issues if i.kind == "stale_baseline_pairing"]
    assert len(stale) == 1
    assert stale[0].variant == "patched"
    assert "2026-05-27" in stale[0].msg
    assert "2026-06-05" in stale[0].msg


# ----------------------------------------------------------------------------
# Rule 3: missing_variant
# ----------------------------------------------------------------------------


def test_patched_without_baseline_fires_missing_variant(tmp_path):
    p = _write_tsv(tmp_path,
        _row(variant="patched", n_reps="3"),
    )
    cov = audit_patch("x", p)
    mv = [i for i in cov.issues if i.kind == "missing_variant"]
    assert len(mv) == 1
    assert mv[0].variant == "baseline"
    assert mv[0].severity == "WARN"


def test_baseline_without_patched_fires_missing_variant(tmp_path):
    p = _write_tsv(tmp_path,
        _row(variant="baseline", n_reps="3"),
    )
    cov = audit_patch("x", p)
    mv = [i for i in cov.issues if i.kind == "missing_variant"]
    assert len(mv) == 1
    assert mv[0].variant == "patched"


def test_baseline_patched_pair_is_clean(tmp_path):
    p = _write_tsv(tmp_path,
        _row(variant="baseline", n_reps="3"),
        _row(variant="patched", n_reps="3"),
    )
    cov = audit_patch("x", p)
    assert "missing_variant" not in _kinds(cov)


# ----------------------------------------------------------------------------
# Rule 4: platform_asymmetry (INFO)
# ----------------------------------------------------------------------------


def test_single_platform_doesnt_trigger_asymmetry(tmp_path):
    # All Win — no platform asymmetry fires (Mac never appears anywhere).
    p = _write_tsv(tmp_path,
        _row(variant="baseline", platform="Windows", n_reps="3"),
        _row(variant="patched", platform="Windows", n_reps="3"),
        _row(tier="medium", variant="baseline", platform="Windows", n_reps="3"),
        _row(tier="medium", variant="patched", platform="Windows", n_reps="3"),
    )
    cov = audit_patch("x", p)
    assert "platform_asymmetry" not in _kinds(cov)


def test_win_tier_missing_on_mac_is_INFO(tmp_path):
    # Patch has both Mac and Win somewhere → cross-platform claim → asymmetry checked.
    p = _write_tsv(tmp_path,
        # small on both platforms
        _row(tier="small", variant="baseline", platform="Windows", n_reps="3"),
        _row(tier="small", variant="patched", platform="Windows", n_reps="3"),
        _row(tier="small", variant="baseline", platform="macOS", n_reps="3"),
        _row(tier="small", variant="patched", platform="macOS", n_reps="3"),
        # large only on Windows
        _row(tier="large", variant="baseline", platform="Windows", n_reps="3"),
        _row(tier="large", variant="patched", platform="Windows", n_reps="3"),
    )
    cov = audit_patch("x", p)
    asym = [i for i in cov.issues if i.kind == "platform_asymmetry"]
    assert len(asym) == 1
    assert asym[0].severity == "INFO"
    assert asym[0].platform == "mac"
    assert asym[0].tier == "large"


# ----------------------------------------------------------------------------
# Rule 5: thread_gap
# ----------------------------------------------------------------------------


def test_single_thread_patch_no_thread_gap(tmp_path):
    # All rows at threads=1; the patch is single-thread by construction.
    p = _write_tsv(tmp_path,
        _row(variant="baseline", threads="1", n_reps="3"),
        _row(variant="patched", threads="1", n_reps="3"),
    )
    cov = audit_patch("x", p)
    assert "thread_gap" not in _kinds(cov)


def test_multi_thread_patch_with_gap_fires_WARN(tmp_path):
    # threads=1 and threads=4 both appear; one (tier, platform) is missing
    # the threads=4 row → WARN.
    p = _write_tsv(tmp_path,
        # small has both thread settings
        _row(tier="small", variant="baseline", threads="1", n_reps="3"),
        _row(tier="small", variant="patched", threads="1", n_reps="3"),
        _row(tier="small", variant="baseline", threads="4", n_reps="3"),
        _row(tier="small", variant="patched", threads="4", n_reps="3"),
        # large only has threads=1
        _row(tier="large", variant="baseline", threads="1", n_reps="3"),
        _row(tier="large", variant="patched", threads="1", n_reps="3"),
    )
    cov = audit_patch("x", p)
    gaps = [i for i in cov.issues if i.kind == "thread_gap"]
    assert len(gaps) == 1
    assert gaps[0].threads == "4"
    assert gaps[0].tier == "large"
    assert gaps[0].severity == "WARN"


# ----------------------------------------------------------------------------
# File-level edge cases
# ----------------------------------------------------------------------------


def test_missing_tsv_is_FAIL(tmp_path):
    cov = audit_patch("x", tmp_path / "nope.tsv")
    assert cov.issues[0].kind == "missing_tsv"
    assert cov.issues[0].severity == "FAIL"


def test_empty_tsv_is_FAIL(tmp_path):
    p = tmp_path / "speedups_finalized.tsv"
    p.write_text(HEADER + "\n")
    cov = audit_patch("x", p)
    assert cov.issues[0].kind == "empty_tsv"
    assert cov.issues[0].severity == "FAIL"


# ----------------------------------------------------------------------------
# Renderers
# ----------------------------------------------------------------------------


def test_render_table_handles_empty(tmp_path):
    assert "no patches" in render_table([])


def test_render_table_with_clean_patch(tmp_path):
    cov = audit_patch("x", _write_tsv(tmp_path,
        _row(variant="baseline", n_reps="3"),
        _row(variant="patched", n_reps="3"),
    ))
    out = render_table([cov])
    assert "clean" in out


def test_render_markdown_with_issues(tmp_path):
    cov = audit_patch("x", _write_tsv(tmp_path, _row(variant="patched", n_reps="3")))
    md = render_markdown([cov])
    assert "## Speedup coverage" in md
    assert "missing_variant" in md


def test_render_json_clean_patch_emits_PASS(tmp_path):
    cov = audit_patch("x", _write_tsv(tmp_path,
        _row(variant="baseline", n_reps="3"),
        _row(variant="patched", n_reps="3"),
    ))
    records = render_json_records([cov])
    assert len(records) == 1
    assert records[0]["severity"] == "PASS"
    assert records[0]["kind"] == "clean"


def test_strict_helpers(tmp_path):
    bad = audit_patch("bad", _write_tsv(tmp_path / "a", _row(status="weird")))
    warned = audit_patch("warn", _write_tsv(tmp_path / "b",
        _row(variant="patched", n_reps="1"),
        _row(variant="baseline", n_reps="3"),
    ))
    clean = audit_patch("clean", _write_tsv(tmp_path / "c",
        _row(variant="baseline", n_reps="3"),
        _row(variant="patched", n_reps="3"),
    ))
    # Each variant uses its own subdir so _write_tsv doesn't overwrite.
    assert has_fail([bad])
    assert not has_fail([warned, clean])
    assert has_warn_or_fail([bad])
    assert has_warn_or_fail([warned])
    assert not has_warn_or_fail([clean])
