"""Deep coverage tests for zyme.scan_speedups.

Targets gaps not covered by tests/test_scan_speedups.py (whose 9 stale
failures from the patched_min_reps 3->2 default change are NOT touched here):
  - _is_experimental / FUSION_PATCHES / _has_headline_schedule
  - _load_oom_skip (file absent) / _oom_skipped
  - _normalize_platform / _read_tsv_rows OSError / _parse_reps / _rep_diff_pct
  - raw-shard helpers: _raw_note_skip / _raw_date / _raw_threads /
    _raw_peak_mb / _row_platform / _matches_platform_filter /
    _discover_raw_speedup_paths / _median
  - OOD-tier INFO downgrades (low_reps + rep variance + thread_gap fullt)
  - thread_gap under the default schedule (mac expects {1,4,8})
  - discover_patch_tsvs / audit_all_patches over the real shipped tree
  - render_table(detail=True) / render_markdown details / has_fail /
    has_warn_or_fail / _platform_filter_label
"""
from __future__ import annotations

from pathlib import Path

import pytest

import zyme.scan_speedups as ss
from zyme.scan_speedups import (
    Issue,
    PatchCoverage,
    _discover_raw_speedup_paths,
    _has_headline_schedule,
    _is_blank_threads,
    _is_experimental,
    _is_ood_tier,
    _load_oom_skip,
    _matches_platform_filter,
    _median,
    _normalize_platform,
    _oom_skipped,
    _parse_int,
    _parse_reps,
    _platform_filter_label,
    _raw_date,
    _raw_note_skip,
    _raw_peak_mb,
    _raw_threads,
    _read_tsv_rows,
    _rep_diff_pct,
    _row_platform,
    audit_all_patches,
    audit_patch,
    discover_patch_tsvs,
    has_fail,
    has_warn_or_fail,
    render_markdown,
    render_table,
)


HEADER = "\t".join([
    "patch", "package_version", "tier", "threads", "platform", "dataset",
    "variant", "status", "n_reps", "sec_reps", "sec_mean", "sec_median",
    "mem_reps", "mem_mean", "mem_median", "speedup_x_reps", "speedup_x_mean",
    "speedup_x_median", "pass_rate", "metrics_json_median", "fw_versions",
    "ts_first", "ts_last",
])


def _row(**over) -> str:
    d = dict(
        patch="x", package_version="X 1.0", tier="small", threads="1",
        platform="Windows", dataset="ds", variant="baseline",
        status="ok", n_reps="3",
        sec_reps="1, 1, 1", sec_mean="1", sec_median="1",
        mem_reps="100, 100, 100", mem_mean="100", mem_median="100",
        speedup_x_reps="", speedup_x_mean="", speedup_x_median="",
        pass_rate="1", metrics_json_median="", fw_versions="0.3.0",
        ts_first="2026-06-01", ts_last="2026-06-01",
    )
    d.update(over)
    return "\t".join(d[c] for c in HEADER.split("\t"))


def _write_tsv(tmp_path: Path, *rows: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "speedups_finalized.tsv"
    p.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return p


def _kinds(cov: PatchCoverage) -> list[str]:
    return [i.kind for i in cov.issues]


# --------------------------------------------------------------------------
# experimental / headline detection
# --------------------------------------------------------------------------

class TestExperimentalDetection:
    def test_has_headline_schedule(self):
        assert _has_headline_schedule("scanpy_normalize") is True
        assert _has_headline_schedule("seurat_markers") is True
        assert _has_headline_schedule("rctd") is False

    def test_is_experimental_fusion(self):
        assert _is_experimental("tradeseq") is True
        assert _is_experimental("mast") is True

    def test_is_experimental_headline(self):
        assert _is_experimental("scanpy_pca") is True

    def test_not_experimental(self):
        assert _is_experimental("rctd") is False


# --------------------------------------------------------------------------
# oom-skip (file absent on this host -> empty set)
# --------------------------------------------------------------------------

class TestOomSkip:
    def test_load_returns_set(self):
        out = _load_oom_skip()
        assert isinstance(out, set)

    def test_cached_on_second_call(self):
        a = _load_oom_skip()
        b = _load_oom_skip()
        assert a is b

    def test_oom_skipped_false_without_filter(self):
        assert _oom_skipped("x", "small", "baseline", None) is False

    def test_oom_skipped_requires_mac_filter(self):
        # Even if a cell were in the skip set, only --mac-only consults it.
        assert _oom_skipped("x", "small", "baseline", "win") is False


# --------------------------------------------------------------------------
# small parse helpers
# --------------------------------------------------------------------------

class TestParseHelpers:
    def test_normalize_platform(self):
        assert _normalize_platform("Windows 10") == "win"
        assert _normalize_platform("macOS 14") == "mac"
        assert _normalize_platform("darwin") == "mac"
        assert _normalize_platform("") == "unknown"
        assert _normalize_platform("linux") == "linux"

    def test_read_tsv_rows_oserror(self, tmp_path: Path):
        assert _read_tsv_rows(tmp_path / "nope.tsv") == []

    def test_parse_int(self):
        assert _parse_int("4") == 4
        assert _parse_int("") is None
        assert _parse_int("x") is None

    def test_parse_reps_ok(self):
        assert _parse_reps("1.0, 2.0, 3.0") == [1.0, 2.0, 3.0]

    def test_parse_reps_empty(self):
        assert _parse_reps("") == []

    def test_parse_reps_bad_token_returns_empty(self):
        assert _parse_reps("1.0, notnum") == []

    def test_rep_diff_pct(self):
        assert _rep_diff_pct([1.0, 1.5]) == pytest.approx(40.0)

    def test_rep_diff_pct_not_two(self):
        assert _rep_diff_pct([1.0]) is None

    def test_rep_diff_pct_zero_mean(self):
        assert _rep_diff_pct([0.0, 0.0]) is None

    def test_is_blank_threads(self):
        assert _is_blank_threads("") is True
        assert _is_blank_threads("unknown") is True
        assert _is_blank_threads("4") is False

    def test_is_ood_tier(self):
        assert _is_ood_tier("ood_large1") is True
        assert _is_ood_tier("small") is False

    def test_median(self):
        assert _median([3.0, 1.0, 2.0]) == 2.0
        assert _median([1.0, 2.0]) == 1.5
        assert _median([]) is None


# --------------------------------------------------------------------------
# raw-shard helpers
# --------------------------------------------------------------------------

class TestRawHelpers:
    def test_raw_note_skip(self):
        assert _raw_note_skip("migrated: full") is True
        assert _raw_note_skip("absorbed: x") is True
        assert _raw_note_skip("ok") is False

    def test_raw_date(self):
        assert _raw_date("2026-06-05T12:00:00") == "2026-06-05"
        assert _raw_date("") == ""

    def test_raw_threads_prefers_system_threads(self):
        assert _raw_threads({"system_threads": "8", "threads": "1"}) == "8"
        assert _raw_threads({"threads": "4"}) == "4"
        assert _raw_threads({}) == ""

    def test_raw_peak_mb(self):
        assert _raw_peak_mb({"peak_mb": "1000"}) == 1000.0
        assert _raw_peak_mb({"peak_mb": "0"}) is None
        assert _raw_peak_mb({"peak_mb": "bad"}) is None

    def test_row_platform(self):
        assert _row_platform({"platform": "macOS"}) == "mac"
        assert _row_platform({"system_os": "Windows"}) == "win"

    def test_matches_platform_filter(self):
        assert _matches_platform_filter({"platform": "macOS"}, "mac") is True
        assert _matches_platform_filter({"platform": "Windows"}, "mac") is False
        assert _matches_platform_filter({"platform": "macOS"}, None) is True

    def test_discover_raw_paths_filters_by_platform(self, tmp_path: Path):
        (tmp_path / "speedups.mac.tsv").write_text("x\n")
        (tmp_path / "speedups.win.tsv").write_text("x\n")
        mac_only = _discover_raw_speedup_paths(tmp_path, "mac")
        assert [p.name for p in mac_only] == ["speedups.mac.tsv"]
        both = _discover_raw_speedup_paths(tmp_path, None)
        assert {p.name for p in both} == {"speedups.mac.tsv", "speedups.win.tsv"}


# --------------------------------------------------------------------------
# OOD-tier INFO downgrades
# --------------------------------------------------------------------------

class TestOodInfoDowngrade:
    def test_low_reps_on_ood_is_info(self, tmp_path: Path):
        p = _write_tsv(tmp_path,
            _row(tier="ood_large1", variant="baseline", threads="1", n_reps="1"),
            _row(tier="ood_large1", variant="patched", threads="1", n_reps="1"),
        )
        cov = audit_patch("x", p)
        low = [i for i in cov.issues if i.kind == "low_reps"]
        assert low and all(i.severity == "INFO" for i in low)

    def test_rep_variance_on_ood_is_info(self, tmp_path: Path):
        # Spread 3s (> default rep_variance_abs_sec floor of 2s) and the
        # n=2 percent diff (85%) exceeds the 10% gate.
        p = _write_tsv(tmp_path,
            _row(tier="ood_large1", variant="baseline", threads="1", n_reps="2",
                 sec_reps="2.0, 5.0", sec_mean="3.5", sec_median="3.5"),
            _row(tier="ood_large1", variant="patched", threads="1", n_reps="2",
                 sec_reps="1.0, 1.0", sec_mean="1.0", sec_median="1.0"),
        )
        cov = audit_patch("x", p)
        hv = [i for i in cov.issues if i.kind == "high_rep_variance"]
        assert hv and hv[0].severity == "INFO"


# --------------------------------------------------------------------------
# thread_gap under the default schedule (mac expects {1,4,8})
# --------------------------------------------------------------------------

class TestThreadGapDefaultSchedule:
    def test_mac_missing_fullt_warns(self, tmp_path: Path):
        # Non-headline patch on mac with only {1,4} -> default schedule {1,4,8}
        # flags missing 8 as a thread_gap WARN.
        p = _write_tsv(tmp_path,
            _row(tier="small", platform="macOS", variant="baseline", threads="1"),
            _row(tier="small", platform="macOS", variant="patched", threads="1"),
            _row(tier="small", platform="macOS", variant="baseline", threads="4"),
            _row(tier="small", platform="macOS", variant="patched", threads="4"),
        )
        cov = audit_patch("x", p)
        gaps = [i for i in cov.issues if i.kind == "thread_gap"]
        assert any(g.threads == "8" and g.severity == "WARN" for g in gaps)

    def test_ood_fullt_gap_is_info(self, tmp_path: Path):
        # On an OOD tier, the missing fullt (8 on mac) is INFO not WARN.
        p = _write_tsv(tmp_path,
            _row(tier="ood_large1", platform="macOS", variant="baseline", threads="1", n_reps="1"),
            _row(tier="ood_large1", platform="macOS", variant="patched", threads="1", n_reps="1"),
            _row(tier="ood_large1", platform="macOS", variant="baseline", threads="4", n_reps="1"),
            _row(tier="ood_large1", platform="macOS", variant="patched", threads="4", n_reps="1"),
        )
        cov = audit_patch("x", p)
        fullt_gap = [i for i in cov.issues
                     if i.kind == "thread_gap" and i.threads == "8"]
        assert fullt_gap and fullt_gap[0].severity == "INFO"

    def test_blank_threads_ignored_on_observed_schedule_platform(self, tmp_path: Path):
        # On a platform NOT in the hardcoded {win,mac} schedules (here "linux"),
        # the expected thread set is the OBSERVED set; blank threads are
        # stripped, leaving an empty expected set -> no thread_gap can fire.
        p = _write_tsv(tmp_path,
            _row(variant="baseline", platform="linux", threads=""),
            _row(variant="patched", platform="linux", threads=""),
        )
        cov = audit_patch("x", p)
        assert "thread_gap" not in _kinds(cov)


# --------------------------------------------------------------------------
# discover_patch_tsvs / audit_all_patches over the real shipped tree
# --------------------------------------------------------------------------

def _framework_root() -> Path:
    # tests/ is autozyme_cli/tests; the framework root is two up from the
    # package dir (.../autozyme_release).
    return Path(__file__).resolve().parents[2]


class TestDiscoverAndAuditAll:
    def test_discover_returns_pairs(self):
        out = discover_patch_tsvs(_framework_root())
        assert isinstance(out, list)
        # The shipped release has many patches with finalized TSVs.
        assert out, "expected at least one shipped patch with a finalized TSV"
        for name, tsv in out:
            assert isinstance(name, str)
            assert tsv.name == "speedups_finalized.tsv"

    def test_skip_experimental_drops_headline(self):
        full = {n for n, _ in discover_patch_tsvs(_framework_root())}
        skipped = {n for n, _ in discover_patch_tsvs(
            _framework_root(), skip_experimental=True)}
        # skip_experimental never adds patches; it only removes.
        assert skipped <= full

    def test_audit_all_returns_coverage(self):
        audits = audit_all_patches(_framework_root(), skip_experimental=True)
        assert all(isinstance(a, PatchCoverage) for a in audits)

    def test_discover_missing_root(self, tmp_path: Path):
        # A framework root with neither package tree -> empty list.
        assert discover_patch_tsvs(tmp_path) == []


# --------------------------------------------------------------------------
# renderers + strict helpers
# --------------------------------------------------------------------------

class TestRenderers:
    def test_platform_filter_label(self):
        assert _platform_filter_label("mac") == "macOS only"
        assert _platform_filter_label("win") == "Windows only"
        assert _platform_filter_label(None) == "all platforms"

    def test_render_table_detail(self, tmp_path: Path):
        cov = audit_patch("x", _write_tsv(tmp_path, _row(variant="patched")))
        out = render_table([cov], detail=True)
        assert "## x" in out
        assert "missing_variant" in out

    def test_render_table_platform_note(self, tmp_path: Path):
        cov = audit_patch("x", _write_tsv(tmp_path,
            _row(variant="baseline"), _row(variant="patched")))
        out = render_table([cov], platform_filter="mac")
        assert "macOS only" in out

    def test_render_markdown_details(self, tmp_path: Path):
        cov = audit_patch("x", _write_tsv(tmp_path, _row(variant="patched")))
        md = render_markdown([cov], platform_filter="win")
        assert "Windows only" in md
        assert "### Details" in md
        assert "missing_variant" in md

    def test_render_markdown_clean_no_details(self, tmp_path: Path):
        # A genuinely clean patch needs the full win schedule {1,4,32} for the
        # tier so no thread_gap fires, plus matched baseline/patched variants.
        rows = []
        for thr in ("1", "4", "32"):
            rows.append(_row(tier="small", platform="Windows",
                             variant="baseline", threads=thr))
            rows.append(_row(tier="small", platform="Windows",
                             variant="patched", threads=thr))
        cov = audit_patch("x", _write_tsv(tmp_path, *rows))
        assert cov.is_clean, _kinds(cov)
        md = render_markdown([cov])
        assert "### Details" not in md


class TestStrictHelpers:
    def test_has_fail_true(self):
        cov = PatchCoverage("x", Path("p"), True)
        cov.issues.append(Issue("FAIL", "bad_status", "x"))
        assert has_fail([cov]) is True

    def test_has_fail_false(self):
        cov = PatchCoverage("x", Path("p"), True)
        cov.issues.append(Issue("WARN", "low_reps", "x"))
        assert has_fail([cov]) is False

    def test_has_warn_or_fail(self):
        warn = PatchCoverage("x", Path("p"), True)
        warn.issues.append(Issue("WARN", "low_reps", "x"))
        clean = PatchCoverage("y", Path("p"), True)
        assert has_warn_or_fail([warn]) is True
        assert has_warn_or_fail([clean]) is False

    def test_patch_coverage_counts_and_clean(self):
        cov = PatchCoverage("x", Path("p"), True)
        assert cov.is_clean is True
        cov.issues.append(Issue("INFO", "platform_asymmetry", "x"))
        assert cov.is_clean is False
        assert cov.counts()["INFO"] == 1
