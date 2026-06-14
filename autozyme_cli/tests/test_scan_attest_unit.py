"""Unit tests for zyme.scan_attest — package patch-coverage audit.

The module walks the autozyme_r / autozyme_py packages, parses each patch's
finalized (or legacy long-format) speedups TSV, classifies per-(platform,tier)
status, computes drift alerts, and renders text/markdown/json. Everything here
is pure given on-disk fixtures: we build tiny finalized/long TSVs in tmp_path
and a fake framework tree to exercise enumeration + parsing + rendering.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import zyme.scan_attest as sa
from zyme.scan_attest import (
    DriftAlert,
    PatchAudit,
    TierStatus,
    TIERS,
    TIERS_5,
    TIERS_6,
    _concordance_min,
    _enumerate_patches,
    _finalized_group_key,
    _float_cell,
    _fmt_conc,
    _fmt_fold,
    _normalize_platform,
    _parse_speedups,
    _platform_tag,
    _short_note,
    audit_packages,
    normalize_tier,
    render_json_records,
    render_markdown,
    render_table,
)


# ==========================================================================
# Finalized-TSV fixture builders
# ==========================================================================

FINALIZED_HEADER = [
    "patch", "tier", "platform", "threads", "dataset", "package_version",
    "variant", "status", "speedup_x_mean", "n_reps", "ts_first", "ts_last",
    "metrics_json_median",
]


def _fin_row(**ov) -> dict:
    base = dict(
        patch="p", tier="small", platform="macOS", threads="1", dataset="ds",
        package_version="1.0", variant="patched", status="ok",
        speedup_x_mean="5.0", n_reps="3", ts_first="2026-06-01",
        ts_last="2026-06-01", metrics_json_median="",
    )
    base.update(ov)
    return base


def _write_finalized(path: Path, *rows: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(FINALIZED_HEADER)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in FINALIZED_HEADER))
    path.write_text("\n".join(lines) + "\n")
    return path


def _audit_from_finalized(path: Path, *rows: dict) -> PatchAudit:
    _write_finalized(path, *rows)
    p = PatchAudit(
        name="p", language="Py", patch_path=path, speedups_path=path,
        speedups_exists=True, lifted_from_task=None,
    )
    _parse_speedups(p)
    return p


# Long-format (raw verify) header — what summarize_batches consumes.
LONG_HEADER = [
    "timestamp", "patch_name", "tier", "dataset", "rep_idx", "variant",
    "sec", "speedup_pct", "speedup_x", "peak_mb", "peak_mb_change_pct",
    "peak_mb_fold", "pass", "metrics_json", "framework_version",
    "package_version", "note", "system_os", "system_cpu", "system_ram_gb",
    "system_threads",
]


def _long_row(**ov) -> dict:
    base = {
        "timestamp": "2026-06-01T00:00:00", "patch_name": "p", "tier": "small",
        "dataset": "ds", "rep_idx": "1", "variant": "baseline", "sec": "2.0",
        "speedup_pct": "", "speedup_x": "", "peak_mb": "100",
        "peak_mb_change_pct": "", "peak_mb_fold": "", "pass": "",
        "metrics_json": "{}", "framework_version": "0.3.0",
        "package_version": "1.0", "note": "", "system_os": "macOS 24.4",
        "system_cpu": "arm64", "system_ram_gb": "16", "system_threads": "1",
    }
    base.update(ov)
    return base


def _write_long(path: Path, *rows: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(LONG_HEADER)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in LONG_HEADER))
    path.write_text("\n".join(lines) + "\n")
    return path


# ==========================================================================
# normalize_tier  +  TIERS constants
# ==========================================================================

class TestTiers:
    def test_tiny_aliases_to_small(self):
        assert normalize_tier("tiny") == "small"

    def test_unchanged_passthrough(self):
        assert normalize_tier("medium") == "medium"
        assert normalize_tier("ood_large1") == "ood_large1"

    def test_tier_families(self):
        assert TIERS_5 == ["small", "medium", "large", "ood_large", "ood_xlarge"]
        assert "ood_large1" in TIERS_6
        # the merged TIERS contains every tier from both families
        assert set(TIERS) == set(TIERS_5) | set(TIERS_6)


# ==========================================================================
# _normalize_platform
# ==========================================================================

class TestNormalizePlatform:
    @pytest.mark.parametrize("raw,expected", [
        (None, "mac"),
        ("", "mac"),
        ("   ", "mac"),
        ("Windows 11", "win"),
        ("windows", "win"),
        ("macOS 24.4", "mac"),
        ("Darwin 23", "mac"),
        ("Linux 6.1", "unknown"),
        ("FreeBSD", "unknown"),
    ])
    def test_mapping(self, raw, expected):
        assert _normalize_platform(raw) == expected


# ==========================================================================
# _float_cell
# ==========================================================================

class TestFloatCell:
    @pytest.mark.parametrize("raw,expected", [
        ("5.0", 5.0),
        ("  3 ", 3.0),
        ("", None),
        (None, None),
        ("NA", None),
        ("nan", None),
        ("NONE", None),
        ("notanumber", None),
        ("-2.5", -2.5),
    ])
    def test_parse(self, raw, expected):
        assert _float_cell(raw) == expected


# ==========================================================================
# _concordance_min
# ==========================================================================

class TestConcordanceMin:
    def test_picks_min_included_metric(self):
        m = {"pearson": 0.99, "spearman": 0.95, "jaccard": 0.97}
        assert _concordance_min(m) == 0.95

    def test_excludes_diff_metrics(self):
        # max_abs_diff / rmse excluded even if in [0,1]
        m = {"pearson": 0.9, "max_abs_diff": 0.01, "rmse": 0.2}
        assert _concordance_min(m) == 0.9

    def test_out_of_range_values_skipped(self):
        m = {"pearson": 1.5, "spearman": 0.8}
        assert _concordance_min(m) == 0.8

    def test_clamps_to_one(self):
        m = {"pearson": 1.0000005}  # within float fuzz, clamps to 1.0
        assert _concordance_min(m) == 1.0

    def test_non_numeric_skipped(self):
        m = {"pearson": "n/a", "spearman": 0.7}
        assert _concordance_min(m) == 0.7

    def test_no_concordance_metrics_returns_none(self):
        assert _concordance_min({"cpu_sec": 5.0, "max_diff": 0.1}) is None

    def test_empty(self):
        assert _concordance_min({}) is None


# ==========================================================================
# _finalized_group_key
# ==========================================================================

class TestFinalizedGroupKey:
    def test_groups_baseline_and_patched_together(self):
        b = _fin_row(variant="baseline")
        p = _fin_row(variant="patched")
        assert _finalized_group_key(b) == _finalized_group_key(p)

    def test_tier_alias_normalized_in_key(self):
        k_tiny = _finalized_group_key(_fin_row(tier="tiny"))
        k_small = _finalized_group_key(_fin_row(tier="small"))
        assert k_tiny == k_small

    def test_different_platform_differs(self):
        assert _finalized_group_key(_fin_row(platform="macOS")) != \
            _finalized_group_key(_fin_row(platform="Windows"))


# ==========================================================================
# _parse_speedups — finalized format
# ==========================================================================

class TestParseFinalized:
    def test_ok_pair_yields_ok_status(self, tmp_path):
        p = _audit_from_finalized(
            tmp_path / "sp.tsv",
            _fin_row(variant="baseline", status="ok"),
            _fin_row(variant="patched", status="ok", speedup_x_mean="5.0"),
        )
        st = p.tier_status[("mac", "small")]
        assert st.state == "ok"
        assert st.speedup_x == 5.0
        assert p.median_fold_by_platform["mac"] == 5.0

    def test_missing_baseline_is_crashed(self, tmp_path):
        p = _audit_from_finalized(
            tmp_path / "sp.tsv",
            _fin_row(variant="patched", status="ok", speedup_x_mean="5.0"),
        )
        st = p.tier_status[("mac", "small")]
        assert st.state == "crashed"
        assert st.speedup_x is None

    def test_bad_status_is_crashed(self, tmp_path):
        p = _audit_from_finalized(
            tmp_path / "sp.tsv",
            _fin_row(variant="baseline", status="OOM"),
            _fin_row(variant="patched", status="OOM", speedup_x_mean=""),
        )
        st = p.tier_status[("mac", "small")]
        assert st.state == "crashed"

    def test_zero_reps_is_crashed(self, tmp_path):
        p = _audit_from_finalized(
            tmp_path / "sp.tsv",
            _fin_row(variant="baseline", status="ok"),
            _fin_row(variant="patched", status="ok", n_reps="0"),
        )
        assert p.tier_status[("mac", "small")].state == "crashed"

    def test_platform_split_kept_separately(self, tmp_path):
        p = _audit_from_finalized(
            tmp_path / "sp.tsv",
            _fin_row(variant="baseline", platform="macOS"),
            _fin_row(variant="patched", platform="macOS", speedup_x_mean="4.0"),
            _fin_row(variant="baseline", platform="Windows"),
            _fin_row(variant="patched", platform="Windows", speedup_x_mean="6.0"),
        )
        assert p.tier_status[("mac", "small")].speedup_x == 4.0
        assert p.tier_status[("win", "small")].speedup_x == 6.0
        assert set(p.platforms_with_data()) == {"win", "mac"}

    def test_concordance_parsed_from_metrics_json(self, tmp_path):
        mj = json.dumps({"pearson": 0.995, "spearman": 0.99})
        p = _audit_from_finalized(
            tmp_path / "sp.tsv",
            _fin_row(variant="baseline"),
            _fin_row(variant="patched", metrics_json_median=mj),
        )
        st = p.tier_status[("mac", "small")]
        assert st.concordance == 0.99

    def test_unknown_tier_skipped(self, tmp_path):
        p = _audit_from_finalized(
            tmp_path / "sp.tsv",
            _fin_row(variant="baseline", tier="bogus_tier"),
            _fin_row(variant="patched", tier="bogus_tier"),
        )
        assert p.tier_status == {}


# ==========================================================================
# _parse_speedups — legacy long format
# ==========================================================================

class TestParseLongFormat:
    def test_long_format_ok_batch(self, tmp_path):
        path = _write_long(
            tmp_path / "speedups.tsv",
            _long_row(variant="baseline", rep_idx="1", sec="2.0"),
            _long_row(variant="baseline", rep_idx="2", sec="2.0"),
            _long_row(variant="patched", rep_idx="1", sec="1.0"),
            _long_row(variant="patched", rep_idx="2", sec="1.0"),
        )
        p = PatchAudit(name="p", language="Py", patch_path=path,
                       speedups_path=path, speedups_exists=True,
                       lifted_from_task=None)
        _parse_speedups(p)
        st = p.tier_status[("mac", "small")]
        assert st.state == "ok"
        # speedup_x = median(baseline)/median(patched) = 2.0/1.0 = 2.0
        assert st.speedup_x == pytest.approx(2.0)

    def test_long_format_sentinel_is_crashed(self, tmp_path):
        # sentinel row: empty variant/sec, has a note + valid tier
        path = _write_long(
            tmp_path / "speedups.tsv",
            _long_row(variant="", sec="", rep_idx="", note="OOM killed"),
        )
        p = PatchAudit(name="p", language="Py", patch_path=path,
                       speedups_path=path, speedups_exists=True,
                       lifted_from_task=None)
        _parse_speedups(p)
        st = p.tier_status[("mac", "small")]
        assert st.state == "crashed"

    def test_drift_detection_from_two_batches(self, tmp_path):
        # older batch slow (2x), newer batch fast (5x) → speedup_up alert
        path = _write_long(
            tmp_path / "speedups.tsv",
            # old batch: baseline 2.0 / patched 1.0 → 2x
            _long_row(timestamp="2026-05-01T00:00:00", variant="baseline", rep_idx="1", sec="2.0"),
            _long_row(timestamp="2026-05-01T00:00:00", variant="patched", rep_idx="1", sec="1.0"),
            # new batch: baseline 5.0 / patched 1.0 → 5x
            _long_row(timestamp="2026-06-01T00:00:00", variant="baseline", rep_idx="1", sec="5.0"),
            _long_row(timestamp="2026-06-01T00:00:00", variant="patched", rep_idx="1", sec="1.0"),
        )
        p = PatchAudit(name="p", language="Py", patch_path=path,
                       speedups_path=path, speedups_exists=True,
                       lifted_from_task=None)
        _parse_speedups(p)
        st = p.tier_status[("mac", "small")]
        assert st.prev_speedup_x is not None
        alerts = p.drift_alerts()
        assert any(a.kind == "speedup_up" for a in alerts)

    def test_no_speedups_exist_noop(self, tmp_path):
        p = PatchAudit(name="p", language="Py", patch_path=tmp_path / "x",
                       speedups_path=tmp_path / "nope.tsv",
                       speedups_exists=False, lifted_from_task=None)
        _parse_speedups(p)
        assert p.tier_status == {}


# ==========================================================================
# PatchAudit per-platform views + status
# ==========================================================================

class TestPatchAuditViews:
    def _audit(self, status_map=None) -> PatchAudit:
        p = PatchAudit(name="p", language="Py", patch_path=Path("x"),
                       speedups_path=Path("y"), speedups_exists=True,
                       lifted_from_task=None)
        for (plat, tier), state in (status_map or {}).items():
            p.tier_status[(plat, tier)] = TierStatus(
                tier=tier, platform=plat, state=state,
                speedup_x=5.0 if state == "ok" else None,
            )
        return p

    def test_state_at_defaults_missing(self):
        p = self._audit()
        assert p.state_at("mac", "small") == "missing"

    def test_tiers_ok_crashed_missing_on(self):
        p = self._audit({
            ("mac", "small"): "ok",
            ("mac", "medium"): "crashed",
        })
        assert p.tiers_ok_on("mac") == ["small"]
        assert p.tiers_crashed_on("mac") == ["medium"]
        assert "large" in p.tiers_missing_on("mac")

    def test_platforms_with_data_ordered(self):
        p = self._audit({
            ("mac", "small"): "ok",
            ("win", "small"): "ok",
        })
        assert p.platforms_with_data() == ["win", "mac"]

    def test_expected_tiers_detects_6_family(self):
        p = self._audit({("mac", "ood_large1"): "ok"})
        assert p.expected_tiers == TIERS_6

    def test_expected_tiers_defaults_5_family(self):
        p = self._audit({("mac", "small"): "ok"})
        assert p.expected_tiers == TIERS_5

    def test_expected_tiers_override(self):
        p = self._audit({("mac", "small"): "ok"})
        p.expected_tiers_override = ["small", "medium"]
        assert p.expected_tiers == ["small", "medium"]

    def test_status_no_attest(self):
        p = self._audit()
        p.speedups_exists = False
        assert p.status == "no_attest"

    def test_status_no_data_when_no_tier_status(self):
        p = self._audit()
        assert p.status == "no_data"

    def test_status_full_when_all_expected_ok(self):
        p = self._audit({("mac", t): "ok" for t in TIERS_5})
        assert p.status == "full"

    def test_status_partial(self):
        p = self._audit({("mac", "small"): "ok"})
        assert p.status == "partial"

    def test_platform_status_none_when_no_rows(self):
        p = self._audit({("mac", "small"): "ok"})
        assert p.platform_status("win") == "none"

    def test_platform_status_full(self):
        p = self._audit({("mac", t): "ok" for t in TIERS_5})
        assert p.platform_status("mac") == "full"

    def test_platform_status_no_data_when_only_crashed(self):
        p = self._audit({("mac", "small"): "crashed"})
        assert p.platform_status("mac") == "no_data"


# ==========================================================================
# DriftAlert.description
# ==========================================================================

class TestDriftAlert:
    def test_speedup_up_description(self):
        a = DriftAlert("p", "mac", "small", "speedup_up", 2.0, 4.0, "t0", "t1")
        d = a.description
        assert "->" in d
        assert "+100%" in d

    def test_speedup_down_description(self):
        a = DriftAlert("p", "mac", "small", "speedup_down", 4.0, 2.0, "t0", "t1")
        assert "-50%" in a.description

    def test_concordance_drop_description(self):
        a = DriftAlert("p", "mac", "small", "concordance_drop", 0.99, 0.97, "t0", "t1")
        d = a.description
        assert "0.9900" in d
        assert "0.9700" in d


# ==========================================================================
# drift_alerts thresholds
# ==========================================================================

class TestDriftThresholds:
    def _audit_with_drift(self, **st_kwargs) -> PatchAudit:
        p = PatchAudit(name="p", language="Py", patch_path=Path("x"),
                       speedups_path=Path("y"), speedups_exists=True,
                       lifted_from_task=None)
        p.tier_status[("mac", "small")] = TierStatus(
            tier="small", platform="mac", state="ok", **st_kwargs)
        return p

    def test_small_change_no_alert(self):
        # 10% change is under the 30% speed threshold
        p = self._audit_with_drift(speedup_x=2.2, prev_speedup_x=2.0)
        assert p.drift_alerts() == []

    def test_speedup_up_alert_above_threshold(self):
        p = self._audit_with_drift(speedup_x=3.0, prev_speedup_x=2.0)  # +50%
        alerts = p.drift_alerts()
        assert len(alerts) == 1
        assert alerts[0].kind == "speedup_up"

    def test_speedup_down_alert(self):
        p = self._audit_with_drift(speedup_x=1.0, prev_speedup_x=2.0)  # -50%
        assert p.drift_alerts()[0].kind == "speedup_down"

    def test_concordance_drop_alert(self):
        p = self._audit_with_drift(
            speedup_x=2.0, prev_speedup_x=2.0,
            concordance=0.98, prev_concordance=0.995)  # -0.015 > 0.005
        kinds = [a.kind for a in p.drift_alerts()]
        assert "concordance_drop" in kinds

    def test_crashed_state_no_drift(self):
        p = PatchAudit(name="p", language="Py", patch_path=Path("x"),
                       speedups_path=Path("y"), speedups_exists=True,
                       lifted_from_task=None)
        p.tier_status[("mac", "small")] = TierStatus(
            tier="small", platform="mac", state="crashed")
        assert p.drift_alerts() == []


# ==========================================================================
# formatting helpers
# ==========================================================================

class TestFmtHelpers:
    @pytest.mark.parametrize("v,expected", [
        (None, "—"),
        (1.5, "1.50x"),
        (2.5, "2.5x"),
        (15.0, "15x"),
        (150.0, "150x"),
        (1500.0, "1.5kx"),
    ])
    def test_fmt_fold(self, v, expected):
        assert _fmt_fold(v) == expected

    def test_fmt_conc(self):
        assert _fmt_conc(None) == "—"
        assert _fmt_conc(0.995) == "0.995"

    def test_platform_tag_none(self):
        p = PatchAudit(name="p", language="Py", patch_path=Path("x"),
                       speedups_path=Path("y"), speedups_exists=True,
                       lifted_from_task=None)
        assert _platform_tag(p) == "—"

    def test_platform_tag_order(self):
        p = PatchAudit(name="p", language="Py", patch_path=Path("x"),
                       speedups_path=Path("y"), speedups_exists=True,
                       lifted_from_task=None)
        p.tier_status[("mac", "small")] = TierStatus("small", "mac", "ok")
        p.tier_status[("win", "small")] = TierStatus("small", "win", "ok")
        assert _platform_tag(p) == "W+M"

    def test_short_note_no_attest(self):
        p = PatchAudit(name="p", language="Py", patch_path=Path("x"),
                       speedups_path=Path("y"), speedups_exists=False,
                       lifted_from_task=None)
        assert "no speedups_finalized" in _short_note(p)

    def test_short_note_empty_tier_status(self):
        p = PatchAudit(name="p", language="Py", patch_path=Path("x"),
                       speedups_path=Path("y"), speedups_exists=True,
                       lifted_from_task=None)
        assert "empty" in _short_note(p)

    def test_short_note_full(self):
        p = PatchAudit(name="p", language="Py", patch_path=Path("x"),
                       speedups_path=Path("y"), speedups_exists=True,
                       lifted_from_task=None)
        for t in TIERS_5:
            p.tier_status[("mac", t)] = TierStatus(t, "mac", "ok")
        assert "M=full" in _short_note(p)

    def test_short_note_crashed_and_missing(self):
        # one ok tier, one crashed tier, rest missing → both pieces appear
        p = PatchAudit(name="p", language="Py", patch_path=Path("x"),
                       speedups_path=Path("y"), speedups_exists=True,
                       lifted_from_task=None)
        p.tier_status[("mac", "small")] = TierStatus("small", "mac", "ok")
        p.tier_status[("mac", "medium")] = TierStatus("medium", "mac", "crashed")
        note = _short_note(p)
        assert "crashed medium" in note
        assert "missing" in note


# ==========================================================================
# Enumeration + audit_packages against a fake framework tree
# ==========================================================================

def _make_fake_framework(tmp_path: Path) -> Path:
    """Build a minimal framework with one R folder-patch + one Py patch."""
    fw = tmp_path / "fw"
    # R folder patch
    rpatch = fw / "autozyme_r" / "inst" / "patches" / "myrpatch"
    rpatch.mkdir(parents=True)
    (rpatch / "patch.R").write_text(
        "# Lifted from autozyme task `test_myrpatch`\nfoo <- function() {}\n"
    )
    _write_finalized(
        rpatch / "speedups_finalized.tsv",
        _fin_row(patch="myrpatch", variant="baseline"),
        _fin_row(patch="myrpatch", variant="patched", speedup_x_mean="3.0"),
    )
    (fw / "autozyme_r" / "inst" / "speedups").mkdir(parents=True)
    # Py patch with speedups_finalized.tsv
    pypatch = fw / "autozyme_py" / "src" / "autozyme" / "mypypatch"
    pypatch.mkdir(parents=True)
    (pypatch / "__init__.py").write_text("# patch\n")
    _write_finalized(
        pypatch / "speedups_finalized.tsv",
        _fin_row(patch="mypypatch", variant="baseline"),
        _fin_row(patch="mypypatch", variant="patched", speedup_x_mean="7.0"),
    )
    return fw


class TestEnumerationAndAudit:
    def test_enumerate_finds_both_patches(self, tmp_path):
        fw = _make_fake_framework(tmp_path)
        patches = _enumerate_patches(fw)
        names = {p.name for p in patches}
        assert "myrpatch" in names
        assert "mypypatch" in names

    def test_lifted_from_extracted(self, tmp_path):
        fw = _make_fake_framework(tmp_path)
        patches = {p.name: p for p in _enumerate_patches(fw)}
        assert patches["myrpatch"].lifted_from_task == "test_myrpatch"

    def test_languages_assigned(self, tmp_path):
        fw = _make_fake_framework(tmp_path)
        patches = {p.name: p for p in _enumerate_patches(fw)}
        assert patches["myrpatch"].language == "R"
        assert patches["mypypatch"].language == "Py"

    def test_audit_packages_parses_speedups(self, tmp_path):
        fw = _make_fake_framework(tmp_path)
        patches = {p.name: p for p in audit_packages(fw)}
        assert patches["mypypatch"].median_fold_by_platform.get("mac") == 7.0
        assert patches["myrpatch"].status == "partial"

    def test_py_patch_without_speedups_is_no_attest(self, tmp_path):
        fw = tmp_path / "fw"
        pypatch = fw / "autozyme_py" / "src" / "autozyme" / "bare"
        pypatch.mkdir(parents=True)
        (pypatch / "__init__.py").write_text("# no speedups\n")
        patches = {p.name: p for p in _enumerate_patches(fw)}
        assert patches["bare"].speedups_exists is False
        assert patches["bare"].status == "no_attest"

    def test_underscore_dirs_skipped(self, tmp_path):
        fw = tmp_path / "fw"
        priv = fw / "autozyme_py" / "src" / "autozyme" / "_private"
        priv.mkdir(parents=True)
        (priv / "__init__.py").write_text("")
        patches = _enumerate_patches(fw)
        assert all(p.name != "_private" for p in patches)


# ==========================================================================
# Rendering — table / markdown / json
# ==========================================================================

def _audited_patches(tmp_path: Path) -> list[PatchAudit]:
    fw = _make_fake_framework(tmp_path)
    return audit_packages(fw)


class TestRendering:
    def test_render_table_smoke(self, tmp_path):
        patches = _audited_patches(tmp_path)
        out = render_table(patches, Path("/fw"))
        assert "Framework: /fw" in out
        assert "Scanned 2 patches" in out
        assert "Summary" in out
        assert "Legend:" in out
        # both patches appear in the rows
        assert "myrpatch" in out
        assert "mypypatch" in out

    def test_render_table_empty(self):
        out = render_table([], Path("/fw"))
        assert "Scanned 0 patches" in out

    def test_render_markdown_smoke(self, tmp_path):
        patches = _audited_patches(tmp_path)
        md = render_markdown(patches, Path("/fw"))
        assert md.startswith("# autozyme attest coverage")
        assert "## Per-patch coverage" in md
        assert "## Summary" in md
        assert "mypypatch" in md

    def test_render_markdown_cross_platform_section(self, tmp_path):
        # build a patch with both win + mac data so cross-platform table fires,
        # with a >1.5x win/mac divergence to trigger the divergence note.
        path = tmp_path / "sp.tsv"
        p = _audit_from_finalized(
            path,
            _fin_row(patch="x", variant="baseline", platform="macOS"),
            _fin_row(patch="x", variant="patched", platform="macOS", speedup_x_mean="2.0"),
            _fin_row(patch="x", variant="baseline", platform="Windows"),
            _fin_row(patch="x", variant="patched", platform="Windows", speedup_x_mean="6.0"),
        )
        md = render_markdown([p], Path("/fw"))
        assert "Cross-platform comparison" in md
        assert "platform divergence" in md

    def test_render_markdown_drift_concordance_drop_row(self, tmp_path):
        good_mj = json.dumps({"pearson": 0.999})
        bad_mj = json.dumps({"pearson": 0.98})
        path = _write_long(
            tmp_path / "speedups.tsv",
            _long_row(timestamp="2026-05-01T00:00:00", variant="baseline", rep_idx="1", sec="2.0"),
            _long_row(timestamp="2026-05-01T00:00:00", variant="patched", rep_idx="1", sec="1.0", metrics_json=good_mj),
            _long_row(timestamp="2026-06-01T00:00:00", variant="baseline", rep_idx="1", sec="2.0"),
            _long_row(timestamp="2026-06-01T00:00:00", variant="patched", rep_idx="1", sec="1.0", metrics_json=bad_mj),
        )
        p = PatchAudit(name="p", language="Py", patch_path=path,
                       speedups_path=path, speedups_exists=True,
                       lifted_from_task=None)
        _parse_speedups(p)
        md = render_markdown([p], Path("/fw"))
        assert "concordance **DROP**" in md

    def test_render_json_records_shape(self, tmp_path):
        patches = _audited_patches(tmp_path)
        records = render_json_records(patches)
        assert len(records) == 2
        rec = next(r for r in records if r["patch"] == "mypypatch")
        assert rec["language"] == "Py"
        assert rec["speedups_exists"] is True
        assert rec["status"] == "partial"
        assert "tiers" in rec
        assert "mac" in rec["tiers"]["small"]
        assert rec["tiers"]["small"]["mac"]["state"] == "ok"
        assert rec["median_fold_by_platform"]["mac"] == 7.0

    def test_render_json_serializable(self, tmp_path):
        patches = _audited_patches(tmp_path)
        records = render_json_records(patches)
        # must round-trip through json without error
        json.dumps(records)

    def test_render_markdown_awaiting_attest_section(self, tmp_path):
        fw = tmp_path / "fw"
        pypatch = fw / "autozyme_py" / "src" / "autozyme" / "bare"
        pypatch.mkdir(parents=True)
        (pypatch / "__init__.py").write_text("# no speedups\n")
        patches = audit_packages(fw)
        md = render_markdown(patches, Path("/fw"))
        assert "Awaiting first attest" in md

    def test_render_table_with_drift(self, tmp_path):
        # two batches with big speedup change → drift section appears in table
        path = _write_long(
            tmp_path / "speedups.tsv",
            _long_row(timestamp="2026-05-01T00:00:00", variant="baseline", rep_idx="1", sec="2.0"),
            _long_row(timestamp="2026-05-01T00:00:00", variant="patched", rep_idx="1", sec="1.0"),
            _long_row(timestamp="2026-06-01T00:00:00", variant="baseline", rep_idx="1", sec="6.0"),
            _long_row(timestamp="2026-06-01T00:00:00", variant="patched", rep_idx="1", sec="1.0"),
        )
        p = PatchAudit(name="p", language="Py", patch_path=path,
                       speedups_path=path, speedups_exists=True,
                       lifted_from_task=None)
        _parse_speedups(p)
        out = render_table([p], Path("/fw"))
        assert "Drift detection" in out
        assert "speedup increased" in out

    def test_render_table_speed_down_and_conc_drop(self, tmp_path):
        # newer batch slower AND lower concordance → both drift kinds render,
        # plus the "may need re-investigation" footer.
        good_mj = json.dumps({"pearson": 0.999})
        bad_mj = json.dumps({"pearson": 0.98})
        path = _write_long(
            tmp_path / "speedups.tsv",
            _long_row(timestamp="2026-05-01T00:00:00", variant="baseline", rep_idx="1", sec="6.0"),
            _long_row(timestamp="2026-05-01T00:00:00", variant="patched", rep_idx="1", sec="1.0", metrics_json=good_mj),
            _long_row(timestamp="2026-06-01T00:00:00", variant="baseline", rep_idx="1", sec="2.0"),
            _long_row(timestamp="2026-06-01T00:00:00", variant="patched", rep_idx="1", sec="1.0", metrics_json=bad_mj),
        )
        p = PatchAudit(name="p", language="Py", patch_path=path,
                       speedups_path=path, speedups_exists=True,
                       lifted_from_task=None)
        _parse_speedups(p)
        out = render_table([p], Path("/fw"))
        assert "speedup decreased" in out
        assert "concordance dropped" in out
        assert "re-investigation or re-attest" in out

    def test_render_markdown_with_drift_table(self, tmp_path):
        path = _write_long(
            tmp_path / "speedups.tsv",
            _long_row(timestamp="2026-05-01T00:00:00", variant="baseline", rep_idx="1", sec="2.0"),
            _long_row(timestamp="2026-05-01T00:00:00", variant="patched", rep_idx="1", sec="1.0"),
            _long_row(timestamp="2026-06-01T00:00:00", variant="baseline", rep_idx="1", sec="1.0"),
            _long_row(timestamp="2026-06-01T00:00:00", variant="patched", rep_idx="1", sec="1.0"),
        )
        p = PatchAudit(name="p", language="Py", patch_path=path,
                       speedups_path=path, speedups_exists=True,
                       lifted_from_task=None)
        _parse_speedups(p)
        md = render_markdown([p], Path("/fw"))
        assert "## Drift detection" in md

    def test_render_markdown_tier_gaps_section(self, tmp_path):
        # a patch with one ok tier + a crashed tier on mac → tier-gaps section
        path = tmp_path / "sp.tsv"
        p = _audit_from_finalized(
            path,
            _fin_row(patch="x", tier="small", variant="baseline"),
            _fin_row(patch="x", tier="small", variant="patched", speedup_x_mean="4.0"),
            _fin_row(patch="x", tier="medium", variant="baseline", status="OOM"),
            _fin_row(patch="x", tier="medium", variant="patched", status="OOM", speedup_x_mean=""),
        )
        md = render_markdown([p], Path("/fw"))
        assert "Tier gaps" in md

    def test_render_table_tier_gaps_and_awaiting(self, tmp_path):
        # partial patch (small ok, medium crashed) → table summary shows
        # tier gaps + tier crashes lines; a no-attest patch → awaiting line.
        path = tmp_path / "sp.tsv"
        partial = _audit_from_finalized(
            path,
            _fin_row(patch="x", tier="small", variant="baseline"),
            _fin_row(patch="x", tier="small", variant="patched", speedup_x_mean="4.0"),
            _fin_row(patch="x", tier="medium", variant="baseline", status="OOM"),
            _fin_row(patch="x", tier="medium", variant="patched", status="OOM", speedup_x_mean=""),
        )
        partial.name = "partialpatch"
        no_attest = PatchAudit(name="noattest", language="Py",
                               patch_path=Path("z"), speedups_path=Path("w"),
                               speedups_exists=False, lifted_from_task=None)
        out = render_table([partial, no_attest], Path("/fw"))
        assert "tier gaps" in out
        assert "tier crashes" in out
        assert "awaiting attest:" in out

    def test_render_table_core_panel_section(self, tmp_path):
        path = tmp_path / "sp.tsv"
        p = _audit_from_finalized(
            path,
            _fin_row(patch="core", variant="baseline"),
            _fin_row(patch="core", variant="patched", speedup_x_mean="4.0"),
        )
        p.name = "corepanel"
        p.is_core = True
        out = render_table([p], Path("/fw"))
        assert "Core panels" in out
        assert "corepanel" in out

    def test_render_markdown_core_panel_section(self, tmp_path):
        path = tmp_path / "sp.tsv"
        p = _audit_from_finalized(
            path,
            _fin_row(patch="core", variant="baseline"),
            _fin_row(patch="core", variant="patched", speedup_x_mean="4.0"),
        )
        p.name = "corepanel"
        p.is_core = True
        p.lifted_from_task = "test_core"
        md = render_markdown([p], Path("/fw"))
        assert "Core panels (tracked separately)" in md
        assert "corepanel" in md


# ==========================================================================
# Manifest expected-tiers loader + bool helper + extra enumeration shapes
# ==========================================================================

class TestManifestAndEnumExtras:
    def test_as_bool_false(self):
        assert sa._as_bool_false(False) is True
        assert sa._as_bool_false("false") is True
        assert sa._as_bool_false("NO") is True
        assert sa._as_bool_false("0") is True
        assert sa._as_bool_false(True) is False
        assert sa._as_bool_false("yes") is False
        assert sa._as_bool_false(None) is False

    def test_manifest_expected_tiers_loaded(self, tmp_path):
        pytest.importorskip("yaml")
        fw = tmp_path / "fw"
        scripts = fw / "scripts"
        scripts.mkdir(parents=True)
        dest = fw / "autozyme_r" / "inst" / "speedups"
        dest.mkdir(parents=True)
        (scripts / "seurat_attest_manifest.yaml").write_text(
            "tasks:\n"
            "  - {id: foo, legacy_key: foo, tiers: [small, medium]}\n"
            "  - {id: skipme, tiers: [small], package_speedups: false}\n"
        )
        out = sa._load_manifest_expected_tiers(fw)
        key = (dest / "seurat_foo.tsv").resolve()
        assert out.get(key) == ["small", "medium"]
        # skipme excluded (package_speedups false)
        skip_key = (dest / "seurat_skipme.tsv").resolve()
        assert skip_key not in out

    def test_r_legacy_dot_r_patch_with_finalized(self, tmp_path):
        # inst/patches/<name>.R + ../speedups/<name>.tsv (legacy layout)
        fw = tmp_path / "fw"
        rp = fw / "autozyme_r" / "inst" / "patches"
        rp.mkdir(parents=True)
        (rp / "legacypatch.R").write_text("# Lifted from autozyme task `test_lp`\n")
        speedups = fw / "autozyme_r" / "inst" / "speedups"
        _write_finalized(
            speedups / "legacypatch.tsv",
            _fin_row(patch="legacypatch", variant="baseline"),
            _fin_row(patch="legacypatch", variant="patched", speedup_x_mean="2.0"),
        )
        patches = {p.name: p for p in _enumerate_patches(fw)}
        assert "legacypatch" in patches
        assert patches["legacypatch"].speedups_exists is True
        assert patches["legacypatch"].lifted_from_task == "test_lp"

    def test_r_legacy_dot_r_patch_missing_speedups(self, tmp_path):
        fw = tmp_path / "fw"
        rp = fw / "autozyme_r" / "inst" / "patches"
        rp.mkdir(parents=True)
        (rp / "orphan.R").write_text("foo <- function() {}\n")
        (fw / "autozyme_r" / "inst" / "speedups").mkdir(parents=True)
        patches = {p.name: p for p in _enumerate_patches(fw)}
        assert patches["orphan"].speedups_exists is False

    def test_r_folder_patch_with_sub_tsvs(self, tmp_path):
        # folder patch with per-method <name>_*.tsv sub-speedups
        fw = tmp_path / "fw"
        d = fw / "autozyme_r" / "inst" / "patches" / "seurat"
        d.mkdir(parents=True)
        (d / "patch.R").write_text("# Lifted from autozyme task `test_seurat`\n")
        speedups = fw / "autozyme_r" / "inst" / "speedups"
        _write_finalized(
            speedups / "seurat_markers.tsv",
            _fin_row(patch="seurat_markers", variant="baseline"),
            _fin_row(patch="seurat_markers", variant="patched", speedup_x_mean="3.0"),
        )
        _write_finalized(
            speedups / "seurat_neighbors.tsv",
            _fin_row(patch="seurat_neighbors", variant="baseline"),
            _fin_row(patch="seurat_neighbors", variant="patched", speedup_x_mean="2.0"),
        )
        names = {p.name for p in _enumerate_patches(fw)}
        assert "seurat_markers" in names
        assert "seurat_neighbors" in names

    def test_py_patch_with_speedups_dir(self, tmp_path):
        # autozyme_py patch with a speedups/ dir of per-op TSVs (no finalized)
        fw = tmp_path / "fw"
        d = fw / "autozyme_py" / "src" / "autozyme" / "scanpy"
        d.mkdir(parents=True)
        (d / "__init__.py").write_text("# Lifted from autozyme task `test_scanpy`\n")
        spd = d / "speedups"
        _write_finalized(
            spd / "leiden.tsv",
            _fin_row(patch="leiden", variant="baseline"),
            _fin_row(patch="leiden", variant="patched", speedup_x_mean="9.0"),
        )
        names = {p.name for p in _enumerate_patches(fw)}
        assert "leiden" in names
