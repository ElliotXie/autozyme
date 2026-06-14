"""Wave-4 coverage tests for zyme.scan_speedups.

Targets REACHABLE lines not covered by tests/test_scan_speedups.py (the 9
pre-existing failures there are ignored) or test_scan_speedups_deep.py:
  - _load_thread_rubric yaml-free regex fallback (PyYAML import blocked)
  - _load_thread_rubric yaml path + malformed-file return
  - _load_oom_skip file-present parse + _oom_skipped under --mac-only
  - _audit_raw_era_issues: baseline_era_drift + stale_baseline_pairing
  - _read_raw_speedup_rows dedup + note-skip + peak-mb-none filter
  - missing_variant: OOM one-sided skip + _REDUCED higher-thread skip
  - platform_asymmetry: dual-platform INFO + headline fullt skip
  - thread_gap: oom-skip suppression
  - render_table empty / render_markdown clean-table-only

Module-global rubric/oom state is saved and restored around tests that mutate
it so test order stays independent.
"""
from __future__ import annotations

import builtins
from pathlib import Path

import pytest

import zyme.scan_speedups as ss
from zyme.scan_speedups import (
    Issue,
    PatchCoverage,
    _audit_raw_era_issues,
    _load_oom_skip,
    _oom_skipped,
    _read_raw_speedup_rows,
    audit_patch,
    render_markdown,
    render_table,
)


# --------------------------------------------------------------------------
# finalized-TSV helpers (mirror the deep-test header)
# --------------------------------------------------------------------------
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


RAW_HEADER = ("timestamp\tpatch_name\ttier\tdataset\trep_idx\tvariant\t"
              "sec\tpeak_mb\tsystem_os\tsystem_threads\tnote")


def _raw_row(ts, variant, peak, *, sec="1.0", tier="small", ds="ds",
             rep="0", os_="macOS", thr="1", note="") -> str:
    return "\t".join([ts, "x", tier, ds, rep, variant, sec, str(peak),
                      os_, thr, note])


def _write_raw_mac(tmp_path: Path, *rows: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "speedups.mac.tsv"
    p.write_text(RAW_HEADER + "\n" + "\n".join(rows) + "\n")
    return p


# --------------------------------------------------------------------------
# rubric-global save/restore fixture (tests that call _load_thread_rubric)
# --------------------------------------------------------------------------
@pytest.fixture
def restore_rubric_globals():
    saved = (
        dict(ss._HEADLINE_THREAD_SCHEDULE),
        dict(ss._DEFAULT_THREAD_SCHEDULE),
        {k: set(v) for k, v in ss._SINGLE_THREADED_PATCHES.items()},
        {k: {p: set(vs) for p, vs in m.items()} for k, m in ss._REDUCED.items()},
        ss.__file__,
    )
    yield
    (ss._HEADLINE_THREAD_SCHEDULE, ss._DEFAULT_THREAD_SCHEDULE,
     ss._SINGLE_THREADED_PATCHES, ss._REDUCED, ss.__file__) = saved


def _point_module_at(tmp: Path, rubric_text: str) -> None:
    """Plant a rubric file under tmp and aim the module's __file__ at it so
    `Path(__file__).resolve().parents[2]` == tmp."""
    rubric = tmp / "paper" / "rubric"
    rubric.mkdir(parents=True, exist_ok=True)
    (rubric / "task_threading.yaml").write_text(rubric_text)
    fake = tmp / "a" / "b" / "scan_speedups.py"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("")
    ss.__file__ = str(fake)


RUBRIC = """schedules:
  headline: {mac: [1, 4, 14], win: [1, 4, 32]}
  default: {mac: [1, 4, 8], win: [1, 4, 32]}
tasks:
  singletask: {schedule: single}
  reducedtask: {reduce_t1: {mac: [baseline], win: [patched]}}
"""


class TestLoadThreadRubricYaml:
    def test_yaml_path_populates(self, tmp_path: Path, restore_rubric_globals):
        _point_module_at(tmp_path, RUBRIC)
        ss._load_thread_rubric()
        assert "singletask" in ss._SINGLE_THREADED_PATCHES["mac"]
        assert ss._REDUCED["reducedtask"]["mac"] == {"baseline"}
        assert ss._REDUCED["reducedtask"]["win"] == {"patched"}
        assert ss._HEADLINE_THREAD_SCHEDULE["mac"] == {"1", "4", "14"}

    def test_malformed_yaml_keeps_defaults(self, tmp_path: Path,
                                           restore_rubric_globals):
        # A YAML file that parses to a scalar (not a mapping) makes
        # r.get(...) raise -> the broad-except branch returns, defaults intact.
        _point_module_at(tmp_path, "just a bare scalar string\n")
        before = dict(ss._HEADLINE_THREAD_SCHEDULE)
        ss._load_thread_rubric()
        assert ss._HEADLINE_THREAD_SCHEDULE == before

    def test_missing_file_returns(self, tmp_path: Path, restore_rubric_globals):
        # Point at an empty tmp (no rubric file). yaml import succeeds, open
        # raises FileNotFoundError -> broad-except returns; defaults intact.
        fake = tmp_path / "a" / "b" / "scan_speedups.py"
        fake.parent.mkdir(parents=True)
        fake.write_text("")
        ss.__file__ = str(fake)
        before = dict(ss._DEFAULT_THREAD_SCHEDULE)
        ss._load_thread_rubric()
        assert ss._DEFAULT_THREAD_SCHEDULE == before


class TestLoadThreadRubricRegexFallback:
    def test_regex_fallback_when_no_pyyaml(self, tmp_path: Path,
                                           restore_rubric_globals,
                                           monkeypatch):
        _point_module_at(tmp_path, RUBRIC)
        real_import = builtins.__import__

        def no_yaml(name, *a, **k):
            if name == "yaml":
                raise ImportError("blocked")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_yaml)
        ss._load_thread_rubric()
        # The regex fallback parses the `tasks:` block the same way.
        assert "singletask" in ss._SINGLE_THREADED_PATCHES["mac"]
        assert ss._REDUCED["reducedtask"]["mac"] == {"baseline"}

    def test_regex_fallback_missing_file_returns(self, tmp_path: Path,
                                                 restore_rubric_globals,
                                                 monkeypatch):
        # No rubric file + PyYAML blocked -> path.read_text raises OSError ->
        # the fallback returns without touching globals.
        fake = tmp_path / "a" / "b" / "scan_speedups.py"
        fake.parent.mkdir(parents=True)
        fake.write_text("")
        ss.__file__ = str(fake)
        real_import = builtins.__import__

        def no_yaml(name, *a, **k):
            if name == "yaml":
                raise ImportError("blocked")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_yaml)
        before = dict(ss._SINGLE_THREADED_PATCHES)
        ss._load_thread_rubric()
        assert ss._SINGLE_THREADED_PATCHES == before


# --------------------------------------------------------------------------
# _load_oom_skip file-present + _oom_skipped under --mac-only
# --------------------------------------------------------------------------
@pytest.fixture
def restore_oom():
    saved = ss._OOM_SKIP
    saved_file = ss.__file__
    yield
    ss._OOM_SKIP = saved
    ss.__file__ = saved_file


class TestOomSkipFilePresent:
    def test_parses_skip_file(self, tmp_path: Path, restore_oom):
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "mac_oom_skip.tsv").write_text(
            "# comment line\n"
            "\n"
            "mypatch\tood_large\tbaseline\n"
            "mypatch\tood_large\tpatched\n"
        )
        fake = tmp_path / "a" / "b" / "scan_speedups.py"
        fake.parent.mkdir(parents=True)
        fake.write_text("")
        ss.__file__ = str(fake)
        ss._OOM_SKIP = None  # force reload
        out = _load_oom_skip()
        assert ("mypatch", "ood_large", "baseline") in out
        assert _oom_skipped("mypatch", "ood_large", "baseline", "mac") is True
        # Only consulted under --mac-only:
        assert _oom_skipped("mypatch", "ood_large", "baseline", None) is False


# --------------------------------------------------------------------------
# _read_raw_speedup_rows + _audit_raw_era_issues
# --------------------------------------------------------------------------
class TestRawEraIssues:
    def test_baseline_era_drift_fires_info(self, tmp_path: Path):
        _write_raw_mac(tmp_path,
            _raw_row("2026-01-01T00:00:00", "baseline", 1000),
            _raw_row("2026-01-01T00:00:00", "baseline", 1050, rep="1"),
            _raw_row("2026-06-01T00:00:00", "baseline", 3000),
            _raw_row("2026-06-01T00:00:00", "baseline", 3100, rep="1"),
        )
        cov = PatchCoverage("x", tmp_path / "tsv", True)
        _audit_raw_era_issues(cov, "x", tmp_path,
                              baseline_era_pct=20.0, baseline_era_abs_mb=1024.0,
                              platform_filter="mac")
        drift = [i for i in cov.issues if i.kind == "baseline_era_drift"]
        assert drift and drift[0].severity == "INFO"

    def test_stale_baseline_pairing_fires_info(self, tmp_path: Path):
        _write_raw_mac(tmp_path,
            _raw_row("2026-06-01T00:00:00", "baseline", 1000),
            _raw_row("2026-03-01T00:00:00", "patched", 800, sec="0.5"),
        )
        cov = PatchCoverage("x", tmp_path / "tsv", True)
        _audit_raw_era_issues(cov, "x", tmp_path,
                              baseline_era_pct=20.0, baseline_era_abs_mb=1024.0,
                              platform_filter="mac")
        stale = [i for i in cov.issues if i.kind == "stale_baseline_pairing"]
        assert stale and stale[0].severity == "INFO"

    def test_no_raw_rows_is_noop(self, tmp_path: Path):
        # No raw shards at all -> early return, no issues.
        cov = PatchCoverage("x", tmp_path / "tsv", True)
        _audit_raw_era_issues(cov, "x", tmp_path,
                              baseline_era_pct=20.0, baseline_era_abs_mb=1024.0)
        assert cov.issues == []

    def test_note_skip_and_peak_filter(self, tmp_path: Path):
        _write_raw_mac(tmp_path,
            _raw_row("2026-01-01", "baseline", 1000),
            _raw_row("2026-01-01", "baseline", 1000, rep="1", note="migrated: x"),
            _raw_row("2026-01-01", "baseline", 0, rep="2"),  # peak 0 -> filtered
        )
        rows = _read_raw_speedup_rows(tmp_path, "mac")
        # only the first row survives (note-skip + peak<=0 dropped)
        assert len(rows) == 1

    def test_dedup_identical_rows(self, tmp_path: Path):
        _write_raw_mac(tmp_path,
            _raw_row("2026-01-01", "baseline", 1000),
            _raw_row("2026-01-01", "baseline", 1000),  # identical signature
        )
        rows = _read_raw_speedup_rows(tmp_path, "mac")
        assert len(rows) == 1


# --------------------------------------------------------------------------
# missing_variant: OOM one-sided + _REDUCED higher-thread skip
# --------------------------------------------------------------------------
class TestMissingVariant:
    def test_oom_only_variant_skipped(self, tmp_path: Path):
        # A cell whose ONLY present variant is OOM is intentionally one-sided.
        p = _write_tsv(tmp_path,
            _row(variant="patched", status="OOM", threads="1"),
        )
        cov = audit_patch("x", p)
        assert "missing_variant" not in _kinds(cov)

    def test_patched_missing_when_baseline_oom_skipped(self, tmp_path: Path):
        p = _write_tsv(tmp_path,
            _row(variant="baseline", status="OOM", threads="1"),
        )
        cov = audit_patch("x", p)
        assert "missing_variant" not in _kinds(cov)

    def test_reduced_baseline_higher_thread_skipped(self, tmp_path: Path,
                                                    monkeypatch):
        # When the rubric reduces baseline to t1 on win, a patched-only cell at
        # threads=4 does NOT fire missing_variant (baseline reduced to t1).
        monkeypatch.setitem(ss._REDUCED, "x", {"win": {"baseline"}})
        p = _write_tsv(tmp_path,
            _row(variant="patched", threads="4", platform="Windows"),
        )
        cov = audit_patch("x", p, platform_filter="win")
        mv = [i for i in cov.issues if i.kind == "missing_variant"]
        assert mv == []

    def test_reduced_patched_higher_thread_skipped(self, tmp_path: Path,
                                                   monkeypatch):
        monkeypatch.setitem(ss._REDUCED, "x", {"win": {"patched"}})
        p = _write_tsv(tmp_path,
            _row(variant="baseline", threads="4", platform="Windows"),
        )
        cov = audit_patch("x", p, platform_filter="win")
        mv = [i for i in cov.issues if i.kind == "missing_variant"]
        assert mv == []


# --------------------------------------------------------------------------
# platform_asymmetry
# --------------------------------------------------------------------------
class TestPlatformAsymmetry:
    def test_asymmetry_info_when_dual_platform(self, tmp_path: Path):
        # Win has a (small, 4) cell that Mac lacks -> INFO asymmetry.
        rows = []
        for plat in ("Windows",):
            for thr in ("1", "4"):
                rows.append(_row(platform=plat, threads=thr, variant="baseline"))
                rows.append(_row(platform=plat, threads=thr, variant="patched"))
        # Mac present only at threads=1
        rows.append(_row(platform="macOS", threads="1", variant="baseline"))
        rows.append(_row(platform="macOS", threads="1", variant="patched"))
        cov = audit_patch("x", _write_tsv(tmp_path, *rows))
        asym = [i for i in cov.issues if i.kind == "platform_asymmetry"]
        assert any(a.threads == "4" and a.severity == "INFO" for a in asym)

    def test_mac_only_cell_flags_missing_win(self, tmp_path: Path):
        rows = [
            _row(platform="macOS", threads="1", variant="baseline"),
            _row(platform="macOS", threads="1", variant="patched"),
            _row(platform="macOS", threads="9", variant="baseline"),
            _row(platform="macOS", threads="9", variant="patched"),
            _row(platform="Windows", threads="1", variant="baseline"),
            _row(platform="Windows", threads="1", variant="patched"),
        ]
        cov = audit_patch("x", _write_tsv(tmp_path, *rows))
        asym = [i for i in cov.issues if i.kind == "platform_asymmetry"]
        # the mac-only (small, 9) cell flags a missing Windows twin
        assert any(a.threads == "9" and a.platform == "win" for a in asym)

    def test_headline_fullt_asymmetry_skipped(self, tmp_path: Path):
        # scanpy_* headline: mac fullt=14, win fullt=32 differ by design, so a
        # threads=14 cell present only on mac is NOT flagged as asymmetry.
        rows = [
            _row(patch="scanpy_x", platform="macOS", threads="14", variant="baseline"),
            _row(patch="scanpy_x", platform="macOS", threads="14", variant="patched"),
            _row(patch="scanpy_x", platform="Windows", threads="1", variant="baseline"),
            _row(patch="scanpy_x", platform="Windows", threads="1", variant="patched"),
        ]
        cov = audit_patch("scanpy_x", _write_tsv(tmp_path, *rows))
        asym = [i for i in cov.issues if i.kind == "platform_asymmetry"]
        # threads=14 (mac fullt) is in headline_fullt_threads -> skipped
        assert all(a.threads != "14" for a in asym)


# --------------------------------------------------------------------------
# thread_gap oom-skip suppression
# --------------------------------------------------------------------------
class TestThreadGapOomSkip:
    def test_oom_skip_suppresses_thread_gap(self, tmp_path: Path, monkeypatch):
        # A whole tier that OOMs on this Mac suppresses its thread_gap WARNs.
        monkeypatch.setattr(ss, "_OOM_SKIP",
                            {("x", "small", "baseline")})
        rows = [
            _row(platform="macOS", threads="1", variant="baseline"),
            _row(platform="macOS", threads="1", variant="patched"),
        ]
        cov = audit_patch("x", _write_tsv(tmp_path, *rows),
                          platform_filter="mac")
        # default mac schedule wants {1,4,8}; without the oom skip there'd be
        # thread_gap WARNs for 4 and 8 — the skip silences them.
        assert "thread_gap" not in _kinds(cov)


# --------------------------------------------------------------------------
# renderers
# --------------------------------------------------------------------------
class TestRenderEdges:
    def test_render_table_no_patches(self):
        assert "no patches" in render_table([])

    def test_render_markdown_table_only_when_clean(self, tmp_path: Path):
        # A clean win patch (full schedule) -> markdown table, no Details.
        rows = []
        for thr in ("1", "4", "32"):
            rows.append(_row(platform="Windows", variant="baseline", threads=thr))
            rows.append(_row(platform="Windows", variant="patched", threads=thr))
        cov = audit_patch("x", _write_tsv(tmp_path, *rows))
        assert cov.is_clean, _kinds(cov)
        md = render_markdown([cov])
        assert "| x |" in md
        assert "### Details" not in md

    def test_render_markdown_skips_clean_in_details(self, tmp_path: Path):
        # One dirty + one clean patch: the Details section lists only the dirty
        # one (the clean patch hits the `continue` in the details loop).
        clean_rows = []
        for thr in ("1", "4", "32"):
            clean_rows.append(_row(platform="Windows", variant="baseline", threads=thr))
            clean_rows.append(_row(platform="Windows", variant="patched", threads=thr))
        clean = audit_patch("cleanp", _write_tsv(tmp_path / "c", *clean_rows))
        dirty = audit_patch("dirtyp",
                            _write_tsv(tmp_path / "d", _row(patch="dirtyp",
                                                            variant="patched")))
        md = render_markdown([clean, dirty])
        assert "#### dirtyp" in md
        assert "#### cleanp" not in md


# --------------------------------------------------------------------------
# extra small reachable branches
# --------------------------------------------------------------------------
class TestSmallBranches:
    def test_matches_filter_other_platform_returns_true(self):
        # A non-mac/non-win filter ("linux") falls through to `return True`.
        from zyme.scan_speedups import _matches_platform_filter
        assert _matches_platform_filter({"platform": "macOS"}, "linux") is True

    def test_empty_after_platform_filter(self, tmp_path: Path):
        # All rows are Windows; filtering to mac empties the set -> early
        # `return cov` with n_rows=0 and no issues.
        p = _write_tsv(tmp_path,
            _row(platform="Windows", variant="baseline"),
            _row(platform="Windows", variant="patched"),
        )
        cov = audit_patch("x", p, platform_filter="mac")
        assert cov.n_rows == 0
        assert cov.issues == []

    def test_low_reps_skips_unparseable_n_reps(self, tmp_path: Path):
        # n_reps that can't parse to int is skipped (no low_reps fired).
        p = _write_tsv(tmp_path,
            _row(variant="baseline", n_reps="oops", threads="1"),
            _row(variant="patched", n_reps="oops", threads="1"),
        )
        cov = audit_patch("x", p)
        assert "low_reps" not in _kinds(cov)

    def test_low_reps_oom_skipped(self, tmp_path: Path, monkeypatch):
        # A low-rep cell that is in the OOM-skip set under --mac-only is not
        # flagged.
        monkeypatch.setattr(ss, "_OOM_SKIP", {("x", "small", "baseline")})
        p = _write_tsv(tmp_path,
            _row(platform="macOS", variant="baseline", n_reps="1", threads="1"),
        )
        cov = audit_patch("x", p, platform_filter="mac")
        assert "low_reps" not in _kinds(cov)

    def test_single_threaded_patch_expects_only_t1(self, tmp_path: Path,
                                                   monkeypatch):
        # A patch registered single-threaded on mac expects only {1}: a t1-only
        # mac TSV produces no thread_gap even though default schedule wants 4/8.
        monkeypatch.setitem(ss._SINGLE_THREADED_PATCHES, "mac", {"x"})
        p = _write_tsv(tmp_path,
            _row(platform="macOS", variant="baseline", threads="1"),
            _row(platform="macOS", variant="patched", threads="1"),
        )
        cov = audit_patch("x", p, platform_filter="mac")
        assert "thread_gap" not in _kinds(cov)

    def test_era_drift_single_era_no_issue(self, tmp_path: Path):
        # Only one measurement date -> len(era_peaks)<2 -> no era-drift issue.
        _write_raw_mac(tmp_path,
            _raw_row("2026-01-01", "baseline", 1000),
            _raw_row("2026-01-01", "baseline", 5000, rep="1"),
        )
        cov = PatchCoverage("x", tmp_path / "tsv", True)
        _audit_raw_era_issues(cov, "x", tmp_path,
                              baseline_era_pct=20.0, baseline_era_abs_mb=1024.0,
                              platform_filter="mac")
        assert [i for i in cov.issues if i.kind == "baseline_era_drift"] == []

    def test_render_json_records_clean_and_issues(self, tmp_path: Path):
        from zyme.scan_speedups import render_json_records
        dirty = audit_patch("dirtyp",
                            _write_tsv(tmp_path / "d", _row(patch="dirtyp",
                                                            variant="patched")))
        clean_rows = []
        for thr in ("1", "4", "32"):
            clean_rows.append(_row(platform="Windows", variant="baseline", threads=thr))
            clean_rows.append(_row(platform="Windows", variant="patched", threads=thr))
        clean = audit_patch("cleanp", _write_tsv(tmp_path / "c", *clean_rows))
        recs = render_json_records([clean, dirty])
        # clean -> one PASS record; dirty -> one record per issue
        kinds = {(r["patch"], r["kind"]) for r in recs}
        assert ("cleanp", "clean") in kinds
        assert any(p == "dirtyp" and k != "clean" for (p, k) in kinds)
