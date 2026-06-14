"""Unit tests for zyme.commands.backfill — the rep-lifting sweep planner.

backfill reads each patch's speedups_finalized.tsv (n_reps), plans which
(patch, tier, threads, platform, variant) cells fall below a target rep count,
and drives `zyme attest` + a shard-aware publish + finalize per cell. The pure
layer is large and testable directly: TSV parsing with tier aliasing + dedup,
the CellState/WorkItem data model, patch/task discovery + index building, the
_plan work-queue construction (under-target / high-variance / OOM-skip /
filters / memory-first sort), the attest command + env construction, the
package_verify -> speedups shard copy (per-platform split + dedup), the
fallback-warning + pv-pass gates, and the resource-admission probe. The
subprocess-launching `_execute_one` / `_orchestrate` are driven via a
monkeypatched execution boundary or covered structurally.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest

import zyme.commands.backfill as bf
from zyme.commands.backfill import (
    CellState,
    ExecResult,
    WorkItem,
    _attest_patch_name,
    _build_attest_cmd,
    _build_task_index,
    _check_no_fallback_warnings,
    _check_pv_new_rows_passed,
    _classify_tier,
    _copy_new_pv_rows_to_speedups,
    _discover_patches,
    _extract_lifted_from,
    _find_task_dir,
    _load_seurat_scanpy_manifests,
    _normalize_platform,
    _parse_csv,
    _parse_int,
    _parse_reps_float,
    _patch_dir,
    _plan,
    _print_plan_summary,
    _read_finalized,
    _resource_fits,
    _seurat_patch_needs_python,
    _subprocess_env,
    cmd_backfill,
    _DEFAULT_SKIP,
)


# ==========================================================================
# finalized-TSV fixture builders
# ==========================================================================

FINALIZED_HEADER = [
    "patch", "tier", "platform", "threads", "dataset", "variant",
    "status", "n_reps", "sec_reps", "sec_mean", "mem_mean",
]


def _fin_row(**ov) -> dict:
    base = {
        "patch": "p", "tier": "small", "platform": "win", "threads": "1",
        "dataset": "ds", "variant": "patched", "status": "ok", "n_reps": "2",
        "sec_reps": "1.0,1.1", "sec_mean": "1.05", "mem_mean": "500",
    }
    base.update(ov)
    return base


def _write_finalized(path: Path, *rows: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(FINALIZED_HEADER)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in FINALIZED_HEADER))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ==========================================================================
# small scalar parsers
# ==========================================================================

class TestScalarParsers:
    @pytest.mark.parametrize("raw,expected", [
        (None, "unknown"), ("", "unknown"), ("  ", "unknown"),
        ("Windows 11", "win"), ("win", "win"),
        ("macOS 24", "mac"), ("Darwin", "mac"), ("mac", "mac"),
        ("Linux", "linux"),
    ])
    def test_normalize_platform(self, raw, expected):
        assert _normalize_platform(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("5", 5), (" 7 ", 7), ("", 0), (None, 0), ("xx", 0),
    ])
    def test_parse_int(self, raw, expected):
        assert _parse_int(raw) == expected

    def test_parse_reps_float_ok(self):
        assert _parse_reps_float("1.0, 2.5 ,3") == [1.0, 2.5, 3.0]

    def test_parse_reps_float_empty(self):
        assert _parse_reps_float("") == []
        assert _parse_reps_float(None) == []

    def test_parse_reps_float_bad_token_aborts(self):
        # a non-numeric token returns [] (whole field rejected)
        assert _parse_reps_float("1.0,abc") == []

    def test_parse_csv(self):
        assert _parse_csv("a, b ,c") == ["a", "b", "c"]
        assert _parse_csv("") == []
        assert _parse_csv(None) == []
        assert _parse_csv(",,") == []


# ==========================================================================
# CellState
# ==========================================================================

class TestCellState:
    def test_diff_pct_n2_computed(self):
        c = CellState(n_reps=2, sec_reps=[1.0, 1.2], status="ok", dataset="d")
        # |1.0-1.2| / 1.1 * 100 = ~18.18%
        assert c.diff_pct_n2 == pytest.approx(18.1818, rel=1e-3)

    def test_diff_pct_n2_none_when_not_two(self):
        assert CellState(3, [1, 1, 1], "ok", "d").diff_pct_n2 is None
        assert CellState(2, [1.0], "ok", "d").diff_pct_n2 is None

    def test_diff_pct_n2_none_when_mean_zero(self):
        assert CellState(2, [0.0, 0.0], "ok", "d").diff_pct_n2 is None


# ==========================================================================
# WorkItem
# ==========================================================================

class TestWorkItem:
    def _wi(self, **ov):
        base = dict(
            patch="p", lang="py", task_dir=Path("/x"), tier="small",
            threads=1, target_reps=3, baseline_n=1, patched_n=2,
            reason="low_reps",
        )
        base.update(ov)
        return WorkItem(**base)

    def test_reps_to_run(self):
        # target 3 - min(baseline 1, patched 2) = 2
        assert self._wi().reps_to_run == 2

    def test_reps_to_run_floor_zero(self):
        assert self._wi(target_reps=1, baseline_n=2, patched_n=2).reps_to_run == 0

    def test_is_small_tier(self):
        assert self._wi(tier="medium").is_small_tier is True
        assert self._wi(tier="large").is_small_tier is False
        assert self._wi(tier="ood_large").is_small_tier is False

    def test_label(self):
        lbl = self._wi(tier="small", threads=4).label()
        assert "p/small/T4" in lbl
        assert "b=1 p=2→3" in lbl


# ==========================================================================
# _read_finalized — tier aliasing + dedup
# ==========================================================================

class TestReadFinalized:
    def test_missing_file_empty(self, tmp_path):
        assert _read_finalized(tmp_path / "nope.tsv") == {}

    def test_basic_parse(self, tmp_path):
        path = _write_finalized(
            tmp_path / "f.tsv",
            _fin_row(tier="small", threads="1", platform="win", variant="patched",
                     n_reps="2", sec_reps="1.0,1.1"),
        )
        out = _read_finalized(path)
        st = out[("small", "1", "win", "patched")]
        assert st.n_reps == 2
        assert st.sec_reps == [1.0, 1.1]
        assert st.sec_mean == 1.05
        assert st.mem_mean == 500.0

    def test_tiny_aliases_to_small(self, tmp_path):
        path = _write_finalized(
            tmp_path / "f.tsv",
            _fin_row(tier="tiny", variant="patched", n_reps="2"),
        )
        out = _read_finalized(path)
        assert ("small", "1", "win", "patched") in out
        assert ("tiny", "1", "win", "patched") not in out

    def test_alias_collapse_keeps_higher_nreps(self, tmp_path):
        path = _write_finalized(
            tmp_path / "f.tsv",
            _fin_row(tier="tiny", variant="patched", n_reps="1"),
            _fin_row(tier="small", variant="patched", n_reps="3"),
        )
        out = _read_finalized(path)
        assert out[("small", "1", "win", "patched")].n_reps == 3

    def test_bad_sec_mean_defaults_zero(self, tmp_path):
        path = _write_finalized(
            tmp_path / "f.tsv",
            _fin_row(variant="patched", sec_mean="oops", mem_mean=""),
        )
        out = _read_finalized(path)
        st = out[("small", "1", "win", "patched")]
        assert st.sec_mean == 0.0
        assert st.mem_mean == 0.0


# ==========================================================================
# _classify_tier
# ==========================================================================

class TestClassifyTier:
    @pytest.mark.parametrize("tier,expected", [
        ("tiny", "small"), ("small", "small"), ("medium", "small"),
        ("large", "large"), ("ood_large", "large"), ("ood_xlarge", "large"),
        ("bogus", "small"), ("", "small"),
    ])
    def test_classify(self, tier, expected):
        assert _classify_tier(tier) == expected


# ==========================================================================
# _attest_patch_name + _patch_dir + _seurat_patch_needs_python
# ==========================================================================

class TestNameAndDirHelpers:
    def test_attest_patch_name_scanpy(self):
        assert _attest_patch_name("scanpy_normalize") == "scanpy"

    def test_attest_patch_name_seurat(self):
        assert _attest_patch_name("seurat_markers") == "seurat"

    def test_attest_patch_name_plain(self):
        assert _attest_patch_name("mgcv") == "mgcv"

    def test_patch_dir_py(self, tmp_path):
        d = _patch_dir(tmp_path, "mgcv", "py")
        assert d == tmp_path / "autozyme_py" / "src" / "autozyme" / "mgcv"

    def test_patch_dir_r(self, tmp_path):
        d = _patch_dir(tmp_path, "seurat_pca", "R")
        assert d == tmp_path / "autozyme_r" / "inst" / "patches" / "seurat_pca"

    def test_seurat_needs_python(self):
        assert _seurat_patch_needs_python("seurat_pca") is True
        assert _seurat_patch_needs_python("seurat_integrate_cca") is True
        assert _seurat_patch_needs_python("seurat_markers") is False


# ==========================================================================
# _extract_lifted_from / _find_task_dir / _discover_patches
# ==========================================================================

class TestDiscovery:
    def test_extract_lifted_from(self, tmp_path):
        p = tmp_path / "patch.R"
        p.write_text("# Lifted from autozyme task `test_mgcv`\n")
        assert _extract_lifted_from(p) == "test_mgcv"

    def test_extract_lifted_from_none(self, tmp_path):
        p = tmp_path / "patch.R"
        p.write_text("no marker here\n")
        assert _extract_lifted_from(p) is None

    def test_extract_lifted_from_missing_file(self, tmp_path):
        assert _extract_lifted_from(tmp_path / "nope") is None

    def test_discover_patches(self, tmp_path):
        fw = tmp_path / "fw"
        pyp = fw / "autozyme_py" / "src" / "autozyme" / "mgcv"
        pyp.mkdir(parents=True)
        (pyp / "speedups_finalized.tsv").write_text("x\n")
        rp = fw / "autozyme_r" / "inst" / "patches" / "seurat_pca"
        rp.mkdir(parents=True)
        (rp / "speedups_finalized.tsv").write_text("x\n")
        out = _discover_patches(fw)
        names = {(n, lang) for n, lang, _ in out}
        assert names == {("mgcv", "py"), ("seurat_pca", "R")}

    def test_discover_skips_underscore_and_unfinalized(self, tmp_path):
        fw = tmp_path / "fw"
        root = fw / "autozyme_py" / "src" / "autozyme"
        (root / "_priv").mkdir(parents=True)
        (root / "_priv" / "speedups_finalized.tsv").write_text("x\n")
        (root / "bare").mkdir()  # no finalized tsv
        out = _discover_patches(fw)
        assert out == []

    def test_find_task_dir(self, tmp_path):
        fw = tmp_path / "fw"
        td = fw / "optimized_task" / "cat" / "test_mgcv"
        td.mkdir(parents=True)
        (td / "task.yaml").write_text("x: 1\n")
        found = _find_task_dir(fw, "test_mgcv")
        assert found == td

    def test_find_task_dir_none(self, tmp_path):
        fw = tmp_path / "fw"
        (fw / "optimized_task").mkdir(parents=True)
        assert _find_task_dir(fw, "missing") is None


# ==========================================================================
# _load_seurat_scanpy_manifests / _build_task_index
# ==========================================================================

class TestTaskIndex:
    def test_manifest_resolution(self, tmp_path):
        pytest.importorskip("yaml")
        fw = tmp_path / "fw"
        scripts = fw / "scripts"
        scripts.mkdir(parents=True)
        td = fw / "optimized_task" / "cat" / "normalize_v1"
        td.mkdir(parents=True)
        (td / "task.yaml").write_text("x: 1\n")
        (scripts / "scanpy_attest_manifest.yaml").write_text(
            "tasks:\n  - {id: normalize, path: optimized_task/cat/normalize_v1}\n"
        )
        out = _load_seurat_scanpy_manifests(fw)
        assert out.get("scanpy_normalize") == td.resolve()

    def test_build_task_index_via_marker(self, tmp_path):
        fw = tmp_path / "fw"
        pyp = fw / "autozyme_py" / "src" / "autozyme" / "mgcv"
        pyp.mkdir(parents=True)
        (pyp / "__init__.py").write_text(
            "# Lifted from autozyme task `test_mgcv`\n"
        )
        td = fw / "optimized_task" / "cat" / "test_mgcv"
        td.mkdir(parents=True)
        (td / "task.yaml").write_text("x: 1\n")
        idx = _build_task_index(fw)
        assert idx["mgcv"] == td


# ==========================================================================
# _plan — the work-queue planner
# ==========================================================================

def _plan_framework(tmp_path: Path, patch: str, *rows: dict,
                    lang: str = "py") -> Path:
    """Build a framework with one patch + finalized rows + a resolvable task dir."""
    fw = tmp_path / "fw"
    if lang == "py":
        pdir = fw / "autozyme_py" / "src" / "autozyme" / patch
        pdir.mkdir(parents=True)
        (pdir / "__init__.py").write_text(
            f"# Lifted from autozyme task `test_{patch}`\n"
        )
    else:
        pdir = fw / "autozyme_r" / "inst" / "patches" / patch
        pdir.mkdir(parents=True)
        (pdir / "patch.R").write_text(
            f"# Lifted from autozyme task `test_{patch}`\n"
        )
    _write_finalized(pdir / "speedups_finalized.tsv", *rows)
    td = fw / "optimized_task" / "cat" / f"test_{patch}"
    td.mkdir(parents=True)
    (td / "task.yaml").write_text("x: 1\n")
    return fw


def _plan_kwargs(**ov):
    base = dict(
        target_reps=2, also_stabilize=False, variance_pct=10.0,
        platform_keep="win", only_patches=set(), skip_patches=set(),
        tier_filter=None, threads_filter=None,
    )
    base.update(ov)
    return base


class TestPlan:
    def test_under_target_makes_workitem(self, tmp_path):
        fw = _plan_framework(
            tmp_path, "mgcv",
            _fin_row(variant="baseline", n_reps="1", sec_reps="2.0"),
            _fin_row(variant="patched", n_reps="1", sec_reps="1.0"),
        )
        items, stats = _plan(fw, **_plan_kwargs())
        assert len(items) == 1
        assert items[0].patch == "mgcv"
        assert stats["cells_under_target"] == 1
        assert items[0].reason == "low_reps"

    def test_already_ok_no_workitem(self, tmp_path):
        fw = _plan_framework(
            tmp_path, "mgcv",
            _fin_row(variant="baseline", n_reps="2", sec_reps="2.0,2.1"),
            _fin_row(variant="patched", n_reps="2", sec_reps="1.0,1.1"),
        )
        items, stats = _plan(fw, **_plan_kwargs())
        assert items == []
        assert stats["cells_already_ok"] == 1

    def test_oom_cell_skipped(self, tmp_path):
        fw = _plan_framework(
            tmp_path, "mgcv",
            _fin_row(variant="baseline", n_reps="0", status="OOM"),
            _fin_row(variant="patched", n_reps="0", status="OOM"),
        )
        items, stats = _plan(fw, **_plan_kwargs())
        assert items == []
        assert stats["skipped_oom"] == 1

    def test_skip_patch_user(self, tmp_path):
        fw = _plan_framework(
            tmp_path, "mgcv",
            _fin_row(variant="patched", n_reps="1"),
        )
        items, stats = _plan(fw, **_plan_kwargs(skip_patches={"mgcv"}))
        assert items == []
        assert stats["skipped_user"] == 1

    def test_only_filter(self, tmp_path):
        fw = _plan_framework(
            tmp_path, "mgcv",
            _fin_row(variant="patched", n_reps="1"),
        )
        items, _ = _plan(fw, **_plan_kwargs(only_patches={"other"}))
        assert items == []

    def test_platform_filter(self, tmp_path):
        fw = _plan_framework(
            tmp_path, "mgcv",
            _fin_row(variant="baseline", platform="mac", n_reps="1"),
            _fin_row(variant="patched", platform="mac", n_reps="1"),
        )
        # planner keeps win by default -> mac cells skipped
        items, _ = _plan(fw, **_plan_kwargs(platform_keep="win"))
        assert items == []
        items2, _ = _plan(fw, **_plan_kwargs(platform_keep="mac"))
        assert len(items2) == 1

    def test_tier_filter(self, tmp_path):
        fw = _plan_framework(
            tmp_path, "mgcv",
            _fin_row(variant="patched", tier="medium", n_reps="1"),
        )
        items, _ = _plan(fw, **_plan_kwargs(tier_filter={"small"}))
        assert items == []
        items2, _ = _plan(fw, **_plan_kwargs(tier_filter={"medium"}))
        assert len(items2) == 1

    def test_threads_filter(self, tmp_path):
        fw = _plan_framework(
            tmp_path, "mgcv",
            _fin_row(variant="patched", threads="8", n_reps="1"),
        )
        items, _ = _plan(fw, **_plan_kwargs(threads_filter={1}))
        assert items == []
        items2, _ = _plan(fw, **_plan_kwargs(threads_filter={8}))
        assert len(items2) == 1

    def test_thread_any_maps_to_1(self, tmp_path):
        fw = _plan_framework(
            tmp_path, "mgcv",
            _fin_row(variant="patched", threads="any", n_reps="1"),
        )
        items, _ = _plan(fw, **_plan_kwargs())
        assert len(items) == 1
        assert items[0].threads == 1

    def test_no_task_dir_skip(self, tmp_path):
        # patch with finalized rows but no resolvable task dir
        fw = tmp_path / "fw"
        pdir = fw / "autozyme_py" / "src" / "autozyme" / "orphan"
        pdir.mkdir(parents=True)
        (pdir / "__init__.py").write_text("no marker\n")
        _write_finalized(pdir / "speedups_finalized.tsv",
                         _fin_row(patch="orphan", variant="patched", n_reps="1"))
        items, stats = _plan(fw, **_plan_kwargs())
        assert items == []
        assert stats["skipped_no_task"] == 1

    def test_also_stabilize_high_variance(self, tmp_path):
        fw = _plan_framework(
            tmp_path, "mgcv",
            # n=2 but the two reps disagree by ~33% (> 10% threshold)
            _fin_row(variant="baseline", n_reps="2", sec_reps="2.0,2.0"),
            _fin_row(variant="patched", n_reps="2", sec_reps="1.0,1.4"),
        )
        items, stats = _plan(fw, **_plan_kwargs(also_stabilize=True,
                                                variance_pct=10.0))
        assert len(items) == 1
        assert items[0].reason == "high_variance"
        assert items[0].target_reps == 3
        assert stats["cells_high_var"] == 1

    def test_also_stabilize_low_variance_skipped(self, tmp_path):
        fw = _plan_framework(
            tmp_path, "mgcv",
            _fin_row(variant="baseline", n_reps="2", sec_reps="2.0,2.0"),
            _fin_row(variant="patched", n_reps="2", sec_reps="1.0,1.02"),
        )
        items, stats = _plan(fw, **_plan_kwargs(also_stabilize=True))
        assert items == []
        assert stats["cells_already_ok"] == 1

    def test_memory_first_sort(self, tmp_path):
        # two patches, cheaper-memory cell should sort first
        fw = tmp_path / "fw"
        for patch, mem in (("heavy", "9000"), ("light", "100")):
            pdir = fw / "autozyme_py" / "src" / "autozyme" / patch
            pdir.mkdir(parents=True)
            (pdir / "__init__.py").write_text(
                f"# Lifted from autozyme task `test_{patch}`\n"
            )
            _write_finalized(
                pdir / "speedups_finalized.tsv",
                _fin_row(patch=patch, variant="baseline", n_reps="1",
                         mem_mean=mem),
                _fin_row(patch=patch, variant="patched", n_reps="1",
                         mem_mean=mem),
            )
            td = fw / "optimized_task" / "cat" / f"test_{patch}"
            td.mkdir(parents=True)
            (td / "task.yaml").write_text("x: 1\n")
        items, _ = _plan(fw, **_plan_kwargs())
        assert [it.patch for it in items] == ["light", "heavy"]


# ==========================================================================
# _build_attest_cmd
# ==========================================================================

class TestBuildAttestCmd:
    def _wi(self, **ov):
        base = dict(
            patch="mgcv", lang="py", task_dir=Path("/tasks/test_mgcv"),
            tier="small", threads=4, target_reps=3, baseline_n=1, patched_n=2,
            reason="low_reps",
        )
        base.update(ov)
        return WorkItem(**base)

    def test_command_shape(self, tmp_path):
        cmd = _build_attest_cmd(self._wi(), tmp_path)
        assert "attest" in cmd
        assert "--name" in cmd and "mgcv" in cmd
        assert "--tiers" in cmd and "small" in cmd
        assert "--threads" in cmd and "4" in cmd
        assert "--rerun-baseline" in cmd
        assert "--no-preflight" in cmd

    def test_scanpy_name_mapped(self, tmp_path):
        cmd = _build_attest_cmd(self._wi(patch="scanpy_normalize"), tmp_path)
        i = cmd.index("--name")
        assert cmd[i + 1] == "scanpy"

    def test_reps_to_run_floor_one(self, tmp_path):
        # baseline_n=patched_n=target -> reps_to_run=0 -> command uses max(1,0)=1
        cmd = _build_attest_cmd(
            self._wi(target_reps=2, baseline_n=2, patched_n=2), tmp_path
        )
        i = cmd.index("--reps")
        assert cmd[i + 1] == "1"


# ==========================================================================
# _subprocess_env
# ==========================================================================

class TestSubprocessEnv:
    def test_prepends_cli_to_pythonpath(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PYTHONPATH", raising=False)
        env = _subprocess_env(tmp_path)
        assert str(tmp_path / "autozyme_cli") in env["PYTHONPATH"]

    def test_sets_publish_skip_and_mode(self, tmp_path):
        env = _subprocess_env(tmp_path)
        assert env["AUTOZYME_ATTEST_PUBLISH_MODE"] == "skip"
        assert env["AUTOZYME_MODE"] == "for_paper_omp"

    def test_preserves_existing_pythonpath(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PYTHONPATH", "/existing")
        env = _subprocess_env(tmp_path)
        assert "/existing" in env["PYTHONPATH"]
        assert str(tmp_path / "autozyme_cli") in env["PYTHONPATH"]


# ==========================================================================
# fallback-warning + pv-pass gates
# ==========================================================================

class TestFallbackWarnings:
    def test_clean_log_ok(self, tmp_path):
        log = tmp_path / "log"
        log.write_text("everything fine here\n")
        ok, note = _check_no_fallback_warnings(log)
        assert ok is True and note == ""

    def test_fallback_pattern_detected(self, tmp_path):
        log = tmp_path / "log"
        log.write_text("WARNING: falling back to upstream because reasons\n")
        ok, note = _check_no_fallback_warnings(log)
        assert ok is False
        assert "patched-fallback" in note

    def test_unreadable_log_does_not_block(self, tmp_path):
        ok, note = _check_no_fallback_warnings(tmp_path / "missing")
        assert ok is True


PV_HEADER = [
    "timestamp", "patch_name", "tier", "dataset", "rep_idx", "variant",
    "sec", "speedup_pct", "speedup_x", "peak_mb", "peak_mb_change_pct",
    "peak_mb_fold", "pass", "metrics_json", "framework_version",
    "package_version", "note", "system_os", "system_cpu", "system_ram_gb",
    "system_threads",
]


def _pv_row(**ov) -> dict:
    base = {
        "timestamp": "2026-06-01T12:00:00", "patch_name": "mgcv", "tier": "small",
        "dataset": "ds", "rep_idx": "1", "variant": "patched", "sec": "1.0",
        "speedup_pct": "50", "speedup_x": "2.0", "peak_mb": "100",
        "peak_mb_change_pct": "", "peak_mb_fold": "", "pass": "1",
        "metrics_json": "{}", "framework_version": "0.3.0",
        "package_version": "1.0", "note": "", "system_os": "Windows 11",
        "system_cpu": "CPU", "system_ram_gb": "127", "system_threads": "1",
    }
    base.update(ov)
    return base


def _write_pv(path: Path, *rows: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(PV_HEADER)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in PV_HEADER))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _wi_for_pv(task_dir, **ov):
    base = dict(
        patch="mgcv", lang="py", task_dir=task_dir, tier="small",
        threads=1, target_reps=2, baseline_n=1, patched_n=1, reason="low_reps",
    )
    base.update(ov)
    return WorkItem(**base)


class TestCheckPvNewRowsPassed:
    def test_missing_pv(self, tmp_path):
        item = _wi_for_pv(tmp_path)
        ok, note = _check_pv_new_rows_passed(item, "2026-06-01T00:00:00", io.StringIO())
        assert ok is False
        assert "missing" in note

    def test_all_passed(self, tmp_path):
        _write_pv(
            tmp_path / "package_verify.tsv",
            _pv_row(variant="baseline", sec="2.0", **{"pass": ""}),
            _pv_row(variant="patched", **{"pass": "1"}, sec="1.0"),
        )
        item = _wi_for_pv(tmp_path)
        ok, note = _check_pv_new_rows_passed(item, "2026-06-01T00:00:00", io.StringIO())
        assert ok is True and note == ""

    def test_failed_patched_surfaces_note(self, tmp_path):
        _write_pv(
            tmp_path / "package_verify.tsv",
            _pv_row(variant="patched", **{"pass": ""}, sec="1.0",
                    note="FileNotFoundError: missing dataset"),
        )
        item = _wi_for_pv(tmp_path)
        ok, note = _check_pv_new_rows_passed(item, "2026-06-01T00:00:00", io.StringIO())
        assert ok is False
        assert "silent-fail" in note

    def test_no_new_rows(self, tmp_path):
        # all rows are older than attest_start -> no new rows
        _write_pv(
            tmp_path / "package_verify.tsv",
            _pv_row(variant="patched", timestamp="2026-05-01T00:00:00"),
        )
        item = _wi_for_pv(tmp_path)
        ok, note = _check_pv_new_rows_passed(item, "2026-06-01T00:00:00", io.StringIO())
        assert ok is False
        assert "no new" in note

    def test_sibling_tier_ignored(self, tmp_path):
        # a failing row for a DIFFERENT tier must not poison our small verdict
        _write_pv(
            tmp_path / "package_verify.tsv",
            _pv_row(variant="patched", tier="small", **{"pass": "1"}, sec="1.0"),
            _pv_row(variant="patched", tier="large", **{"pass": ""}, sec="1.0",
                    note="boom"),
        )
        item = _wi_for_pv(tmp_path, tier="small")
        ok, _ = _check_pv_new_rows_passed(item, "2026-06-01T00:00:00", io.StringIO())
        assert ok is True

    def test_tiny_tier_alias_matches_small(self, tmp_path):
        _write_pv(
            tmp_path / "package_verify.tsv",
            _pv_row(variant="patched", tier="tiny", **{"pass": "1"}, sec="1.0"),
        )
        item = _wi_for_pv(tmp_path, tier="small")
        ok, _ = _check_pv_new_rows_passed(item, "2026-06-01T00:00:00", io.StringIO())
        assert ok is True


# ==========================================================================
# _copy_new_pv_rows_to_speedups — per-platform split + dedup
# ==========================================================================

class TestCopyNewPvRows:
    def test_no_pv_file(self, tmp_path):
        item = _wi_for_pv(tmp_path / "task")
        (tmp_path / "task").mkdir()
        added, skipped = _copy_new_pv_rows_to_speedups(
            item, tmp_path / "fw", "2026-06-01T00:00:00", io.StringIO()
        )
        assert (added, skipped) == (0, 0)

    def test_copies_to_platform_shard(self, tmp_path):
        fw = tmp_path / "fw"
        task = tmp_path / "task"
        task.mkdir()
        _write_pv(
            task / "package_verify.tsv",
            _pv_row(variant="baseline", sec="2.0", system_os="Windows 11"),
            _pv_row(variant="patched", sec="1.0", system_os="Windows 11"),
        )
        item = _wi_for_pv(task)
        added, skipped = _copy_new_pv_rows_to_speedups(
            item, fw, "2026-06-01T00:00:00", io.StringIO()
        )
        assert added == 2 and skipped == 0
        shard = fw / "autozyme_py" / "src" / "autozyme" / "mgcv" / "speedups.win.tsv"
        assert shard.is_file()
        assert "patched" in shard.read_text(encoding="utf-8")

    def test_dedup_against_existing_shard(self, tmp_path):
        fw = tmp_path / "fw"
        task = tmp_path / "task"
        task.mkdir()
        _write_pv(
            task / "package_verify.tsv",
            _pv_row(variant="patched", sec="1.0", system_os="Windows 11"),
        )
        item = _wi_for_pv(task)
        log = io.StringIO()
        # first copy
        a1, _ = _copy_new_pv_rows_to_speedups(item, fw, "2026-06-01T00:00:00", log)
        # second copy of the same rows -> all dedup-skipped
        a2, s2 = _copy_new_pv_rows_to_speedups(item, fw, "2026-06-01T00:00:00", log)
        assert a1 == 1
        assert a2 == 0 and s2 == 1

    def test_no_rows_after_timestamp(self, tmp_path):
        fw = tmp_path / "fw"
        task = tmp_path / "task"
        task.mkdir()
        _write_pv(
            task / "package_verify.tsv",
            _pv_row(variant="patched", timestamp="2026-05-01T00:00:00"),
        )
        item = _wi_for_pv(task)
        added, skipped = _copy_new_pv_rows_to_speedups(
            item, fw, "2026-06-01T00:00:00", io.StringIO()
        )
        assert (added, skipped) == (0, 0)


# ==========================================================================
# _resource_fits — admission probe (no psutil branch + caps)
# ==========================================================================

class TestResourceFits:
    def _wi(self, **ov):
        base = dict(
            patch="p", lang="py", task_dir=Path("/x"), tier="small",
            threads=1, target_reps=2, baseline_n=1, patched_n=1,
            reason="low_reps", cost_mem_mb=100.0,
        )
        base.update(ov)
        return WorkItem(**base)

    def test_per_patch_cap_blocks_second(self):
        item = self._wi(patch="dup")
        in_flight = [self._wi(patch="dup")]
        ok, why = _resource_fits(item, in_flight, None, 16, 0.0)
        assert ok is False
        assert why.startswith("patch=")

    def test_cpu_cap_blocks(self):
        # 16 cores * 0.8 = 12 cap; in-flight 10 + need 4 = 14 > 12
        item = self._wi(patch="a", threads=4)
        in_flight = [self._wi(patch="b", threads=10)]
        ok, why = _resource_fits(item, in_flight, None, 16, 0.0)
        assert ok is False
        assert "cpu_used" in why

    def test_fits_when_no_psutil(self):
        item = self._wi(patch="a", threads=1)
        ok, why = _resource_fits(item, [], None, 16, 0.0)
        assert ok is True and why == ""

    def test_memory_decline(self):
        class FakePsutil:
            @staticmethod
            def virtual_memory():
                # 1000 MB available, total 10000 MB -> buffer = 2000 MB
                return type("M", (), {"available": 1000 * 1024 * 1024})()

        item = self._wi(patch="a", threads=1, cost_mem_mb=500.0)
        ok, why = _resource_fits(item, [], FakePsutil, 16, 10000.0)
        assert ok is False
        assert "mem_avail" in why

    def test_memory_ok(self):
        class FakePsutil:
            @staticmethod
            def virtual_memory():
                return type("M", (), {"available": 9000 * 1024 * 1024})()

        item = self._wi(patch="a", threads=1, cost_mem_mb=100.0)
        ok, why = _resource_fits(item, [], FakePsutil, 16, 10000.0)
        assert ok is True


# ==========================================================================
# _print_plan_summary
# ==========================================================================

class TestPrintPlanSummary:
    def test_empty(self, capsys):
        _print_plan_summary([], {"discovered": 3, "cells_already_ok": 3})
        out = capsys.readouterr().out
        assert "backfill plan" in out
        assert "discovered: 3" in out
        assert "work items: 0" in out

    def test_with_items(self, capsys):
        items = [
            WorkItem(patch="mgcv", lang="py", task_dir=Path("/x"), tier="small",
                     threads=1, target_reps=2, baseline_n=1, patched_n=1,
                     reason="low_reps"),
        ]
        _print_plan_summary(items, {"discovered": 1})
        out = capsys.readouterr().out
        assert "mgcv" in out
        assert "1 cells" in out
        assert "small" in out


# ==========================================================================
# cmd_backfill — top-level (plan mode, no-framework, nothing-to-do)
# ==========================================================================

def _backfill_args(**ov):
    base = dict(
        framework_root=None, platform="win", skip=None, only=None, tiers=None,
        threads=None, target_reps=2, also_stabilize=False, variance_pct=10.0,
        limit=0, plan=True, small_workers=2, large_workers=1, timeout=3600,
        plain=True,
    )
    base.update(ov)
    return argparse.Namespace(**base)


class TestCmdBackfill:
    def test_no_framework_root_errors(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        # find_framework_root returns None for a bare tmp dir
        rc = cmd_backfill(_backfill_args(framework_root=None))
        assert rc == 1
        assert "requires autozyme-framework" in capsys.readouterr().err

    def test_plan_mode_returns_zero(self, tmp_path, capsys):
        fw = _plan_framework(
            tmp_path, "mgcv",
            _fin_row(variant="baseline", n_reps="1"),
            _fin_row(variant="patched", n_reps="1"),
        )
        rc = cmd_backfill(_backfill_args(framework_root=str(fw), plan=True))
        assert rc == 0
        out = capsys.readouterr().out
        assert "backfill plan" in out
        assert "mgcv" in out

    def test_nothing_to_do(self, tmp_path, capsys):
        fw = _plan_framework(
            tmp_path, "mgcv",
            _fin_row(variant="baseline", n_reps="2", sec_reps="2.0,2.1"),
            _fin_row(variant="patched", n_reps="2", sec_reps="1.0,1.1"),
        )
        rc = cmd_backfill(_backfill_args(framework_root=str(fw), plan=False))
        assert rc == 0
        assert "nothing to do" in capsys.readouterr().out

    def test_bad_threads_value_errors(self, tmp_path, capsys):
        fw = _plan_framework(tmp_path, "mgcv", _fin_row(variant="patched"))
        rc = cmd_backfill(_backfill_args(framework_root=str(fw), threads="abc"))
        assert rc == 1
        assert "bad --threads" in capsys.readouterr().err

    def test_default_skip_applied(self, tmp_path):
        # rctd is in the default skip set; a patch named rctd is never planned
        fw = _plan_framework(
            tmp_path, "rctd",
            _fin_row(patch="rctd", variant="patched", n_reps="1"),
        )
        rc = cmd_backfill(_backfill_args(framework_root=str(fw), plan=True))
        assert rc == 0
        # rctd is in _DEFAULT_SKIP
        assert "rctd" in _DEFAULT_SKIP

    def test_limit_truncates(self, tmp_path, monkeypatch, capsys):
        fw = tmp_path / "fw"
        for patch in ("aaa", "bbb"):
            pdir = fw / "autozyme_py" / "src" / "autozyme" / patch
            pdir.mkdir(parents=True)
            (pdir / "__init__.py").write_text(
                f"# Lifted from autozyme task `test_{patch}`\n"
            )
            _write_finalized(
                pdir / "speedups_finalized.tsv",
                _fin_row(patch=patch, variant="baseline", n_reps="1"),
                _fin_row(patch=patch, variant="patched", n_reps="1"),
            )
            td = fw / "optimized_task" / "cat" / f"test_{patch}"
            td.mkdir(parents=True)
            (td / "task.yaml").write_text("x: 1\n")
        rc = cmd_backfill(_backfill_args(framework_root=str(fw), plan=True, limit=1))
        assert rc == 0
        out = capsys.readouterr().out
        assert "work items: 1" in out

    def test_full_run_invokes_orchestrate(self, tmp_path, monkeypatch, capsys):
        fw = _plan_framework(
            tmp_path, "mgcv",
            _fin_row(variant="baseline", n_reps="1"),
            _fin_row(variant="patched", n_reps="1"),
        )
        seen = {}

        def fake_orch(items, framework, log_dir, **kw):
            seen["n"] = len(items)
            return [ExecResult(item=items[0], rc=0, duration_sec=1.0, note="ok")]

        monkeypatch.setattr(bf, "_orchestrate", fake_orch)
        rc = cmd_backfill(_backfill_args(framework_root=str(fw), plan=False))
        assert rc == 0
        assert seen["n"] == 1
        assert "[complete] 1 items, 0 failed" in capsys.readouterr().out

    def test_full_run_reports_failures(self, tmp_path, monkeypatch, capsys):
        fw = _plan_framework(
            tmp_path, "mgcv",
            _fin_row(variant="baseline", n_reps="1"),
            _fin_row(variant="patched", n_reps="1"),
        )

        def fake_orch(items, framework, log_dir, **kw):
            return [ExecResult(item=items[0], rc=2, duration_sec=1.0, note="boom")]

        monkeypatch.setattr(bf, "_orchestrate", fake_orch)
        rc = cmd_backfill(_backfill_args(framework_root=str(fw), plan=False))
        assert rc == 1
        assert "1 failed" in capsys.readouterr().out

    def test_seurat_preflight_gate_blocks(self, tmp_path, monkeypatch, capsys):
        fw = _plan_framework(
            tmp_path, "seurat_pca",
            _fin_row(patch="seurat_pca", variant="baseline", n_reps="1"),
            _fin_row(patch="seurat_pca", variant="patched", n_reps="1"),
            lang="R",
        )
        # seurat_pca needs python; force the preflight probe to fail -> abort rc=2
        monkeypatch.setattr(bf, "_preflight_seurat_python",
                            lambda env: (False, "no python"))
        called = []
        monkeypatch.setattr(bf, "_orchestrate",
                            lambda *a, **k: called.append(1) or [])
        rc = cmd_backfill(_backfill_args(framework_root=str(fw), plan=False))
        assert rc == 2
        assert called == []
        assert "preflight FAIL" in capsys.readouterr().out


# ==========================================================================
# _ProgressDashboard — plain-stdout state machine (no rich)
# ==========================================================================

class TestProgressDashboard:
    def test_plain_progress_tracks_done_failed(self, capsys):
        dash = bf._ProgressDashboard(total=2, use_rich=False)
        with dash:
            dash.start_item("a/small/T1")
            dash.finish_item("a/small/T1", rc=0, duration=1.0, note="ok")
            dash.start_item("b/small/T1")
            dash.finish_item("b/small/T1", rc=2, duration=2.0, note="fail")
        assert dash.done == 2
        assert dash.failed == 1
        out = capsys.readouterr().out
        assert "[1/2]" in out and "OK" in out
        assert "FAIL rc=2" in out

    def test_in_flight_pop_on_finish(self):
        dash = bf._ProgressDashboard(total=1, use_rich=False)
        dash.start_item("x")
        assert "x" in dash.in_flight
        dash.finish_item("x", rc=0, duration=1.0, note="")
        assert "x" not in dash.in_flight


# ==========================================================================
# _publish_and_finalize — subprocess.run boundary monkeypatched
# ==========================================================================

class TestPublishAndFinalize:
    def test_finalize_success(self, tmp_path, monkeypatch):
        fw = tmp_path / "fw"
        task = tmp_path / "task"
        task.mkdir()
        _write_pv(
            task / "package_verify.tsv",
            _pv_row(variant="patched", sec="1.0", system_os="Windows 11"),
        )
        item = _wi_for_pv(task)
        monkeypatch.setattr(
            bf.subprocess, "run",
            lambda *a, **k: type("R", (), {"returncode": 0})(),
        )
        rc, note = bf._publish_and_finalize(
            item, fw, "2026-06-01T00:00:00", io.StringIO()
        )
        assert rc == 0
        assert "ok (+1 rows)" in note

    def test_finalize_failure_propagates(self, tmp_path, monkeypatch):
        fw = tmp_path / "fw"
        task = tmp_path / "task"
        task.mkdir()
        _write_pv(
            task / "package_verify.tsv",
            _pv_row(variant="patched", sec="1.0", system_os="Windows 11"),
        )
        item = _wi_for_pv(task)
        monkeypatch.setattr(
            bf.subprocess, "run",
            lambda *a, **k: type("R", (), {"returncode": 1})(),
        )
        rc, note = bf._publish_and_finalize(
            item, fw, "2026-06-01T00:00:00", io.StringIO()
        )
        assert rc == 1
        assert "finalize_speedups failed" in note


# ==========================================================================
# _execute_one — full per-cell driver with subprocess.Popen monkeypatched
# ==========================================================================

class _FakePopen:
    def __init__(self, *a, **k):
        self.returncode = 0
        self.pid = 1234

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class TestExecuteOne:
    def test_success_publishes(self, tmp_path, monkeypatch):
        fw = tmp_path / "fw"
        task = tmp_path / "task"
        task.mkdir()
        item = _wi_for_pv(task)
        log_dir = tmp_path / "logs"

        def fake_popen(cmd, **kw):
            # Simulate attest writing a passing pv batch with a fresh timestamp.
            _write_pv(
                task / "package_verify.tsv",
                _pv_row(variant="baseline", sec="2.0", timestamp="2027-01-01T00:00:00",
                        system_os="Windows 11"),
                _pv_row(variant="patched", sec="1.0", **{"pass": "1"},
                        timestamp="2027-01-01T00:00:00", system_os="Windows 11"),
            )
            return _FakePopen()

        monkeypatch.setattr(bf.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(
            bf, "_publish_and_finalize",
            lambda item, fw, iso, log: (0, "ok (+2 rows)"),
        )
        res = bf._execute_one(item, fw, log_dir, timeout_sec=0)
        assert res.rc == 0
        assert "ok" in res.note
        # a per-cell log was written
        logs = list(log_dir.glob("*.log"))
        assert len(logs) == 1

    def test_fallback_warning_blocks_publish(self, tmp_path, monkeypatch):
        fw = tmp_path / "fw"
        task = tmp_path / "task"
        task.mkdir()
        item = _wi_for_pv(task)
        log_dir = tmp_path / "logs"

        def fake_popen(cmd, **kw):
            # The attest log captured a silent fallback warning.
            return _FakePopen()

        monkeypatch.setattr(bf.subprocess, "Popen", fake_popen)
        # inject the fallback marker into the log by patching the check
        monkeypatch.setattr(
            bf, "_check_no_fallback_warnings",
            lambda log: (False, "patched-fallback: 'falling back to upstream'"),
        )
        published = []
        monkeypatch.setattr(
            bf, "_publish_and_finalize",
            lambda *a, **k: published.append(1) or (0, "ok"),
        )
        res = bf._execute_one(item, fw, log_dir, timeout_sec=0)
        assert res.rc == 3
        assert published == []  # publish was gated off

    def test_timeout_kills_and_marks_124(self, tmp_path, monkeypatch):
        fw = tmp_path / "fw"
        task = tmp_path / "task"
        task.mkdir()
        item = _wi_for_pv(task)
        log_dir = tmp_path / "logs"

        class _TOPopen(_FakePopen):
            def __init__(self, *a, **k):
                super().__init__()
                self._first = True

            def wait(self, timeout=None):
                if self._first and timeout is not None:
                    self._first = False
                    raise bf.subprocess.TimeoutExpired(cmd="x", timeout=timeout)
                return 0

        monkeypatch.setattr(bf.subprocess, "Popen", lambda *a, **k: _TOPopen())
        res = bf._execute_one(item, fw, log_dir, timeout_sec=5)
        assert res.rc == 124


# ==========================================================================
# _orchestrate — dispatch loop with _execute_one monkeypatched (no subprocess)
# ==========================================================================

class TestOrchestrate:
    def test_runs_all_items_serially(self, tmp_path, monkeypatch):
        items = [
            WorkItem(patch=f"p{i}", lang="py", task_dir=Path("/x"), tier="small",
                     threads=1, target_reps=2, baseline_n=1, patched_n=1,
                     reason="low_reps", cost_mem_mb=10.0)
            for i in range(3)
        ]
        executed = []

        def fake_exec(item, framework, log_dir, timeout_sec):
            executed.append(item.patch)
            return ExecResult(item=item, rc=0, duration_sec=0.1, note="ok")

        monkeypatch.setattr(bf, "_execute_one", fake_exec)
        # no psutil -> non-blocking submit-all path
        monkeypatch.setattr(bf, "_try_import_psutil", lambda: None)
        results = bf._orchestrate(
            items, tmp_path, tmp_path / "logs",
            small_workers=2, large_workers=1, timeout_sec=0, use_rich=False,
        )
        assert len(results) == 3
        assert set(executed) == {"p0", "p1", "p2"}

    def test_worker_exception_recorded(self, tmp_path, monkeypatch):
        items = [
            WorkItem(patch="p", lang="py", task_dir=Path("/x"), tier="small",
                     threads=1, target_reps=2, baseline_n=1, patched_n=1,
                     reason="low_reps", cost_mem_mb=10.0),
        ]

        def boom(*a, **k):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(bf, "_execute_one", boom)
        monkeypatch.setattr(bf, "_try_import_psutil", lambda: None)
        results = bf._orchestrate(
            items, tmp_path, tmp_path / "logs",
            small_workers=1, large_workers=1, timeout_sec=0, use_rich=False,
        )
        assert len(results) == 1
        assert results[0].rc == 255
        assert "exception" in results[0].note

    def test_psutil_gated_admission_runs_all(self, tmp_path, monkeypatch):
        # With psutil present, the dispatcher gates on the resource probe but
        # still admits + completes every item (generous fake budget).
        items = [
            WorkItem(patch=f"p{i}", lang="py", task_dir=Path("/x"), tier="small",
                     threads=1, target_reps=2, baseline_n=1, patched_n=1,
                     reason="low_reps", cost_mem_mb=10.0)
            for i in range(2)
        ]
        executed = []
        monkeypatch.setattr(
            bf, "_execute_one",
            lambda item, fw, ld, ts: executed.append(item.patch)
            or ExecResult(item=item, rc=0, duration_sec=0.01, note="ok"),
        )

        class FakePsutil:
            @staticmethod
            def virtual_memory():
                return type("M", (), {
                    "available": 50_000 * 1024 * 1024,
                    "total": 64_000 * 1024 * 1024,
                })()

        monkeypatch.setattr(bf, "_try_import_psutil", lambda: FakePsutil)
        results = bf._orchestrate(
            items, tmp_path, tmp_path / "logs",
            small_workers=2, large_workers=1, timeout_sec=0, use_rich=False,
        )
        assert len(results) == 2
        assert set(executed) == {"p0", "p1"}


# ==========================================================================
# rich dashboard _render + subprocess R probes
# ==========================================================================

class TestRichRenderAndProbes:
    def test_rich_render_produces_table(self, monkeypatch):
        if not bf._RICH_OK:
            pytest.skip("rich not installed")
        dash = bf._ProgressDashboard(total=3, use_rich=True)
        dash.in_flight["a/small/T1"] = dash.start_time
        dash.recent.append(("b/small/T1", 0, 1.0, "ok"))
        dash.recent.append(("c/small/T1", 2, 2.0, "fail"))
        dash.done = 2
        table = dash._render()
        # a rich Table object is returned (truthy, has columns)
        assert table is not None

    def test_platform_rscript_found(self, monkeypatch):
        monkeypatch.setattr(bf.os, "name", "posix")
        monkeypatch.setattr(
            bf.subprocess, "run",
            lambda *a, **k: type("R", (), {"returncode": 0})(),
        )
        assert bf._platform_rscript() == "Rscript"

    def test_platform_rscript_not_found(self, monkeypatch):
        monkeypatch.setattr(bf.os, "name", "posix")

        def boom(*a, **k):
            raise OSError("no Rscript")

        monkeypatch.setattr(bf.subprocess, "run", boom)
        assert bf._platform_rscript() is None

    def test_preflight_seurat_python_no_rscript(self, monkeypatch):
        monkeypatch.setattr(bf, "_platform_rscript", lambda: None)
        ok, note = bf._preflight_seurat_python({})
        assert ok is False
        assert "Rscript not found" in note

    def test_preflight_seurat_python_probe_fails(self, monkeypatch):
        monkeypatch.setattr(bf, "_platform_rscript", lambda: "Rscript")
        monkeypatch.setattr(
            bf.subprocess, "run",
            lambda *a, **k: type("R", (), {
                "stdout": "FAIL: no Python with numpy+scipy located\n",
                "stderr": "", "returncode": 0,
            })(),
        )
        ok, note = bf._preflight_seurat_python({})
        assert ok is False
        assert "FAIL:" in note

    def test_preflight_seurat_python_ok(self, monkeypatch):
        monkeypatch.setattr(bf, "_platform_rscript", lambda: "Rscript")
        monkeypatch.setattr(
            bf.subprocess, "run",
            lambda *a, **k: type("R", (), {
                "stdout": "/opt/conda/bin/python \n",
                "stderr": "", "returncode": 0,
            })(),
        )
        ok, note = bf._preflight_seurat_python({})
        assert ok is True
        assert "python" in note.lower()
