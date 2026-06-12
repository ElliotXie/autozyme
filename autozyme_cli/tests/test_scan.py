"""Tests for zyme.scan — workspace + framework root detection, task discovery,
and per-task lifecycle phase inference."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from zyme.scan import (
    PHASE_ORDER,
    REFLECT_CATEGORIES,
    detect_phase,
    find_framework_root,
    find_tasks,
    find_workspace_root,
)


# --------------------------------------------------------------------------
# Fixture helpers — build a workspace tree on disk
# --------------------------------------------------------------------------

def _make_workspace(tmp_path: Path) -> Path:
    """Create a workspace whose autozyme-framework/ child is the real one
    (not a symlink). Returns the workspace path."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "autozyme-framework").mkdir()
    return ws


def _make_task(parent: Path, name: str, *, with_yaml: bool = True) -> Path:
    task = parent / name
    task.mkdir(parents=True, exist_ok=True)
    if with_yaml:
        (task / "task.yaml").write_text(f"target_repo: stub\n# task: {name}\n")
    return task


# --------------------------------------------------------------------------
# find_workspace_root / find_framework_root
# --------------------------------------------------------------------------

class TestFindWorkspaceRoot:
    def test_workspace_itself(self, tmp_path: Path):
        ws = _make_workspace(tmp_path)
        assert find_workspace_root(ws) == ws

    def test_subdirectory_of_workspace(self, tmp_path: Path):
        ws = _make_workspace(tmp_path)
        sub = ws / "category" / "test_x"
        sub.mkdir(parents=True)
        assert find_workspace_root(sub) == ws

    def test_outside_workspace_returns_none(self, tmp_path: Path):
        # tmp_path itself doesn't have autozyme-framework/.
        assert find_workspace_root(tmp_path) is None

    def test_symlinked_framework_not_mistaken_for_workspace(self, tmp_path: Path):
        # A category dir that contains `autozyme-framework -> ../autozyme-framework`
        # must NOT be reported as the workspace — only the dir whose framework
        # child is the *real* one qualifies.
        ws = _make_workspace(tmp_path)
        category = ws / "core_singlecell"
        category.mkdir()
        (category / "autozyme-framework").symlink_to(ws / "autozyme-framework")
        # Walking up from inside the category should land at ws, not category.
        sub = category / "test_x"
        sub.mkdir()
        assert find_workspace_root(sub) == ws


class TestFindFrameworkRoot:
    def test_returns_resolved_framework(self, tmp_path: Path):
        ws = _make_workspace(tmp_path)
        fw = find_framework_root(ws)
        assert fw == (ws / "autozyme-framework").resolve()

    def test_none_when_outside_workspace(self, tmp_path: Path):
        assert find_framework_root(tmp_path) is None


# --------------------------------------------------------------------------
# find_tasks
# --------------------------------------------------------------------------

class TestFindTasks:
    def test_discovers_task_dirs_at_top_level(self, tmp_path: Path):
        ws = _make_workspace(tmp_path)
        _make_task(ws, "test_a")
        _make_task(ws, "test_b")
        out = find_tasks([ws], max_depth=2, framework_root=ws / "autozyme-framework")
        names = sorted(p.name for p in out)
        assert names == ["test_a", "test_b"]

    def test_discovers_in_category_dirs(self, tmp_path: Path):
        ws = _make_workspace(tmp_path)
        cat = ws / "core_singlecell"
        cat.mkdir()
        _make_task(cat, "test_x")
        _make_task(cat, "test_y")
        out = find_tasks([ws], max_depth=3, framework_root=ws / "autozyme-framework")
        assert sorted(p.name for p in out) == ["test_x", "test_y"]

    def test_skips_framework_dir(self, tmp_path: Path):
        ws = _make_workspace(tmp_path)
        # Plant a task.yaml inside the framework — must NOT be picked up.
        (ws / "autozyme-framework" / "task.yaml").write_text("")
        _make_task(ws, "test_real")
        out = find_tasks([ws], max_depth=2, framework_root=ws / "autozyme-framework")
        assert [p.name for p in out] == ["test_real"]

    def test_does_not_descend_into_task_dir(self, tmp_path: Path):
        # Once task.yaml is found, walker stops descending.
        ws = _make_workspace(tmp_path)
        outer = _make_task(ws, "test_outer")
        # Inner task.yaml should NOT be found because outer was the boundary.
        nested = outer / "subtask"
        nested.mkdir()
        (nested / "task.yaml").write_text("")
        out = find_tasks([ws], max_depth=4, framework_root=ws / "autozyme-framework")
        assert [p.name for p in out] == ["test_outer"]

    def test_max_depth_caps_descent(self, tmp_path: Path):
        ws = _make_workspace(tmp_path)
        deep = ws / "a" / "b" / "c" / "test_deep"
        deep.mkdir(parents=True)
        (deep / "task.yaml").write_text("")
        # depth from ws to test_deep is 4 (a/b/c/test_deep).
        out_shallow = find_tasks([ws], max_depth=2, framework_root=None)
        out_deep = find_tasks([ws], max_depth=10, framework_root=None)
        assert len(out_shallow) == 0
        assert len(out_deep) == 1

    def test_does_not_follow_symlinks(self, tmp_path: Path):
        ws = _make_workspace(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        _make_task(elsewhere, "test_elsewhere")
        # Symlink from inside ws to elsewhere — followlinks=False means tasks
        # behind the symlink are NOT discovered.
        (ws / "external").symlink_to(elsewhere)
        out = find_tasks([ws], max_depth=4, framework_root=None)
        assert [p.name for p in out] == []

    def test_skips_root_that_doesnt_exist(self, tmp_path: Path):
        # Should not crash on a missing path; just return empty.
        out = find_tasks([tmp_path / "nope"], max_depth=2, framework_root=None)
        assert out == []

    def test_dedupes_when_root_overlaps(self, tmp_path: Path):
        # Calling with two roots that point at the same task should yield it once.
        ws = _make_workspace(tmp_path)
        _make_task(ws, "test_a")
        out = find_tasks([ws, ws / "."], max_depth=3, framework_root=None)
        assert [p.name for p in out] == ["test_a"]

    def test_results_sorted_stably(self, tmp_path: Path):
        ws = _make_workspace(tmp_path)
        _make_task(ws, "test_z")
        _make_task(ws, "test_a")
        _make_task(ws, "test_m")
        out = find_tasks([ws], max_depth=2, framework_root=None)
        names = [p.name for p in out]
        assert names == ["test_a", "test_m", "test_z"]


# --------------------------------------------------------------------------
# detect_phase
# --------------------------------------------------------------------------

class TestDetectPhase:
    def test_unknown_when_no_task_yaml(self, tmp_path: Path):
        # Bare directory — scaffold not done, no other phase done.
        out = detect_phase(tmp_path, framework_root=None)
        assert out["phase"] == "unknown"
        assert out["phases"]["scaffold"]["done"] is False

    def test_scaffold_phase(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        out = detect_phase(tmp_path, framework_root=None)
        assert out["phase"] == "scaffold"
        assert out["phases"]["scaffold"]["done"] is True
        assert out["phases"]["init"]["done"] is False

    def test_init_phase_requires_three_tiers_plus_refs_plus_pipeline(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: tiny, name: t, path: /abs/t.h5}\n"
            "  - {tier: medium, name: m, path: /abs/m.h5}\n"
            "  - {tier: large, name: l, path: /abs/l.h5}\n"
        )
        # Refs + pipeline + evaluate present.
        for tier in ("tiny", "medium", "large"):
            d = tmp_path / f"reference_output_{tier}"
            d.mkdir()
            (d / "marker.txt").write_text("")
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "run.py").write_text("")
        (tmp_path / "evaluate.py").write_text("")
        out = detect_phase(tmp_path, framework_root=None)
        assert out["phases"]["init"]["done"] is True

    def test_init_phase_accepts_current_reference_outputs_layout(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: tiny, name: t, path: /abs/t.h5}\n"
            "  - {tier: medium, name: m, path: /abs/m.h5}\n"
            "  - {tier: large, name: l, path: /abs/l.h5}\n"
        )
        for tier in ("tiny", "medium", "large"):
            d = tmp_path / "reference_outputs" / tier
            d.mkdir(parents=True)
            (d / "marker.txt").write_text("")
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "run.R").write_text("")
        (tmp_path / "evaluate.R").write_text("")
        out = detect_phase(tmp_path, framework_root=None)
        assert out["phases"]["init"]["done"] is True

    def test_init_not_done_when_refs_missing(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: tiny, name: t, path: /abs/t.h5}\n"
            "  - {tier: medium, name: m, path: /abs/m.h5}\n"
            "  - {tier: large, name: l, path: /abs/l.h5}\n"
        )
        (tmp_path / "pipeline").mkdir()
        (tmp_path / "pipeline" / "run.py").write_text("")
        (tmp_path / "evaluate.py").write_text("")
        # No reference_output_* dirs.
        out = detect_phase(tmp_path, framework_root=None)
        assert out["phases"]["init"]["done"] is False

    def test_iterate_phase_requires_rounds_plus_best_plus_artifact(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        (tmp_path / "results.tsv").write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tphase\n"
            "1\tabc1234\ttiny_a\t8.0\t20.0\t500\tkeep\t{}\tH\t\toptimize\n"
        )
        (tmp_path / ".zyme").mkdir()
        (tmp_path / ".zyme" / "best.ref").write_text("abc1234567890\n")
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts" / "001_abc1234_tiny").mkdir()
        out = detect_phase(tmp_path, framework_root=None)
        assert out["phases"]["iterate"]["done"] is True
        assert out["phases"]["iterate"]["rounds"] == 1

    def test_scaling_phase_requires_ood_plus_verify_rows(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n"
            "  - {tier: tiny, name: t, path: /abs/t.h5}\n"
            "  - {tier: ood_large, name: ol, path: /abs/ol.h5}\n"
        )
        (tmp_path / "verify.tsv").write_text(
            "header_a\theader_b\nrow1_a\trow1_b\n"
        )
        out = detect_phase(tmp_path, framework_root=None)
        assert out["phases"]["scaling"]["done"] is True
        assert out["phases"]["scaling"]["ood_tiers"] == ["ood_large"]
        assert out["phases"]["scaling"]["verify_rows"] == 1

    def test_scaling_not_done_without_ood(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "datasets:\n  - {tier: tiny, name: t, path: /abs/t.h5}\n"
        )
        (tmp_path / "verify.tsv").write_text("h\nr\n")
        out = detect_phase(tmp_path, framework_root=None)
        assert out["phases"]["scaling"]["done"] is False

    def test_package_phase_via_python_patch(self, tmp_path: Path):
        # Build a fake framework root with a packaged python patch under
        # autozyme_py/src/autozyme/<task>/__init__.py.
        ws = tmp_path / "ws"
        fw = ws / "autozyme-framework"
        task = ws / "test_foo"
        task.mkdir(parents=True)
        (task / "task.yaml").write_text("task: foo\n")
        patch_dir = fw / "autozyme_py" / "src" / "autozyme" / "foo"
        patch_dir.mkdir(parents=True)
        (patch_dir / "__init__.py").write_text("")
        out = detect_phase(task, framework_root=fw)
        assert out["phases"]["package"]["done"] is True
        assert Path(out["phases"]["package"]["patch_path"]).parts[-2:] == (
            "foo",
            "__init__.py",
        )

    def test_package_phase_via_r_patch(self, tmp_path: Path):
        ws = tmp_path / "ws"
        fw = ws / "autozyme-framework"
        task = ws / "test_bar"
        task.mkdir(parents=True)
        (task / "task.yaml").write_text("task: bar\n")
        r_patches = fw / "autozyme_r" / "inst" / "patches"
        r_patches.mkdir(parents=True)
        (r_patches / "bar.R").write_text("")
        out = detect_phase(task, framework_root=fw)
        assert out["phases"]["package"]["done"] is True

    def test_package_strips_test_prefix_fallback(self, tmp_path: Path):
        # Patch is filed under the bare name but task.yaml's `task:` field
        # has the test_ prefix.
        ws = tmp_path / "ws"
        fw = ws / "autozyme-framework"
        task = ws / "test_xyz"
        task.mkdir(parents=True)
        (task / "task.yaml").write_text("task: test_xyz\n")
        patch_dir = fw / "autozyme_py" / "src" / "autozyme" / "xyz"
        patch_dir.mkdir(parents=True)
        (patch_dir / "__init__.py").write_text("")
        out = detect_phase(task, framework_root=fw)
        assert out["phases"]["package"]["done"] is True

    def test_phase_summary_picks_rightmost_done(self, tmp_path: Path):
        # Scaffold + iterate done but not init/scaling/package — phase
        # summary is "iterate", with "init" listed as a gap.
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        (tmp_path / "results.tsv").write_text(
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tphase\n"
            "1\tabc1234\ttiny_a\t8.0\t20.0\t500\tkeep\t{}\tH\t\toptimize\n"
        )
        (tmp_path / ".zyme").mkdir()
        (tmp_path / ".zyme" / "best.ref").write_text("abc1234\n")
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts" / "001_abc1234_tiny").mkdir()
        out = detect_phase(tmp_path, framework_root=None)
        assert out["phase"] == "iterate"
        assert "init" in out["gaps"]

    def test_reflect_categories_default_false_without_framework(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        out = detect_phase(tmp_path, framework_root=None)
        for c in REFLECT_CATEGORIES:
            assert out["reflect"][c] is False

    def test_active_live_dispatch_from_ancestor_state(self, tmp_path: Path):
        task = tmp_path / "task_a"
        task.mkdir()
        (task / "task.yaml").write_text("target_repo: foo\n")
        dispatch = tmp_path / ".zyme_dispatch"
        dispatch.mkdir()
        (dispatch / "state.json").write_text(json.dumps({
            "master_pid": os.getpid(),
            "agent": "cursor",
            "model": "composer-2-fast",
            "queue": [{
                "name": "task_a",
                "task_dir": str(task),
                "status": "running",
                "reflect_status": None,
                "last_event_at": "2026-05-11T20:00:00Z",
            }],
        }))
        out = detect_phase(task, framework_root=None, active_recent_minutes=0)
        assert out["active"]["active"] is True
        assert out["active"]["status"] == "run"
        assert out["active"]["source"] == "dispatch"

    def test_active_recent_mtime_fallback(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        (tmp_path / "results.tsv").write_text("header\n")
        out = detect_phase(tmp_path, framework_root=None, active_recent_minutes=60)
        assert out["active"]["active"] is True
        assert out["active"]["status"] == "recent"
        assert out["active"]["source"] == "mtime"

    def test_active_recent_fallback_can_be_disabled(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        (tmp_path / "results.tsv").write_text("header\n")
        out = detect_phase(tmp_path, framework_root=None, active_recent_minutes=0)
        assert out["active"]["active"] is False
        assert out["active"]["status"] == "-"


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

class TestConstants:
    def test_phase_order(self):
        assert PHASE_ORDER == ["scaffold", "init", "iterate", "scaling", "package"]

    def test_reflect_categories(self):
        assert REFLECT_CATEGORIES == [
            "initialization", "iteration", "scaling", "packaging",
        ]
