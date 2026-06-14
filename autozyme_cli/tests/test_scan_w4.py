"""Wave-4 coverage tests for zyme.scan.

Targets the REACHABLE lines left uncovered by tests/test_scan.py:
  - _extract_lifted_from + _build_lifted_from_index (R folder/legacy + Py layouts)
  - _find_patch via the lifted-from reverse index (fallback path)
  - _detect_reflect (feedback file matching, both feedback dirs)
  - _latest_keep (header validation, backward walk, thread parse, bad pct)
  - _read_json (corrupt JSON path)
  - _path_matches_task
  - _dispatch_activity: refl / stale / label-None branches + name-match fallback
  - _activity_file_candidates (pipeline/memory/artifacts dirs)
  - detect_phase reflect-via-framework + latest_keep surfaced
  - OSError-guard edge paths (_refs_present, _has_round_dir, _verify_data_rows)

Does NOT touch the pre-existing tests/test_scan.py. Subprocess-only logic
(none here — scan.py is pure filesystem) is fully reachable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import zyme.scan as S
from zyme.scan import (
    _activity_file_candidates,
    _build_lifted_from_index,
    _detect_reflect,
    _dispatch_activity,
    _extract_lifted_from,
    _find_patch,
    _has_round_dir,
    _latest_keep,
    _path_matches_task,
    _read_json,
    _refs_present,
    _verify_data_rows,
    detect_phase,
)


# --------------------------------------------------------------------------
# _extract_lifted_from
# --------------------------------------------------------------------------
class TestExtractLiftedFrom:
    def test_extracts_marker(self, tmp_path: Path):
        f = tmp_path / "patch.R"
        f.write_text("# Lifted from autozyme task `test_foo`\nfoo <- 1\n")
        assert _extract_lifted_from(f) == "test_foo"

    def test_extracts_double_backtick(self, tmp_path: Path):
        f = tmp_path / "patch.R"
        f.write_text("Lifted from autozyme task ``test_bar``\n")
        assert _extract_lifted_from(f) == "test_bar"

    def test_no_marker_returns_none(self, tmp_path: Path):
        f = tmp_path / "patch.R"
        f.write_text("just some code\n")
        assert _extract_lifted_from(f) is None

    def test_missing_file_returns_none(self, tmp_path: Path):
        assert _extract_lifted_from(tmp_path / "nope.R") is None


# --------------------------------------------------------------------------
# _build_lifted_from_index — R folder layout, R legacy single-file, Py layout
# --------------------------------------------------------------------------
class TestBuildLiftedFromIndex:
    def setup_method(self):
        # The index is lru_cached on the framework_root str; clear so each
        # test builds against its own tmp tree.
        _build_lifted_from_index.cache_clear()

    def test_r_folder_layout(self, tmp_path: Path):
        patch_dir = tmp_path / "autozyme_r" / "inst" / "patches" / "mypkg"
        patch_dir.mkdir(parents=True)
        (patch_dir / "patch.R").write_text(
            "# Lifted from autozyme task `test_folder`\n"
        )
        idx = _build_lifted_from_index(str(tmp_path))
        assert idx["test_folder"].endswith("mypkg/patch.R")

    def test_r_legacy_single_file(self, tmp_path: Path):
        patches = tmp_path / "autozyme_r" / "inst" / "patches"
        patches.mkdir(parents=True)
        (patches / "legacy.R").write_text(
            "# Lifted from autozyme task `test_legacy`\n"
        )
        idx = _build_lifted_from_index(str(tmp_path))
        assert idx["test_legacy"].endswith("legacy.R")

    def test_r_non_r_file_ignored(self, tmp_path: Path):
        patches = tmp_path / "autozyme_r" / "inst" / "patches"
        patches.mkdir(parents=True)
        # A .txt file (not .R, not a dir) is skipped.
        (patches / "notes.txt").write_text(
            "# Lifted from autozyme task `test_ignored`\n"
        )
        idx = _build_lifted_from_index(str(tmp_path))
        assert "test_ignored" not in idx

    def test_py_layout(self, tmp_path: Path):
        d = tmp_path / "autozyme_py" / "src" / "autozyme" / "pypatch"
        d.mkdir(parents=True)
        (d / "__init__.py").write_text(
            '"""Lifted from autozyme task `test_py`"""\n'
        )
        idx = _build_lifted_from_index(str(tmp_path))
        assert idx["test_py"].endswith("pypatch/__init__.py")

    def test_py_underscore_dir_skipped(self, tmp_path: Path):
        d = tmp_path / "autozyme_py" / "src" / "autozyme" / "_private"
        d.mkdir(parents=True)
        (d / "__init__.py").write_text(
            '"""Lifted from autozyme task `test_priv`"""\n'
        )
        idx = _build_lifted_from_index(str(tmp_path))
        assert "test_priv" not in idx

    def test_empty_tree_returns_empty(self, tmp_path: Path):
        assert _build_lifted_from_index(str(tmp_path)) == {}


# --------------------------------------------------------------------------
# _find_patch — falls back to the reverse index when direct lookup misses
# --------------------------------------------------------------------------
class TestFindPatchViaIndex:
    def setup_method(self):
        _build_lifted_from_index.cache_clear()

    def test_direct_py_lookup(self, tmp_path: Path):
        d = tmp_path / "autozyme_py" / "src" / "autozyme" / "foo"
        d.mkdir(parents=True)
        (d / "__init__.py").write_text("")
        assert _find_patch(tmp_path, "foo").endswith("foo/__init__.py")

    def test_direct_r_lookup(self, tmp_path: Path):
        patches = tmp_path / "autozyme_r" / "inst" / "patches"
        patches.mkdir(parents=True)
        (patches / "bar.R").write_text("")
        assert _find_patch(tmp_path, "bar").endswith("bar.R")

    def test_fallback_to_lifted_index(self, tmp_path: Path):
        # Patch dir is named after the upstream package (mypkg) but the task
        # is test_thing — direct lookup by "test_thing" misses, the lifted
        # marker links them.
        d = tmp_path / "autozyme_py" / "src" / "autozyme" / "mypkg"
        d.mkdir(parents=True)
        (d / "__init__.py").write_text(
            '"""Lifted from autozyme task `test_thing`"""\n'
        )
        got = _find_patch(tmp_path, "test_thing")
        assert got is not None and got.endswith("mypkg/__init__.py")

    def test_no_match_returns_none(self, tmp_path: Path):
        assert _find_patch(tmp_path, "absent") is None

    def test_empty_stem_skipped(self, tmp_path: Path):
        # An empty stem is skipped without crashing.
        assert _find_patch(tmp_path, "", "also_absent") is None


# --------------------------------------------------------------------------
# _detect_reflect
# --------------------------------------------------------------------------
class TestDetectReflect:
    def test_all_false_when_no_dirs(self, tmp_path: Path):
        out = _detect_reflect(tmp_path, "test_x")
        assert out == {c: False for c in S.REFLECT_CATEGORIES}

    def test_matches_feedback_file(self, tmp_path: Path):
        fb = tmp_path / "reflections" / "prompt_reflect_feedback"
        fb.mkdir(parents=True)
        (fb / "iteration_test_x_note.md").write_text("note")
        out = _detect_reflect(tmp_path, "test_x")
        assert out["iteration"] is True
        assert out["initialization"] is False

    def test_matches_in_second_feedback_dir(self, tmp_path: Path):
        fb = tmp_path / "reflections" / "zyme_cli_feedback"
        fb.mkdir(parents=True)
        (fb / "packaging_test_y_xyz.md").write_text("note")
        out = _detect_reflect(tmp_path, "test_y")
        assert out["packaging"] is True

    def test_non_md_file_ignored(self, tmp_path: Path):
        fb = tmp_path / "reflections" / "prompt_reflect_feedback"
        fb.mkdir(parents=True)
        (fb / "scaling_test_z_note.txt").write_text("note")
        out = _detect_reflect(tmp_path, "test_z")
        assert out["scaling"] is False

    def test_matches_on_dir_name_stem(self, tmp_path: Path):
        # Two stems passed; the second (dir name) matches.
        fb = tmp_path / "reflections" / "prompt_reflect_feedback"
        fb.mkdir(parents=True)
        (fb / "initialization_test_dir.md").write_text("note")
        out = _detect_reflect(tmp_path, "other_name", "test_dir")
        assert out["initialization"] is True


# --------------------------------------------------------------------------
# _latest_keep
# --------------------------------------------------------------------------
HEADER = ("round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\t"
          "status\tmetrics_json\thypothesis\tdescription\tphase\tthread\n")


class TestLatestKeep:
    def test_missing_file(self, tmp_path: Path):
        assert _latest_keep(tmp_path / "nope.tsv") is None

    def test_header_only(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text(HEADER)
        assert _latest_keep(p) is None

    def test_missing_required_column(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        # No speedup_pct column -> None
        p.write_text("round\tcommit\tdataset\tstatus\n1\tabc\tt\tkeep\n")
        assert _latest_keep(p) is None

    def test_returns_latest_keep_with_thread(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text(
            HEADER
            + "1\tabc\ttiny_a\t8.0\t10.0\t500\tkeep\t{}\tH\td\toptimize\t1\n"
            + "2\tdef\tmedium_b\t6.0\t25.0\t600\tkeep\t{}\tH\td\toptimize\t4\n"
        )
        out = _latest_keep(p)
        assert out["round"] == "2"
        assert out["dataset"] == "medium_b"
        assert out["thread"] == 4
        assert out["speedup_pct"] == pytest.approx(25.0)

    def test_skips_non_keep_rows(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text(
            HEADER
            + "1\tabc\ttiny_a\t8.0\t10.0\t500\tkeep\t{}\tH\td\toptimize\t1\n"
            + "2\tdef\tmedium_b\t6.0\t25.0\t600\treject\t{}\tH\td\toptimize\t4\n"
        )
        out = _latest_keep(p)
        # newest keep is round 1 (round 2 was rejected)
        assert out["round"] == "1"

    def test_bad_speedup_pct_becomes_none(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text(
            HEADER
            + "1\tabc\ttiny_a\t8.0\tNaNN\t500\tkeep\t{}\tH\td\toptimize\t1\n"
        )
        out = _latest_keep(p)
        assert out["speedup_pct"] is None

    def test_bad_thread_left_none(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text(
            HEADER
            + "1\tabc\ttiny_a\t8.0\t10.0\t500\tkeep\t{}\tH\td\toptimize\tNaN\n"
        )
        out = _latest_keep(p)
        assert out["thread"] is None

    def test_short_row_skipped(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        # A truncated row shorter than the status column index is skipped.
        p.write_text(HEADER + "1\tabc\n")
        assert _latest_keep(p) is None

    def test_blank_lines_skipped(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text(
            HEADER
            + "\n"
            + "1\tabc\ttiny_a\t8.0\t10.0\t500\tkeep\t{}\tH\td\toptimize\t1\n"
        )
        out = _latest_keep(p)
        assert out["round"] == "1"


# --------------------------------------------------------------------------
# _read_json
# --------------------------------------------------------------------------
class TestReadJson:
    def test_valid(self, tmp_path: Path):
        p = tmp_path / "x.json"
        p.write_text('{"a": 1}')
        assert _read_json(p) == {"a": 1}

    def test_corrupt_returns_none(self, tmp_path: Path):
        p = tmp_path / "x.json"
        p.write_text("{not json")
        assert _read_json(p) is None

    def test_missing_returns_none(self, tmp_path: Path):
        assert _read_json(tmp_path / "nope.json") is None


# --------------------------------------------------------------------------
# _path_matches_task
# --------------------------------------------------------------------------
class TestPathMatchesTask:
    def test_empty_false(self, tmp_path: Path):
        assert _path_matches_task("", tmp_path) is False

    def test_matches_resolved(self, tmp_path: Path):
        task = (tmp_path / "t").resolve()
        task.mkdir()
        assert _path_matches_task(str(task), task) is True

    def test_non_match(self, tmp_path: Path):
        task = (tmp_path / "t").resolve()
        task.mkdir()
        assert _path_matches_task(str(tmp_path / "other"), task) is False


# --------------------------------------------------------------------------
# _dispatch_activity branches
# --------------------------------------------------------------------------
def _write_state(task: Path, state: dict, *, at: Path | None = None) -> None:
    d = (at or task.parent) / ".zyme_dispatch"
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(state))


class TestDispatchActivity:
    def test_no_state_returns_none(self, tmp_path: Path):
        task = tmp_path / "t"
        task.mkdir()
        assert _dispatch_activity(task) is None

    def test_corrupt_state_skipped(self, tmp_path: Path):
        task = tmp_path / "t"
        task.mkdir()
        d = tmp_path / ".zyme_dispatch"
        d.mkdir()
        (d / "state.json").write_text("{broken")
        assert _dispatch_activity(task) is None

    def test_refl_running_alive(self, tmp_path: Path):
        task = tmp_path / "t"
        task.mkdir()
        _write_state(task, {
            "master_pid": os.getpid(),
            "agent": "claude",
            "model": "opus",
            "queue": [{
                "name": "t", "task_dir": str(task),
                "status": "done", "reflect_status": "running",
            }],
        })
        out = _dispatch_activity(task)
        assert out["status"] == "refl"
        assert out["active"] is True
        assert out["confidence"] == "live"

    def test_stale_when_pid_dead(self, tmp_path: Path):
        task = tmp_path / "t"
        task.mkdir()
        # PID 0 is never a live master in pid_alive's eyes.
        _write_state(task, {
            "master_pid": 0,
            "queue": [{
                "name": "t", "task_dir": str(task),
                "status": "running", "reflect_status": None,
            }],
        })
        out = _dispatch_activity(task)
        assert out["status"] == "stale"
        assert out["active"] is False
        assert out["confidence"] == "stale"

    def test_label_none_returns_none(self, tmp_path: Path):
        # A task present in the queue but in a terminal status with no
        # reflect-running yields label=None -> the function returns None.
        task = tmp_path / "t"
        task.mkdir()
        _write_state(task, {
            "master_pid": os.getpid(),
            "queue": [{
                "name": "t", "task_dir": str(task),
                "status": "done", "reflect_status": "done",
            }],
        })
        assert _dispatch_activity(task) is None

    def test_matches_by_name_when_dir_differs(self, tmp_path: Path):
        task = tmp_path / "t"
        task.mkdir()
        _write_state(task, {
            "master_pid": os.getpid(),
            "queue": [{
                # task_dir points elsewhere, but name matches the dir name.
                "name": "t", "task_dir": "/nonexistent/elsewhere",
                "status": "pending", "reflect_status": None,
            }],
        })
        out = _dispatch_activity(task)
        assert out is not None
        assert out["status"] == "pend"

    def test_non_dict_queue_entry_skipped(self, tmp_path: Path):
        task = tmp_path / "t"
        task.mkdir()
        _write_state(task, {
            "master_pid": os.getpid(),
            "queue": ["not a dict", {
                "name": "t", "task_dir": str(task),
                "status": "running", "reflect_status": None,
            }],
        })
        out = _dispatch_activity(task)
        assert out["status"] == "run"


# --------------------------------------------------------------------------
# _activity_file_candidates — pipeline/memory/artifacts traversal
# --------------------------------------------------------------------------
class TestActivityFileCandidates:
    def test_yields_pipeline_and_artifacts_files(self, tmp_path: Path):
        task = tmp_path / "t"
        task.mkdir()
        (task / "results.tsv").write_text("h\n")
        (task / "pipeline").mkdir()
        (task / "pipeline" / "run.py").write_text("x")
        (task / "memory").mkdir()
        (task / "memory" / "note.md").write_text("m")
        arts = task / "artifacts" / "001_abc_tiny"
        arts.mkdir(parents=True)
        (arts / "log.txt").write_text("l")
        names = {p.name for p in _activity_file_candidates(task)}
        assert "run.py" in names
        assert "note.md" in names
        assert "log.txt" in names

    def test_skips_non_dir_artifact_entries(self, tmp_path: Path):
        task = tmp_path / "t"
        task.mkdir()
        arts = task / "artifacts"
        arts.mkdir()
        (arts / "stray.txt").write_text("not a round dir")
        # Should not crash; the stray file is not a round dir so its contents
        # aren't iterated.
        names = [p.name for p in _activity_file_candidates(task)]
        assert "stray.txt" not in names


# --------------------------------------------------------------------------
# OSError-guarded edge paths (reachable by pointing at a file-as-dir)
# --------------------------------------------------------------------------
class TestEdgeGuards:
    def test_refs_present_fewer_than_three(self, tmp_path: Path):
        assert _refs_present(tmp_path, [{"tier": "tiny"}]) is False

    def test_refs_present_missing_dir(self, tmp_path: Path):
        ds = [{"tier": "tiny"}, {"tier": "medium"}, {"tier": "large"}]
        assert _refs_present(tmp_path, ds) is False

    def test_refs_present_empty_dir(self, tmp_path: Path):
        ds = [{"tier": "tiny"}, {"tier": "medium"}, {"tier": "large"}]
        for t in ("tiny", "medium", "large"):
            (tmp_path / "reference_outputs" / t).mkdir(parents=True)
        # empty dirs -> not present
        assert _refs_present(tmp_path, ds) is False

    def test_has_round_dir_no_artifacts(self, tmp_path: Path):
        assert _has_round_dir(tmp_path) is False

    def test_verify_data_rows_missing(self, tmp_path: Path):
        assert _verify_data_rows(tmp_path) == 0

    def test_verify_data_rows_counts(self, tmp_path: Path):
        (tmp_path / "verify.tsv").write_text("h\nr1\nr2\n")
        assert _verify_data_rows(tmp_path) == 2

    def test_verify_data_rows_header_only(self, tmp_path: Path):
        (tmp_path / "verify.tsv").write_text("h\n")
        assert _verify_data_rows(tmp_path) == 0


# --------------------------------------------------------------------------
# detect_phase — reflect via framework + latest_keep surfaced
# --------------------------------------------------------------------------
class TestDetectPhaseWithFramework:
    def setup_method(self):
        _build_lifted_from_index.cache_clear()

    def test_reflect_detected_through_framework(self, tmp_path: Path):
        ws = tmp_path / "ws"
        fw = ws / "autozyme-framework"
        task = ws / "test_refl"
        task.mkdir(parents=True)
        (task / "task.yaml").write_text("task: refl\n")
        fb = fw / "reflections" / "prompt_reflect_feedback"
        fb.mkdir(parents=True)
        (fb / "iteration_refl_note.md").write_text("note")
        out = detect_phase(task, framework_root=fw)
        assert out["reflect"]["iteration"] is True

    def test_latest_keep_surfaced(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: x\n")
        (tmp_path / "results.tsv").write_text(
            HEADER
            + "3\tabc\tlarge_c\t4.0\t40.0\t700\tkeep\t{}\tH\td\toptimize\t8\n"
        )
        out = detect_phase(tmp_path, framework_root=None)
        assert out["latest_keep"]["speedup_pct"] == pytest.approx(40.0)
        assert out["latest_keep"]["thread"] == 8

    def test_report_and_package_verify_flags(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: x\n")
        (tmp_path / "report.html").write_text("<html>")
        (tmp_path / "package_verify.tsv").write_text("h\n")
        out = detect_phase(tmp_path, framework_root=None)
        assert out["report_done"] is True
        assert out["package_verify_done"] is True

    def test_package_via_lifted_index(self, tmp_path: Path):
        # Patch dir named after upstream pkg; task linked only via marker.
        ws = tmp_path / "ws"
        fw = ws / "autozyme-framework"
        task = ws / "test_marker"
        task.mkdir(parents=True)
        (task / "task.yaml").write_text("task: test_marker\n")
        d = fw / "autozyme_py" / "src" / "autozyme" / "upstreampkg"
        d.mkdir(parents=True)
        (d / "__init__.py").write_text(
            '"""Lifted from autozyme task `test_marker`"""\n'
        )
        out = detect_phase(task, framework_root=fw)
        assert out["phases"]["package"]["done"] is True


# --------------------------------------------------------------------------
# _recent_activity / _detect_activity
# --------------------------------------------------------------------------
class TestRecentActivity:
    def test_old_activity_reported_inactive(self, tmp_path: Path):
        # A file touched well in the past, with a 0-minute threshold, lands in
        # the "old" branch: active=False, confidence="old".
        (tmp_path / "task.yaml").write_text("target_repo: x\n")
        (tmp_path / "results.tsv").write_text("h\n")
        os.utime(tmp_path / "results.tsv", (1_000_000, 1_000_000))
        out = detect_phase(tmp_path, framework_root=None, active_recent_minutes=1)
        act = out["active"]
        assert act["active"] is False
        assert act["confidence"] == "old"
        assert act["age_min"] > 1

    def test_no_activity_files(self, tmp_path: Path):
        # Bare task with no activity-candidate files -> status "-", source None.
        (tmp_path / "task.yaml").write_text("target_repo: x\n")
        out = detect_phase(tmp_path, framework_root=None, active_recent_minutes=15)
        act = out["active"]
        assert act["active"] is False
        assert act["status"] == "-"
        assert act["source"] is None

    def test_dispatch_overrides_mtime(self, tmp_path: Path):
        # When a dispatch state owns the task, _detect_activity returns the
        # dispatch dict, not the mtime fallback.
        task = tmp_path / "t"
        task.mkdir()
        (task / "task.yaml").write_text("target_repo: x\n")
        (task / "results.tsv").write_text("h\n")
        _write_state(task, {
            "master_pid": os.getpid(),
            "queue": [{
                "name": "t", "task_dir": str(task),
                "status": "running", "reflect_status": None,
            }],
        })
        out = detect_phase(task, framework_root=None, active_recent_minutes=60)
        assert out["active"]["source"] == "dispatch"


# --------------------------------------------------------------------------
# OSError guards — exercised by making a dir unreadable (skipped under root)
# --------------------------------------------------------------------------
@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses chmod perms")
class TestOSErrorGuards:
    def test_refs_present_unreadable_dir(self, tmp_path: Path):
        ds = [{"tier": "tiny"}, {"tier": "medium"}, {"tier": "large"}]
        d = tmp_path / "reference_outputs" / "tiny"
        d.mkdir(parents=True)
        (d / "x").write_text("a")
        for t in ("medium", "large"):
            (tmp_path / "reference_outputs" / t).mkdir(parents=True)
        os.chmod(d, 0o000)
        try:
            # iterdir() on the unreadable dir raises OSError -> not present.
            assert _refs_present(tmp_path, ds) is False
        finally:
            os.chmod(d, 0o755)

    def test_has_round_dir_unreadable(self, tmp_path: Path):
        arts = tmp_path / "artifacts"
        arts.mkdir()
        os.chmod(arts, 0o000)
        try:
            assert _has_round_dir(tmp_path) is False
        finally:
            os.chmod(arts, 0o755)

    def test_latest_keep_unreadable(self, tmp_path: Path):
        p = tmp_path / "results.tsv"
        p.write_text(HEADER)
        os.chmod(p, 0o000)
        try:
            # read_text raises OSError -> _latest_keep returns None.
            assert _latest_keep(p) is None
        finally:
            os.chmod(p, 0o644)

    def test_extract_lifted_from_unreadable(self, tmp_path: Path):
        f = tmp_path / "patch.R"
        f.write_text("# Lifted from autozyme task `test_x`\n")
        os.chmod(f, 0o000)
        try:
            assert _extract_lifted_from(f) is None
        finally:
            os.chmod(f, 0o644)


class TestDispatchImportFailure:
    def test_pid_alive_import_failure_treated_dead(self, tmp_path: Path, monkeypatch):
        # If zyme.dispatch.state can't import pid_alive, alive stays False and
        # a running task reports as stale.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "zyme.dispatch.state":
                raise ImportError("simulated")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        task = tmp_path / "t"
        task.mkdir()
        _write_state(task, {
            "master_pid": os.getpid(),
            "queue": [{
                "name": "t", "task_dir": str(task),
                "status": "running", "reflect_status": None,
            }],
        })
        out = _dispatch_activity(task)
        # pid_alive unavailable -> alive=False -> running becomes "stale".
        assert out["status"] == "stale"
        assert out["dispatch_alive"] is False
