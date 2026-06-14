"""Unit tests for zyme.commands.datasets — the /datasets/ migration planner.

Covers the pure size/humanize helpers, the MovePlan dataclass, both migration
planners (shared top-level + per-task data/), the single-move executor (real
filesystem moves + compat symlinks under tmp_path), the plan renderer, and the
cmd_datasets_migrate entrypoint (dry-run + execute) with the scan boundary
monkeypatched.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands import datasets as ds


# --------------------------------------------------------------------------
# size + humanize
# --------------------------------------------------------------------------

class TestPathSizeBytes:
    def test_file(self, tmp_path):
        f = tmp_path / "a.bin"
        f.write_bytes(b"x" * 100)
        assert ds._path_size_bytes(f) == 100

    def test_dir_recursive(self, tmp_path):
        d = tmp_path / "d"
        (d / "sub").mkdir(parents=True)
        (d / "a").write_bytes(b"x" * 10)
        (d / "sub" / "b").write_bytes(b"y" * 20)
        assert ds._path_size_bytes(d) == 30

    def test_symlink_is_zero(self, tmp_path):
        target = tmp_path / "t"
        target.write_bytes(b"x" * 50)
        link = tmp_path / "link"
        link.symlink_to(target)
        assert ds._path_size_bytes(link) == 0

    def test_missing(self, tmp_path):
        assert ds._path_size_bytes(tmp_path / "nope") == 0


class TestHumanize:
    @pytest.mark.parametrize("n,expect", [
        (0, "0"),
        (-5, "0"),
        (512, "512B"),
        (1024, "1.0K"),
        (1536, "1.5K"),
        (1024 * 1024, "1.0M"),
        (20 * 1024 * 1024, "20M"),
        (1024 ** 3, "1.0G"),
        (1024 ** 4, "1.0T"),
    ])
    def test_values(self, n, expect):
        assert ds._humanize(n) == expect


# --------------------------------------------------------------------------
# MovePlan
# --------------------------------------------------------------------------

class TestMovePlan:
    def test_construction(self, tmp_path):
        p = ds.MovePlan(tmp_path / "a", tmp_path / "b",
                        compat_link=tmp_path / "a", size_bytes=10,
                        kind="shared_file", note="hi")
        assert p.kind == "shared_file"
        assert p.note == "hi"
        assert p.size_bytes == 10


# --------------------------------------------------------------------------
# shared migration planner
# --------------------------------------------------------------------------

class TestPlanSharedMigration:
    def test_empty_when_no_root(self, tmp_path):
        assert ds.plan_shared_migration(tmp_path / "nope") == []

    def test_plans_moves_skipping_keep_names(self, tmp_path):
        root = tmp_path / "datasets"
        root.mkdir()
        (root / "README.md").write_text("keep me")  # in keep set
        (root / "single_cell").mkdir()  # keep set
        (root / "heart.h5ad").write_bytes(b"x" * 100)  # plan move
        (root / "atlas").mkdir()  # plan move (dir)
        plans = ds.plan_shared_migration(root)
        names = {p.src.name for p in plans}
        assert names == {"heart.h5ad", "atlas"}
        h = next(p for p in plans if p.src.name == "heart.h5ad")
        assert h.kind == "shared_file"
        assert h.dst == root / "shared" / "heart.h5ad"
        a = next(p for p in plans if p.src.name == "atlas")
        assert a.kind == "shared_dir"

    def test_skips_existing_symlink(self, tmp_path):
        root = tmp_path / "datasets"
        root.mkdir()
        target = tmp_path / "real_target"
        target.write_bytes(b"x")
        (root / "already_linked").symlink_to(target)
        assert ds.plan_shared_migration(root) == []


# --------------------------------------------------------------------------
# per-task migration planner
# --------------------------------------------------------------------------

class TestPlanTaskMigration:
    def test_plans_real_data_dir(self, tmp_path):
        root = tmp_path / "datasets"
        root.mkdir()
        task = tmp_path / "test_x"
        (task / "data").mkdir(parents=True)
        (task / "data" / "tier.rds").write_bytes(b"x" * 40)
        plans = ds.plan_task_migration([task], root)
        assert len(plans) == 1
        assert plans[0].kind == "task_data"
        assert plans[0].dst == root / "per_task" / "test_x"
        assert plans[0].note == ""

    def test_skips_symlinked_data(self, tmp_path):
        root = tmp_path / "datasets"
        root.mkdir()
        task = tmp_path / "test_x"
        task.mkdir()
        target = tmp_path / "elsewhere"
        target.mkdir()
        (task / "data").symlink_to(target)
        assert ds.plan_task_migration([task], root) == []

    def test_skips_missing_data(self, tmp_path):
        root = tmp_path / "datasets"
        root.mkdir()
        task = tmp_path / "test_x"
        task.mkdir()
        assert ds.plan_task_migration([task], root) == []

    def test_notes_destination_exists(self, tmp_path):
        root = tmp_path / "datasets"
        (root / "per_task" / "test_x").mkdir(parents=True)
        task = tmp_path / "test_x"
        (task / "data").mkdir(parents=True)
        (task / "data" / "f").write_text("x")
        plans = ds.plan_task_migration([task], root)
        assert "DESTINATION EXISTS" in plans[0].note


# --------------------------------------------------------------------------
# executor
# --------------------------------------------------------------------------

class TestExecuteMove:
    def test_moves_file_and_leaves_compat_symlink(self, tmp_path):
        src = tmp_path / "data" / "heart.h5ad"
        src.parent.mkdir()
        src.write_bytes(b"x" * 10)
        dst = tmp_path / "data" / "shared" / "heart.h5ad"
        plan = ds.MovePlan(src, dst, compat_link=src, size_bytes=10,
                           kind="shared_file")
        ok, msg = ds._execute_move(plan)
        assert ok is True
        assert dst.is_file()
        # compat symlink in place, pointing (relatively) at dst
        assert src.is_symlink()
        assert src.resolve() == dst.resolve()

    def test_moves_dir_with_compat_symlink(self, tmp_path):
        src = tmp_path / "d" / "atlas"
        (src / "inner").mkdir(parents=True)
        (src / "inner" / "f").write_text("x")
        dst = tmp_path / "d" / "shared" / "atlas"
        plan = ds.MovePlan(src, dst, compat_link=src, size_bytes=0,
                           kind="shared_dir")
        ok, _ = ds._execute_move(plan)
        assert ok is True
        assert (dst / "inner" / "f").read_text() == "x"
        assert src.is_symlink()

    def test_skip_when_dest_exists_note(self, tmp_path):
        plan = ds.MovePlan(tmp_path / "a", tmp_path / "b", compat_link=None,
                           size_bytes=0, kind="task_data",
                           note="DESTINATION EXISTS at /b")
        ok, msg = ds._execute_move(plan)
        assert ok is False
        assert "skip (dest exists)" in msg

    def test_skip_when_src_missing(self, tmp_path):
        plan = ds.MovePlan(tmp_path / "missing", tmp_path / "b",
                           compat_link=None, size_bytes=0, kind="shared_file")
        ok, msg = ds._execute_move(plan)
        assert ok is False
        assert "src missing" in msg

    def test_no_compat_link(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"x")
        dst = tmp_path / "out" / "src.bin"
        plan = ds.MovePlan(src, dst, compat_link=None, size_bytes=1,
                           kind="shared_file")
        ok, _ = ds._execute_move(plan)
        assert ok is True
        assert dst.is_file()
        assert not src.exists()

    def test_move_failure_reported(self, tmp_path, monkeypatch):
        src = tmp_path / "src.bin"
        src.write_bytes(b"x")
        dst = tmp_path / "out" / "src.bin"
        plan = ds.MovePlan(src, dst, compat_link=None, size_bytes=1,
                           kind="shared_file")
        import shutil

        def boom(*a, **k):
            raise OSError("disk full")
        monkeypatch.setattr(shutil, "move", boom)
        ok, msg = ds._execute_move(plan)
        assert ok is False
        assert "move FAILED" in msg

    def test_compat_target_still_exists_backs_off(self, tmp_path, monkeypatch):
        # After move, if something is still at compat_link, back off cleanly.
        src = tmp_path / "src.bin"
        src.write_bytes(b"x")
        dst = tmp_path / "out" / "src.bin"
        plan = ds.MovePlan(src, dst, compat_link=src, size_bytes=1,
                           kind="shared_file")
        import shutil
        real_move = shutil.move

        def fake_move(s, d):
            real_move(s, d)
            # re-create something at the old src location to simulate a race
            Path(s).write_text("racer")
        monkeypatch.setattr(shutil, "move", fake_move)
        ok, msg = ds._execute_move(plan)
        assert ok is False
        assert "compat symlink target exists" in msg


# --------------------------------------------------------------------------
# subparser registration
# --------------------------------------------------------------------------

class TestAddSubparser:
    def test_registers_migrate(self):
        import argparse
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        ds.add_datasets_subparser(sub)
        args = parser.parse_args(
            ["datasets", "migrate", "--phase", "tasks", "--execute"])
        assert args.func is ds.cmd_datasets_migrate
        assert args.phase == "tasks"
        assert args.execute is True

    def test_task_flag_nargs(self):
        import argparse
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        ds.add_datasets_subparser(sub)
        args = parser.parse_args(
            ["datasets", "migrate", "--task", "a", "b"])
        assert args.tasks == ["a", "b"]
        assert args.phase == "all"  # default


# --------------------------------------------------------------------------
# renderer
# --------------------------------------------------------------------------

class TestRenderPlans:
    def test_empty(self, tmp_path, capsys):
        ds._render_plans([], [], tmp_path / "datasets")
        out = capsys.readouterr().out
        assert "nothing to move" in out
        assert "Grand total" in out

    def test_with_plans(self, tmp_path, capsys):
        sp = [ds.MovePlan(tmp_path / "a.h5ad", tmp_path / "shared/a.h5ad",
                          compat_link=tmp_path / "a.h5ad", size_bytes=1024,
                          kind="shared_file")]
        tp_task = tmp_path / "test_x"
        (tp_task / "data").mkdir(parents=True)
        tp = [ds.MovePlan(tp_task / "data", tmp_path / "per_task/test_x",
                          compat_link=tp_task / "data", size_bytes=2048,
                          kind="task_data", note="DESTINATION EXISTS")]
        ds._render_plans(sp, tp, tmp_path / "datasets")
        out = capsys.readouterr().out
        assert "Phase 1" in out
        assert "Phase 2" in out
        assert "a.h5ad" in out
        assert "test_x" in out
        assert "note:" in out  # because the task plan has a note


# --------------------------------------------------------------------------
# cmd_datasets_migrate entrypoint
# --------------------------------------------------------------------------

class TestCmdDatasetsMigrate:
    @pytest.fixture
    def workspace(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        fw = ws / "autozyme-framework"
        fw.mkdir(parents=True)
        datasets_root = ws / "datasets"
        datasets_root.mkdir()
        (datasets_root / "heart.h5ad").write_bytes(b"x" * 10)
        task = ws / "test_a"
        (task / "data").mkdir(parents=True)
        (task / "data" / "tier.rds").write_bytes(b"x" * 20)
        monkeypatch.chdir(ws)
        monkeypatch.setattr(ds, "find_workspace_root", lambda cwd: ws)
        monkeypatch.setattr(ds, "find_framework_root", lambda cwd: fw)
        monkeypatch.setattr(ds, "find_tasks",
                            lambda roots, max_depth, framework_root: [task])
        return SimpleNamespace(ws=ws, fw=fw, datasets=datasets_root, task=task)

    def test_no_workspace_exits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ds, "find_workspace_root", lambda cwd: None)
        args = SimpleNamespace(phase="all", tasks=None, execute=False)
        with pytest.raises(SystemExit):
            ds.cmd_datasets_migrate(args)

    def test_no_datasets_dir_exits(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.chdir(ws)
        monkeypatch.setattr(ds, "find_workspace_root", lambda cwd: ws)
        monkeypatch.setattr(ds, "find_framework_root", lambda cwd: None)
        args = SimpleNamespace(phase="all", tasks=None, execute=False)
        with pytest.raises(SystemExit):
            ds.cmd_datasets_migrate(args)

    def test_dry_run(self, workspace, capsys):
        args = SimpleNamespace(phase="all", tasks=None, execute=False)
        ds.cmd_datasets_migrate(args)
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        # nothing moved
        assert (workspace.datasets / "heart.h5ad").is_file()

    def test_execute_moves_files(self, workspace, capsys):
        args = SimpleNamespace(phase="all", tasks=None, execute=True)
        ds.cmd_datasets_migrate(args)
        out = capsys.readouterr().out
        assert "Done" in out
        # shared file moved + compat symlink left
        assert (workspace.datasets / "shared" / "heart.h5ad").is_file()
        assert (workspace.datasets / "heart.h5ad").is_symlink()
        # per-task data moved + symlink
        assert (workspace.datasets / "per_task" / "test_a").is_dir()
        assert (workspace.task / "data").is_symlink()

    def test_task_filter_unknown_exits(self, workspace):
        args = SimpleNamespace(phase="all", tasks=["nonexistent"], execute=False)
        with pytest.raises(SystemExit):
            ds.cmd_datasets_migrate(args)

    def test_task_filter_skips_phase1(self, workspace, capsys):
        args = SimpleNamespace(phase="all", tasks=["test_a"], execute=False)
        ds.cmd_datasets_migrate(args)
        out = capsys.readouterr().out
        # Phase 1 has nothing to move (filter forces do_shared=False).
        assert "Phase 1" in out
        # The task is in the phase-2 plan.
        assert "test_a" in out
