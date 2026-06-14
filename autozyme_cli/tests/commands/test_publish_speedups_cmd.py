"""Unit tests for zyme.commands.publish_speedups.

Covers the pure helpers: task-dir resolution across layouts, dest-shard path
derivation (R folder / R legacy / Py), per-platform partition of TSV text by
system_os, patch-name index + task filter resolution, PublishFilter construction
from args, the not-applicable-threads policy override, path shortening, the
write-mode combine dispatcher, and data-row counting. The cmd entrypoint is then
driven over a fake workspace with the parser/merge boundary monkeypatched.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands import publish_speedups as ps
from zyme.parsers.package_verify_tsv import PublishFilter, PublishFilterError


# --------------------------------------------------------------------------
# _find_task_dir
# --------------------------------------------------------------------------

class TestFindTaskDir:
    def test_in_category_dir(self, tmp_path):
        ws = tmp_path
        d = ws / "test_core_singlecell" / "test_x"
        d.mkdir(parents=True)
        assert ps._find_task_dir(ws, "test_x") == d

    def test_in_framework_optimized(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        fw = tmp_path / "fw"
        d = fw / "optimized_task" / "test_general_bio" / "test_y"
        d.mkdir(parents=True)
        assert ps._find_task_dir(ws, "test_y", framework=fw) == d

    def test_bare_under_workspace(self, tmp_path):
        ws = tmp_path
        d = ws / "test_z"
        d.mkdir()
        assert ps._find_task_dir(ws, "test_z") == d

    def test_not_found(self, tmp_path):
        assert ps._find_task_dir(tmp_path, "missing") is None


# --------------------------------------------------------------------------
# _dest_for_patch
# --------------------------------------------------------------------------

class TestDestForPatch:
    def test_py_combined(self, tmp_path):
        patch = tmp_path / "autozyme_py" / "src" / "autozyme" / "decontx" / "__init__.py"
        patch.parent.mkdir(parents=True)
        got = ps._dest_for_patch(tmp_path, patch)
        assert got == patch.parent / "speedups.tsv"

    def test_py_platform_shard(self, tmp_path):
        patch = tmp_path / "x" / "__init__.py"
        patch.parent.mkdir(parents=True)
        got = ps._dest_for_patch(tmp_path, patch, "mac")
        assert got.name == "speedups.mac.tsv"

    def test_r_folder_layout(self, tmp_path):
        patch = tmp_path / "autozyme_r" / "inst" / "patches" / "decontx" / "patch.R"
        patch.parent.mkdir(parents=True)
        got = ps._dest_for_patch(tmp_path, patch, "win")
        assert got == patch.parent / "speedups.win.tsv"

    def test_r_legacy_single_file(self, tmp_path):
        fw = tmp_path
        patch = fw / "autozyme_r" / "inst" / "patches" / "decontx.R"
        patch.parent.mkdir(parents=True)
        got = ps._dest_for_patch(fw, patch, "mac")
        assert got == fw / "autozyme_r" / "inst" / "speedups" / "decontx.mac.tsv"

    def test_r_legacy_combined(self, tmp_path):
        fw = tmp_path
        patch = fw / "autozyme_r" / "inst" / "patches" / "decontx.R"
        patch.parent.mkdir(parents=True)
        got = ps._dest_for_patch(fw, patch)
        assert got.name == "decontx.tsv"


# --------------------------------------------------------------------------
# _partition_by_platform
# --------------------------------------------------------------------------

class TestPartitionByPlatform:
    HEADER = "tier\tthread\tsystem_os\tspeedup"

    def test_splits_into_mac_win_other(self):
        content = (
            f"{self.HEADER}\n"
            "tiny\t1\tmacOS-14\t2.0\n"
            "tiny\t1\tWindows-11\t1.5\n"
            "tiny\t1\tLinux-6\t3.0\n"
            "tiny\t1\tDarwin\t2.1\n"
        )
        out = ps._partition_by_platform(content)
        assert set(out) == {"mac", "win", "other"}
        # mac bucket gets macOS + Darwin
        assert out["mac"].count("\n") - 1 == 2  # header + 2 rows -> 2 data lines
        assert "Windows" in out["win"]
        assert "Linux" in out["other"]

    def test_empty(self):
        assert ps._partition_by_platform("") == {}

    def test_no_system_os_col_all_other(self):
        content = "tier\tspeedup\ntiny\t2.0\n"
        out = ps._partition_by_platform(content)
        assert set(out) == {"other"}

    def test_skips_blank_lines(self):
        content = f"{self.HEADER}\n\n  \ntiny\t1\tapple silicon\t2.0\n"
        out = ps._partition_by_platform(content)
        assert "mac" in out


# --------------------------------------------------------------------------
# patch-name index + task filter
# --------------------------------------------------------------------------

class TestBuildPatchNameIndex:
    def test_r_folder_uses_parent(self):
        idx = {"test_decontx": "/a/inst/patches/decontx/patch.R"}
        assert ps._build_patch_name_index(idx) == {"decontx": "test_decontx"}

    def test_r_legacy_uses_stem(self):
        idx = {"test_decontx": "/a/inst/patches/decontx.R"}
        assert ps._build_patch_name_index(idx) == {"decontx": "test_decontx"}

    def test_py_uses_parent(self):
        idx = {"test_x": "/a/autozyme/scrublet/__init__.py"}
        assert ps._build_patch_name_index(idx) == {"scrublet": "test_x"}


class TestResolveTaskFilter:
    def test_no_names_selects_all(self):
        idx = {"test_a": "/p/a/patch.R", "test_b": "/p/b/patch.R"}
        selected, unknown = ps._resolve_task_filter([], idx)
        assert selected == {"test_a", "test_b"}
        assert unknown == []

    def test_by_task_dir_name(self):
        idx = {"test_a": "/p/a/patch.R"}
        selected, unknown = ps._resolve_task_filter(["test_a"], idx)
        assert selected == {"test_a"}
        assert unknown == []

    def test_by_patch_name(self):
        idx = {"test_a": "/p/decontx/patch.R"}
        selected, unknown = ps._resolve_task_filter(["decontx"], idx)
        assert selected == {"test_a"}

    def test_unknown_collected(self):
        idx = {"test_a": "/p/a/patch.R"}
        selected, unknown = ps._resolve_task_filter(["test_a", "nope"], idx)
        assert selected == {"test_a"}
        assert unknown == ["nope"]

    def test_blank_names_ignored(self):
        idx = {"test_a": "/p/a/patch.R"}
        selected, unknown = ps._resolve_task_filter(["", "  "], idx)
        assert selected == set()
        assert unknown == []


# --------------------------------------------------------------------------
# PublishFilter from args
# --------------------------------------------------------------------------

class TestPublishFilterFromArgs:
    def test_defaults_full(self):
        args = SimpleNamespace(select="full")
        flt = ps._publish_filter_from_args(args)
        assert flt.select == "full"
        assert flt.tiers is None
        assert flt.tail is None

    def test_tiers_parsed(self):
        args = SimpleNamespace(select="full", tiers="tiny, medium ,")
        flt = ps._publish_filter_from_args(args)
        assert flt.tiers == ("tiny", "medium")

    def test_tail_default_for_tail_select(self):
        args = SimpleNamespace(select="tail")
        flt = ps._publish_filter_from_args(args)
        assert flt.tail == 5

    def test_explicit_tail(self):
        args = SimpleNamespace(select="tail", tail=3)
        flt = ps._publish_filter_from_args(args)
        assert flt.tail == 3

    def test_empty_tiers_dies(self):
        args = SimpleNamespace(select="full", tiers=" , ,")
        with pytest.raises(SystemExit):
            ps._publish_filter_from_args(args)

    def test_gates_passthrough(self):
        args = SimpleNamespace(select="full", all_pass_only=True,
                               require_all_tiers=True, require_all_pass=True)
        flt = ps._publish_filter_from_args(args)
        assert flt.all_pass_only is True
        assert flt.require_all_tiers is True
        assert flt.require_all_pass is True


# --------------------------------------------------------------------------
# _filter_for_task policy
# --------------------------------------------------------------------------

class TestFilterForTask:
    def _flt(self):
        return PublishFilter(select="full")

    def test_passthrough_when_not_restricted(self, tmp_path, monkeypatch):
        (tmp_path / "task.yaml").write_text("threading: default\n")
        monkeypatch.setattr(ps, "parse_threading_mode", lambda p: "default")
        flt = self._flt()
        out = ps._filter_for_task(tmp_path, flt, allow_not_applicable=False)
        assert out is flt  # unchanged identity

    def test_caps_threads_when_not_applicable(self, tmp_path, monkeypatch):
        (tmp_path / "task.yaml").write_text("threading: not_applicable\n")
        monkeypatch.setattr(ps, "parse_threading_mode", lambda p: "not_applicable")
        out = ps._filter_for_task(tmp_path, self._flt(), allow_not_applicable=False)
        assert out.max_threads == 1

    def test_allow_override_keeps_filter(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ps, "parse_threading_mode", lambda p: "not_applicable")
        flt = self._flt()
        out = ps._filter_for_task(tmp_path, flt, allow_not_applicable=True)
        assert out is flt


# --------------------------------------------------------------------------
# _shorten + _count_data_rows + _combine_for_write
# --------------------------------------------------------------------------

class TestShorten:
    def test_relative(self, tmp_path):
        p = tmp_path / "a" / "b.tsv"
        assert ps._shorten(p, tmp_path) == str(Path("a") / "b.tsv")

    def test_not_relative(self, tmp_path):
        p = tmp_path / "x.tsv"
        other = tmp_path.parent / "elsewhere"
        assert ps._shorten(p, other) == str(p)


class TestCountDataRows:
    def test_empty(self):
        assert ps._count_data_rows("") == 0
        assert ps._count_data_rows("   ") == 0

    def test_header_only(self):
        assert ps._count_data_rows("a\tb\n") == 0

    def test_with_rows(self):
        assert ps._count_data_rows("a\tb\n1\t2\n3\t4\n") == 2


class TestCombineForWrite:
    HEADER = "tier\tspeedup"

    def test_overwrite(self):
        existing = f"{self.HEADER}\nx\t1\ny\t2\n"
        new = f"{self.HEADER}\nz\t3\n"
        final, summary, total = ps._combine_for_write(existing, new, 1, "overwrite")
        assert final == new
        assert "overwrite" in summary
        assert total == 1

    def test_append_dispatches(self):
        # The append path delegates to append_published_tsvs; with these minimal
        # headers it dedup-keys to nothing, but the summary still reports append.
        existing = f"{self.HEADER}\nx\t1\n"
        new = f"{self.HEADER}\ny\t2\n"
        final, summary, total = ps._combine_for_write(existing, new, 1, "append")
        assert "append" in summary

    def test_merge_dispatches(self):
        existing = f"{self.HEADER}\nx\t1\n"
        new = f"{self.HEADER}\ny\t2\n"
        final, summary, total = ps._combine_for_write(existing, new, 1, "merge")
        assert "merge" in summary

    def test_unknown_mode_raises(self):
        with pytest.raises(PublishFilterError):
            ps._combine_for_write("a\tb\n", "a\tb\n", 0, "garbage")


# --------------------------------------------------------------------------
# cmd_publish_speedups entrypoint
# --------------------------------------------------------------------------

class TestCmdPublishSpeedups:
    def test_no_workspace_exits(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(ps, "find_workspace_root", lambda cwd: None)
        args = SimpleNamespace(tasks=None, dry_run=True, select="full")
        with pytest.raises(SystemExit):
            ps.cmd_publish_speedups(args)

    def test_no_index_exits(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.chdir(ws)
        monkeypatch.setattr(ps, "find_workspace_root", lambda cwd: ws)
        monkeypatch.setattr(ps, "find_framework_root", lambda cwd: ws / "fw")
        monkeypatch.setattr(ps, "_build_lifted_from_index", lambda fw: {})
        args = SimpleNamespace(tasks=None, dry_run=True, select="full")
        with pytest.raises(SystemExit):
            ps.cmd_publish_speedups(args)

    def test_unknown_task_name_exits(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.chdir(ws)
        monkeypatch.setattr(ps, "find_workspace_root", lambda cwd: ws)
        monkeypatch.setattr(ps, "find_framework_root", lambda cwd: ws / "fw")
        monkeypatch.setattr(ps, "_build_lifted_from_index",
                            lambda fw: {"test_a": "/p/a/patch.R"})
        args = SimpleNamespace(tasks=["does_not_exist"], dry_run=True,
                               select="full")
        with pytest.raises(SystemExit):
            ps.cmd_publish_speedups(args)

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch, capsys):
        ws = tmp_path / "ws"
        fw = ws / "fw"
        # Patch file (R folder layout) the dest writes next to.
        patch = fw / "autozyme_r" / "inst" / "patches" / "decontx" / "patch.R"
        patch.parent.mkdir(parents=True)
        patch.write_text("# patch")
        # Task dir with package_verify.tsv (bare under workspace).
        task = ws / "test_decontx"
        task.mkdir(parents=True)
        (task / "task.yaml").write_text("threading: default\n")
        (task / "package_verify.tsv").write_text(
            "tier\tthread\tsystem_os\tspeedup\ntiny\t1\tmacOS\t2.0\n")

        monkeypatch.chdir(ws)
        monkeypatch.setattr(ps, "find_workspace_root", lambda cwd: ws)
        monkeypatch.setattr(ps, "find_framework_root", lambda cwd: fw)
        monkeypatch.setattr(ps, "_build_lifted_from_index",
                            lambda f: {"test_decontx": str(patch)})
        monkeypatch.setattr(ps, "parse_threading_mode", lambda p: "default")
        monkeypatch.setattr(
            ps, "prepare_publish_content",
            lambda src, flt: (
                "tier\tthread\tsystem_os\tspeedup\ntiny\t1\tmacOS\t2.0\n", 1, "all"))
        monkeypatch.setattr(ps, "prune_published_tsv_text",
                            lambda text, max_threads: (text, 0))

        args = SimpleNamespace(
            tasks=None, dry_run=True, select="full", tiers=None, tail=None,
            platform=None, all_pass_only=False, require_all_tiers=False,
            require_all_pass=False, allow_not_applicable_threads=False,
            write_mode="merge")
        ps.cmd_publish_speedups(args)
        out = capsys.readouterr().out
        assert "DRY-RUN" in out
        # the mac shard was NOT actually written
        assert not (patch.parent / "speedups.mac.tsv").exists()

    def test_skips_task_without_tsv(self, tmp_path, monkeypatch, capsys):
        ws = tmp_path / "ws"
        fw = ws / "fw"
        patch = fw / "autozyme_r" / "inst" / "patches" / "decontx" / "patch.R"
        patch.parent.mkdir(parents=True)
        patch.write_text("# patch")
        task = ws / "test_decontx"
        task.mkdir(parents=True)
        (task / "task.yaml").write_text("threading: default\n")
        # no package_verify.tsv

        monkeypatch.chdir(ws)
        monkeypatch.setattr(ps, "find_workspace_root", lambda cwd: ws)
        monkeypatch.setattr(ps, "find_framework_root", lambda cwd: fw)
        monkeypatch.setattr(ps, "_build_lifted_from_index",
                            lambda f: {"test_decontx": str(patch)})
        args = SimpleNamespace(
            tasks=None, dry_run=True, select="full", tiers=None, tail=None,
            platform=None, all_pass_only=False, require_all_tiers=False,
            require_all_pass=False, allow_not_applicable_threads=False,
            write_mode="merge")
        ps.cmd_publish_speedups(args)
        out = capsys.readouterr().out
        assert "No package_verify.tsv" in out

    def test_real_write_creates_shard(self, tmp_path, monkeypatch, capsys):
        ws = tmp_path / "ws"
        fw = ws / "fw"
        patch = fw / "autozyme_r" / "inst" / "patches" / "decontx" / "patch.R"
        patch.parent.mkdir(parents=True)
        patch.write_text("# patch")
        task = ws / "test_decontx"
        task.mkdir(parents=True)
        (task / "task.yaml").write_text("threading: default\n")
        (task / "package_verify.tsv").write_text(
            "tier\tthread\tsystem_os\tspeedup\ntiny\t1\tmacOS\t2.0\n")

        monkeypatch.chdir(ws)
        monkeypatch.setattr(ps, "find_workspace_root", lambda cwd: ws)
        monkeypatch.setattr(ps, "find_framework_root", lambda cwd: fw)
        monkeypatch.setattr(ps, "_build_lifted_from_index",
                            lambda f: {"test_decontx": str(patch)})
        monkeypatch.setattr(ps, "parse_threading_mode", lambda p: "default")
        new_content = "tier\tthread\tsystem_os\tspeedup\ntiny\t1\tmacOS\t2.0\n"
        monkeypatch.setattr(
            ps, "prepare_publish_content",
            lambda src, flt: (new_content, 1, "all"))
        monkeypatch.setattr(ps, "prune_published_tsv_text",
                            lambda text, max_threads: (text, 0))
        # _combine_for_write -> overwrite mode is simplest deterministic path.
        args = SimpleNamespace(
            tasks=None, dry_run=False, select="full", tiers=None, tail=None,
            platform=None, all_pass_only=False, require_all_tiers=False,
            require_all_pass=False, allow_not_applicable_threads=False,
            write_mode="overwrite")
        ps.cmd_publish_speedups(args)
        out = capsys.readouterr().out
        shard = patch.parent / "speedups.mac.tsv"
        assert shard.is_file()
        assert "macOS" in shard.read_text()
        assert "Wrote" in out

    def test_gate_skip_exits_nonzero(self, tmp_path, monkeypatch, capsys):
        ws = tmp_path / "ws"
        fw = ws / "fw"
        patch = fw / "autozyme_r" / "inst" / "patches" / "decontx" / "patch.R"
        patch.parent.mkdir(parents=True)
        patch.write_text("# patch")
        task = ws / "test_decontx"
        task.mkdir(parents=True)
        (task / "task.yaml").write_text("threading: default\n")
        (task / "package_verify.tsv").write_text(
            "tier\tthread\tsystem_os\tspeedup\ntiny\t1\tmacOS\t2.0\n")

        monkeypatch.chdir(ws)
        monkeypatch.setattr(ps, "find_workspace_root", lambda cwd: ws)
        monkeypatch.setattr(ps, "find_framework_root", lambda cwd: fw)
        monkeypatch.setattr(ps, "_build_lifted_from_index",
                            lambda f: {"test_decontx": str(patch)})
        monkeypatch.setattr(ps, "parse_threading_mode", lambda p: "default")

        def gate_fail(src, flt):
            raise PublishFilterError("require-all-tiers not met")
        monkeypatch.setattr(ps, "prepare_publish_content", gate_fail)

        args = SimpleNamespace(
            tasks=None, dry_run=False, select="full", tiers=None, tail=None,
            platform=None, all_pass_only=False, require_all_tiers=True,
            require_all_pass=False, allow_not_applicable_threads=False,
            write_mode="merge")
        with pytest.raises(SystemExit):
            ps.cmd_publish_speedups(args)
        out = capsys.readouterr().out
        assert "filter/gate" in out

    def test_skips_task_without_dir(self, tmp_path, monkeypatch, capsys):
        ws = tmp_path / "ws"
        fw = ws / "fw"
        fw.mkdir(parents=True)
        monkeypatch.chdir(ws)
        monkeypatch.setattr(ps, "find_workspace_root", lambda cwd: ws)
        monkeypatch.setattr(ps, "find_framework_root", lambda cwd: fw)
        monkeypatch.setattr(ps, "_build_lifted_from_index",
                            lambda f: {"test_ghost": "/p/ghost/patch.R"})
        args = SimpleNamespace(
            tasks=None, dry_run=True, select="full", tiers=None, tail=None,
            platform=None, all_pass_only=False, require_all_tiers=False,
            require_all_pass=False, allow_not_applicable_threads=False,
            write_mode="merge")
        ps.cmd_publish_speedups(args)
        out = capsys.readouterr().out
        assert "No matching task directory" in out
