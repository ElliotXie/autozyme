"""Deep coverage tests for zyme.prompts.sync_prompts AND
zyme.prompts.sync_prompts_to_tasks.

Existing tests/test_sync_prompts*.py only cover a handful of functions. This
file covers the rest of both modules:

sync_prompts:
  - command_like_context heuristics (prose markers, code/table, arrows)
  - in_fenced_code / file_matches_glob / clean_token / flag_from_token
  - all_subcommands / all_flags_for / nested_subcommands / flag_scope_for
  - extract_invocations / collect_per_file / cmd_signature / parallel_basenames
  - validate_invocation unknown-subcommand + shlex-failure paths
  - main() end-to-end over a tiny synthesized prompt tree (clean + drift +
    invalid + invariant) and --strict exit behavior

sync_prompts_to_tasks:
  - detect_flavor / maybe_task_dir / find_tasks dedup
  - diff_files / _extract_markdown_section / _merge_legacy_init_inputs
  - copy_prompt / sync_task (added + skipped) / display_path
  - main() dry-run + apply + --task filter
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme.cli import build_parser
from zyme.prompts import sync_prompts as sp
from zyme.prompts import sync_prompts_to_tasks as spt


def _parts():
    parser = build_parser()
    return parser, sp.all_subcommands(parser)


# --------------------------------------------------------------------------
# command_like_context
# --------------------------------------------------------------------------

class TestCommandLikeContext:
    def test_at_line_start(self):
        line = "zyme run --rerun"
        assert sp.command_like_context(line, 0) is True

    def test_after_dollar_prompt(self):
        line = "$ zyme run"
        idx = line.index("zyme")
        assert sp.command_like_context(line, idx) is True

    def test_in_backticks(self):
        line = "see `zyme run` here"
        idx = line.index("zyme")
        assert sp.command_like_context(line, idx) is True

    def test_prose_marker_excluded(self):
        line = "triggered by zyme reminder"
        idx = line.index("zyme")
        assert sp.command_like_context(line, idx) is False

    def test_uses_marker_excluded(self):
        line = "the pipeline uses zyme verify internally"
        idx = line.index("zyme")
        assert sp.command_like_context(line, idx) is False

    def test_mid_prose_excluded(self):
        line = "we then run zyme to do stuff"
        idx = line.index("zyme")
        # 'run ' is a prose marker -> excluded.
        assert sp.command_like_context(line, idx) is False

    def test_table_cell(self):
        line = "| zyme run | description |"
        idx = line.index("zyme")
        assert sp.command_like_context(line, idx) is True


class TestInFencedCode:
    def test_inside_fence(self):
        text = "intro\n```\nzyme run\n```\n"
        # line index 2 (zyme run) is inside the fence.
        assert sp.in_fenced_code(text, 2) is True

    def test_outside_fence(self):
        text = "intro\n```\nzyme run\n```\nafter\n"
        # line index 4 (after) is outside.
        assert sp.in_fenced_code(text, 4) is False


class TestFileMatchesGlob:
    def test_star_matches_all(self):
        assert sp.file_matches_glob("anything.md", "*") is True

    def test_pattern_match(self):
        assert sp.file_matches_glob("3_validate_scaling.md", "3_*_scaling.md") is True
        assert sp.file_matches_glob("2_iterate.md", "3_*_scaling.md") is False


class TestTokenHelpers:
    def test_clean_token(self):
        assert sp.clean_token("[--flag],") == "--flag"

    def test_flag_from_token(self):
        assert sp.flag_from_token("--tier=tiny") == "--tier"
        assert sp.flag_from_token("positional") is None
        assert sp.flag_from_token("[--oom]") == "--oom"


# --------------------------------------------------------------------------
# argparse introspection
# --------------------------------------------------------------------------

class TestArgparseIntrospection:
    def test_all_subcommands_nonempty(self):
        _parser, subs = _parts()
        assert "run" in subs
        assert "baseline" in subs

    def test_all_flags_for_run(self):
        _parser, subs = _parts()
        flags = sp.all_flags_for(subs["run"])
        assert all(f.startswith("--") for f in flags)

    def test_nested_subcommands_present_for_baseline(self):
        _parser, subs = _parts()
        nested = sp.nested_subcommands(subs["baseline"])
        assert "record" in nested

    def test_nested_subcommands_empty_for_leaf(self):
        _parser, subs = _parts()
        # `run` has no nested subcommands.
        assert sp.nested_subcommands(subs["run"]) == {}

    def test_flag_scope_for_nested(self):
        _parser, subs = _parts()
        flags, path = sp.flag_scope_for("baseline", ["record", "--tier", "tiny"], subs)
        assert path == "baseline record"
        assert "--tier" in flags


# --------------------------------------------------------------------------
# validate_invocation extra branches
# --------------------------------------------------------------------------

class TestValidateInvocationEdge:
    def test_unknown_subcommand(self):
        parser, subs = _parts()
        ok, reason = sp.validate_invocation("frobnicate", "", parser, subs)
        assert not ok
        assert "unknown subcommand" in reason

    def test_shlex_failure(self):
        parser, subs = _parts()
        # An unbalanced quote makes shlex.split raise -> reported.
        ok, reason = sp.validate_invocation("run", 'foo "unterminated', parser, subs)
        assert not ok
        assert "shlex parse failed" in reason

    def test_placeholder_substitution_ok(self):
        parser, subs = _parts()
        ok, _ = sp.validate_invocation("run", "--rerun --n <n>", parser, subs)
        assert ok


# --------------------------------------------------------------------------
# extract_invocations / cmd_signature / parallel_basenames / collect_per_file
# --------------------------------------------------------------------------

class TestExtractInvocations:
    def test_extracts_command_lines_only(self, tmp_path: Path):
        md = tmp_path / "x.md"
        md.write_text(
            "Run `zyme run --rerun` to re-measure.\n"
            "This is triggered by zyme reminder (prose, excluded).\n"
        )
        out = sp.extract_invocations(md)
        cmds = [c for _, c, _, _ in out]
        assert "run" in cmds
        assert "reminder" not in cmds


class TestCmdSignature:
    def test_groups_flag_sets(self):
        invs = [
            (1, "run", "--rerun --n 1", "raw"),
            (2, "run", "--rerun", "raw"),
            (3, "verify", "--tiers small", "raw"),
        ]
        sig = sp.cmd_signature(invs)
        assert "run" in sig and "verify" in sig
        # run has two distinct flag-sets.
        assert len(sig["run"]) == 2

    def test_cmd_filter(self):
        invs = [(1, "run", "--rerun", "raw"), (2, "verify", "--tiers x", "raw")]
        sig = sp.cmd_signature(invs, cmd_filter="run")
        assert set(sig.keys()) == {"run"}


class TestParallelBasenames:
    def test_pairs_by_numeric_prefix(self):
        bio = {"1_init.md": [], "3_validate_scaling.md": []}
        other = {"1_init.md": [], "3_expand_scaling.md": []}
        pairs = sp.parallel_basenames(bio, other)
        prefixes = {b.split("_")[0] for b, _ in pairs}
        assert "1" in prefixes and "3" in prefixes


class TestCollectPerFile:
    def test_missing_field_dir(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(sp, "PROMPTS_ROOT", tmp_path)
        assert sp.collect_per_file("NoSuchField") == {}

    def test_collects_md_files(self, tmp_path: Path, monkeypatch):
        (tmp_path / "Bio").mkdir()
        (tmp_path / "Bio" / "1_init.md").write_text("`zyme run`\n")
        monkeypatch.setattr(sp, "PROMPTS_ROOT", tmp_path)
        out = sp.collect_per_file("Bio")
        assert "1_init.md" in out


# --------------------------------------------------------------------------
# main() end-to-end over a synthetic prompt tree
# --------------------------------------------------------------------------

def _build_prompt_tree(root: Path, *, bad_flag: bool = False,
                       drift: bool = False, invariant: bool = False) -> None:
    bio = root / "Bio"
    other = root / "OtherField"
    bio.mkdir(parents=True)
    other.mkdir(parents=True)
    bio_init = "# Bio init\n\nRun `zyme run --rerun --n 1` to measure.\n"
    other_init = "# OtherField init\n\nRun `zyme run --rerun --n 1` to measure.\n"
    if bad_flag:
        bio_init += "Also `zyme run --totally-bogus-flag`.\n"
    if drift:
        # Bio uses --tiers; OtherField uses --threads for the same subcommand.
        other_init += "Then `zyme run --rerun` alone.\n"
    if invariant:
        # A scaling-prompt file with --rerun but no --n triggers the invariant.
        (bio / "3_validate_scaling.md").write_text(
            "Run `zyme run --rerun` at scale.\n"
        )
    (bio / "1_init.md").write_text(bio_init)
    (other / "1_init.md").write_text(other_init)


class TestMain:
    def test_clean_tree_exit_zero(self, tmp_path: Path, monkeypatch, capsys):
        _build_prompt_tree(tmp_path)
        monkeypatch.setattr(sp, "PROMPTS_ROOT", tmp_path)
        import sys
        monkeypatch.setattr(sys, "argv", ["sync_prompts.py"])
        sp.main()
        out = capsys.readouterr().out
        assert "Section 1" in out
        assert "RESULT" in out

    def test_invalid_command_reported(self, tmp_path: Path, monkeypatch, capsys):
        _build_prompt_tree(tmp_path, bad_flag=True)
        monkeypatch.setattr(sp, "PROMPTS_ROOT", tmp_path)
        import sys
        monkeypatch.setattr(sys, "argv", ["sync_prompts.py"])
        sp.main()
        out = capsys.readouterr().out
        assert "invalid command" in out or "totally-bogus-flag" in out

    def test_strict_exits_nonzero_on_issue(self, tmp_path: Path, monkeypatch):
        _build_prompt_tree(tmp_path, bad_flag=True)
        monkeypatch.setattr(sp, "PROMPTS_ROOT", tmp_path)
        import sys
        monkeypatch.setattr(sys, "argv", ["sync_prompts.py", "--strict"])
        with pytest.raises(SystemExit) as exc:
            sp.main()
        assert exc.value.code == 1

    def test_invariant_violation_reported(self, tmp_path: Path, monkeypatch, capsys):
        _build_prompt_tree(tmp_path, invariant=True)
        monkeypatch.setattr(sp, "PROMPTS_ROOT", tmp_path)
        import sys
        monkeypatch.setattr(sys, "argv", ["sync_prompts.py"])
        sp.main()
        out = capsys.readouterr().out
        assert "Section 3" in out
        # The scaling prompt with --rerun and no --n should hit the invariant.
        assert "invariant" in out.lower()

    def test_cmd_filter_runs(self, tmp_path: Path, monkeypatch, capsys):
        _build_prompt_tree(tmp_path)
        monkeypatch.setattr(sp, "PROMPTS_ROOT", tmp_path)
        import sys
        monkeypatch.setattr(sys, "argv", ["sync_prompts.py", "--cmd", "run"])
        sp.main()
        assert "Section 2" in capsys.readouterr().out


# ==========================================================================
# zyme.prompts.sync_prompts_to_tasks
# ==========================================================================


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# detect_flavor / maybe_task_dir
# --------------------------------------------------------------------------

class TestDetectFlavor:
    def test_otherfield_by_init_header(self, tmp_path: Path):
        task = tmp_path / "test_x"
        _write(task / "prompts" / "1_init.md",
               "# autozyme - init prompt (OtherField)\nYou are an expert "
               "performance engineer.\n")
        assert spt.detect_flavor(task) == "OtherField"

    def test_bio_by_init_header(self, tmp_path: Path):
        task = tmp_path / "test_y"
        _write(task / "prompts" / "1_init.md",
               "You are a genius computational biologist working on single-cell.\n")
        assert spt.detect_flavor(task) == "Bio"

    def test_bio_by_phase3_fallback(self, tmp_path: Path):
        task = tmp_path / "test_z"
        _write(task / "prompts" / "1_init.md", "unrecognized header\n")
        _write(task / "prompts" / "3_validate_scaling.md", "scale\n")
        assert spt.detect_flavor(task) == "Bio"

    def test_otherfield_by_phase3_fallback(self, tmp_path: Path):
        task = tmp_path / "test_w"
        _write(task / "prompts" / "1_init.md", "unrecognized header\n")
        _write(task / "prompts" / "3_expand_scaling.md", "scale\n")
        assert spt.detect_flavor(task) == "OtherField"

    def test_none_when_unrecognized(self, tmp_path: Path):
        task = tmp_path / "test_q"
        _write(task / "prompts" / "1_init.md", "totally generic prompt\n")
        assert spt.detect_flavor(task) is None


class TestMaybeTaskDir:
    def test_non_dir(self, tmp_path: Path):
        f = tmp_path / "test_file.txt"
        f.write_text("x")
        assert spt.maybe_task_dir(f) is None

    def test_wrong_prefix(self, tmp_path: Path):
        d = tmp_path / "notatask"
        (d / "prompts").mkdir(parents=True)
        assert spt.maybe_task_dir(d) is None

    def test_no_prompts_dir(self, tmp_path: Path):
        d = tmp_path / "test_x"
        d.mkdir()
        assert spt.maybe_task_dir(d) is None

    def test_unrecognized_flavor_none(self, tmp_path: Path):
        task = tmp_path / "test_x"
        _write(task / "prompts" / "1_init.md", "generic\n")
        assert spt.maybe_task_dir(task) is None

    def test_valid_task(self, tmp_path: Path):
        task = tmp_path / "test_x"
        _write(task / "prompts" / "1_init.md",
               "You are a computational biologist.\n")
        out = spt.maybe_task_dir(task)
        assert out is not None and out[1] == "Bio"


# --------------------------------------------------------------------------
# find_tasks dedup (root + categorized layout)
# --------------------------------------------------------------------------

class TestFindTasksDedup:
    def test_dedupes_across_layouts(self, tmp_path: Path):
        ws = tmp_path / "ws"
        # Same task discoverable both at root and via category dir would be
        # one physical dir; here we make a root task + a categorized task.
        _write(ws / "test_bio" / "prompts" / "1_init.md",
               "You are a computational biologist.\n")
        _write(ws / "test_general_bio" / "test_inner" / "prompts" / "1_init.md",
               "You are a computational biologist.\n")
        found = {t.name: f for t, f in spt.find_tasks(ws)}
        assert found == {"test_bio": "Bio", "test_inner": "Bio"}


# --------------------------------------------------------------------------
# diff_files / _extract_markdown_section / _merge_legacy_init_inputs
# --------------------------------------------------------------------------

class TestDiffFiles:
    def test_missing_counts_as_diff(self, tmp_path: Path):
        a = tmp_path / "a.md"
        a.write_text("x")
        assert spt.diff_files(a, tmp_path / "absent.md") is True

    def test_same_content_no_diff(self, tmp_path: Path):
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("same")
        b.write_text("same")
        assert spt.diff_files(a, b) is False

    def test_different_content(self, tmp_path: Path):
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("one")
        b.write_text("two")
        assert spt.diff_files(a, b) is True


class TestExtractMarkdownSection:
    def test_returns_section(self):
        text = ("# Title\n\n## Your inputs for this run\n\nfoo\n\n## Scope\n\nbar\n")
        out = spt._extract_markdown_section(text, "## Your inputs for this run")
        assert out is not None
        assert "foo" in out
        assert "Scope" not in out

    def test_returns_none_when_absent(self):
        assert spt._extract_markdown_section("# T\n", "## Nope") is None

    def test_section_runs_to_eof(self):
        text = "## Your inputs for this run\n\nlast block\n"
        out = spt._extract_markdown_section(text, "## Your inputs for this run")
        assert "last block" in out


class TestMergeLegacyInitInputs:
    def test_no_inputs_returns_framework(self):
        out = spt._merge_legacy_init_inputs("no inputs here\n", "framework text\n")
        assert out == "framework text\n"

    def test_replaces_existing_inputs_section(self):
        task = "## Your inputs for this run\n\nTASK SPECIFIC\n"
        fw = "# init\n\n## Your inputs for this run\n\nPLACEHOLDER\n\n## Scope\n\ns\n"
        out = spt._merge_legacy_init_inputs(task, fw)
        assert "TASK SPECIFIC" in out
        assert "PLACEHOLDER" not in out

    def test_inserts_before_scope_when_no_inputs_in_framework(self):
        task = "## Your inputs for this run\n\nTASK SPECIFIC\n"
        fw = "# init\n\n## Role\n\nr\n\n## Scope\n\ns\n"
        out = spt._merge_legacy_init_inputs(task, fw)
        assert "TASK SPECIFIC" in out
        # Inserted just before the Scope heading.
        assert out.index("TASK SPECIFIC") < out.index("## Scope")

    def test_appends_when_no_scope(self):
        task = "## Your inputs for this run\n\nTASK SPECIFIC\n"
        fw = "# init\n\n## Role\n\nr\n"
        out = spt._merge_legacy_init_inputs(task, fw)
        assert out.rstrip().endswith("TASK SPECIFIC")


# --------------------------------------------------------------------------
# copy_prompt + sync_task (added files)
# --------------------------------------------------------------------------

class TestCopyPrompt:
    def test_identical_returns_false(self, tmp_path: Path):
        fw = tmp_path / "fw.md"
        task = tmp_path / "task.md"
        fw.write_text("same\n")
        task.write_text("same\n")
        assert spt.copy_prompt(fw, task, dry_run=False) is False

    def test_copies_when_different(self, tmp_path: Path):
        fw = tmp_path / "2_iterate.md"
        task = tmp_path / "2_iterate_task.md"
        fw.write_text("new\n")
        task.write_text("old\n")
        assert spt.copy_prompt(fw, task, dry_run=False) is True
        assert task.read_text() == "new\n"

    def test_dry_run_does_not_write(self, tmp_path: Path):
        fw = tmp_path / "2_iterate.md"
        task = tmp_path / "task.md"
        fw.write_text("new\n")
        task.write_text("old\n")
        assert spt.copy_prompt(fw, task, dry_run=True) is True
        assert task.read_text() == "old\n"  # unchanged

    def test_init_merge_unchanged_returns_false(self, tmp_path: Path):
        fw = tmp_path / "1_init.md"
        task = tmp_path / "1_init_task.md"
        fw.write_text("# init\n\n## Scope\n\ns\n")
        # task identical to framework (no inputs to preserve) -> no change.
        task.write_text("# init\n\n## Scope\n\ns\n")
        assert spt.copy_prompt(fw, task, dry_run=False) is False


class TestSyncTaskAdded:
    def test_adds_missing_framework_file(self, tmp_path: Path, monkeypatch):
        fw = tmp_path / "framework_prompts"
        _write(fw / "Bio" / "2_iterate.md", "iterate body\n")
        _write(fw / "Bio" / "2.5_iterate_memory.md", "memory body\n")
        task = tmp_path / "ws" / "test_bio"
        _write(task / "prompts" / "2_iterate.md", "iterate body\n")  # identical
        monkeypatch.setattr(spt, "FRAMEWORK_PROMPTS", fw)
        report = spt.sync_task(task, "Bio", dry_run=False)
        assert "2.5_iterate_memory.md" in report["added"]
        assert (task / "prompts" / "2.5_iterate_memory.md").exists()
        assert report["identical"] == ["2_iterate.md"]

    def test_skipped_no_framework(self, tmp_path: Path, monkeypatch):
        fw = tmp_path / "framework_prompts"
        _write(fw / "Bio" / "2_iterate.md", "iterate\n")
        task = tmp_path / "ws" / "test_bio"
        _write(task / "prompts" / "2_iterate.md", "iterate\n")
        _write(task / "prompts" / "task_only.md", "local-only prompt\n")
        monkeypatch.setattr(spt, "FRAMEWORK_PROMPTS", fw)
        report = spt.sync_task(task, "Bio", dry_run=False)
        assert "task_only.md" in report["skipped_no_framework"]


class TestDisplayPath:
    def test_relative(self, tmp_path: Path):
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        assert spt.display_path(sub, tmp_path) == str(Path("a") / "b")

    def test_unrelated_returns_str(self, tmp_path: Path):
        other = Path("/totally/elsewhere")
        assert spt.display_path(other, tmp_path) == str(other)


# --------------------------------------------------------------------------
# main() end-to-end
# --------------------------------------------------------------------------

class TestSyncToTasksMain:
    def _setup(self, tmp_path: Path, monkeypatch):
        fw = tmp_path / "framework_prompts"
        _write(fw / "Bio" / "1_init.md",
               "# init\n\n## Role\n\nrole\n\n## Scope\n\nscope\n")
        _write(fw / "Bio" / "2_iterate.md", "iterate v2\n")
        ws = tmp_path / "ws"
        _write(ws / "test_bio" / "prompts" / "1_init.md",
               "You are a computational biologist.\n\n## Role\n\nold\n")
        _write(ws / "test_bio" / "prompts" / "2_iterate.md", "iterate v1\n")
        monkeypatch.setattr(spt, "FRAMEWORK_PROMPTS", fw)
        return ws

    def test_dry_run(self, tmp_path: Path, monkeypatch, capsys):
        ws = self._setup(tmp_path, monkeypatch)
        import sys
        monkeypatch.setattr(sys, "argv",
                            ["sync.py", "--dry-run", "--workspace", str(ws)])
        spt.main()
        out = capsys.readouterr().out
        assert "DRY RUN" in out
        # 2_iterate.md differs -> reported but not written.
        assert (ws / "test_bio" / "prompts" / "2_iterate.md").read_text() == "iterate v1\n"

    def test_apply_writes(self, tmp_path: Path, monkeypatch, capsys):
        ws = self._setup(tmp_path, monkeypatch)
        import sys
        monkeypatch.setattr(sys, "argv", ["sync.py", "--workspace", str(ws)])
        spt.main()
        assert (ws / "test_bio" / "prompts" / "2_iterate.md").read_text() == "iterate v2\n"

    def test_task_filter_no_match_exits(self, tmp_path: Path, monkeypatch):
        ws = self._setup(tmp_path, monkeypatch)
        import sys
        monkeypatch.setattr(sys, "argv",
                            ["sync.py", "--workspace", str(ws), "--task", "test_ghost"])
        with pytest.raises(SystemExit):
            spt.main()

    def test_task_filter_by_name(self, tmp_path: Path, monkeypatch, capsys):
        ws = self._setup(tmp_path, monkeypatch)
        import sys
        monkeypatch.setattr(sys, "argv",
                            ["sync.py", "--workspace", str(ws), "--task", "test_bio"])
        spt.main()
        assert "test_bio" in capsys.readouterr().out
