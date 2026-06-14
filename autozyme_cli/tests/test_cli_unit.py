"""In-process unit tests for zyme.cli — the parser + pure helpers layer.

The existing test_cli_smoke.py / test_cli_e2e.py drive the CLI as a subprocess
(import smoke, `--help` exits, real git repos). This file is the complementary
IN-PROCESS layer: build_parser() once, then assert argparse parses real argv
into the right Namespace (defaults, choices, store_true, type coercion), that
bad args raise SystemExit, and that the pure helpers behave:

  - _parse_token_budget (k/m/b suffixes, errors)
  - build_parser dispatch wiring (every subcommand sets args.func)
  - flag defaults / choices / type coercion across representative subcommands
  - bad choices / missing required args -> SystemExit
  - _GroupedHelpFormatter top-level grouping
  - _full_cmd_name (nested verb composition)
  - _find_git_root (walks up, stops at fs root)
  - _workspace_git_lock (non-locked passthrough, env-held passthrough,
    acquire/release on a real temp git root)

No subprocesses, no real agents: args.func is never invoked.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme import cli


@pytest.fixture(scope="module")
def parser():
    return cli.build_parser()


def _parse(parser, argv):
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# _parse_token_budget
# --------------------------------------------------------------------------

class TestParseTokenBudget:
    def test_plain_int(self):
        assert cli._parse_token_budget("5000") == 5000

    def test_underscore_and_comma_stripped(self):
        assert cli._parse_token_budget("1_000_000") == 1_000_000
        assert cli._parse_token_budget("1,500") == 1500

    def test_k_suffix(self):
        assert cli._parse_token_budget("10k") == 10_000

    def test_m_suffix(self):
        assert cli._parse_token_budget("2m") == 2_000_000

    def test_b_suffix(self):
        assert cli._parse_token_budget("1b") == 1_000_000_000

    def test_float_with_suffix(self):
        assert cli._parse_token_budget("1.5m") == 1_500_000

    def test_uppercase_suffix(self):
        assert cli._parse_token_budget("3K") == 3000

    def test_zero_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            cli._parse_token_budget("0")

    def test_negative_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            cli._parse_token_budget("-5")

    def test_garbage_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            cli._parse_token_budget("abc")


# --------------------------------------------------------------------------
# build_parser dispatch wiring
# --------------------------------------------------------------------------

class TestDispatchWiring:
    @pytest.mark.parametrize("argv,func_name", [
        (["init", "https://x/repo"], "cmd_init"),
        (["run", "vectorize"], "cmd_run"),
        (["dryrun"], "cmd_dryrun"),
        (["accept"], "cmd_accept"),
        (["reject"], "cmd_reject"),
        (["rollback"], "cmd_rollback"),
        (["iterate"], "cmd_iterate"),
        (["verify"], "cmd_verify"),
        (["status"], "cmd_status"),
        (["plot"], "cmd_plot"),
        (["scan"], "cmd_scan"),
        (["audit"], "cmd_audit"),
        (["cost"], "cmd_cost"),
        (["profile"], "cmd_profile"),
        (["report"], "cmd_report"),
        (["baseline", "reference", "--tier", "tiny"], "cmd_reference"),
        (["baseline", "show", "tiny"], "cmd_baseline_show"),
        (["baseline", "list"], "cmd_baseline_list"),
        (["dispatch", "run", "t", "--prompt", "p.md"], "cmd_dispatch"),
        (["dispatch", "status"], "cmd_dispatch_status"),
        (["prompt", "list"], "cmd_prompt_list"),
        (["registry", "list"], "cmd_registry_list"),
        (["bench", "list"], "cmd_bench_list"),
        (["package", "lint"], "cmd_package_lint"),
        (["validate", "iterate"], "cmd_validate_iterate"),
        (["attest"], "cmd_attest"),
        (["publish-speedups"], "cmd_publish_speedups"),
    ])
    def test_func_set(self, parser, argv, func_name):
        args = _parse(parser, argv)
        assert hasattr(args, "func")
        assert args.func.__name__ == func_name


# --------------------------------------------------------------------------
# defaults / choices / type coercion
# --------------------------------------------------------------------------

class TestDefaultsAndTypes:
    def test_init_defaults(self, parser):
        args = _parse(parser, ["init", "https://x/repo"])
        assert args.target_repo == "https://x/repo"
        assert args.target_function is None
        assert args.field == "Bio"
        assert args.no_clone is False
        assert args.language is None

    def test_init_optional_function_and_flags(self, parser):
        args = _parse(parser, [
            "init", "repo", "FindAllMarkers",
            "--field", "OtherField", "--language", "R", "--no-clone",
        ])
        assert args.target_function == "FindAllMarkers"
        assert args.field == "OtherField"
        assert args.language == "R"
        assert args.no_clone is True

    def test_init_language_bad_choice(self, parser):
        with pytest.raises(SystemExit):
            _parse(parser, ["init", "repo", "fn", "--language", "julia"])

    def test_run_defaults(self, parser):
        args = _parse(parser, ["run", "hypothesis"])
        assert args.hypothesis == "hypothesis"
        assert args.rerun is False
        assert args.phase == "optimize"
        assert args.n_reps is None
        assert args.thread is None

    def test_run_thread_type_coercion(self, parser):
        args = _parse(parser, ["run", "h", "--thread", "8", "--n", "3"])
        assert args.thread == 8 and isinstance(args.thread, int)
        assert args.n_reps == 3

    def test_run_phase_bad_choice(self, parser):
        with pytest.raises(SystemExit):
            _parse(parser, ["run", "h", "--phase", "warmup"])

    def test_run_rerun_store_true(self, parser):
        args = _parse(parser, ["run", "--rerun", "--yes"])
        assert args.rerun is True
        assert args.yes is True

    def test_accept_description_short_and_long(self, parser):
        assert _parse(parser, ["accept", "-m", "msg"]).description == "msg"
        assert _parse(parser, ["accept", "--description", "x"]).description == "x"
        assert _parse(parser, ["accept"]).description == ""

    def test_verify_defaults(self, parser):
        args = _parse(parser, ["verify"])
        assert args.threads == "1,4,8"
        assert args.reps == 1
        assert args.write_mode == "overwrite"
        assert args.output == "verify.tsv"

    def test_verify_write_mode_choices(self, parser):
        for mode in ("overwrite", "append", "topup"):
            assert _parse(parser, ["verify", "--write-mode", mode]).write_mode == mode
        with pytest.raises(SystemExit):
            _parse(parser, ["verify", "--write-mode", "merge"])

    def test_attest_variadic_task_dirs(self, parser):
        args = _parse(parser, ["attest", "t1", "t2", "t3"])
        assert args.task_dirs == ["t1", "t2", "t3"]
        assert args.reps == 2

    def test_attest_lang_choice(self, parser):
        assert _parse(parser, ["attest", "--lang", "py"]).lang == "py"
        assert _parse(parser, ["attest", "--lang", "R"]).lang == "R"
        with pytest.raises(SystemExit):
            _parse(parser, ["attest", "--lang", "julia"])

    def test_profile_backend_choices(self, parser):
        for b in ("full", "cpu", "mem", "native"):
            assert _parse(parser, ["profile", "--backend", b]).backend == b
        with pytest.raises(SystemExit):
            _parse(parser, ["profile", "--backend", "gpu"])

    def test_task_dir_parent_shared(self, parser):
        # task-scoped commands inherit --task-dir from the parent parser
        args = _parse(parser, ["status", "--task-dir", "/some/path"])
        assert args.task_dir == "/some/path"

    def test_status_defaults(self, parser):
        args = _parse(parser, ["status"])
        assert args.task_dir is None
        assert args.last == 5
        assert args.phase == "optimize"

    def test_publish_speedups_choices(self, parser):
        args = _parse(parser, ["publish-speedups", "--select", "latest-per-tier",
                               "--write-mode", "append"])
        assert args.select == "latest-per-tier"
        assert args.write_mode == "append"
        with pytest.raises(SystemExit):
            _parse(parser, ["publish-speedups", "--select", "bogus"])


# --------------------------------------------------------------------------
# required args / subcommand errors
# --------------------------------------------------------------------------

class TestRequiredAndErrors:
    def test_top_level_requires_subcommand(self, parser):
        with pytest.raises(SystemExit):
            _parse(parser, [])

    def test_unknown_subcommand(self, parser):
        with pytest.raises(SystemExit):
            _parse(parser, ["nonexistent-cmd"])

    def test_init_requires_target_repo(self, parser):
        with pytest.raises(SystemExit):
            _parse(parser, ["init"])

    def test_baseline_requires_subcommand(self, parser):
        with pytest.raises(SystemExit):
            _parse(parser, ["baseline"])

    def test_baseline_reference_requires_tier(self, parser):
        with pytest.raises(SystemExit):
            _parse(parser, ["baseline", "reference"])

    def test_dispatch_requires_subcommand(self, parser):
        with pytest.raises(SystemExit):
            _parse(parser, ["dispatch"])

    def test_dispatch_run_requires_prompt(self, parser):
        with pytest.raises(SystemExit):
            _parse(parser, ["dispatch", "run", "task_a"])

    def test_dispatch_resume_mutually_exclusive_required(self, parser):
        # --prompt / --message group is required=True
        with pytest.raises(SystemExit):
            _parse(parser, ["dispatch", "resume", "task"])
        # both at once -> error
        with pytest.raises(SystemExit):
            _parse(parser, ["dispatch", "resume", "task",
                            "--prompt", "p", "--message", "m"])
        # exactly one is fine
        args = _parse(parser, ["dispatch", "resume", "task", "--message", "hi"])
        assert args.message == "hi"

    def test_scan_coverage_platform_mutually_exclusive(self, parser):
        with pytest.raises(SystemExit):
            _parse(parser, ["scan", "--coverage", "--mac-only", "--win-only"])
        args = _parse(parser, ["scan", "--coverage", "--mac-only"])
        assert args.mac_only is True and args.win_only is False

    def test_prompt_save_requires_as(self, parser):
        with pytest.raises(SystemExit):
            _parse(parser, ["prompt", "save", "some/path.md"])
        args = _parse(parser, ["prompt", "save", "p.md", "--as", "v1"])
        assert args.name == "v1"

    def test_registry_query_requires_terms(self, parser):
        with pytest.raises(SystemExit):
            _parse(parser, ["registry", "query"])
        args = _parse(parser, ["registry", "query", "mgcv", "gam"])
        assert args.terms == ["mgcv", "gam"]

    def test_token_budget_type_in_parser(self, parser):
        args = _parse(parser, ["dispatch", "usage", "--token-budget", "5k"])
        assert args.token_budget == 5000
        with pytest.raises(SystemExit):
            _parse(parser, ["dispatch", "usage", "--token-budget", "0"])


# --------------------------------------------------------------------------
# append actions / labels
# --------------------------------------------------------------------------

class TestAppendActions:
    def test_prompt_save_labels_append(self, parser):
        args = _parse(parser, [
            "prompt", "save", "p.md", "--as", "v1",
            "--label", "a=1", "--label", "b=2",
        ])
        assert args.labels == ["a=1", "b=2"]

    def test_bench_doctor_only_append(self, parser):
        args = _parse(parser, ["bench", "doctor", "suite", "--only", "x", "--only", "y"])
        assert args.only == ["x", "y"]


# --------------------------------------------------------------------------
# _GroupedHelpFormatter
# --------------------------------------------------------------------------

class TestGroupedHelpFormatter:
    def test_top_level_help_grouped(self, parser):
        text = parser.format_help()
        # Section headers from _TOP_LEVEL_GROUPS appear in the rendered help.
        assert "Core loop:" in text
        assert "Measurement:" in text
        assert "Packaging:" in text
        # Representative commands are listed under their group.
        assert "init" in text
        assert "attest" in text

    def test_formatter_class_wired(self, parser):
        assert parser.formatter_class is cli._GroupedHelpFormatter

    def test_groups_cover_registered_commands(self, parser):
        # Every command listed in _TOP_LEVEL_GROUPS must be a real subparser
        # (else the grouped help would reference a phantom command).
        sub = next(a for a in parser._actions
                   if isinstance(a, argparse._SubParsersAction))
        registered = set(sub.choices.keys())
        for _, names in cli._TOP_LEVEL_GROUPS:
            for n in names:
                assert n in registered, f"{n} grouped but not registered"


# --------------------------------------------------------------------------
# _full_cmd_name
# --------------------------------------------------------------------------

class TestFullCmdName:
    def test_plain_command(self):
        args = SimpleNamespace(cmd="status")
        assert cli._full_cmd_name(args) == "status"

    def test_baseline_nested(self):
        args = SimpleNamespace(cmd="baseline", baseline_cmd="record")
        assert cli._full_cmd_name(args) == "baseline record"

    def test_dispatch_nested(self):
        args = SimpleNamespace(cmd="dispatch", dispatch_cmd="run")
        assert cli._full_cmd_name(args) == "dispatch run"

    def test_package_nested(self):
        args = SimpleNamespace(cmd="package", package_cmd="lint")
        assert cli._full_cmd_name(args) == "package lint"

    def test_no_cmd_attr(self):
        assert cli._full_cmd_name(SimpleNamespace()) is None

    def test_nested_none_falls_back(self):
        args = SimpleNamespace(cmd="status", baseline_cmd=None)
        assert cli._full_cmd_name(args) == "status"

    def test_via_real_parse(self, parser):
        args = _parse(parser, ["baseline", "reference", "--tier", "tiny"])
        assert cli._full_cmd_name(args) == "baseline reference"


# --------------------------------------------------------------------------
# _find_git_root
# --------------------------------------------------------------------------

class TestFindGitRoot:
    def test_finds_dot_git_at_start(self, tmp_path):
        (tmp_path / ".git").mkdir()
        assert cli._find_git_root(tmp_path) == tmp_path.resolve()

    def test_walks_up_to_parent(self, tmp_path):
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        assert cli._find_git_root(sub) == tmp_path.resolve()

    def test_dot_git_file_counts(self, tmp_path):
        # git worktrees use a .git *file*, not a dir; .exists() matches both.
        (tmp_path / ".git").write_text("gitdir: /elsewhere")
        assert cli._find_git_root(tmp_path) == tmp_path.resolve()

    def test_no_git_returns_none(self, tmp_path):
        # tmp_path is under /private/var/.../pytest-* — no .git up to fs root.
        sub = tmp_path / "deep" / "nest"
        sub.mkdir(parents=True)
        assert cli._find_git_root(sub) is None


# --------------------------------------------------------------------------
# _workspace_git_lock
# --------------------------------------------------------------------------

class TestWorkspaceGitLock:
    def test_non_locked_command_passthrough(self, monkeypatch):
        # 'status' is not in _GIT_LOCKED_COMMANDS -> no lock taken.
        args = SimpleNamespace(cmd="status")
        with cli._workspace_git_lock(args):
            pass  # must not create any lock file or raise

    def test_env_held_passthrough(self, tmp_path, monkeypatch):
        (tmp_path / ".git").mkdir()
        monkeypatch.setenv("ZYME_GIT_LOCK_HELD", "1")
        args = SimpleNamespace(cmd="run", task_dir=str(tmp_path))
        with cli._workspace_git_lock(args):
            pass
        assert not (tmp_path / ".zyme_git.lock").exists()

    def test_no_git_root_passthrough(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ZYME_GIT_LOCK_HELD", raising=False)
        sub = tmp_path / "nogit"
        sub.mkdir()
        args = SimpleNamespace(cmd="run", task_dir=str(sub))
        with cli._workspace_git_lock(args):
            pass

    def test_acquire_and_release(self, tmp_path, monkeypatch):
        (tmp_path / ".git").mkdir()
        monkeypatch.delenv("ZYME_GIT_LOCK_HELD", raising=False)
        args = SimpleNamespace(cmd="run", task_dir=str(tmp_path),
                               baseline_cmd=None, dispatch_cmd=None,
                               prompt_cmd=None, bench_cmd=None, package_cmd=None)
        lock = tmp_path / ".zyme_git.lock"
        with cli._workspace_git_lock(args):
            # Held inside the context.
            assert lock.exists()
            import json as _json
            payload = _json.loads(lock.read_text())
            assert payload["pid"] == os.getpid()
            assert payload["cmd"] == "run"
        # Released on exit.
        assert not lock.exists()

    def test_stale_foreign_lock_not_removed_if_fresh(self, tmp_path, monkeypatch):
        # A fresh foreign lock would block; we don't want the test to spin, so
        # only assert that a fresh lock with another pid is left intact when we
        # DON'T enter the lock (non-locked command). This keeps the test fast.
        (tmp_path / ".git").mkdir()
        lock = tmp_path / ".zyme_git.lock"
        lock.write_text('{"pid": 999999}')
        args = SimpleNamespace(cmd="status", task_dir=str(tmp_path))
        with cli._workspace_git_lock(args):
            pass
        # status doesn't lock -> foreign lock untouched.
        assert lock.read_text() == '{"pid": 999999}'

    def test_in_locked_set_constant(self):
        assert "run" in cli._GIT_LOCKED_COMMANDS
        assert "accept" in cli._GIT_LOCKED_COMMANDS
        assert "status" not in cli._GIT_LOCKED_COMMANDS


# --------------------------------------------------------------------------
# main() dispatch — exercised in-process with a stub func (no real command)
# --------------------------------------------------------------------------

class TestMainDispatch:
    def test_readonly_command_bypasses_audit(self, monkeypatch, tmp_path):
        # `audit` is in the read-only set -> main() calls func directly, no
        # AuditContext. We assert that by making func record it ran.
        called = {}

        def fake_func(args):
            called["ran"] = True
            called["cmd"] = args.cmd

        fake_args = SimpleNamespace(cmd="audit", func=fake_func, task_dir=str(tmp_path))
        monkeypatch.setattr(cli, "build_parser",
                            lambda: SimpleNamespace(parse_args=lambda: fake_args))
        monkeypatch.setattr(cli.sys, "argv", ["zyme", "audit"])
        cli.main()
        assert called == {"ran": True, "cmd": "audit"}

    def test_audited_command_runs_func(self, monkeypatch, tmp_path):
        # A non-read-only, non-git-locked command ('status') runs through the
        # AuditContext + git-lock wrappers. With no task.yaml, audit writes
        # nothing, the lock is a passthrough, and func still runs.
        called = {}

        def fake_func(args):
            called["ran"] = True

        fake_args = SimpleNamespace(cmd="status", func=fake_func,
                                    task_dir=str(tmp_path))
        monkeypatch.setattr(cli, "build_parser",
                            lambda: SimpleNamespace(parse_args=lambda: fake_args))
        monkeypatch.setattr(cli.sys, "argv", ["zyme", "status"])
        monkeypatch.delenv("CLAUDECODE", raising=False)
        cli.main()
        assert called == {"ran": True}
        # status is workspace-level here (no task.yaml) -> no audit log.
        assert not (tmp_path / ".zyme" / "audit.jsonl").exists()


# --------------------------------------------------------------------------
# _GroupedHelpFormatter — empty-group branch
# --------------------------------------------------------------------------

class TestFormatterEmptyGroup:
    def test_group_with_no_registered_commands_skipped(self, monkeypatch):
        # Inject a group whose commands are NOT registered; the formatter must
        # skip it (the `if not present: continue` branch) without error and
        # without printing the phantom header.
        original = cli._TOP_LEVEL_GROUPS
        patched = list(original) + [("Phantom", ["does_not_exist_cmd"])]
        monkeypatch.setattr(cli, "_TOP_LEVEL_GROUPS", patched)
        parser = cli.build_parser()
        text = parser.format_help()
        assert "Phantom:" not in text
        assert "does_not_exist_cmd" not in text
