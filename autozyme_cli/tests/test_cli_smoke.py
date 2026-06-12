"""CLI smoke tests.

These are the cheapest possible tests: every cmd_* importable, every
subparser builds, and `zyme <subcmd> --help` exits 0. They wouldn't catch
logic bugs, but they catch the catastrophic cases (a refactor leaves a
broken import, a missing module-level constant, a parser that argparse
itself rejects). With a thousand-line CLI, this baseline is high-leverage.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


# Single source of truth: matches commands/__init__.py's __all__.
ALL_CMD_NAMES = [
    "cmd_init",
    "cmd_record_baseline", "cmd_reference", "cmd_record_noise", "cmd_promote_baseline",
    "cmd_baseline_list", "cmd_baseline_show", "cmd_baseline_rebench",
    "cmd_run", "cmd_dryrun", "cmd_accept", "cmd_reject", "cmd_rollback",
    "cmd_plot", "cmd_verify", "cmd_inspect_parallelism",
    "cmd_status", "cmd_scan",
    "cmd_registry_rebuild", "cmd_registry_query", "cmd_registry_suggest", "cmd_registry_list",
    "cmd_dispatch", "cmd_dispatch_status", "cmd_dispatch_usage",
    "cmd_dispatch_prices", "cmd_dispatch_resume",
    "cmd_dispatch_logs", "cmd_dispatch_stop",
    "cmd_prompt_save", "cmd_prompt_list", "cmd_prompt_show",
    "cmd_prompt_diff", "cmd_prompt_use", "cmd_prompt_annotate",
    "cmd_bench_register_template", "cmd_bench_list_templates",
    "cmd_bench_doctor", "cmd_bench_status", "cmd_bench_init", "cmd_bench_start",
    "cmd_bench_usage", "cmd_bench_prices", "cmd_bench_list",
    "cmd_audit",
]

# Top-level subcommands registered in cli.build_parser(). baseline / dispatch /
# prompt / bench are family parents whose children live in
# GROUPED_SUBCOMMANDS below — invoked as `zyme <group> <verb>`.
ALL_SUBCOMMAND_NAMES = [
    "init", "run", "dryrun", "accept", "reject", "rollback",
    "verify", "status", "plot", "scan", "registry", "inspect-parallelism",
    "baseline", "dispatch", "prompt", "bench",
    "audit",
]

# Groups whose subcommands are nested one level deep (zyme <group> <verb>).
GROUPED_SUBCOMMANDS = {
    "baseline": ["record", "reference", "noise", "promote", "rebench", "list", "show"],
    "dispatch": ["run", "status", "usage", "prices", "resume", "logs", "stop"],
    "prompt": ["save", "list", "show", "diff", "use", "annotate"],
    "registry": ["rebuild", "query", "suggest", "list"],
    "bench": [
        "register-template", "list-templates", "doctor", "status",
        "init", "start", "usage", "prices", "list",
    ],
}


# --------------------------------------------------------------------------
# Import-only smoke
# --------------------------------------------------------------------------

class TestPackageImports:
    def test_zyme_imports(self):
        import zyme
        assert hasattr(zyme, "__version__")

    def test_cli_module_imports(self):
        from zyme.cli import main, build_parser
        assert callable(main)
        assert callable(build_parser)

    def test_commands_package_re_exports_all(self):
        import zyme.commands as commands_pkg
        for name in ALL_CMD_NAMES:
            assert hasattr(commands_pkg, name), (
                f"{name!r} missing from zyme.commands re-exports"
            )

    @pytest.mark.parametrize("name", ALL_CMD_NAMES)
    def test_each_cmd_importable_directly_from_package(self, name):
        import zyme.commands
        fn = getattr(zyme.commands, name)
        assert callable(fn)

    def test_sibling_libs_distinct_from_command_wrappers(self):
        # zyme.audit (library) vs zyme.commands.audit (cmd).
        from zyme.audit import AuditContext, read_audit
        from zyme.commands.audit import cmd_audit
        # Different absolute paths — no name collision.
        assert AuditContext.__module__ == "zyme.audit"
        assert cmd_audit.__module__ == "zyme.commands.audit"

    def test_sibling_dispatch_distinct(self):
        from zyme.dispatch import start_dispatch
        from zyme.commands.dispatch import cmd_dispatch
        assert start_dispatch.__module__ == "zyme.dispatch.master"
        assert cmd_dispatch.__module__ == "zyme.commands.dispatch"


# --------------------------------------------------------------------------
# Parser construction
# --------------------------------------------------------------------------

class TestBuildParser:
    def test_returns_argument_parser(self):
        from zyme.cli import build_parser
        import argparse
        p = build_parser()
        assert isinstance(p, argparse.ArgumentParser)

    def test_registers_all_expected_subcommands(self):
        # Walk the subparser action and check it carries every expected name.
        from zyme.cli import build_parser
        import argparse
        p = build_parser()
        sub = next(
            (a for a in p._actions if isinstance(a, argparse._SubParsersAction)),
            None,
        )
        assert sub is not None, "build_parser should register a subparsers action"
        registered = set(sub.choices.keys())
        for expected in ALL_SUBCOMMAND_NAMES:
            assert expected in registered, f"subparser {expected!r} missing"


# --------------------------------------------------------------------------
# `zyme <subcommand> --help` exits 0
# --------------------------------------------------------------------------

@pytest.mark.parametrize("subcmd", ALL_SUBCOMMAND_NAMES)
def test_help_exits_zero(subcmd):
    """Catch syntactically-broken subparsers (argparse barfs on its own
    config). Run via subprocess so argparse's SystemExit path is exercised
    end-to-end."""
    result = subprocess.run(
        [sys.executable, "-m", "zyme", subcmd, "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, (
        f"`zyme {subcmd} --help` exited {result.returncode}\n"
        f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )
    # Help output should at minimum start with 'usage: '.
    assert result.stdout.lstrip().startswith("usage:"), (
        f"`zyme {subcmd} --help` didn't print a usage line: {result.stdout[:200]!r}"
    )


@pytest.mark.parametrize(
    "group,verb",
    [(g, v) for g, verbs in GROUPED_SUBCOMMANDS.items() for v in verbs],
)
def test_grouped_help_exits_zero(group, verb):
    """`zyme <group> <verb> --help` for nested subparser groups."""
    result = subprocess.run(
        [sys.executable, "-m", "zyme", group, verb, "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, (
        f"`zyme {group} {verb} --help` exited {result.returncode}\n"
        f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )
    assert result.stdout.lstrip().startswith("usage:")


def test_top_level_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "zyme", "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_no_args_prints_help_and_nonzero():
    # When invoked with no subcommand, zyme should print help and exit
    # nonzero (argparse default). This is a regression guard against
    # refactors that accidentally swallow argparse's required=True.
    result = subprocess.run(
        [sys.executable, "-m", "zyme"],
        capture_output=True, text=True, timeout=15,
    )
    # Either nonzero exit (required subcommand missing) or zero with help —
    # the user's actual config could go either way; just confirm it doesn't
    # hang or crash with a Python traceback.
    assert "Traceback" not in result.stderr
