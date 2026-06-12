"""Smoke test: every `zyme package <sub>` command must be reachable via --help.

If a future refactor breaks the wiring (forgot to register a subparser, missed
an import in commands/__init__.py), pytest catches it before agents do.
"""
from __future__ import annotations

import subprocess
import sys


SUBCOMMANDS = (
    "lint",
    "check-intercept",
    "check-versions",
    "preflight",
    "smoke-parity",
    "sync-manifests",
)


def _run_help(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "zyme", *args, "--help"],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_package_group_help():
    rc, out = _run_help(["package"])
    assert rc == 0, out
    for sub in SUBCOMMANDS:
        assert sub in out, f"{sub} missing from `zyme package --help`"


def test_each_package_subcommand_help():
    for sub in SUBCOMMANDS:
        rc, out = _run_help(["package", sub])
        assert rc == 0, f"`zyme package {sub} --help` returned {rc}:\n{out}"


def test_attest_has_no_preflight_flag():
    rc, out = _run_help(["attest"])
    assert rc == 0, out
    assert "--no-preflight" in out, "attest is missing --no-preflight"
