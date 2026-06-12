"""`zyme package preflight` — one-button gate before `zyme attest`.

Runs the cheap, deterministic checks in order:

  1. lint          — CAVEATS-derived static checks across all patches
  2. portability   — refresh ``.zyme/portability_scan.json`` and require an
                     acceptable verdict
  3. smoke-parity  — one smoke iteration at the tiny tier; evaluate must pass

Stops at the first FAIL by default (``--continue`` keeps going so the agent
sees every problem in one round). Exits non-zero if any step fails.

This is what ``zyme attest`` runs implicitly before measurement (unless
``--no-preflight`` is passed), and what an agent should run manually while
iterating on a fresh patch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from zyme.commands.attest import _infer_patch_name
from zyme.commands.package.lint import cmd_package_lint
from zyme.commands.package.smoke_parity import cmd_package_smoke_parity
from zyme.scan import find_framework_root
from zyme.scan_portability import scan_task, save_portability_scan
from zyme.utils import die, task_dir_from_args


# Verdicts the portability scan can emit. CLEAN and MAC_BONUS_ONLY are fine
# to ship; NEEDS_REVIEW and UNSAFE block.
_ACCEPTABLE_VERDICTS = {"CLEAN", "MAC_BONUS_ONLY"}


def _step_lint(task_dir: Path, framework_root: Path | None) -> int:
    """Run lint scoped to the patch this task lifts. Falls back to --all when
    we can't infer the patch name (early in packaging, before register_patch
    is written). The lint is the same shape either way.
    """
    patch_name = _infer_patch_name(task_dir)
    args = SimpleNamespace(patch=patch_name, all=patch_name is None)
    print(f"\n=== preflight: lint ({'patch=' + patch_name if patch_name else '--all'}) ===")
    return cmd_package_lint(args)


def _step_portability(task_dir: Path, framework_root: Path | None) -> int:
    print("\n=== preflight: portability scan ===")
    res = scan_task(task_dir, framework_root=framework_root)
    save_portability_scan(task_dir, res)
    verdict = getattr(res, "verdict", "UNKNOWN")
    print(f"  verdict: {verdict}")
    hits = getattr(res, "hits", []) or []
    if hits:
        print(f"  {len(hits)} hit(s):")
        for h in hits[:10]:
            ref = getattr(h, "ref", None) or getattr(h, "location", "")
            label = getattr(h, "label", "") or getattr(h, "pattern", "")
            print(f"    - {ref}  {label}")
        if len(hits) > 10:
            print(f"    ... and {len(hits) - 10} more")
    if verdict not in _ACCEPTABLE_VERDICTS:
        print(f"\n[FAIL] portability verdict {verdict!r} is not shippable; "
              f"fix the hits or apply OS-specific branches before attest.")
        return 1
    print("[OK] portability acceptable")
    return 0


def _step_smoke_parity(task_dir: Path, patch_name: str | None) -> int:
    print("\n=== preflight: smoke-parity (tiny) ===")
    args = SimpleNamespace(
        task_dir=str(task_dir),
        patch=patch_name,
        tier="tiny",
        lang=None,
    )
    return cmd_package_smoke_parity(args)


def cmd_package_preflight(args) -> int:
    task_dir = task_dir_from_args(args)
    framework_root = find_framework_root(task_dir)
    cont = bool(getattr(args, "continue_on_fail", False))
    skip_parity = bool(getattr(args, "skip_parity", False))
    failures: list[str] = []

    rc = _step_lint(task_dir, framework_root)
    if rc != 0:
        failures.append("lint")
        if not cont:
            return _summarize(failures)

    rc = _step_portability(task_dir, framework_root)
    if rc != 0:
        failures.append("portability")
        if not cont:
            return _summarize(failures)

    if not skip_parity:
        patch_name = _infer_patch_name(task_dir)
        rc = _step_smoke_parity(task_dir, patch_name)
        if rc != 0:
            failures.append("smoke-parity")
            if not cont:
                return _summarize(failures)

    return _summarize(failures)


def _summarize(failures: list[str]) -> int:
    print("")
    if failures:
        print(f"[FAIL] preflight: {len(failures)} step(s) failed: {', '.join(failures)}")
        return 1
    print("[OK] preflight clean")
    return 0
