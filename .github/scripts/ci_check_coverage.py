#!/usr/bin/env python3
"""Tier A meta-test: every registered py patch has a contract test.

Fails CI with a clear diff when a patch is added without a corresponding
test_<name>.py, or when a contract test exists for a patch that's been
removed. Catches the silent-regression class where someone ships a new
patch but forgets to write its per-API contract test.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Patches that register multiple sub-targets under one patch name and are
# intentionally split into multiple contract test files (one per sub-target).
KNOWN_SPLITS = {
    "scanpy": {
        "normalize_total", "log1p", "pca", "scale", "leiden",
        "rank_genes_groups", "highly_variable_genes",
        "regress_out",
    },
}

# Contract tests that don't map 1:1 to a patch (meta / cross-cutting).
META_TESTS = {"all_patches_activate"}


def main() -> int:
    import autozyme

    patches = set(autozyme.list_patches())

    repo_root = Path(__file__).resolve().parents[2]
    test_dir = repo_root / "autozyme_py" / "tests" / "contract"
    test_files = {
        p.stem.removeprefix("test_")
        for p in test_dir.glob("test_*.py")
    }

    expected = set()
    for p in patches:
        if p in KNOWN_SPLITS:
            expected.update(KNOWN_SPLITS[p])
        else:
            expected.add(p)

    missing = expected - test_files
    extra = test_files - expected - META_TESTS

    if missing:
        print(
            f"ERROR: patches without contract test: {sorted(missing)}",
            file=sys.stderr,
        )
        print(
            "  -> add autozyme_py/tests/contract/test_<name>.py for each.",
            file=sys.stderr,
        )
    if extra:
        print(
            f"ERROR: contract tests for unknown patches: {sorted(extra)}",
            file=sys.stderr,
        )
        print(
            "  -> remove the stale test, re-add the patch, or extend "
            "KNOWN_SPLITS / META_TESTS in .github/scripts/ci_check_coverage.py.",
            file=sys.stderr,
        )
    if missing or extra:
        return 1

    print(
        f"OK: {len(patches)} patches -> {len(test_files)} contract tests, "
        "fully accounted for."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
