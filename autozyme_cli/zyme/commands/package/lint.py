"""`zyme package lint` — static checks against CAVEATS-derived rules.

Scans patch files under ``autozyme_r/inst/patches/<name>/patch.R`` and
``autozyme_py/src/autozyme/<name>/__init__.py``, runs the rules in
``lint_rules.py``, and prints findings grouped by patch.

Exits 1 if any FAIL is reported. WARN findings print but don't change exit
code, so CI gates that want stricter behavior can grep for ``[WARN]``.

Resolution rules:
  - ``zyme package lint --patch <name>`` lints one patch in either language.
  - ``zyme package lint --all`` lints every patch in both languages.
  - Bare ``zyme package lint`` looks at cwd: a task directory shows the
    patch derived from task.yaml::target_function; under autozyme_r/ or
    autozyme_py/ it's equivalent to --all over that language.
"""
from __future__ import annotations

import sys
from pathlib import Path

from zyme.commands.package.lint_rules import (
    LintContext,
    LintFinding,
    PY_RULES,
    R_RULES,
)
from zyme.scan import find_framework_root
from zyme.utils import die


def _r_patch_files(framework_root: Path) -> list[tuple[str, Path]]:
    """List (patch_name, patch.R) under autozyme_r/inst/patches/."""
    root = framework_root / "autozyme_r" / "inst" / "patches"
    if not root.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        f = child / "patch.R"
        if f.is_file():
            out.append((child.name, f))
    return out


def _py_patch_files(framework_root: Path) -> list[tuple[str, Path]]:
    """List (patch_name, __init__.py) under autozyme_py/src/autozyme/<name>/.

    Excludes private/dunder names and the package's own core modules.
    """
    root = framework_root / "autozyme_py" / "src" / "autozyme"
    if not root.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        init = child / "__init__.py"
        if not init.is_file():
            continue
        # Skip if the file doesn't actually call register_patch — patches
        # always do; non-patch subpackages don't.
        text = init.read_text(encoding="utf-8", errors="replace")
        if "register_patch" not in text:
            continue
        out.append((child.name, init))
    return out


def _r_context(name: str, path: Path, framework_root: Path) -> LintContext:
    desc_path = framework_root / "autozyme_r" / "DESCRIPTION"
    ns_path = framework_root / "autozyme_r" / "NAMESPACE"
    return LintContext(
        patch_name=name,
        patch_file=path,
        patch_text=path.read_text(encoding="utf-8", errors="replace"),
        language="R",
        description_path=desc_path,
        namespace_path=ns_path,
        description_text=desc_path.read_text(encoding="utf-8", errors="replace") if desc_path.is_file() else "",
        namespace_text=ns_path.read_text(encoding="utf-8", errors="replace") if ns_path.is_file() else "",
    )


def _py_context(name: str, path: Path) -> LintContext:
    return LintContext(
        patch_name=name,
        patch_file=path,
        patch_text=path.read_text(encoding="utf-8", errors="replace"),
        language="py",
    )


def _resolve_targets(
    args, framework_root: Path
) -> list[tuple[LintContext, tuple]]:
    """Build (context, rule_tuple) pairs from CLI args.

    Without --all and without --patch we lint everything (treat as --all);
    leaving the command useful in pre-commit hooks where no args are passed.
    """
    name = getattr(args, "patch", None)
    do_all = bool(getattr(args, "all", False)) or name is None
    r_files = _r_patch_files(framework_root)
    py_files = _py_patch_files(framework_root)
    pairs: list[tuple[LintContext, tuple]] = []
    if name:
        matched_r = [(n, p) for n, p in r_files if n == name]
        matched_py = [(n, p) for n, p in py_files if n == name]
        if not matched_r and not matched_py:
            die(f"no patch named {name!r} found under autozyme_r or autozyme_py")
        for n, p in matched_r:
            pairs.append((_r_context(n, p, framework_root), R_RULES))
        for n, p in matched_py:
            pairs.append((_py_context(n, p), PY_RULES))
        return pairs
    if do_all:
        for n, p in r_files:
            pairs.append((_r_context(n, p, framework_root), R_RULES))
        for n, p in py_files:
            pairs.append((_py_context(n, p), PY_RULES))
    return pairs


def run_lint(args, framework_root: Path | None = None) -> tuple[int, list[LintFinding]]:
    """Run lint and return (exit_code, findings). Reused by `preflight` and
    by the `attest` gate so they can decide what to do with findings without
    re-implementing CLI output formatting.
    """
    fr = framework_root or find_framework_root(Path.cwd())
    if fr is None:
        die("not inside an autozyme-framework workspace; cd into the framework or pass --framework-root")
    pairs = _resolve_targets(args, fr)
    all_findings: list[LintFinding] = []
    for ctx, rules in pairs:
        for rule in rules:
            try:
                findings = rule(ctx)
            except Exception as e:  # pragma: no cover — rule bugs surface here
                findings = [LintFinding(
                    rule_id=rule.__name__,
                    severity="WARN",
                    file=ctx.patch_file,
                    line=1,
                    message=f"lint rule crashed: {e!r}",
                )]
            all_findings.extend(findings)
    exit_code = 1 if any(f.severity == "FAIL" for f in all_findings) else 0
    return exit_code, all_findings


def cmd_package_lint(args) -> int:
    exit_code, findings = run_lint(args)
    if not findings:
        print("[OK] lint clean")
        return 0
    by_patch: dict[Path, list[LintFinding]] = {}
    for f in findings:
        by_patch.setdefault(f.file, []).append(f)
    for file in sorted(by_patch.keys()):
        print(f"\n{file}")
        for f in by_patch[file]:
            print(f"  {f.severity:<4}  {f.rule_id:<22} (line {f.line}) {f.message}")
    n_fail = sum(1 for f in findings if f.severity == "FAIL")
    n_warn = sum(1 for f in findings if f.severity == "WARN")
    print(f"\n[{'FAIL' if exit_code else 'WARN'}] {n_fail} fail, {n_warn} warn")
    return exit_code
