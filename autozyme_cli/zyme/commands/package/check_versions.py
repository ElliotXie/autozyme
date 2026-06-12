"""`zyme package check-versions` — flag patches whose tested_against has drifted.

Every patch declares ``tested_against = "<upstream_pkg> <X.Y.Z>"`` — the
upstream version it was lifted against. After upstream releases a new
version, the patch keeps working until the upstream API actually changes,
but the drift is invisible until something breaks at attest time.

This command walks all patches, parses their ``tested_against`` strings,
queries the installed upstream version, and prints a table of drift status.

Columns: ``patch | upstream | tested_against | installed | status``.
Statuses: ``ok`` (exact match), ``drift`` (different), ``missing`` (upstream
not installed), ``unknown`` (parse failed). Returns non-zero when any patch
reports drift or missing — call from CI to gate on upstream pin freshness.
"""
from __future__ import annotations

import ast
import importlib.metadata
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from zyme.scan import find_framework_root
from zyme.utils import die


@dataclass
class _VersionRow:
    patch: str
    language: str        # "py" or "R"
    upstream: str
    tested: str
    installed: str | None
    status: str          # "ok" | "drift" | "missing" | "unknown"


def _parse_tested_against(s: str) -> tuple[str, str] | None:
    """Extract (upstream_pkg, version) from a `tested_against` string.

    Convention: ``"<pkg> <X.Y.Z>"`` — the package name and version split by
    whitespace. We tolerate dotted package names (``scanpy.tools``) and
    version strings beginning with a digit (the version always starts with
    a digit; the package name never does).
    """
    s = s.strip()
    # Split on the first whitespace before a digit — that's the package
    # name / version boundary. Falls back to plain split for robustness.
    m = re.match(r"^([\w.\-]+)\s+([0-9][\w.\-+]*)\s*$", s)
    if m:
        return m.group(1), m.group(2)
    return None


def _scan_python(framework_root: Path) -> list[tuple[str, str, str]]:
    """Walk Python patches, return list of (patch_name, upstream, tested_version)."""
    root = framework_root / "autozyme_py" / "src" / "autozyme"
    out: list[tuple[str, str, str]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        init = child / "__init__.py"
        if not init.is_file():
            continue
        text = init.read_text(encoding="utf-8", errors="replace")
        if "register_patch" not in text or "tested_against" not in text:
            continue
        try:
            mod = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(mod):
            if not isinstance(node, ast.Call):
                continue
            fname = node.func.attr if isinstance(node.func, ast.Attribute) else (
                node.func.id if isinstance(node.func, ast.Name) else None
            )
            if fname != "register_patch":
                continue
            patch_name = None
            tested = None
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    patch_name = kw.value.value
                elif kw.arg == "tested_against" and isinstance(kw.value, ast.Constant):
                    tested = kw.value.value
            if not (patch_name and tested):
                continue
            parsed = _parse_tested_against(tested)
            if not parsed:
                continue
            upstream, version = parsed
            out.append((patch_name, upstream, version))
    return out


def _scan_r(framework_root: Path) -> list[tuple[str, str, str]]:
    """Walk R patches, return list of (patch_name, upstream, tested_version)."""
    root = framework_root / "autozyme_r" / "inst" / "patches"
    out: list[tuple[str, str, str]] = []
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        patch_file = child / "patch.R"
        if not patch_file.is_file():
            continue
        text = patch_file.read_text(encoding="utf-8", errors="replace")
        # Locate each register_patch( ... ) body, then pull name + tested_against
        # from within it (so unrelated `name =` arguments don't poison the result).
        for m in re.finditer(r"register_patch\s*\(", text):
            start = m.end()
            depth = 1
            i = start
            while i < len(text) and depth > 0:
                ch = text[i]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                i += 1
            if depth != 0:
                continue
            body = text[start:i - 1]
            name_m = re.search(r'\bname\s*=\s*["\']([^"\']+)["\']', body)
            tested_m = re.search(
                r'\btested_against\s*=\s*["\']([^"\']+)["\']', body
            )
            if not (name_m and tested_m):
                break
            parsed = _parse_tested_against(tested_m.group(1))
            if parsed:
                out.append((name_m.group(1), parsed[0], parsed[1]))
            break
    return out


def _installed_python(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def _installed_r_batch(pkgs: list[str]) -> dict[str, str | None]:
    """Spawn one Rscript invocation to query packageVersion for all R upstreams.

    Returns {pkg: version-or-None}. R is slow to start; batching keeps the
    command snappy even when many R patches exist.
    """
    if not pkgs:
        return {}
    # Build an R one-liner that prints JSON. tryCatch returns NA for missing.
    pkgs_lit = ",".join(f'"{p}"' for p in pkgs)
    script = (
        f"pkgs <- c({pkgs_lit}); "
        "out <- list(); "
        "for (p in pkgs) { "
        "  v <- tryCatch(as.character(utils::packageVersion(p)), "
        "                error = function(e) NA_character_); "
        "  out[[p]] <- if (is.na(v)) NULL else v "
        "}; "
        "cat(jsonlite::toJSON(out, auto_unbox = TRUE))"
    )
    try:
        proc = subprocess.run(
            ["Rscript", "-e", script],
            capture_output=True, text=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # No Rscript on PATH (or timed out) — treat all as unknown.
        return {p: None for p in pkgs}
    if proc.returncode != 0:
        return {p: None for p in pkgs}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {p: None for p in pkgs}
    return {p: data.get(p) for p in pkgs}


def _build_rows(framework_root: Path) -> list[_VersionRow]:
    rows: list[_VersionRow] = []
    py_rows = _scan_python(framework_root)
    r_rows = _scan_r(framework_root)
    r_pkgs = sorted({u for _, u, _ in r_rows})
    r_installed = _installed_r_batch(r_pkgs)
    for patch, upstream, tested in py_rows:
        installed = _installed_python(upstream)
        if installed is None:
            status = "missing"
        elif installed == tested:
            status = "ok"
        else:
            status = "drift"
        rows.append(_VersionRow(patch, "py", upstream, tested, installed, status))
    for patch, upstream, tested in r_rows:
        installed = r_installed.get(upstream)
        if installed is None:
            status = "missing"
        elif installed == tested:
            status = "ok"
        else:
            status = "drift"
        rows.append(_VersionRow(patch, "R", upstream, tested, installed, status))
    return rows


def _print_table(rows: list[_VersionRow], filter_status: set[str] | None = None) -> None:
    if filter_status:
        rows = [r for r in rows if r.status in filter_status]
    if not rows:
        print("(no rows)")
        return
    widths = {
        "patch":    max(5, max(len(r.patch) for r in rows)),
        "lang":     4,
        "upstream": max(8, max(len(r.upstream) for r in rows)),
        "tested":   max(6, max(len(r.tested) for r in rows)),
        "installed": max(9, max(len(r.installed or "-") for r in rows)),
        "status":   7,
    }
    header = (
        f"{'patch':<{widths['patch']}}  "
        f"{'lang':<{widths['lang']}}  "
        f"{'upstream':<{widths['upstream']}}  "
        f"{'tested':<{widths['tested']}}  "
        f"{'installed':<{widths['installed']}}  "
        f"status"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        installed = r.installed or "-"
        print(
            f"{r.patch:<{widths['patch']}}  "
            f"{r.language:<{widths['lang']}}  "
            f"{r.upstream:<{widths['upstream']}}  "
            f"{r.tested:<{widths['tested']}}  "
            f"{installed:<{widths['installed']}}  "
            f"{r.status}"
        )


def cmd_package_check_versions(args) -> int:
    fr_arg = getattr(args, "framework_root", None)
    framework_root = Path(fr_arg).resolve() if fr_arg else find_framework_root(Path.cwd())
    if framework_root is None or not framework_root.is_dir():
        die("not inside an autozyme-framework workspace; pass --framework-root")

    rows = _build_rows(framework_root)
    if not rows:
        print("(no patches with tested_against found)")
        return 0

    only_drift = bool(getattr(args, "only_drift", False))
    filter_set = {"drift", "missing"} if only_drift else None
    _print_table(rows, filter_status=filter_set)

    n_drift = sum(1 for r in rows if r.status == "drift")
    n_missing = sum(1 for r in rows if r.status == "missing")
    n_ok = sum(1 for r in rows if r.status == "ok")
    print(f"\n{n_ok} ok, {n_drift} drift, {n_missing} missing "
          f"({len(rows)} patches scanned)")
    return 1 if (n_drift or n_missing) else 0
