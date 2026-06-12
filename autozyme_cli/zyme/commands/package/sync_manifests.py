"""`zyme package sync-manifests` — reconcile UPSTREAMS / .zyme_upstreams.

Both autozyme_py and autozyme_r maintain a manifest mapping patch name ->
list of upstream packages. The package agent must keep these in sync with
each patch's ``register_patch()`` call; this command does it mechanically.

  - Python side: walk ``autozyme_py/src/autozyme/*/__init__.py``, parse
    with ``ast``, extract ``register_patch(name=..., targets=[...], ...)``
    arguments. The first element of each target tuple is the upstream
    module path; we take its top-level package as the upstream identifier.
    ``tested_upstream_versions`` keys are merged in (covers patches that
    pull a co-dependency like ``pyro`` without listing it as a target).

  - R side: walk ``autozyme_r/inst/patches/*/patch.R``, regex-extract
    ``register_patch(name = "...", upstream = "...")``. R's call form
    declares the upstream package directly, so no AST is needed.

Outputs a diff against the existing manifests; with ``--apply`` rewrites
the manifest blocks in place. The rewrite is bounded by sentinel comments
inserted on first apply, so other content in ``_subsets.py`` /
``subsets.R`` is preserved verbatim.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from zyme.scan import find_framework_root
from zyme.utils import die


# Sentinel comments that bracket the auto-generated block. The block is
# overwritten on each apply; everything outside it is left untouched.
_PY_BEGIN = "# AUTOZYME-GENERATED-UPSTREAMS-BEGIN"
_PY_END = "# AUTOZYME-GENERATED-UPSTREAMS-END"
_R_BEGIN = "# AUTOZYME-GENERATED-UPSTREAMS-BEGIN"
_R_END = "# AUTOZYME-GENERATED-UPSTREAMS-END"


def _top_level_pkg(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def _scan_python(framework_root: Path) -> dict[str, list[str]]:
    """Return {patch_name: sorted unique upstream packages}.

    Includes underscore-prefixed test patches (e.g. ``_test_json``) because
    those are legitimately in the UPSTREAMS manifest. We exclude only
    private core modules (``_core``, ``_subsets``, etc.) by requiring a
    ``register_patch`` call inside the file.
    """
    root = framework_root / "autozyme_py" / "src" / "autozyme"
    out: dict[str, set[str]] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        init = child / "__init__.py"
        if not init.is_file():
            continue
        text = init.read_text(encoding="utf-8", errors="replace")
        if "register_patch" not in text:
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
            name = None
            upstreams: set[str] = set()
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    name = kw.value.value
                elif kw.arg == "targets" and isinstance(kw.value, (ast.List, ast.Tuple)):
                    for elt in kw.value.elts:
                        if not isinstance(elt, (ast.Tuple, ast.List)) or not elt.elts:
                            continue
                        first = elt.elts[0]
                        if isinstance(first, ast.Constant) and isinstance(first.value, str):
                            upstreams.add(_top_level_pkg(first.value))
                elif kw.arg == "tested_upstream_versions" and isinstance(kw.value, ast.Dict):
                    for k in kw.value.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            upstreams.add(_top_level_pkg(k.value))
            if name and upstreams:
                out.setdefault(name, set()).update(upstreams)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _scan_r(framework_root: Path) -> dict[str, str]:
    """Return {patch_name: upstream package}.

    Looks specifically inside ``register_patch(...)`` call bodies, not
    anywhere in the file. Without that scoping, unrelated ``name = "..."``
    arguments to other helpers (e.g. ``object.name = "object1"``) would
    poison the result.
    """
    root = framework_root / "autozyme_r" / "inst" / "patches"
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        patch_file = child / "patch.R"
        if not patch_file.is_file():
            continue
        text = patch_file.read_text(encoding="utf-8", errors="replace")
        # Locate each register_patch( ... ) span by brace-balanced match.
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
            # \b avoids matching object.name=, fast.name=, etc.
            name_m = re.search(r'\bname\s*=\s*["\']([^"\']+)["\']', body)
            ups_m = re.search(r'\bupstream\s*=\s*["\']([^"\']+)["\']', body)
            if name_m and ups_m:
                out[name_m.group(1)] = ups_m.group(1)
                break  # one register_patch per patch.R
    return dict(sorted(out.items()))


def _load_current_python(path: Path) -> dict[str, list[str]]:
    """Parse the existing UPSTREAMS dict literal out of _subsets.py."""
    text = path.read_text(encoding="utf-8", errors="replace")
    mod = ast.parse(text)
    for node in mod.body:
        if not isinstance(node, ast.AnnAssign) and not isinstance(node, ast.Assign):
            continue
        targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "UPSTREAMS":
                value = node.value
                if isinstance(value, ast.Dict):
                    out: dict[str, list[str]] = {}
                    for k, v in zip(value.keys, value.values):
                        if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                            continue
                        if isinstance(v, ast.List):
                            out[k.value] = [
                                e.value for e in v.elts
                                if isinstance(e, ast.Constant) and isinstance(e.value, str)
                            ]
                    return out
    return {}


def _load_current_r(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Match .zyme_upstreams <- list(\n  name = "Pkg",\n  ...\n)
    block = re.search(r"\.zyme_upstreams\s*<-\s*list\((.*?)\)", text, re.DOTALL)
    if not block:
        return {}
    body = block.group(1)
    out: dict[str, str] = {}
    for m in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["\']([^"\']+)["\']', body):
        out[m.group(1)] = m.group(2)
    return out


def _diff_py(current: dict[str, list[str]], target: dict[str, list[str]]) -> list[str]:
    lines: list[str] = []
    added = sorted(set(target) - set(current))
    removed = sorted(set(current) - set(target))
    common = sorted(set(target) & set(current))
    for k in added:
        lines.append(f"  + {k!r}: {target[k]}")
    for k in removed:
        lines.append(f"  - {k!r}: {current[k]}")
    for k in common:
        if current[k] != target[k]:
            lines.append(f"  ~ {k!r}: {current[k]} -> {target[k]}")
    return lines


def _diff_r(current: dict[str, str], target: dict[str, str]) -> list[str]:
    lines: list[str] = []
    added = sorted(set(target) - set(current))
    removed = sorted(set(current) - set(target))
    common = sorted(set(target) & set(current))
    for k in added:
        lines.append(f"  + {k!r}: {target[k]!r}")
    for k in removed:
        lines.append(f"  - {k!r}: {current[k]!r}")
    for k in common:
        if current[k] != target[k]:
            lines.append(f"  ~ {k!r}: {current[k]!r} -> {target[k]!r}")
    return lines


def _format_py_block(target: dict[str, list[str]]) -> str:
    inner = "\n".join(
        f'    "{k}": {target[k]!r},'.replace("'", '"')
        for k in target
    )
    return (
        f"{_PY_BEGIN}\n"
        f"UPSTREAMS: dict[str, list[str]] = {{\n"
        f"{inner}\n"
        f"}}\n"
        f"{_PY_END}"
    )


def _format_r_block(target: dict[str, str]) -> str:
    width = max((len(k) for k in target), default=0)
    rows = [f"  {k.ljust(width)} = \"{v}\"" for k, v in target.items()]
    inner = ",\n".join(rows)
    return (
        f"{_R_BEGIN}\n"
        f".zyme_upstreams <- list(\n"
        f"{inner}\n"
        f")\n"
        f"{_R_END}"
    )


def _find_balanced_block(text: str, header_pat: re.Pattern, open_ch: str, close_ch: str) -> tuple[int, int] | None:
    """Locate a ``header(...balanced...)`` span using header regex + bracket counting.

    Returns (start, end) byte offsets where ``text[start:end]`` is the full
    block (header + opening bracket + balanced body + closing bracket), or
    None if no match. Brace-counting is more reliable than nesting-blind
    regex for type-annotated dict literals and multi-line R list() blocks.
    """
    m = header_pat.search(text)
    if not m:
        return None
    # Find the open bracket starting at or after m.end().
    i = text.find(open_ch, m.end() - 1)
    if i < 0:
        return None
    depth = 1
    j = i + 1
    while j < len(text) and depth > 0:
        ch = text[j]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
        j += 1
    if depth != 0:
        return None
    return m.start(), j


def _rewrite_with_sentinels(
    path: Path,
    begin: str,
    end: str,
    block: str,
    fallback_header: re.Pattern,
    open_ch: str,
    close_ch: str,
) -> None:
    """Replace ``begin..end`` block in place; if sentinels are absent (first
    apply), fall back to a header-anchored balanced-bracket span.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if begin in text and end in text:
        new_text = re.sub(
            re.escape(begin) + r".*?" + re.escape(end),
            block,
            text,
            count=1,
            flags=re.DOTALL,
        )
    else:
        span = _find_balanced_block(text, fallback_header, open_ch, close_ch)
        if span is None:
            die(f"could not locate existing block to replace in {path}; "
                f"add sentinels manually: {begin} ... {end}")
        start, stop = span
        new_text = text[:start] + block + text[stop:]
    path.write_text(new_text, encoding="utf-8")


def cmd_package_sync_manifests(args) -> int:
    fr_arg = getattr(args, "framework_root", None)
    framework_root = Path(fr_arg).resolve() if fr_arg else find_framework_root(Path.cwd())
    if framework_root is None or not framework_root.is_dir():
        die("not inside an autozyme-framework workspace; pass --framework-root")

    py_target = _scan_python(framework_root)
    r_target = _scan_r(framework_root)

    py_path = framework_root / "autozyme_py" / "src" / "autozyme" / "_subsets.py"
    r_path = framework_root / "autozyme_r" / "R" / "subsets.R"
    py_current = _load_current_python(py_path) if py_path.is_file() else {}
    r_current = _load_current_r(r_path) if r_path.is_file() else {}

    py_diff = _diff_py(py_current, py_target)
    r_diff = _diff_r(r_current, r_target)

    print(f"\n-- autozyme_py UPSTREAMS  ({py_path}) --")
    if not py_diff:
        print("  (in sync)")
    else:
        for ln in py_diff:
            print(ln)

    print(f"\n-- autozyme_r .zyme_upstreams  ({r_path}) --")
    if not r_diff:
        print("  (in sync)")
    else:
        for ln in r_diff:
            print(ln)

    if not py_diff and not r_diff:
        return 0

    if not getattr(args, "apply", False):
        print("\nRun with --apply to write the changes.")
        return 1

    if py_diff:
        # Fallback header: `UPSTREAMS[: dict[...]] =` then a `{...}` block.
        # Brace counting handles nested type annotations.
        fb_py = re.compile(r"UPSTREAMS\s*(?::[^=]*?)?=\s*")
        _rewrite_with_sentinels(
            py_path, _PY_BEGIN, _PY_END, _format_py_block(py_target),
            fb_py, "{", "}",
        )
        print(f"\n[OK] wrote {py_path}")
    if r_diff:
        fb_r = re.compile(r"\.zyme_upstreams\s*<-\s*list\s*")
        _rewrite_with_sentinels(
            r_path, _R_BEGIN, _R_END, _format_r_block(r_target),
            fb_r, "(", ")",
        )
        print(f"[OK] wrote {r_path}")
    return 0
