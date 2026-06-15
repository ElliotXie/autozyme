"""Fingerprint check for reference.{py,R} inflation patterns.

Layer 2 of the Cat A defense (see Current_problem.md). Detects synthetic
duplication patterns that optimizers pattern-match and short-circuit,
producing speedups that don't generalize.

Three AST checks against reference.py (Python) — full coverage:
  1. `[expr] * N` / `N * [expr]` — list-repeat for file lists
  2. `np.tile(...)` / `np.repeat(...)` — array tile-sizing
  3. `for _ in range(N): <call>(same_args)` — wall-time inflation via repeat-loop

Regex-based partial check for reference.R:
  - `rep(<expr>, N)` — R's equivalent of list-repeat

Override via `task.yaml::synthesis: <reason>` + CLI `--accept-synthesis`,
documented in 1_init.md step 2.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path


class FingerprintViolation(Exception):
    """Raised when a reference.{py,R} fingerprint check fires."""


def check_reference_fingerprint(reference_path: Path) -> None:
    """Scan reference.{py,R} for forbidden inflation patterns.

    Raises FingerprintViolation with an actionable message if any pattern
    fires; returns silently otherwise. No-op when the file doesn't exist
    (caller is responsible for checking that earlier in the pipeline).
    """
    if not reference_path.exists():
        return
    suffix = reference_path.suffix.lower()
    source = reference_path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".py":
        violations = _check_python(source)
    elif suffix in (".r", ".rscript"):
        violations = _check_r(source)
    else:
        return
    if violations:
        raise FingerprintViolation(_format_violations(reference_path, violations))


# ----------------------------------------------------------------------------
# Python AST checks
# ----------------------------------------------------------------------------

def _check_python(source: str) -> list[dict]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Syntax errors are someone else's problem — don't block the
        # baseline command on them.
        return []
    violations: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            v = _check_list_mul(node)
            if v:
                violations.append(v)
        elif isinstance(node, ast.Call):
            v = _check_tile_call(node)
            if v:
                violations.append(v)
        elif isinstance(node, ast.For):
            v = _check_target_loop(node)
            if v:
                violations.append(v)
    return violations


def _check_list_mul(node: ast.BinOp) -> dict | None:
    """Flag `[expr] * N` and `N * [expr]` where `[expr]` is a single-element
    list of a non-trivial expression (a path-like Call, Attribute, Subscript,
    or string Constant).

    Single-element lists of None / 0 / False / "" are fine (init patterns).
    Multi-element lists are not flagged either — they're already explicit
    distinct content.
    """
    list_side = None
    other_side = None
    if isinstance(node.left, ast.List):
        list_side, other_side = node.left, node.right
    elif isinstance(node.right, ast.List):
        list_side, other_side = node.right, node.left
    else:
        return None
    if len(list_side.elts) != 1:
        return None
    elt = list_side.elts[0]
    # Allow trivial init values: None / 0 / 0.0 / False / "" / b""
    if isinstance(elt, ast.Constant) and elt.value in (None, 0, 0.0, False, "", b""):
        return None
    # Otherwise, the element is a "real" expression — flag.
    try:
        code = ast.unparse(node)
    except Exception:
        code = "[...] * N"
    return {
        "pattern": "list_mul",
        "line": node.lineno,
        "code": code[:100],
        "hint": "use a list of distinct paths / sources, not the same one repeated",
    }


def _check_tile_call(node: ast.Call) -> dict | None:
    """Flag `np.tile(...)` / `np.repeat(...)` / `numpy.tile(...)` / `numpy.repeat(...)`.

    Both functions create arrays where slice-N equals slice-N+k for some k,
    which lets an optimizer recover the base via modulo indexing.
    """
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr not in ("tile", "repeat"):
        return None
    base = func.value
    if not (isinstance(base, ast.Name) and base.id in ("np", "numpy")):
        return None
    try:
        code = ast.unparse(node)
    except Exception:
        code = f"{base.id}.{func.attr}(...)"
    return {
        "pattern": "numpy_tile",
        "line": node.lineno,
        "code": code[:100],
        "hint": "build the tier from distinct cohorts / longer instances / perturbed copies",
    }


def _check_target_loop(node: ast.For) -> dict | None:
    """Flag `for _ in range(N): <call>(...)` — wall-time inflation by repeated calls.

    Conditions (all must hold):
      - loop variable is `_` OR a Name that is never read in the body
      - iterator is `range(...)` (constant N >= 2, or any non-constant)
      - body contains at least one Call
    """
    # Iterator must be range(...)
    if not (isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"):
        return None
    # If range(N) with N a constant int, require N >= 2 (range(1) is silly anyway)
    if node.iter.args:
        first = node.iter.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, int) and first.value < 2:
            return None
    # Loop variable must be `_` or a Name that is never read in the body
    target = node.target
    if not isinstance(target, ast.Name):
        return None
    var_name = target.id
    if var_name != "_":
        for sub in node.body:
            for child in ast.walk(sub):
                if (isinstance(child, ast.Name)
                        and child.id == var_name
                        and isinstance(child.ctx, ast.Load)):
                    return None  # variable is used → legit batch loop
    # Body must contain at least one Call
    has_call = False
    for sub in node.body:
        for child in ast.walk(sub):
            if isinstance(child, ast.Call):
                has_call = True
                break
        if has_call:
            break
    if not has_call:
        return None
    # Build a short preview of what's being looped
    try:
        range_repr = ast.unparse(node.iter)[:40]
        first_stmt = ast.unparse(node.body[0])[:60] if node.body else "..."
        code = f"for {var_name} in {range_repr}: {first_stmt}"
    except Exception:
        code = "for _ in range(N): <call>(...)"
    return {
        "pattern": "target_loop",
        "line": node.lineno,
        "code": code[:120],
        "hint": "loop over distinct inputs, not repeated calls with the same args",
    }


# ----------------------------------------------------------------------------
# R regex checks (partial coverage — flags the obvious patterns)
# ----------------------------------------------------------------------------

_R_REP_PATTERN = re.compile(r"\brep\s*\(\s*([^,()]+?)\s*,\s*(?:times\s*=\s*)?(\d+)\s*\)")
_R_REPLICATE_PATTERN = re.compile(r"\breplicate\s*\(\s*(\d+)\s*,")


def _check_r(source: str) -> list[dict]:
    violations: list[dict] = []
    for i, raw in enumerate(source.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Pattern: rep(<x>, N) with N >= 2 — duplicate-expansion
        m = _R_REP_PATTERN.search(raw)
        if m:
            try:
                n = int(m.group(2))
            except ValueError:
                n = 0
            if n >= 2:
                violations.append({
                    "pattern": "r_rep",
                    "line": i,
                    "code": stripped[:100],
                    "hint": "rep(x, N) duplicates the same value — use distinct sources",
                })
                continue
        # Pattern: replicate(N, expr) — runs the same expression N times
        m = _R_REPLICATE_PATTERN.search(raw)
        if m:
            try:
                n = int(m.group(1))
            except ValueError:
                n = 0
            if n >= 2:
                violations.append({
                    "pattern": "r_replicate",
                    "line": i,
                    "code": stripped[:100],
                    "hint": "replicate(N, expr) re-runs the same call — use distinct inputs",
                })
    return violations


# ----------------------------------------------------------------------------
# Error formatter
# ----------------------------------------------------------------------------

def _format_violations(path: Path, violations: list[dict]) -> str:
    lines = [
        "",
        f"Fingerprint check failed for {path}.",
        "",
        "This reference script contains an inflation pattern that an optimizer",
        "will detect and short-circuit, producing speedups that don't generalize.",
        "See `1_init.md` step 2 — Data source rules (hard).",
        "",
    ]
    for v in violations:
        lines.append(f"  [{v['pattern']}] line {v['line']}:")
        lines.append(f"    {v['code']}")
        lines.append(f"    → {v['hint']}")
        lines.append("")
    lines.extend([
        "To fix, follow step 2's escalation ladder:",
        "  1. Preferred  — different cohort / longer trajectory / deeper-sampling",
        "                  instance of the same data type.",
        "  2. Acceptable — independent perturbations (each frame gets unique small",
        "                  noise so per-item outputs truly differ).",
        "  3. Last resort — narrow the tier set (only tiny exists; document why",
        "                   in README).",
        "",
        "If this synthesis is genuinely intrinsic to the workload (PDE solver,",
        "Lomb-Scargle on generated signals, license-blocked data):",
        "  - Add a top-level `synthesis: <one-sentence reason>` to task.yaml.",
        "  - Re-run with `--accept-synthesis`.",
    ])
    return "\n".join(lines)
