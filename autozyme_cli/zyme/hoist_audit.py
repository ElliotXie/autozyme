"""Hoist audit for pipeline/run.{py,R}.

Detects work moved OUTSIDE the timer that the target function would otherwise
execute inside its own call. The decontX round-58 pattern — the surgical and
reviewer-indefensible form — looks like this:

    .orig_initZ <- getFromNamespace(".decontxInitializeZ", "celda")
    .precomputed <- .orig_initZ(counts, ...)         # ← runs UMAP+DBSCAN BEFORE t0
    install_override(".decontxInitializeZ", "celda",
                     function(...) .precomputed)     # override returns cache
    t0 <- Sys.time()
    res <- celda::decontX(counts, ...)               # init is now O(1)

Work is bit-exact upstream, but moved outside the timer — `speed_sec` shows
a gain a real `decontX()` caller never receives.

Two structural checks per language (Python AST, R regex):
  1. Alias bound to a private upstream symbol (R: `getFromNamespace`, Py: import
     or attribute access onto a name starting with `_`) + a call to that alias
     before t0.
  2. Direct triple-colon (R) / private-attribute (Py) call before t0.

Override via `task.yaml::hoist_exempt: <one-line reason>` — skips the check
silently. Documented in 2_iterate.md "Speedup must reach the user".
"""
from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from zyme.argdiff import _call_name_chain, _find_timer_lineno


class HoistViolation(Exception):
    """Raised when a pipeline/run.{py,R} hoist check fires."""


def scan_pipeline_hoist(pipeline_path: Path) -> list[dict]:
    """Scan pipeline/run.{py,R} and return the list of violation records.

    Empty list means no violation. Caller decides whether to raise, log,
    print, or bypass.
    """
    if not pipeline_path.exists():
        return []
    suffix = pipeline_path.suffix.lower()
    source = pipeline_path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".py":
        return _check_python(source)
    if suffix in (".r", ".rscript"):
        return _check_r(source)
    return []


def check_pipeline_hoist(pipeline_path: Path) -> None:
    """Legacy entry: raises HoistViolation if any pattern fires.

    Retained for callers that want the simple "scan + raise" semantics. New
    callers should use `scan_pipeline_hoist` + structured handling instead.
    """
    violations = scan_pipeline_hoist(pipeline_path)
    if violations:
        raise HoistViolation(format_violations_message(pipeline_path, violations))


def format_violations_message(
    pipeline_path: Path,
    violations: list[dict],
    *,
    hypothesis: str | None = None,
) -> str:
    """Build the human-readable warning string. Public so commands/run.py can
    print + log the same text. `hypothesis` is woven into the bypass example
    so the agent can copy-paste the corrected command verbatim."""
    return _format_violations(pipeline_path, violations, hypothesis=hypothesis)


def append_hoist_log(
    task_dir: Path,
    *,
    round_num: int | None,
    commit: str | None,
    pipeline_rel: str,
    violations: list[dict],
    outcome: str,
    bypass_reason: str | None = None,
    hypothesis: str | None = None,
) -> None:
    """Append one JSON line to `.zyme/hoist_log.jsonl`.

    `outcome` is one of: "blocked" | "exempted" | "bypassed". Failures here
    are non-fatal (we don't want a logging glitch to crash the run).
    """
    try:
        log_dir = task_dir / ".zyme"
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "round": round_num,
            "commit": commit,
            "hypothesis": hypothesis,
            "pipeline": pipeline_rel,
            "outcome": outcome,
            "bypass_reason": bypass_reason,
            "violations": violations,
        }
        log_path = log_dir / "hoist_log.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Best-effort logging; never block the actual run on log failure.
        pass


def read_hoist_exempt(task_dir: Path) -> str | None:
    """Return the `hoist_exempt: <reason>` value from task.yaml if set, else None.

    Simple line-scan (no YAML dep) for v1; matches the pattern used by other
    task.yaml readers in helpers.py.
    """
    yaml_path = task_dir / "task.yaml"
    if not yaml_path.exists():
        return None
    try:
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("hoist_exempt:"):
                value = s.split(":", 1)[1].strip()
                # Strip inline comment + quotes.
                if "#" in value:
                    value = value.split("#", 1)[0].strip()
                value = value.strip("\"'")
                return value or None
    except OSError:
        return None
    return None


# ----------------------------------------------------------------------------
# Python AST checks
# ----------------------------------------------------------------------------

def _check_python(source: str) -> list[dict]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    timer_lineno = _find_timer_lineno(tree)
    if timer_lineno is None:
        return []

    private_aliases = _collect_private_aliases(tree)
    if not private_aliases:
        # Still possible to flag direct private-attribute calls below; continue.
        pass

    violations: list[dict] = []
    for call in _module_level_calls_before(tree, timer_lineno):
        chain = _call_name_chain(call.func)
        if not chain:
            continue
        # Case 1: bare name that resolves to a private alias.
        if len(chain) == 1 and chain[0] in private_aliases:
            violations.append({
                "pattern": "private_alias_call",
                "line": call.lineno,
                "code": _safe_unparse(call),
                "resolved": private_aliases[chain[0]],
                "hint": "upstream-internal symbol invoked before t0 — its work is "
                        "moved outside the timer",
            })
        # Case 2: `pkg._sub._private(...)` direct call (any non-tail segment
        # starts with underscore → upstream-internal access).
        elif len(chain) >= 2 and any(p.startswith("_") for p in chain[:-1]):
            violations.append({
                "pattern": "private_attr_call",
                "line": call.lineno,
                "code": _safe_unparse(call),
                "resolved": ".".join(chain),
                "hint": "upstream-internal attribute called before t0 — move it "
                        "inside the timed window or drop it",
            })
    return violations


def _collect_private_aliases(tree: ast.Module) -> dict[str, str]:
    """Map name -> resolved chain for module-level bindings to private upstream
    symbols.

    Triggers on:
      - `alias = <pkg>._<x>._<y>` (Assign with Attribute RHS, any private segment)
      - `from <pkg>._sub import _foo as alias`
      - `from <pkg> import _foo as alias`  (top-level private name)
    """
    out: dict[str, str] = {}
    for node in tree.body:  # module level only
        if isinstance(node, ast.Assign):
            if not isinstance(node.value, (ast.Name, ast.Attribute)):
                continue
            chain = _call_name_chain(node.value)
            if not chain or not any(p.startswith("_") for p in chain):
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = ".".join(chain)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            module_is_private = any(p.startswith("_") for p in module.split(".") if p)
            for alias in node.names:
                asname = alias.asname or alias.name
                resolved = f"{module}.{alias.name}" if module else alias.name
                if module_is_private or alias.name.startswith("_"):
                    out[asname] = resolved
    return out


def _module_level_calls_before(
    tree: ast.Module, timer_lineno: int,
) -> list[ast.Call]:
    """Calls that execute at module-load AND happen before the timer line.

    Skips bodies of FunctionDef / AsyncFunctionDef / ClassDef so we don't flag
    helper functions that are merely defined before t0 — they only fire when
    called from inside the timed region.
    """
    out: list[ast.Call] = []

    def _scan(stmts):
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(stmt, ast.If):
                _scan(stmt.body); _scan(stmt.orelse); continue
            if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                _scan(stmt.body); _scan(stmt.orelse); continue
            if isinstance(stmt, (ast.With, ast.AsyncWith)):
                _scan(stmt.body); continue
            if isinstance(stmt, ast.Try):
                _scan(stmt.body)
                for h in stmt.handlers:
                    _scan(h.body)
                _scan(stmt.orelse); _scan(stmt.finalbody); continue
            for sub in ast.walk(stmt):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(sub, ast.Call) and sub.lineno < timer_lineno:
                    out.append(sub)

    _scan(tree.body)
    return out


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)[:120]
    except Exception:
        return "<unparseable>"


# ----------------------------------------------------------------------------
# R regex checks
# ----------------------------------------------------------------------------

_R_T0_PATTERN = re.compile(r"\bSys\.time\s*\(\s*\)|\bproc\.time\s*\(\s*\)|\btictoc::tic\s*\(")
_R_GETNS_ASSIGN = re.compile(
    r"^\s*([\.\w]+)\s*(?:<-|=)\s*getFromNamespace\s*\(\s*['\"]([^'\"]+)['\"]\s*,"
    r"\s*['\"]([^'\"]+)['\"]\s*\)"
)
_R_TRIPLE_COLON_CALL = re.compile(r"\b([\.\w]+)\s*:::\s*([\.\w]+)\s*\(")
_R_COMMENT = re.compile(r"#.*$")


def _strip_comment(line: str) -> str:
    return _R_COMMENT.sub("", line)


def _strip_braced_blocks(source: str) -> str:
    """Blank out everything inside `{ ... }` (any nesting depth), preserving
    newlines so line numbers stay valid. String literals are skipped so a
    quoted `{` doesn't open a fake block. This collapses function bodies,
    `local({...})` blocks, and `if/for/while` bodies — leaving only the
    top-level R script for the regex pass.
    """
    out: list[str] = []
    depth = 0
    in_string: str | None = None  # None, '"', or "'"
    i = 0
    while i < len(source):
        ch = source[i]
        if in_string is not None:
            # Inside a string: preserve at depth 0, blank otherwise.
            out.append(ch if depth == 0 else (ch if ch == "\n" else " "))
            if ch == "\\" and i + 1 < len(source):
                # Skip the escaped char (still preserve newlines for lineno).
                nxt = source[i + 1]
                out.append(nxt if depth == 0 else (nxt if nxt == "\n" else " "))
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_string = ch
            out.append(ch if depth == 0 else (ch if ch == "\n" else " "))
            i += 1
            continue
        if ch == "#":
            # Strip comment to end-of-line; preserve newline.
            while i < len(source) and source[i] != "\n":
                out.append(" " if depth > 0 else " ")  # blank the comment too
                i += 1
            continue
        if depth == 0:
            out.append(ch)
            if ch == "{":
                depth = 1
            i += 1
            continue
        # depth > 0
        if ch == "{":
            depth += 1
            out.append(" ")
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append(ch)  # preserve the closing brace at top level
            else:
                out.append(" ")
        else:
            out.append("\n" if ch == "\n" else " ")
        i += 1
    return "".join(out)


def _check_r(source: str) -> list[dict]:
    # Pre-process: blank everything inside `{ ... }` so the regex pass only
    # sees top-level (script-level) R code. This eliminates the false-positive
    # class where a private call sits inside an override wrapper body.
    source = _strip_braced_blocks(source)
    lines = source.splitlines()
    # Find timer boundary line — first Sys.time()/proc.time()/tic() not in a comment.
    timer_line = None
    for i, raw in enumerate(lines, 1):
        s = _strip_comment(raw)
        if _R_T0_PATTERN.search(s):
            timer_line = i
            break
    if timer_line is None:
        return []
    # Collect getFromNamespace aliases: alias -> (private_name, pkg)
    ns_aliases: dict[str, tuple[str, str]] = {}
    for raw in lines:
        s = _strip_comment(raw)
        m = _R_GETNS_ASSIGN.search(s)
        if m:
            ns_aliases[m.group(1)] = (m.group(2), m.group(3))

    violations: list[dict] = []
    # Scan code BEFORE timer_line (line index 1..timer_line-1)
    for i, raw in enumerate(lines[: timer_line - 1], 1):
        s = _strip_comment(raw).strip()
        if not s:
            continue

        # Case 1: a getFromNamespace alias is called.
        # Skip the assignment line itself (alias <- getFromNamespace(...)).
        ns_match = _R_GETNS_ASSIGN.search(s)
        own_assign_alias = ns_match.group(1) if ns_match else None
        for alias, (priv, pkg) in ns_aliases.items():
            if alias == own_assign_alias:
                continue
            # Match `<alias>(` not preceded by a word/dot character (avoid
            # accidental substring matches like `.orig_initZ_compiled(...)`
            # matching alias `.orig_initZ`).
            if re.search(rf"(?<![\w\.]){re.escape(alias)}\s*\(", s):
                violations.append({
                    "pattern": "r_getns_call",
                    "line": i,
                    "code": s[:120],
                    "resolved": f"{pkg}:::{priv}",
                    "hint": "upstream-internal call before t0 — move it inside "
                            "the timed window or drop it",
                })

        # Case 2: pkg:::private(...) direct call.
        m = _R_TRIPLE_COLON_CALL.search(s)
        if m:
            pkg, priv = m.group(1), m.group(2)
            violations.append({
                "pattern": "r_triple_colon",
                "line": i,
                "code": s[:120],
                "resolved": f"{pkg}:::{priv}",
                "hint": "upstream private function called before t0 — pkg:::name "
                        "should not run on real data outside the timer",
            })
    return violations


# ----------------------------------------------------------------------------
# Error formatter
# ----------------------------------------------------------------------------

def _format_violations(
    path: Path, violations: list[dict], *, hypothesis: str | None = None,
) -> str:
    hyp_quoted = f'"{hypothesis}"' if hypothesis else '"<hypothesis>"'
    lines = [
        "",
        f"Hoist check failed for {path}.",
        "",
        "Pipeline invokes upstream-internal functions BEFORE the timer (t0).",
        "Work moved outside t0 doesn't reduce wall-time for a downstream user —",
        "they pay the full cost on a fresh call. See `2_iterate.md` →",
        "  'Speedup must reach the user' → Rule B / Rule C.",
        "",
    ]
    for v in violations:
        lines.append(f"  [{v['pattern']}] line {v['line']}:")
        lines.append(f"    {v['code']}")
        lines.append(f"    resolved → {v['resolved']}")
        lines.append(f"    → {v['hint']}")
        lines.append("")
    lines.extend([
        "To fix:",
        "  - Move the call INSIDE the t0..t1 window. The reference doesn't get",
        "    pre-computed init; pipeline should not either.",
        "  - If you're priming JIT / lazy-loads, use a SYNTHETIC dummy with",
        "    different args — this check ignores private calls whose results",
        "    aren't referenced inside the timed region.",
        "",
        "If you believe the check misfired for THIS round (e.g., the flagged",
        "call doesn't actually move work outside t0 — synthetic dummy, runtime",
        "knob a user would also flip, etc.), bypass once with:",
        "",
        f"    zyme run {hyp_quoted} --bypass-hoist \"<one-line reason>\"",
        "",
        "Your reason is appended to `.zyme/hoist_log.jsonl`. Use sparingly:",
        "every bypass is part of a paper trail the reviewer can audit later.",
        "",
        "If the hoist is permanently part of the task's contract (target IS a",
        "cached lookup, documented runtime precondition, etc.), declare it in",
        "task.yaml instead:",
        "    hoist_exempt: <one-line reason>",
    ])
    return "\n".join(lines)
