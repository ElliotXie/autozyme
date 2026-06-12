"""Lint rules derived from CAVEATS.md.

Each rule is a function ``rule_<id>(ctx) -> list[LintFinding]``. The dispatcher
in ``lint.py`` collects rules by language tag and invokes them with a
``LintContext`` per patch. Rules return zero or more ``LintFinding`` records.

Phase 1–3 rules (8 implemented):

  R-LIBRARY-GUARD         (R#1, R#3, R#8 unified)
    MAST / spacexr / WGCNA touched by smoke `call` but smoke `load` only
    calls requireNamespace() — needs library(). Detects three reflection
    flavors with one rule because the fix is identical.

  R-RCPP-INLINE           (R#4)
    Rcpp::cppFunction("...") or sourceCpp(code=...) at file scope. Must move
    to src/<kernel>.cpp.

  R-DATATABLE-NS          (R#7)
    Patch uses `:=` but DESCRIPTION lacks `data.table` in Imports OR NAMESPACE
    lacks `importFrom(data.table, ":=")`. Package-level check.

  R-SETMETHOD-STRING      (R#2)
    setMethod("name", ...) with a string-literal first arg. Must pass the
    generic function object.

  R-ENV-ASNAMESPACE       (former footgun #2)
    `environment(fn) <- asNamespace(pkg)` rebinds the closure to upstream's
    locked namespace; byte compiler refuses to scan it.

  R-PIPE-IN-MCLAPPLY      (former footgun #3)
    Patch uses `%>%` (or `%||%`, `%<>%`, `%+%`) inside an mclapply / future
    worker without binding the operator at file scope. Forked workers can't
    see autozyme's imports.

  PY-NUMBA-TF-GUARD       (Py#1)
    `import numba` without preceding `os.environ.setdefault("NUMBA_NUM_THREADS",
    "1")` guarded by `if "tensorflow" in sys.modules:`.

  PY-CACHE-CLEAR-SCOPE    (Py#2)
    `<x>.cache.clear()` / `<x>._ENTRY_POINT_CACHE.clear()` at module scope
    instead of inside a function body.

Still deferred (out of scope here):

  R#5 stale-API shim — requires a deprecation registry that maps upstream/version
       to the set of defunct APIs the patch would need to shim. Not lintable
       without that registry.
  R#6 formula-LHS globalenv — requires AST flow analysis to know which symbols
       a formula references and whether they reach the smoke call frame.
  Py#3 loky+numba doc-only — heuristic detector for the combo; the "fix" is a
       README note, not code, so a lint warning is the wrong shape.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# data types
# ---------------------------------------------------------------------------

# Severity: FAIL blocks attest (exit nonzero from `zyme package lint`); WARN
# prints but doesn't block. Keep this list small — most rules should be FAIL.
SEVERITIES = ("FAIL", "WARN")


@dataclass
class LintFinding:
    rule_id: str
    severity: str
    file: Path
    line: int
    message: str

    def format_one(self) -> str:
        return f"[{self.severity}] {self.rule_id} {self.file}:{self.line}: {self.message}"


@dataclass
class LintContext:
    """Inputs available to every rule.

    Both R and Python rules receive the same context shape. Rules that don't
    apply to a given language inspect ``language`` and bail.
    """
    patch_name: str
    patch_file: Path                   # patch.R or __init__.py
    patch_text: str
    language: str                      # "R" or "py"
    # R-only package-level paths (set by the dispatcher when scanning R patches)
    description_path: Path | None = None
    namespace_path: Path | None = None
    description_text: str = ""
    namespace_text: str = ""
    # Python-only: parsed AST (cached so multiple Python rules can reuse).
    py_ast: ast.Module | None = None


# ---------------------------------------------------------------------------
# R rules
# ---------------------------------------------------------------------------

# Upstream packages whose reflection-on-attach behavior demands library() in
# smoke load — covers R#1 (MAST callName), R#3 (RCTD S4 class cache), R#8 (WGCNA
# cor() override). Add new packages here as we discover them; the fix is the
# same shape every time.
_LIBRARY_GUARD_PKGS = ("MAST", "spacexr", "WGCNA")


def _smoke_load_block(text: str) -> tuple[str, int] | None:
    """Extract the body of ``smoke = list(load = function(...) { ... })``.

    R doesn't have a real AST library here. We use a brace-balanced extractor
    starting at ``load = function``. Returns (block_text, start_line) or None
    when no smoke load is found. The start_line is 1-indexed so findings can
    cite roughly the right line.
    """
    m = re.search(r"load\s*=\s*function\s*\([^)]*\)\s*\{", text)
    if not m:
        return None
    start = m.end()  # first char after the opening brace
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    start_line = text.count("\n", 0, m.start()) + 1
    return text[start:i - 1], start_line


def rule_r_library_guard(ctx: LintContext) -> list[LintFinding]:
    """For each reflection-sensitive upstream we know about, the patch file
    must include ``library(<pkg>)`` somewhere — top of file, inside a named
    smoke load helper, or inline in ``load = function``. Where it lives
    doesn't matter as long as it runs before the timed region. The trap is
    only ever ``requireNamespace()`` alone.
    """
    if ctx.language != "R":
        return []
    findings: list[LintFinding] = []
    text = ctx.patch_text
    for pkg in _LIBRARY_GUARD_PKGS:
        touched = (
            re.search(rf"\b{pkg}::", text) is not None
            or re.search(rf'requireNamespace\(\s*["\']{pkg}["\']', text) is not None
            or re.search(rf'requireNamespace\(\s*{pkg}\b', text) is not None
        )
        if not touched:
            continue
        attached = (
            re.search(rf'library\(\s*{pkg}\b', text) is not None
            or re.search(rf'library\(\s*["\']{pkg}["\']', text) is not None
        )
        if attached:
            continue
        findings.append(LintFinding(
            rule_id="R-LIBRARY-GUARD",
            severity="FAIL",
            file=ctx.patch_file,
            line=1,
            message=(
                f"patch touches {pkg!r} but never calls library({pkg}). "
                f"{pkg} uses reflection on the search path / S4 cache / call "
                f"form; requireNamespace() is not enough. Add "
                f"suppressPackageStartupMessages(library({pkg})) inside "
                f"smoke load."
            ),
        ))
    return findings


def rule_r_rcpp_inline(ctx: LintContext) -> list[LintFinding]:
    if ctx.language != "R":
        return []
    findings: list[LintFinding] = []
    for m in re.finditer(r"Rcpp\s*::\s*cppFunction\s*\(", ctx.patch_text):
        line = ctx.patch_text.count("\n", 0, m.start()) + 1
        findings.append(LintFinding(
            rule_id="R-RCPP-INLINE",
            severity="FAIL",
            file=ctx.patch_file,
            line=line,
            message=(
                "Rcpp::cppFunction() at file scope triggers GC corruption "
                "after activation; move the kernel to autozyme_r/src/<name>.cpp "
                "and use tools/scaffold_cpp_patch.R."
            ),
        ))
    for m in re.finditer(r"sourceCpp\s*\(\s*code\s*=", ctx.patch_text):
        line = ctx.patch_text.count("\n", 0, m.start()) + 1
        findings.append(LintFinding(
            rule_id="R-RCPP-INLINE",
            severity="FAIL",
            file=ctx.patch_file,
            line=line,
            message=(
                "sourceCpp(code=...) inline compile in a patch file. "
                "Move kernel to autozyme_r/src/<name>.cpp."
            ),
        ))
    return findings


def rule_r_setmethod_string(ctx: LintContext) -> list[LintFinding]:
    if ctx.language != "R":
        return []
    findings: list[LintFinding] = []
    # First arg must NOT be a string literal. Allow getFromNamespace / getMethod
    # / a bare identifier (function object). Examples to flag:
    #   setMethod("getCurves", sig, fn)
    #   setMethod( "fit"  , "Class", fast )
    # Examples to accept:
    #   setMethod(getCurves, sig, fn)
    #   setMethod(utils::getFromNamespace("getCurves", "slingshot"), sig, fn)
    for m in re.finditer(r'setMethod\s*\(\s*"([^"]+)"', ctx.patch_text):
        name = m.group(1)
        line = ctx.patch_text.count("\n", 0, m.start()) + 1
        findings.append(LintFinding(
            rule_id="R-SETMETHOD-STRING",
            severity="FAIL",
            file=ctx.patch_file,
            line=line,
            message=(
                f'setMethod("{name}", ...) uses a string literal — autozyme '
                f'namespace does not import upstream S4 generics, so name-'
                f'based lookup fails. Pass the generic function object via '
                f'utils::getFromNamespace("{name}", "<pkg>") instead.'
            ),
        ))
    return findings


def rule_r_datatable_ns(ctx: LintContext) -> list[LintFinding]:
    """Package-level: any patch using `:=` requires data.table in DESCRIPTION
    Imports AND `importFrom(data.table, ":=")` in NAMESPACE. We fire one
    finding per package per missing piece (not per patch) — the dispatcher
    dedupes by stashing the finding under the first patch that triggers it.
    """
    if ctx.language != "R":
        return []
    # Per-patch trigger: does this patch use `:=`?
    uses_assign = re.search(r":=", ctx.patch_text) is not None
    if not uses_assign:
        return []
    findings: list[LintFinding] = []
    desc = ctx.description_text or ""
    nsfile = ctx.namespace_text or ""
    # DESCRIPTION must list data.table in Imports.
    desc_imports = re.search(r"^Imports:[\s\S]*?(?=\n[A-Z]\w*:|\Z)", desc, re.MULTILINE)
    desc_has_dt = bool(desc_imports and re.search(r"\bdata\.table\b", desc_imports.group(0)))
    ns_has_import = re.search(
        r'importFrom\(\s*data\.table\s*,\s*"?:="?\s*\)', nsfile
    ) is not None
    if not desc_has_dt:
        findings.append(LintFinding(
            rule_id="R-DATATABLE-NS",
            severity="FAIL",
            file=ctx.description_path or ctx.patch_file,
            line=1,
            message=(
                f"patch {ctx.patch_name!r} uses data.table `:=` but DESCRIPTION "
                f"Imports lacks `data.table`. data.table's `[.data.table` "
                f"checks cedta(); without `data.table` in Imports the autozyme "
                f"namespace is not data.table-aware."
            ),
        ))
    if not ns_has_import:
        findings.append(LintFinding(
            rule_id="R-DATATABLE-NS",
            severity="FAIL",
            file=ctx.namespace_path or ctx.patch_file,
            line=1,
            message=(
                f"patch {ctx.patch_name!r} uses data.table `:=` but NAMESPACE "
                f'lacks importFrom(data.table, ":="). bare Imports: entry is '
                f"not enough — only importFrom() registers data.table in "
                f"getNamespaceImports()."
            ),
        ))
    return findings


def rule_r_env_asnamespace(ctx: LintContext) -> list[LintFinding]:
    """``environment(fn) <- asNamespace(pkg)`` rebinds the closure environment
    of an autozyme function to upstream's namespace. The byte compiler then
    refuses to scan it and patch activation crashes. Pattern is unambiguous;
    no false-positive shape exists in practice.
    """
    if ctx.language != "R":
        return []
    findings: list[LintFinding] = []
    for m in re.finditer(
        r"environment\s*\(\s*[A-Za-z_.][\w.]*\s*\)\s*<-\s*asNamespace\s*\(",
        ctx.patch_text,
    ):
        line = ctx.patch_text.count("\n", 0, m.start()) + 1
        findings.append(LintFinding(
            rule_id="R-ENV-ASNAMESPACE",
            severity="FAIL",
            file=ctx.patch_file,
            line=line,
            message=(
                "`environment(fn) <- asNamespace(pkg)` rebinds the closure "
                "to upstream's locked namespace; the byte compiler refuses to "
                "scan it. Capture upstream helpers at file scope via "
                "`getFromNamespace(\"helper\", \"pkg\")` instead."
            ),
        ))
    return findings


# Non-base operators that get lexically captured by mclapply worker closures
# only when bound at file scope. dplyr/magrittr's namespace is not visible to
# forked workers under autozyme's namespace.
_PIPE_OPERATORS = ("%>%", "%||%", "%<>%", "%+%")


def rule_r_pipe_in_mclapply(ctx: LintContext) -> list[LintFinding]:
    """If the patch calls ``mclapply`` / ``mcparallel`` / ``future_map`` AND
    uses a non-base operator like ``%>%``, that operator must be bound at
    file scope (e.g. `` `%>%` <- dplyr::`%>%` ``) so the forked worker can
    resolve it lexically. autozyme's namespace doesn't re-export pipes; the
    closure sent into the worker can't reach them via search-path lookup.
    """
    if ctx.language != "R":
        return []
    text = ctx.patch_text
    uses_parallel = re.search(
        r"\b(?:parallel\s*::\s*)?(?:mclapply|mcparallel|mcMap)\b|"
        r"\bfuture\s*::\s*(?:future_map|future_lapply)\b|"
        r"\bfuture_map\b",
        text,
    ) is not None
    if not uses_parallel:
        return []
    findings: list[LintFinding] = []
    for op in _PIPE_OPERATORS:
        if op not in text:
            continue
        # File-scope binding: backtick or bare assignment to the operator
        # name from a known pkg (dplyr/magrittr/rlang/data.table).
        # E.g.   `%>%` <- dplyr::`%>%`
        bind_pat = re.compile(
            rf"`?{re.escape(op)}`?\s*<-\s*"
            rf"(?:dplyr|magrittr|rlang|data\.table|purrr)\s*::\s*`?{re.escape(op)}`?"
        )
        if bind_pat.search(text):
            continue
        findings.append(LintFinding(
            rule_id="R-PIPE-IN-MCLAPPLY",
            severity="FAIL",
            file=ctx.patch_file,
            line=1,
            message=(
                f"patch uses ``{op}`` inside a forked-worker context "
                f"(mclapply / mcparallel / future_map) but the operator is "
                f"not bound at file scope. Forked workers can't see "
                f"autozyme's imports — add `` `{op}` <- dplyr::`{op}` `` "
                f"(or magrittr/rlang/data.table source) at file scope so the "
                f"closure captures it lexically."
            ),
        ))
    return findings


# ---------------------------------------------------------------------------
# Python rules
# ---------------------------------------------------------------------------

def _ast_for(ctx: LintContext) -> ast.Module | None:
    if ctx.language != "py":
        return None
    if ctx.py_ast is not None:
        return ctx.py_ast
    try:
        ctx.py_ast = ast.parse(ctx.patch_text)
    except SyntaxError:
        return None
    return ctx.py_ast


def _walk_module_level(mod: ast.Module) -> Iterator[ast.stmt]:
    """Yield top-level statements only — do NOT descend into FunctionDef etc."""
    yield from mod.body


def _is_numba_threads_setdefault(stmt: ast.stmt) -> bool:
    """``os.environ.setdefault("NUMBA_NUM_THREADS", ...)`` — expression statement."""
    if not isinstance(stmt, ast.Expr):
        return False
    call = stmt.value
    if not isinstance(call, ast.Call):
        return False
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "setdefault":
        return False
    target = call.func.value
    # os.environ
    if not (
        isinstance(target, ast.Attribute)
        and target.attr == "environ"
        and isinstance(target.value, ast.Name)
        and target.value.id == "os"
    ):
        return False
    if not call.args:
        return False
    key = call.args[0]
    return isinstance(key, ast.Constant) and key.value == "NUMBA_NUM_THREADS"


def _is_tf_in_sysmodules_test(test: ast.expr) -> bool:
    """``"tensorflow" in sys.modules`` — the guard condition."""
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.In):
        return False
    left = test.left
    right = test.comparators[0]
    left_ok = isinstance(left, ast.Constant) and left.value == "tensorflow"
    right_ok = (
        isinstance(right, ast.Attribute)
        and right.attr == "modules"
        and isinstance(right.value, ast.Name)
        and right.value.id == "sys"
    )
    return left_ok and right_ok


def _uses_parallel_numba(mod: ast.Module) -> bool:
    """The TF deadlock is triggered by parallel numba — either ``parallel=True``
    inside an ``@njit`` decorator or ``prange``. njit alone (single-thread) is
    safe, so we only fire the guard rule when the patch actually opts into the
    risky pattern.
    """
    for node in ast.walk(mod):
        # @njit(parallel=True) or @numba.jit(parallel=True)
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else (
                func.id if isinstance(func, ast.Name) else None
            )
            if name in ("njit", "jit"):
                for kw in node.keywords:
                    if (
                        kw.arg == "parallel"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        return True
        # prange import or use
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("numba"):
            if any(a.name == "prange" for a in node.names):
                return True
        if isinstance(node, ast.Name) and node.id == "prange":
            return True
    return False


def rule_py_numba_tf_guard(ctx: LintContext) -> list[LintFinding]:
    mod = _ast_for(ctx)
    if mod is None:
        return []
    # Only flag when the patch uses parallel numba — single-thread @njit
    # doesn't hit the TBB/TF deadlock.
    if not _uses_parallel_numba(mod):
        return []
    # Find the first top-level `import numba` (or `from numba import ...`).
    numba_import_line: int | None = None
    for stmt in _walk_module_level(mod):
        if isinstance(stmt, ast.Import) and any(a.name == "numba" or a.name.startswith("numba.") for a in stmt.names):
            numba_import_line = stmt.lineno
            break
        if isinstance(stmt, ast.ImportFrom) and (stmt.module or "").startswith("numba"):
            numba_import_line = stmt.lineno
            break
    if numba_import_line is None:
        return []
    # Walk top-level statements that precede the numba import; look for the
    # guarded setdefault. The guard can be:
    #   if "tensorflow" in sys.modules:
    #       <maybe other stmts>
    #       os.environ.setdefault("NUMBA_NUM_THREADS", ...)
    # Either the if-body's direct stmts or any nested block under it counts.
    has_guard = False
    for stmt in _walk_module_level(mod):
        if stmt.lineno >= numba_import_line:
            break
        if isinstance(stmt, ast.If) and _is_tf_in_sysmodules_test(stmt.test):
            for inner in ast.walk(stmt):
                if isinstance(inner, ast.stmt) and _is_numba_threads_setdefault(inner):
                    has_guard = True
                    break
        if has_guard:
            break
    if has_guard:
        return []
    return [LintFinding(
        rule_id="PY-NUMBA-TF-GUARD",
        severity="FAIL",
        file=ctx.patch_file,
        line=numba_import_line,
        message=(
            "`import numba` runs without a preceding "
            '`if "tensorflow" in sys.modules: '
            'os.environ.setdefault("NUMBA_NUM_THREADS", "1")` guard. '
            "Co-loading TF + parallel numba deadlocks the TBB pool. "
            "Add the guard before the numba import."
        ),
    )]


def rule_py_cache_clear_scope(ctx: LintContext) -> list[LintFinding]:
    mod = _ast_for(ctx)
    if mod is None:
        return []
    findings: list[LintFinding] = []
    # Look for any top-level expression that calls `<x>.clear()` where x looks
    # like an entry-point / lru cache. The trap is doing this at import time;
    # the fix is doing it inside the fast fn.
    for stmt in _walk_module_level(mod):
        if not isinstance(stmt, ast.Expr):
            continue
        call = stmt.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
            continue
        if call.func.attr not in ("clear", "cache_clear"):
            continue
        # Heuristic: the receiver looks cache-y — _ENTRY_POINT_CACHE, *_cache,
        # lru_cache, or anything ending in _CACHE.
        receiver = call.func.value
        receiver_name = _attr_chain_tail(receiver)
        if not receiver_name:
            continue
        looks_cachey = (
            receiver_name.endswith("_CACHE")
            or receiver_name.endswith("_cache")
            or receiver_name == "lru_cache"
        )
        if not looks_cachey:
            continue
        findings.append(LintFinding(
            rule_id="PY-CACHE-CLEAR-SCOPE",
            severity="WARN",
            file=ctx.patch_file,
            line=stmt.lineno,
            message=(
                f"{receiver_name}.{call.func.attr}() runs at module/import "
                f"time. Upstream entry-point caches must be cleared inside the "
                f"fast function (each fast-call), not at register time — "
                f"`verify_patch` toggles restore→activate within one process."
            ),
        ))
    return findings


def _attr_chain_tail(node: ast.expr) -> str | None:
    """For `a.b.c` return 'c'; for `_CACHE` return '_CACHE'."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


# ---------------------------------------------------------------------------
# rule registry
# ---------------------------------------------------------------------------

R_RULES = (
    rule_r_library_guard,
    rule_r_rcpp_inline,
    rule_r_setmethod_string,
    rule_r_datatable_ns,
    rule_r_env_asnamespace,
    rule_r_pipe_in_mclapply,
)

PY_RULES = (
    rule_py_numba_tf_guard,
    rule_py_cache_clear_scope,
)
