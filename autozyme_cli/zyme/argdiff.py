"""Target-call kwarg diff between reference.{py,R} and pipeline/run.{py,R}.

Layer C-companion: detects when pipeline passes API-level kwargs to the
target function that differ from baseline's call (e.g., pipeline switches
`counts_file_path` from `.tsv` to `.pickle`). Such divergence requires
the baseline to be re-recorded at the new kwarg value before claiming
speedup — see 2_iterate.md "API-level kwarg" red line.

Threading-related kwargs are excluded (they have their own discipline via
`upstream_parallelism` + `baseline_threads`).

Python-only AST parsing for now; R support deferred.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path


# Kwargs that have their own re-record discipline (threading); skip them here.
_THREAD_KWARGS = frozenset({
    "threads", "n_threads", "nthreads", "n_jobs", "n_workers",
    "num_workers", "num_threads", "workers", "parallel", "BPPARAM",
    "mc.cores", "cores", "ncores", "n_cores",
})


# Kwargs that legitimately differ between reference and pipeline by framework
# convention — output destinations, log paths, suffix tags. Skipping these
# avoids the most common false-positive class.
_FRAMEWORK_OUTPUT_KWARGS = frozenset({
    "output_path", "output_dir", "output_suffix", "output_prefix",
    "output_urlpath", "output_filename",
    "out_path", "out_dir", "out_file",
    "save_path", "save_dir",
    "log_path", "log_dir", "log_file",
    "result_dir", "results_dir",
    "verbose",
})


_QUOTED_STR = re.compile(r"""['"]([^'"]*)['"]""")
_PATH_CTOR = re.compile(r"\b(Path|os\.path\.(join|abspath|dirname)|__file__|parent)\b")
_EXT_RE = re.compile(r"(\.[A-Za-z][A-Za-z0-9]{0,6})$")


def _looks_like_path_expr(value: str) -> bool:
    """Heuristic: does this resolved expression look like a filesystem path?

    Triggers on Path() / os.path.* / __file__ constructions, regardless of
    the kwarg name.
    """
    return bool(_PATH_CTOR.search(value))


def _path_signature(value: str) -> str:
    """Reduce a path-expression source to its semantic identity for comparison.

    Strategy: scan all quoted string literals; return the file extension of
    the last one that has one (e.g. `.tsv`, `.pickle`, `.nc`, `.parquet`).
    This collapses cosmetic differences — f-string `f'tas_{TIER}.nc'` vs
    hardcoded `'data/tas_tiny.nc'` both reduce to `.nc` — while a true
    format swap (`.tsv` → `.pickle`) still differs.

    Falls back to the last quoted string when no extension is found, and to
    verbatim when there are no quoted strings at all (e.g., bare variable
    references that didn't resolve).
    """
    strings = _QUOTED_STR.findall(value)
    if not strings:
        return value
    for s in reversed(strings):
        m = _EXT_RE.search(s)
        if m:
            return m.group(1).lower()
    return strings[-1]


def _normalize_for_diff(name: str, value: str | None) -> str | None:
    """Path-like expressions reduce to extension/basename signature; others
    compare verbatim. See `_path_signature` for the path semantics.
    """
    if value is None:
        return None
    if _looks_like_path_expr(value):
        return _path_signature(value)
    return value


def diff_target_call_kwargs(
    task_dir: Path,
    target_function: str,
    upstream_parallelism: list[str] | None = None,
) -> list[dict]:
    """Compare target-function call kwargs in reference.py vs pipeline/run.py.

    Returns a list of divergence records. Empty list means no divergence
    (or one of the files is missing / target not found in one of them).

    Each divergence record:
      {"kwarg": <name>, "ref": <resolved-source>, "pipe": <resolved-source>}
    """
    ref_path = task_dir / "reference.py"
    pipe_path = task_dir / "pipeline" / "run.py"
    # Only Python for now. If either file is .R or missing, return [].
    if not (ref_path.exists() and pipe_path.exists()):
        return []
    target_name = _target_short_name(target_function)
    if not target_name:
        return []

    ref_kwargs = _extract_call_kwargs(ref_path, target_name)
    pipe_kwargs = _extract_call_kwargs(pipe_path, target_name)
    if ref_kwargs is None or pipe_kwargs is None:
        return []  # target call not found in one of the files

    skip = set(_THREAD_KWARGS) | set(_FRAMEWORK_OUTPUT_KWARGS)
    if upstream_parallelism:
        skip.update(upstream_parallelism)

    divergences = []
    all_keys = set(ref_kwargs) | set(pipe_kwargs)
    for k in sorted(all_keys):
        if k in skip:
            continue
        rv_norm = _normalize_for_diff(k, ref_kwargs.get(k))
        pv_norm = _normalize_for_diff(k, pipe_kwargs.get(k))
        if rv_norm != pv_norm:
            divergences.append({
                "kwarg": k,
                "ref": ref_kwargs.get(k),
                "pipe": pipe_kwargs.get(k),
            })
    return divergences


def check_target_prewarm(task_dir: Path, target_function: str) -> str | None:
    """Detect target-call BEFORE the timer in pipeline/run.py with args
    matching an in-timer call (xclim-style pre-warm hack).

    A legitimate JIT warmup using a synthetic dummy input (`fast_target(dummy)`)
    won't match the production args and is safely ignored.

    Returns: actionable warning string, or None.
    Python-only (R deferred).
    """
    pipe_path = task_dir / "pipeline" / "run.py"
    if not pipe_path.exists():
        return None
    target_short = _target_short_name(target_function)
    if not target_short:
        return None
    try:
        source = pipe_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None

    timer_lineno = _find_timer_lineno(tree)
    if timer_lineno is None:
        return None

    # Collect override-alias names from `install_override("<target_short_tail>", ..., <alias>)`.
    # `xclim` calls fast_growing_season_length() at module level (the alias),
    # and growing_season_length() inside the timer (the overridden name). Both
    # must be matched.
    target_chain = target_short.split(".")
    target_tail = target_chain[-1]
    override_aliases: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "install_override"
                and len(node.args) >= 3):
            first = node.args[0]
            third = node.args[2]
            if (isinstance(first, ast.Constant)
                    and isinstance(first.value, str)
                    and first.value == target_tail):
                if isinstance(third, ast.Name):
                    override_aliases.add(third.id)
                elif isinstance(third, ast.Attribute):
                    chain = _call_name_chain(third)
                    if chain:
                        override_aliases.add(chain[-1])

    # Find every target-or-alias call at MODULE EXECUTION level (skip calls
    # inside `def` / `class` bodies — those are only invoked when their
    # enclosing function is called, not at module load).
    outside_calls: list[ast.Call] = []
    inside_calls: list[ast.Call] = []

    def _scan_for_calls(stmts):
        for stmt in stmts:
            # Skip function and class bodies — calls inside them aren't
            # executed at module load (they fire only when the function/class
            # is called/instantiated).
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            # Descend into nested control flow that DOES execute at module level.
            if isinstance(stmt, ast.If):
                _scan_for_calls(stmt.body)
                _scan_for_calls(stmt.orelse)
                continue
            if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                _scan_for_calls(stmt.body)
                _scan_for_calls(stmt.orelse)
                continue
            if isinstance(stmt, (ast.With, ast.AsyncWith)):
                _scan_for_calls(stmt.body)
                continue
            if isinstance(stmt, ast.Try):
                _scan_for_calls(stmt.body)
                for h in stmt.handlers:
                    _scan_for_calls(h.body)
                _scan_for_calls(stmt.orelse)
                _scan_for_calls(stmt.finalbody)
                continue
            # Otherwise, ast.walk only the inside of this statement (no
            # further FunctionDef descent because they were filtered above).
            for sub in ast.walk(stmt):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Defensive: skip in case ast.walk re-enters
                    continue
                if not isinstance(sub, ast.Call):
                    continue
                chain = _call_name_chain(sub.func)
                if not chain:
                    continue
                is_target = (len(chain) >= len(target_chain)
                             and chain[-len(target_chain):] == target_chain)
                is_alias = chain[-1] in override_aliases
                if not (is_target or is_alias):
                    continue
                if sub.lineno < timer_lineno:
                    outside_calls.append(sub)
                else:
                    inside_calls.append(sub)

    _scan_for_calls(tree.body)

    if not outside_calls or not inside_calls:
        return None

    module_assigns = _collect_module_assigns(tree)

    def _call_args(call: ast.Call) -> dict[str, str]:
        d = {}
        for kw in call.keywords:
            if kw.arg is None:
                d[f"<kwargs:{ast.unparse(kw.value)}>"] = ast.unparse(kw.value)
            else:
                d[kw.arg] = _resolved_unparse(kw.value, module_assigns)
        for i, arg in enumerate(call.args):
            d[f"<pos{i}>"] = _resolved_unparse(arg, module_assigns)
        return d

    inside_arg_sets = [_call_args(c) for c in inside_calls]

    for out_call in outside_calls:
        out_args = _call_args(out_call)
        for in_args in inside_arg_sets:
            if _args_match(out_args, in_args):
                first_inside = inside_calls[0]
                return _format_prewarm(
                    pipe_path, out_call.lineno, first_inside.lineno,
                    out_args, target_function,
                )
    return None


def _args_match(out_args: dict[str, str], in_args: dict[str, str]) -> bool:
    """Compare two arg dicts using the same path-like normalization as kwarg diff.

    Match = same set of keys AND same normalized value per key AND at least
    one arg present. Empty-args matches (`target()` outside vs `target()`
    inside) are degenerate — they carry no input-identity signal and would
    fire on every method-call-with-no-args. Skip those.
    """
    if not out_args or not in_args:
        return False
    if set(out_args.keys()) != set(in_args.keys()):
        return False
    for k, v_out in out_args.items():
        v_in = in_args[k]
        if _normalize_for_diff(k, v_out) != _normalize_for_diff(k, v_in):
            return False
    return True


_TIMER_NAMES = frozenset({"perf_counter", "process_time", "monotonic", "time"})


def _find_timer_lineno(tree: ast.Module) -> int | None:
    """First call to `time.perf_counter()` (or peer) — the timer boundary line."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _call_name_chain(node.func)
        if not chain:
            continue
        tail = chain[-1]
        if tail in _TIMER_NAMES:
            # Must be qualified by `time.` (or unambiguous like `perf_counter`)
            if tail == "perf_counter":
                return node.lineno
            if len(chain) >= 2 and chain[-2] == "time":
                return node.lineno
    return None


def _format_prewarm(
    pipe_path: Path, outside_line: int, inside_line: int,
    args: dict[str, str], target_function: str,
) -> str:
    arg_preview = ", ".join(
        f"{k}={v[:50] + '…' if v and len(v) > 50 else v}"
        for k, v in list(args.items())[:4]
    )
    return (
        f"\nPre-warm pattern detected in {pipe_path}:\n"
        f"\n"
        f"  line {outside_line}: target called with production args "
        f"BEFORE timer starts\n"
        f"  line {inside_line}: same args called INSIDE timer\n"
        f"\n"
        f"  matched args: {arg_preview}\n"
        f"\n"
        f"This warms input-specific state (page tables, cache, JIT-on-real-data)\n"
        f"that real users can't pre-pay for — the timed call benefits from work\n"
        f"the reference doesn't get. Either:\n"
        f"  - Drop the pre-warm line and let JIT/caches warm inside the timer\n"
        f"    (reference does it cold; pipeline should too).\n"
        f"  - If JIT compile cost is genuinely amortizable, warm on a SYNTHETIC\n"
        f"    tiny dummy (`{target_function.split('::')[-1].split('.')[-1]}(small_dummy, ...)`)\n"
        f"    before the timer — the AST check ignores those because the args differ.\n"
        f"\n"
        f"See `2_iterate.md` → 'no in-memory timer-gaming' rule."
    )


# ----------------------------------------------------------------------------
# Side-channel parallelism detection
# ----------------------------------------------------------------------------
#
# argdiff's kwarg check only sees what's passed *to* the target call. Pipelines
# can engage parallelism through orthogonal channels — `numba.set_num_threads()`,
# `os.environ['OMP_NUM_THREADS']`, `torch.set_num_threads()` — that don't show
# up in any kwarg diff. If reference.py is serial and pipeline does this, the
# baseline is unfair in the same way as an unmatched kwarg.
#
# Python-only (R deferred — same scope as the rest of argdiff).

_SIDE_CHANNEL_ENV_KEYS = frozenset({
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS", "GOTO_NUM_THREADS",
})

_SIDE_CHANNEL_CALL_TAILS = frozenset({
    "set_num_threads",   # numba, torch
    "setNumThreads",     # cv2
})


def _scan_side_channel_setters(file_path: Path) -> list[dict]:
    """AST-walk a .py file for thread-count setters that don't go through the
    target call's kwargs.

    Each record: {"kind": "call"|"env", "symbol": <name>, "value": <expr>, "lineno": <int>}.
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []

    found: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = _call_name_chain(node.func)
            if chain and chain[-1] in _SIDE_CHANNEL_CALL_TAILS:
                arg_repr = ast.unparse(node.args[0]) if node.args else ""
                found.append({
                    "kind": "call",
                    "symbol": ".".join(chain),
                    "value": arg_repr,
                    "lineno": node.lineno,
                })
                continue
            # `os.environ.setdefault("OMP_NUM_THREADS", "N")` /
            # `os.environ.__setitem__(...)`
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("setdefault", "__setitem__"):
                base = node.func.value
                if (isinstance(base, ast.Attribute) and base.attr == "environ"
                        and len(node.args) >= 2
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                        and node.args[0].value in _SIDE_CHANNEL_ENV_KEYS):
                    found.append({
                        "kind": "env",
                        "symbol": node.args[0].value,
                        "value": ast.unparse(node.args[1]),
                        "lineno": node.lineno,
                    })
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Attribute)
                        and tgt.value.attr == "environ"
                        and isinstance(tgt.slice, ast.Constant)
                        and isinstance(tgt.slice.value, str)
                        and tgt.slice.value in _SIDE_CHANNEL_ENV_KEYS):
                    found.append({
                        "kind": "env",
                        "symbol": tgt.slice.value,
                        "value": ast.unparse(node.value),
                        "lineno": node.lineno,
                    })
    return found


def diff_side_channel_parallelism(task_dir: Path) -> list[dict]:
    """Detect thread-count setters present in pipeline/run.py but not mirrored
    in reference.py. Returns one record per pipeline-only setter.

    Asymmetry = pipeline gets parallelism that the single-threaded reference
    doesn't engage, so speedup mixes "algorithmic gain" with "raw parallelism
    upstream's default doesn't have." Warning, not gate (see M_thread_baseline_
    fairness.md's outcome-B carve-out for the legitimate case).
    """
    ref_path = task_dir / "reference.py"
    pipe_path = task_dir / "pipeline" / "run.py"
    if not (ref_path.exists() and pipe_path.exists()):
        return []
    ref_keys = {(s["kind"], s["symbol"]) for s in _scan_side_channel_setters(ref_path)}
    return [s for s in _scan_side_channel_setters(pipe_path)
            if (s["kind"], s["symbol"]) not in ref_keys]


def format_side_channel_divergences(divergences: list[dict]) -> str:
    """Pretty-print side-channel divergences as a runner warning."""
    lines = [
        "",
        "Side-channel parallelism in pipeline/run.py not mirrored in reference.py:",
        "",
    ]
    for d in divergences:
        if d["kind"] == "call":
            lines.append(f"  line {d['lineno']}: {d['symbol']}({d['value']})")
        else:
            lines.append(f"  line {d['lineno']}: os.environ[{d['symbol']!r}] = {d['value']}")
    lines.extend([
        "",
        "These bypass argdiff's kwarg check — they don't appear in the target call's",
        "signature, but they DO change how much CPU the timed region uses. If the",
        "speedup partly comes from this added parallelism, baseline is unfair.",
        "",
        "Triage (see M_thread_baseline_fairness.md):",
        "  - Outcome A — upstream exposes an equivalent knob: mirror the setter in",
        "    reference.py and `zyme baseline reference --tier <t> --thread <N> --force`.",
        "  - Outcome B — upstream lacks an equivalent: leave reference serial; document",
        "    in memory/discoveries.md that the gain includes a parallel layer upstream lacks.",
    ])
    return "\n".join(lines)


def format_divergences(divergences: list[dict], target_function: str) -> str:
    """Pretty-print divergences as a warning string suitable for the runner."""
    lines = [
        "",
        f"API-level kwarg divergence on `{target_function}` between reference and pipeline:",
        "",
    ]
    for d in divergences:
        lines.append(f"  {d['kwarg']}:")
        lines.append(f"    reference     = {d['ref']}")
        lines.append(f"    pipeline      = {d['pipe']}")
        lines.append("")
    lines.extend([
        "This means the target call doesn't compare apples-to-apples. If the kwarg",
        "is something upstream natively supports (input format, batch size,",
        "algorithm mode), re-record the baseline at pipeline's value first:",
        "  1. Edit reference.{py,R} to mirror pipeline's kwarg value.",
        "  2. Run `zyme baseline reference --tier <t> --force` to overwrite the",
        "     prior baseline at the new kwarg.",
        "  3. Re-run this round to get an honest speedup against the new baseline.",
        "",
        "See `2_iterate.md` → API-level kwarg red line.",
    ])
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# AST helpers
# ----------------------------------------------------------------------------

def _target_short_name(target_function: str) -> str:
    """Pull the call-site short name from a target spec.

    Examples:
      'cellphonedb.src.core.methods.cpdb_statistical_analysis_method::call' -> 'call'
      'spacexr::run.RCTD'                                                   -> 'run.RCTD'
      'MDAnalysis.analysis.rms::RMSD.run'                                   -> 'RMSD.run'
      'np.linalg::svd'                                                       -> 'svd'
    """
    if not target_function:
        return ""
    s = target_function.strip()
    # Split on '::' first if present, else last '.'
    if "::" in s:
        return s.rsplit("::", 1)[1].strip()
    return s.rsplit(".", 1)[-1].strip()


def _extract_call_kwargs(file_path: Path, target_short: str) -> dict[str, str] | None:
    """Find first ast.Call whose attribute/name chain ends with target_short,
    return its kwargs resolved (one or two levels of variable substitution).

    Returns None if no match was found.
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None

    module_assigns = _collect_module_assigns(tree)
    target_chain = target_short.split(".")  # for "RMSD.run" → ["RMSD", "run"]

    matching = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _call_name_chain(node.func)
        if not chain:
            continue
        # Match: pipeline.chain ends with target_chain
        # e.g. ["cpdb_statistical_analysis_method", "call"] ends with ["call"]
        if len(chain) >= len(target_chain) and chain[-len(target_chain):] == target_chain:
            matching = node
            break
    if matching is None:
        return None

    out: dict[str, str] = {}
    for kw in matching.keywords:
        if kw.arg is None:
            continue  # **kwargs splat — skip
        out[kw.arg] = _resolved_unparse(kw.value, module_assigns)
    # Positional args keyed by index (less common but capture them too).
    for i, arg in enumerate(matching.args):
        out[f"<pos{i}>"] = _resolved_unparse(arg, module_assigns)
    return out


def _call_name_chain(func: ast.expr) -> list[str]:
    """Turn `a.b.c` AST attribute chain into ['a','b','c']; `f` into ['f']."""
    if isinstance(func, ast.Name):
        return [func.id]
    if isinstance(func, ast.Attribute):
        base = _call_name_chain(func.value)
        if base is None:
            return []
        return base + [func.attr]
    return []


def _collect_module_assigns(tree: ast.Module) -> dict[str, ast.expr]:
    """Walk module-level + obvious top-level `if` bodies, map Name -> value.

    Only simple `name = value` assignments. Last assignment wins (later in file).
    """
    out: dict[str, ast.expr] = {}
    def _walk_body(stmts):
        for node in stmts:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        out[tgt.id] = node.value
            elif isinstance(node, ast.If):
                _walk_body(node.body)
                _walk_body(node.orelse)
            elif isinstance(node, ast.Try):
                _walk_body(node.body)
                _walk_body(node.handlers)
                _walk_body(node.orelse)
                _walk_body(node.finalbody)
            elif isinstance(node, ast.ExceptHandler):
                _walk_body(node.body)
            elif isinstance(node, ast.With):
                _walk_body(node.body)
    _walk_body(tree.body)
    return out


def _resolved_unparse(
    node: ast.expr,
    module_assigns: dict[str, ast.expr],
    max_depth: int = 4,
) -> str:
    """Unparse `node` after substituting module-level Names with their values
    (recursively, up to `max_depth` levels). Cycle-safe via visited set.
    """
    visited: set[str] = set()

    class Resolver(ast.NodeTransformer):
        def __init__(self, depth: int):
            self.depth = depth

        def visit_Name(self, n: ast.Name):
            if (isinstance(n.ctx, ast.Load)
                    and n.id in module_assigns
                    and n.id not in visited
                    and self.depth < max_depth):
                visited.add(n.id)
                substitute = module_assigns[n.id]
                # Recursively resolve names inside the substitute
                try:
                    return Resolver(self.depth + 1).visit(_clone(substitute))
                finally:
                    visited.discard(n.id)
            return n

    try:
        resolved = Resolver(0).visit(_clone(node))
        return ast.unparse(resolved)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return "<unparseable>"


def _clone(node: ast.AST) -> ast.AST:
    """Deep-clone an AST node by round-tripping through source."""
    return ast.parse(ast.unparse(node), mode="eval").body
