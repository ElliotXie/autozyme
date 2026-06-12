"""Profile enrichment: layer attribution + call-count + per-layer aggregates.

Adds three signals on top of the raw hotspot list that the parsers produce:

  * Per-hotspot `layer` tag — task / library:<pkg> / base-r / base-py /
    primitive / builtin / unknown. Tells the agent the editable scope of
    each hot function without needing a separate source dive.

  * Per-hotspot `n_calls_est` (R) — number of distinct stack-entry events
    for the function in the Rprof.out sample stream. Estimate (±20%) for
    sub-sample-interval calls. Python's cProfile already records exact
    `ncalls` so the Py path just promotes that field.

  * Top-level `layer_breakdown` aggregate — one line per layer with summed
    self_time + percentage. This is the single most actionable new signal:
    it tells the agent "X% of wall is in editable task code, Y% in
    library, Z% in primitives" at a glance, instead of mentally tagging
    every hotspot.

  * Top-level `per_layer_top` — top-5 hotspots per layer group, so the
    dominant editable function in each bucket is visible even when the
    global top-15 is dominated by primitives.

This module is a post-process step: it mutates the dict that
`parsers.normalize()` returned, in place. No new I/O against the raw
profile artifacts is required for Python (cProfile already encodes
everything we need). R needs the Rprof.out to re-scan for call counts —
that path is the only place we touch raw data again.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional


_BASE_R_PACKAGES = {
    "base", "utils", "stats", "graphics", "grDevices", "methods",
    "datasets", "tools", "parallel", "compiler", "splines",
}

_BASE_PY_MODULES = {
    "builtins", "abc", "collections", "contextlib", "copy", "copyreg",
    "enum", "functools", "gc", "importlib", "inspect", "io", "itertools",
    "keyword", "linecache", "logging", "operator", "os", "pickle",
    "platform", "posixpath", "re", "shutil", "site", "string", "struct",
    "subprocess", "sys", "tempfile", "threading", "time", "tokenize",
    "traceback", "types", "warnings", "weakref",
    "_thread", "_warnings", "_io", "_signal", "_socket",
    "_collections_abc",
}

_R_PRIMITIVE_PREFIXES = (".C(", ".Fortran(", ".Call(", ".External(",
                         ".External2(", ".Primitive(", ".C ", ".Fortran ",
                         ".Call ", ".External ", ".External2 ", ".Primitive ")


def parse_target_pkg_from_task_yaml(task_dir: Path) -> Optional[str]:
    """Extract the package name from task.yaml::target_function.

    target_function is conventionally `pkg::func` (R) or `pkg.mod.func`
    (Python). Returns the leading package token, or None if absent.
    """
    yaml = task_dir / "task.yaml"
    if not yaml.exists():
        return None
    text = yaml.read_text(errors="ignore")
    m = re.search(r"^target_function:\s*([^\s#<>]+)", text, re.MULTILINE)
    if not m:
        return None
    tf = m.group(1).strip()
    if not tf or tf.startswith("<"):
        return None
    # R: pkg::func
    if "::" in tf:
        return tf.split("::", 1)[0]
    # Py: pkg.mod.func — take leading token
    if "." in tf:
        return tf.split(".", 1)[0]
    return tf


# ---------------------------------------------------------------------------
# R: parse Rprof.out for stack-transition call counts
# ---------------------------------------------------------------------------

def count_r_calls(rprof_path: Path) -> dict[str, int]:
    """Return per-function call-count estimate from an Rprof.out file.

    Strategy: walk the sample stream sequentially; each time a function
    appears in a sample's stack that wasn't in the previous sample's stack,
    that's a fresh entry — count it. Approximate (sub-sample-interval
    calls are missed) but lets the agent recognise per-iteration loop
    bodies vs one-shot setup functions.
    """
    counts: dict[str, int] = defaultdict(int)
    if not rprof_path.exists():
        return {}
    try:
        lines = rprof_path.read_text(errors="ignore").splitlines()
    except Exception:
        return {}
    if not lines:
        return {}
    # First line is the sample.interval=NNN header (or in newer R: blank).
    prev_set: set[str] = set()
    for ln in lines[1:]:
        if not ln:
            prev_set = set()
            continue
        toks = [t.strip('"') for t in ln.split() if t]
        toks = [t for t in toks if t]
        cur = set(toks)
        for fn in cur - prev_set:
            counts[fn] += 1
        prev_set = cur
    return dict(counts)


# ---------------------------------------------------------------------------
# Layer classification
# ---------------------------------------------------------------------------

def _classify_r_function(name: str, target_pkg: Optional[str],
                         loaded_ns_index: dict[str, str] | None = None) -> str:
    """Return a layer tag for one R function name."""
    if not name:
        return "unknown"
    # Strip leading/trailing quotes.
    f = name.strip('"')
    # Primitive calls
    if any(f.startswith(p) for p in _R_PRIMITIVE_PREFIXES):
        return "primitive"
    if f in {".C", ".Fortran", ".Call", ".External", ".External2", ".Primitive"}:
        return "primitive"
    # Native symbol from --backend native: "lib.so:symbol" / "lib.dylib:symbol"
    # Must come before :: check because native symbols can contain C++ "::"
    _native_m = re.match(r"^([^:]+\.(?:so|dylib)):(.+)$", f)
    if _native_m:
        lib = _native_m.group(1)
        pkg = re.sub(r"\.\d+", "", lib)             # libRblas.0.dylib -> libRblas.dylib
        pkg = re.sub(r"\.(so|dylib)$", "", pkg)     # Matrix.so -> Matrix
        pkg = re.sub(r"^lib", "", pkg)               # libR.dylib -> R after above
        pkg = re.sub(r"^sourceCpp_\d+$", "", pkg)   # sourceCpp_4 -> "" (inline Rcpp)
        if not pkg or pkg in {"R", "system_m", "z"}:
            return "base-r"
        if pkg in {"Rblas", "Rlapack"}:
            return "primitive"
        if target_pkg and pkg == target_pkg:
            return "task"
        return f"library:{pkg}"
    # Namespace-qualified call: "pkg::fn" or "pkg:::fn" — package is explicit.
    if "::" in f:
        pkg = f.split("::", 1)[0]
        if target_pkg and pkg == target_pkg:
            return "task"
        if pkg in _BASE_R_PACKAGES:
            return "base-r"
        return f"library:{pkg}"
    # Namespace-indexed lookup (best signal)
    if loaded_ns_index:
        pkg = loaded_ns_index.get(f)
        if pkg:
            if target_pkg and pkg == target_pkg:
                return "task"
            if pkg in _BASE_R_PACKAGES:
                return "base-r"
            return f"library:{pkg}"
    # Closure heuristic — family$ls / fam$Dd / object$method
    # These are typical NB family / S3 method closures bound at runtime;
    # mgcv NB family methods are the canonical example.
    if "$" in f:
        base, _, method = f.partition("$")
        if base in {"family", "fam", "object", "model"}:
            if method in {"ls", "Dd", "linkinv", "mu.eta", "variance",
                          "dev.resids", "validmu", "initialize", "valideta",
                          "aic", "Dpsi", "g2g", "g3g", "g4g", "Dd2",
                          "linkfun", "simulate"}:
                # heuristic: NB family functions from mgcv/stats
                if target_pkg == "mgcv":
                    return "task"
                return "library:mgcv"
    # Common base-R name patterns (without namespace data)
    if f in {"<Anonymous>", "FUN", "anonymous"}:
        return "anonymous"
    # Fall back
    return "unknown"


def _extract_module_from_builtin_name(name: str) -> Optional[str]:
    """Extract a top-level package from C-extension function signatures.

    cProfile records C-extension functions in two forms:
      - "<built-in method numpy.core._multiarray_umath.implement_array_function>"
        → module = "numpy"
      - "<method 'reduce' of 'numpy.ufunc' objects>"
        → module = "numpy"
      - "<built-in method builtins.len>"
        → module = "builtins"
    Returns the top-level package name, or None if unparseable.
    """
    # Form 1: <built-in method module.path.func>
    m = re.match(r"<built-in method ([a-zA-Z_][a-zA-Z0-9_.]+)>", name)
    if m:
        full = m.group(1)
        return full.split(".")[0]
    # Form 2: <method 'X' of 'module.Class' objects>
    m = re.match(r"<method '[^']+' of '([a-zA-Z_][a-zA-Z0-9_.]+)' objects>", name)
    if m:
        full = m.group(1)
        return full.split(".")[0]
    return None


def _classify_py_function(raw: dict, target_pkg: Optional[str]) -> str:
    """Return a layer tag for one Python cProfile entry."""
    file = (raw or {}).get("file") or ""
    name = (raw or {}).get("func") or ""

    # C-extension / built-in functions: cProfile records these with file="~"
    # or file="" and a descriptive name. Many carry the originating module
    # in the name string — extract it for accurate layer attribution instead
    # of lumping everything into a generic "builtin" bucket.
    if file in ("~", "") or file.startswith("<"):
        mod = _extract_module_from_builtin_name(name)
        if mod:
            if target_pkg and (mod == target_pkg or mod.startswith(target_pkg)):
                return "task"
            if mod in _BASE_PY_MODULES or mod == "builtins":
                return "base-py"
            return f"library:{mod}"
        return "builtin"
    if name.startswith("<built-in method ") or name.startswith("<method "):
        mod = _extract_module_from_builtin_name(name)
        if mod:
            if target_pkg and mod == target_pkg:
                return "task"
            if mod in _BASE_PY_MODULES or mod == "builtins":
                return "base-py"
            return f"library:{mod}"
        return "builtin"

    # Derive module from filename
    parts = Path(file).parts
    module: Optional[str] = None
    # Look for site-packages / dist-packages
    for i, part in enumerate(parts):
        if part in ("site-packages", "dist-packages"):
            if i + 1 < len(parts):
                cand = parts[i + 1]
                if cand.endswith(".py"):
                    cand = cand[:-3]
                module = cand
                break
    # Stdlib (.../python3.X/...)
    if module is None:
        for i, part in enumerate(parts):
            if re.match(r"^python\d", part) and i + 1 < len(parts):
                cand = parts[i + 1]
                if cand.endswith(".py"):
                    cand = cand[:-3]
                module = cand
                break
    if module is None:
        # File not in site-packages or stdlib. Check for:
        # 1. upstream_repo/ in path → cloned target package source → task
        # 2. filename matches target_pkg → user script for this task
        path_parts = Path(file).parts
        if "upstream_repo" in path_parts:
            return "task"
        if target_pkg and Path(file).stem == target_pkg:
            return "task"
        return "unknown"

    if target_pkg and (module == target_pkg or module.startswith(target_pkg + ".")):
        return "task"
    if module in _BASE_PY_MODULES or module.startswith("_"):
        return "base-py"
    return f"library:{module}"


# ---------------------------------------------------------------------------
# R: build a {func -> pkg} index by running a small R subprocess once.
# This is best-effort; falls back to heuristic-only classification if R is
# missing or the subprocess errors.
# ---------------------------------------------------------------------------

def build_r_namespace_index(executor: Optional[dict],
                            aux_pkgs: Iterable[str] = ()) -> dict[str, str]:
    """Run a one-shot R subprocess that loads aux_pkgs and dumps
    {function_name -> first_loaded_namespace} as a JSON map. Returns {} on
    any failure — caller must be defensive."""
    import json
    import subprocess

    rscript = "Rscript"
    if executor:
        rs = executor.get("rscript")
        if rs:
            rscript = rs

    # Wrap each library() in invisible() — library returns a value at top
    # level which would otherwise auto-print and break our jsonlite-only
    # stdout contract.
    # Quote the package name — `library(foo, character.only=TRUE)` only
    # works if foo is a quoted string, not a bare symbol.
    aux_load = "; ".join(
        f'invisible(suppressPackageStartupMessages(tryCatch(library("{p}", character.only=TRUE), error=function(e) NULL)))'
        for p in aux_pkgs
    )
    r_code = (
        "if (!requireNamespace('jsonlite', quietly=TRUE)) {"
        "  cat('{}'); quit(status=0)"
        "};"
        + aux_load +
        # Also load Suggests+Enhances of every loaded package so lazy deps
        # (e.g. glmGamPoi via sctransform Enhances / Seurat Suggests) appear
        # in the index.  Filter to installed-only, skip already-loaded.
        "; .inst <- rownames(installed.packages());"
        " .already <- loadedNamespaces();"
        " .to_try <- character(0);"
        " for (pkg in .already) {"
        "   desc <- tryCatch(packageDescription(pkg), error=function(e) NULL);"
        "   if (is.null(desc)) next;"
        "   for (.fld in c('Suggests', 'Enhances')) {"
        "     val <- desc[[.fld]];"
        "     if (!is.null(val)) {"
        "       .s <- trimws(gsub('[(].*?[)]', '', strsplit(val, ',')[[1]]));"
        "       .to_try <- c(.to_try, .s)"
        "     }"
        "   }"
        " };"
        " .to_try <- setdiff(unique(.to_try[nchar(.to_try) > 0 & .to_try != 'R']), .already);"
        " .to_try <- .to_try[.to_try %in% .inst];"
        " for (s in .to_try) tryCatch(loadNamespace(s), error=function(e) NULL);"
        " idx <- list();"
        " for (pkg in loadedNamespaces()) {"
        "   ns <- tryCatch(getNamespace(pkg), error=function(e) NULL);"
        "   if (is.null(ns)) next;"
        "   for (nm in ls(ns, all.names=TRUE)) {"
        "     if (is.null(idx[[nm]])) idx[[nm]] <- pkg"
        "   }"
        " };"
        " cat(jsonlite::toJSON(idx, auto_unbox=TRUE))"
    )
    try:
        out = subprocess.run([rscript, "--vanilla", "-e", r_code],
                             capture_output=True, text=True, timeout=90)
        if out.returncode != 0:
            return {}
        text = out.stdout.strip()
        if not text:
            return {}
        # Be defensive: find the first '{' and parse from there. If the
        # caller invocation leaked any text before the JSON (warnings that
        # ignored suppressPackageStartupMessages, lazy-load notices, etc.)
        # we still recover.
        brace = text.find("{")
        if brace > 0:
            text = text[brace:]
        return json.loads(text)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _extract_pipeline_function_names(task_dir: Path) -> set[str]:
    """Extract function names defined in pipeline/run.R and sibling .cpp files.

    These are agent-written functions that should be classified as 'task'."""
    names: set[str] = set()
    pipeline_dir = task_dir / "pipeline"
    run_r = pipeline_dir / "run.R"
    if not run_r.exists():
        return names
    text = run_r.read_text(errors="ignore")
    for m in re.finditer(r"^(\w+)\s*(?:<-|=)\s*function\s*\(", text, re.MULTILINE):
        names.add(m.group(1))
    for m in re.finditer(
        r"//\s*\[\[Rcpp::export\]\]\s*\n\s*\w[\w:<>]*\s+(\w+)\s*\(", text
    ):
        names.add(m.group(1))
    for cpp in pipeline_dir.glob("*.cpp"):
        cpp_text = cpp.read_text(errors="ignore")
        for m in re.finditer(
            r"//\s*\[\[Rcpp::export\]\]\s*\n\s*\w[\w:<>]*\s+(\w+)\s*\(", cpp_text
        ):
            names.add(m.group(1))
    names -= {"get_script_dir"}
    return names


def enrich(profile_data: dict, *, target_pkg: Optional[str],
           rprof_path: Optional[Path] = None,
           executor: Optional[dict] = None,
           aux_pkgs: Iterable[str] = (),
           task_dir: Optional[Path] = None) -> None:
    """Enrich profile_data in place with layer, n_calls_est, layer_breakdown,
    per_layer_top.

    Required:
      profile_data: the dict returned by parsers.normalize()
      target_pkg:   the task's target package name (e.g. "tradeSeq", "fipy");
                    may be None — classification still runs, but task-tagging
                    won't fire.

    Optional:
      rprof_path:   path to the Rprof.out (for R parsers) so call counts can
                    be computed by stack-transition counting. If absent,
                    `n_calls_est` is left as whatever the parser provided.
      executor:     task.yaml executor block — propagates to R subprocess.
      aux_pkgs:     extra R packages to load in the namespace-indexer
                    subprocess (improves classification accuracy).
      task_dir:     task root directory — when provided, function names from
                    pipeline/run.R are classified as 'task'.
    """
    lang = profile_data.get("lang")
    hotspots = profile_data.get("hotspots") or []
    if not hotspots:
        # Still attach empty aggregates so downstream readers see the new
        # schema cleanly.
        profile_data["layer_breakdown"] = []
        profile_data["per_layer_top"] = {}
        profile_data["schema_version"] = "2"
        return

    # --- R-specific: call-count + namespace index -------------------------
    r_ns_index: dict[str, str] = {}
    r_call_counts: dict[str, int] = {}
    if lang == "R":
        if rprof_path is not None:
            r_call_counts = count_r_calls(rprof_path)
        # Build namespace index by spawning R once (best-effort).
        ns_aux = list(aux_pkgs)
        if target_pkg:
            ns_aux.insert(0, target_pkg)
        if ns_aux:
            r_ns_index = build_r_namespace_index(executor, ns_aux)

    # --- pipeline function names (agent-written → task layer) ---------------
    pipeline_fns: set[str] = set()
    if task_dir is not None and lang == "R":
        pipeline_fns = _extract_pipeline_function_names(task_dir)

    # --- annotate each hotspot --------------------------------------------
    for h in hotspots:
        label = h.get("label", "")
        if lang == "py":
            layer = _classify_py_function(h.get("raw") or {}, target_pkg)
            # cProfile already gives us exact call count via raw.ncalls
            ncalls = (h.get("raw") or {}).get("ncalls")
            if ncalls is not None:
                h["n_calls"] = int(ncalls)
        else:
            bare = label.strip('"')
            if bare in pipeline_fns:
                layer = "task"
            else:
                layer = _classify_r_function(label, target_pkg, r_ns_index)
            if bare in r_call_counts:
                h["n_calls_est"] = int(r_call_counts[bare])
            elif label in r_call_counts:
                h["n_calls_est"] = int(r_call_counts[label])
        h["layer"] = layer
        # Layer group (collapse library:* into "library")
        h["layer_group"] = "library" if layer.startswith("library:") else layer

    # --- layer_breakdown aggregate ----------------------------------------
    totals: dict[str, float] = defaultdict(float)
    for h in hotspots:
        st = h.get("self_time_s") or 0.0
        totals[h["layer_group"]] += st
    grand = sum(totals.values()) or 1.0
    breakdown = [
        {"layer_group": k, "self_time_s": round(v, 3),
         "pct": round(100.0 * v / grand, 1)}
        for k, v in sorted(totals.items(), key=lambda x: -x[1])
    ]
    profile_data["layer_breakdown"] = breakdown

    # --- per_layer_top: top-5 hotspots per layer_group --------------------
    per_layer: dict[str, list[dict]] = defaultdict(list)
    for h in sorted(hotspots, key=lambda x: -(x.get("self_time_s") or 0.0)):
        per_layer[h["layer_group"]].append({
            "label": h.get("label"),
            "self_pct": h.get("self_pct"),
            "self_time_s": h.get("self_time_s"),
            "n_calls": h.get("n_calls") or h.get("n_calls_est"),
            "layer": h.get("layer"),
        })
    profile_data["per_layer_top"] = {
        k: v[:5] for k, v in per_layer.items()
    }

    # --- python_native_split: Scalene backends give per-hotspot py% / native%
    # which is the most direct "is this Python dispatch overhead or actual C
    # compute?" signal. Aggregate into a single top-level block. Works with
    # ANY C extension (numpy, scipy, torch, TF, numba) without per-framework
    # plugins — Scalene's sampling naturally separates Python from compiled
    # code. Only emitted when at least one hotspot carries the raw fields.
    total_py = total_native = total_system = 0.0
    has_split = False
    for h in hotspots:
        raw = h.get("raw") or {}
        py_pct = raw.get("cpu_python_pct")
        nat_pct = raw.get("cpu_native_pct")
        if py_pct is not None or nat_pct is not None:
            has_split = True
            st = h.get("self_time_s") or 0.0
            total_py += st * (py_pct or 0) / 100.0
            total_native += st * (nat_pct or 0) / 100.0
            sys_pct = raw.get("cpu_system_pct") or 0
            total_system += st * sys_pct / 100.0
    if has_split:
        grand_pns = (total_py + total_native + total_system) or 1.0
        profile_data["python_native_split"] = {
            "python_s": round(total_py, 3),
            "python_pct": round(100.0 * total_py / grand_pns, 1),
            "native_s": round(total_native, 3),
            "native_pct": round(100.0 * total_native / grand_pns, 1),
            "system_s": round(total_system, 3),
            "system_pct": round(100.0 * total_system / grand_pns, 1),
        }

    # Bump schema flag so readers can detect enriched output.
    profile_data["schema_version"] = "2"
