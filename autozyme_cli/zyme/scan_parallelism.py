"""Scan upstream source for parallelism backends.

Backs the `zyme inspect-parallelism` subcommand: walks an upstream repo
checkout and detects every parallelism mechanism the call chain might
touch, emitting a structured report + draft `parallelism_profile` YAML
block for the init agent to paste (after review) into `task.yaml`.

Detection has two confidence tiers:

  DETECTED  — direct source-code match (e.g. `bplapply(`, `#pragma omp`).
              These are load-bearing — listed verbatim with file:line.
  LIKELY    — inferred from compiled-dep declarations (e.g. `Imports: mgcv`
              implies BLAS). Cannot be resolved without runtime probing,
              so flagged as "you should verify or accept the inference."

Caveats: regex-based, not AST. Comments and strings are matched too. False
positives are cheap (agent removes the line); false negatives are
expensive (silent under-inventory). Patterns err toward over-detection.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Source extensions we scan. Binary / vendored / lock files skipped.
_TEXT_EXTS = {
    ".R", ".r", ".Rmd",
    ".py", ".pyx", ".pxd", ".pyi",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx",
    ".jl", ".f90", ".f", ".F90",
    ".sh", ".yaml", ".yml", ".toml", ".cfg",
    "DESCRIPTION", "NAMESPACE", "Makevars", "Makevars.in",
    "setup.py", "setup.cfg", "pyproject.toml", "requirements.txt",
}

_SKIP_DIRS = {
    ".git", "node_modules", "vendor", "build", "dist", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".tox", ".venv", "venv", "env",
    "target", "Cargo.lock",
}

# Backend types align with the schema in task.yaml::parallelism_profile.upstream_backends.
# Each entry: (backend_type, list of (regex, language_hint) tuples)
_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "mclapply": [
        (r"\bmclapply\s*\(", "R"),
        (r"\bparLapply\s*\(", "R"),
        (r"\bparLapplyLB\s*\(", "R"),
        (r"\bclusterApply\w*\s*\(", "R"),
        (r"\bmakeCluster\s*\(", "R"),
        (r"\bMulticoreParam\s*\(", "R"),
        (r"\bSnowParam\s*\(", "R"),
        (r"\bbplapply\s*\(", "R"),
        (r"\bbpmapply\s*\(", "R"),
        (r"\bbpvec\s*\(", "R"),
        (r"\bBiocParallel::register\b", "R"),
        (r"\bfuture_lapply\s*\(", "R"),
        (r"%dopar%", "R"),
        (r"\bregisterDoParallel\s*\(", "R"),
    ],
    "mp_pool": [
        (r"\bmultiprocessing\.Pool\s*\(", "Python"),
        (r"\bmp\.Pool\s*\(", "Python"),
        (r"\bProcessPoolExecutor\s*\(", "Python"),
        (r"\bThreadPoolExecutor\s*\(", "Python"),
        (r"^from\s+multiprocessing\s+import", "Python"),
        (r"^from\s+concurrent\.futures\s+import", "Python"),
    ],
    "joblib": [
        (r"\bjoblib\.Parallel\s*\(", "Python"),
        (r"^from\s+joblib\s+import", "Python"),
        (r"^import\s+joblib\b", "Python"),
        (r"\bn_jobs\s*=", "Python"),  # broad; sklearn / many libs
    ],
    "openmp": [
        (r"#pragma\s+omp\s+", "C/C++"),
        (r"#include\s+[<\"]omp\.h[>\"]", "C/C++"),
        (r"\bomp_set_num_threads\s*\(", "C/C++"),
        (r"\bomp_get_num_threads\s*\(", "C/C++"),
        (r"-fopenmp\b", "Makevars"),
    ],
    "rcpp_parallel": [
        (r"\bRcppParallel::", "R"),
        (r"#include\s+[<\"]RcppParallel\.h[>\"]", "C/C++"),
        (r"\bparallelFor\s*\(", "C/C++"),
        (r"\bparallelReduce\s*\(", "C/C++"),
        (r"\bsetThreadOptions\s*\(", "any"),
    ],
    "data_table": [
        (r"\bsetDTthreads\s*\(", "R"),
        (r"\bdata\.table::setDTthreads\b", "R"),
        (r"\bgetDTthreads\s*\(", "R"),
    ],
    "numba": [
        (r"@(?:numba\.)?(?:jit|njit|prange|vectorize|guvectorize)\b", "Python"),
        (r"\bnumba\.set_num_threads\s*\(", "Python"),
        (r"\bnumba\.prange\s*\(", "Python"),
        (r"^from\s+numba\s+import", "Python"),
        (r"^import\s+numba\b", "Python"),
    ],
    "torch": [
        (r"\btorch\.set_num_threads\s*\(", "Python"),
        (r"\btorch\.set_num_interop_threads\s*\(", "Python"),
        (r"^import\s+torch\b", "Python"),
        (r"^from\s+torch\s+import", "Python"),
    ],
    "tf": [
        (r"\btf\.config\.threading\.", "Python"),
        (r"^import\s+tensorflow\b", "Python"),
        (r"^from\s+tensorflow\s+import", "Python"),
    ],
    "cuda": [
        (r"\btorch\.cuda\.", "Python"),
        (r"\.cuda\(\)", "Python"),
        (r"^import\s+cupy\b", "Python"),
        (r"^import\s+cuml\b", "Python"),
        (r"\bCUDA_VISIBLE_DEVICES\b", "any"),
        (r"#include\s+[<\"]cuda\.h[>\"]", "C/C++"),
        (r"\b__global__\s+", "C/C++"),
    ],
    "mpi": [
        (r"^from\s+mpi4py\b", "Python"),
        (r"^import\s+mpi4py\b", "Python"),
        (r"\bMPI_(?:Init|Comm|Send|Recv|Bcast|Reduce)\b", "C/C++"),
        (r"#include\s+[<\"]mpi\.h[>\"]", "C/C++"),
    ],
    "env_thread_vars": [
        # Reading any of the standard thread env vars
        (
            r"(Sys\.getenv|os\.environ\.get|os\.environ\[)\s*\(?\s*['\"]"
            r"(OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS|"
            r"VECLIB_MAXIMUM_THREADS|NUMEXPR_NUM_THREADS|RAYON_NUM_THREADS|"
            r"TBB_NUM_THREADS|NUMBA_NUM_THREADS|GOTO_NUM_THREADS|"
            r"BLIS_NUM_THREADS|POLARS_MAX_THREADS|JULIA_NUM_THREADS)",
            "any",
        ),
    ],
    "kwargs_parallel": [
        # Function-signature / call-site parallelism kwargs (broad; informational)
        (r"\bparallel\s*=\s*(?:TRUE|FALSE|True|False)", "any"),
        (r"\bnthreads\s*=", "any"),
        (r"\bnum_workers\s*=", "Python"),
        (r"\bworkers\s*=", "any"),
        (r"\bmc\.cores\s*=", "R"),
        (r"\bBPPARAM\s*=", "R"),
    ],
}

# Compiled-dep packages that strongly imply BLAS / OpenMP usage at runtime.
# These are inferred (LIKELY tier), not detected.
_R_DEPS_IMPLY_BLAS = {
    "mgcv", "lme4", "MASS", "Matrix", "RcppArmadillo", "RcppEigen",
    "irlba", "RSpectra", "fields", "stats",
}
_R_DEPS_IMPLY_OPENMP = {
    "RcppArmadillo", "RcppEigen", "data.table",
    "mgcv",       # exposes nthreads in gam.control; uses OpenMP for smoothing fits
    "lme4",       # transitively via RcppEigen
    "glmnet",     # uses OpenMP in src
}
_R_DEPS_IMPLY_RCPP_PARALLEL = {"RcppParallel"}
_R_DEPS_IMPLY_DATA_TABLE = {"data.table"}

_PY_DEPS_IMPLY_BLAS = {
    "numpy", "scipy", "scikit-learn", "sklearn", "pandas",
    "torch", "tensorflow", "jax", "polars", "statsmodels",
}
_PY_DEPS_IMPLY_NUMBA = {"numba"}
_PY_DEPS_IMPLY_OPENMP = {"numexpr"}


@dataclass
class Hit:
    """A single regex match in a source file."""
    backend: str
    file: str           # path relative to repo root
    line: int
    text: str           # raw matched line, stripped


@dataclass
class DepInference:
    """A backend inferred from declared dependency."""
    backend: str
    via: str            # human description of the inference path


@dataclass
class DefaultKnob:
    """A user-controllable parallel knob with its default value, extracted
    from a function signature in production code.

    These tell the agent what upstream looks like *out of the box*: e.g.
    `parallel = FALSE` means upstream defaults to single-thread, which
    is the answer to `upstream_default_threads` 9 times out of 10.
    """
    knob: str           # e.g. "parallel", "BPPARAM", "n_jobs", "mc.cores"
    default: str        # raw default expression, e.g. "FALSE", "bpparam()", "-1", "1L"
    file: str
    line: int
    function_hint: str  # nearest preceding `name <- function(...)` / `def name(...)`


@dataclass
class Inventory:
    """Aggregated scan results."""
    repo_path: Path
    files_scanned: int = 0
    files_by_lang: dict[str, int] = field(default_factory=dict)
    hits: list[Hit] = field(default_factory=list)
    inferred: list[DepInference] = field(default_factory=list)
    deps_seen: dict[str, set[str]] = field(default_factory=dict)  # {"R-Imports": {"mgcv", ...}}
    knobs: list[DefaultKnob] = field(default_factory=list)
    # When target-function filtering is applied, knobs not on the call chain
    # are moved here so the report can show them collapsed.
    off_chain_knobs: list[DefaultKnob] = field(default_factory=list)


# Compile patterns once.
_COMPILED: dict[str, list[tuple[re.Pattern, str]]] = {
    backend: [(re.compile(rx), lang) for rx, lang in pats]
    for backend, pats in _PATTERNS.items()
}


def _path_priority(rel_path: str) -> int:
    """Lower = better. Production code outranks tests outranks docs.

    Used to pick the most informative file:line as the "via" hint when a
    backend has multiple hits — agents reading "via: tests/test_*.py"
    rightly suspect the match is incidental.
    """
    p = rel_path.lower().replace("\\", "/")
    parts = p.split("/")
    # Documentation / vignettes / examples
    if any(seg in {"vignettes", "docs", "doc", "examples", "example"} for seg in parts):
        return 30
    if p.endswith((".rmd", ".rst", ".md", ".ipynb")):
        return 30
    # Tests
    if any(seg in {"tests", "test", "testthat", "spec", "specs"} for seg in parts):
        return 20
    if any(seg.startswith("test_") or seg.startswith("test-") for seg in parts):
        return 20
    # Production: R/, src/, <pkg>/. NOTE: `p` is already lowercased above, so the
    # R-package source dir compares as "r" (matching it as "R" never fired).
    if parts and parts[0] in {"r", "src", "inst"}:
        return 0
    # Python: typically <pkg_name>/<modules>.py at root
    if parts and len(parts) > 1 and parts[0] not in {"build", "dist", "scripts"}:
        return 5
    return 10


def _classify_lang(path: Path) -> str:
    name = path.name
    suffix = path.suffix
    if name in {"DESCRIPTION", "NAMESPACE"}:
        return "R-meta"
    if name.startswith("Makevars"):
        return "Makevars"
    if name in {"setup.py", "setup.cfg", "pyproject.toml", "requirements.txt"}:
        return "Python-meta"
    if suffix in {".R", ".r", ".Rmd"}:
        return "R"
    if suffix in {".py", ".pyx", ".pxd", ".pyi"}:
        return "Python"
    if suffix in {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx"}:
        return "C/C++"
    if suffix in {".jl"}:
        return "Julia"
    if suffix in {".f90", ".f", ".F90"}:
        return "Fortran"
    return "other"


def _iter_source_files(repo: Path):
    """Yield text source files under repo, skipping vendored/binary dirs."""
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        # Skip if any parent is in skip list
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        # Match by name (DESCRIPTION etc.) or extension
        if path.name in _TEXT_EXTS or path.suffix in _TEXT_EXTS:
            yield path
        elif path.name.startswith("Makevars") or path.name in {
            "DESCRIPTION", "NAMESPACE", "setup.py", "setup.cfg",
            "pyproject.toml", "requirements.txt",
        }:
            yield path


# Names of kwargs we treat as "user-controllable parallel knobs" — when these
# appear in a function signature with a default value, we surface the default.
_KNOB_NAMES = {
    "parallel", "BPPARAM", "mc.cores", "nthreads", "n_threads",
    "n_jobs", "num_workers", "workers", "threads", "ncores", "n_cores",
}

# Match a knob with default in either:
#   R signatures:   `, parallel = FALSE,`  /  `BPPARAM = BiocParallel::bpparam()`
#   Python sigs:    `def f(..., n_jobs=-1, ...)` / `parallel: bool = False`
# Captures (knob_name, default_expr).
_KNOB_DEFAULT_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_.]*)\s*"
    r"(?::\s*[A-Za-z_][\w\[\], ]*)?"        # optional Python type annotation
    r"\s*=\s*"
    r"([A-Z][A-Z]+|True|False|[A-Za-z_][A-Za-z0-9_:.]*\(\)|[+\-]?\d+L?|"     # FALSE/TRUE / True/False / func() / 4 / 4L
    r"\"[^\"]*\"|'[^']*')"
)

# Match a function definition opening so we can attach the knob to it.
# R allows names starting with `.` (private convention) — `.fitGAM <- function(...)`.
_R_FN_RE = re.compile(r"^\s*([\w.]+)\s*<-\s*function\s*\(")
_PY_FN_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(")


def _scan_function_signatures(text: str, rel: str, lang: str, inv: Inventory) -> None:
    """Walk the file looking for function signatures, then extract any
    parallel knobs with defaults from the signature block.

    A signature block can span multiple lines:
        fitGAM <- function(counts,
                           ...,
                           parallel = FALSE,
                           BPPARAM = BiocParallel::bpparam(),
                           ...) {
    We collect lines from the `function(` opener until the matching `)`,
    then scan the joined block once.
    """
    if lang not in {"R", "Python"}:
        return
    fn_re = _R_FN_RE if lang == "R" else _PY_FN_RE

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = fn_re.match(lines[i])
        if not m:
            i += 1
            continue
        fn_name = m.group(1)
        # Collect the signature block: everything from the `(` until the
        # matching `)`. Track paren depth.
        sig_start_line = i + 1
        sig_buf: list[str] = []
        depth = 0
        started = False
        while i < len(lines):
            ln = lines[i]
            for ch in ln:
                if ch == "(":
                    depth += 1
                    started = True
                elif ch == ")":
                    depth -= 1
            sig_buf.append(ln)
            i += 1
            if started and depth == 0:
                break
        sig_text = " ".join(sig_buf)
        # Extract knobs
        for km in _KNOB_DEFAULT_RE.finditer(sig_text):
            name, default = km.group(1), km.group(2)
            if name not in _KNOB_NAMES:
                continue
            inv.knobs.append(DefaultKnob(
                knob=name,
                default=default,
                file=rel,
                line=sig_start_line,
                function_hint=fn_name,
            ))


def _scan_file(path: Path, repo_root: Path, inv: Inventory) -> None:
    """Scan one file for all patterns, append hits to inventory."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return
    if len(text) > 5_000_000:  # skip files >5MB; likely vendored data
        return

    rel = str(path.relative_to(repo_root))
    inv.files_scanned += 1
    lang = _classify_lang(path)
    inv.files_by_lang[lang] = inv.files_by_lang.get(lang, 0) + 1

    # Function-signature pass: extract user-controllable knob defaults.
    # Only scan production-priority paths — knob defaults in tests / vignettes
    # are confusingly different from the real API surface.
    if _path_priority(rel) <= 5:
        _scan_function_signatures(text, rel, lang, inv)

    # Per-line scan so we can report file:line.
    is_r_or_py = lang in {"R", "Python", "Python-meta", "R-meta"}
    is_c = lang == "C/C++"
    for lineno, line in enumerate(text.splitlines(), start=1):
        # Skip very long lines (often minified data)
        if len(line) > 2000:
            continue
        # Skip comment-only lines so commented-out code doesn't poison the
        # via hint. Doesn't handle block comments (/* */); not worth the
        # complexity since parallelism calls rarely live in block comments.
        stripped = line.lstrip()
        if is_r_or_py and stripped.startswith("#"):
            continue
        if is_c and (stripped.startswith("//") or stripped.startswith("/*")):
            continue
        for backend, patterns in _COMPILED.items():
            for rx, _lang in patterns:
                if rx.search(line):
                    inv.hits.append(Hit(
                        backend=backend,
                        file=rel,
                        line=lineno,
                        text=line.strip()[:200],
                    ))
                    break  # one hit per (line, backend)


def _parse_r_description(path: Path) -> dict[str, set[str]]:
    """Extract Imports/Depends/LinkingTo/Suggests from R DESCRIPTION."""
    fields_out: dict[str, set[str]] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return fields_out

    # DESCRIPTION fields can span multiple lines (continuation indented).
    # Simple parse: find "Field:" then capture until next non-indented line.
    target_fields = {"Imports", "Depends", "LinkingTo", "Suggests"}
    current_field = None
    buf: list[str] = []

    def flush():
        nonlocal buf, current_field
        if current_field and current_field in target_fields:
            joined = " ".join(buf)
            # Strip version specifiers like "mgcv (>= 1.8)"
            entries = [
                re.sub(r"\s*\(.*?\)", "", e).strip()
                for e in joined.split(",")
            ]
            entries = [e for e in entries if e and e != "R"]
            fields_out.setdefault(current_field, set()).update(entries)
        buf = []

    for line in text.splitlines():
        m = re.match(r"^([A-Za-z]+):\s*(.*)$", line)
        if m:
            flush()
            current_field = m.group(1)
            buf = [m.group(2).strip()]
        elif line.startswith((" ", "\t")) and current_field:
            buf.append(line.strip())
        else:
            flush()
            current_field = None
    flush()
    return fields_out


# Project-metadata keys (pyproject.toml / setup.cfg / setup.py) that look like
# `key = ...` and must NOT be mistaken for dependency names. Applied to BOTH
# the bare-requirements regex (where `=` is a version operator) and the quoted
# regex in the fallback scanner.
_DEP_METADATA_KEYS = {
    "python", "version", "name", "description", "license", "author", "url",
    "homepage", "dependencies", "requires", "requires-python", "readme",
    "classifiers", "keywords", "maintainers", "authors", "license-files",
    "requires-dist", "optional-dependencies", "scripts", "packages",
}


def _dep_name_from_spec(spec: str) -> str | None:
    """Leading package name from a PEP 508 requirement (drop extras/version)."""
    m = re.match(r"[A-Za-z][A-Za-z0-9_.\-]*", spec.strip())
    if not m:
        return None
    return m.group(0).lower().replace("_", "-")


def _regex_scan_deps(text: str, deps: set[str]) -> None:
    """Best-effort dependency-name scan for setup.py / setup.cfg / requirements
    (and as a fallback for pyproject.toml when no TOML parser is available)."""
    # requirements.txt format: bare `numpy>=1.21` per line. NOTE: a TOML/
    # setup.cfg assignment `name = "x"` also matches here (`=` is in the
    # operator class), so the same metadata-key exclusion is applied.
    for m in re.finditer(
        r"^\s*([A-Za-z][A-Za-z0-9_.\-]*)\s*[<>=!~]",
        text, re.MULTILINE,
    ):
        name = m.group(1).lower().replace("_", "-")
        if name not in _DEP_METADATA_KEYS:
            deps.add(name)
    # Quoted string forms used by setup.py / pyproject.toml:
    #   'numpy>=1.21'  (setup.py INSTALL_REQUIRES)
    #   "numpy"         (pyproject.toml dependencies)
    # The name must follow the opening quote immediately and the optional
    # version specifier may not span a newline, so the regex cannot pair the
    # closing quote of one `key = "value"` with the opening quote of the next.
    for m in re.finditer(
        r'["\']([a-zA-Z][a-zA-Z0-9_.\-]{1,40})\s*(?:[<>=!~][^"\'\n]*)?["\']',
        text,
    ):
        name = m.group(1).lower().replace("_", "-")
        if name not in _DEP_METADATA_KEYS:
            deps.add(name)


def _parse_python_deps(repo: Path) -> set[str]:
    """Extract top-level dependency names from pyproject.toml / setup.py /
    requirements.txt. Names only — versions stripped."""
    deps: set[str] = set()

    # pyproject.toml: parse with a real TOML parser when available so project
    # metadata keys/values (name, license, ...) are never mistaken for deps.
    pp = repo / "pyproject.toml"
    if pp.is_file():
        try:
            text = pp.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError):
            text = None
        if text is not None:
            try:
                import tomllib as _toml  # Python 3.11+
            except ImportError:
                try:
                    import tomli as _toml  # backport
                except ImportError:
                    _toml = None
            data = None
            if _toml is not None:
                try:
                    data = _toml.loads(text)
                except Exception:
                    data = None
            if isinstance(data, dict):
                proj = data.get("project") or {}
                specs: list = list(proj.get("dependencies") or [])
                for grp in (proj.get("optional-dependencies") or {}).values():
                    specs.extend(grp or [])
                specs.extend((data.get("build-system") or {}).get("requires") or [])
                for spec in specs:
                    name = _dep_name_from_spec(str(spec))
                    if name:
                        deps.add(name)
            else:
                # No TOML parser / unparseable: degrade to the regex scanner.
                _regex_scan_deps(text, deps)

    for fname in ("requirements.txt", "setup.py", "setup.cfg"):
        p = repo / fname
        if not p.is_file():
            continue
        try:
            _regex_scan_deps(p.read_text(encoding="utf-8", errors="replace"), deps)
        except (OSError, UnicodeError):
            continue
    return deps


def _infer_from_deps(inv: Inventory) -> None:
    """Populate inv.inferred from declared compiled dependencies."""
    # R deps: union of Imports + Depends + LinkingTo (Suggests is informational)
    r_active = set()
    for f in ("Imports", "Depends", "LinkingTo"):
        r_active |= inv.deps_seen.get(f"R-{f}", set())
    if r_active:
        blas_hits = r_active & _R_DEPS_IMPLY_BLAS
        if blas_hits:
            inv.inferred.append(DepInference(
                backend="blas",
                via=f"R DESCRIPTION lists {', '.join(sorted(blas_hits))} (linear algebra)",
            ))
        omp_hits = r_active & _R_DEPS_IMPLY_OPENMP
        if omp_hits:
            inv.inferred.append(DepInference(
                backend="openmp",
                via=f"R DESCRIPTION lists {', '.join(sorted(omp_hits))} (OpenMP-linked)",
            ))
        rcpp_hits = r_active & _R_DEPS_IMPLY_RCPP_PARALLEL
        if rcpp_hits:
            inv.inferred.append(DepInference(
                backend="rcpp_parallel",
                via=f"R DESCRIPTION LinkingTo {', '.join(sorted(rcpp_hits))}",
            ))
        dt_hits = r_active & _R_DEPS_IMPLY_DATA_TABLE
        if dt_hits:
            inv.inferred.append(DepInference(
                backend="data_table",
                via=f"R DESCRIPTION lists {', '.join(sorted(dt_hits))} (uses TBB threading)",
            ))

    # Python deps
    py_deps = inv.deps_seen.get("Python", set())
    if py_deps:
        blas_hits = py_deps & _PY_DEPS_IMPLY_BLAS
        if blas_hits:
            inv.inferred.append(DepInference(
                backend="blas",
                via=f"Python deps include {', '.join(sorted(blas_hits))} (BLAS-backed)",
            ))
        numba_hits = py_deps & _PY_DEPS_IMPLY_NUMBA
        if numba_hits:
            inv.inferred.append(DepInference(
                backend="numba",
                via=f"Python deps include {', '.join(sorted(numba_hits))}",
            ))
        omp_hits = py_deps & _PY_DEPS_IMPLY_OPENMP
        if omp_hits:
            inv.inferred.append(DepInference(
                backend="openmp",
                via=f"Python deps include {', '.join(sorted(omp_hits))} (OpenMP-linked)",
            ))


# ---------------------------------------------------------------------------
# Target-function call-chain filter
# ---------------------------------------------------------------------------

_CALL_RE = re.compile(r"(?<!\w)(\.?[A-Za-z_][\w.]*)\s*\(")

# R S4 method: setMethod(f = "name", ...) or setMethod("name", ...)
_R_S4_RE = re.compile(
    r'^\s*setMethod\s*\(\s*(?:f\s*=\s*)?["\']([^"\']+)["\']'
)


def _build_fn_index(repo: Path) -> dict[str, list[tuple[str, int, int]]]:
    """Map function_name → [(rel_path, body_start_line, body_end_line), ...].

    Body extent is approximated: from the definition line to the next
    top-level definition (or EOF). Good enough for grep-style call tracing.
    """
    index: dict[str, list[tuple[str, int, int]]] = {}
    for path in _iter_source_files(repo):
        rel = str(path.relative_to(repo))
        if _path_priority(rel) > 5:
            continue
        lang = _classify_lang(path)
        if lang not in {"R", "Python"}:
            continue
        fn_re = _R_FN_RE if lang == "R" else _PY_FN_RE
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except (OSError, UnicodeError):
            continue
        defs: list[tuple[str, int]] = []
        for i, ln in enumerate(lines):
            m = fn_re.match(ln)
            if m:
                defs.append((m.group(1), i))
            elif lang == "R":
                m = _R_S4_RE.match(ln)
                if m:
                    defs.append((m.group(1), i))
        for j, (name, start) in enumerate(defs):
            end = defs[j + 1][1] if j + 1 < len(defs) else len(lines)
            index.setdefault(name, []).append((rel, start, end))
    return index


def _extract_body_calls(repo: Path, fn_index: dict, fn_name: str) -> set[str]:
    """Return set of function names called within fn_name's body."""
    entries = fn_index.get(fn_name, [])
    called: set[str] = set()
    for rel, start, end in entries:
        try:
            text = (repo / rel).read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError):
            continue
        body = "\n".join(text.splitlines()[start:end])
        for m in _CALL_RE.finditer(body):
            raw = m.group(1)
            called.add(raw)
            # For Python module.func style, also add the bare name
            if "." in raw and not raw.startswith("."):
                called.add(raw.split(".")[-1])
    return called


def _trace_reachable(repo: Path, target: str, max_depth: int = 3) -> set[str]:
    """BFS from target function, returning all reachable function names
    (within the repo's production code) up to max_depth hops."""
    fn_index = _build_fn_index(repo)
    reachable: set[str] = {target}
    frontier: set[str] = {target}
    # Also try the last component for Pkg::func or module.func style names
    bare = target.split("::")[-1].split(".")[-1]
    if bare != target:
        reachable.add(bare)
        frontier.add(bare)
    for _ in range(max_depth):
        next_frontier: set[str] = set()
        for fn in frontier:
            calls = _extract_body_calls(repo, fn_index, fn)
            new = (calls & set(fn_index.keys())) - reachable
            next_frontier |= new
        if not next_frontier:
            break
        reachable |= next_frontier
        frontier = next_frontier
    return reachable


def filter_knobs_by_target(inv: Inventory, repo: Path, target: str) -> None:
    """Move knobs whose function_hint is NOT reachable from target to
    inv.off_chain_knobs. Mutates inv in place."""
    reachable = _trace_reachable(repo, target)
    on_chain: list[DefaultKnob] = []
    off_chain: list[DefaultKnob] = []
    for k in inv.knobs:
        if k.function_hint in reachable:
            on_chain.append(k)
        else:
            off_chain.append(k)
    inv.knobs = on_chain
    inv.off_chain_knobs = off_chain


def scan(repo: Path) -> Inventory:
    """Run full scan. Returns Inventory with hits + inferred backends."""
    inv = Inventory(repo_path=repo)

    # Pass 1: parse DESCRIPTION + Python dep files
    desc_path = repo / "DESCRIPTION"
    if desc_path.exists():
        for field_name, names in _parse_r_description(desc_path).items():
            inv.deps_seen[f"R-{field_name}"] = names

    py_deps = _parse_python_deps(repo)
    if py_deps:
        inv.deps_seen["Python"] = py_deps

    # Pass 2: scan source files
    for path in _iter_source_files(repo):
        _scan_file(path, repo, inv)

    # Pass 3: infer from deps
    _infer_from_deps(inv)

    return inv


def _hits_by_backend(inv: Inventory) -> dict[str, list[Hit]]:
    """Group hits by backend, with each list sorted by path priority + file:line.

    Higher-priority paths come first so the YAML draft's "via" hint points
    at production code rather than a test or vignette match.
    """
    out: dict[str, list[Hit]] = {}
    for h in inv.hits:
        out.setdefault(h.backend, []).append(h)
    for backend, hits in out.items():
        hits.sort(key=lambda h: (_path_priority(h.file), h.file, h.line))
    return out


def _suggest_yaml(inv: Inventory) -> str:
    """Build a draft parallelism_profile YAML block from the inventory."""
    by_backend = _hits_by_backend(inv)
    inferred_only = {d.backend for d in inv.inferred} - set(by_backend.keys())

    lines = ["  parallelism_profile:", "    upstream_backends:"]

    # Detected backends — one entry each, with first hit as the via hint.
    # Map our internal backend name to the schema's `type` value.
    # `None` = excluded from the YAML upstream_backends list (informational only):
    #   - cuda: autozyme is CPU-only; GPU paths are out of scope. Reported
    #     separately so the agent confirms reference.{py,R} hits the CPU branch.
    #   - env_thread_vars / kwargs_parallel: searches for *symptoms* of
    #     parallelism (env-var reads, function-arg names) rather than concrete
    #     backends. Useful as a debugging signal, but the structural backends
    #     above already capture the actual mechanism.
    type_map = {
        "mclapply": "mclapply",
        "mp_pool": "mp.Pool",
        "joblib": "joblib",
        "openmp": "openmp",
        "rcpp_parallel": "rcpp_parallel",
        "data_table": "data_table",
        "numba": "numba",
        "torch": "torch",
        "tf": "tf",
        "cuda": None,                   # autozyme is CPU-only
        "mpi": "mpi",
        "env_thread_vars": None,
        "kwargs_parallel": None,
    }

    for backend, hits in sorted(by_backend.items()):
        if type_map.get(backend) is None:
            continue
        first = hits[0]
        type_val = type_map[backend]
        via = f"{first.file}:{first.line}"
        lines.append(f'      - {{type: {type_val}, via: "{via}"}}')

    # Inferred backends (deps-only)
    for inf in inv.inferred:
        if inf.backend in inferred_only:
            lines.append(
                f'      - {{type: {inf.backend}, via: "implicit ({inf.via})"}}'
            )

    if not by_backend and not inv.inferred:
        lines.append('      []                                    # no parallelism detected')

    lines.append("    upstream_default_threads: <FILL>            # threads upstream uses with all defaults — verify by running reference")
    lines.append("    parallelism_class: <FILL>                   # embarrassingly_parallel | inherently_serial | mixed — needs human judgment")
    return "\n".join(lines)


def format_report(inv: Inventory, max_hits_per_backend: int = 5) -> str:
    """Render a human-readable report of the inventory."""
    by_backend = _hits_by_backend(inv)
    lines = []
    lines.append(f"=== Parallelism inventory: {inv.repo_path} ===\n")
    lines.append(f"Files scanned: {inv.files_scanned}")
    lang_summary = ", ".join(
        f"{k}={v}" for k, v in sorted(inv.files_by_lang.items())
    )
    lines.append(f"  by language: {lang_summary}\n")

    # Detected (CPU-relevant backends — these go into upstream_backends)
    cpu_structural = [
        b for b in by_backend
        if b not in {"env_thread_vars", "kwargs_parallel", "cuda"}
    ]
    lines.append("DETECTED (direct source matches):")
    if not cpu_structural:
        lines.append("  (none)")
    else:
        for backend in sorted(cpu_structural):
            hits = by_backend[backend]
            lines.append(f"  {backend} — {len(hits)} match{'es' if len(hits) > 1 else ''}")
            for h in hits[:max_hits_per_backend]:
                lines.append(f"    {h.file}:{h.line}  {h.text[:120]}")
            if len(hits) > max_hits_per_backend:
                lines.append(f"    ... +{len(hits) - max_hits_per_backend} more")
    lines.append("")

    # GPU section — autozyme is CPU-only, but agents need to know GPU paths
    # exist so they can verify reference.{py,R} hits the CPU branch.
    if "cuda" in by_backend:
        hits = by_backend["cuda"]
        lines.append(f"GPU PATHS DETECTED ({len(hits)} match{'es' if len(hits) > 1 else ''}, OUT OF SCOPE):")
        lines.append("  autozyme is CPU-only — these are informational, NOT added to upstream_backends.")
        lines.append("  Action: confirm reference.{py,R} runs the CPU branch (no .to('cuda'),")
        lines.append("          no model.cuda(), torch.cuda.is_available() returns False on the host).")
        for h in hits[:max_hits_per_backend]:
            lines.append(f"    {h.file}:{h.line}  {h.text[:120]}")
        if len(hits) > max_hits_per_backend:
            lines.append(f"    ... +{len(hits) - max_hits_per_backend} more")
        lines.append("")

    # Inferred
    inferred_backends = {d.backend for d in inv.inferred} - set(by_backend.keys())
    if inferred_backends:
        lines.append("LIKELY (inferred from compiled deps):")
        for inf in inv.inferred:
            if inf.backend in inferred_backends:
                lines.append(f"  {inf.backend}")
                lines.append(f"    via: {inf.via}")
        lines.append("")

    # User-controllable knobs (function signature defaults). This is what
    # tells you upstream's actual default thread count without running it —
    # `parallel = FALSE` means upstream is serial out of the box.
    all_knobs_empty = not inv.knobs and not inv.off_chain_knobs
    if inv.knobs:
        header = "USER-CONTROLLABLE KNOBS (function signature defaults)"
        if inv.off_chain_knobs:
            header += f" — {len(inv.knobs)} on call chain"
        lines.append(f"{header}:")
        by_fn: dict[tuple[str, str], list[DefaultKnob]] = {}
        for k in inv.knobs:
            by_fn.setdefault((k.file, k.function_hint), []).append(k)
        for (file, fn), knobs in sorted(by_fn.items()):
            knob_strs = ", ".join(f"{k.knob}={k.default}" for k in knobs)
            lines.append(f"  {fn}() in {file}:{knobs[0].line}")
            lines.append(f"    {knob_strs}")
        lines.append("  → Read this row for `upstream_default_threads`: a `parallel=FALSE`,")
        lines.append("    `n_jobs=1`, or `mc.cores=1` default usually means 1.")
        if inv.off_chain_knobs:
            n_off = len(set((k.file, k.function_hint) for k in inv.off_chain_knobs))
            lines.append(f"  ({n_off} more function{'s' if n_off > 1 else ''} with knobs outside target call chain — omitted)")
        lines.append("")
    elif inv.off_chain_knobs:
        n_off = len(set((k.file, k.function_hint) for k in inv.off_chain_knobs))
        lines.append("USER-CONTROLLABLE KNOBS: none on target call chain.")
        lines.append(f"  ({n_off} function{'s' if n_off > 1 else ''} with knobs found elsewhere in repo — not reachable from target)")
        lines.append("")
    elif all_knobs_empty:
        lines.append("USER-CONTROLLABLE KNOBS: none found in function signatures.")
        lines.append("  Note: parallelism may still be controlled by global state")
        lines.append("  (e.g. future::plan(), options(mc.cores), OMP_NUM_THREADS,")
        lines.append("  threadpool_limits). Check whether the target function")
        lines.append("  dispatches to a parallel backend without a function-level knob.")
        lines.append("")

    # Informational signals
    for label, backend in (("Env thread-var reads", "env_thread_vars"),
                           ("Parallelism kwargs", "kwargs_parallel")):
        if backend in by_backend:
            hits = by_backend[backend]
            lines.append(f"{label} ({len(hits)} match{'es' if len(hits) > 1 else ''}, informational):")
            for h in hits[:max_hits_per_backend]:
                lines.append(f"    {h.file}:{h.line}  {h.text[:120]}")
            if len(hits) > max_hits_per_backend:
                lines.append(f"    ... +{len(hits) - max_hits_per_backend} more")
            lines.append("")

    # Not detected (cuda excluded — it has its own GPU-out-of-scope section)
    all_known = {
        "mclapply", "mp_pool", "joblib", "openmp", "rcpp_parallel",
        "data_table", "numba", "torch", "tf", "mpi",
    }
    detected_or_inferred = set(by_backend) | {d.backend for d in inv.inferred}
    missing = sorted(all_known - detected_or_inferred)
    if missing:
        lines.append(f"NOT DETECTED: {', '.join(missing)}")
        lines.append("")

    # Draft YAML
    lines.append("DRAFT task.yaml block (review + edit before pasting):")
    lines.append("")
    lines.append(_suggest_yaml(inv))
    lines.append("")

    # Caveats
    lines.append("CAVEATS:")
    lines.append("  - Regex-based; comments/strings may match. False positives are cheap, false negatives are not.")
    lines.append("  - parallelism_class needs call-graph analysis — script can't decide.")
    lines.append("  - upstream_default_threads = what reference.{py,R} runs at; verify by")
    lines.append("    running once with no env vars set and inspecting cpu/wall ratio.")
    lines.append("  - BLAS detection is implicit (deps-based); OS-specific linkage may differ.")
    lines.append("  - When in doubt, list it. Over-listing wastes nothing.")

    return "\n".join(lines)
