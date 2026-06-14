"""Unit tests for zyme.scan_parallelism — the upstream parallelism scanner.

The module is regex-based and pure: it walks a repo tree, matches parallelism
patterns, parses DESCRIPTION / Python dep files, infers backends from declared
deps, traces a call chain, and renders a human report + draft YAML. Every layer
is exercised here against small on-disk repo fixtures built in tmp_path — no
real upstream checkout needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme.scan_parallelism import (
    DefaultKnob,
    DepInference,
    Hit,
    Inventory,
    _build_fn_index,
    _classify_lang,
    _extract_body_calls,
    _hits_by_backend,
    _infer_from_deps,
    _iter_source_files,
    _parse_python_deps,
    _parse_r_description,
    _path_priority,
    _scan_file,
    _scan_function_signatures,
    _suggest_yaml,
    _trace_reachable,
    filter_knobs_by_target,
    format_report,
    scan,
)


# --------------------------------------------------------------------------
# _classify_lang
# --------------------------------------------------------------------------

class TestClassifyLang:
    @pytest.mark.parametrize("name,expected", [
        ("DESCRIPTION", "R-meta"),
        ("NAMESPACE", "R-meta"),
        ("Makevars", "Makevars"),
        ("Makevars.in", "Makevars"),
        ("setup.py", "Python-meta"),
        ("setup.cfg", "Python-meta"),
        ("pyproject.toml", "Python-meta"),
        ("requirements.txt", "Python-meta"),
        ("foo.R", "R"),
        ("foo.r", "R"),
        ("foo.Rmd", "R"),
        ("foo.py", "Python"),
        ("foo.pyx", "Python"),
        ("foo.pxd", "Python"),
        ("foo.pyi", "Python"),
        ("foo.cpp", "C/C++"),
        ("foo.cc", "C/C++"),
        ("foo.h", "C/C++"),
        ("foo.hpp", "C/C++"),
        ("foo.jl", "Julia"),
        ("foo.f90", "Fortran"),
        ("foo.F90", "Fortran"),
        ("foo.txt", "other"),
    ])
    def test_classification(self, name, expected):
        assert _classify_lang(Path("/some/dir") / name) == expected


# --------------------------------------------------------------------------
# _path_priority
# --------------------------------------------------------------------------

class TestPathPriority:
    def test_lowercase_production_dirs_rank_0(self):
        # src/ and inst/ are already lowercase so they hit the priority-0 branch.
        assert _path_priority("src/kernel.cpp") == 0
        assert _path_priority("inst/patch.R") == 0

    def test_uppercase_R_dir_quirk(self):
        # NOTE (latent bug): _path_priority lowercases rel_path before the
        # production check `parts[0] in {"R", "src", "inst"}`. After lowercasing
        # "R/" becomes "r/", which is NOT in that uppercase set, so R-package
        # source under R/ is ranked 5 (generic Python-module tier), not 0.
        # Documenting current behavior, not endorsing it.
        assert _path_priority("R/foo.R") == 5
        assert _path_priority("R\\foo.R") == 5

    def test_tests_rank_20(self):
        assert _path_priority("tests/test_foo.py") == 20
        assert _path_priority("testthat/test-bar.R") == 20
        assert _path_priority("pkg/test_thing.py") == 20

    def test_docs_rank_30(self):
        assert _path_priority("vignettes/intro.Rmd") == 30
        assert _path_priority("docs/usage.py") == 30
        assert _path_priority("notebook.ipynb") == 30
        assert _path_priority("README.md") == 30

    def test_python_pkg_module_rank_5(self):
        # A module nested one+ level deep under a non-build top dir.
        assert _path_priority("mypkg/core.py") == 5

    def test_top_level_fallback_rank_10(self):
        assert _path_priority("toplevel.py") == 10

    def test_windows_test_separators_normalized(self):
        assert _path_priority("tests\\test_x.py") == 20


# --------------------------------------------------------------------------
# _iter_source_files  +  skip dirs
# --------------------------------------------------------------------------

class TestIterSourceFiles:
    def test_yields_known_extensions(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.R").write_text("x <- 1\n")
        (tmp_path / "c.cpp").write_text("int x;\n")
        (tmp_path / "ignore.bin").write_text("binary")
        out = sorted(p.name for p in _iter_source_files(tmp_path))
        assert out == ["a.py", "b.R", "c.cpp"]

    def test_yields_named_meta_files(self, tmp_path):
        (tmp_path / "DESCRIPTION").write_text("Package: foo\n")
        (tmp_path / "NAMESPACE").write_text("export(foo)\n")
        (tmp_path / "Makevars").write_text("PKG_CFLAGS=-fopenmp\n")
        out = sorted(p.name for p in _iter_source_files(tmp_path))
        assert out == ["DESCRIPTION", "Makevars", "NAMESPACE"]

    def test_skips_vendored_dirs(self, tmp_path):
        (tmp_path / "good.py").write_text("x=1\n")
        for skip in (".git", "node_modules", "build", "__pycache__", ".venv"):
            d = tmp_path / skip
            d.mkdir()
            (d / "hidden.py").write_text("x=1\n")
        out = sorted(p.name for p in _iter_source_files(tmp_path))
        assert out == ["good.py"]


# --------------------------------------------------------------------------
# _scan_file — pattern detection
# --------------------------------------------------------------------------

def _scan_text(tmp_path: Path, name: str, text: str) -> Inventory:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    inv = Inventory(repo_path=tmp_path)
    _scan_file(p, tmp_path, inv)
    return inv


class TestScanFile:
    def test_mclapply_detected(self, tmp_path):
        inv = _scan_text(tmp_path, "R/x.R", "res <- mclapply(seq_len(10), f)\n")
        assert any(h.backend == "mclapply" for h in inv.hits)

    def test_openmp_pragma_detected(self, tmp_path):
        inv = _scan_text(tmp_path, "src/k.cpp", "#pragma omp parallel for\n")
        assert any(h.backend == "openmp" for h in inv.hits)

    def test_joblib_and_njobs(self, tmp_path):
        inv = _scan_text(tmp_path, "pkg/m.py",
                         "from joblib import Parallel\nout = f(n_jobs=4)\n")
        backends = {h.backend for h in inv.hits}
        assert "joblib" in backends

    def test_numba_decorator(self, tmp_path):
        inv = _scan_text(tmp_path, "pkg/m.py", "@njit\ndef f(): pass\n")
        assert any(h.backend == "numba" for h in inv.hits)

    def test_cuda_detected(self, tmp_path):
        inv = _scan_text(tmp_path, "pkg/m.py", "x = tensor.cuda()\n")
        assert any(h.backend == "cuda" for h in inv.hits)

    def test_env_thread_var_read(self, tmp_path):
        inv = _scan_text(tmp_path, "pkg/m.py",
                         "import os\nn = os.environ.get('OMP_NUM_THREADS')\n")
        assert any(h.backend == "env_thread_vars" for h in inv.hits)

    def test_comment_lines_skipped_python(self, tmp_path):
        # A commented-out mclapply-like line must not poison detection.
        inv = _scan_text(tmp_path, "pkg/m.py", "# out = f(n_jobs=4)\n")
        assert all(h.backend != "joblib" for h in inv.hits)

    def test_comment_lines_skipped_c(self, tmp_path):
        inv = _scan_text(tmp_path, "src/k.cpp", "// #pragma omp parallel\n")
        assert all(h.backend != "openmp" for h in inv.hits)

    def test_files_scanned_and_lang_counts(self, tmp_path):
        inv = _scan_text(tmp_path, "pkg/m.py", "x = 1\n")
        assert inv.files_scanned == 1
        assert inv.files_by_lang.get("Python") == 1

    def test_huge_file_skipped(self, tmp_path):
        p = tmp_path / "big.py"
        p.write_text("x" + " " * 5_000_001)
        inv = Inventory(repo_path=tmp_path)
        _scan_file(p, tmp_path, inv)
        # Over the 5MB guard — no scanning happened.
        assert inv.files_scanned == 0

    def test_very_long_line_skipped(self, tmp_path):
        # Line > 2000 chars is skipped (minified data guard).
        line = "x = " + "mclapply(" * 300 + "\n"
        inv = _scan_text(tmp_path, "pkg/m.py", line)
        assert all(h.backend != "mclapply" for h in inv.hits)

    def test_hit_text_truncated_to_200(self, tmp_path):
        long_tail = "mclapply(f)  # " + "z" * 500
        inv = _scan_text(tmp_path, "R/x.R", long_tail + "\n")
        hit = next(h for h in inv.hits if h.backend == "mclapply")
        assert len(hit.text) <= 200


# --------------------------------------------------------------------------
# _scan_function_signatures — knob extraction
# --------------------------------------------------------------------------

class TestScanFunctionSignatures:
    def test_r_single_line_parallel_false(self, tmp_path):
        inv = Inventory(repo_path=tmp_path)
        text = "fitGAM <- function(counts, parallel = FALSE, nthreads = 1L) {\n}\n"
        _scan_function_signatures(text, "R/fit.R", "R", inv)
        knobs = {k.knob: k.default for k in inv.knobs}
        assert knobs.get("parallel") == "FALSE"
        assert knobs.get("nthreads") == "1L"
        assert all(k.function_hint == "fitGAM" for k in inv.knobs)

    def test_r_multiline_signature_block(self, tmp_path):
        inv = Inventory(repo_path=tmp_path)
        text = (
            "fitGAM <- function(counts,\n"
            "                   x,\n"
            "                   parallel = FALSE,\n"
            "                   BPPARAM = bpparam()) {\n"
            "  body\n"
            "}\n"
        )
        _scan_function_signatures(text, "R/fit.R", "R", inv)
        knobs = {k.knob: k.default for k in inv.knobs}
        assert knobs.get("parallel") == "FALSE"
        assert knobs.get("BPPARAM") == "bpparam()"

    def test_python_def_with_njobs(self, tmp_path):
        inv = Inventory(repo_path=tmp_path)
        text = "def run(x, n_jobs=-1, workers=4): pass\n"
        _scan_function_signatures(text, "pkg/m.py", "Python", inv)
        knobs = {k.knob: k.default for k in inv.knobs}
        assert knobs.get("n_jobs") == "-1"
        assert knobs.get("workers") == "4"

    def test_python_bool_default_is_captured(self, tmp_path):
        # B4 fix: the default-value regex now matches Python's mixed-case
        # True/False (in addition to R-style FALSE/TRUE), so a `parallel` knob
        # defaulting to Python `False` yields a knob row with default "False".
        inv = Inventory(repo_path=tmp_path)
        text = "def run(parallel: bool = False): pass\n"
        _scan_function_signatures(text, "pkg/m.py", "Python", inv)
        knobs = {k.knob: k.default for k in inv.knobs}
        assert knobs.get("parallel") == "False"
        # R-style booleans must still work.
        inv2 = Inventory(repo_path=tmp_path)
        _scan_function_signatures("run <- function(parallel = TRUE) {}\n",
                                  "pkg/m.R", "R", inv2)
        assert {k.knob: k.default for k in inv2.knobs}.get("parallel") == "TRUE"

    def test_python_numeric_default_captured(self, tmp_path):
        inv = Inventory(repo_path=tmp_path)
        text = "def run(n_jobs: int = -1): pass\n"
        _scan_function_signatures(text, "pkg/m.py", "Python", inv)
        # numeric default IS captured even with a type annotation.
        knobs = {k.knob: k.default for k in inv.knobs}
        assert knobs.get("n_jobs") == "-1"

    def test_non_knob_kwargs_ignored(self, tmp_path):
        inv = Inventory(repo_path=tmp_path)
        text = "def run(alpha=0.5, beta='x'): pass\n"
        _scan_function_signatures(text, "pkg/m.py", "Python", inv)
        assert inv.knobs == []

    def test_other_langs_noop(self, tmp_path):
        inv = Inventory(repo_path=tmp_path)
        _scan_function_signatures("anything parallel=FALSE", "x.cpp", "C/C++", inv)
        assert inv.knobs == []


# --------------------------------------------------------------------------
# _parse_r_description
# --------------------------------------------------------------------------

class TestParseRDescription:
    def test_single_line_imports(self, tmp_path):
        p = tmp_path / "DESCRIPTION"
        p.write_text("Package: foo\nImports: mgcv, Matrix\n")
        out = _parse_r_description(p)
        assert out["Imports"] == {"mgcv", "Matrix"}

    def test_multiline_continuation(self, tmp_path):
        p = tmp_path / "DESCRIPTION"
        p.write_text(
            "Package: foo\n"
            "Imports:\n"
            "    mgcv (>= 1.8),\n"
            "    Matrix,\n"
            "    RcppArmadillo\n"
            "Suggests: testthat\n"
        )
        out = _parse_r_description(p)
        assert out["Imports"] == {"mgcv", "Matrix", "RcppArmadillo"}
        assert out["Suggests"] == {"testthat"}

    def test_version_specifiers_stripped(self, tmp_path):
        p = tmp_path / "DESCRIPTION"
        p.write_text("LinkingTo: RcppParallel (>= 5.0.0)\n")
        out = _parse_r_description(p)
        assert out["LinkingTo"] == {"RcppParallel"}

    def test_bare_R_dropped(self, tmp_path):
        p = tmp_path / "DESCRIPTION"
        p.write_text("Depends: R (>= 4.0), mgcv\n")
        out = _parse_r_description(p)
        assert out["Depends"] == {"mgcv"}

    def test_nonexistent_field_absent(self, tmp_path):
        p = tmp_path / "DESCRIPTION"
        p.write_text("Package: foo\nTitle: A package\n")
        out = _parse_r_description(p)
        assert out == {}

    def test_block_terminated_by_non_field_line(self, tmp_path):
        # A non-indented line that isn't a `Field:` header (e.g. a bare blank
        # or stray text) exercises the `else: flush(); current_field = None`
        # block-reset branch between two target fields.
        p = tmp_path / "DESCRIPTION"
        p.write_text(
            "Imports: mgcv\n"
            "\n"
            "Depends: Matrix\n"
        )
        out = _parse_r_description(p)
        assert out["Imports"] == {"mgcv"}
        assert out["Depends"] == {"Matrix"}


# --------------------------------------------------------------------------
# _parse_python_deps
# --------------------------------------------------------------------------

class TestParsePythonDeps:
    def test_requirements_txt(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("numpy>=1.21\nscipy==1.10\n")
        deps = _parse_python_deps(tmp_path)
        assert "numpy" in deps
        assert "scipy" in deps

    def test_pyproject_quoted(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\ndependencies = ["numba", "scikit-learn>=1.0"]\n'
        )
        deps = _parse_python_deps(tmp_path)
        assert "numba" in deps
        assert "scikit-learn" in deps

    def test_quoted_metadata_values_excluded(self, tmp_path):
        # The quoted-string regex filters known metadata *values* (python,
        # version, name, license, author, ...). Here the real dep is torch.
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "version"\ndependencies = ["torch"]\n'
        )
        deps = _parse_python_deps(tmp_path)
        assert "torch" in deps
        assert "version" not in deps  # filtered metadata value

    def test_underscores_normalized_to_dash(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("scikit_learn>=1.0\n")
        deps = _parse_python_deps(tmp_path)
        assert "scikit-learn" in deps

    def test_no_dep_files(self, tmp_path):
        assert _parse_python_deps(tmp_path) == set()

    def test_toml_metadata_keys_not_leaked_as_deps(self, tmp_path):
        # B5 fix: a TOML/setup.cfg assignment `name = "..."` matches the bare-
        # requirements regex too (`=` is in its operator class), so the
        # metadata-key exclusion is now applied to that regex as well. The real
        # dep (pandas) survives; the assignment keys do not become deps.
        (tmp_path / "pyproject.toml").write_text(
            "[project]\n"
            'name = "myproj"\n'
            'version = "1.0"\n'
            'license = "MIT"\n'
            'requires-python = ">=3.10"\n'
            'dependencies = ["pandas>=2.0"]\n'
        )
        deps = _parse_python_deps(tmp_path)
        assert "pandas" in deps
        for key in ("name", "version", "license", "requires-python", "dependencies"):
            assert key not in deps


# --------------------------------------------------------------------------
# _infer_from_deps
# --------------------------------------------------------------------------

class TestInferFromDeps:
    def test_r_blas_inference(self):
        inv = Inventory(repo_path=Path("."))
        inv.deps_seen["R-Imports"] = {"mgcv", "irlba"}
        _infer_from_deps(inv)
        backends = {d.backend for d in inv.inferred}
        assert "blas" in backends

    def test_r_openmp_inference(self):
        inv = Inventory(repo_path=Path("."))
        inv.deps_seen["R-Imports"] = {"glmnet"}
        _infer_from_deps(inv)
        assert any(d.backend == "openmp" for d in inv.inferred)

    def test_r_rcpp_parallel_via_linkingto(self):
        inv = Inventory(repo_path=Path("."))
        inv.deps_seen["R-LinkingTo"] = {"RcppParallel"}
        _infer_from_deps(inv)
        assert any(d.backend == "rcpp_parallel" for d in inv.inferred)

    def test_r_data_table_inference(self):
        inv = Inventory(repo_path=Path("."))
        inv.deps_seen["R-Depends"] = {"data.table"}
        _infer_from_deps(inv)
        assert any(d.backend == "data_table" for d in inv.inferred)

    def test_python_blas_and_numba(self):
        inv = Inventory(repo_path=Path("."))
        inv.deps_seen["Python"] = {"numpy", "numba"}
        _infer_from_deps(inv)
        backends = {d.backend for d in inv.inferred}
        assert {"blas", "numba"} <= backends

    def test_python_openmp_via_numexpr(self):
        inv = Inventory(repo_path=Path("."))
        inv.deps_seen["Python"] = {"numexpr"}
        _infer_from_deps(inv)
        assert any(d.backend == "openmp" for d in inv.inferred)

    def test_no_deps_no_inference(self):
        inv = Inventory(repo_path=Path("."))
        _infer_from_deps(inv)
        assert inv.inferred == []


# --------------------------------------------------------------------------
# _hits_by_backend — grouping + priority sort
# --------------------------------------------------------------------------

class TestHitsByBackend:
    def test_groups_and_sorts_by_priority(self):
        inv = Inventory(repo_path=Path("."))
        inv.hits = [
            Hit("mclapply", "tests/test_x.R", 5, "mclapply(f)"),
            Hit("mclapply", "R/core.R", 12, "mclapply(g)"),
            Hit("openmp", "src/k.cpp", 3, "#pragma omp"),
        ]
        out = _hits_by_backend(inv)
        # production R/core.R (priority 0) sorts before tests (priority 20)
        assert out["mclapply"][0].file == "R/core.R"
        assert "openmp" in out

    def test_empty(self):
        inv = Inventory(repo_path=Path("."))
        assert _hits_by_backend(inv) == {}


# --------------------------------------------------------------------------
# call-chain tracing
# --------------------------------------------------------------------------

class TestCallChainTracing:
    def _make_repo(self, tmp_path) -> Path:
        (tmp_path / "R").mkdir()
        (tmp_path / "R" / "core.R").write_text(
            "run <- function(x) {\n"
            "  helper(x)\n"
            "}\n"
            "helper <- function(x) {\n"
            "  inner(x)\n"
            "}\n"
            "inner <- function(x) {\n"
            "  mclapply(x, f)\n"
            "}\n"
            "unrelated <- function(x) {\n"
            "  parLapply(cl, x, g)\n"
            "}\n"
        )
        return tmp_path

    def test_build_fn_index(self, tmp_path):
        repo = self._make_repo(tmp_path)
        idx = _build_fn_index(repo)
        assert {"run", "helper", "inner", "unrelated"} <= set(idx.keys())

    def test_extract_body_calls(self, tmp_path):
        repo = self._make_repo(tmp_path)
        idx = _build_fn_index(repo)
        calls = _extract_body_calls(repo, idx, "inner")
        assert "mclapply" in calls

    def test_trace_reachable_follows_chain(self, tmp_path):
        repo = self._make_repo(tmp_path)
        reach = _trace_reachable(repo, "run")
        assert {"run", "helper", "inner"} <= reach
        assert "unrelated" not in reach

    def test_build_fn_index_skips_tests_and_c(self, tmp_path):
        # Production R file is indexed; a tests/ file and a .cpp file are not
        # (priority > 5 / non-R-or-Python language → skipped).
        (tmp_path / "R").mkdir()
        (tmp_path / "R" / "core.R").write_text("prod <- function(x) { x }\n")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_x.R").write_text("teststub <- function() {}\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "k.cpp").write_text("int cfn() { return 0; }\n")
        idx = _build_fn_index(tmp_path)
        assert "prod" in idx
        assert "teststub" not in idx
        assert "cfn" not in idx

    def test_build_fn_index_r_s4_method(self, tmp_path):
        # setMethod("run", ...) registers an S4 method name in the index.
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "methods.R").write_text(
            'setMethod("runRCTD", "spacexr", function(object) {\n'
            "  mclapply(x, f)\n"
            "})\n"
        )
        idx = _build_fn_index(tmp_path)
        assert "runRCTD" in idx

    def test_trace_reachable_resolves_bare_name(self, tmp_path):
        # target given as Pkg::func — the bare last component is seeded too.
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "core.R").write_text(
            "runRCTD <- function(x) { helper(x) }\n"
            "helper <- function(x) { x }\n"
        )
        reach = _trace_reachable(tmp_path, "spacexr::runRCTD")
        assert "runRCTD" in reach
        assert "helper" in reach

    def test_extract_body_calls_python_module_dot_func(self, tmp_path):
        # Python `np.dot(...)` adds both `np.dot` and the bare `dot`.
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "m.py").write_text(
            "def run(x):\n"
            "    return np.dot(x, x)\n"
        )
        idx = _build_fn_index(tmp_path)
        calls = _extract_body_calls(tmp_path, idx, "run")
        assert "np.dot" in calls
        assert "dot" in calls

    def test_filter_knobs_by_target_splits_on_off_chain(self, tmp_path):
        repo = self._make_repo(tmp_path)
        inv = Inventory(repo_path=repo)
        inv.knobs = [
            DefaultKnob("parallel", "FALSE", "R/core.R", 1, "inner"),
            DefaultKnob("parallel", "FALSE", "R/core.R", 9, "unrelated"),
        ]
        filter_knobs_by_target(inv, repo, "run")
        on = [k.function_hint for k in inv.knobs]
        off = [k.function_hint for k in inv.off_chain_knobs]
        assert "inner" in on
        assert "unrelated" in off


# --------------------------------------------------------------------------
# scan() — full pipeline against a tiny on-disk repo
# --------------------------------------------------------------------------

class TestScanIntegration:
    def test_full_scan_r_repo(self, tmp_path):
        (tmp_path / "DESCRIPTION").write_text(
            "Package: foo\nImports: mgcv, RcppParallel\nLinkingTo: RcppParallel\n"
        )
        (tmp_path / "R").mkdir()
        (tmp_path / "R" / "core.R").write_text(
            "run <- function(x, parallel = FALSE) {\n"
            "  mclapply(x, f)\n"
            "}\n"
        )
        inv = scan(tmp_path)
        assert inv.files_scanned >= 2
        backends = {h.backend for h in inv.hits}
        assert "mclapply" in backends
        assert any(d.backend == "rcpp_parallel" for d in inv.inferred)
        assert "R-Imports" in inv.deps_seen
        # parallel=FALSE knob found in production code.
        assert any(k.knob == "parallel" for k in inv.knobs)

    def test_full_scan_python_repo(self, tmp_path):
        # requirements.txt lines need a version specifier to be parsed (the
        # parser keys off `[<>=!~]`); bare names are not captured.
        (tmp_path / "requirements.txt").write_text("numpy>=1.21\nnumba>=0.55\n")
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "m.py").write_text(
            "from joblib import Parallel\n"
            "def run(x, n_jobs=-1):\n"
            "    return f(x)\n"
        )
        inv = scan(tmp_path)
        backends = {h.backend for h in inv.hits}
        assert "joblib" in backends
        assert any(d.backend == "numba" for d in inv.inferred)


# --------------------------------------------------------------------------
# _suggest_yaml — draft YAML block
# --------------------------------------------------------------------------

class TestSuggestYaml:
    def test_detected_backend_emitted(self):
        inv = Inventory(repo_path=Path("."))
        inv.hits = [Hit("mclapply", "R/core.R", 12, "mclapply(f)")]
        out = _suggest_yaml(inv)
        assert "type: mclapply" in out
        assert "R/core.R:12" in out

    def test_cuda_excluded_from_yaml(self):
        inv = Inventory(repo_path=Path("."))
        inv.hits = [Hit("cuda", "pkg/m.py", 3, "x.cuda()")]
        out = _suggest_yaml(inv)
        # cuda maps to None in type_map — not emitted as a backend line.
        assert "type: cuda" not in out

    def test_env_vars_and_kwargs_excluded(self):
        inv = Inventory(repo_path=Path("."))
        inv.hits = [
            Hit("env_thread_vars", "pkg/m.py", 1, "os.environ['OMP_NUM_THREADS']"),
            Hit("kwargs_parallel", "pkg/m.py", 2, "nthreads=4"),
        ]
        out = _suggest_yaml(inv)
        assert "type: env_thread_vars" not in out
        assert "type: kwargs_parallel" not in out

    def test_inferred_only_backend_appended(self):
        inv = Inventory(repo_path=Path("."))
        inv.inferred = [DepInference("blas", "Python deps include numpy")]
        out = _suggest_yaml(inv)
        assert "type: blas" in out
        assert "implicit" in out

    def test_no_parallelism_emits_empty_list(self):
        inv = Inventory(repo_path=Path("."))
        out = _suggest_yaml(inv)
        assert "no parallelism detected" in out

    def test_fill_placeholders_present(self):
        inv = Inventory(repo_path=Path("."))
        out = _suggest_yaml(inv)
        assert "upstream_default_threads: <FILL>" in out
        assert "parallelism_class: <FILL>" in out


# --------------------------------------------------------------------------
# format_report — rendering
# --------------------------------------------------------------------------

class TestFormatReport:
    def test_empty_inventory(self):
        inv = Inventory(repo_path=Path("/repo"))
        out = format_report(inv)
        assert "Parallelism inventory: /repo" in out
        assert "DETECTED (direct source matches):" in out
        assert "(none)" in out
        assert "NOT DETECTED:" in out
        assert "CAVEATS:" in out

    def test_detected_section_lists_hits(self):
        inv = Inventory(repo_path=Path("/repo"))
        inv.files_scanned = 1
        inv.hits = [Hit("mclapply", "R/core.R", 12, "mclapply(f)")]
        out = format_report(inv)
        assert "mclapply — 1 match" in out
        assert "R/core.R:12" in out

    def test_gpu_section_when_cuda_present(self):
        inv = Inventory(repo_path=Path("/repo"))
        inv.hits = [Hit("cuda", "pkg/m.py", 3, "x.cuda()")]
        out = format_report(inv)
        assert "GPU PATHS DETECTED" in out
        assert "OUT OF SCOPE" in out

    def test_inferred_section(self):
        inv = Inventory(repo_path=Path("/repo"))
        inv.inferred = [DepInference("blas", "Python deps include numpy")]
        out = format_report(inv)
        assert "LIKELY (inferred from compiled deps):" in out
        assert "blas" in out

    def test_knobs_section_on_chain(self):
        inv = Inventory(repo_path=Path("/repo"))
        inv.knobs = [DefaultKnob("parallel", "FALSE", "R/fit.R", 3, "fitGAM")]
        out = format_report(inv)
        assert "USER-CONTROLLABLE KNOBS" in out
        assert "fitGAM() in R/fit.R:3" in out
        assert "parallel=FALSE" in out

    def test_knobs_section_off_chain_only(self):
        inv = Inventory(repo_path=Path("/repo"))
        inv.off_chain_knobs = [DefaultKnob("parallel", "FALSE", "R/x.R", 3, "other")]
        out = format_report(inv)
        assert "none on target call chain" in out

    def test_knobs_none_found(self):
        inv = Inventory(repo_path=Path("/repo"))
        out = format_report(inv)
        assert "none found in function signatures" in out

    def test_informational_env_and_kwargs(self):
        inv = Inventory(repo_path=Path("/repo"))
        inv.hits = [
            Hit("env_thread_vars", "pkg/m.py", 1, "os.environ['OMP_NUM_THREADS']"),
            Hit("kwargs_parallel", "pkg/m.py", 2, "nthreads=4"),
        ]
        out = format_report(inv)
        assert "Env thread-var reads" in out
        assert "Parallelism kwargs" in out
        assert "informational" in out

    def test_max_hits_truncation(self):
        inv = Inventory(repo_path=Path("/repo"))
        inv.hits = [Hit("mclapply", "R/core.R", i, "mclapply(f)") for i in range(10)]
        out = format_report(inv, max_hits_per_backend=3)
        assert "+7 more" in out

    def test_gpu_section_truncates(self):
        inv = Inventory(repo_path=Path("/repo"))
        inv.hits = [Hit("cuda", "pkg/m.py", i, "x.cuda()") for i in range(10)]
        out = format_report(inv, max_hits_per_backend=2)
        assert "+8 more" in out

    def test_informational_section_truncates(self):
        inv = Inventory(repo_path=Path("/repo"))
        inv.hits = [
            Hit("env_thread_vars", "pkg/m.py", i, "os.environ['OMP_NUM_THREADS']")
            for i in range(10)
        ]
        out = format_report(inv, max_hits_per_backend=2)
        assert "+8 more" in out

    def test_knobs_on_chain_with_off_chain_count(self):
        inv = Inventory(repo_path=Path("/repo"))
        inv.knobs = [DefaultKnob("parallel", "FALSE", "R/a.R", 1, "f1")]
        inv.off_chain_knobs = [
            DefaultKnob("n_jobs", "1", "R/b.R", 2, "f2"),
            DefaultKnob("workers", "4", "R/c.R", 3, "f3"),
        ]
        out = format_report(inv)
        assert "on call chain" in out
        assert "outside target call chain" in out


# --------------------------------------------------------------------------
# dataclass smoke
# --------------------------------------------------------------------------

class TestDataclasses:
    def test_inventory_defaults(self):
        inv = Inventory(repo_path=Path("/r"))
        assert inv.files_scanned == 0
        assert inv.hits == []
        assert inv.inferred == []
        assert inv.knobs == []
        assert inv.off_chain_knobs == []
        assert inv.deps_seen == {}
        assert inv.files_by_lang == {}

    def test_hit_fields(self):
        h = Hit("openmp", "src/k.cpp", 5, "#pragma omp")
        assert (h.backend, h.file, h.line, h.text) == ("openmp", "src/k.cpp", 5, "#pragma omp")
