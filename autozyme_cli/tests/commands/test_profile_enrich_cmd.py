"""Unit tests for zyme.commands.profile.enrich — pure layer classification,
call-count estimation, and the enrich() aggregation step.

No R subprocess is spawned: enrich() is driven on pre-built profile dicts,
and build_r_namespace_index (the only subprocess path) is exercised via a
monkeypatched subprocess.run so the JSON-shaped output is supplied directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from zyme.commands.profile import enrich


# ---------------------------------------------------------------------------
# parse_target_pkg_from_task_yaml
# ---------------------------------------------------------------------------

def test_parse_target_pkg_r_double_colon(tmp_path):
    (tmp_path / "task.yaml").write_text("target_function: tradeSeq::fitGAM\n")
    assert enrich.parse_target_pkg_from_task_yaml(tmp_path) == "tradeSeq"


def test_parse_target_pkg_py_dotted(tmp_path):
    (tmp_path / "task.yaml").write_text("target_function: scanpy.pp.normalize\n")
    assert enrich.parse_target_pkg_from_task_yaml(tmp_path) == "scanpy"


def test_parse_target_pkg_bare(tmp_path):
    (tmp_path / "task.yaml").write_text("target_function: foo\n")
    assert enrich.parse_target_pkg_from_task_yaml(tmp_path) == "foo"


def test_parse_target_pkg_missing_yaml(tmp_path):
    assert enrich.parse_target_pkg_from_task_yaml(tmp_path) is None


def test_parse_target_pkg_no_field(tmp_path):
    (tmp_path / "task.yaml").write_text("target_repo: https://x\n")
    assert enrich.parse_target_pkg_from_task_yaml(tmp_path) is None


def test_parse_target_pkg_placeholder_skipped(tmp_path):
    (tmp_path / "task.yaml").write_text("target_function: <fill-me-in>\n")
    assert enrich.parse_target_pkg_from_task_yaml(tmp_path) is None


# ---------------------------------------------------------------------------
# count_r_calls — pure Rprof.out stack-transition counting
# ---------------------------------------------------------------------------

def test_count_r_calls_counts_fresh_entries(tmp_path):
    prof = tmp_path / "Rprof.out"
    # header line then sample stacks (leaf-first). A function entering after
    # being absent in the prior sample is a fresh entry.
    prof.write_text(
        "sample.interval=20000\n"
        '"a" "b"\n'      # a,b enter -> a:1 b:1
        '"a" "b"\n'      # both present in prev -> no new
        '"c" "b"\n'      # c enters -> c:1 (b still present)
        '"a" "b"\n'      # a re-enters -> a:2
    )
    counts = enrich.count_r_calls(prof)
    assert counts["a"] == 2
    assert counts["b"] == 1
    assert counts["c"] == 1


def test_count_r_calls_blank_line_resets(tmp_path):
    prof = tmp_path / "Rprof.out"
    prof.write_text(
        "sample.interval=20000\n"
        '"a"\n'
        "\n"        # blank resets prev_set
        '"a"\n'     # a re-counted as fresh
    )
    counts = enrich.count_r_calls(prof)
    assert counts["a"] == 2


def test_count_r_calls_missing_file(tmp_path):
    assert enrich.count_r_calls(tmp_path / "absent.out") == {}


def test_count_r_calls_empty_file(tmp_path):
    prof = tmp_path / "Rprof.out"
    prof.write_text("")
    assert enrich.count_r_calls(prof) == {}


# ---------------------------------------------------------------------------
# _classify_r_function
# ---------------------------------------------------------------------------

def test_classify_r_primitive():
    assert enrich._classify_r_function(".Call", None) == "primitive"
    assert enrich._classify_r_function(".Fortran(foo)", None) == "primitive"
    assert enrich._classify_r_function(".External2", None) == "primitive"


def test_classify_r_namespace_qualified():
    assert enrich._classify_r_function("Matrix::crossprod", None) == "library:Matrix"
    assert enrich._classify_r_function("base::sum", None) == "base-r"
    assert enrich._classify_r_function("stats::lm", None) == "base-r"


def test_classify_r_target_pkg_via_namespace():
    assert enrich._classify_r_function("tradeSeq::fitGAM", "tradeSeq") == "task"


def test_classify_r_native_symbol():
    # libRblas / libRlapack are BLAS primitives.
    assert enrich._classify_r_function("libRblas.0.dylib:dgemm", None) == "primitive"
    # libR core -> base-r.
    assert enrich._classify_r_function("libR.dylib:Rf_eval", None) == "base-r"
    # a real package .so -> library:<pkg>.
    assert enrich._classify_r_function("Matrix.so:symbol", None) == "library:Matrix"
    # target package native symbol -> task.
    assert enrich._classify_r_function("mypkg.so:kernel", "mypkg") == "task"


def test_classify_r_native_inline_rcpp_is_base_r():
    # sourceCpp_N (inline Rcpp) collapses to "" -> base-r.
    assert enrich._classify_r_function("sourceCpp_4.so:foo", None) == "base-r"


def test_classify_r_namespace_index():
    idx = {"fitPixels": "spacexr"}
    assert enrich._classify_r_function("fitPixels", None, idx) == "library:spacexr"
    assert enrich._classify_r_function("fitPixels", "spacexr", idx) == "task"


def test_classify_r_namespace_index_base_r():
    # ns_index resolves a function to a base-R package -> base-r.
    idx = {"lm": "stats"}
    assert enrich._classify_r_function("lm", None, idx) == "base-r"


def test_classify_r_anonymous():
    assert enrich._classify_r_function("<Anonymous>", None) == "anonymous"
    assert enrich._classify_r_function("FUN", None) == "anonymous"


def test_classify_r_closure_heuristic_mgcv():
    assert enrich._classify_r_function("family$ls", None) == "library:mgcv"
    assert enrich._classify_r_function("family$ls", "mgcv") == "task"


def test_classify_r_unknown_and_empty():
    assert enrich._classify_r_function("randomfn", None) == "unknown"
    assert enrich._classify_r_function("", None) == "unknown"


# ---------------------------------------------------------------------------
# _extract_module_from_builtin_name
# ---------------------------------------------------------------------------

def test_extract_module_builtin_form1():
    name = "<built-in method numpy.core._multiarray_umath.implement_array_function>"
    assert enrich._extract_module_from_builtin_name(name) == "numpy"


def test_extract_module_builtin_form2():
    name = "<method 'reduce' of 'numpy.ufunc' objects>"
    assert enrich._extract_module_from_builtin_name(name) == "numpy"


def test_extract_module_builtin_builtins():
    assert enrich._extract_module_from_builtin_name(
        "<built-in method builtins.len>") == "builtins"


def test_extract_module_builtin_unparseable():
    assert enrich._extract_module_from_builtin_name("<lambda>") is None


# ---------------------------------------------------------------------------
# _classify_py_function
# ---------------------------------------------------------------------------

def test_classify_py_builtin_numpy():
    raw = {"file": "~", "func":
           "<built-in method numpy.core._multiarray_umath.implement_array_function>"}
    assert enrich._classify_py_function(raw, None) == "library:numpy"


def test_classify_py_builtin_base_py():
    raw = {"file": "~", "func": "<built-in method builtins.len>"}
    assert enrich._classify_py_function(raw, None) == "base-py"


def test_classify_py_builtin_unparseable():
    raw = {"file": "~", "func": "<lambda>"}
    assert enrich._classify_py_function(raw, None) == "builtin"


def test_classify_py_site_packages_library():
    raw = {"file": "/env/lib/python3.12/site-packages/scanpy/preprocessing.py",
           "func": "normalize"}
    assert enrich._classify_py_function(raw, None) == "library:scanpy"


def test_classify_py_site_packages_target_is_task():
    raw = {"file": "/env/lib/python3.12/site-packages/scanpy/pp.py",
           "func": "normalize"}
    assert enrich._classify_py_function(raw, "scanpy") == "task"


def test_classify_py_stdlib_base_py():
    # pickle IS in the curated _BASE_PY_MODULES set -> base-py.
    raw = {"file": "/usr/lib/python3.12/pickle.py", "func": "loads"}
    assert enrich._classify_py_function(raw, None) == "base-py"


def test_classify_py_stdlib_uncurated_is_library():
    # A stdlib module NOT in the curated base set is reported as library:<mod>.
    # (json is stdlib but intentionally outside _BASE_PY_MODULES.)
    raw = {"file": "/usr/lib/python3.12/json/decoder.py", "func": "decode"}
    assert enrich._classify_py_function(raw, None) == "library:json"


def test_classify_py_upstream_repo_is_task():
    raw = {"file": "/task/upstream_repo/src/foo.py", "func": "bar"}
    assert enrich._classify_py_function(raw, None) == "task"


def test_classify_py_user_script_matches_target():
    raw = {"file": "/task/myalgo.py", "func": "run"}
    assert enrich._classify_py_function(raw, "myalgo") == "task"


def test_classify_py_unknown_path():
    raw = {"file": "/random/place/script.py", "func": "go"}
    assert enrich._classify_py_function(raw, None) == "unknown"


def test_classify_py_builtin_target_pkg_is_task():
    raw = {"file": "~",
           "func": "<built-in method mypkg.core.kernel>"}
    assert enrich._classify_py_function(raw, "mypkg") == "task"


def test_classify_py_builtin_name_with_real_file():
    # second branch: a real file path but the func name is itself a
    # <built-in method ...> string -> module extracted from the name.
    raw = {"file": "/proj/run.py", "func": "<built-in method numpy.dot>"}
    assert enrich._classify_py_function(raw, None) == "library:numpy"


def test_classify_py_builtin_name_real_file_base_py():
    raw = {"file": "/proj/run.py", "func": "<built-in method builtins.len>"}
    assert enrich._classify_py_function(raw, None) == "base-py"


def test_classify_py_builtin_name_real_file_target_task():
    raw = {"file": "/proj/run.py", "func": "<built-in method mypkg.go>"}
    assert enrich._classify_py_function(raw, "mypkg") == "task"


def test_classify_py_builtin_name_real_file_unparseable():
    raw = {"file": "/proj/run.py", "func": "<method nonsense"}
    assert enrich._classify_py_function(raw, None) == "builtin"


# ---------------------------------------------------------------------------
# build_r_namespace_index — monkeypatch subprocess boundary.
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_build_r_namespace_index_parses_json(monkeypatch):
    import subprocess
    monkeypatch.setattr(
        subprocess, "run",
        lambda c, **k: _Result(0, json.dumps({"fitGAM": "tradeSeq"})))
    idx = enrich.build_r_namespace_index({"rscript": "Rscript"}, ["tradeSeq"])
    assert idx == {"fitGAM": "tradeSeq"}


def test_build_r_namespace_index_recovers_leading_noise(monkeypatch):
    import subprocess
    # leading warning text before the JSON brace should be recovered.
    monkeypatch.setattr(
        subprocess, "run",
        lambda c, **k: _Result(0, 'Loading required package: foo\n{"f": "pkg"}'))
    idx = enrich.build_r_namespace_index(None, [])
    assert idx == {"f": "pkg"}


def test_build_r_namespace_index_nonzero_rc(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda c, **k: _Result(1, "err"))
    assert enrich.build_r_namespace_index(None, []) == {}


def test_build_r_namespace_index_empty_output(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda c, **k: _Result(0, "  "))
    assert enrich.build_r_namespace_index(None, []) == {}


def test_build_r_namespace_index_subprocess_raises(monkeypatch):
    import subprocess

    def boom(cmd, **kw):
        raise FileNotFoundError("Rscript")

    monkeypatch.setattr(subprocess, "run", boom)
    assert enrich.build_r_namespace_index(None, []) == {}


# ---------------------------------------------------------------------------
# _extract_pipeline_function_names
# ---------------------------------------------------------------------------

def test_extract_pipeline_function_names_r(tmp_path):
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    (pipeline / "run.R").write_text(
        "my_helper <- function(x) { x + 1 }\n"
        "another = function() 2\n"
        "get_script_dir <- function() {}\n"   # explicitly excluded
    )
    names = enrich._extract_pipeline_function_names(tmp_path)
    assert "my_helper" in names
    assert "another" in names
    assert "get_script_dir" not in names


def test_extract_pipeline_function_names_cpp_export(tmp_path):
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    (pipeline / "run.R").write_text("noop <- function() {}\n")
    (pipeline / "kernel.cpp").write_text(
        "// [[Rcpp::export]]\n"
        "NumericVector fast_kernel(NumericVector x) { return x; }\n"
    )
    names = enrich._extract_pipeline_function_names(tmp_path)
    assert "fast_kernel" in names


def test_extract_pipeline_function_names_no_run_r(tmp_path):
    assert enrich._extract_pipeline_function_names(tmp_path) == set()


# ---------------------------------------------------------------------------
# enrich() — the public aggregation entry point.
# ---------------------------------------------------------------------------

def _py_profile(hotspots):
    return {"lang": "py", "hotspots": hotspots, "schema_version": "1"}


def test_enrich_empty_hotspots_attaches_empty_aggregates():
    data = _py_profile([])
    enrich.enrich(data, target_pkg=None)
    assert data["layer_breakdown"] == []
    assert data["per_layer_top"] == {}
    assert data["schema_version"] == "2"


def test_enrich_py_tags_layers_and_promotes_ncalls():
    data = _py_profile([
        {"label": "scanpy/pp.py:10:normalize", "self_time_s": 0.6, "self_pct": 60.0,
         "raw": {"file": "/env/site-packages/scanpy/pp.py", "func": "normalize",
                 "ncalls": 3}},
        {"label": "~:0:len", "self_time_s": 0.4, "self_pct": 40.0,
         "raw": {"file": "~", "func": "<built-in method builtins.len>"}},
    ])
    enrich.enrich(data, target_pkg="scanpy")
    h0, h1 = data["hotspots"]
    assert h0["layer"] == "task"
    assert h0["layer_group"] == "task"
    assert h0["n_calls"] == 3
    assert h1["layer"] == "base-py"
    # layer_breakdown sums self_time per layer_group, sorted descending.
    groups = {b["layer_group"]: b for b in data["layer_breakdown"]}
    assert groups["task"]["self_time_s"] == 0.6
    assert groups["base-py"]["self_time_s"] == 0.4
    assert groups["task"]["pct"] == pytest.approx(60.0)
    # per_layer_top has up to 5 per group.
    assert data["per_layer_top"]["task"][0]["label"] == "scanpy/pp.py:10:normalize"
    assert data["schema_version"] == "2"


def test_enrich_collapses_library_layer_group():
    data = _py_profile([
        {"label": "numpy", "self_time_s": 0.5, "self_pct": 50.0,
         "raw": {"file": "/env/site-packages/numpy/core.py", "func": "dot"}},
    ])
    enrich.enrich(data, target_pkg=None)
    assert data["hotspots"][0]["layer"] == "library:numpy"
    assert data["hotspots"][0]["layer_group"] == "library"


def test_enrich_python_native_split_from_scalene_fields():
    data = _py_profile([
        {"label": "f", "self_time_s": 1.0, "self_pct": 100.0,
         "raw": {"file": "/env/site-packages/np/x.py", "func": "f",
                 "cpu_python_pct": 25.0, "cpu_native_pct": 75.0}},
    ])
    enrich.enrich(data, target_pkg=None)
    pns = data["python_native_split"]
    assert pns["python_s"] == pytest.approx(0.25)
    assert pns["native_s"] == pytest.approx(0.75)
    assert pns["python_pct"] == pytest.approx(25.0)
    assert pns["native_pct"] == pytest.approx(75.0)


def test_enrich_no_python_native_split_when_absent():
    data = _py_profile([
        {"label": "f", "self_time_s": 1.0, "self_pct": 100.0,
         "raw": {"file": "/env/site-packages/np/x.py", "func": "f"}},
    ])
    enrich.enrich(data, target_pkg=None)
    assert "python_native_split" not in data


def test_enrich_r_uses_call_counts_and_pipeline_fns(tmp_path, monkeypatch):
    # Build an Rprof.out so count_r_calls returns real estimates.
    prof = tmp_path / "Rprof.out"
    prof.write_text(
        "sample.interval=20000\n"
        '"my_helper" "run"\n'
        '"my_helper" "run"\n'
        '"Matrix::crossprod" "run"\n'
    )
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    (pipeline / "run.R").write_text("my_helper <- function() {}\n")

    # Avoid spawning R in the namespace indexer.
    monkeypatch.setattr(enrich, "build_r_namespace_index", lambda ex, aux: {})

    data = {
        "lang": "R",
        "schema_version": "1",
        "hotspots": [
            {"label": "my_helper", "self_time_s": 0.4, "self_pct": 40.0, "raw": {}},
            {"label": "Matrix::crossprod", "self_time_s": 0.6, "self_pct": 60.0,
             "raw": {}},
        ],
    }
    enrich.enrich(data, target_pkg="mypkg", rprof_path=prof, task_dir=tmp_path)
    h0, h1 = data["hotspots"]
    # my_helper is in pipeline/run.R -> task layer.
    assert h0["layer"] == "task"
    assert h0["n_calls_est"] == 1  # the 'run' frame counts 'my_helper' once
    # Matrix:: -> library.
    assert h1["layer"] == "library:Matrix"
    assert data["schema_version"] == "2"
