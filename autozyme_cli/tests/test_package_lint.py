"""Per-rule lint tests.

Each rule has at least one trap sample (must FAIL) and one fix sample (must
PASS). The trap/fix bodies are minimal — just enough to trigger the regex /
AST match.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme.commands.package.lint_rules import (
    LintContext,
    rule_py_cache_clear_scope,
    rule_py_numba_tf_guard,
    rule_r_datatable_ns,
    rule_r_env_asnamespace,
    rule_r_library_guard,
    rule_r_pipe_in_mclapply,
    rule_r_rcpp_inline,
    rule_r_setmethod_string,
)


def _r_ctx(text: str, *, desc: str = "", ns: str = "") -> LintContext:
    return LintContext(
        patch_name="test_patch",
        patch_file=Path("test_patch/patch.R"),
        patch_text=text,
        language="R",
        description_path=Path("DESCRIPTION"),
        namespace_path=Path("NAMESPACE"),
        description_text=desc,
        namespace_text=ns,
    )


def _py_ctx(text: str) -> LintContext:
    return LintContext(
        patch_name="test_patch",
        patch_file=Path("test_patch/__init__.py"),
        patch_text=text,
        language="py",
    )


# --------------------------------------------------------------------------
# R-LIBRARY-GUARD
# --------------------------------------------------------------------------

def test_r_library_guard_traps_mast_without_library():
    text = """
    if (requireNamespace("MAST", quietly = TRUE)) {
      register_patch(name = "mast", upstream = "MAST",
        targets = list(lrTest = fast_lrTest),
        smoke = list(load = function(td, tier) { 1 })
      )
    }
    """
    findings = rule_r_library_guard(_r_ctx(text))
    assert any(f.rule_id == "R-LIBRARY-GUARD" for f in findings)


def test_r_library_guard_accepts_mast_with_library_anywhere():
    text = """
    if (requireNamespace("MAST", quietly = TRUE)) {
      .my_load <- function(td, tier) {
        suppressPackageStartupMessages(library(MAST))
        1
      }
      register_patch(name = "mast", upstream = "MAST",
        targets = list(lrTest = fast_lrTest),
        smoke = list(load = .my_load)
      )
    }
    """
    assert rule_r_library_guard(_r_ctx(text)) == []


def test_r_library_guard_skips_unrelated_upstream():
    text = 'register_patch(name = "vegan", upstream = "vegan")'
    assert rule_r_library_guard(_r_ctx(text)) == []


# --------------------------------------------------------------------------
# R-RCPP-INLINE
# --------------------------------------------------------------------------

def test_r_rcpp_inline_traps_cppfunction():
    text = 'Rcpp::cppFunction("IntegerVector f(IntegerVector x) { return x*2; }")'
    findings = rule_r_rcpp_inline(_r_ctx(text))
    assert any(f.rule_id == "R-RCPP-INLINE" for f in findings)


def test_r_rcpp_inline_traps_sourcecpp_code():
    text = 'sourceCpp(code = "void f() {}")'
    findings = rule_r_rcpp_inline(_r_ctx(text))
    assert any(f.rule_id == "R-RCPP-INLINE" for f in findings)


def test_r_rcpp_inline_accepts_no_inline_cpp():
    text = "fast_fn <- function(x) x * 2"
    assert rule_r_rcpp_inline(_r_ctx(text)) == []


# --------------------------------------------------------------------------
# R-SETMETHOD-STRING
# --------------------------------------------------------------------------

def test_r_setmethod_string_traps_literal_name():
    text = 'setMethod("getCurves", signature(x = "PseudotimeOrdering"), fast)'
    findings = rule_r_setmethod_string(_r_ctx(text))
    assert any(f.rule_id == "R-SETMETHOD-STRING" for f in findings)


def test_r_setmethod_string_accepts_function_object():
    text = 'setMethod(utils::getFromNamespace("getCurves", "slingshot"), sig, fast)'
    assert rule_r_setmethod_string(_r_ctx(text)) == []


# --------------------------------------------------------------------------
# R-DATATABLE-NS
# --------------------------------------------------------------------------

def test_r_datatable_ns_traps_missing_imports():
    text = 'dt[, col := value]'
    findings = rule_r_datatable_ns(_r_ctx(text, desc="Package: foo\n", ns=""))
    assert any(f.rule_id == "R-DATATABLE-NS" for f in findings)


def test_r_datatable_ns_accepts_when_properly_imported():
    desc = "Package: foo\nImports:\n    data.table,\n    Rcpp\n"
    ns = 'importFrom(data.table, ":=")\n'
    text = 'dt[, col := value]'
    assert rule_r_datatable_ns(_r_ctx(text, desc=desc, ns=ns)) == []


def test_r_datatable_ns_skips_patch_not_using_assign():
    text = "x <- 1"
    assert rule_r_datatable_ns(_r_ctx(text, desc="", ns="")) == []


# --------------------------------------------------------------------------
# R-ENV-ASNAMESPACE
# --------------------------------------------------------------------------

def test_r_env_asnamespace_traps_rebind():
    text = "environment(fast_fn) <- asNamespace(\"slingshot\")"
    findings = rule_r_env_asnamespace(_r_ctx(text))
    assert any(f.rule_id == "R-ENV-ASNAMESPACE" for f in findings)


def test_r_env_asnamespace_accepts_getfromnamespace():
    text = 'helper <- utils::getFromNamespace("helper", "slingshot")'
    assert rule_r_env_asnamespace(_r_ctx(text)) == []


# --------------------------------------------------------------------------
# R-PIPE-IN-MCLAPPLY
# --------------------------------------------------------------------------

def test_r_pipe_in_mclapply_traps_unbound_pipe():
    text = """
    parallel::mclapply(items, function(x) {
      x %>% transform()
    })
    """
    findings = rule_r_pipe_in_mclapply(_r_ctx(text))
    assert any(f.rule_id == "R-PIPE-IN-MCLAPPLY" for f in findings)


def test_r_pipe_in_mclapply_accepts_filescope_binding():
    text = """
    `%>%` <- dplyr::`%>%`
    parallel::mclapply(items, function(x) {
      x %>% transform()
    })
    """
    assert rule_r_pipe_in_mclapply(_r_ctx(text)) == []


def test_r_pipe_in_mclapply_skips_no_parallel():
    text = "x %>% y"
    assert rule_r_pipe_in_mclapply(_r_ctx(text)) == []


def test_r_pipe_in_mclapply_skips_parallel_without_pipe():
    text = 'parallel::mclapply(1:10, function(i) i * 2)'
    assert rule_r_pipe_in_mclapply(_r_ctx(text)) == []


# --------------------------------------------------------------------------
# PY-NUMBA-TF-GUARD
# --------------------------------------------------------------------------

def test_py_numba_tf_guard_traps_parallel_without_guard():
    text = """
import numba
from numba import njit, prange

@njit(parallel=True)
def f(x):
    for i in prange(len(x)):
        pass
"""
    findings = rule_py_numba_tf_guard(_py_ctx(text))
    assert any(f.rule_id == "PY-NUMBA-TF-GUARD" for f in findings)


def test_py_numba_tf_guard_accepts_with_guard():
    text = """
import os
import sys

if "tensorflow" in sys.modules:
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numba
from numba import njit, prange

@njit(parallel=True)
def f(x):
    for i in prange(len(x)):
        pass
"""
    assert rule_py_numba_tf_guard(_py_ctx(text)) == []


def test_py_numba_tf_guard_skips_serial_njit():
    text = """
import numba
from numba import njit

@njit
def f(x):
    return x * 2
"""
    assert rule_py_numba_tf_guard(_py_ctx(text)) == []


# --------------------------------------------------------------------------
# PY-CACHE-CLEAR-SCOPE
# --------------------------------------------------------------------------

def test_py_cache_clear_scope_traps_module_level_clear():
    text = """
from obspy.core.util.misc import _ENTRY_POINT_CACHE
_ENTRY_POINT_CACHE.clear()
"""
    findings = rule_py_cache_clear_scope(_py_ctx(text))
    assert any(f.rule_id == "PY-CACHE-CLEAR-SCOPE" for f in findings)


def test_py_cache_clear_scope_accepts_clear_inside_function():
    text = """
def fast_filter(self, *args):
    from obspy.core.util.misc import _ENTRY_POINT_CACHE
    _ENTRY_POINT_CACHE.clear()
    return self
"""
    assert rule_py_cache_clear_scope(_py_ctx(text)) == []
