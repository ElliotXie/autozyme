"""Wave-3 coverage for zyme.commands.package.lint_rules.

tests/test_package_lint.py covers each rule's primary trap/fix sample. This
file fills the reachable internals the existing file leaves uncovered:

  - LintFinding.format_one
  - _smoke_load_block (found / not-found / unbalanced)
  - every rule's `language != ...` early-bail
  - _ast_for: cache reuse + SyntaxError -> None
  - _is_numba_threads_setdefault: non-Expr / non-Call / wrong-attr / no-os /
    no-args / wrong-key branches
  - _is_tf_in_sysmodules_test: non-Compare / multi-op / non-In / wrong-side
  - _uses_parallel_numba: njit(parallel=True), numba.jit(parallel=True),
    `from numba import prange`, bare prange Name, and the no-parallel case
  - rule_py_numba_tf_guard: parallel-but-no-numba-import -> no finding
  - rule_py_cache_clear_scope: non-cachey receiver skip, cache_clear attr,
    _CACHE/_cache/lru_cache receivers, non-Expr/non-attr-call skips
  - _attr_chain_tail: Attribute / Name / other

All AST-driven, no subprocess.
"""
from __future__ import annotations

import ast
from pathlib import Path

from zyme.commands.package import lint_rules as lr
from zyme.commands.package.lint_rules import (
    LintContext,
    LintFinding,
    _ast_for,
    _attr_chain_tail,
    _is_numba_threads_setdefault,
    _is_tf_in_sysmodules_test,
    _smoke_load_block,
    _uses_parallel_numba,
    rule_py_cache_clear_scope,
    rule_py_numba_tf_guard,
    rule_r_datatable_ns,
    rule_r_env_asnamespace,
    rule_r_library_guard,
    rule_r_pipe_in_mclapply,
    rule_r_rcpp_inline,
    rule_r_setmethod_string,
)


def _py_ctx(text: str) -> LintContext:
    return LintContext(
        patch_name="t", patch_file=Path("t/__init__.py"),
        patch_text=text, language="py",
    )


def _r_ctx(text: str) -> LintContext:
    return LintContext(
        patch_name="t", patch_file=Path("t/patch.R"),
        patch_text=text, language="R",
    )


def _expr(src: str) -> ast.stmt:
    """Parse a single statement and return it."""
    return ast.parse(src).body[0]


# --------------------------------------------------------------------------
# LintFinding.format_one
# --------------------------------------------------------------------------

def test_finding_format_one():
    f = LintFinding(rule_id="R-X", severity="FAIL",
                    file=Path("a/patch.R"), line=12, message="boom")
    out = f.format_one()
    assert out == "[FAIL] R-X a/patch.R:12: boom"


# --------------------------------------------------------------------------
# _smoke_load_block
# --------------------------------------------------------------------------

class TestSmokeLoadBlock:
    def test_not_found(self):
        assert _smoke_load_block("no load function here") is None

    def test_extracts_balanced_body(self):
        text = (
            "x <- 1\n"
            "load = function(td, tier) {\n"
            "  library(MAST)\n"
            "  do_thing()\n"
            "}\n"
        )
        out = _smoke_load_block(text)
        assert out is not None
        body, start_line = out
        assert "library(MAST)" in body
        # the `load = function` is on line 2.
        assert start_line == 2

    def test_unbalanced_braces_returns_none(self):
        # opening brace never closes -> depth stays > 0.
        text = "load = function(td) {\n  unclosed(\n"
        assert _smoke_load_block(text) is None

    def test_nested_braces_handled(self):
        text = "load = function(x) { if (a) { b() } ; c() }"
        out = _smoke_load_block(text)
        assert out is not None
        body, _ = out
        assert "c()" in body


# --------------------------------------------------------------------------
# Every rule bails on the wrong language
# --------------------------------------------------------------------------

class TestLanguageBails:
    def test_r_rules_skip_py(self):
        ctx = _py_ctx("x = 1")
        assert rule_r_library_guard(ctx) == []
        assert rule_r_rcpp_inline(ctx) == []
        assert rule_r_setmethod_string(ctx) == []
        assert rule_r_datatable_ns(ctx) == []
        assert rule_r_env_asnamespace(ctx) == []
        assert rule_r_pipe_in_mclapply(ctx) == []

    def test_py_rules_skip_r(self):
        ctx = _r_ctx("x <- 1")
        assert rule_py_numba_tf_guard(ctx) == []
        assert rule_py_cache_clear_scope(ctx) == []


# --------------------------------------------------------------------------
# _ast_for
# --------------------------------------------------------------------------

class TestAstFor:
    def test_wrong_language_returns_none(self):
        assert _ast_for(_r_ctx("x <- 1")) is None

    def test_parses_and_caches(self):
        ctx = _py_ctx("x = 1\n")
        mod1 = _ast_for(ctx)
        assert isinstance(mod1, ast.Module)
        # cached on the ctx, returned again identically.
        assert ctx.py_ast is mod1
        assert _ast_for(ctx) is mod1

    def test_syntax_error_returns_none(self):
        assert _ast_for(_py_ctx("def (:\n")) is None


# --------------------------------------------------------------------------
# _is_numba_threads_setdefault
# --------------------------------------------------------------------------

class TestIsNumbaThreadsSetdefault:
    def test_true_case(self):
        stmt = _expr('os.environ.setdefault("NUMBA_NUM_THREADS", "1")')
        assert _is_numba_threads_setdefault(stmt) is True

    def test_not_an_expr(self):
        stmt = _expr("a = 1")
        assert _is_numba_threads_setdefault(stmt) is False

    def test_expr_not_a_call(self):
        stmt = _expr("a")  # bare name expression
        assert _is_numba_threads_setdefault(stmt) is False

    def test_wrong_method_name(self):
        stmt = _expr('os.environ.update({"x": "1"})')
        assert _is_numba_threads_setdefault(stmt) is False

    def test_not_os_environ_receiver(self):
        stmt = _expr('d.setdefault("NUMBA_NUM_THREADS", "1")')
        assert _is_numba_threads_setdefault(stmt) is False

    def test_no_args(self):
        stmt = _expr("os.environ.setdefault()")
        assert _is_numba_threads_setdefault(stmt) is False

    def test_wrong_key(self):
        stmt = _expr('os.environ.setdefault("OMP_NUM_THREADS", "1")')
        assert _is_numba_threads_setdefault(stmt) is False

    def test_setdefault_on_func_not_attribute(self):
        # call.func is a Name, not an Attribute -> bail.
        stmt = _expr('setdefault("NUMBA_NUM_THREADS", "1")')
        assert _is_numba_threads_setdefault(stmt) is False


# --------------------------------------------------------------------------
# _is_tf_in_sysmodules_test
# --------------------------------------------------------------------------

class TestIsTfInSysModulesTest:
    def _test_expr(self, src: str) -> ast.expr:
        node = ast.parse(src).body[0]
        assert isinstance(node, ast.If)
        return node.test

    def test_true_case(self):
        t = self._test_expr('if "tensorflow" in sys.modules: pass')
        assert _is_tf_in_sysmodules_test(t) is True

    def test_not_a_compare(self):
        t = self._test_expr("if flag: pass")
        assert _is_tf_in_sysmodules_test(t) is False

    def test_multi_op_compare(self):
        t = self._test_expr("if 1 < x < 3: pass")
        assert _is_tf_in_sysmodules_test(t) is False

    def test_not_in_operator(self):
        t = self._test_expr('if "tensorflow" == sys.modules: pass')
        assert _is_tf_in_sysmodules_test(t) is False

    def test_wrong_left_constant(self):
        t = self._test_expr('if "torch" in sys.modules: pass')
        assert _is_tf_in_sysmodules_test(t) is False

    def test_wrong_right_side(self):
        t = self._test_expr('if "tensorflow" in some.other: pass')
        assert _is_tf_in_sysmodules_test(t) is False


# --------------------------------------------------------------------------
# _uses_parallel_numba
# --------------------------------------------------------------------------

class TestUsesParallelNumba:
    def test_njit_parallel_true(self):
        mod = ast.parse("@njit(parallel=True)\ndef f(): pass\n")
        assert _uses_parallel_numba(mod) is True

    def test_numba_jit_parallel_true(self):
        mod = ast.parse("@numba.jit(parallel=True)\ndef f(): pass\n")
        assert _uses_parallel_numba(mod) is True

    def test_from_numba_import_prange(self):
        mod = ast.parse("from numba import njit, prange\n")
        assert _uses_parallel_numba(mod) is True

    def test_bare_prange_name(self):
        mod = ast.parse("for i in prange(10):\n    pass\n")
        assert _uses_parallel_numba(mod) is True

    def test_serial_njit_no_parallel(self):
        mod = ast.parse("@njit\ndef f(): pass\n")
        assert _uses_parallel_numba(mod) is False

    def test_njit_parallel_false_not_flagged(self):
        mod = ast.parse("@njit(parallel=False)\ndef f(): pass\n")
        assert _uses_parallel_numba(mod) is False

    def test_unrelated_import_from_not_flagged(self):
        mod = ast.parse("from numba import njit\n")
        assert _uses_parallel_numba(mod) is False


# --------------------------------------------------------------------------
# rule_py_numba_tf_guard — parallel but numba import absent
# --------------------------------------------------------------------------

def test_numba_tf_guard_parallel_but_no_numba_import():
    # uses prange (parallel) but never `import numba` / `from numba import` at
    # top level -> numba_import_line stays None -> no finding (line 486-487).
    text = "for i in prange(10):\n    pass\n"
    assert rule_py_numba_tf_guard(_py_ctx(text)) == []


def test_numba_tf_guard_import_numba_module_form():
    # `import numba` (Import node) with parallel njit, no guard -> finding.
    text = "import numba\n@numba.njit(parallel=True)\ndef f(): pass\n"
    findings = rule_py_numba_tf_guard(_py_ctx(text))
    assert any(f.rule_id == "PY-NUMBA-TF-GUARD" for f in findings)


def test_numba_tf_guard_from_import_is_the_numba_line():
    # `from numba import ...` as the FIRST top-level numba reference (no
    # preceding `import numba`) -> the ImportFrom branch (lines 483-485) sets
    # numba_import_line. Parallel via njit(parallel=True), no guard -> finding.
    text = "from numba import njit, prange\n@njit(parallel=True)\ndef f(): pass\n"
    findings = rule_py_numba_tf_guard(_py_ctx(text))
    assert any(f.rule_id == "PY-NUMBA-TF-GUARD" for f in findings)


# --------------------------------------------------------------------------
# rule_py_cache_clear_scope — receiver heuristics
# --------------------------------------------------------------------------

class TestCacheClearScope:
    def test_lru_cache_receiver(self):
        text = "lru_cache.clear()\n"
        findings = rule_py_cache_clear_scope(_py_ctx(text))
        assert any(f.rule_id == "PY-CACHE-CLEAR-SCOPE" for f in findings)

    def test_lowercase_cache_suffix(self):
        text = "my_func_cache.clear()\n"
        findings = rule_py_cache_clear_scope(_py_ctx(text))
        assert len(findings) == 1

    def test_cache_clear_method_name(self):
        # cache_clear() (functools lru wrapper) also flagged.
        text = "SOME_CACHE.cache_clear()\n"
        findings = rule_py_cache_clear_scope(_py_ctx(text))
        assert len(findings) == 1

    def test_non_cachey_receiver_skipped(self):
        text = "my_list.clear()\n"
        assert rule_py_cache_clear_scope(_py_ctx(text)) == []

    def test_non_clear_method_skipped(self):
        text = "_ENTRY_POINT_CACHE.reset()\n"
        assert rule_py_cache_clear_scope(_py_ctx(text)) == []

    def test_non_expr_statement_skipped(self):
        text = "x = some_CACHE.clear()\n"  # assignment, not bare Expr
        assert rule_py_cache_clear_scope(_py_ctx(text)) == []

    def test_call_func_not_attribute_skipped(self):
        text = "clear()\n"  # plain Call, func is a Name
        assert rule_py_cache_clear_scope(_py_ctx(text)) == []

    def test_receiver_without_chain_tail_skipped(self):
        # receiver is a subscript-like expr with no Name/Attribute tail.
        text = "d['x'].clear()\n"
        assert rule_py_cache_clear_scope(_py_ctx(text)) == []

    def test_syntax_error_returns_empty(self):
        assert rule_py_cache_clear_scope(_py_ctx("def (:\n")) == []


# --------------------------------------------------------------------------
# _attr_chain_tail
# --------------------------------------------------------------------------

class TestAttrChainTail:
    def test_attribute(self):
        node = ast.parse("a.b.c").body[0].value
        assert _attr_chain_tail(node) == "c"

    def test_name(self):
        node = ast.parse("_CACHE").body[0].value
        assert _attr_chain_tail(node) == "_CACHE"

    def test_other_returns_none(self):
        node = ast.parse("d['k']").body[0].value  # Subscript
        assert _attr_chain_tail(node) is None
