"""Unit tests for zyme.hoist_audit — Python AST + R regex detection of work
hoisted outside the timer, the exempt reader, the JSONL logger, and the
violation formatter.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from zyme import hoist_audit as HA


# ---------------------------------------------------------------------------
# scan_pipeline_hoist — dispatch + edge cases
# ---------------------------------------------------------------------------

def test_scan_missing_file_returns_empty(tmp_path):
    assert HA.scan_pipeline_hoist(tmp_path / "nope.py") == []


def test_scan_unknown_suffix_returns_empty(tmp_path):
    p = tmp_path / "run.txt"
    p.write_text("anything")
    assert HA.scan_pipeline_hoist(p) == []


def _py(tmp_path, src):
    p = tmp_path / "run.py"
    p.write_text(src)
    return HA.scan_pipeline_hoist(p)


def _r(tmp_path, src):
    p = tmp_path / "run.R"
    p.write_text(src)
    return HA.scan_pipeline_hoist(p)


# ---------------------------------------------------------------------------
# Python AST checks
# ---------------------------------------------------------------------------

def test_py_no_timer_returns_empty(tmp_path):
    src = "import foo\nx = foo._private(data)\n"
    assert _py(tmp_path, src) == []


def test_py_syntax_error_returns_empty(tmp_path):
    assert _py(tmp_path, "def (:\n") == []


def test_py_private_alias_call_before_timer(tmp_path):
    src = (
        "import time\n"
        "from pkg._sub import _init as init_fn\n"
        "cached = init_fn(data)\n"
        "t0 = time.perf_counter()\n"
        "res = target(data)\n"
    )
    v = _py(tmp_path, src)
    assert len(v) == 1
    assert v[0]["pattern"] == "private_alias_call"
    assert v[0]["resolved"] == "pkg._sub._init"


def test_py_private_alias_from_assign(tmp_path):
    src = (
        "import time\n"
        "import pkg\n"
        "alias = pkg._mod._fn\n"
        "out = alias(x)\n"
        "t0 = time.perf_counter()\n"
        "r = target(x)\n"
    )
    v = _py(tmp_path, src)
    assert len(v) == 1
    assert v[0]["pattern"] == "private_alias_call"
    assert v[0]["resolved"] == "pkg._mod._fn"


def test_py_direct_private_attr_call_before_timer(tmp_path):
    src = (
        "import time\n"
        "import pkg\n"
        "val = pkg._internal._compute(data)\n"
        "t0 = time.perf_counter()\n"
        "r = target(data)\n"
    )
    v = _py(tmp_path, src)
    assert len(v) == 1
    assert v[0]["pattern"] == "private_attr_call"
    assert "pkg._internal._compute" == v[0]["resolved"]


def test_py_call_after_timer_not_flagged(tmp_path):
    src = (
        "import time\n"
        "import pkg\n"
        "t0 = time.perf_counter()\n"
        "val = pkg._internal._compute(data)\n"
    )
    assert _py(tmp_path, src) == []


def test_py_private_call_inside_function_not_flagged(tmp_path):
    # Definition before timer, but only called inside a helper body.
    src = (
        "import time\n"
        "import pkg\n"
        "def helper():\n"
        "    return pkg._internal._compute(data)\n"
        "t0 = time.perf_counter()\n"
        "r = target(data)\n"
    )
    assert _py(tmp_path, src) == []


def test_py_public_call_not_flagged(tmp_path):
    src = (
        "import time\n"
        "import pkg\n"
        "val = pkg.public.compute(data)\n"
        "t0 = time.perf_counter()\n"
        "r = target(data)\n"
    )
    assert _py(tmp_path, src) == []


def test_py_perf_counter_bare_import(tmp_path):
    # `perf_counter` unqualified still recognized as the timer.
    src = (
        "from time import perf_counter\n"
        "import pkg\n"
        "v = pkg._x._y(data)\n"
        "t0 = perf_counter()\n"
        "r = target(data)\n"
    )
    v = _py(tmp_path, src)
    assert len(v) == 1


def test_py_private_call_inside_if_before_timer_flagged(tmp_path):
    # The recursive _scan descends into module-level if blocks.
    src = (
        "import time\n"
        "import pkg\n"
        "if True:\n"
        "    v = pkg._x._y(data)\n"
        "t0 = time.perf_counter()\n"
        "r = target(data)\n"
    )
    v = _py(tmp_path, src)
    assert len(v) == 1


def test_py_private_call_inside_for_before_timer(tmp_path):
    src = (
        "import time\n"
        "import pkg\n"
        "for i in range(2):\n"
        "    v = pkg._x._y(data)\n"
        "t0 = time.perf_counter()\n"
        "r = target(data)\n"
    )
    assert len(_py(tmp_path, src)) == 1


def test_py_private_call_inside_while_before_timer(tmp_path):
    src = (
        "import time\n"
        "import pkg\n"
        "while False:\n"
        "    v = pkg._x._y(data)\n"
        "t0 = time.perf_counter()\n"
        "r = target(data)\n"
    )
    assert len(_py(tmp_path, src)) == 1


def test_py_private_call_inside_with_before_timer(tmp_path):
    src = (
        "import time\n"
        "import pkg\n"
        "with open('x') as fh:\n"
        "    v = pkg._x._y(data)\n"
        "t0 = time.perf_counter()\n"
        "r = target(data)\n"
    )
    assert len(_py(tmp_path, src)) == 1


def test_py_private_call_inside_try_before_timer(tmp_path):
    src = (
        "import time\n"
        "import pkg\n"
        "try:\n"
        "    v = pkg._x._y(data)\n"
        "except Exception:\n"
        "    w = pkg._a._b(data)\n"
        "else:\n"
        "    pass\n"
        "finally:\n"
        "    pass\n"
        "t0 = time.perf_counter()\n"
        "r = target(data)\n"
    )
    # both the try body and except body calls fire
    assert len(_py(tmp_path, src)) == 2


def test_py_call_without_name_chain_skipped(tmp_path):
    # A call whose func has no Name/Attribute chain (subscript result called)
    # is skipped (line 164 `continue`).
    src = (
        "import time\n"
        "registry = {}\n"
        "registry['fn'](data)\n"
        "t0 = time.perf_counter()\n"
        "r = target(data)\n"
    )
    assert _py(tmp_path, src) == []


def test_py_importfrom_private_module(tmp_path):
    # `from pkg._priv import thing` -> module private -> alias flagged.
    src = (
        "import time\n"
        "from pkg._priv import thing as t\n"
        "x = t(data)\n"
        "t0 = time.perf_counter()\n"
        "r = target(data)\n"
    )
    v = _py(tmp_path, src)
    assert len(v) == 1
    assert v[0]["resolved"] == "pkg._priv.thing"


# ---------------------------------------------------------------------------
# R regex checks
# ---------------------------------------------------------------------------

def test_r_no_timer_returns_empty(tmp_path):
    src = "x <- celda:::.decontxInitializeZ(counts)\n"
    assert _r(tmp_path, src) == []


def test_r_getns_alias_call_before_timer(tmp_path):
    src = (
        ".orig_initZ <- getFromNamespace('.decontxInitializeZ', 'celda')\n"
        "precomp <- .orig_initZ(counts)\n"
        "t0 <- Sys.time()\n"
        "res <- celda::decontX(counts)\n"
    )
    v = _r(tmp_path, src)
    assert len(v) == 1
    assert v[0]["pattern"] == "r_getns_call"
    assert v[0]["resolved"] == "celda:::.decontxInitializeZ"


def test_r_assignment_line_itself_not_flagged(tmp_path):
    # The getFromNamespace assignment line should not be a violation by itself.
    src = (
        ".orig_initZ <- getFromNamespace('.x', 'celda')\n"
        "t0 <- Sys.time()\n"
        "res <- celda::decontX(counts)\n"
    )
    assert _r(tmp_path, src) == []


def test_r_triple_colon_call_before_timer(tmp_path):
    src = (
        "precomp <- celda:::.decontxInitializeZ(counts)\n"
        "t0 <- Sys.time()\n"
        "res <- celda::decontX(counts)\n"
    )
    v = _r(tmp_path, src)
    assert any(x["pattern"] == "r_triple_colon" for x in v)
    assert v[0]["resolved"] == "celda:::.decontxInitializeZ"


def test_r_call_after_timer_not_flagged(tmp_path):
    src = (
        "t0 <- Sys.time()\n"
        "precomp <- celda:::.decontxInitializeZ(counts)\n"
    )
    assert _r(tmp_path, src) == []


def test_r_private_call_inside_braces_not_flagged(tmp_path):
    # Brace-stripping collapses override wrapper bodies -> not top-level.
    src = (
        "install_override('.fn', 'celda', function(...) {\n"
        "  celda:::.decontxInitializeZ(counts)\n"
        "})\n"
        "t0 <- Sys.time()\n"
        "res <- celda::decontX(counts)\n"
    )
    assert _r(tmp_path, src) == []


def test_r_proc_time_and_tic_recognized(tmp_path):
    for timer in ("proc.time()", "tictoc::tic()"):
        src = (
            "v <- pkg:::.priv(counts)\n"
            f"t0 <- {timer}\n"
            "res <- pkg::run(counts)\n"
        )
        v = _r(tmp_path, src)
        assert len(v) >= 1, timer


def test_r_comment_timer_ignored(tmp_path):
    # Sys.time() in a comment is not a real timer boundary.
    src = (
        "# Sys.time() reference\n"
        "v <- pkg:::.priv(counts)\n"
        "t0 <- Sys.time()\n"
        "res <- pkg::run(counts)\n"
    )
    v = _r(tmp_path, src)
    # the real timer is line 3; the call on line 2 is before it -> flagged.
    assert len(v) >= 1


def test_r_alias_substring_not_false_matched(tmp_path):
    # `.orig_initZ_compiled(` must NOT match alias `.orig_initZ`.
    src = (
        ".orig_initZ <- getFromNamespace('.x', 'celda')\n"
        "y <- .orig_initZ_compiled(counts)\n"
        "t0 <- Sys.time()\n"
        "res <- celda::decontX(counts)\n"
    )
    v = _r(tmp_path, src)
    assert all(x["pattern"] != "r_getns_call" for x in v)


# ---------------------------------------------------------------------------
# _strip_braced_blocks / _strip_comment helpers
# ---------------------------------------------------------------------------

def test_strip_comment():
    assert HA._strip_comment("x <- 1  # hi").rstrip() == "x <- 1"
    assert HA._strip_comment("# whole line").strip() == ""


def test_strip_braced_blocks_preserves_line_count():
    src = "a\nf <- function() {\n  inner()\n}\nb\n"
    out = HA._strip_braced_blocks(src)
    assert out.count("\n") == src.count("\n")
    assert "inner" not in out
    assert out.splitlines()[0] == "a"


def test_strip_braced_blocks_ignores_brace_in_string():
    src = 'x <- "{ not a block }"\ny <- 1\n'
    out = HA._strip_braced_blocks(src)
    # string content preserved at depth 0
    assert "{ not a block }" in out


def test_strip_braced_blocks_nested_and_string_inside():
    # nested braces (line 325-326) + a string with an escaped char inside a
    # block (lines 298-301) + plain chars inside the block (line 332).
    src = (
        'f <- function() {\n'
        '  g <- function() { paste("a\\"b{c}") }\n'
        '  inner_call()\n'
        '}\n'
        'after <- 1\n'
    )
    out = HA._strip_braced_blocks(src)
    assert out.count("\n") == src.count("\n")
    assert "inner_call" not in out
    assert "paste" not in out
    assert out.splitlines()[-1] == "after <- 1"


def test_strip_braced_blocks_escape_at_depth_zero():
    # backslash escape inside a top-level string (depth 0) -> preserved branch.
    src = 'x <- "a\\"b"\ny <- 1\n'
    out = HA._strip_braced_blocks(src)
    assert 'a\\"b' in out


# ---------------------------------------------------------------------------
# check_pipeline_hoist — raise wrapper
# ---------------------------------------------------------------------------

def test_check_pipeline_hoist_raises(tmp_path):
    p = tmp_path / "run.R"
    p.write_text(
        "precomp <- celda:::.decontxInitializeZ(counts)\n"
        "t0 <- Sys.time()\n"
        "res <- celda::decontX(counts)\n"
    )
    with pytest.raises(HA.HoistViolation):
        HA.check_pipeline_hoist(p)


def test_check_pipeline_hoist_clean_no_raise(tmp_path):
    p = tmp_path / "run.py"
    p.write_text("import time\nt0 = time.perf_counter()\ntarget(x)\n")
    HA.check_pipeline_hoist(p)  # no exception


# ---------------------------------------------------------------------------
# read_hoist_exempt
# ---------------------------------------------------------------------------

def test_read_hoist_exempt_missing_yaml(tmp_path):
    assert HA.read_hoist_exempt(tmp_path) is None


def test_read_hoist_exempt_absent_field(tmp_path):
    (tmp_path / "task.yaml").write_text("target_repo: x\ntarget_function: f\n")
    assert HA.read_hoist_exempt(tmp_path) is None


def test_read_hoist_exempt_present(tmp_path):
    (tmp_path / "task.yaml").write_text(
        "target_function: f\nhoist_exempt: target IS a cached lookup\n")
    assert HA.read_hoist_exempt(tmp_path) == "target IS a cached lookup"


def test_read_hoist_exempt_strips_quotes_and_comment(tmp_path):
    (tmp_path / "task.yaml").write_text(
        'hoist_exempt: "documented runtime precondition"  # note\n')
    assert HA.read_hoist_exempt(tmp_path) == "documented runtime precondition"


def test_read_hoist_exempt_empty_value_is_none(tmp_path):
    (tmp_path / "task.yaml").write_text("hoist_exempt:   \n")
    assert HA.read_hoist_exempt(tmp_path) is None


def test_read_hoist_exempt_oserror_returns_none(tmp_path, monkeypatch):
    # task.yaml exists but read_text raises OSError -> None (lines 137-138).
    (tmp_path / "task.yaml").write_text("hoist_exempt: x\n")
    orig = Path.read_text

    def boom(self, *a, **k):
        if self.name == "task.yaml":
            raise OSError("read fail")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", boom)
    assert HA.read_hoist_exempt(tmp_path) is None


# ---------------------------------------------------------------------------
# append_hoist_log
# ---------------------------------------------------------------------------

def test_append_hoist_log_writes_jsonl(tmp_path):
    violations = [{"pattern": "r_triple_colon", "line": 1, "code": "x", "resolved": "p:::q"}]
    HA.append_hoist_log(
        tmp_path, round_num=3, commit="abc123", pipeline_rel="pipeline/run.R",
        violations=violations, outcome="blocked", hypothesis="[algorithmic] Y",
    )
    log = tmp_path / ".zyme" / "hoist_log.jsonl"
    assert log.exists()
    rec = json.loads(log.read_text().strip())
    assert rec["round"] == 3
    assert rec["commit"] == "abc123"
    assert rec["outcome"] == "blocked"
    assert rec["violations"] == violations
    assert "ts" in rec


def test_append_hoist_log_swallows_oserror(tmp_path, monkeypatch):
    # mkdir raises OSError -> logging is best-effort, must not propagate.
    monkeypatch.setattr(Path, "mkdir",
                        lambda self, *a, **k: (_ for _ in ()).throw(OSError("ro fs")))
    HA.append_hoist_log(
        tmp_path, round_num=1, commit="c", pipeline_rel="p", violations=[],
        outcome="blocked",
    )  # no exception
    assert not (tmp_path / ".zyme" / "hoist_log.jsonl").exists()


def test_append_hoist_log_appends_multiple(tmp_path):
    for i in range(3):
        HA.append_hoist_log(
            tmp_path, round_num=i, commit=None, pipeline_rel="pipeline/run.py",
            violations=[], outcome="bypassed", bypass_reason="r%d" % i,
        )
    lines = (tmp_path / ".zyme" / "hoist_log.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[2])["bypass_reason"] == "r2"


# ---------------------------------------------------------------------------
# format_violations_message
# ---------------------------------------------------------------------------

def test_format_violations_message_includes_fields(tmp_path):
    p = tmp_path / "run.R"
    violations = [{
        "pattern": "r_getns_call", "line": 2, "code": "precomp <- .orig(x)",
        "resolved": "celda:::.initZ", "hint": "moved outside the timer",
    }]
    msg = HA.format_violations_message(p, violations, hypothesis="[algorithmic] decontX")
    assert "Hoist check failed" in msg
    assert "r_getns_call" in msg
    assert "celda:::.initZ" in msg
    assert "moved outside the timer" in msg
    assert '"[algorithmic] decontX"' in msg
    assert "hoist_exempt" in msg


def test_format_violations_message_default_hypothesis_placeholder(tmp_path):
    p = tmp_path / "run.py"
    msg = HA.format_violations_message(p, [{
        "pattern": "private_attr_call", "line": 1, "code": "x",
        "resolved": "a._b", "hint": "h",
    }])
    assert '"<hypothesis>"' in msg
