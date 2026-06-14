"""Unit tests for zyme.fingerprint — reference.{py,R} inflation-pattern detector.

Pure AST/regex checks. No upstream, no network. Each test builds a tiny
reference source inline, writes it to tmp_path, and asserts on whether the
fingerprint check fires (and on the structured violation dicts from the
internal _check_python / _check_r helpers).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme import fingerprint as fp
from zyme.fingerprint import (
    FingerprintViolation,
    check_reference_fingerprint,
    _check_python,
    _check_r,
    _format_violations,
)


# --------------------------------------------------------------------------
# _check_python — list-multiply pattern
# --------------------------------------------------------------------------
class TestListMul:
    def test_path_call_list_repeat_flagged(self):
        v = _check_python("paths = [Path('x.h5ad')] * 8\n")
        assert len(v) == 1
        assert v[0]["pattern"] == "list_mul"
        assert v[0]["line"] == 1
        assert "distinct paths" in v[0]["hint"]

    def test_n_on_left_also_flagged(self):
        v = _check_python("paths = 8 * [load('x')]\n")
        assert [x["pattern"] for x in v] == ["list_mul"]

    def test_string_constant_list_repeat_flagged(self):
        v = _check_python("xs = ['data/a.h5ad'] * 4\n")
        assert [x["pattern"] for x in v] == ["list_mul"]

    def test_attribute_element_flagged(self):
        v = _check_python("xs = [cfg.path] * 3\n")
        assert [x["pattern"] for x in v] == ["list_mul"]

    def test_subscript_element_flagged(self):
        v = _check_python("xs = [files[0]] * 3\n")
        assert [x["pattern"] for x in v] == ["list_mul"]

    @pytest.mark.parametrize("trivial", ["None", "0", "0.0", "False", "''", "b''"])
    def test_trivial_init_values_not_flagged(self, trivial):
        # [None]*N etc. is a legit preallocation idiom, never flagged.
        assert _check_python(f"buf = [{trivial}] * 100\n") == []

    def test_multi_element_list_not_flagged(self):
        assert _check_python("xs = [a, b] * 3\n") == []

    def test_non_list_mult_not_flagged(self):
        # plain arithmetic
        assert _check_python("z = 3 * width\n") == []

    def test_two_lists_multiplied_not_flagged(self):
        # B7 fix: the multiplier must be scalar-like, not itself a list/tuple
        # literal, so `[a] * [b]` is no longer flagged as a list-repeat.
        v = _check_python("z = [a] * [b]\n")
        assert [x["pattern"] for x in v] == []

    def test_list_times_scalar_still_flagged(self):
        # The genuine "replicate a list N times" pattern is still flagged.
        assert [x["pattern"] for x in _check_python("z = [load()] * 5\n")] == ["list_mul"]
        assert [x["pattern"] for x in _check_python("z = n * [load()]\n")] == ["list_mul"]


# --------------------------------------------------------------------------
# _check_python — numpy tile/repeat pattern
# --------------------------------------------------------------------------
class TestTileCall:
    @pytest.mark.parametrize("base", ["np", "numpy"])
    @pytest.mark.parametrize("fn", ["tile", "repeat"])
    def test_tile_repeat_flagged(self, base, fn):
        v = _check_python(f"arr = {base}.{fn}(x, 5)\n")
        assert [a["pattern"] for a in v] == ["numpy_tile"]
        assert v[0]["line"] == 1

    def test_other_numpy_call_not_flagged(self):
        assert _check_python("arr = np.zeros(5)\n") == []

    def test_tile_on_unknown_base_not_flagged(self):
        # foo.tile is not np/numpy
        assert _check_python("arr = foo.tile(x, 5)\n") == []

    def test_bare_tile_name_not_flagged(self):
        # tile(...) as a plain Name call, not an Attribute, is ignored
        assert _check_python("arr = tile(x, 5)\n") == []


# --------------------------------------------------------------------------
# _check_python — repeat-loop pattern
# --------------------------------------------------------------------------
class TestTargetLoop:
    def test_underscore_loop_with_call_flagged(self):
        src = "for _ in range(10):\n    run(adata)\n"
        v = _check_python(src)
        assert [a["pattern"] for a in v] == ["target_loop"]
        assert v[0]["line"] == 1

    def test_unused_named_var_flagged(self):
        # i is never read in the body -> still an inflation loop
        src = "for i in range(5):\n    process(data)\n"
        v = _check_python(src)
        assert [a["pattern"] for a in v] == ["target_loop"]

    def test_used_loop_var_not_flagged(self):
        # i IS read -> legit per-item batch loop
        src = "for i in range(5):\n    process(data[i])\n"
        assert _check_python(src) == []

    def test_range_1_not_flagged(self):
        src = "for _ in range(1):\n    run(x)\n"
        assert _check_python(src) == []

    def test_range_0_not_flagged(self):
        src = "for _ in range(0):\n    run(x)\n"
        assert _check_python(src) == []

    def test_non_constant_range_flagged(self):
        src = "for _ in range(n):\n    run(x)\n"
        v = _check_python(src)
        assert [a["pattern"] for a in v] == ["target_loop"]

    def test_loop_without_call_not_flagged(self):
        # body has no Call -> not an inflation loop
        src = "for _ in range(5):\n    total += 1\n"
        assert _check_python(src) == []

    def test_non_range_iter_not_flagged(self):
        src = "for _ in items:\n    run(x)\n"
        assert _check_python(src) == []

    def test_tuple_target_not_flagged(self):
        # `for a, b in ...` target is not a Name -> ignored
        src = "for a, b in range(5):\n    run(x)\n"
        assert _check_python(src) == []


# --------------------------------------------------------------------------
# _check_python — robustness
# --------------------------------------------------------------------------
class TestCheckPythonRobust:
    def test_syntax_error_returns_empty(self):
        # Don't block baseline on a broken reference.py.
        assert _check_python("def (:\n") == []

    def test_clean_source_returns_empty(self):
        src = (
            "def reference(paths):\n"
            "    out = []\n"
            "    for p in paths:\n"
            "        out.append(load(p))\n"
            "    return out\n"
        )
        assert _check_python(src) == []

    def test_multiple_violations_collected(self):
        src = (
            "a = [Path('x')] * 4\n"
            "b = np.tile(z, 3)\n"
            "for _ in range(9):\n"
            "    run(y)\n"
        )
        patterns = sorted(a["pattern"] for a in _check_python(src))
        assert patterns == ["list_mul", "numpy_tile", "target_loop"]


# --------------------------------------------------------------------------
# _check_r — regex checks
# --------------------------------------------------------------------------
class TestCheckR:
    def test_rep_bare_arg_flagged(self):
        v = _check_r("x <- rep(obj, 5)\n")
        assert [a["pattern"] for a in v] == ["r_rep"]
        assert v[0]["line"] == 1

    def test_rep_string_arg_flagged(self):
        v = _check_r("x <- rep('a.rds', 7)\n")
        assert [a["pattern"] for a in v] == ["r_rep"]

    def test_rep_nested_call_arg_flagged(self):
        # B6 fix: the _R_REP_PATTERN now allows one level of nested parens in
        # the first arg, so a function-call first argument is detected.
        assert [a["pattern"] for a in _check_r("x <- rep(readRDS('a.rds'), 7)\n")] == ["r_rep"]

    def test_rep_times_kwarg_flagged(self):
        v = _check_r("x <- rep(obj, times = 3)\n")
        assert [a["pattern"] for a in v] == ["r_rep"]

    def test_rep_n_1_not_flagged(self):
        assert _check_r("x <- rep(obj, 1)\n") == []

    def test_replicate_flagged(self):
        v = _check_r("x <- replicate(4, run_model())\n")
        assert [a["pattern"] for a in v] == ["r_replicate"]

    def test_replicate_n_1_not_flagged(self):
        assert _check_r("x <- replicate(1, run_model())\n") == []

    def test_comment_line_ignored(self):
        assert _check_r("# x <- rep(obj, 5)\n") == []

    def test_blank_lines_ignored(self):
        assert _check_r("\n   \n") == []

    def test_line_numbers_are_one_based(self):
        src = "a <- 1\nb <- 2\nx <- rep(obj, 3)\n"
        v = _check_r(src)
        assert v[0]["line"] == 3

    def test_clean_r_source(self):
        assert _check_r("x <- lapply(paths, readRDS)\n") == []


# --------------------------------------------------------------------------
# check_reference_fingerprint — end-to-end on tmp files
# --------------------------------------------------------------------------
class TestCheckReferenceFingerprint:
    def test_missing_file_is_noop(self, tmp_path: Path):
        # No file -> returns None silently.
        assert check_reference_fingerprint(tmp_path / "absent.py") is None

    def test_clean_python_passes(self, tmp_path: Path):
        p = tmp_path / "reference.py"
        p.write_text("xs = [load(a), load(b)]\n")
        assert check_reference_fingerprint(p) is None

    def test_dirty_python_raises(self, tmp_path: Path):
        p = tmp_path / "reference.py"
        p.write_text("xs = [load('a')] * 16\n")
        with pytest.raises(FingerprintViolation) as ei:
            check_reference_fingerprint(p)
        msg = str(ei.value)
        assert "Fingerprint check failed" in msg
        assert "list_mul" in msg

    def test_clean_r_passes(self, tmp_path: Path):
        p = tmp_path / "reference.R"
        p.write_text("x <- lapply(files, readRDS)\n")
        assert check_reference_fingerprint(p) is None

    def test_dirty_r_raises(self, tmp_path: Path):
        p = tmp_path / "reference.R"
        p.write_text("x <- rep(obj, 7)\n")
        with pytest.raises(FingerprintViolation):
            check_reference_fingerprint(p)

    def test_rscript_suffix_handled(self, tmp_path: Path):
        p = tmp_path / "reference.rscript"
        p.write_text("x <- replicate(5, run())\n")
        with pytest.raises(FingerprintViolation):
            check_reference_fingerprint(p)

    def test_unknown_suffix_is_noop(self, tmp_path: Path):
        # .txt isn't .py or .r -> not scanned, never raises even with bad content.
        p = tmp_path / "reference.txt"
        p.write_text("xs = [load('a')] * 16\n")
        assert check_reference_fingerprint(p) is None

    def test_suffix_is_case_insensitive(self, tmp_path: Path):
        p = tmp_path / "reference.PY"
        p.write_text("xs = [load('a')] * 16\n")
        with pytest.raises(FingerprintViolation):
            check_reference_fingerprint(p)


# --------------------------------------------------------------------------
# _format_violations — message contains every violation's details
# --------------------------------------------------------------------------
class TestFormatViolations:
    def test_message_includes_all_fields(self, tmp_path: Path):
        violations = [
            {"pattern": "list_mul", "line": 3, "code": "[x] * 5", "hint": "use distinct"},
            {"pattern": "r_rep", "line": 9, "code": "rep(y, 4)", "hint": "no rep"},
        ]
        msg = _format_violations(tmp_path / "ref.py", violations)
        assert "[list_mul] line 3" in msg
        assert "[x] * 5" in msg
        assert "use distinct" in msg
        assert "[r_rep] line 9" in msg
        assert "--accept-synthesis" in msg
        assert "synthesis:" in msg


# --------------------------------------------------------------------------
# Module-level invariants
# --------------------------------------------------------------------------
def test_violation_is_exception_subclass():
    assert issubclass(FingerprintViolation, Exception)


def test_check_python_does_not_mutate_input():
    src = "for _ in range(3):\n    run(x)\n"
    before = src
    _check_python(src)
    assert src == before
