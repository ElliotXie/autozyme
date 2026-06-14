"""In-process unit tests for zyme.argdiff.

argdiff is pure-Python AST parsing: no upstream, no subprocess. Every helper
is exercised here against small synthetic reference.py / pipeline/run.py
fixtures written into tmp dirs, plus direct calls into the private helpers.

Covers:
  - _target_short_name (spec parsing: '::' / '.' / empty)
  - _looks_like_path_expr / _path_signature / _normalize_for_diff
  - _call_name_chain / _collect_module_assigns / _resolved_unparse / _clone
  - _extract_call_kwargs (kwargs, positional, var substitution, not-found)
  - diff_target_call_kwargs (added / removed / changed / reordered / thread-
    skip / output-skip / path-normalized / missing-file)
  - _find_timer_lineno / _args_match / check_target_prewarm (prewarm + dummy)
  - _scan_side_channel_setters / diff_side_channel_parallelism
  - format_divergences / format_side_channel_divergences / _format_prewarm
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from zyme import argdiff


# --------------------------------------------------------------------------
# Helpers to build task fixtures
# --------------------------------------------------------------------------

def _make_task(tmp_path: Path, reference: str | None, pipeline: str | None) -> Path:
    """Write reference.py and/or pipeline/run.py into a tmp task dir."""
    if reference is not None:
        (tmp_path / "reference.py").write_text(textwrap.dedent(reference))
    if pipeline is not None:
        pdir = tmp_path / "pipeline"
        pdir.mkdir(exist_ok=True)
        (pdir / "run.py").write_text(textwrap.dedent(pipeline))
    return tmp_path


# --------------------------------------------------------------------------
# _target_short_name
# --------------------------------------------------------------------------

class TestTargetShortName:
    def test_double_colon_form(self):
        assert argdiff._target_short_name(
            "cellphonedb.src.core.methods.mod::call") == "call"

    def test_double_colon_keeps_dotted_tail(self):
        assert argdiff._target_short_name("spacexr::run.RCTD") == "run.RCTD"
        assert argdiff._target_short_name(
            "MDAnalysis.analysis.rms::RMSD.run") == "RMSD.run"

    def test_dotted_only(self):
        assert argdiff._target_short_name("np.linalg.svd") == "svd"

    def test_bare_name(self):
        assert argdiff._target_short_name("foo") == "foo"

    def test_empty(self):
        assert argdiff._target_short_name("") == ""

    def test_whitespace_stripped(self):
        assert argdiff._target_short_name("  pkg::fn  ") == "fn"


# --------------------------------------------------------------------------
# path heuristics
# --------------------------------------------------------------------------

class TestPathHeuristics:
    def test_looks_like_path_expr_true(self):
        assert argdiff._looks_like_path_expr("Path('a/b.nc')")
        assert argdiff._looks_like_path_expr("os.path.join(d, 'x')")
        assert argdiff._looks_like_path_expr("__file__")
        assert argdiff._looks_like_path_expr("p.parent")

    def test_looks_like_path_expr_false(self):
        assert not argdiff._looks_like_path_expr("42")
        assert not argdiff._looks_like_path_expr("'a string'")
        assert not argdiff._looks_like_path_expr("some_var")

    def test_path_signature_extension(self):
        assert argdiff._path_signature("Path('data/tas_tiny.nc')") == ".nc"
        assert argdiff._path_signature("f'tas_{TIER}.pickle'") == ".pickle"

    def test_path_signature_last_string_wins(self):
        # multiple quoted strings -> last with an extension
        assert argdiff._path_signature(
            "os.path.join('dir', 'sub', 'file.tsv')") == ".tsv"

    def test_path_signature_no_extension_falls_back_to_last_string(self):
        assert argdiff._path_signature("Path('justadir')") == "justadir"

    def test_path_signature_no_strings_verbatim(self):
        assert argdiff._path_signature("p.parent") == "p.parent"

    def test_normalize_none(self):
        assert argdiff._normalize_for_diff("k", None) is None

    def test_normalize_path_reduces(self):
        assert argdiff._normalize_for_diff(
            "counts", "Path('data/x.tsv')") == ".tsv"

    def test_normalize_nonpath_verbatim(self):
        assert argdiff._normalize_for_diff("batch_size", "32") == "32"

    def test_normalize_collapses_cosmetic_path_diff(self):
        # Both wrapped in Path() -> both reduce to .nc. The Path() wrapper is
        # what makes _looks_like_path_expr fire; a bare f-string would NOT
        # (see test_normalize_bare_fstring_is_verbatim below).
        a = argdiff._normalize_for_diff("f", "Path(f'tas_{TIER}.nc')")
        b = argdiff._normalize_for_diff("f", "Path('data/tas_tiny.nc')")
        assert a == b == ".nc"

    def test_normalize_bare_fstring_is_verbatim(self):
        # No Path()/os.path/__file__ token -> not treated as a path, even
        # though _path_signature alone could reduce it.
        out = argdiff._normalize_for_diff("f", "f'tas_{TIER}.nc'")
        assert out == "f'tas_{TIER}.nc'"
        # _path_signature in isolation DOES reduce it.
        assert argdiff._path_signature("f'tas_{TIER}.nc'") == ".nc"

    def test_normalize_distinguishes_real_format_swap(self):
        a = argdiff._normalize_for_diff("f", "Path('x.tsv')")
        b = argdiff._normalize_for_diff("f", "Path('x.pickle')")
        assert a != b


# --------------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------------

class TestCallNameChain:
    def _chain(self, src: str):
        node = ast.parse(src, mode="eval").body
        assert isinstance(node, ast.Call)
        return argdiff._call_name_chain(node.func)

    def test_simple_name(self):
        assert self._chain("foo()") == ["foo"]

    def test_attribute_chain(self):
        assert self._chain("a.b.c()") == ["a", "b", "c"]

    def test_call_result_attribute_keeps_only_attr_tail(self):
        # f().g() -> base of the .g attribute is a Call (returns []), so the
        # chain is just the trailing attribute name. (Not [] — base [] is
        # falsy but `base is None` is False, so the attr is appended.)
        assert self._chain("f().g()") == ["g"]

    def test_subscript_base_not_a_name(self):
        # d['k']() -> func is a Name? no, it's a Subscript -> []
        assert self._chain("d['k']()") == []


class TestCollectModuleAssigns:
    def test_simple_and_last_wins(self):
        tree = ast.parse("x = 1\nx = 2\ny = 'a'\n")
        assigns = argdiff._collect_module_assigns(tree)
        assert set(assigns) == {"x", "y"}
        assert ast.literal_eval(ast.unparse(assigns["x"])) == 2

    def test_walks_if_and_try_bodies(self):
        tree = ast.parse(
            "if FLAG:\n    a = 1\nelse:\n    b = 2\n"
            "try:\n    c = 3\nexcept Exception:\n    d = 4\n"
        )
        assigns = argdiff._collect_module_assigns(tree)
        assert {"a", "b", "c", "d"} <= set(assigns)

    def test_walks_with_body(self):
        tree = ast.parse("with ctx() as c:\n    w = 1\n")
        assigns = argdiff._collect_module_assigns(tree)
        assert "w" in assigns

    def test_walks_try_finally_and_orelse(self):
        tree = ast.parse(
            "try:\n    a = 1\nexcept Exception:\n    b = 2\n"
            "else:\n    e = 3\nfinally:\n    f = 4\n"
        )
        assigns = argdiff._collect_module_assigns(tree)
        assert {"a", "b", "e", "f"} <= set(assigns)


class TestResolvedUnparse:
    def test_substitutes_name(self):
        tree = ast.parse("TIER = 'tiny'\nx = TIER\n")
        assigns = argdiff._collect_module_assigns(tree)
        node = assigns["x"]  # Name TIER
        assert argdiff._resolved_unparse(node, assigns) == "'tiny'"

    def test_cycle_safe(self):
        # a = b ; b = a -> resolver must not loop forever
        tree = ast.parse("a = b\nb = a\n")
        assigns = argdiff._collect_module_assigns(tree)
        # Should return *something* without hanging.
        out = argdiff._resolved_unparse(assigns["a"], assigns)
        assert isinstance(out, str)

    def test_unknown_name_unchanged(self):
        out = argdiff._resolved_unparse(ast.Name(id="zzz", ctx=ast.Load()), {})
        assert out == "zzz"

    def test_clone_roundtrips(self):
        node = ast.parse("a + b", mode="eval").body
        cloned = argdiff._clone(node)
        assert ast.unparse(cloned) == "a + b"


# --------------------------------------------------------------------------
# _extract_call_kwargs
# --------------------------------------------------------------------------

class TestExtractCallKwargs:
    def test_kwargs_and_positional(self, tmp_path):
        f = tmp_path / "r.py"
        f.write_text("result = my_fn(data, batch_size=32, mode='fast')\n")
        out = argdiff._extract_call_kwargs(f, "my_fn")
        assert out == {"<pos0>": "data", "batch_size": "32", "mode": "'fast'"}

    def test_var_substitution(self, tmp_path):
        f = tmp_path / "r.py"
        f.write_text("BS = 64\nresult = my_fn(batch_size=BS)\n")
        out = argdiff._extract_call_kwargs(f, "my_fn")
        assert out == {"batch_size": "64"}

    def test_splat_kwargs_skipped(self, tmp_path):
        f = tmp_path / "r.py"
        f.write_text("result = my_fn(a=1, **extra)\n")
        out = argdiff._extract_call_kwargs(f, "my_fn")
        assert out == {"a": "1"}

    def test_dotted_target_via_attribute_object(self, tmp_path):
        # A real `a.b.run(...)` attribute chain matches a dotted target.
        f = tmp_path / "r.py"
        f.write_text("mod.RMSD.run(verbose=True)\n")
        out = argdiff._extract_call_kwargs(f, "RMSD.run")
        assert out == {"verbose": "True"}

    def test_dotted_target_via_call_result_does_not_match(self, tmp_path):
        # RMSD(u).run(...) -> chain is ['run'] only (base is a Call), so the
        # two-segment target 'RMSD.run' does NOT match. Documents the limit.
        f = tmp_path / "r.py"
        f.write_text("obj = RMSD(u).run(verbose=True)\n")
        assert argdiff._extract_call_kwargs(f, "RMSD.run") is None

    def test_not_found_returns_none(self, tmp_path):
        f = tmp_path / "r.py"
        f.write_text("other()\n")
        assert argdiff._extract_call_kwargs(f, "my_fn") is None

    def test_syntax_error_returns_none(self, tmp_path):
        f = tmp_path / "r.py"
        f.write_text("def (:\n")
        assert argdiff._extract_call_kwargs(f, "my_fn") is None

    def test_first_match_wins(self, tmp_path):
        f = tmp_path / "r.py"
        f.write_text("my_fn(a=1)\nmy_fn(a=2)\n")
        out = argdiff._extract_call_kwargs(f, "my_fn")
        assert out == {"a": "1"}


# --------------------------------------------------------------------------
# diff_target_call_kwargs
# --------------------------------------------------------------------------

class TestDiffTargetCallKwargs:
    def test_identical_no_divergence(self, tmp_path):
        _make_task(
            tmp_path,
            "x = my_fn(batch_size=32, mode='fast')\n",
            "x = my_fn(batch_size=32, mode='fast')\n",
        )
        assert argdiff.diff_target_call_kwargs(tmp_path, "lib::my_fn") == []

    def test_reordered_no_divergence(self, tmp_path):
        # order doesn't matter; keys are compared as a set
        _make_task(
            tmp_path,
            "x = my_fn(batch_size=32, mode='fast')\n",
            "x = my_fn(mode='fast', batch_size=32)\n",
        )
        assert argdiff.diff_target_call_kwargs(tmp_path, "lib::my_fn") == []

    def test_changed_value(self, tmp_path):
        _make_task(
            tmp_path,
            "x = my_fn(batch_size=32)\n",
            "x = my_fn(batch_size=64)\n",
        )
        out = argdiff.diff_target_call_kwargs(tmp_path, "lib::my_fn")
        assert out == [{"kwarg": "batch_size", "ref": "32", "pipe": "64"}]

    def test_added_kwarg_in_pipeline(self, tmp_path):
        _make_task(
            tmp_path,
            "x = my_fn(a=1)\n",
            "x = my_fn(a=1, b=2)\n",
        )
        out = argdiff.diff_target_call_kwargs(tmp_path, "lib::my_fn")
        assert out == [{"kwarg": "b", "ref": None, "pipe": "2"}]

    def test_removed_kwarg_in_pipeline(self, tmp_path):
        _make_task(
            tmp_path,
            "x = my_fn(a=1, b=2)\n",
            "x = my_fn(a=1)\n",
        )
        out = argdiff.diff_target_call_kwargs(tmp_path, "lib::my_fn")
        assert out == [{"kwarg": "b", "ref": "2", "pipe": None}]

    def test_thread_kwarg_skipped(self, tmp_path):
        _make_task(
            tmp_path,
            "x = my_fn(n_jobs=1)\n",
            "x = my_fn(n_jobs=8)\n",
        )
        assert argdiff.diff_target_call_kwargs(tmp_path, "lib::my_fn") == []

    def test_output_kwarg_skipped(self, tmp_path):
        _make_task(
            tmp_path,
            "x = my_fn(output_path='a.tsv')\n",
            "x = my_fn(output_path='b.tsv')\n",
        )
        assert argdiff.diff_target_call_kwargs(tmp_path, "lib::my_fn") == []

    def test_upstream_parallelism_extra_skip(self, tmp_path):
        _make_task(
            tmp_path,
            "x = my_fn(chunks=1)\n",
            "x = my_fn(chunks=4)\n",
        )
        # without the extra skip it'd diverge
        assert argdiff.diff_target_call_kwargs(tmp_path, "lib::my_fn") != []
        assert argdiff.diff_target_call_kwargs(
            tmp_path, "lib::my_fn", upstream_parallelism=["chunks"]) == []

    def test_path_format_swap_detected(self, tmp_path):
        _make_task(
            tmp_path,
            "x = my_fn(counts_file_path='data/x.tsv')\n",
            "x = my_fn(counts_file_path='data/x.pickle')\n",
        )
        out = argdiff.diff_target_call_kwargs(tmp_path, "lib::my_fn")
        assert len(out) == 1
        assert out[0]["kwarg"] == "counts_file_path"

    def test_path_cosmetic_diff_ignored(self, tmp_path):
        _make_task(
            tmp_path,
            "x = my_fn(counts=Path('data/tas_tiny.nc'))\n",
            "TIER='tiny'\nx = my_fn(counts=Path(f'tas_{TIER}.nc'))\n",
        )
        assert argdiff.diff_target_call_kwargs(tmp_path, "lib::my_fn") == []

    def test_missing_reference_file(self, tmp_path):
        _make_task(tmp_path, None, "x = my_fn(a=1)\n")
        assert argdiff.diff_target_call_kwargs(tmp_path, "lib::my_fn") == []

    def test_target_not_in_one_file(self, tmp_path):
        _make_task(tmp_path, "x = my_fn(a=1)\n", "y = other()\n")
        assert argdiff.diff_target_call_kwargs(tmp_path, "lib::my_fn") == []

    def test_empty_target_function(self, tmp_path):
        _make_task(tmp_path, "x = my_fn(a=1)\n", "x = my_fn(a=2)\n")
        assert argdiff.diff_target_call_kwargs(tmp_path, "") == []

    def test_multiple_divergences_sorted(self, tmp_path):
        _make_task(
            tmp_path,
            "x = my_fn(zebra=1, alpha=1)\n",
            "x = my_fn(zebra=2, alpha=2)\n",
        )
        out = argdiff.diff_target_call_kwargs(tmp_path, "lib::my_fn")
        assert [d["kwarg"] for d in out] == ["alpha", "zebra"]


# --------------------------------------------------------------------------
# _find_timer_lineno
# --------------------------------------------------------------------------

class TestFindTimerLineno:
    def test_perf_counter_bare(self):
        tree = ast.parse("import time\nt0 = perf_counter()\n")
        assert argdiff._find_timer_lineno(tree) == 2

    def test_time_dot_perf_counter(self):
        tree = ast.parse("import time\nt0 = time.perf_counter()\n")
        assert argdiff._find_timer_lineno(tree) == 2

    def test_time_dot_monotonic(self):
        tree = ast.parse("import time\nt0 = time.monotonic()\n")
        assert argdiff._find_timer_lineno(tree) == 2

    def test_bare_monotonic_not_matched(self):
        # 'monotonic' without time. qualification must not match
        tree = ast.parse("t0 = monotonic()\n")
        assert argdiff._find_timer_lineno(tree) is None

    def test_no_timer(self):
        tree = ast.parse("x = 1\n")
        assert argdiff._find_timer_lineno(tree) is None


# --------------------------------------------------------------------------
# _args_match
# --------------------------------------------------------------------------

class TestArgsMatch:
    def test_empty_never_matches(self):
        assert argdiff._args_match({}, {}) is False
        assert argdiff._args_match({"a": "1"}, {}) is False

    def test_same_keys_and_values(self):
        assert argdiff._args_match({"a": "1"}, {"a": "1"}) is True

    def test_different_keys(self):
        assert argdiff._args_match({"a": "1"}, {"b": "1"}) is False

    def test_value_mismatch(self):
        assert argdiff._args_match({"a": "1"}, {"a": "2"}) is False

    def test_path_normalized_match(self):
        assert argdiff._args_match(
            {"f": "Path('x.nc')"}, {"f": "Path('data/x.nc')"}) is True


# --------------------------------------------------------------------------
# check_target_prewarm
# --------------------------------------------------------------------------

class TestCheckTargetPrewarm:
    def test_prewarm_detected(self, tmp_path):
        pipe = """\
        import time
        data = load('big.h5ad')
        my_fn(data, mode='fast')      # pre-warm before timer
        t0 = time.perf_counter()
        my_fn(data, mode='fast')      # timed
        """
        _make_task(tmp_path, None, pipe)
        msg = argdiff.check_target_prewarm(tmp_path, "lib::my_fn")
        assert msg is not None
        assert "Pre-warm pattern detected" in msg
        assert "BEFORE timer" in msg

    def test_synthetic_dummy_not_flagged(self, tmp_path):
        pipe = """\
        import time
        my_fn(dummy, mode='warm')     # synthetic warmup, different args
        t0 = time.perf_counter()
        my_fn(real_data, mode='fast')
        """
        _make_task(tmp_path, None, pipe)
        assert argdiff.check_target_prewarm(tmp_path, "lib::my_fn") is None

    def test_no_timer_returns_none(self, tmp_path):
        _make_task(tmp_path, None, "my_fn(data)\nmy_fn(data)\n")
        assert argdiff.check_target_prewarm(tmp_path, "lib::my_fn") is None

    def test_call_inside_def_ignored(self, tmp_path):
        pipe = """\
        import time
        def helper():
            my_fn(data, mode='fast')   # inside def, not module-level
        t0 = time.perf_counter()
        my_fn(data, mode='fast')
        """
        _make_task(tmp_path, None, pipe)
        # only one module-level call (inside timer) -> no outside call -> None
        assert argdiff.check_target_prewarm(tmp_path, "lib::my_fn") is None

    def test_no_pipeline_returns_none(self, tmp_path):
        assert argdiff.check_target_prewarm(tmp_path, "lib::my_fn") is None

    def test_empty_target_returns_none(self, tmp_path):
        _make_task(tmp_path, None, "import time\nt0=time.perf_counter()\n")
        assert argdiff.check_target_prewarm(tmp_path, "") is None

    def test_empty_args_degenerate_not_flagged(self, tmp_path):
        # bare my_fn() before and inside -> _args_match rejects empty args
        pipe = """\
        import time
        my_fn()
        t0 = time.perf_counter()
        my_fn()
        """
        _make_task(tmp_path, None, pipe)
        assert argdiff.check_target_prewarm(tmp_path, "lib::my_fn") is None

    def test_install_override_alias_matched(self, tmp_path):
        pipe = """\
        import time
        install_override("my_fn", original, fast_my_fn)
        fast_my_fn(data, mode='fast')   # alias call before timer
        t0 = time.perf_counter()
        my_fn(data, mode='fast')        # real name inside timer
        """
        _make_task(tmp_path, None, pipe)
        msg = argdiff.check_target_prewarm(tmp_path, "lib::my_fn")
        assert msg is not None

    def test_prewarm_inside_if_block(self, tmp_path):
        pipe = """\
        import time
        if True:
            my_fn(data, mode='fast')
        t0 = time.perf_counter()
        my_fn(data, mode='fast')
        """
        _make_task(tmp_path, None, pipe)
        assert argdiff.check_target_prewarm(tmp_path, "lib::my_fn") is not None

    def test_prewarm_inside_for_and_with_blocks(self, tmp_path):
        # Exercises the For + With descent branches in _scan_for_calls.
        pipe = """\
        import time
        for _ in range(1):
            my_fn(data, mode='fast')
        t0 = time.perf_counter()
        with ctx():
            my_fn(data, mode='fast')
        """
        _make_task(tmp_path, None, pipe)
        assert argdiff.check_target_prewarm(tmp_path, "lib::my_fn") is not None

    def test_prewarm_inside_try_block(self, tmp_path):
        # Exercises the Try descent branch.
        pipe = """\
        import time
        try:
            my_fn(data, mode='fast')
        except Exception:
            pass
        t0 = time.perf_counter()
        my_fn(data, mode='fast')
        """
        _make_task(tmp_path, None, pipe)
        assert argdiff.check_target_prewarm(tmp_path, "lib::my_fn") is not None

    def test_prewarm_install_override_attribute_alias(self, tmp_path):
        # third arg of install_override is an Attribute chain, not a Name.
        pipe = """\
        import time
        install_override("my_fn", original, mod.fast_my_fn)
        mod.fast_my_fn(data, mode='fast')
        t0 = time.perf_counter()
        my_fn(data, mode='fast')
        """
        _make_task(tmp_path, None, pipe)
        assert argdiff.check_target_prewarm(tmp_path, "lib::my_fn") is not None

    def test_prewarm_syntax_error_returns_none(self, tmp_path):
        _make_task(tmp_path, None, "def (:\n")
        assert argdiff.check_target_prewarm(tmp_path, "lib::my_fn") is None


# --------------------------------------------------------------------------
# side-channel parallelism
# --------------------------------------------------------------------------

class TestSideChannel:
    def test_scan_call_setter(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("import numba\nnumba.set_num_threads(8)\n")
        found = argdiff._scan_side_channel_setters(f)
        assert len(found) == 1
        assert found[0]["kind"] == "call"
        assert found[0]["symbol"] == "numba.set_num_threads"
        assert found[0]["value"] == "8"

    def test_scan_env_assign(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("import os\nos.environ['OMP_NUM_THREADS'] = '4'\n")
        found = argdiff._scan_side_channel_setters(f)
        assert len(found) == 1
        assert found[0]["kind"] == "env"
        assert found[0]["symbol"] == "OMP_NUM_THREADS"
        assert found[0]["value"] == "'4'"

    def test_scan_env_setdefault(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("import os\nos.environ.setdefault('MKL_NUM_THREADS', '2')\n")
        found = argdiff._scan_side_channel_setters(f)
        assert len(found) == 1
        assert found[0]["symbol"] == "MKL_NUM_THREADS"

    def test_scan_irrelevant_env_ignored(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("import os\nos.environ['FOO'] = 'bar'\n")
        assert argdiff._scan_side_channel_setters(f) == []

    def test_scan_syntax_error_empty(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("def (:\n")
        assert argdiff._scan_side_channel_setters(f) == []

    def test_diff_pipeline_only_setter(self, tmp_path):
        _make_task(
            tmp_path,
            "x = my_fn(data)\n",
            "import numba\nnumba.set_num_threads(8)\nx = my_fn(data)\n",
        )
        out = argdiff.diff_side_channel_parallelism(tmp_path)
        assert len(out) == 1
        assert out[0]["symbol"] == "numba.set_num_threads"

    def test_diff_mirrored_setter_not_flagged(self, tmp_path):
        ref = "import numba\nnumba.set_num_threads(8)\nx = my_fn(data)\n"
        _make_task(tmp_path, ref, ref)
        assert argdiff.diff_side_channel_parallelism(tmp_path) == []

    def test_diff_missing_file(self, tmp_path):
        _make_task(tmp_path, None, "import numba\nnumba.set_num_threads(8)\n")
        assert argdiff.diff_side_channel_parallelism(tmp_path) == []


# --------------------------------------------------------------------------
# formatters
# --------------------------------------------------------------------------

class TestFormatters:
    def test_format_divergences(self):
        divs = [{"kwarg": "batch_size", "ref": "32", "pipe": "64"}]
        out = argdiff.format_divergences(divs, "lib::my_fn")
        assert "API-level kwarg divergence" in out
        assert "batch_size" in out
        assert "reference     = 32" in out
        assert "pipeline      = 64" in out

    def test_format_side_channel_call_and_env(self):
        divs = [
            {"kind": "call", "symbol": "numba.set_num_threads",
             "value": "8", "lineno": 3},
            {"kind": "env", "symbol": "OMP_NUM_THREADS",
             "value": "'4'", "lineno": 5},
        ]
        out = argdiff.format_side_channel_divergences(divs)
        assert "Side-channel parallelism" in out
        assert "line 3: numba.set_num_threads(8)" in out
        assert "line 5: os.environ['OMP_NUM_THREADS'] = '4'" in out

    def test_format_prewarm_truncates_long_value(self, tmp_path):
        long_val = "x" * 80
        out = argdiff._format_prewarm(
            tmp_path / "pipeline" / "run.py", 3, 5,
            {"data": long_val}, "lib::my_fn",
        )
        assert "Pre-warm pattern detected" in out
        # value > 50 chars gets the ellipsis truncation
        assert "…" in out
