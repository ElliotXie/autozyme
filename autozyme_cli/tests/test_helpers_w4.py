"""Wave-4 coverage tests for zyme.helpers.

Targets REACHABLE lines left uncovered by test_helpers_unit.py:
  - _resolve_module class-attribute walk that hits AttributeError then succeeds
  - patch_namespace swap-hint branches (module=None+swapped; attr-missing+reverse)
  - patch_namespace strict-aliases >5 truncation + getattr-raising alias skip
  - patch_namespace lenient auto-patch failure paths (alias_mod gone / setattr fails)
  - _parse_datasets_minimal coerce int/float fallbacks, quoted-value strip,
    datasets-list malformed-entry WARN
  - auto_structure / _shape_of / _non_degenerate edge dtypes:
    shape unknown -> None, asarray failure -> None, ref-degenerate TypeError
    fallback, ref non-degenerate but test empty, categorical degenerate,
    unrecognized dtype -> None

Genuinely-unreachable (documented, not tested):
  - peak_memory_mb Windows ctypes path (lines 570-610): this is a macOS host;
    platform.system() never == "Windows".
"""
from __future__ import annotations

import sys
import types

import pytest

from zyme import helpers as H

np = pytest.importorskip("numpy")


# --------------------------------------------------------------------------
# find_framework_root default-start + filesystem-root exhaustion
# --------------------------------------------------------------------------
class TestFindFrameworkRootEdges:
    def test_default_start_uses_this_files_dir(self):
        # start=None exercises the default-start branch (helpers.py's own dir).
        # In the release tree there is no autozyme-framework above the package,
        # so it raises — but the default-start line is covered either way.
        with pytest.raises(RuntimeError):
            H.find_framework_root(start=None)

    def test_default_start_finds_when_framework_present(self, tmp_path,
                                                        monkeypatch):
        # Plant a framework above a fake helpers.py location and point the
        # module __file__ there so the start=None default resolves to it.
        fw = tmp_path / "autozyme-framework"
        (fw / "autozyme_cli").mkdir(parents=True)
        fake = tmp_path / "autozyme-framework" / "autozyme_cli" / "zyme" / "helpers.py"
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text("")
        monkeypatch.setattr(H, "__file__", str(fake))
        found = H.find_framework_root()  # start=None
        assert found.endswith("autozyme-framework")

    def test_walks_to_filesystem_root_then_raises(self):
        # Starting at "/" with the framework absent above it exhausts the walk
        # via the parent==cur break, then raises RuntimeError.
        with pytest.raises(RuntimeError, match="not found"):
            H.find_framework_root(start="/", max_depth=50)


# --------------------------------------------------------------------------
# _resolve_module — class attribute walk with an intermediate AttributeError
# --------------------------------------------------------------------------
class TestResolveModuleWalk:
    def teardown_method(self):
        sys.modules.pop("zyme_w4_walkmod", None)

    def test_attr_walk_skips_bad_split_then_succeeds(self):
        # Build a module with a nested attribute chain `outer.inner`. A dotted
        # path "zyme_w4_walkmod.outer.inner" can't import as a module, so the
        # attribute walk kicks in; the first split (head=...outer.inner... no)
        # may AttributeError, a later split resolves.
        mod = types.ModuleType("zyme_w4_walkmod")

        class Outer:
            pass

        outer = Outer()
        outer.inner = "INNER_VALUE"
        mod.outer = outer
        sys.modules["zyme_w4_walkmod"] = mod
        got = H._resolve_module("zyme_w4_walkmod.outer.inner")
        assert got == "INNER_VALUE"

    def test_attr_walk_attributeerror_returns_none(self):
        mod = types.ModuleType("zyme_w4_walkmod")
        mod.outer = object()  # has no `.missing`
        sys.modules["zyme_w4_walkmod"] = mod
        assert H._resolve_module("zyme_w4_walkmod.outer.missing") is None


# --------------------------------------------------------------------------
# patch_namespace swap-hint + alias edge branches
# --------------------------------------------------------------------------
@pytest.fixture
def victim():
    name = "zyme_w4_victim"
    mod = types.ModuleType(name)

    def original(x):
        return ("orig", x)

    mod.target = original
    sys.modules[name] = mod
    try:
        yield name, mod, original
    finally:
        for n in (name, "zyme_w4_alias", *[f"zyme_w4_alias_{i}" for i in range(8)]):
            sys.modules.pop(n, None)


class TestPatchNamespaceSwapHints:
    def test_module_none_swapped_hint(self):
        # Caller swapped args: module_path is actually the func NAME and
        # func_name is an importable module that HAS an attribute named after
        # module_path's last segment. Triggers the swap-hint RuntimeError text.
        # os.getcwd exists; calling patch_namespace("getcwd", "os", fn) is the
        # correct order, so to trip the hint we pass them swapped: the
        # "module_path" ('getcwd') is unresolvable, but _resolve_module(func_name
        # = 'os') succeeds and has attribute 'getcwd'.
        with pytest.raises(RuntimeError, match="swap the arguments"):
            H.patch_namespace("os", "getcwd", lambda: None)

    def test_attr_missing_reverse_hint(self, victim):
        # module resolves, but func_name absent AND the module HAS an attribute
        # named like module_path's last segment -> attr-missing swap hint.
        name, mod, _ = victim
        mod.zyme_w4_victim = "decoy"  # last segment of module_path
        with pytest.raises(RuntimeError, match="swap the arguments"):
            H.patch_namespace("no_such_attr", name, lambda: None)


class TestPatchNamespaceAliases:
    def test_strict_more_than_five_aliases_truncated(self, victim):
        name, mod, original = victim
        # Bind the same original under 7 alias modules -> strict raises with
        # "... and N more" truncation text.
        for i in range(7):
            am = types.ModuleType(f"zyme_w4_alias_{i}")
            am.target = original
            sys.modules[f"zyme_w4_alias_{i}"] = am
        with pytest.raises(RuntimeError, match="and 2 more"):
            H.patch_namespace("target", name, lambda x: x, strict_aliases=True)

    def test_getattr_raising_alias_module_skipped(self, victim):
        name, mod, original = victim

        class Raising(types.ModuleType):
            def __getattribute__(self, n):
                if n == "target":
                    raise RuntimeError("boom")
                return super().__getattribute__(n)

        bad = Raising("zyme_w4_alias")
        sys.modules["zyme_w4_alias"] = bad

        def fast(x):
            return x

        # The raising alias is skipped during the scan; patch still succeeds.
        H.patch_namespace("target", name, fast)
        assert mod.target is fast

    def test_lenient_alias_autopatch_success_prints(self, victim, capsys):
        # A reachable, writable alias is auto-patched -> the success print fires.
        name, mod, original = victim
        alias = types.ModuleType("zyme_w4_alias")
        alias.target = original
        sys.modules["zyme_w4_alias"] = alias

        def fast(x):
            return x

        H.patch_namespace("target", name, fast)
        out = capsys.readouterr().out
        assert "auto-patched" in out
        assert alias.target is fast

    def test_lenient_alias_setattr_fails_recorded(self, victim, capsys):
        name, mod, original = victim

        class FrozenAttr(types.ModuleType):
            def __setattr__(self, n, v):
                if n == "target":
                    raise AttributeError("read-only")
                super().__setattr__(n, v)

        frozen = FrozenAttr("zyme_w4_alias")
        # set the alias the original is bound to (bypass our __setattr__ guard)
        object.__setattr__(frozen, "target", original)
        sys.modules["zyme_w4_alias"] = frozen

        def fast(x):
            return x

        H.patch_namespace("target", name, fast)
        err = capsys.readouterr().out
        assert "could not auto-patch" in err
        assert mod.target is fast


# --------------------------------------------------------------------------
# inline_upstream non-regular-function target
# --------------------------------------------------------------------------
class TestInlineUpstreamNonCode:
    def teardown_method(self):
        sys.modules.pop("zyme_w4_up", None)

    def test_callable_without_code_raises(self):
        # `len` is callable but is a builtin without __code__/__globals__ ->
        # the "not a regular Python function" RuntimeError.
        mod = types.ModuleType("zyme_w4_up")
        mod.fn = len  # builtin_function_or_method
        sys.modules["zyme_w4_up"] = mod
        with pytest.raises(RuntimeError, match="not a regular Python function"):
            H.inline_upstream("zyme_w4_up.fn")


# --------------------------------------------------------------------------
# peak_memory_mb resource.getrusage failure path (non-Windows)
# --------------------------------------------------------------------------
class TestPeakMemoryFailure:
    def test_getrusage_exception_returns_zero(self, monkeypatch, capsys):
        import resource
        monkeypatch.setattr(H, "_zyme_peak_mb_warned", False)

        def boom(*a, **k):
            raise OSError("no rusage")

        monkeypatch.setattr(resource, "getrusage", boom)
        assert H.peak_memory_mb() == 0.0
        assert "[peak_mb] warning" in capsys.readouterr().err


# --------------------------------------------------------------------------
# _parse_datasets_minimal coerce + quoted-value + malformed-list WARN
# --------------------------------------------------------------------------
class TestParseDatasetsCoerce:
    def test_param_int_float_and_string_coercion(self, tmp_path):
        p = tmp_path / "task.yaml"
        p.write_text(
            "datasets:\n"
            "  - {tier: tiny, name: a, path: p, "
            "params: {i: 5, f: 1.5, s: hello, q: \"quoted\"}}\n"
        )
        e = H._parse_datasets_minimal(str(p))[0]
        assert e["params"] == {"i": 5, "f": 1.5, "s": "hello", "q": "quoted"}

    def test_quoted_top_level_value_stripped(self, tmp_path):
        p = tmp_path / "task.yaml"
        p.write_text(
            "datasets:\n"
            '  - {tier: tiny, name: "a", path: \'p\'}\n'
        )
        e = H._parse_datasets_minimal(str(p))[0]
        assert e["name"] == "a"
        assert e["path"] == "p"

    def test_malformed_dataset_entry_warned(self, tmp_path, capsys):
        # A `key value` (no colon) directly in the datasets list entry -> WARN.
        p = tmp_path / "task.yaml"
        p.write_text(
            "datasets:\n"
            "  - {tier: tiny, name: a, path: p, bare_no_colon}\n"
        )
        e = H._parse_datasets_minimal(str(p))[0]
        assert e["name"] == "a"
        assert "WARN" in capsys.readouterr().err

    def test_blank_kv_entries_skipped(self, tmp_path):
        # Trailing comma -> an empty split token is skipped without error.
        p = tmp_path / "task.yaml"
        p.write_text(
            "datasets:\n"
            "  - {tier: tiny, name: a, path: p,}\n"
        )
        e = H._parse_datasets_minimal(str(p))[0]
        assert e["name"] == "a"

    def test_blank_kv_inside_params_dict_skipped(self, tmp_path):
        # A trailing comma inside the nested params dict yields an empty kv
        # token that parse_val skips (the `if not kv: continue` branch).
        p = tmp_path / "task.yaml"
        p.write_text(
            "datasets:\n"
            "  - {tier: tiny, name: a, path: p, params: {n: 5,}}\n"
        )
        e = H._parse_datasets_minimal(str(p))[0]
        assert e["params"] == {"n": 5}

    def test_non_list_indented_line_in_datasets_skipped(self, tmp_path):
        # An indented line that isn't a `- ` list item is skipped.
        p = tmp_path / "task.yaml"
        p.write_text(
            "datasets:\n"
            "  not_a_list_item: 1\n"
            "  - {tier: tiny, name: a, path: p}\n"
        )
        entries = H._parse_datasets_minimal(str(p))
        assert [e["name"] for e in entries] == ["a"]


# --------------------------------------------------------------------------
# auto_structure / _shape_of / _non_degenerate edge dtypes
# --------------------------------------------------------------------------
class TestShapeAndDegenerateEdges:
    def test_shape_match_none_when_shape_unknown(self):
        # A slot whose value has no determinable shape: np.asarray raises AND
        # len() raises -> _shape_of returns None -> shape_match recorded as None.
        class NoShape:
            def __array__(self, *a, **k):
                raise ValueError("no array")
            # no __len__ either

        ref = {"s": NoShape()}
        test = {"s": NoShape()}
        out = H.auto_structure_check_all_slots(ref, test)
        assert out["s_present"] == 1.0
        assert out["s_shape_match"] is None

    def test_shape_of_asarray_failure_falls_back(self):
        # np.asarray of a ragged nested list raises -> falls back to len().
        ragged = [[1, 2], [3]]
        assert H._shape_of(ragged, np) == (2,)

    def test_non_degenerate_asarray_failure_returns_none(self):
        # A value np.asarray can't build (ragged + object) -> None.
        class Boom:
            def __array__(self, *a, **k):
                raise ValueError("no array")

        assert H._non_degenerate_check(Boom(), Boom(), np) is None

    def test_ref_degenerate_numeric_exact_match(self):
        # ref all-equal numeric -> test must match exactly (non-NaN path).
        assert H._non_degenerate_check([3, 3, 3], [3, 3, 3], np) == 1.0
        assert H._non_degenerate_check([3, 3, 3], [3, 3, 4], np) == 0.0

    def test_ref_nondegenerate_test_empty_fails(self):
        # ref varies; test all non-finite -> test_finite empty -> 0.0.
        out = H._non_degenerate_check([1.0, 2.0, 3.0], [np.nan, np.nan, np.nan], np)
        assert out == 0.0

    def test_categorical_ref_degenerate_match(self):
        # all-same strings ref -> test must equal exactly.
        assert H._non_degenerate_check(["a", "a"], ["a", "a"], np) == 1.0
        assert H._non_degenerate_check(["a", "a"], ["a", "b"], np) == 0.0

    def test_unrecognized_dtype_returns_none(self):
        # datetime64 is neither numeric/bool nor U/S/O kind -> None.
        arr = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
        assert H._non_degenerate_check(arr, arr, np) is None
