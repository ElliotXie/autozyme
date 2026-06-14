"""Unit tests for zyme.helpers — shared task-side utilities (pure Python).

Covers: find_framework_root, module resolution + patch_namespace /
install_override / inject_override aliasing, inline_upstream cloning,
time_it / emit_summary, peak_memory_mb, the task.yaml tier/threads helpers
(_find_task_yaml, _parse_datasets_minimal, get_tier_params, get_threads),
and the auto_structure_check_all_slots family (numpy-aware shape + degeneracy).

numpy is a hard dep of the package, so the auto_structure tests use it directly
rather than importorskip.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from zyme import helpers as H


# --------------------------------------------------------------------------
# find_framework_root
# --------------------------------------------------------------------------
class TestFindFrameworkRoot:
    def _make_fw(self, base: Path) -> Path:
        fw = base / "autozyme-framework"
        (fw / "autozyme_cli").mkdir(parents=True)
        return fw

    def test_finds_from_nested_start(self, tmp_path: Path):
        fw = self._make_fw(tmp_path)
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        found = H.find_framework_root(start=str(deep))
        assert Path(found) == fw.resolve()

    def test_finds_when_start_is_root_itself(self, tmp_path: Path):
        self._make_fw(tmp_path)
        found = H.find_framework_root(start=str(tmp_path))
        assert Path(found).name == "autozyme-framework"

    def test_rejects_dir_without_cli_child(self, tmp_path: Path):
        # an autozyme-framework dir WITHOUT autozyme_cli/ doesn't count
        (tmp_path / "autozyme-framework").mkdir()
        with pytest.raises(RuntimeError):
            H.find_framework_root(start=str(tmp_path), max_depth=1)

    def test_raises_when_absent(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="not found"):
            H.find_framework_root(start=str(tmp_path), max_depth=2)

    def test_respects_max_depth(self, tmp_path: Path):
        fw = self._make_fw(tmp_path)
        deep = tmp_path / "x" / "y" / "z" / "w"
        deep.mkdir(parents=True)
        # too shallow a budget to reach the framework
        with pytest.raises(RuntimeError):
            H.find_framework_root(start=str(deep), max_depth=1)
        # enough budget finds it
        assert Path(H.find_framework_root(start=str(deep), max_depth=8)) == fw.resolve()


# --------------------------------------------------------------------------
# _resolve_module
# --------------------------------------------------------------------------
class TestResolveModule:
    def test_resolves_imported_module(self):
        assert H._resolve_module("os") is os

    def test_resolves_via_importlib(self):
        # json is importable but exercise the importlib path even if cached
        mod = H._resolve_module("json")
        assert mod is not None and mod.__name__ == "json"

    def test_resolves_class_attribute_walk(self):
        # os.path is an attribute reachable via attribute walk from os
        mod = H._resolve_module("os.path")
        assert mod is os.path

    def test_returns_none_for_garbage(self):
        assert H._resolve_module("totally.not.a.module.xyz") is None


# --------------------------------------------------------------------------
# patch_namespace / install_override / inject_override
# --------------------------------------------------------------------------
@pytest.fixture
def victim_module():
    """A throwaway module registered in sys.modules we can patch + restore."""
    name = "zyme_test_victim_mod"
    mod = types.ModuleType(name)

    def original(x):
        return ("orig", x)

    mod.target = original
    sys.modules[name] = mod
    try:
        yield name, mod, original
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop("zyme_test_alias_mod", None)


class TestPatchNamespace:
    def test_basic_patch_returns_original(self, victim_module):
        name, mod, original = victim_module

        def fast(x):
            return ("fast", x)

        ret = H.patch_namespace("target", name, fast)
        assert ret is original
        assert mod.target is fast
        assert mod.target(5) == ("fast", 5)

    def test_install_override_is_alias(self, victim_module):
        name, mod, original = victim_module

        def fast(x):
            return ("fast", x)

        ret = H.install_override("target", name, fast)
        assert ret is original
        assert mod.target is fast

    def test_inject_override_arg_order(self, victim_module):
        # inject_override(func_name, new_func, module_name) -- swapped arg order
        name, mod, original = victim_module

        def fast(x):
            return ("fast", x)

        ret = H.inject_override("target", fast, module_name=name)
        assert ret is original
        assert mod.target is fast

    def test_missing_module_raises(self):
        def fast():
            pass
        with pytest.raises(RuntimeError, match="neither an importable"):
            H.patch_namespace("anything", "no.such.module.zzz", fast)

    def test_missing_attr_raises(self, victim_module):
        name, mod, _ = victim_module
        with pytest.raises(RuntimeError, match="not found in"):
            H.patch_namespace("no_such_attr", name, lambda: None)

    def test_strict_aliases_raises_on_stale_alias(self, victim_module):
        name, mod, original = victim_module
        # bind the SAME original under a second module -> a stale alias
        alias = types.ModuleType("zyme_test_alias_mod")
        alias.target = original
        sys.modules["zyme_test_alias_mod"] = alias

        with pytest.raises(RuntimeError, match="strict_aliases"):
            H.patch_namespace("target", name, lambda x: x, strict_aliases=True)
        # strict mode raises BEFORE patching
        assert mod.target is original

    def test_lenient_aliases_autopatched(self, victim_module):
        name, mod, original = victim_module
        alias = types.ModuleType("zyme_test_alias_mod")
        alias.target = original
        sys.modules["zyme_test_alias_mod"] = alias

        def fast(x):
            return x

        H.patch_namespace("target", name, fast, strict_aliases=False)
        assert mod.target is fast
        # the alias was auto-patched too
        assert alias.target is fast


# --------------------------------------------------------------------------
# inline_upstream
# --------------------------------------------------------------------------
def _make_upstream(monkeypatch):
    """Register a fake module with a plain python function + a global it reads."""
    mod = types.ModuleType("zyme_test_upstream")
    ns = {}

    def helper():
        return "real_helper"

    # function whose body does a LOAD_GLOBAL on `helper`
    src = "def co(x):\n    return (helper(), x)\n"
    exec(src, mod.__dict__)
    mod.helper = helper
    sys.modules["zyme_test_upstream"] = mod
    return mod


class TestInlineUpstream:
    def teardown_method(self):
        sys.modules.pop("zyme_test_upstream", None)

    def test_clones_and_shim_overrides_global(self, monkeypatch):
        mod = _make_upstream(monkeypatch)
        clone = H.inline_upstream("zyme_test_upstream.co")
        # clone shares code, has its own shim overlay
        assert clone(1) == ("real_helper", 1)
        clone.shim["helper"] = lambda: "fast_helper"
        assert clone(2) == ("fast_helper", 2)
        # upstream untouched
        assert mod.co(3) == ("real_helper", 3)

    def test_bad_qualified_name_raises(self):
        with pytest.raises(ValueError):
            H.inline_upstream("noseparator")

    def test_unresolvable_module_raises(self):
        with pytest.raises(RuntimeError, match="cannot resolve module"):
            H.inline_upstream("no.such.mod.fn")

    def test_missing_function_raises(self, monkeypatch):
        _make_upstream(monkeypatch)
        with pytest.raises(RuntimeError, match="not found"):
            H.inline_upstream("zyme_test_upstream.absent")

    def test_non_function_target_raises(self, monkeypatch):
        mod = _make_upstream(monkeypatch)
        mod.not_a_fn = 42
        with pytest.raises(RuntimeError):
            H.inline_upstream("zyme_test_upstream.not_a_fn")

    def test_clones_kwonly_defaults(self, monkeypatch):
        # a function with keyword-only defaults -> __kwdefaults__ copied to clone
        mod = types.ModuleType("zyme_test_upstream")
        exec("def co(x, *, k=7):\n    return x + k\n", mod.__dict__)
        sys.modules["zyme_test_upstream"] = mod
        clone = H.inline_upstream("zyme_test_upstream.co")
        assert clone.__kwdefaults__ == {"k": 7}
        assert clone(3) == 10
        assert clone(3, k=0) == 3


# --------------------------------------------------------------------------
# time_it / emit_summary
# --------------------------------------------------------------------------
class TestTiming:
    def test_time_it_returns_result_and_elapsed(self):
        result, elapsed = H.time_it(lambda a, b: a + b, 2, 3)
        assert result == 5
        assert elapsed >= 0.0

    def test_time_it_passes_kwargs(self):
        result, _ = H.time_it(lambda a, b=0: a - b, 10, b=4)
        assert result == 6

    def test_emit_summary_lines(self, capsys):
        H.emit_summary(speed_sec=1.5, peak_mb=100.0, cpu_sec=1.2,
                       custom_float=2.25, custom_str="abc", custom_int=7)
        out = capsys.readouterr().out
        assert "speed_sec: 1.500000" in out
        assert "peak_mb: 100.0" in out
        assert "cpu_sec: 1.200000" in out
        assert "custom_float: 2.250000" in out
        assert "custom_str: abc" in out
        assert "custom_int: 7" in out

    def test_emit_summary_defaults_fill_peak_and_cpu(self, capsys):
        H.emit_summary(speed_sec=0.5)
        out = capsys.readouterr().out
        assert "speed_sec: 0.500000" in out
        assert "peak_mb:" in out
        assert "cpu_sec:" in out


# --------------------------------------------------------------------------
# peak_memory_mb
# --------------------------------------------------------------------------
class TestPeakMemory:
    def test_returns_positive_float(self):
        v = H.peak_memory_mb()
        assert isinstance(v, float)
        assert v > 0.0   # a running process always has some RSS

    def test_warn_peak_mb_once_is_idempotent(self, capsys, monkeypatch):
        monkeypatch.setattr(H, "_zyme_peak_mb_warned", False)
        H._warn_peak_mb_once("first")
        H._warn_peak_mb_once("second")
        err = capsys.readouterr().err
        assert err.count("[peak_mb] warning:") == 1
        assert "first" in err
        assert "second" not in err


# --------------------------------------------------------------------------
# get_threads
# --------------------------------------------------------------------------
class TestGetThreads:
    def test_reads_env(self, monkeypatch):
        monkeypatch.setenv("ZYME_THREADS", "6")
        assert H.get_threads() == 6

    def test_invalid_env_falls_through_to_default(self, monkeypatch):
        monkeypatch.setenv("ZYME_THREADS", "notanint")
        assert H.get_threads(default=3) == 3

    def test_default_used_when_unset(self, monkeypatch):
        monkeypatch.delenv("ZYME_THREADS", raising=False)
        assert H.get_threads(default=4) == 4

    def test_falls_back_to_cpu_count(self, monkeypatch):
        monkeypatch.delenv("ZYME_THREADS", raising=False)
        v = H.get_threads()
        assert isinstance(v, int) and v >= 1


# --------------------------------------------------------------------------
# task.yaml parsing: _find_task_yaml / _parse_datasets_minimal / get_tier_params
# --------------------------------------------------------------------------
TASK_YAML = """\
target_repo: x
datasets:
  - {tier: tiny, name: tiny_a, path: data/t.h5ad, params: {n_cells: 500, n_genes: 200}}
  - {tier: medium, name: med_a, path: data/m.h5ad}
metrics:
  - {name: speedup, comparator: gte, threshold: 1.0}
"""


class TestTaskYaml:
    def test_find_task_yaml_walks_up(self, tmp_path, monkeypatch):
        (tmp_path / "task.yaml").write_text(TASK_YAML)
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        monkeypatch.chdir(sub)
        assert H._find_task_yaml() == str(tmp_path / "task.yaml")

    def test_find_task_yaml_none_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert H._find_task_yaml() is None

    def test_parse_datasets_minimal(self, tmp_path):
        p = tmp_path / "task.yaml"
        p.write_text(TASK_YAML)
        entries = H._parse_datasets_minimal(str(p))
        names = {e["name"] for e in entries}
        assert names == {"tiny_a", "med_a"}
        tiny = next(e for e in entries if e["tier"] == "tiny")
        assert tiny["params"] == {"n_cells": 500, "n_genes": 200}
        # medium has no params -> defaulted to {}
        med = next(e for e in entries if e["tier"] == "medium")
        assert med["params"] == {}

    def test_parse_datasets_minimal_missing_file(self, tmp_path):
        assert H._parse_datasets_minimal(str(tmp_path / "nope.yaml")) == []

    def test_get_tier_params_explicit_tier(self, tmp_path, monkeypatch):
        (tmp_path / "task.yaml").write_text(TASK_YAML)
        monkeypatch.chdir(tmp_path)
        assert H.get_tier_params("tiny") == {"n_cells": 500, "n_genes": 200}
        assert H.get_tier_params("medium") == {}

    def test_get_tier_params_from_env(self, tmp_path, monkeypatch):
        (tmp_path / "task.yaml").write_text(TASK_YAML)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ZYME_TIER", "tiny")
        assert H.get_tier_params() == {"n_cells": 500, "n_genes": 200}

    def test_get_tier_params_no_tier_raises(self, tmp_path, monkeypatch):
        (tmp_path / "task.yaml").write_text(TASK_YAML)
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ZYME_TIER", raising=False)
        with pytest.raises(RuntimeError, match="ZYME_TIER"):
            H.get_tier_params()

    def test_get_tier_params_no_yaml_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            H.get_tier_params("tiny")

    def test_get_tier_params_unknown_tier_raises(self, tmp_path, monkeypatch):
        (tmp_path / "task.yaml").write_text(TASK_YAML)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(KeyError, match="ood_large"):
            H.get_tier_params("ood_large")

    def test_parse_datasets_dotted_keys(self, tmp_path):
        # dotted param keys (na.rm) must survive
        p = tmp_path / "task.yaml"
        p.write_text(
            "datasets:\n"
            "  - {tier: tiny, name: a, path: p, params: {na.rm: 1, burn.in: 10}}\n"
        )
        e = H._parse_datasets_minimal(str(p))[0]
        assert e["params"] == {"na.rm": 1, "burn.in": 10}

    def test_parse_datasets_section_ends_on_dedent(self, tmp_path):
        # a non-indented, non-list line after datasets: ends the section
        p = tmp_path / "task.yaml"
        p.write_text(
            "datasets:\n"
            "  - {tier: tiny, name: a, path: p}\n"
            "metrics:\n"
            "  - {tier: ignored, name: should_not_appear, path: x}\n"
        )
        entries = H._parse_datasets_minimal(str(p))
        assert [e["name"] for e in entries] == ["a"]

    def test_parse_datasets_drops_entry_without_name_or_path(self, tmp_path):
        # entry missing name+path is silently dropped (not appended)
        p = tmp_path / "task.yaml"
        p.write_text(
            "datasets:\n"
            "  - {tier: tiny, name: only_name}\n"
            "  - {tier: tiny, name: good, path: p}\n"
        )
        entries = H._parse_datasets_minimal(str(p))
        assert [e["name"] for e in entries] == ["good"]

    def test_parse_datasets_malformed_param_entry_warned(self, tmp_path, capsys):
        # a `key value` (no colon) param entry is dropped with a WARN to stderr
        p = tmp_path / "task.yaml"
        p.write_text(
            "datasets:\n"
            "  - {tier: tiny, name: a, path: p, params: {good: 1, bad_no_colon}}\n"
        )
        e = H._parse_datasets_minimal(str(p))[0]
        assert e["params"] == {"good": 1}
        assert "WARN" in capsys.readouterr().err

    def test_parse_datasets_comment_and_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "task.yaml"
        p.write_text(
            "datasets:\n"
            "  # a comment inside the list region\n"
            "\n"
            "  - {tier: tiny, name: a, path: p}\n"
        )
        entries = H._parse_datasets_minimal(str(p))
        assert [e["name"] for e in entries] == ["a"]


# --------------------------------------------------------------------------
# auto_structure_check_all_slots family
# --------------------------------------------------------------------------
np = pytest.importorskip("numpy")


class TestShapeOf:
    def test_numpy_shape(self):
        assert H._shape_of(np.zeros((3, 4)), np) == (3, 4)

    def test_list_fallback_len(self):
        # something numpy can't asarray cleanly still falls back to len()
        assert H._shape_of([1, 2, 3], np) == (3,)

    def test_scalar_object_is_zero_d(self):
        # np.asarray(object()) succeeds as a 0-d array -> shape ().
        assert H._shape_of(object(), np) == ()

    def test_no_numpy_no_len_returns_none(self):
        # without numpy, a non-sized object has no len() -> None
        assert H._shape_of(object(), None) is None


class TestNonDegenerate:
    def test_ref_varies_test_varies_pass(self):
        assert H._non_degenerate_check([1, 2, 3], [4, 5, 6], np) == 1.0

    def test_ref_varies_test_constant_fail(self):
        assert H._non_degenerate_check([1, 2, 3], [9, 9, 9], np) == 0.0

    def test_ref_constant_test_matches_pass(self):
        assert H._non_degenerate_check([5, 5, 5], [5, 5, 5], np) == 1.0

    def test_ref_constant_test_differs_fail(self):
        assert H._non_degenerate_check([5, 5, 5], [5, 5, 6], np) == 0.0

    def test_ref_all_nan_no_claim(self):
        assert H._non_degenerate_check([np.nan, np.nan], [1, 2], np) == 1.0

    def test_string_categorical_pass(self):
        assert H._non_degenerate_check(["a", "b", "c"], ["x", "y", "z"], np) == 1.0

    def test_string_categorical_degenerate_fail(self):
        assert H._non_degenerate_check(["a", "b"], ["z", "z"], np) == 0.0

    def test_no_numpy_returns_none(self):
        assert H._non_degenerate_check([1, 2], [3, 4], None) is None


class TestAutoStructureCheck:
    def test_ref_none_returns_empty(self):
        assert H.auto_structure_check_all_slots(None, {}) == {}

    def test_ref_not_dict_returns_empty(self):
        assert H.auto_structure_check_all_slots([1, 2, 3], {}) == {}

    def test_present_shape_nondegenerate_all_pass(self):
        ref = {"emb": np.array([1.0, 2.0, 3.0])}
        test = {"emb": np.array([4.0, 5.0, 6.0])}
        out = H.auto_structure_check_all_slots(ref, test)
        assert out["emb_present"] == 1.0
        assert out["emb_shape_match"] == 1.0
        assert out["emb_non_degenerate"] == 1.0

    def test_missing_slot_present_zero(self):
        ref = {"emb": np.array([1.0, 2.0])}
        out = H.auto_structure_check_all_slots(ref, {})
        assert out["emb_present"] == 0.0
        # no further checks once absent
        assert "emb_shape_match" not in out

    def test_shape_mismatch(self):
        ref = {"emb": np.array([1.0, 2.0, 3.0])}
        test = {"emb": np.array([1.0, 2.0])}
        out = H.auto_structure_check_all_slots(ref, test)
        assert out["emb_shape_match"] == 0.0
        assert "emb_non_degenerate" not in out

    def test_stub_to_constant_caught(self):
        ref = {"score": np.array([0.1, 0.5, 0.9])}
        test = {"score": np.array([0.0, 0.0, 0.0])}
        out = H.auto_structure_check_all_slots(ref, test)
        assert out["score_non_degenerate"] == 0.0

    def test_waived_slot_skipped(self):
        ref = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
        test = {"a": np.array([1.0, 2.0])}
        out = H.auto_structure_check_all_slots(ref, test, waived=["b"])
        assert "b_present" not in out
        assert out["a_present"] == 1.0

    def test_nested_dict_recursion(self):
        ref = {"grp": {"x": np.array([1.0, 2.0])}}
        test = {"grp": {"x": np.array([3.0, 4.0])}}
        out = H.auto_structure_check_all_slots(ref, test)
        assert out["grp.x_present"] == 1.0
        assert out["grp.x_shape_match"] == 1.0

    def test_ref_slot_none_skipped(self):
        ref = {"a": None, "b": np.array([1.0, 2.0])}
        test = {"a": 5, "b": np.array([3.0, 4.0])}
        out = H.auto_structure_check_all_slots(ref, test)
        assert "a_present" not in out
        assert out["b_present"] == 1.0


# --------------------------------------------------------------------------
# emit_auto_structure_summary
# --------------------------------------------------------------------------
class TestEmitAutoStructureSummary:
    def test_empty_noop(self, capsys):
        H.emit_auto_structure_summary({})
        assert capsys.readouterr().out == ""

    def test_all_perfect_one_line(self, capsys):
        H.emit_auto_structure_summary({"a_present": 1.0, "b_present": 1.0})
        out = capsys.readouterr().out
        assert "2/2 perfect" in out

    def test_failures_listed(self, capsys):
        H.emit_auto_structure_summary(
            {"a_present": 1.0, "b_shape_match": 0.0, "c_non_degenerate": 0.0})
        out = capsys.readouterr().out
        assert "1/3 pass, 2 FAIL" in out
        assert "b_shape_match: 0.000000" in out
        assert "c_non_degenerate: 0.000000" in out


# --------------------------------------------------------------------------
# profiling context managers (no-op when ZYME_PROFILE unset)
# --------------------------------------------------------------------------
class TestProfilingNoop:
    def test_with_profile_noop(self, monkeypatch):
        monkeypatch.delenv("ZYME_PROFILE", raising=False)
        ran = []
        with H.with_profile():
            ran.append(1)
        assert ran == [1]

    def test_with_subprofile_noop(self, monkeypatch):
        monkeypatch.delenv("ZYME_PROFILE", raising=False)
        ran = []
        with H.with_subprofile("blk"):
            ran.append(1)
        assert ran == [1]

    def test_with_subprofile_active_prints(self, monkeypatch, capsys):
        monkeypatch.setenv("ZYME_PROFILE", "1")
        with H.with_subprofile("blk"):
            pass
        err = capsys.readouterr().err
        assert "[subprofile] blk:" in err

    def test_with_memprof_noop(self, monkeypatch):
        monkeypatch.delenv("ZYME_PROFILE", raising=False)
        ran = []
        with H.with_memprof():
            ran.append(1)
        assert ran == [1]

    def test_with_override_timing_marker_once(self, monkeypatch, capsys):
        # one-shot [override active] marker per name
        monkeypatch.setattr(H, "_zyme_overrides_seen", set())
        with H.with_override_timing("pkg.fn"):
            pass
        with H.with_override_timing("pkg.fn"):
            pass
        out = capsys.readouterr().out
        assert out.count("[override active] pkg.fn") == 1

    def test_record_override_timing_is_noop(self):
        assert H.record_override_timing("pkg.fn", 1.0) is None


# --------------------------------------------------------------------------
# profiling context managers (ACTIVE — exercise the real backends)
# --------------------------------------------------------------------------
class TestProfilingActive:
    def _profile_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZYME_PROFILE", "1")
        monkeypatch.setenv("ZYME_PROFILE_DIR", str(tmp_path))

    def test_cpu_backend_writes_profile_out(self, tmp_path, monkeypatch, capsys):
        self._profile_dir(tmp_path, monkeypatch)
        monkeypatch.setenv("ZYME_PROFILE_BACKEND", "cpu")
        with H.with_profile():
            sum(range(1000))
        assert (tmp_path / "profile.out").exists()
        err = capsys.readouterr().err
        assert "backend=cprofile" in err

    def test_unknown_backend_falls_back_to_cpu(self, tmp_path, monkeypatch, capsys):
        self._profile_dir(tmp_path, monkeypatch)
        monkeypatch.setenv("ZYME_PROFILE_BACKEND", "bogus")
        with H.with_profile():
            pass
        err = capsys.readouterr().err
        assert "unknown ZYME_PROFILE_BACKEND" in err
        assert (tmp_path / "profile.out").exists()

    def test_full_backend_without_scalene_warns(self, tmp_path, monkeypatch, capsys):
        self._profile_dir(tmp_path, monkeypatch)
        monkeypatch.setenv("ZYME_PROFILE_BACKEND", "full")
        monkeypatch.delenv("ZYME_SCALENE_ACTIVE", raising=False)
        ran = []
        with H.with_profile():
            ran.append(1)
        assert ran == [1]
        assert "scalene wrapper" in capsys.readouterr().err

    def test_full_backend_with_scalene_active(self, tmp_path, monkeypatch, capsys):
        self._profile_dir(tmp_path, monkeypatch)
        monkeypatch.setenv("ZYME_PROFILE_BACKEND", "full")
        monkeypatch.setenv("ZYME_SCALENE_ACTIVE", "1")
        with H.with_profile():
            pass
        assert "backend=scalene" in capsys.readouterr().err

    def test_mem_backend_external_wrapper_noop(self, tmp_path, monkeypatch, capsys):
        # ZYME_MEMRAY_ACTIVE set -> helper defers to external wrapper, just prints.
        self._profile_dir(tmp_path, monkeypatch)
        monkeypatch.setenv("ZYME_PROFILE_BACKEND", "mem")
        monkeypatch.setenv("ZYME_MEMRAY_ACTIVE", "1")
        ran = []
        with H.with_profile():
            ran.append(1)
        assert ran == [1]
        assert "backend=memray" in capsys.readouterr().err

    def test_mem_backend_scoped_tracker(self, tmp_path, monkeypatch, capsys):
        # No external wrapper -> in-process memray Tracker. memray is installed
        # in this env; the block must still run and write the trace.
        pytest.importorskip("memray")
        self._profile_dir(tmp_path, monkeypatch)
        monkeypatch.setenv("ZYME_PROFILE_BACKEND", "mem")
        monkeypatch.delenv("ZYME_MEMRAY_ACTIVE", raising=False)
        ran = []
        with H.with_profile():
            ran.append([0] * 1000)
        assert ran  # body executed
        err = capsys.readouterr().err
        assert "backend=memray" in err

    def test_with_memprof_active(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("ZYME_PROFILE", "1")
        with H.with_memprof():
            _ = [0] * 5000
        assert "[memprof]" in capsys.readouterr().err


# --------------------------------------------------------------------------
# auto_structure: pandas DataFrame recursion (pandas is installed in this env)
# --------------------------------------------------------------------------
class TestAutoStructureDataFrame:
    def test_dataframe_columns_checked_individually(self):
        pd = pytest.importorskip("pandas")
        ref = {"de": pd.DataFrame({"score": [0.1, 0.5, 0.9], "name": ["a", "b", "c"]})}
        test = {"de": pd.DataFrame({"score": [0.2, 0.6, 0.8], "name": ["x", "y", "z"]})}
        out = H.auto_structure_check_all_slots(ref, test)
        # per-column metrics emitted under the slot prefix
        assert out["de.score_present"] == 1.0
        assert out["de.score_non_degenerate"] == 1.0
        assert out["de.name_present"] == 1.0

    def test_dataframe_column_stubbed_to_constant_caught(self):
        pd = pytest.importorskip("pandas")
        ref = {"de": pd.DataFrame({"score": [0.1, 0.5, 0.9]})}
        test = {"de": pd.DataFrame({"score": [0.0, 0.0, 0.0]})}
        out = H.auto_structure_check_all_slots(ref, test)
        assert out["de.score_non_degenerate"] == 0.0
