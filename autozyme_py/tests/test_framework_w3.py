"""Wave-3 gap-fill across the autozyme framework-core modules.

Picks up the reachable branches wave-1/2 left in:
  - _threads          : env-var int() failures, option int() failure, cap coercion
  - _utils            : non-bounded absolute-path fallthrough, traversal in 2nd cand
  - _intercept_probe  : counter increment, install(patch in registry), write OSError
  - _benchmark        : reps < 1 rejection
  - _smoke            : spec loader None failure
  - _verify_worker    : non-darwin/psutil RSS, missing-patch / no-smoke error raises
  - _core             : tested_upstream_versions non-list, quiet inactive marker,
                        deactivate-unregistered skip, activate disabled-env list path
  - __init__          : _banner disabled + zero-patch short-circuits

stdlib-only; the synthetic ``_test_json`` patch stands in for a real upstream.
"""
from __future__ import annotations

import builtins
import json
import os
import sys

import pytest

import autozyme
from autozyme import _core as core
from autozyme._core import _REGISTRY, _import_submodule


# ==========================================================================
# _threads — error fall-through branches
# ==========================================================================
def test_auto_threads_env_non_int_falls_through(monkeypatch):
    # garbage in the first env var -> ValueError swallowed (lines 81-82),
    # then a valid value in a later var wins.
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
              "AUTOZYMER_THREADS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ZYME_THREADS", "not-an-int")
    monkeypatch.setenv("AUTOZYME_THREADS", "5")
    from autozyme import _threads as T
    assert T.auto_threads() == 5


def test_auto_threads_env_negative_ignored(monkeypatch):
    # "-1" parses but n>=1 fails -> falls through to AUTOZYME_THREADS
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
              "AUTOZYMER_THREADS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ZYME_THREADS", "-1")
    monkeypatch.setenv("AUTOZYME_THREADS", "3")
    from autozyme import _threads as T
    assert T.auto_threads() == 3


def test_auto_threads_option_non_int_falls_to_hardware(monkeypatch):
    # module option is a non-coercible object -> TypeError swallowed (88-89),
    # falls through to the hardware default.
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
              "AUTOZYMER_THREADS"):
        monkeypatch.delenv(v, raising=False)
    from autozyme import _threads as T
    monkeypatch.setattr(T, "_AUTOZYME_THREADS_OPTION", object())
    n = T.auto_threads()
    assert isinstance(n, int) and n >= 1


def test_auto_threads_cap_uncoercible_raises(monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
              "AUTOZYMER_THREADS"):
        monkeypatch.delenv(v, raising=False)
    from autozyme import _threads as T
    monkeypatch.setattr(T, "_AUTOZYME_THREADS_OPTION", None)
    with pytest.raises(ValueError, match="cap must be int or None"):
        T.auto_threads(cap="eight")  # lines 104-105


def test_auto_threads_cap_below_one_raises(monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
              "AUTOZYMER_THREADS"):
        monkeypatch.delenv(v, raising=False)
    from autozyme import _threads as T
    monkeypatch.setattr(T, "_AUTOZYME_THREADS_OPTION", None)
    with pytest.raises(ValueError, match="cap must be >= 1"):
        T.auto_threads(cap=0)  # line 110


def test_auto_threads_option_positive_int_wins(monkeypatch):
    # A valid positive option (no env vars) returns via lines 88-89.
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
              "AUTOZYMER_THREADS"):
        monkeypatch.delenv(v, raising=False)
    from autozyme import _threads as T
    monkeypatch.setattr(T, "_AUTOZYME_THREADS_OPTION", 5)
    assert T.auto_threads(cap=2) == 5  # option wins over cap


def test_auto_threads_option_zero_falls_through(monkeypatch):
    # Option coerces to int but n >= 1 fails -> falls through past 88-89 to hw.
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
              "AUTOZYMER_THREADS"):
        monkeypatch.delenv(v, raising=False)
    from autozyme import _threads as T
    monkeypatch.setattr(T, "_AUTOZYME_THREADS_OPTION", 0)
    n = T.auto_threads()
    assert isinstance(n, int) and n >= 1


# ==========================================================================
# _utils — branches wave-1 left (28-29 commonpath ValueError; 68 abs trust)
# ==========================================================================
def test_resolve_dataset_path_bounded_candidate_rejected_then_resolved(tmp_path):
    # First candidate (task_dir/raw) is INSIDE -> resolves immediately, but to
    # exercise line 68's `continue` we craft a raw_path whose first bounded
    # candidate is a traversal outside the task, then the 3rd candidate
    # (data/<basename>) resolves. The traversal candidate exists but fails
    # _within -> continue (line 68).
    from autozyme import _utils as U
    task = tmp_path / "task"
    (task / "data").mkdir(parents=True)
    # the real file lives at task/data/secret.csv (basename fallback target)
    (task / "data" / "secret.csv").write_text("real")
    # an outside file the traversal candidate would point at
    (tmp_path / "secret.csv").write_text("outside")
    got = U.resolve_dataset_path(str(task), "../secret.csv")
    # bounded traversal candidate skipped (68); basename fallback wins
    assert os.path.samefile(got, task / "data" / "secret.csv")



def test_resolve_dataset_path_absolute_outside_task(tmp_path):
    # An absolute path OUTSIDE the task dir is trusted (must_bound False),
    # so it resolves even though _within would reject it. Covers line 68's
    # `must_bound` False short-circuit on the absolute-path candidate.
    outside = tmp_path.parent / "trusted_abs.csv"
    outside.write_text("ok")
    task = tmp_path / "task"
    task.mkdir()
    from autozyme import _utils as U
    got = U.resolve_dataset_path(str(task), str(outside))
    assert os.path.samefile(got, outside)


def test_within_commonpath_valueerror_on_mismatched_drive(monkeypatch):
    # Force os.path.commonpath to raise ValueError (it does cross-drive on
    # Windows; we simulate by monkeypatching) -> _within returns False (28-29).
    from autozyme import _utils as U

    def _boom(_paths):
        raise ValueError("paths on different drives")

    monkeypatch.setattr(U.os.path, "commonpath", _boom)
    assert U._within("/a/b", "/a/b/c") is False


# ==========================================================================
# _intercept_probe — counter increment + install(registered) + write OSError
# ==========================================================================
def test_intercept_counted_increments(monkeypatch, tmp_path):
    from autozyme import _intercept_probe as IP

    saved_make = core._make_dispatcher
    saved_installed = IP._INSTALLED
    saved_orig = IP._ORIG_MAKE_DISPATCHER
    saved_counts = dict(IP._COUNTS)
    saved_keys = dict(IP._KEYS)
    try:
        IP._INSTALLED = False
        IP._COUNTS.clear()
        IP._KEYS.clear()

        def fast(*a, **k):
            return "fast"

        def orig(*a, **k):
            return "orig"

        # record a key for fast so the counter knows what to bump
        IP._KEYS[id(fast)] = "mod:attr"
        IP.install()  # wraps _make_dispatcher
        disp = core._make_dispatcher(fast, orig)
        assert disp() == "fast"
        assert disp() == "fast"
        # counter incremented twice (line 61)
        assert IP._COUNTS["mod:attr"] == 2
        # under disabled() the original runs; the counter still wraps but the
        # real dispatcher routes to orig.
        with autozyme.disabled():
            assert disp() == "orig"
    finally:
        core._make_dispatcher = saved_make
        IP._INSTALLED = saved_installed
        IP._ORIG_MAKE_DISPATCHER = saved_orig
        IP._COUNTS.clear()
        IP._COUNTS.update(saved_counts)
        IP._KEYS.clear()
        IP._KEYS.update(saved_keys)


def test_intercept_install_records_registered_patch(monkeypatch):
    from autozyme import _intercept_probe as IP

    saved_make = core._make_dispatcher
    saved_installed = IP._INSTALLED
    saved_orig = IP._ORIG_MAKE_DISPATCHER
    saved_keys = dict(IP._KEYS)
    try:
        IP._INSTALLED = True  # skip the global install block; test 84-85 only
        IP._KEYS.clear()

        def fast(*a, **k):
            return None

        autozyme.register_patch("ip_reg_w3", [("json", "loads", fast)])
        # patch_name present + in registry -> _record_target_keys runs (line 85)
        IP.install("ip_reg_w3")
        assert IP._KEYS[id(fast)] == "json:loads"
    finally:
        core._make_dispatcher = saved_make
        IP._INSTALLED = saved_installed
        IP._ORIG_MAKE_DISPATCHER = saved_orig
        IP._KEYS.clear()
        IP._KEYS.update(saved_keys)
        _REGISTRY.pop("ip_reg_w3", None)


def test_intercept_write_counts_oserror_swallowed(monkeypatch, tmp_path):
    from autozyme import _intercept_probe as IP
    # Point at a path whose parent does not exist -> write_text raises OSError,
    # which the except clause swallows (lines 94-95).
    bad = tmp_path / "no_such_dir" / "counts.json"
    monkeypatch.setenv("ZYME_INTERCEPT_OUT", str(bad))
    saved = dict(IP._COUNTS)
    try:
        IP._COUNTS.clear()
        IP._COUNTS["a:b"] = 1
        IP._write_counts()  # must NOT raise
        assert not bad.exists()
    finally:
        IP._COUNTS.clear()
        IP._COUNTS.update(saved)


# ==========================================================================
# _benchmark — reps < 1 rejection (lines 56, 58)
# ==========================================================================
def test_benchmark_reps_zero_rejected(tmp_path):
    from autozyme._benchmark import benchmark
    _import_submodule("_test_json")
    try:
        with pytest.raises(ValueError, match="reps must be"):
            benchmark("_test_json", str(tmp_path), reps=0, verbose=False)
        # negative also rejected by the < 1 guard
        with pytest.raises(ValueError, match="reps must be"):
            benchmark("_test_json", str(tmp_path), reps=-3, verbose=False)
    finally:
        autozyme.deactivate("_test_json")


def test_benchmark_import_succeeds_but_unregistered_keyerror(tmp_path, monkeypatch):
    from autozyme import _benchmark as B
    # _import_submodule returns without registering -> _REGISTRY.get is None ->
    # KeyError "no patch registered" (lines 54-56).
    monkeypatch.setattr(B, "_import_submodule", lambda name: None)
    with pytest.raises(KeyError, match="no patch registered"):
        B.benchmark("ghost_unreg_bench_w3", str(tmp_path), verbose=False)


def test_benchmark_no_smoke_recipe_rejected(tmp_path):
    from autozyme._benchmark import benchmark
    # register a patch with NO smoke -> ValueError (no smoke recipe branch)
    autozyme.register_patch("nosmoke_w3", [("json", "loads", lambda s: s)])
    try:
        with pytest.raises(ValueError, match="no smoke recipe"):
            benchmark("nosmoke_w3", str(tmp_path), verbose=False)
    finally:
        _REGISTRY.pop("nosmoke_w3", None)


# ==========================================================================
# _smoke — spec loader None (line 16)
# ==========================================================================
def test_resolve_smoke_spec_none_raises(tmp_path, monkeypatch):
    from autozyme import _smoke as SM
    attest = tmp_path / "attest"
    attest.mkdir()
    (attest / "smoke.py").write_text("smoke = {}\n")

    class _SpecNoLoader:
        loader = None

    monkeypatch.setattr(SM.importlib.util, "spec_from_file_location",
                        lambda *a, **k: _SpecNoLoader())
    with pytest.raises(RuntimeError, match="could not load attest smoke module"):
        SM.resolve_smoke(str(tmp_path), None)


# ==========================================================================
# _verify_worker — non-darwin RSS, psutil fallback, error raises
# ==========================================================================
def test_verify_worker_peak_rss_linux_branch(monkeypatch):
    from autozyme import _verify_worker as W
    # Force the non-darwin POSIX branch: ru_maxrss treated as KiB -> /1024.
    monkeypatch.setattr(W.sys, "platform", "linux")

    class _RU:
        ru_maxrss = 2048  # KiB

    class _FakeResource:
        RUSAGE_SELF = 0

        @staticmethod
        def getrusage(_who):
            return _RU()

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "resource":
            return _FakeResource
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert W._peak_rss_mb() == pytest.approx(2048 / 1024)


def test_verify_worker_peak_rss_psutil_fallback(monkeypatch):
    from autozyme import _verify_worker as W
    real_import = builtins.__import__

    class _MemInfo:
        peak_wset = 5 * 1024 * 1024  # 5 MiB in bytes

    class _Proc:
        def memory_info(self):
            return _MemInfo()

    class _FakePsutil:
        Process = _Proc

    def fake_import(name, *a, **k):
        if name == "resource":
            raise ImportError("no resource module")
        if name == "psutil":
            return _FakePsutil
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert W._peak_rss_mb() == pytest.approx(5.0)


def test_verify_worker_peak_rss_none_when_all_fail(monkeypatch):
    from autozyme import _verify_worker as W
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in ("resource", "psutil"):
            raise ImportError("absent")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert W._peak_rss_mb() is None


def test_verify_worker_main_unregistered_patch_raises(tmp_path):
    from autozyme import _verify_worker as W
    # _import_submodule raises ImportError for an unknown patch; main should
    # propagate (the patch never registers).
    with pytest.raises(ImportError):
        W.main(["--patch", "definitely_not_a_patch_zzz",
                "--task-dir", str(tmp_path), "--tier", "tiny",
                "--output-dir", str(tmp_path / "o")])


def test_verify_worker_main_no_smoke_raises(tmp_path, monkeypatch):
    from autozyme import _verify_worker as W
    # Register a smoke-less patch and make _import_submodule a no-op so main()
    # reaches the `patch.smoke is None` guard (line 88-91).
    autozyme.register_patch("ws_nosmoke_w3", [("json", "loads", lambda s: s)])
    monkeypatch.setattr(W, "_import_submodule", lambda name: None,
                        raising=False)
    try:
        with pytest.raises(RuntimeError, match="has no smoke recipe"):
            W.main(["--patch", "ws_nosmoke_w3", "--task-dir", str(tmp_path),
                    "--tier", "tiny", "--output-dir", str(tmp_path / "o")])
    finally:
        _REGISTRY.pop("ws_nosmoke_w3", None)


def test_verify_worker_main_resolve_smoke_none_raises(tmp_path, monkeypatch):
    from autozyme import _verify_worker as W
    # Patch HAS a smoke dict so the patch.smoke guard passes, but resolve_smoke
    # returns None -> the "no smoke recipe for patch" raise (lines 95-99).
    smoke = {"load": lambda td, t: None, "call": lambda i: i,
             "save": lambda r, od, *, tier: None}
    autozyme.register_patch("ws_resolvenone_w3",
                            [("json", "loads", lambda s: s)], smoke=smoke)
    monkeypatch.setattr(W, "_import_submodule", lambda name: None,
                        raising=False)
    monkeypatch.setattr("autozyme._smoke.resolve_smoke",
                        lambda task_dir, patch: None)
    try:
        with pytest.raises(RuntimeError, match="no smoke recipe for patch"):
            W.main(["--patch", "ws_resolvenone_w3", "--task-dir", str(tmp_path),
                    "--tier", "tiny", "--output-dir", str(tmp_path / "o")])
    finally:
        _REGISTRY.pop("ws_resolvenone_w3", None)


# ==========================================================================
# _core — gap branches wave-1/2 missed
# ==========================================================================
def test_register_patch_tested_versions_value_not_list():
    # str key but non-list value -> TypeError (line 211-213, str-key path)
    try:
        with pytest.raises(TypeError, match="must be list"):
            autozyme.register_patch(
                "tuv_notlist_w3", [("json", "loads", lambda s: s)],
                tested_upstream_versions={"json": "1.0.0"},  # value not a list
            )
    finally:
        _REGISTRY.pop("tuv_notlist_w3", None)


def test_register_patch_conflict_raises():
    # Two patches claiming the same (upstream, attr) -> ValueError naming the
    # already-claiming patch (lines 189, 193-194).
    autozyme.register_patch("claim_a_w3", [("json", "loads", lambda s: s)])
    try:
        with pytest.raises(ValueError, match=r"already claimed by patch 'claim_a_w3'"):
            autozyme.register_patch("claim_b_w3", [("json", "loads", lambda s: s)])
    finally:
        _REGISTRY.pop("claim_a_w3", None)
        _REGISTRY.pop("claim_b_w3", None)


def test_register_patch_same_name_reregister_skips_self(monkeypatch):
    # Re-registering the SAME name with the SAME targets must NOT self-conflict:
    # the loop skips the same-named entry (line 189). Ensure the json::dumps
    # claim isn't already taken by inserting on a fresh attr.
    autozyme.register_patch("reself_w3", [("json", "loads", lambda s: s)])
    try:
        # second registration with same name + same target re-binds cleanly
        autozyme.register_patch("reself_w3", [("json", "loads", lambda s: "v2")])
        assert "reself_w3" in _REGISTRY
    finally:
        _REGISTRY.pop("reself_w3", None)


def test_emit_inactive_marker_quiet(monkeypatch, capsys):
    monkeypatch.setenv("AUTOZYME_QUIET", "1")
    core._emit_inactive_marker("ghost_w3", "upstream not installed: foo")
    # quiet short-circuit (line 378) -> nothing on stderr
    assert capsys.readouterr().err == ""


def test_deactivate_unregistered_name_skips(monkeypatch):
    # A name that resolves (is in _AVAILABLE) but was never imported/registered
    # -> the `p is None: continue` skip (line 582). Use a fake available name.
    monkeypatch.setattr(core, "_AVAILABLE", core._AVAILABLE + ["ghostonly_w3"])
    # Not in _REGISTRY -> deactivate loops, finds None, continues, no error.
    autozyme.deactivate("ghostonly_w3")


def test_activate_list_disabled_env_returns_all_false(monkeypatch):
    # AUTOZYME_DISABLED set + a LIST target -> dict of all-False (line 543),
    # distinct from the single-name False path wave-1 covered.
    # Distinct (module, attr) pairs that nothing else (incl. _test_json which
    # owns json::dumps) claims, so registration never trips the conflict guard.
    autozyme.register_patch("envoff_a_w3", [("json", "load", lambda s: s)])
    autozyme.register_patch("envoff_b_w3", [("json", "JSONDecoder", lambda: None)])
    monkeypatch.setattr(core, "_AVAILABLE",
                        core._AVAILABLE + ["envoff_a_w3", "envoff_b_w3"])
    monkeypatch.setenv("AUTOZYME_DISABLED", "1")
    try:
        out = autozyme.activate(["envoff_a_w3", "envoff_b_w3"])
        assert out == {"envoff_a_w3": False, "envoff_b_w3": False}
    finally:
        _REGISTRY.pop("envoff_a_w3", None)
        _REGISTRY.pop("envoff_b_w3", None)


def test_activate_list_skips_uninstalled(monkeypatch, capsys):
    # In the multi-target loop, an uninstalled patch emits the inactive marker
    # and records False (lines 558-561) without raising.
    autozyme.register_patch("uninst_w3", [("json", "JSONEncoder", lambda: None)])
    monkeypatch.setattr(core, "_AVAILABLE", core._AVAILABLE + ["uninst_w3"])
    monkeypatch.setattr(core, "_probe_patch_installed",
                        lambda name: (False, "upstream not installed: zzz")
                        if name == "uninst_w3" else (True, None))
    monkeypatch.delenv("AUTOZYME_DISABLED", raising=False)
    monkeypatch.delenv("AUTOZYME_DISABLE", raising=False)
    try:
        out = autozyme.activate(["uninst_w3"])
        assert out == {"uninst_w3": False}
    finally:
        _REGISTRY.pop("uninst_w3", None)


def test_activate_one_resolve_import_error_partial(capsys):
    # One target resolves (json::loads), another's UPSTREAM MODULE is missing
    # -> _resolve_target raises ImportError -> tracked as missing (394-400),
    # partial activation still True.
    def fast_loads(s, **k):
        return "x"

    def fast_ghost(*a, **k):
        return None

    autozyme.register_patch(
        "ar_partial_w3",
        [("json", "loads", fast_loads),
         ("no_such_module_zzz_w3.sub", "fn", fast_ghost)],
    )
    try:
        p = _REGISTRY["ar_partial_w3"]
        assert core._activate_one(p) is True   # partial -> still True
        assert p.injected is True
        err = capsys.readouterr().err
        assert "partial activation" in err
        core._deactivate_one(p)
    finally:
        _REGISTRY.pop("ar_partial_w3", None)


def test_activate_single_uninstalled_emits_and_false(monkeypatch, capsys):
    # Single-name activate path where the patch's upstream is not installed
    # -> _emit_inactive_marker + return False (lines 549-551).
    autozyme.register_patch("single_uninst_w3", [("json", "loads", lambda s: s)])
    monkeypatch.setattr(core, "_AVAILABLE", core._AVAILABLE + ["single_uninst_w3"])
    monkeypatch.setattr(core, "_probe_patch_installed",
                        lambda name: (False, "upstream not installed: q")
                        if name == "single_uninst_w3" else (True, None))
    monkeypatch.delenv("AUTOZYME_DISABLED", raising=False)
    monkeypatch.delenv("AUTOZYME_DISABLE", raising=False)
    try:
        assert autozyme.activate("single_uninst_w3") is False
        assert "NOT activated" in capsys.readouterr().err
    finally:
        _REGISTRY.pop("single_uninst_w3", None)


def test_activate_list_installed_path_activates(monkeypatch):
    # Multi-target loop where the patch IS installed -> _activate_one runs and
    # records its result (line 562-563), distinct from the uninstalled branch.
    def fast(s, **k):
        return "patched"

    autozyme.register_patch("list_ok_w3", [("json", "loads", fast)])
    monkeypatch.setattr(core, "_AVAILABLE", core._AVAILABLE + ["list_ok_w3"])
    monkeypatch.setattr(core, "_probe_patch_installed",
                        lambda name: (True, None))
    monkeypatch.setattr(core, "_import_submodule", lambda name: None)
    monkeypatch.delenv("AUTOZYME_DISABLED", raising=False)
    monkeypatch.delenv("AUTOZYME_DISABLE", raising=False)
    try:
        out = autozyme.activate(["list_ok_w3"])
        assert out == {"list_ok_w3": True}
        autozyme.deactivate("list_ok_w3")
    finally:
        _REGISTRY.pop("list_ok_w3", None)


def test_inspect_active_resolve_import_error_target_view(monkeypatch):
    # An active patch whose target's _resolve_target later raises ImportError
    # during inspect() -> current=None branch in the target-view loop (649-650).
    autozyme.register_patch("insp_imp_w3", [("json", "loads", lambda s: s)])
    monkeypatch.setattr(core, "_AVAILABLE", core._AVAILABLE + ["insp_imp_w3"])
    monkeypatch.setattr(core, "_probe_patch_installed", lambda name: (True, None))
    monkeypatch.setattr(core, "_import_submodule", lambda name: None)

    real_resolve = core._resolve_target

    def flaky_resolve(path):
        if path == "json":
            raise ImportError("json vanished mid-inspect")
        return real_resolve(path)

    monkeypatch.setattr(core, "_resolve_target", flaky_resolve)
    try:
        res = core.inspect("insp_imp_w3")
        # resolve raised inside the per-target loop -> current None recorded
        assert res["targets"][0]["currently_bound_to_fast"] is False
    finally:
        _REGISTRY.pop("insp_imp_w3", None)


def test_verify_worker_main_patch_none_after_import(tmp_path, monkeypatch):
    from autozyme import _verify_worker as W
    # _import_submodule "succeeds" but the patch never lands in _REGISTRY ->
    # patch is None -> RuntimeError "did not register on import" (lines 84-87).
    # main() does `from autozyme._core import _import_submodule`, so we patch
    # the source symbol on _core (the local import binds to it at call time).
    monkeypatch.setattr(core, "_import_submodule", lambda name: None)
    with pytest.raises(RuntimeError, match="did not register on import"):
        W.main(["--patch", "ghost_unregistered_w3", "--task-dir", str(tmp_path),
                "--tier", "tiny", "--output-dir", str(tmp_path / "o")])


def test_inspect_import_error_returns_uninstalled(monkeypatch):
    # _probe says installed, but _import_submodule raises ImportError ->
    # inspect returns the uninstalled-shaped dict (lines 634-640).
    monkeypatch.setattr(core, "_AVAILABLE", core._AVAILABLE + ["impfail_w3"])
    monkeypatch.setattr(core, "_probe_patch_installed",
                        lambda name: (True, None))

    def _boom(name):
        raise ImportError("submodule import blew up")

    monkeypatch.setattr(core, "_import_submodule", _boom)
    res = autozyme.inspect("impfail_w3")
    assert res["status"] == "uninstalled"
    assert res["targets"] == []
    assert "blew up" in res["error"]


# ==========================================================================
# __init__._banner short-circuits
# ==========================================================================
def test_banner_disabled_short_circuit(monkeypatch, capsys):
    import autozyme as A
    monkeypatch.setenv("AUTOZYME_DISABLED", "1")
    A._banner()  # line 59 return
    assert capsys.readouterr().err == ""


def test_banner_zero_patches_short_circuit(monkeypatch, capsys):
    import autozyme as A
    monkeypatch.delenv("AUTOZYME_DISABLED", raising=False)
    monkeypatch.setattr(A, "list_patches", lambda: [])
    A._banner()  # n_patches == 0 -> line 63 return, no print
    assert capsys.readouterr().err == ""


def test_banner_prints_when_patches_present(monkeypatch, capsys):
    import autozyme as A
    monkeypatch.delenv("AUTOZYME_DISABLED", raising=False)
    monkeypatch.setattr(A, "list_patches", lambda: ["a", "b"])
    monkeypatch.setattr(A, "list_subsets", lambda: ["s"])
    A._banner()
    err = capsys.readouterr().err
    assert "autozyme" in err and "2 patches available" in err
