"""Unit tests for the remaining framework-core modules:

  - _utils         : dataset-path resolution + traversal guard
  - _subsets       : declarative manifests (SUBSETS / UPSTREAMS / CONFLICTS)
  - _smoke         : attest/smoke.py resolution vs patch default
  - _intercept_probe : pure pieces of the intercept counter
  - _verify_worker : arg parsing + peak-rss + end-to-end main() on _test_json
  - _benchmark     : _summarize + benchmark() on _test_json
  - __main__       : the `python -m autozyme` dashboard (captured stdout)
  - _core          : gap-filling for inspect / env_snapshot / disabled /
                     partial activation / did-you-mean / version helpers

Uses the stdlib-only synthetic ``_test_json`` patch so no heavy upstream is
needed. Anything requiring a scientific lib is importorskip'd.
"""
from __future__ import annotations

import builtins
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import autozyme
from autozyme import _utils as U
from autozyme import _subsets as SUB
from autozyme import _smoke as SM
from autozyme import _core as core
from autozyme._core import _REGISTRY, _import_submodule


# ==========================================================================
# _utils — resolve_dataset_path + _within
# ==========================================================================
def test_within_inside(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    f = d / "x.csv"
    f.write_text("x")
    assert U._within(str(tmp_path), str(f)) is True


def test_within_traversal_rejected(tmp_path):
    # task_dir/../../etc -> not inside
    assert U._within(str(tmp_path), str(tmp_path.parent / "outside")) is False


def test_within_sibling_prefix_not_accepted(tmp_path):
    # /a/b should not accept /a/bc (commonpath, not string prefix)
    base = tmp_path / "b"
    base.mkdir()
    sibling = tmp_path / "bc"
    sibling.mkdir()
    assert U._within(str(base), str(sibling)) is False


def test_resolve_dataset_path_relative(tmp_path):
    (tmp_path / "data").mkdir()
    f = tmp_path / "data" / "tiny.csv"
    f.write_text("x")
    got = U.resolve_dataset_path(str(tmp_path), "data/tiny.csv")
    assert os.path.samefile(got, f)


def test_resolve_dataset_path_dot_slash(tmp_path):
    (tmp_path / "data").mkdir()
    f = tmp_path / "data" / "tiny.csv"
    f.write_text("x")
    got = U.resolve_dataset_path(str(tmp_path), "./data/tiny.csv")
    assert os.path.samefile(got, f)


def test_resolve_dataset_path_basename_fallback(tmp_path):
    (tmp_path / "data").mkdir()
    f = tmp_path / "data" / "tiny.csv"
    f.write_text("x")
    # stale path whose only valid resolution is data/<basename>
    got = U.resolve_dataset_path(str(tmp_path), "old/layout/tiny.csv")
    assert os.path.samefile(got, f)


def test_resolve_dataset_path_absolute_trusted(tmp_path):
    f = tmp_path / "abs.csv"
    f.write_text("x")
    got = U.resolve_dataset_path(str(tmp_path), str(f))
    assert os.path.samefile(got, f)


def test_resolve_dataset_path_traversal_rejected(tmp_path):
    # create a file outside the task dir, reference it via ../
    outside = tmp_path.parent / "secret.csv"
    outside.write_text("secret")
    task = tmp_path / "task"
    task.mkdir()
    with pytest.raises(FileNotFoundError):
        U.resolve_dataset_path(str(task), "../secret.csv")


def test_resolve_dataset_path_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="could not resolve"):
        U.resolve_dataset_path(str(tmp_path), "nope/missing.csv")


# ==========================================================================
# _subsets manifests
# ==========================================================================
def test_subsets_structure():
    assert "scrna_core" in SUB.SUBSETS
    assert SUB.SUBSETS["scrna_core"] == ["scanpy", "sccoda", "scvelo"]
    assert all(isinstance(v, list) for v in SUB.SUBSETS.values())


def test_upstreams_manifest():
    assert SUB.UPSTREAMS["scanpy"] == ["scanpy"]
    assert "tensorflow" in SUB.UPSTREAMS["sccoda"]
    # synthetic test patch is registered against json
    assert SUB.UPSTREAMS["_test_json"] == ["json"]


def test_conflicts_manifest():
    assert len(SUB.CONFLICTS) >= 1
    pair, reason = SUB.CONFLICTS[0]
    assert isinstance(pair, frozenset)
    assert pair == frozenset({"sccoda", "xclim"})
    assert isinstance(reason, str) and reason


# ==========================================================================
# _smoke.resolve_smoke
# ==========================================================================
class _PatchWithSmoke:
    smoke = {"load": lambda *a: None, "call": lambda *a: None, "save": lambda *a: None}


def test_resolve_smoke_falls_back_to_patch(tmp_path):
    got = SM.resolve_smoke(str(tmp_path), _PatchWithSmoke())
    assert got is _PatchWithSmoke.smoke


def test_resolve_smoke_none_patch(tmp_path):
    assert SM.resolve_smoke(str(tmp_path), None) is None


def _write_attest_smoke(tmp_path, body):
    attest = tmp_path / "attest"
    attest.mkdir()
    (attest / "smoke.py").write_text(body)


def test_resolve_smoke_task_file_wins(tmp_path):
    _write_attest_smoke(tmp_path, (
        "def _l(td, t): return {}\n"
        "def _c(i): return i\n"
        "def _s(r, od, *, tier): pass\n"
        "smoke = dict(load=_l, call=_c, save=_s)\n"
    ))
    got = SM.resolve_smoke(str(tmp_path), _PatchWithSmoke())
    assert got is not _PatchWithSmoke.smoke
    assert set(got.keys()) == {"load", "call", "save"}


def test_resolve_smoke_task_file_missing_smoke_var(tmp_path):
    _write_attest_smoke(tmp_path, "x = 1\n")
    with pytest.raises(RuntimeError, match="must define smoke"):
        SM.resolve_smoke(str(tmp_path), _PatchWithSmoke())


def test_resolve_smoke_task_file_smoke_not_dict(tmp_path):
    _write_attest_smoke(tmp_path, "smoke = 5\n")
    with pytest.raises(RuntimeError, match="must be a dict"):
        SM.resolve_smoke(str(tmp_path), _PatchWithSmoke())


def test_resolve_smoke_task_file_missing_keys(tmp_path):
    _write_attest_smoke(tmp_path, "smoke = dict(load=lambda *a: None)\n")
    with pytest.raises(RuntimeError, match="missing keys"):
        SM.resolve_smoke(str(tmp_path), _PatchWithSmoke())


# ==========================================================================
# _intercept_probe — pure pieces
# ==========================================================================
def test_intercept_install_from_env_disabled(monkeypatch):
    from autozyme import _intercept_probe as IP
    monkeypatch.delenv("ZYME_INSTRUMENT_INTERCEPTS", raising=False)
    assert IP.install_from_env() is False


def test_intercept_record_target_keys_and_key_lookup():
    from autozyme import _intercept_probe as IP

    def fast(): pass
    patch = type("P", (), {"targets": [("scanpy.preprocessing", "scale", fast)]})()
    IP._record_target_keys(patch)
    assert IP._key_for_fast_fn(fast) == "scanpy.preprocessing:scale"
    # unknown fn -> None
    assert IP._key_for_fast_fn(lambda: None) is None


def test_intercept_write_counts_no_env(monkeypatch):
    from autozyme import _intercept_probe as IP
    monkeypatch.delenv("ZYME_INTERCEPT_OUT", raising=False)
    # no env var -> no-op, no exception
    IP._write_counts()


def test_intercept_write_counts_to_file(monkeypatch, tmp_path):
    from autozyme import _intercept_probe as IP
    out = tmp_path / "counts.json"
    monkeypatch.setenv("ZYME_INTERCEPT_OUT", str(out))
    monkeypatch.setattr(IP, "_COUNTS", {"a:b": 3})
    IP._write_counts()
    assert json.loads(out.read_text()) == {"a:b": 3}


def test_intercept_install_from_env_enabled(monkeypatch):
    from autozyme import _intercept_probe as IP
    monkeypatch.setenv("ZYME_INSTRUMENT_INTERCEPTS", "1")
    monkeypatch.delenv("ZYME_INTERCEPT_PATCH", raising=False)
    # save + restore the wrapped dispatcher so we don't leak the monkeypatch
    saved_make = core._make_dispatcher
    saved_installed = IP._INSTALLED
    saved_orig = IP._ORIG_MAKE_DISPATCHER
    try:
        assert IP.install_from_env() is True
        assert IP._INSTALLED is True
        # wrapped dispatcher still produces a working dispatcher
        def fast(*a, **k): return "fast"
        def orig(*a, **k): return "orig"
        disp = core._make_dispatcher(fast, orig)
        assert disp() == "fast"
        with autozyme.disabled():
            assert disp() == "orig"
    finally:
        core._make_dispatcher = saved_make
        IP._INSTALLED = saved_installed
        IP._ORIG_MAKE_DISPATCHER = saved_orig


# ==========================================================================
# _verify_worker — arg parse + peak rss + main() on _test_json
# ==========================================================================
def test_verify_worker_parse_args():
    from autozyme import _verify_worker as W
    ns = W._parse_args([
        "--patch", "demo", "--task-dir", "/t", "--tier", "tiny",
        "--output-dir", "/o", "--activate",
    ])
    assert ns.patch == "demo"
    assert ns.task_dir == "/t"
    assert ns.tier == "tiny"
    assert ns.output_dir == "/o"
    assert ns.activate is True


def test_verify_worker_parse_args_no_activate():
    from autozyme import _verify_worker as W
    ns = W._parse_args([
        "--patch", "demo", "--task-dir", "/t", "--tier", "tiny",
        "--output-dir", "/o",
    ])
    assert ns.activate is False


def test_verify_worker_peak_rss_type():
    from autozyme import _verify_worker as W
    v = W._peak_rss_mb()
    # POSIX (mac/linux) -> float; otherwise None
    assert v is None or (isinstance(v, float) and v > 0)


def test_verify_worker_main_baseline_and_patched(tmp_path, capsys):
    from autozyme import _verify_worker as W
    # _test_json is the stdlib-only synthetic patch
    _import_submodule("_test_json")
    out_dir = tmp_path / "out"
    rc = W.main(["--patch", "_test_json", "--task-dir", str(tmp_path),
                 "--tier", "tiny", "--output-dir", str(out_dir)])
    assert rc == 0
    captured = capsys.readouterr()
    # last non-empty stdout line is JSON with elapsed_sec
    last = [ln for ln in captured.out.splitlines() if ln.strip()][-1]
    payload = json.loads(last)
    assert "elapsed_sec" in payload and payload["elapsed_sec"] >= 0
    assert "peak_mb" in payload
    # smoke.save wrote output.txt
    assert (out_dir / "output.txt").read_text().strip() == "6"


def test_verify_worker_main_activate(tmp_path, capsys):
    from autozyme import _verify_worker as W
    _import_submodule("_test_json")
    out_dir = tmp_path / "out"
    rc = W.main(["--patch", "_test_json", "--task-dir", str(tmp_path),
                 "--tier", "tiny", "--output-dir", str(out_dir), "--activate"])
    assert rc == 0
    autozyme.deactivate("_test_json")


# ==========================================================================
# _benchmark
# ==========================================================================
def test_benchmark_summarize():
    from autozyme._benchmark import _summarize
    s = _summarize([2.0, 4.0, 6.0])
    assert s["min"] == 2.0
    assert s["max"] == 6.0
    assert s["median"] == 4.0
    assert s["mean"] == 4.0
    assert s["stdev"] > 0
    assert s["all"] == [2.0, 4.0, 6.0]


def test_benchmark_summarize_single():
    from autozyme._benchmark import _summarize
    s = _summarize([3.0])
    assert s["stdev"] == 0.0


def test_benchmark_runs_on_test_json(tmp_path):
    from autozyme._benchmark import benchmark
    _import_submodule("_test_json")
    try:
        res = benchmark("_test_json", str(tmp_path), tier="tiny", reps=2,
                        verbose=False)
        for key in ("baseline", "patched", "speedup_x", "speedup_pct"):
            assert key in res
            assert "median" in res[key]
        assert len(res["baseline"]["all"]) == 2
    finally:
        autozyme.deactivate("_test_json")


def test_benchmark_rejects_bad_reps(tmp_path):
    from autozyme._benchmark import benchmark
    _import_submodule("_test_json")
    with pytest.raises(ValueError, match="reps must be"):
        benchmark("_test_json", str(tmp_path), reps=0, verbose=False)
    with pytest.raises(ValueError, match="positive integer"):
        benchmark("_test_json", str(tmp_path), reps="x", verbose=False)


def test_benchmark_unknown_patch(tmp_path):
    from autozyme._benchmark import benchmark
    with pytest.raises((KeyError, ImportError)):
        benchmark("definitely_not_a_patch", str(tmp_path), verbose=False)


def test_benchmark_verbose_prints(tmp_path, capsys):
    from autozyme._benchmark import benchmark
    _import_submodule("_test_json")
    try:
        benchmark("_test_json", str(tmp_path), tier="tiny", reps=1, verbose=True)
    finally:
        autozyme.deactivate("_test_json")
    err = capsys.readouterr().err
    assert "benchmark" in err
    assert "summary" in err


# --------------------------------------------------------------------------
# _verify._run_worker — drives the real worker subprocess via _test_json
# --------------------------------------------------------------------------
def test_run_worker_returns_elapsed_and_peak(tmp_path):
    from autozyme._verify import _run_worker
    _import_submodule("_test_json")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    elapsed, peak = _run_worker("_test_json", str(tmp_path), "tiny",
                                str(out_dir), activate=False, verbose=False)
    assert isinstance(elapsed, float) and elapsed >= 0
    assert peak is None or isinstance(peak, float)
    assert (out_dir / "output.txt").read_text().strip() == "6"


def test_run_worker_bad_patch_raises(tmp_path):
    from autozyme._verify import _run_worker
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(RuntimeError, match="verify-worker subprocess exited"):
        _run_worker("definitely_not_a_patch_zzz", str(tmp_path), "tiny",
                    str(out_dir), activate=False, verbose=False)


# ==========================================================================
# __main__ dashboard
# ==========================================================================
def test_main_dashboard_runs(capsys):
    from autozyme import __main__ as M
    rc = M.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "autozyme" in out
    assert "patches discovered" in out
    assert "subsets" in out
    assert "Activate with" in out


def test_main_patch_line_uninstalled(monkeypatch):
    from autozyme import __main__ as M
    monkeypatch.setattr(M, "_probe_patch_installed",
                        lambda name: (False, "upstream not installed: foo"))
    ch, desc = M._patch_line("foo")
    assert ch == "x"
    assert "not installed" in desc


def test_main_patch_line_installed(monkeypatch):
    from autozyme import __main__ as M
    monkeypatch.setattr(M, "_probe_patch_installed", lambda name: (True, None))
    monkeypatch.setattr(M, "_installed_version", lambda pkg: "9.9.9")
    monkeypatch.setitem(M.UPSTREAMS, "fakepatch", ["fakepkg"])
    ch, desc = M._patch_line("fakepatch")
    assert ch == "v"
    assert "fakepkg 9.9.9" in desc


# ==========================================================================
# _core gap-filling
# ==========================================================================
def test_top_level_pkg():
    assert core._top_level_pkg("cell2location.models._x") == "cell2location"
    assert core._top_level_pkg("scanpy") == "scanpy"


def test_installed_version_known_and_unknown():
    assert core._installed_version("pytest") is not None
    assert core._installed_version("definitely_not_installed_pkg_zzz") is None


def test_base_version_strips_local_and_dev():
    assert core._base_version("0.9.6.dev5+g6c5e37d1d") == "0.9.6"
    assert core._base_version("4.0.2+4.gb847e2c") == "4.0.2"
    assert core._base_version("1.2.3") == "1.2.3"


def test_base_version_nonstandard_fallback():
    # non-PEP440 string still returns something sane (pre-"+" portion)
    assert core._base_version("weird-ver+local") == "weird-ver"


def test_did_you_mean():
    assert "scanpy" in core._did_you_mean("scnpy", ["scanpy", "scvelo"])
    assert core._did_you_mean("zzzzzz", ["scanpy"]) == ""
    multi = core._did_you_mean("scn", ["scanpy", "scvelo", "scnpy"], n=2)
    assert "one of" in multi or "Did you mean" in multi


def test_resolve_activation_target_subset_expands():
    out = core._resolve_activation_target("scrna_core")
    assert out == ["scanpy", "sccoda", "scvelo"]


def test_resolve_activation_target_dedup_preserves_order():
    out = core._resolve_activation_target(["scrna_core", "scanpy"])
    # scanpy appears once despite being in both
    assert out.count("scanpy") == 1
    assert out[0] == "scanpy"


def test_resolve_activation_target_unknown_raises():
    with pytest.raises(KeyError, match="neither a known patch"):
        core._resolve_activation_target("notapatch_xyz")


def test_resolve_activation_target_bad_type():
    with pytest.raises(TypeError, match="expects str"):
        core._resolve_activation_target(42)


def test_list_subsets_and_subset():
    subs = autozyme.list_subsets()
    assert "scrna_core" in subs
    assert autozyme.subset("scrna_core") == ["scanpy", "sccoda", "scvelo"]


def test_subset_unknown_raises():
    with pytest.raises(KeyError, match="no subset named"):
        autozyme.subset("not_a_subset")


def test_disabled_context_var():
    assert core.is_disabled() is False
    with autozyme.disabled():
        assert core.is_disabled() is True
    assert core.is_disabled() is False


def test_make_dispatcher_forwards_and_strips_zyme():
    def fast(*a, **k): return "fast"
    def orig(*a, **k): return ("orig", k)
    disp = core._make_dispatcher(fast, orig)
    assert disp() == "fast"
    assert disp.__autozyme_fast__ is fast
    assert disp.__autozyme_original__ is orig
    with autozyme.disabled():
        # zyme kwarg stripped before forwarding to original
        result, kwargs = disp(zyme=True)
        assert result == "orig"
        assert "zyme" not in kwargs


def test_register_patch_validates_targets():
    with pytest.raises(TypeError, match="must be a 3-tuple"):
        autozyme.register_patch("bad1", [("json", "dumps")])
    with pytest.raises(TypeError, match="upstream_path must be"):
        autozyme.register_patch("bad2", [("", "dumps", lambda: None)])
    with pytest.raises(TypeError, match="attr must be"):
        autozyme.register_patch("bad3", [("json", "", lambda: None)])
    with pytest.raises(TypeError, match="must be callable"):
        autozyme.register_patch("bad4", [("json", "dumps", "notcallable")])


def test_register_patch_validates_tested_upstream_versions():
    # target a json attr NOT claimed by the auto-registered _test_json patch
    try:
        with pytest.raises(TypeError, match="tested_upstream_versions must be dict"):
            autozyme.register_patch("tuv1", [("json", "loads", lambda s: s)],
                                    tested_upstream_versions=["x"])
        with pytest.raises(ValueError, match="is empty"):
            autozyme.register_patch("tuv2", [("json", "loads", lambda s: s)],
                                    tested_upstream_versions={"json": []})
        with pytest.raises(TypeError, match="must be str"):
            autozyme.register_patch("tuv3", [("json", "loads", lambda s: s)],
                                    tested_upstream_versions={"json": [1.0]})
    finally:
        for n in ("tuv1", "tuv2", "tuv3"):
            _REGISTRY.pop(n, None)


def test_register_patch_smoke_missing_keys():
    try:
        with pytest.raises(ValueError, match="smoke recipe missing"):
            autozyme.register_patch("smk", [("json", "loads", lambda s: s)],
                                    smoke={"load": lambda: None})
    finally:
        _REGISTRY.pop("smk", None)


def test_resolve_target_module_and_class():
    import email.message
    assert core._resolve_target("json") is __import__("json")
    # class via attribute walk
    assert core._resolve_target("email.message.Message") is email.message.Message


def test_resolve_target_unresolvable_raises():
    with pytest.raises(ImportError):
        core._resolve_target("no.such.module.at.all")


def test_inspect_uninstalled(monkeypatch):
    monkeypatch.setattr(core, "_probe_patch_installed",
                        lambda name: (False, "upstream not installed"))
    monkeypatch.setattr(core, "_AVAILABLE", ["ghost"])
    res = autozyme.inspect("ghost")
    assert res["status"] == "uninstalled"
    assert res["targets"] == []


def test_inspect_active_patch_via_test_json(tmp_path):
    _import_submodule("_test_json")
    autozyme.activate("_test_json")
    try:
        res = autozyme.inspect("_test_json")
        assert res["name"] == "_test_json"
        assert res["status"] == "active"
        assert len(res["targets"]) == 1
        t = res["targets"][0]
        assert t["upstream"] == "json" and t["attr"] == "dumps"
        assert t["currently_bound_to_fast"] is True
    finally:
        autozyme.deactivate("_test_json")


def test_env_snapshot_structure():
    snap = autozyme.env_snapshot()
    assert "autozyme_version" in snap
    assert "python_version" in snap
    assert "platform" in snap
    assert isinstance(snap["patches"], list)


def test_status_includes_registry_only_entries():
    autozyme.register_patch("statusonly", [("json", "loads", lambda s: s)])
    try:
        s = autozyme.status()
        assert s["statusonly"] == "inactive"
    finally:
        _REGISTRY.pop("statusonly", None)


def test_list_patches_installed_filter():
    all_patches = autozyme.list_patches()
    installed = autozyme.list_patches(installed=True)
    assert set(installed).issubset(set(all_patches))


def test_activate_disabled_env_short_circuits(monkeypatch):
    autozyme.register_patch("envoff", [("json", "loads", lambda s: s)])
    monkeypatch.setattr(core, "_AVAILABLE", core._AVAILABLE + ["envoff"])
    monkeypatch.setenv("AUTOZYME_DISABLED", "1")
    try:
        assert autozyme.activate("envoff") is False
        assert autozyme.activate(["envoff"]) == {"envoff": False}
    finally:
        _REGISTRY.pop("envoff", None)


def test_activate_one_partial_drift(monkeypatch, capsys):
    # one valid target (json.loads, not claimed by _test_json) + one whose
    # attr is missing on the upstream
    def fast_loads(s, **k): return "x"
    def fast_missing(*a, **k): return None
    autozyme.register_patch(
        "partialdrift",
        [("json", "loads", fast_loads),
         ("json", "no_such_attr_zzz", fast_missing)],
    )
    try:
        p = _REGISTRY["partialdrift"]
        assert core._activate_one(p) is True   # partial activation still True
        assert p.injected is True
        err = capsys.readouterr().err
        assert "partial activation" in err
        core._deactivate_one(p)
    finally:
        _REGISTRY.pop("partialdrift", None)


def test_activate_one_zero_targets_returns_false():
    def fast(*a, **k): return None
    autozyme.register_patch("alldrift",
                            [("json", "no_attr_a_zzz", fast)])
    try:
        p = _REGISTRY["alldrift"]
        assert core._activate_one(p) is False
        assert p.injected is False
    finally:
        _REGISTRY.pop("alldrift", None)


def test_deactivate_idempotent_unactivated():
    autozyme.register_patch("neveractive", [("json", "loads", lambda s: s)])
    try:
        # never activated -> deactivate is a silent no-op
        autozyme.deactivate("neveractive")
    finally:
        _REGISTRY.pop("neveractive", None)


def test_emit_activation_marker_quiet(monkeypatch, capsys):
    monkeypatch.setenv("AUTOZYME_QUIET", "1")
    p = core._Patch(name="q", targets=[("json", "dumps", lambda o: o)])
    core._emit_activation_marker(p)
    assert capsys.readouterr().err == ""  # silenced


def test_emit_activation_marker_drift(monkeypatch, capsys):
    monkeypatch.delenv("AUTOZYME_QUIET", raising=False)
    monkeypatch.setattr(core, "_installed_version", lambda pkg: "2.0.0")
    p = core._Patch(name="d", targets=[("json", "dumps", lambda o: o)],
                    tested_against="json 1.0.0")
    core._emit_activation_marker(p)
    err = capsys.readouterr().err
    assert "activated d" in err
    assert "WARN: lifted against json 1.0.0" in err


def test_emit_inactive_marker(monkeypatch, capsys):
    monkeypatch.delenv("AUTOZYME_QUIET", raising=False)
    core._emit_inactive_marker("ghost", "upstream not installed: foo")
    err = capsys.readouterr().err
    assert "NOT activated" in err
    assert "foo" in err
