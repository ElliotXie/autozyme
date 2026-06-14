"""Deep tests for autozyme._core gaps not reached by wave-1 tests.

test_core.py + test_framework_misc_unit.py (wave 1) cover the happy path of
register/activate/deactivate/inspect/env_snapshot plus did-you-mean and the
disabled() dispatcher. This file fills the remaining branch gaps:

  * _resolve_target attribute-walk that finds the head module but the tail
    attribute is missing (AttributeError continue, 128-131),
  * _probe_patch_installed deriving upstreams from a registered patch's
    targets and the find_spec ValueError path (276-296),
  * _import_submodule's "imported but didn't register" RuntimeError (248),
  * _emit_activation_marker: malformed tested_against, multi-version known set,
    AUTOZYME_QUIET, no-drift same-version (351-368),
  * _activate_one partial-drift recommendation lines for both
    tested_upstream_versions (single + multi) and tested_against (426-440),
  * _check_conflicts RuntimeWarning (510-519),
  * activate() list path swallowing ImportError per-name (562-565),
  * inspect() single-name guard + bound-target view,
  * env_snapshot() uninstalled + registered-patch entries.

All synthetic, stdlib-only (json / email targets). No heavy upstream needed.
"""
from __future__ import annotations

import warnings

import pytest

import autozyme
from autozyme import _core as core
from autozyme._core import _REGISTRY


# ==========================================================================
# _resolve_target — attribute-walk where the tail attr is missing
# ==========================================================================
def test_resolve_target_head_module_missing_tail_attr():
    """json is importable but json.no_such_attr does not exist -> the walk
    hits AttributeError and ultimately raises ImportError (128-136)."""
    with pytest.raises(ImportError):
        core._resolve_target("json.no_such_attr_zzz")


def test_resolve_target_deep_attr_walk_success():
    import email.message
    # email.message importable; .Message.get is a real method chain
    obj = core._resolve_target("email.message.Message")
    assert obj is email.message.Message


# ==========================================================================
# _probe_patch_installed — derive upstreams from targets / fallback to name
# ==========================================================================
def test_probe_installed_derives_upstreams_from_targets():
    """A registered patch NOT in UPSTREAMS derives its upstream package set
    from its target paths (276-277). json has a find_spec -> installed True."""
    core._PROBE_CACHE.pop("probe_derive", None)
    autozyme.register_patch("probe_derive", [("json", "loads", lambda s: s)])
    try:
        installed, err = core._probe_patch_installed("probe_derive")
        assert installed is True and err is None
    finally:
        _REGISTRY.pop("probe_derive", None)
        core._PROBE_CACHE.pop("probe_derive", None)


def test_probe_unregistered_unknown_name_falls_back_to_name():
    """An unknown name (not in UPSTREAMS, not registered) assumes a same-named
    upstream package (278) -> missing -> (False, msg)."""
    core._PROBE_CACHE.pop("nonexistent_pkg_zzz", None)
    installed, err = core._probe_patch_installed("nonexistent_pkg_zzz")
    assert installed is False
    assert "not installed" in err
    core._PROBE_CACHE.pop("nonexistent_pkg_zzz", None)


def test_probe_find_spec_value_error_is_missing(monkeypatch):
    """find_spec raising ValueError (e.g. a namespace edge) is treated as a
    missing upstream (283-285)."""
    import importlib.util
    core._PROBE_CACHE.pop("probe_valueerr", None)
    autozyme.register_patch("probe_valueerr",
                            [("weirdpkg_zzz", "fn", lambda: None)])

    def boom(pkg):
        raise ValueError("bad spec")

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    try:
        installed, err = core._probe_patch_installed("probe_valueerr")
        assert installed is False and "not installed" in err
    finally:
        _REGISTRY.pop("probe_valueerr", None)
        core._PROBE_CACHE.pop("probe_valueerr", None)


def test_probe_result_is_cached():
    core._PROBE_CACHE.pop("nonexistent_cache_zzz", None)
    r1 = core._probe_patch_installed("nonexistent_cache_zzz")
    assert "nonexistent_cache_zzz" in core._PROBE_CACHE
    r2 = core._probe_patch_installed("nonexistent_cache_zzz")
    assert r1 == r2
    core._PROBE_CACHE.pop("nonexistent_cache_zzz", None)


# ==========================================================================
# _import_submodule — registered-on-import contract failure (244-251)
# ==========================================================================
def test_import_submodule_unknown_raises_importerror():
    with pytest.raises(ImportError, match="could not load autozyme"):
        core._import_submodule("definitely_not_a_submodule_zzz")


def test_import_submodule_imported_but_not_registered(monkeypatch):
    """If the submodule imports cleanly but never calls register_patch, we get
    a RuntimeError (247-251)."""
    import importlib

    monkeypatch.setattr(importlib, "import_module", lambda mod: object())
    with pytest.raises(RuntimeError, match="did not register a patch"):
        core._import_submodule("ghostmod_zzz")


# ==========================================================================
# _emit_activation_marker — drift / no-drift / malformed tested_against
# ==========================================================================
def test_emit_marker_malformed_tested_against(monkeypatch, capsys):
    """tested_against without a space -> rsplit ValueError -> no drift section
    but still prints the activation line (350-352)."""
    monkeypatch.delenv("AUTOZYME_QUIET", raising=False)
    monkeypatch.setattr(core, "_installed_version", lambda pkg: "1.0.0")
    p = core._Patch(name="m1", targets=[("json", "dumps", lambda o: o)],
                    tested_against="nospaceversion")
    core._emit_activation_marker(p)
    err = capsys.readouterr().err
    assert "activated m1" in err
    assert "WARN" not in err  # malformed -> no drift warning


def test_emit_marker_no_drift_same_version(monkeypatch, capsys):
    """Installed version matches tested_against -> no WARN (359-363 not hit)."""
    monkeypatch.delenv("AUTOZYME_QUIET", raising=False)
    monkeypatch.setattr(core, "_installed_version", lambda pkg: "1.0.0")
    p = core._Patch(name="m2", targets=[("json", "dumps", lambda o: o)],
                    tested_against="json 1.0.0")
    core._emit_activation_marker(p)
    err = capsys.readouterr().err
    assert "activated m2" in err
    assert "WARN" not in err


def test_emit_marker_known_versions_from_tested_upstream(monkeypatch, capsys):
    """tested_upstream_versions widens the known-good set so an installed
    version listed there is NOT flagged as drift (356-358)."""
    monkeypatch.delenv("AUTOZYME_QUIET", raising=False)
    monkeypatch.setattr(core, "_installed_version", lambda pkg: "2.5.0")
    p = core._Patch(
        name="m3", targets=[("json", "dumps", lambda o: o)],
        tested_against="json 1.0.0",
        tested_upstream_versions={"json": ["2.5.0", "2.6.0"]},
    )
    core._emit_activation_marker(p)
    err = capsys.readouterr().err
    assert "activated m3" in err
    assert "WARN" not in err  # 2.5.0 is in the known set


def test_emit_marker_version_unknown(monkeypatch, capsys):
    """When _installed_version returns None the marker prints '(version
    unknown)' for that pkg (338-342)."""
    monkeypatch.delenv("AUTOZYME_QUIET", raising=False)
    monkeypatch.setattr(core, "_installed_version", lambda pkg: None)
    p = core._Patch(name="m4", targets=[("json", "dumps", lambda o: o)])
    core._emit_activation_marker(p)
    err = capsys.readouterr().err
    assert "version unknown" in err


# ==========================================================================
# _activate_one — partial-drift recommendation lines
# ==========================================================================
def test_activate_one_partial_drift_tested_upstream_single(capsys):
    """Partial activation + tested_upstream_versions with one version -> the
    'for full speedup install: pkg==X' line (426-434)."""
    autozyme.register_patch(
        "pd_single",
        [("json", "loads", lambda s, **k: s),
         ("json", "no_attr_zzz", lambda *a, **k: None)],
        tested_upstream_versions={"json": ["9.9.9"]},
    )
    try:
        p = _REGISTRY["pd_single"]
        assert core._activate_one(p) is True
        err = capsys.readouterr().err
        assert "partial activation" in err
        assert "json==9.9.9" in err
        core._deactivate_one(p)
    finally:
        _REGISTRY.pop("pd_single", None)


def test_activate_one_partial_drift_tested_upstream_multi(capsys):
    """Multi-version tested_upstream_versions -> 'pkg in {a, b}' form (427-430)."""
    autozyme.register_patch(
        "pd_multi",
        [("json", "loads", lambda s, **k: s),
         ("json", "no_attr_zzz", lambda *a, **k: None)],
        tested_upstream_versions={"json": ["1.0.0", "2.0.0"]},
    )
    try:
        p = _REGISTRY["pd_multi"]
        assert core._activate_one(p) is True
        err = capsys.readouterr().err
        assert "partial activation" in err
        assert "json in {1.0.0, 2.0.0}" in err
        core._deactivate_one(p)
    finally:
        _REGISTRY.pop("pd_multi", None)


def test_activate_one_partial_drift_tested_against_only(capsys):
    """Partial activation with only tested_against (no structured versions) ->
    the 'lifted against X -- install that exact upstream' line (435-439)."""
    autozyme.register_patch(
        "pd_ta",
        [("json", "loads", lambda s, **k: s),
         ("json", "no_attr_zzz", lambda *a, **k: None)],
        tested_against="json 3.1.4",
    )
    try:
        p = _REGISTRY["pd_ta"]
        assert core._activate_one(p) is True
        err = capsys.readouterr().err
        assert "lifted against json 3.1.4" in err
        core._deactivate_one(p)
    finally:
        _REGISTRY.pop("pd_ta", None)


def test_activate_one_already_injected_returns_true():
    """Re-activating an injected patch short-circuits to True (387-388)."""
    autozyme.register_patch("already_inj", [("json", "loads", lambda s, **k: s)])
    try:
        p = _REGISTRY["already_inj"]
        assert core._activate_one(p) is True
        assert p.injected is True
        # second call: already injected -> True without re-binding
        assert core._activate_one(p) is True
        core._deactivate_one(p)
    finally:
        _REGISTRY.pop("already_inj", None)


def test_deactivate_one_resolve_failure_continues(monkeypatch):
    """If _resolve_target raises mid-deactivate, that target is skipped and the
    rest still restore (451-452).

    NOTE: forcing the restore to fail leaves ``json.loads`` bound to the fast
    lambda; we capture + restore it by hand in ``finally`` so the leaked
    binding can't poison json.loads() in later tests/files.
    """
    import json as _json
    real_loads = _json.loads
    autozyme.register_patch("deact_fail", [("json", "loads", lambda s, **k: s)])
    try:
        p = _REGISTRY["deact_fail"]
        core._activate_one(p)
        # force _resolve_target to raise during deactivate
        monkeypatch.setattr(
            core, "_resolve_target",
            lambda u: (_ for _ in ()).throw(ImportError("gone")))
        core._deactivate_one(p)  # must not raise
        assert p.injected is False
        # the forced failure means json.loads was NOT restored by the framework
        assert _json.loads is not real_loads
    finally:
        _json.loads = real_loads  # undo the leaked binding by hand
        _REGISTRY.pop("deact_fail", None)


# ==========================================================================
# _check_conflicts — RuntimeWarning when a CONFLICTS pair becomes live
# ==========================================================================
def test_check_conflicts_warns(monkeypatch):
    """Inject a synthetic conflict pair and confirm activate-time warning."""
    fake_pair = (frozenset({"cc_a", "cc_b"}), "they fight")
    monkeypatch.setattr(core, "CONFLICTS", [fake_pair])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        core._check_conflicts(["cc_a", "cc_b"])
    assert any("interact badly" in str(w.message) for w in rec)


def test_check_conflicts_no_warn_when_disjoint(monkeypatch):
    fake_pair = (frozenset({"cc_x", "cc_y"}), "nope")
    monkeypatch.setattr(core, "CONFLICTS", [fake_pair])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        core._check_conflicts(["cc_x"])  # only one of the pair
    assert not rec


# ==========================================================================
# activate() list/subset path — swallows ImportError per-name (562-565)
# ==========================================================================
def test_activate_list_swallows_import_error(monkeypatch):
    """activate(list) catches ImportError per-name and records False, rather
    than aborting the whole batch."""
    monkeypatch.setattr(core, "_AVAILABLE", core._AVAILABLE + ["boomname"])
    monkeypatch.setattr(core, "_probe_patch_installed",
                        lambda n: (True, None))

    def boom(n):
        raise ImportError("kaboom")

    monkeypatch.setattr(core, "_import_submodule", boom)
    # pass a list (not a single str) so we take the multi-name branch
    out = autozyme.activate(["boomname"])
    assert out == {"boomname": False}


def test_activate_list_mixed_installed(monkeypatch):
    """activate(list) with one uninstalled name records False + emits the
    inactive marker (558-561)."""
    monkeypatch.setattr(core, "_AVAILABLE", core._AVAILABLE + ["uninst_zzz"])

    def probe(n):
        return (False, "upstream not installed: uninst_zzz")

    monkeypatch.setattr(core, "_probe_patch_installed", probe)
    out = autozyme.activate(["uninst_zzz"])
    assert out == {"uninst_zzz": False}


# ==========================================================================
# inspect() — multi-name guard + bound-target view
# ==========================================================================
def test_inspect_multi_name_raises():
    with pytest.raises(ValueError, match="single patch name"):
        autozyme.inspect("scrna_core")  # a subset -> resolves to many


def test_inspect_inactive_target_not_bound():
    """inspect on a registered-but-inactive patch: currently_bound_to_fast is
    False and original is None (no activation captured)."""
    autozyme.register_patch("insp_inactive",
                            [("json", "loads", lambda s, **k: s)])
    try:
        core._AVAILABLE.append("insp_inactive")
        res = autozyme.inspect("insp_inactive")
        assert res["status"] == "inactive"
        t = res["targets"][0]
        assert t["currently_bound_to_fast"] is False
        assert t["original"] is None
    finally:
        if "insp_inactive" in core._AVAILABLE:
            core._AVAILABLE.remove("insp_inactive")
        _REGISTRY.pop("insp_inactive", None)


# ==========================================================================
# env_snapshot() — uninstalled vs registered entries
# ==========================================================================
def test_env_snapshot_uninstalled_entry(monkeypatch):
    """An available patch whose upstream is missing -> 'uninstalled' entry
    (703-705)."""
    monkeypatch.setattr(core, "_AVAILABLE", ["snap_missing"])
    monkeypatch.setattr(core, "_probe_patch_installed",
                        lambda n: (False, "upstream not installed: foo"))
    snap = autozyme.env_snapshot()
    names = {p["name"]: p for p in snap["patches"]}
    assert names["snap_missing"]["status"] == "uninstalled"
    assert "error" in names["snap_missing"]


def test_env_snapshot_registered_entry(monkeypatch):
    """A registered (installed) patch contributes status + tested_against
    (713-716)."""
    autozyme.register_patch("snap_reg", [("json", "loads", lambda s, **k: s)],
                            tested_against="json 1.2.3")
    try:
        monkeypatch.setattr(core, "_AVAILABLE", ["snap_reg"])
        monkeypatch.setattr(core, "_probe_patch_installed",
                            lambda n: (True, None))
        snap = autozyme.env_snapshot()
        entry = {p["name"]: p for p in snap["patches"]}["snap_reg"]
        assert entry["status"] == "inactive"
        assert entry["tested_against"] == "json 1.2.3"
        assert "installed_versions" in entry
    finally:
        _REGISTRY.pop("snap_reg", None)


def test_env_snapshot_available_not_registered(monkeypatch):
    """An installed but not-yet-imported patch: status inactive, tested_against
    None (717-719)."""
    monkeypatch.setattr(core, "_AVAILABLE", ["snap_lazy"])
    monkeypatch.setattr(core, "_probe_patch_installed", lambda n: (True, None))
    snap = autozyme.env_snapshot()
    entry = {p["name"]: p for p in snap["patches"]}["snap_lazy"]
    assert entry["status"] == "inactive"
    assert entry["tested_against"] is None
