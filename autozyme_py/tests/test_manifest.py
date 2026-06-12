"""Manifest consistency tests for the patch registry.

Catches the "I added a patch and forgot to wire it up" class of bug
without running any patch's actual fast functions:

  1. Every submodule registers under its own name (no register_patch
     name= typo vs the directory name).
  2. UPSTREAMS in _subsets.py and the submodule list are in sync.
  3. For every patch whose upstream IS installed, activate() binds every
     declared target to a fast fn (catches typos in dotted paths).

Patches whose upstream isn't installed are skipped per-patch, so the
file passes on any subset of installed upstreams (CI matrix friendly).
"""
from __future__ import annotations

import pytest

import autozyme
from autozyme._core import (
    _AVAILABLE,
    _REGISTRY,
    _import_submodule,
    _probe_patch_installed,
)
from autozyme._subsets import UPSTREAMS


def _is_installed(name: str) -> bool:
    return _probe_patch_installed(name)[0]


@pytest.mark.parametrize("name", _AVAILABLE)
def test_submodule_registers_under_its_own_name(name):
    """register_patch(name=...) inside autozyme/<name>/__init__.py must
    equal the directory name. Without this, _import_submodule succeeds
    but _REGISTRY[name] is missing — and the next activate(name) raises.
    """
    if not _is_installed(name):
        pytest.skip(f"upstream for {name!r} not installed")
    try:
        _import_submodule(name)
    except ImportError as e:
        # find_spec-based probe says "installed" but the actual import
        # chain fails (e.g. ABI mismatch in a transitive C extension).
        # That's an upstream env issue, not a patch bug — skip cleanly.
        pytest.skip(f"upstream for {name!r} importable per probe but "
                    f"fails at import: {e}")
    assert name in _REGISTRY, (
        f"autozyme.{name} imported but did not register a patch named "
        f"{name!r}. Check register_patch(name=...) inside "
        f"autozyme/{name}/__init__.py."
    )


def test_upstreams_manifest_matches_submodules():
    """UPSTREAMS and the submodule list must agree. Mismatches break
    list_patches(installed=True) and env_snapshot(), which probe upstream
    availability via UPSTREAMS without sourcing patches.
    """
    available = set(_AVAILABLE)
    manifest = set(UPSTREAMS.keys())

    missing = available - manifest
    assert not missing, (
        f"submodules with no UPSTREAMS entry: {sorted(missing)}. "
        f"Add to autozyme/_subsets.py::UPSTREAMS."
    )

    # Allow underscore-prefixed manifest entries — those are test-only
    # synthetic patches (e.g. _test_json) that intentionally live outside
    # _AVAILABLE (which skips _-prefixed names in _populate_available).
    orphans = {n for n in manifest if not n.startswith("_")} - available
    assert not orphans, (
        f"UPSTREAMS entries with no submodule: {sorted(orphans)}. "
        f"Remove from autozyme/_subsets.py::UPSTREAMS."
    )


@pytest.mark.parametrize("name", _AVAILABLE)
def test_tested_upstream_versions_schema(name):
    """If a plugin declares `tested_upstream_versions=`, each version string
    must be parseable by `packaging.version.Version`. The structural shape
    (dict[str, list[str]]) is already enforced by `register_patch` itself.

    Plugins are free to omit the field — they just won't appear in Tier C's
    drift-detection matrix.
    """
    packaging_version = pytest.importorskip("packaging.version").Version

    if not _is_installed(name):
        pytest.skip(f"upstream for {name!r} not installed")
    try:
        _import_submodule(name)
    except ImportError as e:
        pytest.skip(f"upstream for {name!r} importable per probe but "
                    f"fails at import: {e}")
    patch = _REGISTRY[name]
    versions = patch.tested_upstream_versions
    if versions is None:
        pytest.skip(f"{name!r} does not declare tested_upstream_versions")
    for pkg, vers in versions.items():
        for v in vers:
            try:
                packaging_version(v)
            except Exception as e:
                pytest.fail(
                    f"{name!r} tested_upstream_versions[{pkg!r}] entry "
                    f"{v!r} is not a valid PEP 440 version: {e}"
                )


@pytest.mark.parametrize("name", _AVAILABLE)
def test_activate_binds_every_declared_target(name):
    """Activate and confirm every declared target's dotted path resolved
    and got bound to the fast fn. Catches wrong upstream paths
    (typo'd module / renamed class) that _activate_one otherwise handles
    by silently returning False.
    """
    if not _is_installed(name):
        pytest.skip(f"upstream for {name!r} not installed")
    try:
        ok = autozyme.activate(name)
    except ImportError as e:
        pytest.skip(f"upstream for {name!r} importable per probe but "
                    f"fails at import: {e}")
    try:
        assert ok is True, (
            f"activate({name!r}) returned False even though the upstream "
            f"probe reports installed — likely a wrong dotted path in "
            f"register_patch(targets=...)."
        )
        info = autozyme.inspect(name)
        unbound = [
            f"{t['upstream']}::{t['attr']}"
            for t in info["targets"]
            if not t["currently_bound_to_fast"]
        ]
        assert not unbound, (
            f"{name}: targets not bound after activate(): {unbound}. "
            f"Check dotted paths in register_patch(targets=...)."
        )
    finally:
        autozyme.deactivate(name)
