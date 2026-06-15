"""Universal activation smoke for every patch shipped in autozyme.

Goal: minimum-viable coverage for every patch — does ``import + activate +
deactivate`` round-trip without crashing? This is the floor below the
per-API contract tests (test_normalize_total.py etc.); a patch that
fails THIS test is broken at registration / import / namespace-binding
level, which is the cheapest class of bug to catch.

Skips cleanly when a patch's upstream isn't installed (the common case
on a vanilla CI runner without 30 scientific Python deps). The result
is: full pass = no patch has shipped with an import-time / registration
crash; skips merely report what wasn't reachable. Whoever installs an
upstream gets the test for free on the next CI run.

For deep per-API contracts (kwarg permutations, copy=True semantics,
delegation gates), see the sibling ``test_<patch_name>.py`` files. New
patches: ship at least an activation smoke (here) and ideally a per-API
contract file too — see test_normalize_total.py as the canonical
template.
"""
from __future__ import annotations

import pytest


def _list_patches():
    """Pull the shipped-patch list from autozyme. Defers the import so
    pytest collection doesn't crash if autozyme itself fails to import."""
    try:
        import autozyme
    except ImportError as e:  # pragma: no cover -- collection-time guard
        pytest.skip(f"autozyme not importable: {e}", allow_module_level=True)
    return autozyme.list_patches()


# Module-level expansion: each shipped patch becomes its own test ID, so
# a failure points directly at the offending patch by name rather than
# burying it in a generic loop.
_ALL_PATCHES = _list_patches()


def _is_upstream_installed(name: str) -> tuple[bool, str | None]:
    from autozyme._core import _probe_patch_installed
    return _probe_patch_installed(name)


def _strict_version_incompatibility(name: str) -> str | None:
    from autozyme._core import _REGISTRY, _import_submodule, _strict_version_error

    _import_submodule(name)
    return _strict_version_error(_REGISTRY[name])


@pytest.fixture(autouse=True)
def _restore_all_between_tests():
    """Each activation test runs with a clean slate.

    Conflicts (two patches targeting the same upstream symbol) raise
    inside activate(); restoring everything in between guarantees order-
    independence so test ordering doesn't determine pass/fail.
    """
    yield
    import autozyme
    try:
        autozyme.deactivate_all()
    except Exception:
        # Don't mask the actual test failure if teardown also fails;
        # just swallow so the original assertion shows up.
        pass


def test_list_patches_nonempty():
    """Sanity: autozyme must ship at least one patch."""
    assert len(_ALL_PATCHES) > 0


@pytest.mark.parametrize("patch_name", _ALL_PATCHES)
def test_patch_activate_smoke(patch_name):
    """``activate(name)`` must not crash, regardless of upstream presence."""
    import autozyme

    installed, err = _is_upstream_installed(patch_name)
    if not installed:
        pytest.skip(f"upstream missing for {patch_name!r}: {err}")
    strict_err = _strict_version_incompatibility(patch_name)
    if strict_err:
        pytest.skip(strict_err)

    result = autozyme.activate(patch_name)
    assert result is True, (
        f"activate({patch_name!r}) returned {result!r} despite upstream "
        f"being detected as installed — registration / binding gap?"
    )
    assert autozyme.status().get(patch_name) == "active", (
        f"after activate({patch_name!r}), status reports "
        f"{autozyme.status().get(patch_name)!r} (expected 'active')"
    )


@pytest.mark.parametrize("patch_name", _ALL_PATCHES)
def test_patch_activate_restore_roundtrip(patch_name):
    """``activate -> deactivate`` round-trip leaves status at 'inactive'."""
    import autozyme

    installed, err = _is_upstream_installed(patch_name)
    if not installed:
        pytest.skip(f"upstream missing for {patch_name!r}: {err}")
    strict_err = _strict_version_incompatibility(patch_name)
    if strict_err:
        pytest.skip(strict_err)

    autozyme.activate(patch_name)
    autozyme.deactivate(patch_name)
    assert autozyme.status().get(patch_name) == "inactive", (
        f"after deactivate({patch_name!r}), status still reports "
        f"{autozyme.status().get(patch_name)!r} (expected 'inactive')"
    )


@pytest.mark.parametrize("patch_name", _ALL_PATCHES)
def test_patch_all_targets_bound(patch_name):
    """Every declared target must actually bind to its fast fn — not just the
    patch reporting 'active'.

    activate() returns True (and status -> 'active') if *any* target binds, so a
    patch can look active while one of its N targets silently failed to bind
    (e.g. upstream renamed an internal). That ships a silent no-op for that op —
    exactly the "looks accelerated but isn't" failure class. inspect() reports
    per-target binding, so assert ALL targets are bound.
    """
    import autozyme

    installed, err = _is_upstream_installed(patch_name)
    if not installed:
        pytest.skip(f"upstream missing for {patch_name!r}: {err}")
    strict_err = _strict_version_incompatibility(patch_name)
    if strict_err:
        pytest.skip(strict_err)

    autozyme.activate(patch_name)
    targets = autozyme.inspect(patch_name).get("targets", [])
    assert targets, f"{patch_name!r}: inspect() reports no targets"
    unbound = [t for t in targets if not t.get("currently_bound_to_fast")]
    assert not unbound, (
        f"{patch_name!r}: {len(unbound)}/{len(targets)} target(s) NOT bound to "
        f"the fast fn (silent partial no-op): "
        + ", ".join(f"{t.get('upstream')}.{t.get('attr')}" for t in unbound)
    )


def test_registered_patch_with_missing_upstream_returns_false():
    """When upstream is absent, activate(name) must return False, not crash.

    This is the contract that lets users `import autozyme` on minimal envs
    without paying the cost of every scientific package being installed.
    Patches that crash here (instead of returning False) break the lazy-
    install promise documented in autozyme.__init__.

    Use a synthetic registered patch instead of parametrizing over the shipped
    patches. On the release verification host every real upstream may be
    installed, and a negative environment test should still run without
    producing skip noise.
    """
    import autozyme
    from autozyme import _core

    patch_name = "missing_upstream_probe"

    def _fast_fn(*_args, **_kwargs):  # pragma: no cover - never called
        raise AssertionError("missing-upstream patch should not activate")

    try:
        autozyme.register_patch(
            patch_name,
            [("autozyme_definitely_missing_upstream_for_test", "noop", _fast_fn)],
        )
        _core._PROBE_CACHE.pop(patch_name, None)

        installed, err = _is_upstream_installed(patch_name)
        assert installed is False
        assert "upstream not installed" in (err or "")

        # Don't crash — return False cleanly.
        result = autozyme.activate(patch_name)
        assert result is False, (
            f"activate({patch_name!r}) with missing upstream returned "
            f"{result!r} (expected False)"
        )
    finally:
        try:
            autozyme.deactivate(patch_name)
        except Exception:
            pass
        _core._REGISTRY.pop(patch_name, None)
        _core._PROBE_CACHE.pop(patch_name, None)
