"""Smoke tests using `json` (stdlib) as a synthetic upstream."""
from __future__ import annotations

import json

import autozyme
from autozyme._core import _REGISTRY


def test_status_callable_on_empty_registry():
    s = autozyme.status()
    assert isinstance(s, dict)


def test_register_inject_restore_roundtrip():
    original_dumps = json.dumps

    def fast_dumps(obj, **kwargs):
        kwargs.setdefault("separators", (",", ":"))
        return original_dumps(obj, **kwargs)

    autozyme.register_patch(
        name="json_demo",
        targets=[("json", "dumps", fast_dumps)],
    )
    try:
        assert autozyme.activate("json_demo") is True
        assert autozyme.status()["json_demo"] == "active"
        assert json.dumps({"a": 1, "b": 2}) == '{"a":1,"b":2}'

        autozyme.deactivate("json_demo")
        assert autozyme.status()["json_demo"] == "inactive"
        assert json.dumps({"a": 1, "b": 2}) == '{"a": 1, "b": 2}'
    finally:
        _REGISTRY.pop("json_demo", None)


def test_register_resolves_class_via_attribute_walk():
    """Class-method patching: upstream path resolves to a class, not a module."""
    import email.message
    original_get = email.message.Message.get

    def fast_get(self, name, failobj=None):
        return f"patched:{name}"

    autozyme.register_patch(
        name="email_demo",
        targets=[("email.message.Message", "get", fast_get)],
    )
    try:
        assert autozyme.activate("email_demo") is True
        m = email.message.Message()
        m["Subject"] = "x"
        assert m.get("Subject") == "patched:Subject"

        autozyme.deactivate("email_demo")
        assert m.get("Subject") == "x"
    finally:
        email.message.Message.get = original_get
        _REGISTRY.pop("email_demo", None)


def test_set_threads_writes_env():
    import os
    autozyme.set_threads(3)
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "3"


def test_unknown_patch_raises():
    import pytest
    with pytest.raises(KeyError):
        autozyme.deactivate("does_not_exist")
    with pytest.raises(KeyError):
        autozyme.activate("does_not_exist")


def test_missing_upstream_short_circuits_before_patch_import(monkeypatch):
    """A user may install autozyme without every heavy upstream.

    activate(name) should then return False from the cheap upstream probe,
    not import the patch module and crash on transitive deps like numpy/torch.
    """
    import autozyme._core as core

    monkeypatch.setattr(core, "_AVAILABLE", ["missing_demo"])
    monkeypatch.setattr(
        core,
        "_probe_patch_installed",
        lambda name: (False, "upstream not installed: missing_demo"),
    )

    def explode(name):
        raise AssertionError("_import_submodule should not run")

    monkeypatch.setattr(core, "_import_submodule", explode)

    assert autozyme.activate("missing_demo") is False
    assert autozyme.activate(["missing_demo"]) == {"missing_demo": False}


def test_register_rejects_conflicting_target():
    import pytest

    autozyme.register_patch("conflict_a", [("json", "dumps", lambda obj: obj)])
    try:
        with pytest.raises(ValueError, match="already claimed"):
            autozyme.register_patch("conflict_b", [("json", "dumps", lambda obj: obj)])
    finally:
        _REGISTRY.pop("conflict_a", None)
        _REGISTRY.pop("conflict_b", None)


def test_register_allows_disjoint_targets_on_same_upstream():
    autozyme.register_patch("a_dumps", [("json", "dumps", lambda obj: obj)])
    autozyme.register_patch("a_loads", [("json", "loads", lambda s: s)])
    try:
        names = set(autozyme.status().keys())
        assert {"a_dumps", "a_loads"}.issubset(names)
    finally:
        _REGISTRY.pop("a_dumps", None)
        _REGISTRY.pop("a_loads", None)


def test_strict_upstream_version_refuses_activation(capsys):
    original_dumps = json.dumps

    def fast_dumps(obj, **kwargs):
        return "patched"

    autozyme.register_patch(
        "strict_version_demo",
        [("json", "dumps", fast_dumps)],
        tested_upstream_versions={"pip": ["0.0"]},
        strict_upstream_versions=True,
    )
    try:
        assert autozyme.activate("strict_version_demo") is False
        assert json.dumps is original_dumps
        stderr = capsys.readouterr().err
        assert "strict_version_demo NOT activated" in stderr
        assert "requires pip==0.0" in stderr
    finally:
        json.dumps = original_dumps
        _REGISTRY.pop("strict_version_demo", None)
