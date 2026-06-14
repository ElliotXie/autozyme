"""End-to-end / wrapper-line tests for autozyme.sccoda.

Wave-1 (`test_sccoda_unit.py`) tested fast_get_y_hat + _scoped_num_leapfrog_steps.
The existing `test_sccoda.py` drives the FULL HMC (sample_hmc) end-to-end, which
already covers fast_sample_hmc / _build_fast_tlpf / fast_lfi_call's scoped path /
fast_get_y_hat on the real path.

This file targets the remaining COVERAGE-VISIBLE wrapper branches that the full
HMC run does NOT reliably hit, WITHOUT paying another multi-second TF-graph HMC
trace:

  - `fast_lfi_call`: the OUT-OF-SCOPE fallback (`_sccoda_leapfrog_scope` False ->
    `_call_orig_lfi`), i.e. a non-scCODA TFP HMC call delegates to upstream.
  - `fast_lfi_call`: the scoped-but-no-static-step-count fallback (n_steps None).
  - `_call_orig_lfi`: the thin upstream forwarder.
  - the contextvar scope toggles.
  - activate/restore lifecycle on the 3 sccoda targets.

NOTE (env): sccoda imports tensorflow at module import. Cross-module import
order can trip numba/TF (the xclim Bug-5 / TF-TBB deadlock). This file is kept
TF-light (no graph compile / HMC) and is safe to run alongside the others, but
if a numba/TF clash appears, run it alone -- see report.
"""
from __future__ import annotations

import os

import pytest

# Quiet TF before sccoda imports it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

np = pytest.importorskip("numpy")
pytest.importorskip("sccoda")
pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_probability")

import autozyme
from autozyme import sccoda as azsccoda


def test_lfi_call_out_of_scope_delegates_to_upstream(monkeypatch):
    """Outside the sccoda leapfrog scope, fast_lfi_call forwards verbatim to the
    captured upstream SimpleLeapfrogIntegrator.__call__ (a non-scCODA TFP HMC
    call must be untouched)."""
    autozyme.activate("sccoda")
    sentinel = object()
    seen = {}

    def _fake_orig(self, momentum_parts, state_parts, target=None,
                   target_grad_parts=None, kinetic_energy_fn=None, name=None):
        seen["called"] = True
        seen["state"] = state_parts
        return sentinel

    monkeypatch.setattr(azsccoda, "_orig_lfi_call", _fake_orig)
    # Default scope is False (not inside a scCODA sample_hmc).
    assert azsccoda._sccoda_leapfrog_scope.get() is False
    out = azsccoda.fast_lfi_call(
        self=object(), momentum_parts=[1.0], state_parts=["state"],
    )
    assert out is sentinel
    assert seen["called"] and seen["state"] == ["state"]


def test_call_orig_lfi_forwarder(monkeypatch):
    """_call_orig_lfi is a thin forwarder to the captured upstream __call__."""
    autozyme.activate("sccoda")
    seen = {}

    def _fake_orig(self, momentum_parts, state_parts, target=None,
                   target_grad_parts=None, kinetic_energy_fn=None, name=None):
        seen["args"] = (momentum_parts, state_parts, target, name)
        return "ORIG"

    monkeypatch.setattr(azsccoda, "_orig_lfi_call", _fake_orig)
    out = azsccoda._call_orig_lfi(
        self=object(), momentum_parts=[2.0], state_parts=["s"],
        target="t", name="nm",
    )
    assert out == "ORIG"
    assert seen["args"] == ([2.0], ["s"], "t", "nm")


def test_lfi_call_scoped_without_static_steps_falls_back(monkeypatch):
    """Inside the scope but with no resolvable static step count AND no scoped
    steps set, fast_lfi_call still delegates to upstream (n_steps None)."""
    autozyme.activate("sccoda")
    seen = {}

    def _fake_orig(self, momentum_parts, state_parts, target=None,
                   target_grad_parts=None, kinetic_energy_fn=None, name=None):
        seen["called"] = True
        return "FALLBACK"

    monkeypatch.setattr(azsccoda, "_orig_lfi_call", _fake_orig)

    # A fake integrator whose _num_steps has no static value and no .numpy().
    class _FakeIntegrator:
        _num_steps = object()  # tf.get_static_value(...) -> None, no .numpy()

    scope_tok = azsccoda._sccoda_leapfrog_scope.set(True)
    steps_tok = azsccoda._sccoda_leapfrog_steps.set(None)
    try:
        out = azsccoda.fast_lfi_call(
            self=_FakeIntegrator(), momentum_parts=[1.0], state_parts=["s"],
        )
    finally:
        azsccoda._sccoda_leapfrog_scope.reset(scope_tok)
        azsccoda._sccoda_leapfrog_steps.reset(steps_tok)
    assert out == "FALLBACK"
    assert seen["called"]


def test_scoped_num_leapfrog_steps_branches():
    """_scoped_num_leapfrog_steps: None default, int parse, garbage -> None."""
    tok = azsccoda._sccoda_leapfrog_steps.set(None)
    try:
        assert azsccoda._scoped_num_leapfrog_steps() is None
    finally:
        azsccoda._sccoda_leapfrog_steps.reset(tok)

    tok = azsccoda._sccoda_leapfrog_steps.set("10")
    try:
        assert azsccoda._scoped_num_leapfrog_steps() == 10
    finally:
        azsccoda._sccoda_leapfrog_steps.reset(tok)

    tok = azsccoda._sccoda_leapfrog_steps.set("not-an-int")
    try:
        assert azsccoda._scoped_num_leapfrog_steps() is None
    finally:
        azsccoda._sccoda_leapfrog_steps.reset(tok)


def test_activate_restore_lifecycle():
    """activate binds the 3 sccoda targets; deactivate restores them."""
    from tensorflow_probability.python.mcmc.internal import leapfrog_integrator as lfi
    import sccoda.model.scCODA_model as scm

    autozyme.deactivate("sccoda")
    orig_lfi = lfi.SimpleLeapfrogIntegrator.__call__
    orig_y_hat = scm.scCODAModel.get_y_hat
    orig_hmc = scm.scCODAModel.sample_hmc

    assert autozyme.activate("sccoda") is True
    assert lfi.SimpleLeapfrogIntegrator.__call__ is not orig_lfi
    info = autozyme.inspect("sccoda")
    assert info["status"] == "active"
    assert len(info["targets"]) == 3

    autozyme.deactivate("sccoda")
    assert lfi.SimpleLeapfrogIntegrator.__call__ is orig_lfi
    assert scm.scCODAModel.get_y_hat is orig_y_hat
    assert scm.scCODAModel.sample_hmc is orig_hmc
