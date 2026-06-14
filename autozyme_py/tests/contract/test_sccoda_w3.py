"""Wave-3 heavy-path tests for autozyme.sccoda.

Wave-1 (`test_sccoda_unit.py`) tested fast_get_y_hat + _scoped_num_leapfrog_steps.
Wave-2 (`test_sccoda_e2e.py`) tested the fast_lfi_call OUT-OF-SCOPE fallback,
the scoped-but-no-static-steps fallback, _call_orig_lfi, and the contextvar
toggles. The existing `test_sccoda.py` drives the full HMC.

What none of them drive in isolation is the actual UNROLLED leapfrog kernel
inside fast_lfi_call (the scoped + valid-static-step path, lines 156-186) on a
real SimpleLeapfrogIntegrator, nor the `zyme=False` dispatch in fast_sample_hmc.
Wave-2 noted these needed a real HMC. Here we drive them WITHOUT paying a
multi-second TF-graph HMC trace, by building a tiny real SimpleLeapfrogIntegrator
on a 3-D Gaussian target and asserting the unrolled integration is BIT-EXACT vs
the captured upstream `SimpleLeapfrogIntegrator.__call__`. We cover:

  - the unrolled n-step leapfrog (no kinetic_energy_fn) -> parity vs upstream,
  - the `kinetic_energy_fn` velocity-parts branch (get_velocity_parts, 163-165),
  - the out-of-range step-count fallback (n_steps > _MAX_UNROLLED_LEAPFROG_STEPS),
  - fast_sample_hmc's `zyme=False` dispatch: RuntimeError when no original is
    captured, and verbatim forward to the captured original when it is.

NOTE (env): sccoda imports tensorflow at module import. This file builds tiny
TF tensors + one real leapfrog integration (no @tf.function HMC graph), so it
stays well under the time budget. If a numba/TF import-order clash appears, run
it alone.

The numpy-step-count branch of fast_lfi_call (lines 137-140: _num_steps a
symbolic tensor with `tf.get_static_value -> None` but a live `.numpy()`) is
only reachable when _num_steps is a graph-captured eager tensor inside a
@tf.function trace -- in eager mode tf.get_static_value always resolves an
eager tensor, so it cannot be hit without compiling the full HMC graph. It is
exercised by the real-HMC `test_sccoda.py`; not duplicated here. See report.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

np = pytest.importorskip("numpy")
tf = pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_probability")
pytest.importorskip("sccoda")

import autozyme
from autozyme import sccoda as azsccoda
from tensorflow_probability.python.mcmc.internal import leapfrog_integrator as _lfi


def _gaussian_target(x):
    """Standard-normal log density (up to a constant) on the last axis."""
    return -0.5 * tf.reduce_sum(x ** 2, axis=-1)


def _make_integrator(n_steps, step=0.1):
    step_sizes = [tf.constant(step, dtype=tf.float64)]
    return _lfi.SimpleLeapfrogIntegrator(
        _gaussian_target, step_sizes, num_steps=n_steps
    )


def _state_momentum():
    state = [tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)]
    momentum = [tf.constant([0.5, -0.5, 0.25], dtype=tf.float64)]
    return state, momentum


@pytest.fixture(autouse=True)
def _activate():
    autozyme.activate("sccoda")
    yield
    azsccoda._sccoda_leapfrog_scope.set(False)
    azsccoda._sccoda_leapfrog_steps.set(None)
    autozyme.deactivate_all()


def _run_scoped(integ, momentum, state, n_steps, **kw):
    scope_tok = azsccoda._sccoda_leapfrog_scope.set(True)
    steps_tok = azsccoda._sccoda_leapfrog_steps.set(n_steps)
    try:
        return azsccoda.fast_lfi_call(integ, momentum, state, **kw)
    finally:
        azsccoda._sccoda_leapfrog_steps.reset(steps_tok)
        azsccoda._sccoda_leapfrog_scope.reset(scope_tok)


def test_unrolled_leapfrog_matches_upstream():
    """The scoped unrolled integration (static int step count) reproduces the
    upstream tf.while_loop leapfrog bit-for-bit on a real integrator."""
    integ = _make_integrator(4)
    state, momentum = _state_momentum()

    # Upstream reference (patch is active but scope is OFF -> delegates).
    ref_mp, ref_sp, ref_t, ref_tg = azsccoda._call_orig_lfi(integ, momentum, state)
    # Scoped fast unrolled path.
    fast_mp, fast_sp, fast_t, fast_tg = _run_scoped(integ, momentum, state, 4)

    np.testing.assert_allclose(
        fast_sp[0].numpy(), ref_sp[0].numpy(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        fast_mp[0].numpy(), ref_mp[0].numpy(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        fast_t.numpy(), ref_t.numpy(), rtol=1e-12, atol=1e-12
    )


def test_unrolled_leapfrog_single_step():
    """A 1-step unrolled integration also matches upstream (boundary half-step
    momentum kicks)."""
    integ = _make_integrator(1)
    state, momentum = _state_momentum()
    ref_mp, ref_sp, _, _ = azsccoda._call_orig_lfi(integ, momentum, state)
    fast_mp, fast_sp, _, _ = _run_scoped(integ, momentum, state, 1)
    np.testing.assert_allclose(fast_sp[0].numpy(), ref_sp[0].numpy(),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fast_mp[0].numpy(), ref_mp[0].numpy(),
                               rtol=1e-12, atol=1e-12)


def test_unrolled_leapfrog_with_kinetic_energy_fn_matches_upstream():
    """Passing a kinetic_energy_fn takes the get_velocity_parts branch
    (lines 163-165); result still matches upstream with the same fn."""
    integ = _make_integrator(3)
    state, momentum = _state_momentum()

    def kinetic(parts):
        # Standard Gaussian kinetic energy sum(0.5 * m^2).
        return 0.5 * tf.reduce_sum(parts[0] ** 2)

    ref_mp, ref_sp, _, _ = azsccoda._call_orig_lfi(
        integ, momentum, state, kinetic_energy_fn=kinetic
    )
    fast_mp, fast_sp, _, _ = _run_scoped(
        integ, momentum, state, 3, kinetic_energy_fn=kinetic
    )
    np.testing.assert_allclose(fast_sp[0].numpy(), ref_sp[0].numpy(),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fast_mp[0].numpy(), ref_mp[0].numpy(),
                               rtol=1e-12, atol=1e-12)


def test_out_of_range_step_count_falls_back_to_upstream():
    """A step count above _MAX_UNROLLED_LEAPFROG_STEPS (50) takes the fallback
    branch (line 150-154); the result equals the upstream integration."""
    n = azsccoda._MAX_UNROLLED_LEAPFROG_STEPS + 5
    integ = _make_integrator(n)
    state, momentum = _state_momentum()
    ref_mp, ref_sp, _, _ = azsccoda._call_orig_lfi(integ, momentum, state)
    fast_mp, fast_sp, _, _ = _run_scoped(integ, momentum, state, n)
    # Fallback delegates to the same upstream integrator -> identical.
    np.testing.assert_allclose(fast_sp[0].numpy(), ref_sp[0].numpy(),
                               rtol=1e-12, atol=1e-12)


def test_sample_hmc_zyme_false_raises_without_captured_original():
    """fast_sample_hmc(zyme=False) raises if the original sample_hmc was never
    captured (i.e. the patch isn't active)."""
    autozyme.deactivate("sccoda")
    with pytest.raises(RuntimeError, match="original sample_hmc not captured"):
        azsccoda.fast_sample_hmc(object(), zyme=False)


def test_sample_hmc_zyme_false_forwards_to_captured_original(monkeypatch):
    """fast_sample_hmc(zyme=False) forwards verbatim to the captured upstream
    sample_hmc with all params, without entering the fast HMC path."""
    import sccoda.model.scCODA_model as scm

    seen = {}

    def _stub_orig(self, **kwargs):
        seen.update(kwargs)
        seen["self"] = self
        return "UPSTREAM_RESULT"

    # The dispatch reads scCODAModel.sample_hmc.__autozyme_original__.
    fn = scm.scCODAModel.sample_hmc
    monkeypatch.setattr(fn, "__autozyme_original__", _stub_orig, raising=False)

    sentinel_self = object()
    out = azsccoda.fast_sample_hmc(
        sentinel_self, num_results=37, num_burnin=4, num_leapfrog_steps=7,
        step_size=0.02, verbose=False, zyme=False,
    )
    assert out == "UPSTREAM_RESULT"
    assert seen["self"] is sentinel_self
    assert seen["num_results"] == 37
    assert seen["num_burnin"] == 4
    assert seen["num_leapfrog_steps"] == 7
    assert seen["step_size"] == 0.02
