"""End-to-end / wrapper-line tests for autozyme.fipy.

Wave-1 (`test_fipy_unit.py`) tested the signature helpers (_matrix_signature,
_array_state_sig, _constraint_state_sig, ...). The existing `test_fipy.py` solves
a steady-state DiffusionTerm ONCE -- which only exercises the cache-MISS (first
call) path of the three caching layers.

This file covers the COVERAGE-VISIBLE cache-HIT branches that the single-solve
test never reaches, by running a TRANSIENT loop:

    eqn = TransientTerm() == DiffusionTerm(coeff=const)
    for _ in range(n_steps): eqn.solve(var=v, dt=dt)

The 2nd+ steps hit fast_term_solve / fast_binary_buildAndAddMatrices /
fast_solve_ cache-hit fast paths. We assert step-by-step parity vs a vanilla
(autozyme.disabled) run of the identical loop.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fipy")
np = pytest.importorskip("numpy")

import autozyme
from autozyme import fipy as azfipy


def _transient_solution(n_steps, dt=0.5, disabled=False):
    import fipy as fp

    mesh = fp.Grid1D(nx=20)
    var = fp.CellVariable(name="phi", mesh=mesh, value=0.0)
    var.constrain(1.0, mesh.facesLeft)
    var.constrain(0.0, mesh.facesRight)
    eqn = fp.TransientTerm() == fp.DiffusionTerm(coeff=1.0)

    snapshots = []
    ctx = autozyme.disabled() if disabled else _nullcontext()
    with ctx:
        for _ in range(n_steps):
            eqn.solve(var=var, dt=dt)
            snapshots.append(np.array(var.value))
    return snapshots


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def test_transient_loop_hits_cache_and_matches_vanilla():
    """A 5-step transient loop under the patch matches the vanilla loop at
    every step -- exercising both the first-step cache-miss and the
    subsequent-step cache-hit fast paths."""
    autozyme.activate("fipy")
    # Clear caches so this loop builds them fresh.
    azfipy._LU_CACHE.clear()
    azfipy._BINARY_FULL_CACHE.clear()
    azfipy._TERM_SOLVE_CACHE.clear()
    azfipy._LU_LAST_OBJ[0] = None
    azfipy._LU_LAST_OBJ[1] = None

    fast = _transient_solution(5)
    with autozyme.disabled():
        ref = _transient_solution(5, disabled=True)

    assert len(fast) == len(ref) == 5
    for i, (f, r) in enumerate(zip(fast, ref)):
        np.testing.assert_allclose(f, r, rtol=1e-6, atol=1e-9,
                                   err_msg=f"transient step {i} drift")


def test_term_solve_cache_populated_after_first_step():
    """After the first solve, the Term.solve + LU caches hold entries (proving
    the cache-miss path stored them for the cache-hit branch)."""
    autozyme.activate("fipy")
    azfipy._TERM_SOLVE_CACHE.clear()
    azfipy._LU_CACHE.clear()

    _transient_solution(3)
    assert len(azfipy._TERM_SOLVE_CACHE) >= 1
    assert len(azfipy._LU_CACHE) >= 1


def test_changing_dt_misses_cache():
    """Changing dt between solves changes the cache key -> a fresh upstream
    path runs (cache miss). Solving at two dts both succeed and remain finite."""
    import fipy as fp

    autozyme.activate("fipy")
    mesh = fp.Grid1D(nx=15)
    var = fp.CellVariable(name="phi", mesh=mesh, value=0.0)
    var.constrain(1.0, mesh.facesLeft)
    var.constrain(0.0, mesh.facesRight)
    eqn = fp.TransientTerm() == fp.DiffusionTerm(coeff=1.0)

    eqn.solve(var=var, dt=0.2)
    v1 = np.array(var.value)
    eqn.solve(var=var, dt=0.7)  # different dt -> different key -> miss
    v2 = np.array(var.value)
    assert np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))


def test_steady_state_single_solve_still_works():
    """The original single-solve DiffusionTerm path (cache-miss only) is
    unaffected by the transient cache machinery."""
    import fipy as fp

    autozyme.activate("fipy")
    mesh = fp.Grid1D(nx=30)
    var = fp.CellVariable(name="phi", mesh=mesh, value=0.0)
    var.constrain(1.0, mesh.facesLeft)
    var.constrain(0.0, mesh.facesRight)
    fp.DiffusionTerm(coeff=1.0).solve(var=var)
    vals = var.value
    assert vals[0] > 0.9 and vals[-1] < 0.1


def test_activate_restore_all_three_layers():
    from fipy.solvers.scipy.linearLUSolver import LinearLUSolver
    from fipy.terms.binaryTerm import _BinaryTerm
    from fipy.terms.term import Term

    autozyme.deactivate("fipy")
    orig_solve = LinearLUSolver._solve_
    orig_build = _BinaryTerm._buildAndAddMatrices
    orig_term = Term.solve

    assert autozyme.activate("fipy") is True
    assert Term.solve is not orig_term
    info = autozyme.inspect("fipy")
    assert info["status"] == "active"
    assert len(info["targets"]) == 3

    autozyme.deactivate("fipy")
    assert LinearLUSolver._solve_ is orig_solve
    assert _BinaryTerm._buildAndAddMatrices is orig_build
    assert Term.solve is orig_term
