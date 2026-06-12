"""Contract tests for the fipy patch (PDE solver caching).

Patched surface (3 layers of caching):
  - LinearLUSolver._solve_              (content-keyed splu cache)
  - _BinaryTerm._buildAndAddMatrices    (combined-matrix cache)
  - Term.solve                          (user-facing entry; short-
                                         circuits to cached LU + RHS)

The cache logic is most meaningful on REPEATED ``eqn.solve()`` calls
with the same matrix structure (typical of timestepping). A contract
test verifies basic round-trip + zyme=False parity on a single solve.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fipy")


@pytest.fixture
def diffusion_eqn():
    """1-D diffusion on a 30-cell mesh; canonical fipy tutorial setup."""
    import fipy as fp

    mesh = fp.Grid1D(nx=30)
    var = fp.CellVariable(name="phi", mesh=mesh, value=0.0)
    var.constrain(1.0, mesh.facesLeft)
    var.constrain(0.0, mesh.facesRight)
    eqn = fp.DiffusionTerm(coeff=1.0)
    return eqn, var


def test_term_solve_runs_and_updates_var(diffusion_eqn):
    """Term.solve advances the CellVariable to the steady state."""
    import autozyme
    autozyme.activate("fipy")

    eqn, var = diffusion_eqn
    eqn.solve(var=var)
    vals = var.value
    # Steady-state Laplace solution: linear from 1 at left to 0 at right.
    assert vals[0] > 0.9
    assert vals[-1] < 0.1
    # Monotone decreasing.
    diffs = vals[1:] - vals[:-1]
    assert (diffs <= 1e-9).all(), "solution not monotone decreasing"


def test_term_solve_zyme_false_matches_vanilla(diffusion_eqn):
    """Patched and zyme=False solutions match within float tolerance."""
    import autozyme
    import numpy as np
    autozyme.activate("fipy")

    eqn, var = diffusion_eqn
    eqn.solve(var=var)
    fast_vals = np.array(var.value)

    # Rebuild fixture to avoid in-place contamination.
    import fipy as fp
    mesh_v = fp.Grid1D(nx=30)
    var_v = fp.CellVariable(name="phi", mesh=mesh_v, value=0.0)
    var_v.constrain(1.0, mesh_v.facesLeft)
    var_v.constrain(0.0, mesh_v.facesRight)
    eqn_v = fp.DiffusionTerm(coeff=1.0)
    with autozyme.disabled():
        eqn_v.solve(var=var_v)
    ref_vals = np.array(var_v.value)

    np.testing.assert_allclose(fast_vals, ref_vals, rtol=1e-6, atol=1e-9,
                               err_msg="fipy solution drift")
