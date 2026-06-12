"""Contract tests for the prody patch (ANM dynamics).

Patched surface (2 targets):
  - ANMBase.buildHessian  (fast_build_hessian, takes **kwargs --
    Bug 2-class candidate for silent kwarg drop)
  - prody.dynamics.anm.solveEig (module-level eigendecomposition)

User-facing entry: ``anm = prody.ANM()``, ``anm.buildHessian(atoms)``,
``anm.calcModes(n_modes)``. Contract pins:
  - buildHessian populates a (3N, 3N) Hessian for N atoms
  - calcModes populates eigenvalues + eigenvectors of expected shape
  - zyme=False matches patched Hessian numerically
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
prody = pytest.importorskip("prody")


@pytest.fixture
def tiny_atoms():
    """50-atom AtomGroup with random coords. CA-only model is the
    canonical ANM input (one site per residue)."""
    n = 50
    ag = prody.AtomGroup("test")
    rng = np.random.default_rng(0)
    coords = rng.random((n, 3)) * 30.0
    ag.setCoords(coords)
    ag.setNames(["CA"] * n)
    ag.setResnames(["ALA"] * n)
    ag.setResnums(np.arange(1, n + 1))
    return ag


@pytest.fixture
def mode_atoms():
    """120-atom synthetic C-alpha model large enough for calcModes."""
    n = 120
    ag = prody.AtomGroup("mode_test")
    rng = np.random.default_rng(0)
    coords = rng.random((n, 3)) * 30.0
    ag.setCoords(coords)
    ag.setNames(["CA"] * n)
    ag.setResnames(["ALA"] * n)
    ag.setResnums(np.arange(1, n + 1))
    return ag


def test_anm_build_hessian_returns_3n_by_3n(tiny_atoms):
    """ANM.buildHessian populates a (3N, 3N) Hessian matrix."""
    import autozyme
    autozyme.activate("prody")

    anm = prody.ANM("test_anm")
    anm.buildHessian(tiny_atoms, cutoff=15.0, gamma=1.0)
    H = anm.getHessian()
    n = tiny_atoms.numAtoms()
    assert H is not None
    assert H.shape == (3 * n, 3 * n)


def test_anm_build_hessian_zyme_false_matches_vanilla(tiny_atoms):
    """Patched Hessian matches vanilla within float tolerance.

    Patched fast_build_hessian stores a SPARSE Hessian (a deliberate
    optimization); vanilla stores DENSE. Densify both before compare so
    we test the numerical contract, not the storage-format contract.
    Users who do ``np.allclose(anm.getHessian(), ref)`` directly will hit
    a sparse/dense mismatch — that's a real downstream-affecting drift
    but separate from "are the Hessian values equal."
    """
    import autozyme
    from scipy.sparse import issparse
    autozyme.activate("prody")

    def _densify(H):
        return H.toarray() if issparse(H) else np.asarray(H)

    with autozyme.disabled():
        anm_v = prody.ANM("vanilla")
        anm_v.buildHessian(tiny_atoms, cutoff=15.0, gamma=1.0)
        H_v = _densify(anm_v.getHessian())
    anm_f = prody.ANM("patched")
    anm_f.buildHessian(tiny_atoms, cutoff=15.0, gamma=1.0)
    H_f = _densify(anm_f.getHessian())

    np.testing.assert_allclose(H_f, H_v, rtol=1e-6, atol=1e-8,
                               err_msg="Hessian drift patched vs vanilla")


def test_anm_calc_modes_returns_eigvals_eigvecs(mode_atoms):
    """ANM.calcModes populates eigenvalues + eigenvectors via solveEig."""
    import autozyme
    autozyme.activate("prody")

    anm = prody.ANM("test_anm")
    anm.buildHessian(mode_atoms, cutoff=15.0, gamma=1.0)
    anm.calcModes(n_modes=6)
    eigvals = anm.getEigvals()
    eigvecs = anm.getEigvecs()
    n = mode_atoms.numAtoms()
    assert len(eigvals) == 6
    assert eigvecs.shape == (3 * n, 6)


def test_anm_calc_modes_zyme_false_matches_eigvals(mode_atoms):
    """Eigenvalues match vanilla within float tolerance."""
    import autozyme
    autozyme.activate("prody")

    with autozyme.disabled():
        anm_v = prody.ANM("vanilla")
        anm_v.buildHessian(mode_atoms, cutoff=15.0, gamma=1.0)
        anm_v.calcModes(n_modes=6)
        ev_v = anm_v.getEigvals()
    anm_f = prody.ANM("patched")
    anm_f.buildHessian(mode_atoms, cutoff=15.0, gamma=1.0)
    anm_f.calcModes(n_modes=6)
    ev_f = anm_f.getEigvals()

    np.testing.assert_allclose(ev_f, ev_v, rtol=1e-4, atol=1e-6,
                               err_msg="ANM eigenvalue drift")


def test_anm_build_hessian_honors_cutoff_kwarg(tiny_atoms):
    """fast_build_hessian captures **kwargs -- verify ``cutoff`` is actually
    honored (Bug 2-class check: a wrapper that silently drops kwargs would
    return identical Hessians regardless of cutoff)."""
    import autozyme
    autozyme.activate("prody")

    anm_loose = prody.ANM("loose")
    anm_loose.buildHessian(tiny_atoms, cutoff=10.0, gamma=1.0)

    anm_tight = prody.ANM("tight")
    anm_tight.buildHessian(tiny_atoms, cutoff=25.0, gamma=1.0)

    # Different cutoffs MUST give different Hessians; if they're identical
    # the patch is silently ignoring the kwarg.
    H_loose = anm_loose.getHessian()
    H_tight = anm_tight.getHessian()
    diff = np.abs(np.asarray(H_loose) - np.asarray(H_tight)).max()
    assert diff > 1e-6, (
        "cutoff kwarg appears to be silently dropped — Hessians "
        "identical despite cutoff=10 vs cutoff=25"
    )
