"""Unit tests for autozyme.mdanalysis_rmsd.

This module is almost entirely coupled to MDAnalysis trajectory objects: the hot
logic (COM centering, QCP rotation RMSD, Kahan time accumulation, bulk DCD
reads) lives inside fast_single_frame / fast_compute and operates on Universe /
Reader / Timestep instances, not on free-standing numpy arrays. There is no
pure-python kernel to import and test in isolation.

So we test the patch the way a user exercises it: build a tiny IN-MEMORY
Universe (MemoryReader, no DCD files needed), activate the patch, and check the
patched RMSD against (a) the unpatched baseline and (b) MDAnalysis's own
superposition-RMSD reference. These run in well under a second and need no data
files. MDAnalysis must be importable (it is a hard import of the module).
"""
from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")
mda = pytest.importorskip("MDAnalysis")
from MDAnalysis.analysis import rms
from MDAnalysis.coordinates.memory import MemoryReader

import autozyme


@pytest.fixture(autouse=True)
def _silence_mda_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    yield


def _universe(coords):
    n_atoms = coords.shape[1]
    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.load_new(np.ascontiguousarray(coords, dtype=np.float32), format=MemoryReader)
    u.add_TopologyAttr("name", ["CA"] * n_atoms)
    u.add_TopologyAttr("masses", [12.0] * n_atoms)
    return u


def _run_rmsd(coords):
    u = _universe(coords)
    ref = _universe(coords[0:1])
    R = rms.RMSD(u, ref, select="name CA", ref_frame=0)
    R.run()
    return R.results.rmsd.copy()


@pytest.fixture
def traj():
    rng = np.random.default_rng(0)
    return rng.standard_normal((6, 8, 3)).astype(np.float32)  # 6 frames, 8 atoms


def test_patched_rmsd_matches_baseline(traj):
    with autozyme.disabled():
        base = _run_rmsd(traj)
    autozyme.activate("mdanalysis_rmsd")
    try:
        fast = _run_rmsd(traj)
    finally:
        autozyme.deactivate_all()
    # Frame / time columns identical; RMSD column equal to QCP fp noise.
    np.testing.assert_array_equal(base[:, 0], fast[:, 0])
    np.testing.assert_allclose(base[:, 2], fast[:, 2], rtol=1e-5, atol=1e-6)


def test_patched_rmsd_matches_mda_reference(traj):
    # MDAnalysis's own minimal-RMSD between superposed coordinate sets is the
    # independent reference for the QCP rotation kernel the patch drives.
    autozyme.activate("mdanalysis_rmsd")
    try:
        fast = _run_rmsd(traj)
    finally:
        autozyme.deactivate_all()
    ref0 = traj[0].astype(np.float64)
    for i in range(traj.shape[0]):
        expected = rms.rmsd(traj[i].astype(np.float64), ref0, center=True, superposition=True)
        assert fast[i, 2] == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_patched_rmsd_zero_against_self(traj):
    # Frame 0 vs ref=frame 0 -> RMSD ~ 0.
    autozyme.activate("mdanalysis_rmsd")
    try:
        fast = _run_rmsd(traj)
    finally:
        autozyme.deactivate_all()
    assert fast[0, 2] == pytest.approx(0.0, abs=1e-5)


def test_patched_rmsd_translation_invariant(traj):
    # Rigidly translating every frame must not change the (centered,
    # superposed) RMSD the patch computes.
    shifted = traj + np.array([10.0, -5.0, 3.0], dtype=np.float32)
    autozyme.activate("mdanalysis_rmsd")
    try:
        base = _run_rmsd(traj)
        moved = _run_rmsd(shifted)
    finally:
        autozyme.deactivate_all()
    np.testing.assert_allclose(base[:, 2], moved[:, 2], rtol=1e-4, atol=1e-5)


def test_patched_rmsd_frame_indices_sequential(traj):
    autozyme.activate("mdanalysis_rmsd")
    try:
        fast = _run_rmsd(traj)
    finally:
        autozyme.deactivate_all()
    np.testing.assert_array_equal(fast[:, 0], np.arange(traj.shape[0]))
