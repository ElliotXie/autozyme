"""End-to-end / wrapper-line tests for autozyme.mdanalysis_rmsd.

Wave-1 (`test_mdanalysis_rmsd_unit.py`) + the existing `test_mdanalysis_rmsd.py`
drive a full in-memory RMSD.run (frames == 0..n-1, no weights, no group
selections). This file covers the remaining COVERAGE-VISIBLE branches in
fast_compute / fast_single_frame that those don't reach:

  - the scope-guard delegation in fast_compute: a sliced run (step / start)
    where frames != arange(n) -> _orig_compute (which still dispatches the
    fast per-frame kernel).
  - the weighted center-of-mass path (weights='mass').
  - the groupselections path (fast_single_frame -> _orig_single_frame).
  - the n==0 early return.

The DCD bulk-read path + fast_read_next_timestep / fast_frame_to_ts need real
DCD trajectory files (MDAnalysisTests), which are not installed here -- see the
agent report note. We cover everything reachable from an in-memory MemoryReader.
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
def _silence():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    yield


def _universe(coords):
    n_atoms = coords.shape[1]
    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.load_new(np.ascontiguousarray(coords, dtype=np.float32), format=MemoryReader)
    u.add_TopologyAttr("name", ["CA"] * n_atoms)
    u.add_TopologyAttr("type", ["C"] * n_atoms)
    u.add_TopologyAttr("mass", [12.0 + (i % 3) for i in range(n_atoms)])
    return u


@pytest.fixture
def traj():
    rng = np.random.default_rng(0)
    return rng.standard_normal((8, 10, 3)).astype(np.float32)


def test_sliced_run_delegates_and_matches_vanilla(traj):
    """A stepped run (frames != 0..n-1) takes the scope-guard delegation to
    _orig_compute; parity vs vanilla on the same slice."""
    autozyme.activate("mdanalysis_rmsd")
    u, ref = _universe(traj), _universe(traj[0:1])
    fast = rms.RMSD(u, ref, select="name CA", ref_frame=0).run(step=2)
    fast_arr = fast.results.rmsd.copy()

    with autozyme.disabled():
        u2, ref2 = _universe(traj), _universe(traj[0:1])
        van = rms.RMSD(u2, ref2, select="name CA", ref_frame=0).run(step=2)
    np.testing.assert_allclose(fast_arr, van.results.rmsd, rtol=1e-6, atol=1e-6)
    # step=2 over 8 frames -> 4 rows.
    assert fast_arr.shape[0] == 4


def test_start_offset_run_delegates(traj):
    """A run with start>0 also delegates (frames != arange(n))."""
    autozyme.activate("mdanalysis_rmsd")
    u, ref = _universe(traj), _universe(traj[0:1])
    fast = rms.RMSD(u, ref, select="name CA", ref_frame=0).run(start=2)
    with autozyme.disabled():
        u2, ref2 = _universe(traj), _universe(traj[0:1])
        van = rms.RMSD(u2, ref2, select="name CA", ref_frame=0).run(start=2)
    np.testing.assert_allclose(fast.results.rmsd, van.results.rmsd,
                               rtol=1e-6, atol=1e-6)


@pytest.mark.xfail(
    reason="SUSPECTED BUG: fast_compute's weighted center-of-mass branch uses "
    "np.einsum('cij,j->ci', buf, w) which contracts the size-3 coordinate axis "
    "with w (length n_atoms), raising a broadcast ValueError. It should be "
    "'cij,i->cj' (contract the atom axis i). fast_single_frame uses the correct "
    "np.dot(w, buf). Weighted RMSD on the full in-order (non-DCD-bulk) path is "
    "therefore broken; the benchmark only ran unweighted backbone RMSD so it "
    "never tripped. Documented, not fixed (tests-only campaign).",
    raises=ValueError,
    strict=True,
)
def test_weighted_com_path_matches_vanilla(traj):
    """weights='mass' should take the weighted center-of-mass branch in both
    fast_single_frame and fast_compute -- currently raises in fast_compute."""
    autozyme.activate("mdanalysis_rmsd")
    u, ref = _universe(traj), _universe(traj[0:1])
    fast = rms.RMSD(u, ref, select="name CA", weights="mass",
                    ref_frame=0).run().results.rmsd.copy()
    with autozyme.disabled():
        u2, ref2 = _universe(traj), _universe(traj[0:1])
        van = rms.RMSD(u2, ref2, select="name CA", weights="mass",
                       ref_frame=0).run().results.rmsd.copy()
    np.testing.assert_allclose(fast, van, rtol=1e-5, atol=1e-6)


def test_groupselections_routes_to_original(traj):
    """A non-empty groupselections makes fast_single_frame delegate to
    _orig_single_frame for the secondary RMSD; parity vs vanilla."""
    autozyme.activate("mdanalysis_rmsd")
    u, ref = _universe(traj), _universe(traj[0:1])
    fast = rms.RMSD(u, ref, select="name CA", groupselections=["name CA"],
                    ref_frame=0).run().results.rmsd.copy()
    with autozyme.disabled():
        u2, ref2 = _universe(traj), _universe(traj[0:1])
        van = rms.RMSD(u2, ref2, select="name CA", groupselections=["name CA"],
                       ref_frame=0).run().results.rmsd.copy()
    # Columns: frame, time, backbone-RMSD, then one per groupselection.
    assert fast.shape[1] >= 4
    np.testing.assert_allclose(fast[:, :3], van[:, :3], rtol=1e-5, atol=1e-6)


def test_activate_restore_lifecycle():
    from MDAnalysis.coordinates.DCD import DCDReader

    autozyme.deactivate("mdanalysis_rmsd")
    orig_single = rms.RMSD._single_frame
    orig_compute = rms.RMSD._compute
    orig_dcd = DCDReader._read_next_timestep

    assert autozyme.activate("mdanalysis_rmsd") is True
    assert rms.RMSD._single_frame is not orig_single
    info = autozyme.inspect("mdanalysis_rmsd")
    assert info["status"] == "active"
    assert len(info["targets"]) == 3

    autozyme.deactivate("mdanalysis_rmsd")
    assert rms.RMSD._single_frame is orig_single
    assert rms.RMSD._compute is orig_compute
    assert DCDReader._read_next_timestep is orig_dcd
