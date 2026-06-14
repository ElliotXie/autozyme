"""Wave-3 heavy-path tests for autozyme.mdanalysis_rmsd: the DCD bulk path.

Wave-1 (`test_mdanalysis_rmsd_unit.py`) + wave-2 (`test_mdanalysis_rmsd_e2e.py`)
drove a full in-memory RMSD.run via a MemoryReader, plus the scope-guard
delegation / groupselections branches. Wave-2 noted that the DCD bulk-read fast
path (`fast_compute`'s `use_bulk` branch, lines 219-275) and
`fast_read_next_timestep` (lines 313-322) need real DCD trajectory files
(MDAnalysisTests), which are NOT installed here, so it left them uncovered.

MDAnalysisTests is indeed absent on this machine. BUT we don't need it: we
synthesize a tiny DCD on the fly with MDAnalysis's own DCDWriter, then build a
Universe over it. That gives a REAL DCDReader whose `_file.readframes` exists,
so the bulk path (`DCDFile.readframes(indices=..., order="fac")` collapse +
per-segment COM centering + the QCP RMSD loop) and the in-place
`fast_read_next_timestep` actually fire. This file covers:

  - the unweighted DCD bulk-read RMSD path end-to-end, bit-exact vs the
    unpatched baseline (both over the same real DCDReader),
  - the single-segment seg_files / seg_n_frames machinery + the times/frame
    column population,
  - `fast_read_next_timestep`: in-place ts reuse while iterating a patched
    DCDReader -> frames / times / positions match the unpatched read,
  - REGRESSION (new bug site): the WEIGHTED bulk path (line 252) reuses the
    same wrong einsum `"cij,j->ci"` as the B13 non-bulk bug -- it raises a
    broadcast ValueError. The wave-2 B13 xfail pins the NON-bulk (MemoryReader)
    site; this is a DISTINCT second site on the DCD bulk path, pinned here.

Not covered (documented in the report):
  - `fast_frame_to_ts` (lines 325-336) is defined but NOT in the patch's
    `targets`; the public patch never routes through it (the patched
    `fast_read_next_timestep` calls the upstream `self._frame_to_ts`), so it is
    unreachable dead code via the public API.
"""
from __future__ import annotations

import os
import tempfile
import warnings

import pytest

np = pytest.importorskip("numpy")
mda = pytest.importorskip("MDAnalysis")
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.coordinates.DCD import DCDWriter
from MDAnalysis.analysis import rms

import autozyme


@pytest.fixture(autouse=True)
def _silence():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    yield
    autozyme.deactivate_all()


@pytest.fixture(scope="module")
def dcd_file():
    """Write a tiny synthetic DCD trajectory and return its path + coords.

    Module-scoped: writing the DCD once is the only non-trivial setup cost.
    """
    n_atoms, n_frames = 12, 25
    rng = np.random.default_rng(1)
    coords = (rng.standard_normal((n_frames, n_atoms, 3)).astype(np.float32)
              * 5.0)
    src = mda.Universe.empty(n_atoms, trajectory=True)
    src.load_new(np.ascontiguousarray(coords, dtype=np.float32),
                 format=MemoryReader)
    src.add_TopologyAttr("name", ["CA"] * n_atoms)
    src.add_TopologyAttr("type", ["C"] * n_atoms)
    src.add_TopologyAttr("mass", [12.0 + (i % 3) for i in range(n_atoms)])

    tmpd = tempfile.mkdtemp(prefix="autozyme_dcd_w3_")
    path = os.path.join(tmpd, "traj.dcd")
    with DCDWriter(path, n_atoms=n_atoms) as W:
        for _ in src.trajectory:
            W.write(src.atoms)
    return {"path": path, "n_atoms": n_atoms, "n_frames": n_frames}


def _make_dcd_universe(meta):
    """A Universe whose trajectory is a real DCDReader (has _file.readframes)."""
    n_atoms = meta["n_atoms"]
    u = mda.Universe.empty(n_atoms, trajectory=False)
    u.add_TopologyAttr("name", ["CA"] * n_atoms)
    u.add_TopologyAttr("type", ["C"] * n_atoms)
    u.add_TopologyAttr("mass", [12.0 + (i % 3) for i in range(n_atoms)])
    u.load_new(meta["path"])
    return u


def test_trajectory_is_real_dcd_reader(dcd_file):
    """Sanity: the synthetic universe yields a DCDReader with the readframes
    bulk API the fast path keys on (so the bulk branch is genuinely taken)."""
    u = _make_dcd_universe(dcd_file)
    traj = u.trajectory
    assert type(traj).__name__ == "DCDReader"
    assert hasattr(traj, "_file") and hasattr(traj._file, "readframes")
    assert traj.n_frames == dcd_file["n_frames"]


def test_dcd_bulk_path_unweighted_matches_baseline(dcd_file):
    """The unweighted DCD bulk-read RMSD is bit-exact vs the unpatched baseline
    over the same real DCDReader (drives fast_compute's use_bulk branch)."""
    # Baseline: patches off, but still a real DCDReader.
    with autozyme.disabled():
        u, ref = _make_dcd_universe(dcd_file), _make_dcd_universe(dcd_file)
        base = rms.RMSD(u, ref, select="all", ref_frame=0).run().results.rmsd.copy()

    autozyme.activate("mdanalysis_rmsd")
    # Universe construction with patches disabled (the smoke recipe contract);
    # R.run() then enters the patched bulk path.
    with autozyme.disabled():
        u2, ref2 = _make_dcd_universe(dcd_file), _make_dcd_universe(dcd_file)
    fast = rms.RMSD(u2, ref2, select="all", ref_frame=0).run().results.rmsd.copy()

    assert fast.shape == base.shape == (dcd_file["n_frames"], 3)
    # Frame indices + RMSD column bit-exact (same QCP kernel, same reads).
    np.testing.assert_array_equal(base[:, 0], fast[:, 0])
    np.testing.assert_allclose(base[:, 2], fast[:, 2], rtol=1e-5, atol=1e-5)
    # The reference frame's self-RMSD is ~0.
    assert fast[0, 2] == pytest.approx(0.0, abs=1e-4)


def test_dcd_bulk_path_times_and_frames_populated(dcd_file):
    """The bulk path fills the frame + time columns and the results frames/times
    arrays consistently with the baseline read."""
    with autozyme.disabled():
        u, ref = _make_dcd_universe(dcd_file), _make_dcd_universe(dcd_file)
        base_R = rms.RMSD(u, ref, select="all", ref_frame=0).run()

    autozyme.activate("mdanalysis_rmsd")
    with autozyme.disabled():
        u2, ref2 = _make_dcd_universe(dcd_file), _make_dcd_universe(dcd_file)
    fast_R = rms.RMSD(u2, ref2, select="all", ref_frame=0).run()

    np.testing.assert_array_equal(
        np.asarray(fast_R.frames), np.asarray(base_R.frames)
    )
    np.testing.assert_allclose(
        np.asarray(fast_R.times), np.asarray(base_R.times), rtol=1e-6, atol=1e-6
    )
    # frame column is the sequential 0..n-1 index.
    np.testing.assert_array_equal(
        fast_R.results.rmsd[:, 0],
        np.arange(dcd_file["n_frames"], dtype=fast_R.results.rmsd.dtype),
    )


def test_fast_read_next_timestep_inplace_iteration_matches_baseline(dcd_file):
    """Iterating a patched DCDReader frame-by-frame drives fast_read_next_timestep
    (in-place ts reuse); frames / times / positions match the unpatched read."""
    with autozyme.disabled():
        ub = _make_dcd_universe(dcd_file)
        base = [
            (ts.frame, float(ts.time), ub.atoms.positions.copy())
            for ts in ub.trajectory
        ]

    autozyme.activate("mdanalysis_rmsd")
    # Construction disabled; iteration (the patched _read_next_timestep) active.
    with autozyme.disabled():
        up = _make_dcd_universe(dcd_file)
    fast = [
        (ts.frame, float(ts.time), up.atoms.positions.copy())
        for ts in up.trajectory
    ]

    assert len(fast) == len(base) == dcd_file["n_frames"]
    assert [b[0] for b in base] == [f[0] for f in fast]
    np.testing.assert_allclose(
        [b[1] for b in base], [f[1] for f in fast], rtol=1e-6, atol=1e-6
    )
    for (_, _, pb), (_, _, pf) in zip(base, fast):
        np.testing.assert_allclose(pb, pf, rtol=1e-5, atol=1e-5)


@pytest.mark.xfail(
    reason="SUSPECTED BUG (2nd site of B13): fast_compute's DCD BULK weighted "
    "center-of-mass branch (line 252) uses np.einsum('cij,j->ci', buf, w), "
    "contracting the size-3 coordinate axis with w (length n_atoms) -> broadcast "
    "ValueError. It should be 'cij,i->cj' (contract atom axis i). The wave-2 B13 "
    "xfail pins the NON-bulk (MemoryReader) fallback at line 295; this is the "
    "DISTINCT bulk-path site, reached only with a real DCDReader. Documented, "
    "not fixed (tests-only campaign).",
    raises=ValueError,
    strict=True,
)
def test_dcd_bulk_weighted_path_is_broken(dcd_file):
    """weights='mass' over the DCD bulk path should match the baseline but
    currently raises the same wrong-einsum ValueError as B13."""
    autozyme.activate("mdanalysis_rmsd")
    with autozyme.disabled():
        u, ref = _make_dcd_universe(dcd_file), _make_dcd_universe(dcd_file)
    rms.RMSD(u, ref, select="all", weights="mass", ref_frame=0).run()
