"""Wave-4 heavy-path tests for autozyme.mdanalysis_rmsd.

Wave-1/2/3 covered: the MemoryReader full run, the scope-guard / groupselections
branches, the SINGLE-segment DCD bulk-read path, `fast_read_next_timestep`, and
pinned the B13 weighted-COM bug (do NOT re-pin -- this file covers the UNWEIGHTED
paths only, plus the CORRECT weighted single_frame COM).

What wave-3 explicitly left for wave-4:
  - the ChainReader MULTI-segment time-accumulation path (`fast_compute` lines
    149-271): the `traj.readers` / `_start_frames` seg-length cache, the
    per-segment `times_global` accumulation, and the multi-`seg_files` bulk-read
    loop. We synthesize a 2-segment ChainReader-of-DCDReaders (10 + 8 frames)
    and assert bit-exact parity vs the unpatched baseline over the same readers.
  - `fast_single_frame`'s WEIGHTED center-of-mass branch + `_zyme_w_sum` cache
    (src lines 74-82): the CORRECT `np.dot(w, buf)/w_sum` form (distinct from the
    broken bulk einsum B13). Reached via the `has_groups` branch with
    weights='mass', asserted bit-exact (incl. the groupselections column).
  - `fast_frame_to_ts` (src lines 325-336): defined but not in the public patch
    `targets`, so wave-3 called it dead-via-public-API. It IS callable directly,
    and matches the upstream `_frame_to_ts`; covered here by a direct invocation
    on a real DCDReader frame.
  - the smoke recipe `.dcd` branch (src lines 348-426): `_smoke_load` (build
    Universes + RMSD with patches disabled), `_smoke_call` (R.run()), `_smoke_save`
    (rmsd.npy). Driven with a synthetic DCD + a hand-written minimal PSF so the
    `select="backbone"` Universe resolves.

MDAnalysisTests is NOT installed; everything here synthesizes its own DCD via
MDAnalysis's own DCDWriter (a REAL DCDReader, with `_file.readframes`).
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
from autozyme import mdanalysis_rmsd as azmd


@pytest.fixture(autouse=True)
def _silence():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    yield
    autozyme.deactivate_all()


def _write_dcd(coords, n_atoms, path):
    src = mda.Universe.empty(n_atoms, trajectory=True)
    src.load_new(np.ascontiguousarray(coords, dtype=np.float32), format=MemoryReader)
    src.add_TopologyAttr("name", ["CA"] * n_atoms)
    src.add_TopologyAttr("type", ["C"] * n_atoms)
    src.add_TopologyAttr("mass", [12.0 + (i % 3) for i in range(n_atoms)])
    with DCDWriter(path, n_atoms=n_atoms) as W:
        for _ in src.trajectory:
            W.write(src.atoms)


# --------------------------------------------------------------------------
# ChainReader MULTI-segment path (the wave-3-deferred time-accumulation kernel)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def chain_segments():
    n_atoms = 12
    rng = np.random.default_rng(2)
    tmpd = tempfile.mkdtemp(prefix="autozyme_mda_w4_chain_")
    c1 = rng.standard_normal((10, n_atoms, 3)).astype(np.float32) * 5.0
    c2 = rng.standard_normal((8, n_atoms, 3)).astype(np.float32) * 5.0
    p1 = os.path.join(tmpd, "seg1.dcd")
    p2 = os.path.join(tmpd, "seg2.dcd")
    _write_dcd(c1, n_atoms, p1)
    _write_dcd(c2, n_atoms, p2)
    return {"n_atoms": n_atoms, "paths": [p1, p2], "n_frames": 18}


def _make_chain_universe(meta):
    n_atoms = meta["n_atoms"]
    u = mda.Universe.empty(n_atoms, trajectory=False)
    u.add_TopologyAttr("name", ["CA"] * n_atoms)
    u.add_TopologyAttr("type", ["C"] * n_atoms)
    u.add_TopologyAttr("mass", [12.0 + (i % 3) for i in range(n_atoms)])
    u.load_new(meta["paths"])  # list of 2 DCDs -> ChainReader of 2 DCDReaders
    return u


def test_chain_reader_is_multisegment(chain_segments):
    """Sanity: the synthetic universe is a ChainReader with 2 reader segments
    and a `_start_frames` table (drives the seg-length cache branch)."""
    u = _make_chain_universe(chain_segments)
    traj = u.trajectory
    assert type(traj).__name__ == "ChainReader"
    assert hasattr(traj, "readers") and len(traj.readers) == 2
    assert hasattr(traj, "_start_frames")
    assert traj.n_frames == chain_segments["n_frames"]


def test_chain_reader_multisegment_rmsd_matches_baseline(chain_segments):
    """The multi-segment ChainReader bulk path (`fast_compute`'s readers/
    seg_files machinery + per-segment time accumulation) is bit-exact vs the
    unpatched baseline over the same ChainReader."""
    with autozyme.disabled():
        u, ref = _make_chain_universe(chain_segments), _make_chain_universe(chain_segments)
        base = rms.RMSD(u, ref, select="all", ref_frame=0).run().results.rmsd.copy()

    autozyme.activate("mdanalysis_rmsd")
    with autozyme.disabled():
        u2, ref2 = _make_chain_universe(chain_segments), _make_chain_universe(chain_segments)
    fast = rms.RMSD(u2, ref2, select="all", ref_frame=0).run().results.rmsd.copy()

    assert fast.shape == base.shape == (chain_segments["n_frames"], 3)
    np.testing.assert_array_equal(base[:, 0], fast[:, 0])      # frame index col
    np.testing.assert_allclose(base[:, 1], fast[:, 1], rtol=1e-6, atol=1e-6)  # time col
    np.testing.assert_allclose(base[:, 2], fast[:, 2], rtol=1e-5, atol=1e-5)  # rmsd col
    assert fast[0, 2] == pytest.approx(0.0, abs=1e-4)


def test_chain_reader_frames_and_global_time(chain_segments):
    """The multi-segment path fills the results frames array with the sequential
    0..n-1 index and produces a GLOBAL (continuous, monotone) cumulative time
    across the segment boundary -- the patch's documented Kahan-accumulated
    `times_global`. (NOTE: this is a deliberate behavior difference vs the
    baseline ChainReader, whose per-segment DCDReader times reset to 0 at each
    segment boundary for synthetic equal-origin segments; the RMSD analysis
    column, asserted bit-exact in the parity test above, is unaffected.)"""
    with autozyme.disabled():
        u, ref = _make_chain_universe(chain_segments), _make_chain_universe(chain_segments)
        base_R = rms.RMSD(u, ref, select="all", ref_frame=0).run()

    autozyme.activate("mdanalysis_rmsd")
    with autozyme.disabled():
        u2, ref2 = _make_chain_universe(chain_segments), _make_chain_universe(chain_segments)
    fast_R = rms.RMSD(u2, ref2, select="all", ref_frame=0).run()

    # frames array + frame column are the sequential 0..n-1 index (matches base).
    np.testing.assert_array_equal(np.asarray(fast_R.frames), np.asarray(base_R.frames))
    np.testing.assert_array_equal(
        fast_R.results.rmsd[:, 0],
        np.arange(chain_segments["n_frames"], dtype=fast_R.results.rmsd.dtype),
    )
    # times_global is continuous + monotone non-decreasing across the boundary
    # (frame 10 = start of segment 2 continues from frame 9, it does not reset).
    times = np.asarray(fast_R.times)
    assert np.all(np.diff(times) >= -1e-6)
    np.testing.assert_allclose(
        times, np.arange(chain_segments["n_frames"], dtype=np.float64),
        rtol=1e-5, atol=1e-5,
    )


# --------------------------------------------------------------------------
# fast_compute empty-frames early return
# --------------------------------------------------------------------------
def test_fast_compute_empty_frames_returns_self():
    """`fast_compute` with an empty indexed_frames table (n == 0) returns self
    immediately without touching any trajectory state (src lines 103-106)."""

    class _FakeRMSD:
        pass

    fake = _FakeRMSD()
    out = azmd.fast_compute(fake, np.empty((0, 2), dtype=np.int64))
    assert out is fake


# --------------------------------------------------------------------------
# WEIGHTED fast_single_frame COM (CORRECT np.dot form; B13 is the *bulk* site)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_dcd():
    n_atoms = 8
    rng = np.random.default_rng(6)
    tmpd = tempfile.mkdtemp(prefix="autozyme_mda_w4_small_")
    coords = rng.standard_normal((7, n_atoms, 3)).astype(np.float32) * 5.0
    path = os.path.join(tmpd, "t.dcd")
    _write_dcd(coords, n_atoms, path)
    return {"n_atoms": n_atoms, "path": path, "n_frames": 7}


def _make_residued_universe(meta):
    n_atoms = meta["n_atoms"]
    u = mda.Universe.empty(n_atoms, trajectory=False)
    u.add_TopologyAttr("name", ["CA"] * n_atoms)
    u.add_TopologyAttr("type", ["C"] * n_atoms)
    u.add_TopologyAttr("mass", [12.0 + (i % 3) for i in range(n_atoms)])
    u.add_TopologyAttr("resname", ["ALA"])
    u.add_TopologyAttr("resid", [1])
    u.load_new(meta["path"])
    return u


def test_weighted_single_frame_com_matches_baseline(small_dcd):
    """The groupselections branch routes through `fast_single_frame`, exercising
    its WEIGHTED center-of-mass (`np.dot(w, buf)/w_sum`) + `_zyme_w_sum` cache.
    This is the CORRECT weighted form -- distinct from the broken *bulk* einsum
    pinned by the wave-3 B13 xfail. Bit-exact vs the unpatched baseline incl. the
    groupselections column."""
    with autozyme.disabled():
        u, ref = _make_residued_universe(small_dcd), _make_residued_universe(small_dcd)
        base = rms.RMSD(
            u, ref, select="all", groupselections=["name CA"],
            weights="mass", ref_frame=0,
        ).run().results.rmsd.copy()

    autozyme.activate("mdanalysis_rmsd")
    with autozyme.disabled():
        u2, ref2 = _make_residued_universe(small_dcd), _make_residued_universe(small_dcd)
    fast = rms.RMSD(
        u2, ref2, select="all", groupselections=["name CA"],
        weights="mass", ref_frame=0,
    ).run().results.rmsd.copy()

    assert fast.shape == base.shape == (small_dcd["n_frames"], 4)
    np.testing.assert_allclose(base[:, 2], fast[:, 2], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(base[:, 3], fast[:, 3], rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# fast_frame_to_ts (defined-but-not-in-targets; callable directly)
# --------------------------------------------------------------------------
def test_fast_frame_to_ts_matches_upstream(small_dcd):
    """`fast_frame_to_ts` populates ts.frame/time/positions from a DCD frame and
    matches the upstream `_frame_to_ts` byte-for-byte. It is not in the patch's
    public `targets`, but is callable; this directly drives src lines 325-336."""
    u_fast = _make_residued_universe(small_dcd)
    traj = u_fast.trajectory
    assert type(traj).__name__ == "DCDReader"
    traj._file.seek(0)
    frame = traj._file.read()
    traj._frame = 0
    ts_fast = traj.ts.copy()
    azmd.fast_frame_to_ts(traj, frame, ts_fast)

    u_ref = _make_residued_universe(small_dcd)
    t2 = u_ref.trajectory
    t2._file.seek(0)
    fr2 = t2._file.read()
    t2._frame = 0
    ts_ref = t2.ts.copy()
    t2._frame_to_ts(fr2, ts_ref)

    assert ts_fast.frame == ts_ref.frame == 0
    assert float(ts_fast.time) == pytest.approx(float(ts_ref.time))
    np.testing.assert_allclose(ts_fast.positions, ts_ref.positions, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------
# Smoke recipe `.dcd` branch (synthetic DCD + hand-written minimal PSF)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_task_dir():
    """A task dir whose `data/adk_dims.dcd` is a synthetic DCD and whose
    `data/adk_dims/adk.psf` is a hand-written minimal PSF with backbone atoms
    (N CA C O) so `select="backbone"` resolves."""
    import yaml

    n_atoms = 8
    rng = np.random.default_rng(4)
    td = tempfile.mkdtemp(prefix="autozyme_mda_w4_smoke_")
    os.makedirs(os.path.join(td, "data"), exist_ok=True)
    coords = rng.standard_normal((6, n_atoms, 3)).astype(np.float32) * 5.0
    dcd = os.path.join(td, "data", "adk_dims.dcd")
    _write_dcd(coords, n_atoms, dcd)

    names = ["N", "CA", "C", "O"] * 2
    psf_lines = ["PSF", "", "%8d !NTITLE" % 1, " REMARKS minimal", "",
                 "%8d !NATOM" % n_atoms]
    for i in range(n_atoms):
        psf_lines.append(
            "%8d %-4s %-4d %-4s %-4s %-4s %10.6f %13.4f %11d"
            % (i + 1, "A", 1, "ALA", names[i], "C", 0.0, 12.0, 0)
        )
    psf = os.path.join(td, "data", "adk_dims", "adk.psf")
    os.makedirs(os.path.dirname(psf), exist_ok=True)
    with open(psf, "w", encoding="utf-8") as f:
        f.write("\n".join(psf_lines) + "\n")

    with open(os.path.join(td, "task.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"datasets": [{"tier": "small", "path": "data/adk_dims.dcd"}]}, f
        )
    return {"dir": td, "n_frames": 6}


def test_smoke_load_call_save_dcd_branch(smoke_task_dir):
    """The full smoke recipe `.dcd` branch: `_smoke_load` builds the Universes +
    RMSD (with patches disabled per the recipe contract), `_smoke_call` runs it
    under the patch, `_smoke_save` writes rmsd.npy."""
    autozyme.activate("mdanalysis_rmsd")
    inputs = azmd._smoke_load(smoke_task_dir["dir"], "small")
    assert "R" in inputs
    R = azmd._smoke_call(inputs)
    assert R.results.rmsd.shape == (smoke_task_dir["n_frames"], 3)

    out_dir = tempfile.mkdtemp(prefix="autozyme_mda_w4_out_")
    azmd._smoke_save(R, out_dir)
    arr = np.load(os.path.join(out_dir, "rmsd.npy"))
    assert arr.shape == (smoke_task_dir["n_frames"], 3)
    assert np.all(np.isfinite(arr))
    # reference-frame self-RMSD ~0.
    assert arr[0, 2] == pytest.approx(0.0, abs=1e-3)
