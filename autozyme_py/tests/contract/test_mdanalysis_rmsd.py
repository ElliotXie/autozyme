"""Contract tests for the mdanalysis_rmsd patch.

Patched surface (3 targets):
  - MDAnalysis.analysis.rms.RMSD._single_frame  (per-frame RMSD computation)
  - MDAnalysis.analysis.rms.RMSD._compute       (whole-trajectory loop)
  - MDAnalysis.coordinates.DCD.DCDReader._read_next_timestep
    (DCD reader hot path)

User-facing entry: ``RMSD(u, u_ref, select=...).run()``. The production
speed path includes DCD bulk reads, but the core RMSD patch is also reached
by a multi-frame MemoryReader trajectory. That lets this contract exercise
``_compute`` and ``_single_frame`` without shipping DCD fixtures.
"""
from __future__ import annotations

import pytest

pytest.importorskip("MDAnalysis")
np = pytest.importorskip("numpy")


def _make_universe(coords):
    import MDAnalysis as mda

    n_atoms = coords.shape[1]
    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.add_TopologyAttr("names", ["CA"] * n_atoms)
    u.add_TopologyAttr("types", ["C"] * n_atoms)
    u.add_TopologyAttr("masses", [12.0] * n_atoms)
    u.load_new(coords, order="fac")
    return u


def _trajectory():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(8, 3)).astype(np.float32)
    coords = np.empty((7, base.shape[0], 3), dtype=np.float32)
    for frame in range(coords.shape[0]):
        drift = np.array([0.05 * frame, 0.02 * frame, 0.0], dtype=np.float32)
        coords[frame] = base + drift
    return coords


def test_rmsd_run_multiframe_returns_rmsd_array():
    """RMSD.run over a multi-frame trajectory must complete and populate rows."""
    import autozyme
    from MDAnalysis.analysis import rms

    autozyme.activate("mdanalysis_rmsd")
    coords = _trajectory()
    out = rms.RMSD(
        _make_universe(coords),
        _make_universe(coords[:1]),
        select="all",
        ref_frame=0,
    ).run()
    assert out.results.rmsd.shape == (coords.shape[0], 3)
    np.testing.assert_array_equal(out.results.rmsd[:, 0], np.arange(coords.shape[0]))


def test_rmsd_zyme_false_matches_vanilla_multiframe():
    """Patched MemoryReader fallback and vanilla RMSD agree on a real loop."""
    import autozyme
    from MDAnalysis.analysis import rms

    autozyme.activate("mdanalysis_rmsd")
    coords = _trajectory()
    fast = rms.RMSD(
        _make_universe(coords),
        _make_universe(coords[:1]),
        select="all",
        ref_frame=0,
    ).run().results.rmsd.copy()
    with autozyme.disabled():
        ref = rms.RMSD(
            _make_universe(coords),
            _make_universe(coords[:1]),
            select="all",
            ref_frame=0,
        ).run().results.rmsd.copy()
    np.testing.assert_allclose(fast, ref, rtol=1e-7, atol=1e-7)
