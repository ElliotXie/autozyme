"""Unit tests for the pure helpers in autozyme.astropy_boxleastsquares.

The contract test (test_astropy_boxleastsquares.py) drives BLS.power end to end.
Here we test the self-contained period-chunking helpers directly:

  - _thread_count        worker-count env resolution
  - _bls_chunk_edges     work-balanced period-grid partition

astropy must import for the module to load. fast_bls_fast itself needs the
upstream bls_fast kernel; we cover it via importorskip but focus on the pure
chunking math.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("astropy")

from autozyme import astropy_boxleastsquares as azbls


# --------------------------------------------------------------------------
# _thread_count
# --------------------------------------------------------------------------
def test_thread_count_reads_zyme_threads(monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "AUTOZYMER_THREADS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ZYME_THREADS", "5")
    assert azbls._thread_count() == 5


def test_thread_count_precedence(monkeypatch):
    # ZYME_THREADS is checked before OMP_NUM_THREADS.
    monkeypatch.setenv("ZYME_THREADS", "3")
    monkeypatch.setenv("OMP_NUM_THREADS", "9")
    assert azbls._thread_count() == 3


def test_thread_count_skips_garbage(monkeypatch):
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "AUTOZYMER_THREADS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ZYME_THREADS", "garbage")
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    # invalid ZYME_THREADS skipped; OMP wins.
    assert azbls._thread_count() == 6


def test_thread_count_default_when_all_unset(monkeypatch):
    import os
    for v in ("ZYME_THREADS", "AUTOZYME_THREADS", "AUTOZYMER_THREADS",
              "OMP_NUM_THREADS"):
        monkeypatch.delenv(v, raising=False)
    n = azbls._thread_count()
    assert n == (os.cpu_count() or 1) + 6


# --------------------------------------------------------------------------
# _bls_chunk_edges
# --------------------------------------------------------------------------
def test_bls_chunk_edges_partitions_full_grid():
    t = np.linspace(0, 100, 500)
    period = np.linspace(0.5, 5.0, 2000)
    duration = np.array([0.1, 0.2])
    edges = azbls._bls_chunk_edges(t, period, duration, oversample=10, n_workers=4)
    assert edges[0] == 0
    assert edges[-1] == len(period)
    # Monotone non-decreasing, covers the whole grid with no gaps/overlaps.
    assert np.all(np.diff(edges) >= 0)
    assert len(edges) == 5


def test_bls_chunk_edges_chunks_are_contiguous_cover():
    t = np.linspace(0, 50, 300)
    period = np.linspace(1.0, 4.0, 1000)
    duration = np.array([0.05, 0.15])
    n_workers = 5
    edges = azbls._bls_chunk_edges(t, period, duration, oversample=8, n_workers=n_workers)
    # Reconstruct the full index range from the per-chunk slices.
    covered = []
    for i in range(n_workers):
        covered.extend(range(int(edges[i]), int(edges[i + 1])))
    assert covered == list(range(len(period)))


def test_bls_chunk_edges_degenerate_bin_duration_uniform():
    # Non-finite / non-positive bin duration -> uniform linspace fallback.
    t = np.linspace(0, 10, 100)
    period = np.linspace(1.0, 3.0, 800)
    duration = np.array([0.0])  # -> bin_duration = 0 -> fallback branch
    edges = azbls._bls_chunk_edges(t, period, duration, oversample=10, n_workers=4)
    expected = np.linspace(0, len(period), 5, dtype=np.int64)
    np.testing.assert_array_equal(edges, expected)


def test_bls_chunk_edges_balances_work():
    # Heavier weight on long periods -> later chunks should be NARROWER in
    # index span (each carries comparable work). Verify the first chunk spans
    # more periods than the last.
    t = np.linspace(0, 100, 200)
    period = np.linspace(0.5, 50.0, 4000)  # wide range -> strong weight gradient
    duration = np.array([0.1])
    edges = azbls._bls_chunk_edges(t, period, duration, oversample=10, n_workers=4)
    spans = np.diff(edges)
    assert spans[0] > spans[-1]
