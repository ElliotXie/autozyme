"""End-to-end / wrapper-line tests for autozyme.squidpy_cooccurrence.

Wave-1 (`test_squidpy_cooccurrence_unit.py`) tested the numba kernels directly
(coverage-invisible njit bodies); the existing `test_squidpy_cooccurrence.py`
drives `sq.gr.co_occurrence` once. This file targets the COVERAGE-VISIBLE python
wrapper lines that neither hits:

  - `fast_co_occurrence_helper`'s python dispatch: the per-tile path AND the
    all-tiles production path (>= _ALL_TILES_THRESHOLD split-pairs), plus the
    `queue` Signal.UPDATE/FINISH branch.
  - activate -> inspect -> deactivate lifecycle on the single bind site.

The helper is called the way squidpy's `parallelize(...)` wrapper calls it, so
we drive it with the exact (idx_splits, spatial_splits, labs_splits, ...)
contract and assert parity between the per-tile and all-tiles python paths.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("numba")
pytest.importorskip("squidpy")

import autozyme
from autozyme import squidpy_cooccurrence as cooc


def _splits(seed=0, n_splits=3, per=8, num=3):
    rng = np.random.default_rng(seed)
    spatial_splits = [
        np.ascontiguousarray(rng.random((per, 2)) * 5.0, dtype=np.float32)
        for _ in range(n_splits)
    ]
    labs_splits = [
        np.ascontiguousarray(rng.integers(0, num, size=per), dtype=np.int32)
        for _ in range(n_splits)
    ]
    labs_unique = np.arange(num, dtype=np.int32)
    interval = np.linspace(0.0, 8.0, 6).astype(np.float32)
    # All same+cross tile pairs.
    idx_splits = [(i, j) for i in range(n_splits) for j in range(i, n_splits)]
    return idx_splits, spatial_splits, labs_splits, labs_unique, interval


def test_helper_per_tile_path_runs():
    """The python per-tile dispatch (< threshold) returns one array per tile."""
    idx, sp, labs, uniq, interval = _splits(seed=1)
    out = cooc.fast_co_occurrence_helper(idx, sp, labs, uniq, interval)
    assert len(out) == len(idx)
    for arr in out:
        assert arr.shape == (uniq.shape[0], uniq.shape[0], interval.shape[0] - 1)
        assert np.all(np.isfinite(arr))


def test_helper_all_tiles_matches_per_tile(monkeypatch):
    """Production all-tiles python path (>= threshold) == per-tile path.

    We lower the all-tiles threshold so a small fixture exercises the
    `_process_all_tiles_2d` branch + the out_lst gather, then compare to the
    per-tile branch (threshold restored high) on the same inputs.
    """
    idx, sp, labs, uniq, interval = _splits(seed=2, n_splits=3, per=10)

    monkeypatch.setattr(cooc, "_ALL_TILES_THRESHOLD", 1)
    all_tiles = cooc.fast_co_occurrence_helper(idx, sp, labs, uniq, interval)

    monkeypatch.setattr(cooc, "_ALL_TILES_THRESHOLD", 10_000_000)
    per_tile = cooc.fast_co_occurrence_helper(idx, sp, labs, uniq, interval)

    assert len(all_tiles) == len(per_tile) == len(idx)
    for a, b in zip(all_tiles, per_tile):
        np.testing.assert_allclose(
            np.asarray(a, np.float64), np.asarray(b, np.float64),
            rtol=1e-5, atol=1e-6,
        )


def test_helper_queue_emits_update_and_finish():
    """The queue branch pushes one UPDATE per tile + a trailing FINISH."""
    from squidpy._utils import Signal

    idx, sp, labs, uniq, interval = _splits(seed=3, n_splits=2, per=6)

    class _Q:
        def __init__(self):
            self.items = []

        def put(self, x):
            self.items.append(x)

    q = _Q()
    cooc.fast_co_occurrence_helper(idx, sp, labs, uniq, interval, queue=q)
    assert q.items[-1] is Signal.FINISH
    assert q.items.count(Signal.UPDATE) == len(idx)


def test_helper_queue_on_all_tiles_path(monkeypatch):
    """The all-tiles branch also feeds the queue (one UPDATE per tile)."""
    from squidpy._utils import Signal

    idx, sp, labs, uniq, interval = _splits(seed=4, n_splits=2, per=6)
    monkeypatch.setattr(cooc, "_ALL_TILES_THRESHOLD", 1)

    class _Q:
        def __init__(self):
            self.items = []

        def put(self, x):
            self.items.append(x)

    q = _Q()
    cooc.fast_co_occurrence_helper(idx, sp, labs, uniq, interval, queue=q)
    assert q.items[-1] is Signal.FINISH
    assert q.items.count(Signal.UPDATE) == len(idx)


def test_activate_inspect_deactivate_lifecycle():
    """activate binds the dispatcher; inspect reports it bound; deactivate
    restores the upstream original at the single bind site."""
    import squidpy.gr._ppatterns as ppat

    autozyme.deactivate("squidpy_cooccurrence")
    original = ppat._co_occurrence_helper

    assert autozyme.activate("squidpy_cooccurrence") is True
    bound = ppat._co_occurrence_helper
    assert bound is not original
    assert getattr(bound, "__autozyme_fast__", None) is cooc.fast_co_occurrence_helper
    assert getattr(bound, "__autozyme_original__", None) is original

    info = autozyme.inspect("squidpy_cooccurrence")
    assert info["status"] == "active"
    assert info["targets"][0]["currently_bound_to_fast"] is True

    autozyme.deactivate("squidpy_cooccurrence")
    assert ppat._co_occurrence_helper is original


def test_disabled_block_forwards_to_original(monkeypatch):
    """Inside autozyme.disabled() the dispatcher forwards to the upstream
    helper, not the fast one."""
    import squidpy.gr._ppatterns as ppat

    autozyme.activate("squidpy_cooccurrence")
    calls = {"fast": 0}
    real_fast = cooc.fast_co_occurrence_helper

    def _counting_fast(*a, **k):
        calls["fast"] += 1
        return real_fast(*a, **k)

    # Re-bind the dispatcher's fast fn so we can observe whether it ran.
    monkeypatch.setattr(cooc, "fast_co_occurrence_helper", _counting_fast)
    autozyme.deactivate("squidpy_cooccurrence")
    autozyme.activate("squidpy_cooccurrence")

    idx, sp, labs, uniq, interval = _splits(seed=5, n_splits=2, per=5)
    with autozyme.disabled():
        ppat._co_occurrence_helper(idx, sp, labs, uniq, interval)
    assert calls["fast"] == 0  # disabled() forwarded to the original
    autozyme.deactivate("squidpy_cooccurrence")
