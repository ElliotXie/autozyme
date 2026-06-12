"""Contract tests for the cellphonedb patch (cell-cell communication).

Patched surface: 7 internal targets in cpdb_statistical_analysis_helper.
Public entry: ``cellphonedb.src.core.methods.method_analysis.call(...)``,
which orchestrates the permutation-based ligand-receptor analysis.

Three of the patched functions are pure data shuffling/aggregation that
could be tested in isolation (fast_shuffle_meta in particular). The
rest depend on threaded ndarray buffers + the cpdb interaction database
parquet, which makes a synthetic fixture impractical. Test what's
unit-testable; defer end-to-end to integration.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("cellphonedb")


def test_shuffle_meta_returns_same_shape_dataframe():
    """fast_shuffle_meta(meta) returns a permuted DataFrame of same shape."""
    import autozyme
    autozyme.activate("cellphonedb")
    from cellphonedb.src.core.methods.cpdb_statistical_analysis_helper import (
        shuffle_meta,
    )

    meta = pd.DataFrame({
        "cell_id": [f"c{i}" for i in range(50)],
        "cell_type": pd.Categorical((["A"] * 25) + (["B"] * 25)),
    })
    out = shuffle_meta(meta)
    assert out.shape == meta.shape
    # Label set preserved under permutation.
    assert set(out["cell_type"]) == set(meta["cell_type"])


def test_shuffle_meta_zyme_false_matches_label_distribution():
    """Patched and vanilla both produce a permutation; only label
    counts must match exactly (the specific order is RNG-dependent)."""
    import autozyme
    autozyme.activate("cellphonedb")
    from cellphonedb.src.core.methods.cpdb_statistical_analysis_helper import (
        shuffle_meta,
    )

    meta = pd.DataFrame({
        "cell_id": [f"c{i}" for i in range(40)],
        "cell_type": pd.Categorical((["A"] * 20) + (["B"] * 20)),
    })
    np.random.seed(0)
    fast = shuffle_meta(meta)
    np.random.seed(0)
    with autozyme.disabled():
        ref = shuffle_meta(meta)
    # Label counts identical (permutation conserves them).
    fast_counts = fast["cell_type"].value_counts().sort_index()
    ref_counts = ref["cell_type"].value_counts().sort_index()
    pd.testing.assert_series_equal(fast_counts, ref_counts)


def test_build_clusters_matches_vanilla_simple_counts():
    """Patched build_clusters matches upstream means/percents on a toy matrix."""
    import autozyme
    autozyme.activate("cellphonedb")
    from cellphonedb.src.core.methods.cpdb_statistical_analysis_helper import (
        build_clusters,
    )

    meta = pd.DataFrame({
        "cell_id": [f"c{i}" for i in range(6)],
        "cell_type": pd.Categorical(["A", "A", "B", "B", "C", "C"]),
    })
    counts = pd.DataFrame(
        {
            "c0": [1.0, 0.0, 2.0],
            "c1": [3.0, 0.0, 0.0],
            "c2": [0.0, 5.0, 1.0],
            "c3": [2.0, 1.0, 0.0],
            "c4": [4.0, 2.0, 0.0],
            "c5": [0.0, 3.0, 6.0],
        },
        index=[101, 102, 103],
        dtype=np.float32,
    )

    fast = build_clusters(meta.copy(), counts.copy(), {}, skip_percent=False)
    with autozyme.disabled():
        ref = build_clusters(meta.copy(), counts.copy(), {}, skip_percent=False)

    pd.testing.assert_frame_equal(fast["means"], ref["means"])
    # Existing patch-of-record returns the same percent values as upstream,
    # but preserves the float32 input dtype where upstream promotes to float64.
    # Keep this contract value-based so release tests do not change runtime
    # semantics without a fresh attest.
    pd.testing.assert_frame_equal(
        fast["percents"], ref["percents"], check_dtype=False
    )
