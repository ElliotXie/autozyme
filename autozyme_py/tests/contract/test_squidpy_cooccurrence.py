"""Contract tests for the squidpy_cooccurrence patch.

Patched surface (1 target):
  - squidpy.gr._ppatterns._co_occurrence_helper

Public entry: ``squidpy.gr.co_occurrence(adata, cluster_key)``. Computes
the spatial co-occurrence probability of cluster labels at increasing
radii. Fixture: small AnnData with synthetic spatial coords + cluster
labels that form deliberate spatial structure.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
sq = pytest.importorskip("squidpy")


@pytest.fixture
def spatial_adata():
    """200-cell AnnData with 2D coords + 3 spatially-clustered groups."""
    import anndata as ad
    from scipy import sparse

    rng = np.random.default_rng(0)
    n_cells = 200
    n_genes = 50
    # 3 clusters arranged at corners of a triangle.
    centers = np.array([[0, 0], [10, 0], [5, 8]])
    per_cluster = n_cells // 3
    coords = []
    labels = []
    for i, c in enumerate(centers):
        coords.append(c + rng.normal(scale=1.0, size=(per_cluster, 2)))
        labels.extend([str(i)] * per_cluster)
    coords = np.vstack(coords)
    # Pad to n_cells.
    if coords.shape[0] < n_cells:
        extra = n_cells - coords.shape[0]
        coords = np.vstack([coords, rng.normal(scale=5.0, size=(extra, 2))])
        labels.extend(["0"] * extra)

    X = sparse.csr_matrix(rng.random((n_cells, n_genes), dtype=np.float32))
    a = ad.AnnData(X)
    a.obs["cluster"] = pd.Categorical(labels)
    a.obsm["spatial"] = coords
    return a


pd = pytest.importorskip("pandas")


def test_co_occurrence_runs_and_populates_uns(spatial_adata):
    """squidpy.gr.co_occurrence writes results to adata.uns."""
    import autozyme
    if not autozyme.activate("squidpy_cooccurrence"):
        pytest.skip("squidpy_cooccurrence is strict to squidpy==1.6.5")

    sq.gr.co_occurrence(spatial_adata, cluster_key="cluster",
                        spatial_key="spatial", show_progress_bar=False)
    # squidpy stores results under "{cluster_key}_co_occurrence".
    assert "cluster_co_occurrence" in spatial_adata.uns


def test_co_occurrence_zyme_false_matches_patched_shape(spatial_adata):
    """Patched + zyme=False produce co-occurrence matrices of same shape."""
    import autozyme
    import anndata as ad
    if not autozyme.activate("squidpy_cooccurrence"):
        pytest.skip("squidpy_cooccurrence is strict to squidpy==1.6.5")

    a_fast = spatial_adata.copy()
    a_vanilla = spatial_adata.copy()
    sq.gr.co_occurrence(a_fast, cluster_key="cluster", spatial_key="spatial",
                        show_progress_bar=False)
    with autozyme.disabled():
        sq.gr.co_occurrence(a_vanilla, cluster_key="cluster",
                            spatial_key="spatial", show_progress_bar=False)

    res_f = a_fast.uns["cluster_co_occurrence"]
    res_v = a_vanilla.uns["cluster_co_occurrence"]
    assert type(res_f) is type(res_v)
    # Both contain "occ" + "interval" arrays.
    assert set(res_f.keys()) == set(res_v.keys())
    assert res_f["occ"].shape == res_v["occ"].shape
