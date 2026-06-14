"""Wave-4 smoke-recipe tests for autozyme.squidpy_cooccurrence.

KERNEL CEILING: this module is ~97% numba @njit kernel bodies that coverage.py
cannot see -- `_occur_count_fused_2d` (src 102-182), `_occur_count_fused_nd`
(187-216), `_cumsum_normalize` (226-262), and `_process_all_tiles_2d` (272-400).
Waves 1-2 already drive every COVERAGE-VISIBLE python line: the
`fast_co_occurrence_helper` dispatch (per-tile + all-tiles + queue branches), the
`_occur_count_fused` 2D/nd selector, and `_parse_all_tiles_threshold` (all env
branches).

The ONLY reachable python lines waves 1-2 left uncovered are the smoke recipe
(src 468-512): `_smoke_load` (read the h5ad + resolve cluster_key from task.yaml,
incl. the `astype("category")` coercion), `_smoke_call` (the real
`sq.gr.co_occurrence(..., copy=True)` -- the canonical timed call that runs the
patched `_co_occurrence_helper` end-to-end), and `_smoke_save` (co_occurrence.npz).
This file builds the smallest spatial AnnData (80 cells, 2D coords, 3 clusters)
and drives the full load -> call -> save round-trip under the patch.
"""
from __future__ import annotations

import os
import tempfile
import warnings

import pytest

np = pytest.importorskip("numpy")
ad = pytest.importorskip("anndata")
pytest.importorskip("squidpy")
pytest.importorskip("numba")

import autozyme
from autozyme import squidpy_cooccurrence as azsquidpy


@pytest.fixture(autouse=True)
def _silence():
    warnings.filterwarnings("ignore")
    yield
    autozyme.deactivate_all()


def _make_spatial_adata(category_dtype=True):
    rng = np.random.default_rng(0)
    n = 80
    X = rng.random((n, 5)).astype(np.float32)
    a = ad.AnnData(X)
    a.obsm["spatial"] = rng.random((n, 2)).astype(np.float32) * 100
    a.obs["cell_type"] = rng.choice(["A", "B", "C"], n)
    if category_dtype:
        a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    return a


@pytest.fixture(scope="module")
def smoke_task_dir():
    import yaml

    a = _make_spatial_adata(category_dtype=True)
    td = tempfile.mkdtemp(prefix="autozyme_squidpy_w4_")
    os.makedirs(os.path.join(td, "data"), exist_ok=True)
    a.write_h5ad(os.path.join(td, "data", "tiny.h5ad"))
    with open(os.path.join(td, "task.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "tier": "small",
                        "path": "data/tiny.h5ad",
                        "params": {"cluster_key": "cell_type"},
                    }
                ]
            },
            f,
        )
    return td


def test_smoke_load_resolves_cluster_key(smoke_task_dir):
    """`_smoke_load` reads the h5ad + the cluster_key from task.yaml params, and
    returns both."""
    inputs = azsquidpy._smoke_load(smoke_task_dir, "small")
    assert set(inputs.keys()) == {"adata", "cluster_key"}
    assert inputs["cluster_key"] == "cell_type"
    assert str(inputs["adata"].obs["cell_type"].dtype) == "category"


def test_smoke_load_coerces_non_category_cluster_key():
    """When the cluster column is not categorical, `_smoke_load` coerces it to
    `category` (src line 484). We use an INTEGER cluster column because anndata
    round-trips string/object obs columns back as `category` on read (which would
    skip line 484), whereas an int column survives as int64 through the h5ad."""
    import yaml

    rng = np.random.default_rng(1)
    n = 80
    a = ad.AnnData(rng.random((n, 5)).astype(np.float32))
    a.obsm["spatial"] = rng.random((n, 2)).astype(np.float32) * 100
    a.obs["cell_type"] = rng.integers(0, 3, n)  # int64 -> stays non-category
    td = tempfile.mkdtemp(prefix="autozyme_squidpy_w4_obj_")
    os.makedirs(os.path.join(td, "data"), exist_ok=True)
    a.write_h5ad(os.path.join(td, "data", "tiny.h5ad"))
    with open(os.path.join(td, "task.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "tier": "small",
                        "path": "data/tiny.h5ad",
                        "params": {"cluster_key": "cell_type"},
                    }
                ]
            },
            f,
        )
    inputs = azsquidpy._smoke_load(td, "small")
    assert str(inputs["adata"].obs["cell_type"].dtype) == "category"


def test_smoke_call_co_occurrence_and_save(smoke_task_dir):
    """The full smoke recipe under the patch: load -> `_smoke_call`
    (`sq.gr.co_occurrence(copy=True)`, which runs the patched
    `_co_occurrence_helper`) -> `_smoke_save` (co_occurrence.npz with occ +
    interval). The occ array is (n_clusters, n_clusters, n_intervals)."""
    autozyme.activate("squidpy_cooccurrence")
    inputs = azsquidpy._smoke_load(smoke_task_dir, "small")
    result = azsquidpy._smoke_call(inputs)
    assert set(result.keys()) == {"occ", "interval"}
    occ = np.asarray(result["occ"])
    assert occ.ndim == 3 and occ.shape[0] == occ.shape[1] == 3  # 3 clusters
    assert np.all(np.isfinite(occ[~np.isnan(occ)]))

    out_dir = tempfile.mkdtemp(prefix="autozyme_squidpy_w4_out_")
    azsquidpy._smoke_save(result, out_dir)
    npz_path = os.path.join(out_dir, "co_occurrence.npz")
    assert os.path.isfile(npz_path)
    z = np.load(npz_path)
    assert set(z.keys()) == {"occ", "interval"}
    np.testing.assert_array_equal(z["occ"].shape, occ.shape)
    # interval has n_intervals+1 edges for n_intervals occ bins.
    assert z["interval"].shape[0] == occ.shape[2] + 1
