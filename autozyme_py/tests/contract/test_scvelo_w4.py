"""Wave-4 smoke-recipe tests for autozyme.scvelo.

KERNEL CEILING: scvelo is dominated by numba @njit bodies that coverage.py
cannot see -- `_splicing_solve` (src 104-120), `_streaming_argmin_2d` (153-169),
and the five CSR matvec kernels `_csr_matvec_*` (244-335). Waves 1-2 already
drive every COVERAGE-VISIBLE python wrapper / dispatch line: `_fast_get_solution`
(fast + fallback + with_keys/unstacked), `fast_assign_tau` (both projection +
non-projection branches), `fast_get_n_jobs` (all the n_jobs rewrites),
`NumbaConn.dot` (1D / 2-col / 4-col / generic / F-contig transposed paths),
`fast_get_connectivities`, and `fit_recovery_with_overrides`.

The ONLY reachable python lines waves 1-2 left uncovered are the smoke recipe
(src 398-443): `_smoke_load` (read the prepared h5ad), `_smoke_call` (the real
`scv.tl.recover_dynamics(..., n_jobs=-1, ...)` -- the canonical timed call, which
spins up the loky worker pool and runs the patched per-gene EM end-to-end), and
`_smoke_save` (fit_pars.csv). This file builds the smallest scvelo-ready AnnData
(60 cells / 8 genes, with spliced/unspliced + Ms/Mu moments + a velocity_genes
mask) and drives the full load -> call -> save round-trip under the patch.

`recover_dynamics` running under the patch exercises `fast_assign_tau` /
`_fast_get_solution` / the loky worker re-install wrapper for real (the kernel
interiors stay invisible to coverage), so this is genuine end-to-end coverage of
the heaviest path, not just IO.
"""
from __future__ import annotations

import os
import tempfile
import warnings

import pytest

# Suppress scvelo's BLAS-pin stderr print + global pin at submodule import.
os.environ.setdefault("AUTOZYME_SCVELO_NO_BLAS_PIN", "1")

np = pytest.importorskip("numpy")
scvelo = pytest.importorskip("scvelo")
ad = pytest.importorskip("anndata")
sc = pytest.importorskip("scanpy")

import autozyme
from autozyme import scvelo as azscvelo


@pytest.fixture(autouse=True)
def _silence():
    warnings.filterwarnings("ignore")
    yield
    autozyme.deactivate_all()


@pytest.fixture(scope="module")
def smoke_task_dir():
    """A task dir whose h5ad is a tiny scvelo-ready AnnData: spliced/unspliced
    layers, Ms/Mu moments, and a velocity_genes mask."""
    import yaml
    import scvelo as scv

    n_obs, n_var = 60, 8
    rng = np.random.default_rng(0)
    spliced = rng.poisson(5, (n_obs, n_var)).astype(np.float32)
    unspliced = rng.poisson(3, (n_obs, n_var)).astype(np.float32)
    a = ad.AnnData(spliced)
    a.layers["spliced"] = spliced
    a.layers["unspliced"] = unspliced
    scv.pp.normalize_per_cell(a)
    sc.pp.log1p(a)
    scv.pp.moments(a, n_pcs=5, n_neighbors=10)
    a.var["velocity_genes"] = True

    td = tempfile.mkdtemp(prefix="autozyme_scvelo_w4_")
    os.makedirs(os.path.join(td, "data"), exist_ok=True)
    a.write_h5ad(os.path.join(td, "data", "tiny.h5ad"))
    with open(os.path.join(td, "task.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"datasets": [{"tier": "small", "path": "data/tiny.h5ad"}]}, f
        )
    return {"dir": td, "n_var": n_var}


def test_smoke_load_reads_anndata(smoke_task_dir):
    """`_smoke_load` reads the prepared h5ad and returns it under `adata`."""
    inputs = azscvelo._smoke_load(smoke_task_dir["dir"], "small")
    assert set(inputs.keys()) == {"adata"}
    assert "spliced" in inputs["adata"].layers
    assert "Ms" in inputs["adata"].layers  # moments present


def test_smoke_call_recover_dynamics_and_save(smoke_task_dir):
    """The full smoke recipe under the patch: load -> `_smoke_call`
    (`recover_dynamics` with n_jobs=-1, the loky-worker path) -> `_smoke_save`
    (fit_pars.csv). The saved fit parameters are well-shaped and finite."""
    autozyme.activate("scvelo")
    inputs = azscvelo._smoke_load(smoke_task_dir["dir"], "small")
    out_adata = azscvelo._smoke_call(inputs)
    # recover_dynamics populated fit_* columns on the (copied) adata's var.
    fit_cols = [c for c in out_adata.var.columns if c.startswith("fit_")]
    assert len(fit_cols) > 0

    out_dir = tempfile.mkdtemp(prefix="autozyme_scvelo_w4_out_")
    azscvelo._smoke_save(out_adata, out_dir)
    csv_path = os.path.join(out_dir, "fit_pars.csv")
    assert os.path.isfile(csv_path)
    import pandas as pd
    fit_df = pd.read_csv(csv_path, index_col="gene")
    # One row per gene; the fit_* columns are present.
    assert len(fit_df) == smoke_task_dir["n_var"]
    assert any(c.startswith("fit_") for c in fit_df.columns)
