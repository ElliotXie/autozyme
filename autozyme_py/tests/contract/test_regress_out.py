"""Contract tests for ``sc.pp.regress_out`` after ``autozyme.activate("scanpy")``."""
from __future__ import annotations

import pytest


def _with_numeric_covariates(adata):
    np = pytest.importorskip("numpy")

    adata = adata.copy()
    n = adata.n_obs
    adata.obs["cov_linear"] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    # Constant zero makes the Gram matrix singular and exercises the pinv
    # path that avoids upstream's slow per-gene statsmodels fallback.
    adata.obs["cov_zero"] = np.zeros(n, dtype=np.float32)
    return adata


def _dense_x(adata):
    np = pytest.importorskip("numpy")
    sparse = pytest.importorskip("scipy.sparse")

    x = adata.X
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x)


def test_regress_out_sparse_singular_matches_vanilla(tiny_adata):
    import autozyme
    import numpy as np
    import scanpy as sc

    a_fast = _with_numeric_covariates(tiny_adata)
    a_vanilla = _with_numeric_covariates(tiny_adata)
    keys = ["cov_linear", "cov_zero"]

    with autozyme.disabled():
        ref = sc.pp.regress_out(a_vanilla, keys, n_jobs=1)
    out = sc.pp.regress_out(a_fast, keys, n_jobs=1)

    assert type(out) is type(ref)
    np.testing.assert_allclose(
        _dense_x(a_fast),
        _dense_x(a_vanilla),
        rtol=2e-5,
        atol=2e-5,
    )


def test_regress_out_copy_true_returns_distinct(tiny_adata):
    import scanpy as sc

    out = sc.pp.regress_out(
        _with_numeric_covariates(tiny_adata),
        ["cov_linear", "cov_zero"],
        n_jobs=1,
        copy=True,
    )
    assert out is not None
    assert out is not tiny_adata


def test_regress_out_zyme_false_delegates(tiny_adata):
    import autozyme
    import numpy as np
    import scanpy as sc

    a_escape = _with_numeric_covariates(tiny_adata)
    a_vanilla = _with_numeric_covariates(tiny_adata)
    keys = ["cov_linear", "cov_zero"]

    with autozyme.disabled():
        sc.pp.regress_out(a_vanilla, keys, n_jobs=1)
    sc.pp.regress_out(a_escape, keys, n_jobs=1, zyme=False)

    np.testing.assert_allclose(
        _dense_x(a_escape),
        _dense_x(a_vanilla),
        rtol=2e-5,
        atol=2e-5,
    )
