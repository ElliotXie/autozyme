"""Unit tests for the pure-numpy kernels in autozyme.dipy.

The contract test (test_dipy.py) exercises the full DTI fit through dipy. Here
we test the self-contained linear-algebra kernels directly against numpy/scipy
references:

  - _chol7_solve     batched 7x7 SPD Cholesky solve  vs np.linalg.solve
  - _pinv_wls_fit    SVD-pinv weighted least squares  vs lstsq normal equations
  - _fast_wls_fit_tensor_inner(return_lower_triangular=True)  vs WLS closed form

dipy must import for the module to load; the kernels above touch no dipy
internals except eig_from_lo_tri, which we avoid via return_lower_triangular.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("dipy")

from autozyme import dipy as azdipy


def _spd_batch(n, d=7, seed=0):
    rng = np.random.default_rng(seed)
    out = np.empty((n, d, d))
    for k in range(n):
        a = rng.standard_normal((d, d))
        out[k] = a @ a.T + d * np.eye(d)  # well-conditioned SPD
    return out


def test_chol7_solve_matches_numpy_solve():
    n = 5
    lhs = _spd_batch(n, seed=1)
    rng = np.random.default_rng(2)
    rhs = rng.standard_normal((n, 7))
    x, bad = azdipy._chol7_solve(lhs, rhs)
    assert not bad.any()
    ref = np.linalg.solve(lhs, rhs)
    np.testing.assert_allclose(x, ref, rtol=1e-8, atol=1e-9)


def test_chol7_solve_residual_small():
    lhs = _spd_batch(4, seed=3)
    rng = np.random.default_rng(4)
    rhs = rng.standard_normal((4, 7))
    x, bad = azdipy._chol7_solve(lhs, rhs)
    resid = np.einsum("kij,kj->ki", lhs, x) - rhs
    assert np.max(np.abs(resid)) < 1e-8
    assert not bad.any()


def test_chol7_solve_flags_non_spd_as_bad():
    # A singular / non-positive-definite matrix must be flagged for fallback.
    lhs = _spd_batch(3, seed=5)
    lhs[1] = np.zeros((7, 7))            # zero matrix: not SPD
    lhs[2, 0, 0] = -50.0                 # negative pivot
    rng = np.random.default_rng(6)
    rhs = rng.standard_normal((3, 7))
    x, bad = azdipy._chol7_solve(lhs, rhs)
    assert bad[1] and bad[2]
    assert not bad[0]


def test_chol7_solve_preserves_batch_shape():
    # Multi-dim batch (e.g. (a, b, 7, 7)) -> output (a, b, 7).
    lhs = _spd_batch(6, seed=7).reshape(2, 3, 7, 7)
    rng = np.random.default_rng(8)
    rhs = rng.standard_normal((2, 3, 7))
    x, bad = azdipy._chol7_solve(lhs, rhs)
    assert x.shape == (2, 3, 7)
    assert bad.shape == (2, 3)
    ref = np.linalg.solve(lhs.reshape(-1, 7, 7), rhs.reshape(-1, 7)).reshape(2, 3, 7)
    np.testing.assert_allclose(x, ref, rtol=1e-7, atol=1e-9)


def test_pinv_wls_fit_matches_weighted_normal_equations():
    rng = np.random.default_rng(11)
    g, p = 30, 7
    design = rng.standard_normal((g, p))
    log_s = rng.standard_normal(g)
    w = np.abs(rng.standard_normal(g)) + 0.5  # sqrt-weights, positive
    got = azdipy._pinv_wls_fit(design, log_s, w)
    # Reference: solve (X'W2 X) b = X'W2 y  with W2 = diag(w**2).
    W2 = np.diag(w ** 2)
    ref = np.linalg.solve(design.T @ W2 @ design, design.T @ W2 @ log_s)
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9)


def _wellcond_design(seed=0, g=60):
    # A well-conditioned generic 7-column design. The real DTI design matrix is
    # near-rank-deficient (cond ~1e16), which would make ANY direct normal-
    # equation solve (kernel and numpy reference alike) blow up — that is a
    # property of the DTI parameterisation, not of this kernel. We test the WLS
    # math on a well-conditioned design where the closed form is meaningful.
    rng = np.random.default_rng(seed)
    design = rng.standard_normal((g, 7))
    # last column = negative intercept (mirrors DTI's S0 column sign)
    design[:, -1] = -1.0
    return design


def test_wls_inner_lower_triangular_matches_wls_closed_form():
    design = _wellcond_design(seed=1)
    g = design.shape[0]
    rng = np.random.default_rng(2)
    # The kernel takes raw signal `data` and logs it internally; feed positive
    # signal so log is well-defined. log_s below is the reference target.
    data = np.abs(rng.standard_normal((2, g))) + 1.0
    log_s = np.log(data)
    # Explicit weights (these are w**2 / variance weights per `w = sqrt(weights)`).
    weights = np.abs(rng.standard_normal((2, g))) + 0.3
    fit, leverages = azdipy._fast_wls_fit_tensor_inner(
        design, data, weights=weights, return_lower_triangular=True,
    )
    assert leverages is None
    assert fit.shape == (2, 7)
    # Reference per voxel: solve (X'diag(w^2)X) b = X'diag(w^2)y with W2 = weights.
    for v in range(2):
        W2 = np.diag(weights[v])
        ref = np.linalg.solve(design.T @ W2 @ design, design.T @ W2 @ log_s[v])
        np.testing.assert_allclose(fit[v], ref, rtol=1e-6, atol=1e-8)


def test_wls_inner_min_signal_clamps_log():
    # With min_signal set the data is clamped then logged; verify the clamp by
    # comparing two inputs that differ only below the floor.
    design = _wellcond_design(seed=3)
    g = design.shape[0]
    rng = np.random.default_rng(4)
    data = np.abs(rng.standard_normal((1, g))) + 1.0
    weights = np.ones((1, g))
    floor = 0.5
    d_lo = data.copy()
    d_lo[0, 0] = 1e-6   # below floor
    f1, _ = azdipy._fast_wls_fit_tensor_inner(
        design, d_lo, weights=weights, return_lower_triangular=True, min_signal=floor,
    )
    d_floor = data.copy()
    d_floor[0, 0] = floor
    f2, _ = azdipy._fast_wls_fit_tensor_inner(
        design, d_floor, weights=weights, return_lower_triangular=True, min_signal=floor,
    )
    np.testing.assert_allclose(f1, f2, rtol=1e-10, atol=1e-12)
