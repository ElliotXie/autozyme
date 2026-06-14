"""Unit tests for the pure numba/numpy kernels in autozyme.scvelo.

The contract test (test_scvelo.py) drives recover_dynamics end to end. Here we
test the self-contained kernels directly against numpy/scipy references:

  - _splicing_solve         transcriptional-kinetics ODE solution (closed form)
  - _streaming_argmin_2d    nearest-time assignment vs numpy argmin
  - fast_get_n_jobs         worker-count resolution logic
  - _csr_matvec_*           CSR @ dense kernels vs scipy
  - NumbaConn.dot           wrapper dispatch vs scipy csr.dot

scvelo only needs to import for the module to load.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.sparse")
pytest.importorskip("numba")
pytest.importorskip("scvelo")

from autozyme import scvelo as azscvelo


# --------------------------------------------------------------------------
# _splicing_solve : closed-form splicing-kinetics ODE
# --------------------------------------------------------------------------
def _ref_splicing(alpha, beta, gamma, t, u0, s0):
    # Reference closed form (same algebra as the kernel, written vectorised).
    inv = 1.0 / (gamma - beta) if gamma != beta else 0.0
    a_b = alpha / beta if beta != 0 else 0.0
    a_g = alpha / gamma if gamma != 0 else 0.0
    c = (alpha - u0 * beta) * inv
    eu = np.exp(-beta * t)
    es = np.exp(-gamma * t)
    u = u0 * eu + a_b * (1.0 - eu)
    s = s0 * es + a_g * (1.0 - es) + c * (es - eu)
    return u, s


def test_splicing_solve_matches_reference():
    t = np.linspace(0, 5, 50)
    u, s = azscvelo._splicing_solve(2.0, 0.7, 0.3, t, 0.1, 0.05)
    ru, rs = _ref_splicing(2.0, 0.7, 0.3, t, 0.1, 0.05)
    np.testing.assert_allclose(u, ru, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(s, rs, rtol=1e-12, atol=1e-13)


def test_splicing_solve_steady_state_at_large_t():
    # As t -> inf: u -> alpha/beta, s -> alpha/gamma (induction from 0).
    t = np.array([1e4])
    alpha, beta, gamma = 3.0, 1.0, 0.5
    u, s = azscvelo._splicing_solve(alpha, beta, gamma, t, 0.0, 0.0)
    assert u[0] == pytest.approx(alpha / beta, rel=1e-9)
    assert s[0] == pytest.approx(alpha / gamma, rel=1e-9)


def test_splicing_solve_t0_returns_initial_state():
    t = np.array([0.0])
    u, s = azscvelo._splicing_solve(2.0, 0.7, 0.3, t, 0.42, 0.17)
    assert u[0] == pytest.approx(0.42)
    assert s[0] == pytest.approx(0.17)


def test_splicing_solve_gamma_equals_beta_no_nan():
    # gamma == beta sets inv = 0 (degenerate) — must not divide by zero.
    t = np.linspace(0, 3, 20)
    u, s = azscvelo._splicing_solve(1.5, 0.6, 0.6, t, 0.0, 0.0)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(s))
    ru, rs = _ref_splicing(1.5, 0.6, 0.6, t, 0.0, 0.0)
    np.testing.assert_allclose(u, ru, rtol=1e-12)
    np.testing.assert_allclose(s, rs, rtol=1e-12)


# --------------------------------------------------------------------------
# _streaming_argmin_2d
# --------------------------------------------------------------------------
def test_streaming_argmin_matches_full_distance():
    rng = np.random.default_rng(0)
    x_obs = rng.standard_normal((30, 2))
    xt = rng.standard_normal((17, 2))
    xt_sq = np.einsum("ij,ij->i", xt, xt)
    got = azscvelo._streaming_argmin_2d(
        np.ascontiguousarray(x_obs), np.ascontiguousarray(xt), xt_sq
    )
    # Reference: full pairwise squared distance argmin. Note the kernel drops the
    # constant ||x_obs||^2 term (it cancels in the argmin), so compare on the
    # same partial-distance objective xt_sq - 2 x.xt.
    partial = xt_sq[None, :] - 2.0 * (x_obs @ xt.T)
    ref = np.argmin(partial, axis=1)
    np.testing.assert_array_equal(got, ref)


def test_streaming_argmin_equivalent_to_true_argmin():
    # On the full Euclidean distance, the partial objective gives the same argmin.
    rng = np.random.default_rng(1)
    x_obs = rng.standard_normal((12, 2))
    xt = rng.standard_normal((9, 2))
    xt_sq = np.einsum("ij,ij->i", xt, xt)
    got = azscvelo._streaming_argmin_2d(
        np.ascontiguousarray(x_obs), np.ascontiguousarray(xt), xt_sq
    )
    full = ((x_obs[:, None, :] - xt[None, :, :]) ** 2).sum(axis=2)
    np.testing.assert_array_equal(got, np.argmin(full, axis=1))


# --------------------------------------------------------------------------
# fast_get_n_jobs
# auto_threads() resolves cpu via ZYME_THREADS (highest precedence env var),
# so we pin that to get a deterministic "cpu" count regardless of the host.
# --------------------------------------------------------------------------
def _pin_cpu(monkeypatch, n):
    # ZYME_THREADS outranks OMP_NUM_THREADS (set by the env recipe).
    monkeypatch.setenv("ZYME_THREADS", str(n))


def test_get_n_jobs_explicit_one_kept(monkeypatch):
    _pin_cpu(monkeypatch, 8)
    assert azscvelo.fast_get_n_jobs(1) == 1


def test_get_n_jobs_none_uses_cpu(monkeypatch):
    _pin_cpu(monkeypatch, 6)
    assert azscvelo.fast_get_n_jobs(None) == 6


def test_get_n_jobs_clamped_to_cpu(monkeypatch):
    _pin_cpu(monkeypatch, 4)
    # request more than cpu -> clamped to cpu
    assert azscvelo.fast_get_n_jobs(100) == 4
    # in-range request preserved
    assert azscvelo.fast_get_n_jobs(3) == 3


def test_get_n_jobs_negative(monkeypatch):
    _pin_cpu(monkeypatch, 8)
    # n_jobs = -1 -> cpu + 1 + (-1) = cpu
    assert azscvelo.fast_get_n_jobs(-1) == 8
    # n_jobs = -2 -> cpu + 1 - 2 = cpu - 1
    assert azscvelo.fast_get_n_jobs(-2) == 7
    # very negative -> floors at 1
    assert azscvelo.fast_get_n_jobs(-100) == 1


# --------------------------------------------------------------------------
# CSR matvec kernels + NumbaConn.dot
# --------------------------------------------------------------------------
def _rand_csr(n, density=0.3, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=density, format="csr", random_state=rng)
    A.data = rng.standard_normal(A.nnz)
    return A.astype(np.float64)


@pytest.mark.parametrize("n_cols", [2, 4])
def test_csr_matvec_2d_matches_scipy(n_cols):
    A = _rand_csr(40, seed=1)
    rng = np.random.default_rng(2)
    X = np.ascontiguousarray(rng.standard_normal((40, n_cols)))
    out = np.empty((40, n_cols))
    indptr = A.indptr.astype(np.int32)
    indices = A.indices.astype(np.int32)
    data = np.ascontiguousarray(A.data, dtype=np.float64)
    if n_cols == 2:
        azscvelo._csr_matvec_2col(indptr, indices, data, X, out)
    else:
        azscvelo._csr_matvec_4col(indptr, indices, data, X, out)
    np.testing.assert_allclose(out, A @ X, rtol=1e-11, atol=1e-12)


def test_csr_matvec_1d_matches_scipy():
    A = _rand_csr(35, seed=3)
    rng = np.random.default_rng(4)
    X = np.ascontiguousarray(rng.standard_normal(35))
    out = np.empty(35)
    azscvelo._csr_matvec_1d(
        A.indptr.astype(np.int32), A.indices.astype(np.int32),
        np.ascontiguousarray(A.data), X, out,
    )
    np.testing.assert_allclose(out, A @ X, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("n_cols", [2, 4])
def test_csr_matvec_transposed_kernels(n_cols):
    A = _rand_csr(30, seed=5)
    rng = np.random.default_rng(6)
    # transposed kernels read X[c, j] -> X has shape (n_cols, n)
    X_T = np.ascontiguousarray(rng.standard_normal((n_cols, 30)))
    out_T = np.empty((n_cols, 30))
    indptr = A.indptr.astype(np.int32)
    indices = A.indices.astype(np.int32)
    data = np.ascontiguousarray(A.data)
    if n_cols == 2:
        azscvelo._csr_matvec_2col_T(indptr, indices, data, X_T, out_T)
    else:
        azscvelo._csr_matvec_4col_T(indptr, indices, data, X_T, out_T)
    # out_T[c] = A @ X_T[c]
    np.testing.assert_allclose(out_T.T, A @ X_T.T, rtol=1e-11, atol=1e-12)


def test_csr_matvec_generic_matches_scipy():
    A = _rand_csr(25, seed=7)
    rng = np.random.default_rng(8)
    n_cols = 5
    X = np.ascontiguousarray(rng.standard_normal((25, n_cols)))
    out = np.empty((25, n_cols))
    azscvelo._csr_matvec_generic(
        A.indptr.astype(np.int32), A.indices.astype(np.int32),
        np.ascontiguousarray(A.data), X, out, n_cols,
    )
    np.testing.assert_allclose(out, A @ X, rtol=1e-11, atol=1e-12)


def test_numbaconn_dot_1d_2d_4d_match_scipy():
    A = _rand_csr(50, seed=9)
    conn = azscvelo.NumbaConn(A)
    assert conn.shape == A.shape
    rng = np.random.default_rng(10)
    for ncol in (1, 2, 4, 6):
        if ncol == 1:
            X = rng.standard_normal(50)
        else:
            X = rng.standard_normal((50, ncol))
        got = conn.dot(np.ascontiguousarray(X))
        np.testing.assert_allclose(got, A @ X, rtol=1e-10, atol=1e-12)


def test_numbaconn_dot_fortran_contiguous_path():
    # Exercises the F-contiguous transposed fast path (n<=2000, n_cols in {2,4}).
    A = _rand_csr(64, seed=11)
    conn = azscvelo.NumbaConn(A)
    rng = np.random.default_rng(12)
    for ncol in (2, 4):
        X = np.asfortranarray(rng.standard_normal((64, ncol)))
        assert X.flags["F_CONTIGUOUS"] and not X.flags["C_CONTIGUOUS"]
        got = conn.dot(X)
        np.testing.assert_allclose(got, A @ X, rtol=1e-10, atol=1e-12)
