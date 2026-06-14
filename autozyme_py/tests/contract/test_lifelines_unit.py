"""Unit tests for the pure numba/numpy kernels in autozyme.lifelines.

The contract test (test_lifelines.py) drives CoxPHFitter.fit end to end. Here we
test the self-contained pieces directly:

  - _efron_kernel_jit   numba Efron partial-likelihood gradient/Hessian, checked
                        against an independent numpy implementation of the Efron
                        tied-time Cox partial likelihood.
  - _all_finite_frame / _all_finite_array   blocked finite scanners.

lifelines must import for the module to load. The kernel itself uses no lifelines
internals (it takes plain numpy arrays).
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("numba")
pytest.importorskip("lifelines")

from autozyme import lifelines as azll


# --------------------------------------------------------------------------
# Independent numpy reference for the Efron partial-likelihood
# gradient/Hessian/log-likelihood, matching the kernel's contract.
# --------------------------------------------------------------------------
def _efron_reference(X, E, weights, beta, T):
    """Compute (hessian, gradient, log_lik) for Cox Efron partial likelihood.

    Mirrors lifelines' _get_efron_values_batch semantics: data must be sorted by
    DESCENDING time (largest time first), `counts` = run lengths of tied times.
    log_lik here is the EM-loop accumulation BEFORE the `+ we_X @ beta` fixup
    (the kernel returns that partial value).
    """
    n, d = X.shape
    scores = weights * np.exp(X @ beta)

    hessian = np.zeros((d, d))
    gradient = np.zeros(d)
    log_lik = 0.0
    risk_phi = 0.0
    risk_phi_x = np.zeros(d)
    risk_phi_x_x = np.zeros((d, d))

    # counts: tied-time groups in ascending position from the end.
    _, counts = np.unique(-T, return_counts=True)
    pos = n
    for cnt in counts:
        s_start = pos - cnt
        s_end = pos
        Xg = X[s_start:s_end]
        sg = scores[s_start:s_end]
        Eg = E[s_start:s_end]
        Wg = weights[s_start:s_end]

        tied_death_counts = 0
        for i in range(cnt - 1, -1, -1):
            if Eg[i] == 1:
                tied_death_counts += 1
            else:
                break

        phi_x_g = sg.reshape(-1, 1) * Xg
        risk_phi += sg.sum()
        risk_phi_x += phi_x_g.sum(axis=0)
        risk_phi_x_x += Xg.T @ phi_x_g

        if tied_death_counts == 0:
            pos -= cnt
            continue

        d_start = cnt - tied_death_counts
        Xd = Xg[d_start:]
        Wd = Wg[d_start:]

        weight_count = Wd.sum()
        x_death_sum = (Wd[:, None] * Xd).sum(axis=0)
        weighted_average = weight_count / tied_death_counts

        if tied_death_counts > 1:
            phi_x_d = phi_x_g[d_start:]
            tie_phi = sg[d_start:].sum()
            tie_phi_x = phi_x_d.sum(axis=0)
            tie_phi_x_x = Xd.T @ phi_x_d

            sum_denom = 0.0
            sum_p_denom = 0.0
            sum_summand = np.zeros(d)
            sum_outer = np.zeros((d, d))
            for p in range(tied_death_counts):
                prop = p / tied_death_counts
                denom = 1.0 / (risk_phi - prop * tie_phi)
                sum_denom += denom
                sum_p_denom += prop * denom
                log_lik += weighted_average * np.log(denom)
                s_vec = (risk_phi_x - prop * tie_phi_x) * denom
                sum_summand += s_vec
                sum_outer += np.outer(s_vec, s_vec)
            gradient += x_death_sum - weighted_average * sum_summand
            a1 = risk_phi_x_x * sum_denom - tie_phi_x_x * sum_p_denom
            hessian += weighted_average * (sum_outer - a1)
        else:
            denom = 1.0 / risk_phi
            log_lik += weighted_average * np.log(denom)
            sum_summand = risk_phi_x * denom
            gradient += x_death_sum - weighted_average * sum_summand
            a1 = risk_phi_x_x * denom
            a2 = np.outer(sum_summand, sum_summand)
            hessian += weighted_average * (a2 - a1)

        pos -= cnt
    return hessian, gradient, log_lik


def _make_cox_data(n=40, d=3, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    if ties:
        T = rng.integers(1, 6, size=n).astype(float)  # heavily tied
    else:
        T = rng.uniform(0.1, 10.0, size=n)
    E = (rng.random(n) > 0.3).astype(np.int64)  # ~70% events
    weights = np.ones(n)
    # Sort descending by time (kernel contract).
    order = np.argsort(-T)
    return (np.ascontiguousarray(X[order]), E[order], weights[order],
            T[order])


def _counts(T):
    _, counts = np.unique(-T, return_counts=True)
    return counts.astype(np.int64)


@pytest.mark.parametrize("ties", [False, True])
def test_efron_kernel_matches_numpy_reference(ties):
    X, E, weights, T = _make_cox_data(n=50, d=3, seed=1, ties=ties)
    beta = np.array([0.1, -0.2, 0.05])
    scores = weights * np.exp(X @ beta)
    counts = _counts(T)
    h, g, ll = azll._efron_kernel_jit(X, E, weights, scores, counts)
    rh, rg, rll = _efron_reference(X, E, weights, beta, T)
    np.testing.assert_allclose(g, rg, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(h, rh, rtol=1e-9, atol=1e-10)
    assert ll == pytest.approx(rll, rel=1e-9, abs=1e-10)


def test_efron_hessian_symmetric():
    X, E, weights, T = _make_cox_data(n=40, d=4, seed=2, ties=True)
    beta = np.zeros(4)
    scores = weights * np.exp(X @ beta)
    h, g, ll = azll._efron_kernel_jit(X, E, weights, scores, _counts(T))
    np.testing.assert_allclose(h, h.T, rtol=1e-10, atol=1e-12)


def test_efron_weighted_matches_reference():
    X, E, weights, T = _make_cox_data(n=30, d=2, seed=3, ties=True)
    rng = np.random.default_rng(99)
    weights = rng.uniform(0.5, 2.0, size=X.shape[0])
    beta = np.array([0.3, -0.1])
    scores = weights * np.exp(X @ beta)
    h, g, ll = azll._efron_kernel_jit(X, E, weights, scores, _counts(T))
    rh, rg, rll = _efron_reference(X, E, weights, beta, T)
    np.testing.assert_allclose(g, rg, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(h, rh, rtol=1e-8, atol=1e-9)
    assert ll == pytest.approx(rll, rel=1e-8, abs=1e-9)


# --------------------------------------------------------------------------
# _all_finite_frame / _all_finite_array
# --------------------------------------------------------------------------
def test_all_finite_frame_true_for_clean():
    df = pd.DataFrame(np.arange(12.0).reshape(4, 3))
    assert azll._all_finite_frame(df) is True


def test_all_finite_frame_false_for_nan_inf():
    df = pd.DataFrame(np.ones((5, 2)))
    df.iloc[3, 1] = np.nan
    assert azll._all_finite_frame(df) is False
    df2 = pd.DataFrame(np.ones((5, 2)))
    df2.iloc[0, 0] = np.inf
    assert azll._all_finite_frame(df2) is False


def test_all_finite_frame_block_boundary():
    # Force multiple blocks (block_rows small) with the bad value in a later
    # block, so the blocked scan must reach it.
    df = pd.DataFrame(np.ones((10, 2)))
    df.iloc[9, 0] = np.nan
    assert azll._all_finite_frame(df, block_rows=3) is False
    df_clean = pd.DataFrame(np.ones((10, 2)))
    assert azll._all_finite_frame(df_clean, block_rows=3) is True


def test_all_finite_array_series_and_ndarray():
    s = pd.Series([1.0, 2.0, 3.0])
    assert azll._all_finite_array(s) is True
    s_bad = pd.Series([1.0, np.nan, 3.0])
    assert azll._all_finite_array(s_bad) is False
    arr = np.array([1.0, 2.0, np.inf])
    assert azll._all_finite_array(arr) is False
    assert azll._all_finite_array(np.array([1.0, 2.0])) is True
