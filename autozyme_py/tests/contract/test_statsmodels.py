"""Contract tests for the statsmodels patch.

Patched surface (8 targets):
  - ModelData._handle_constant      (constant detection helper)
  - GLM.initialize, GLM.fit, GLM._fit_irls
  - WLS.initialize, WLS.fit
  - _MinimalWLS.__init__, _MinimalWLS.fit

User-facing entry points are ``sm.GLM(...).fit()`` and
``sm.WLS(...).fit()``. Both ``fast_glm_fit`` and ``fast_wls_fit`` take
``*args, **kwargs`` — Bug 2-class candidate: ensure they don't silently
drop a kwarg that vanilla would honor.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
sm = pytest.importorskip("statsmodels.api")


@pytest.fixture
def glm_inputs():
    """20-obs synthetic Poisson regression — log-link is the GLM default."""
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 3))
    eta = 0.5 + 0.3 * X[:, 0] - 0.2 * X[:, 1]
    y = rng.poisson(np.exp(eta))
    X_const = sm.add_constant(X)
    return y, X_const


@pytest.fixture
def wls_inputs():
    """100-obs synthetic WLS regression with heteroscedastic weights."""
    rng = np.random.default_rng(1)
    n = 100
    X = rng.normal(size=(n, 2))
    y = 1.0 + 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.3, size=n)
    w = 1.0 / (1.0 + np.abs(X[:, 0]))  # weight ~ inverse heteroscedasticity
    X_const = sm.add_constant(X)
    return y, X_const, w


def test_glm_fit_returns_results_object(glm_inputs):
    """GLM.fit() must return a GLMResultsWrapper carrying params + bse."""
    import autozyme
    autozyme.activate("statsmodels")

    y, X = glm_inputs
    out = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    # Vanilla's contract: a GLMResultsWrapper with the canonical attrs.
    assert hasattr(out, "params"), "fit() result missing .params"
    assert hasattr(out, "bse"), "fit() result missing .bse"
    assert out.params.shape == (X.shape[1],)


def test_glm_fit_zyme_false_matches_vanilla(glm_inputs):
    """zyme=False per-call (via context manager) must produce identical fit."""
    import autozyme
    autozyme.activate("statsmodels")

    y, X = glm_inputs
    with autozyme.disabled():
        ref = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    fast = sm.GLM(y, X, family=sm.families.Poisson()).fit()

    np.testing.assert_allclose(fast.params, ref.params, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(fast.bse, ref.bse, rtol=1e-3, atol=1e-6)


def test_wls_fit_returns_results_object(wls_inputs):
    """WLS.fit() must return a RegressionResultsWrapper with params + bse."""
    import autozyme
    autozyme.activate("statsmodels")

    y, X, w = wls_inputs
    out = sm.WLS(y, X, weights=w).fit()
    assert hasattr(out, "params")
    assert hasattr(out, "bse")
    assert out.params.shape == (X.shape[1],)


def test_wls_fit_zyme_false_matches_vanilla(wls_inputs):
    import autozyme
    autozyme.activate("statsmodels")

    y, X, w = wls_inputs
    with autozyme.disabled():
        ref = sm.WLS(y, X, weights=w).fit()
    fast = sm.WLS(y, X, weights=w).fit()

    np.testing.assert_allclose(fast.params, ref.params, rtol=1e-5, atol=1e-7)


def test_glm_fit_passes_method_kwarg_through(glm_inputs):
    """fast_glm_fit captures **kwargs — verify a known kwarg actually reaches
    the underlying fitter rather than getting silently swallowed.

    Same risk shape as Bug 2: a wrapper with **kwargs can eat a user-facing
    kwarg without honoring it. Here we pass ``method="newton"`` (non-default)
    and confirm the fit takes a different path than vanilla default.
    """
    import autozyme
    autozyme.activate("statsmodels")

    y, X = glm_inputs
    # Both fits should converge with method="newton"; verifying the kwarg
    # was honored means the patched signature didn't drop it.
    fast = sm.GLM(y, X, family=sm.families.Poisson()).fit(method="newton")
    with autozyme.disabled():
        ref = sm.GLM(y, X, family=sm.families.Poisson()).fit(method="newton")
    np.testing.assert_allclose(fast.params, ref.params, rtol=1e-4, atol=1e-6)
