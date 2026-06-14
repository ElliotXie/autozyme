"""Wave-4 wrapper/dispatch/gating-line tests for autozyme.statsmodels.

Wave-1 (`_unit`) covered the pure helpers (`_all_ones`, `_normalized_cov...`,
`_is_poisson_log_model`, `fast_handle_constant` basics) and the default-True
path of `_can_fast_poisson_irls`. The existing `test_statsmodels.py` drives
GLM/WLS.fit end to end. The COVERAGE-VISIBLE lines still missing are:

  - every individual ``return False`` gating branch in `_can_fast_poisson_irls`
    (attach_wls / wls_method / tol_criterion / rtol / _offset_exposure /
    freq_weights / var_weights / iweights / n_trials / start_params-exception),
    each on its own line (150,152,154,156,158,160,162,164,166,171-172).
  - `fast_handle_constant`'s non-finite delegation path (lines 188-189).
  - `fast_glm_initialize`'s ``hasattr(self,'family') and not can_fast`` upstream
    delegate (208) and the freq_weights branch (215-216).
  - `fast_glm_fit`'s ``zyme=False`` short-circuit (224).
  - `fast_wls_fit` / `fast_wls_initialize` state-None upstream branches
    (316, 326-329).
  - the smoke recipe (468-494) via a tiny synthetic Poisson npz.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
sm = pytest.importorskip("statsmodels.api")
yaml = pytest.importorskip("yaml")

import autozyme
from autozyme import statsmodels as azsm


def _poisson_model(n=40, p=2, seed=1):
    rng = np.random.default_rng(seed)
    X = sm.add_constant(rng.standard_normal((n, p)))
    y = rng.poisson(2.0, size=n).astype(float)
    return sm.GLM(y, X, family=sm.families.Poisson())


# --------------------------------------------------------------------------
# _can_fast_poisson_irls — every individual `return False` gating branch
# --------------------------------------------------------------------------
def test_can_fast_rejects_attach_wls():
    m = _poisson_model()
    assert azsm._can_fast_poisson_irls(m, kwargs={"attach_wls": True}) is False


def test_can_fast_rejects_non_lstsq_wls_method():
    m = _poisson_model()
    assert azsm._can_fast_poisson_irls(m, kwargs={"wls_method": "pinv"}) is False


def test_can_fast_rejects_non_deviance_tol_criterion():
    m = _poisson_model()
    assert azsm._can_fast_poisson_irls(
        m, kwargs={"tol_criterion": "params"}
    ) is False


def test_can_fast_rejects_nonzero_rtol():
    m = _poisson_model()
    assert azsm._can_fast_poisson_irls(m, kwargs={"rtol": 1e-3}) is False


def test_can_fast_rejects_offset_exposure():
    m = _poisson_model()
    # A non-zero offset/exposure makes the lean IRLS unsafe.
    m._offset_exposure = np.ones(m.endog.shape[0])
    assert azsm._can_fast_poisson_irls(m) is False


def test_can_fast_rejects_nonunit_freq_weights():
    m = _poisson_model()
    m.freq_weights = np.full(m.endog.shape[0], 2.0)
    assert azsm._can_fast_poisson_irls(m) is False


def test_can_fast_rejects_nonunit_var_weights():
    m = _poisson_model()
    m.var_weights = np.full(m.endog.shape[0], 3.0)
    assert azsm._can_fast_poisson_irls(m) is False


def test_can_fast_rejects_nonunit_iweights():
    m = _poisson_model()
    m.iweights = np.full(m.endog.shape[0], 2.0)
    assert azsm._can_fast_poisson_irls(m) is False


def test_can_fast_rejects_nonunit_n_trials():
    m = _poisson_model()
    m.n_trials = np.full(m.endog.shape[0], 5.0)
    assert azsm._can_fast_poisson_irls(m) is False


def test_can_fast_rejects_start_params_exception():
    """A start_params object whose .shape[0] access raises -> the except branch
    returns False (lines 168-172)."""
    m = _poisson_model()

    class _NoShape:
        # np.asarray(...) -> 0-d object array; .shape[0] raises IndexError
        pass

    assert azsm._can_fast_poisson_irls(m, start_params=_NoShape()) is False


# --------------------------------------------------------------------------
# fast_handle_constant — non-finite exog delegates to upstream (188-189)
# --------------------------------------------------------------------------
def test_handle_constant_non_numeric_exog_typeerror_delegates():
    """An object-dtype exog where np.isfinite raises TypeError must delegate to
    the captured upstream _handle_constant (lines 188-189)."""
    exog = np.array([["a", "b"], ["c", "d"]], dtype=object)
    self = SimpleNamespace(exog=exog, k_constant=None, const_idx=None)
    # Upstream is reached; it may raise or set attrs, but the key contract is
    # that the fast intercept-detection short-circuit did NOT fire.
    try:
        azsm.fast_handle_constant(self, True)
    except Exception:
        pass
    # The fast path would have set const_idx=0; delegation must not.
    assert self.const_idx != 0 or self.k_constant != 1


def test_handle_constant_nonfinite_intercept_delegates():
    """A column-0 that is all-ones but the matrix has a NaN elsewhere: the
    finite-check fails so the fast all-ones short-circuit is skipped and the
    call delegates to upstream (covers the `finite_exog` False edge of 190).
    Upstream raises MissingDataError on inf/nan exog — the fast fn must reproduce
    that exact behavior (i.e. it delegated rather than mis-tagging the column)."""
    from statsmodels.tools.sm_exceptions import MissingDataError

    exog = np.column_stack([np.ones(5), np.arange(5.0)])
    exog[2, 1] = np.nan
    fast_self = SimpleNamespace(exog=exog, k_constant=None, const_idx=None)
    with pytest.raises(MissingDataError):
        azsm.fast_handle_constant(fast_self, True)
    # And the captured upstream raises the same error on an identical stub.
    ref_self = SimpleNamespace(exog=exog.copy(), k_constant=None, const_idx=None)
    with pytest.raises(MissingDataError):
        azsm._orig_handle_constant(ref_self, True)


# --------------------------------------------------------------------------
# fast_glm_initialize — upstream delegate + freq_weights branch (208, 215-216)
# --------------------------------------------------------------------------
def test_glm_initialize_non_poisson_delegates(glm_gaussian=None):
    """A Gaussian GLM with a `family` attribute is not fast-eligible, so
    fast_glm_initialize delegates to upstream (line 208). We compare df_model /
    df_resid against an independent vanilla GLM."""
    autozyme.activate("statsmodels")
    rng = np.random.default_rng(5)
    X = sm.add_constant(rng.standard_normal((30, 2)))
    y = rng.standard_normal(30)
    fast = sm.GLM(y, X, family=sm.families.Gaussian())
    with autozyme.disabled():
        ref = sm.GLM(y, X, family=sm.families.Gaussian())
    assert fast.df_model == ref.df_model
    assert fast.df_resid == ref.df_resid


def test_glm_initialize_freq_weights_branch():
    """fast_glm_initialize with freq_weights matching nobs computes wnobs from
    the weight sum (lines 212-216)."""
    autozyme.activate("statsmodels")
    rng = np.random.default_rng(6)
    n = 40
    X = sm.add_constant(rng.standard_normal((n, 2)))
    y = rng.poisson(2.0, size=n).astype(float)
    fw = np.full(n, 2.0)
    # Poisson + freq_weights -> _can_fast_poisson_irls is False (non-unit
    # freq_weights), so initialize() delegates to upstream; but constructing
    # with freq_weights still exercises the patched initialize during __init__.
    model = sm.GLM(y, X, family=sm.families.Poisson(), freq_weights=fw)
    with autozyme.disabled():
        ref = sm.GLM(y, X, family=sm.families.Poisson(), freq_weights=fw)
    assert model.df_model == ref.df_model
    np.testing.assert_allclose(model.df_resid, ref.df_resid)


# --------------------------------------------------------------------------
# fast_glm_fit — zyme=False short-circuit (line 224)
# --------------------------------------------------------------------------
def test_glm_fit_zyme_false_kwarg_uses_upstream():
    """Passing zyme=False directly to GLM.fit pops the flag and calls the
    captured upstream fit unchanged (line 224)."""
    autozyme.activate("statsmodels")
    rng = np.random.default_rng(7)
    n = 60
    X = sm.add_constant(rng.standard_normal((n, 2)))
    y = rng.poisson(np.exp(0.3 + 0.2 * X[:, 1]))
    fast_disabled = sm.GLM(y, X, family=sm.families.Poisson()).fit(zyme=False)
    with autozyme.disabled():
        ref = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    np.testing.assert_allclose(fast_disabled.params, ref.params,
                               rtol=1e-6, atol=1e-8)


# --------------------------------------------------------------------------
# fast_wls_fit / fast_wls_initialize — state-None upstream branches
# --------------------------------------------------------------------------
def test_wls_fit_outside_poisson_state_uses_upstream():
    """A plain WLS.fit() with no active Poisson-IRLS context (state is None)
    must take the upstream branch (line 316) and match vanilla."""
    autozyme.activate("statsmodels")
    rng = np.random.default_rng(8)
    n = 80
    X = sm.add_constant(rng.standard_normal((n, 2)))
    y = 1.0 + 2.0 * X[:, 1] + rng.normal(scale=0.3, size=n)
    w = 1.0 / (1.0 + np.abs(X[:, 1]))
    out = sm.WLS(y, X, weights=w).fit()
    with autozyme.disabled():
        ref = sm.WLS(y, X, weights=w).fit()
    np.testing.assert_allclose(out.params, ref.params, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(out.normalized_cov_params,
                               ref.normalized_cov_params, rtol=1e-6, atol=1e-8)


def test_wls_initialize_outside_state_uses_upstream():
    """fast_wls_initialize with state None delegates to upstream (lines
    326-329); the WLS object built outside any Poisson context still has the
    upstream-initialized whitened design."""
    autozyme.activate("statsmodels")
    rng = np.random.default_rng(9)
    n = 50
    X = sm.add_constant(rng.standard_normal((n, 2)))
    y = rng.standard_normal(n)
    w = np.full(n, 0.5)
    model = sm.WLS(y, X, weights=w)
    # Upstream initialize populates wexog / wendog from sqrt(w)*X.
    assert hasattr(model, "wexog")
    np.testing.assert_allclose(model.wexog, np.sqrt(w)[:, None] * np.asarray(X),
                               rtol=1e-10, atol=1e-12)


def test_wls_fit_inside_poisson_state_returns_cached():
    """Inside a fast-Poisson-IRLS contextvar scope, fast_wls_fit returns the
    cached params + normalized_cov from the scratch state (line 311-319), and
    fast_wls_initialize sets the cheap nobs/df shortcut (322-329). We set the
    token directly to exercise the in-context dispatch without rebuilding the
    whole IRLS loop."""
    rng = np.random.default_rng(11)
    n = 30
    X = sm.add_constant(rng.standard_normal((n, 2)))
    y = rng.standard_normal(n)
    w = np.full(n, 0.5)
    model = sm.WLS(y, X, weights=w)

    state = azsm._new_fast_poisson_state()
    params = np.array([0.5, -0.2, 0.1])
    XtX = np.asarray(X).T @ np.asarray(X)
    state["params"] = params
    state["XtX"] = XtX
    token = azsm._fast_poisson_irls_state.set(state)
    try:
        # state-not-None branch of fast_wls_fit (line 316).
        res = azsm.fast_wls_fit(model, "pinv")
        np.testing.assert_array_equal(res.params, params)
        np.testing.assert_allclose(res.normalized_cov_params,
                                   np.linalg.inv(XtX), rtol=1e-9)
        # state-not-None branch of fast_wls_initialize (lines 322-329).
        azsm.fast_wls_initialize(model)
        assert model.nobs == float(n)
        assert model._df_model is None and model.rank is None
    finally:
        azsm._fast_poisson_irls_state.reset(token)


# --------------------------------------------------------------------------
# fast_glm_initialize — explicit post-construction call delegates (line 208)
# --------------------------------------------------------------------------
def test_glm_initialize_explicit_call_non_poisson_delegates():
    """Calling .initialize() AFTER construction on a non-Poisson model (family
    now set, _can_fast_poisson_irls False) takes the upstream delegate at line
    208. df_model/df_resid must match a vanilla re-initialize."""
    autozyme.activate("statsmodels")
    rng = np.random.default_rng(12)
    X = sm.add_constant(rng.standard_normal((40, 2)))
    y = rng.standard_normal(40)
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    # Re-initialize post-construction: now self.family exists -> line 207 True
    # -> delegate to upstream (208).
    model.initialize()
    with autozyme.disabled():
        ref = sm.GLM(y, X, family=sm.families.Gaussian())
        ref.initialize()
    assert model.df_model == ref.df_model
    assert model.df_resid == ref.df_resid


# --------------------------------------------------------------------------
# fast IRLS impl — start_params branch (lines 405-407)
# --------------------------------------------------------------------------
def test_glm_fit_with_start_params_uses_start_branch():
    """A fast Poisson GLM fit with a valid start_params vector enters the
    ``else`` start-params branch of the IRLS impl (lines 405-407: lin_pred =
    X @ start_params). Result matches a vanilla fit from the same start."""
    autozyme.activate("statsmodels")
    rng = np.random.default_rng(13)
    n = 120
    X = sm.add_constant(rng.standard_normal((n, 2)))
    y = rng.poisson(np.exp(0.3 + 0.25 * X[:, 1] - 0.1 * X[:, 2]))
    start = np.array([0.2, 0.1, -0.05])
    fast = sm.GLM(y, X, family=sm.families.Poisson()).fit(start_params=start)
    with autozyme.disabled():
        ref = sm.GLM(y, X, family=sm.families.Poisson()).fit(start_params=start)
    np.testing.assert_allclose(fast.params, ref.params, rtol=1e-4, atol=1e-6)


# --------------------------------------------------------------------------
# smoke recipe (468-494)
# --------------------------------------------------------------------------
def _make_task_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    n = 200
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    beta = np.array([0.4, 0.3, -0.2])
    y = rng.poisson(np.exp(X @ beta)).astype(np.int64)
    np.savez(data_dir / "small.npz", X=X.astype(np.float64), y=y)
    task = {"datasets": [{"tier": "small", "name": "synth",
                          "path": "./data/small.npz"}]}
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    return str(tmp_path)


def test_smoke_load_returns_X_y(tmp_path):
    """_smoke_load reads the npz into fortran-ordered X and y (lines 468-480)."""
    task_dir = _make_task_dir(tmp_path)
    out = azsm._smoke_load(task_dir, "small")
    assert out["X"].shape[0] == out["y"].shape[0] == 200
    assert out["X"].flags["F_CONTIGUOUS"]


def test_smoke_call_and_save_roundtrip(tmp_path):
    """_smoke_call fits a Poisson GLM; _smoke_save writes params/scale/llf/
    n_iter/converged (covers 483-494)."""
    autozyme.activate("statsmodels")
    task_dir = _make_task_dir(tmp_path)
    inputs = azsm._smoke_load(task_dir, "small")
    result = azsm._smoke_call(inputs)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    azsm._smoke_save(result, str(out_dir))
    saved = np.load(out_dir / "fit.npz")
    assert set(saved.files) == {"params", "scale", "llf", "n_iter", "converged"}
    assert saved["params"].shape == (3,)
    # Patched smoke fit should match a vanilla GLM fit on the same data.
    with autozyme.disabled():
        ref = sm.GLM(inputs["y"], inputs["X"],
                     family=sm.families.Poisson()).fit()
    np.testing.assert_allclose(saved["params"], ref.params,
                               rtol=1e-4, atol=1e-6)
