"""Contract tests for the lifelines patch.

Patched surface (4 targets, all methods on SemiParametricPHFitter — the
engine ``CoxPHFitter`` dispatches to for the semiparametric Cox PH fit):
  - _get_efron_values_batch     (the heavy partial-likelihood inner loop)
  - _preprocess_dataframe       (input coercion)
  - _check_values_pre_fitting   (validation guard)
  - predict_log_partial_hazard  (post-fit prediction)

User-facing entry: ``cph.fit(df, duration_col, event_col)`` and
``cph.predict_log_partial_hazard(X)``. Contract pins:
  - .fit() returns the fitter; .summary populated
  - zyme=False matches the patched coefficient + SE numerically
  - predict_log_partial_hazard returns a Series of expected length
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
lifelines = pytest.importorskip("lifelines")


@pytest.fixture
def coxph_inputs():
    """200-subject synthetic survival data with 3 covariates.

    Generates a Cox PH model with known coefficients so the patch's fit
    has real signal to recover (vs noise-only fixtures where convergence
    paths can diverge between fast / vanilla).
    """
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    # True coefs: [0.5, -0.3, 0.1]. Generate Exp survival with hazard h_i.
    true_beta = np.array([0.5, -0.3, 0.1])
    hazard = np.exp(X @ true_beta)
    duration = rng.exponential(scale=1.0 / hazard)
    # ~30% censoring at duration > 1.5
    event = (duration < 1.5).astype(int)
    duration = np.minimum(duration, 1.5)

    df = pd.DataFrame(
        X, columns=["x1", "x2", "x3"]
    )
    df["duration"] = duration
    df["event"] = event
    return df


def test_cox_fit_returns_fitter_with_summary(coxph_inputs):
    """cph.fit() must return the fitter; .summary populated with 3 coefs."""
    import autozyme
    autozyme.activate("lifelines")
    from lifelines import CoxPHFitter

    cph = CoxPHFitter()
    out = cph.fit(coxph_inputs, duration_col="duration", event_col="event")
    assert out is cph, "fit() should return self for chaining"
    assert hasattr(cph, "summary"), ".summary missing after fit"
    assert cph.summary.shape[0] == 3  # 3 covariates


def test_cox_fit_zyme_false_matches_vanilla(coxph_inputs):
    """Patched coefficients + standard errors must match vanilla."""
    import autozyme
    autozyme.activate("lifelines")
    from lifelines import CoxPHFitter

    with autozyme.disabled():
        cph_v = CoxPHFitter().fit(
            coxph_inputs, duration_col="duration", event_col="event"
        )
    cph_f = CoxPHFitter().fit(
        coxph_inputs, duration_col="duration", event_col="event"
    )

    np.testing.assert_allclose(
        cph_f.params_.values, cph_v.params_.values,
        rtol=1e-4, atol=1e-6,
        err_msg="Cox coefficient drift between patched and vanilla",
    )
    np.testing.assert_allclose(
        cph_f.standard_errors_.values, cph_v.standard_errors_.values,
        rtol=1e-3, atol=1e-6,
        err_msg="Cox SE drift between patched and vanilla",
    )


def test_predict_log_partial_hazard_shape(coxph_inputs):
    """predict_log_partial_hazard returns a Series of length n_obs."""
    import autozyme
    autozyme.activate("lifelines")
    from lifelines import CoxPHFitter

    cph = CoxPHFitter().fit(
        coxph_inputs, duration_col="duration", event_col="event"
    )
    X_new = coxph_inputs[["x1", "x2", "x3"]].head(20)
    pred = cph.predict_log_partial_hazard(X_new)
    assert len(pred) == 20


def test_predict_log_partial_hazard_zyme_false_matches(coxph_inputs):
    """predict matches vanilla on identical fitted model + X."""
    import autozyme
    autozyme.activate("lifelines")
    from lifelines import CoxPHFitter

    df = coxph_inputs
    X_new = df[["x1", "x2", "x3"]].head(50)

    with autozyme.disabled():
        cph_v = CoxPHFitter().fit(df, duration_col="duration",
                                  event_col="event")
        pred_v = cph_v.predict_log_partial_hazard(X_new)
    cph_f = CoxPHFitter().fit(df, duration_col="duration",
                              event_col="event")
    pred_f = cph_f.predict_log_partial_hazard(X_new)

    np.testing.assert_allclose(pred_f.values, pred_v.values,
                               rtol=1e-4, atol=1e-6)
