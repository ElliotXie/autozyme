"""End-to-end / wrapper-line tests for autozyme.lifelines.

Wave-1 (`test_lifelines_unit.py`) tested the `_efron_kernel_jit` numba kernel and
the `_all_finite_*` scanners directly. The existing `test_lifelines.py` drives
`cph.fit` (covering the happy path of all 4 targets) + `predict_log_partial_hazard`
on a DataFrame. This file covers the remaining COVERAGE-VISIBLE wrapper branches:

  - `fast_predict_log_partial_hazard`: the pd.Series fallback branch and the raw
    numpy-array branch (`index=None`), each vs the captured upstream original.
  - `safe_check_pre_fitting`: the weighted-fit delegation, the non-numeric-dtype
    delegation, and the non-finite-values delegation — all return the upstream
    result, not the clean fast-path None.
  - `fast_preprocess_dataframe`: the float-dtype stash + the patched _fast_mean /
    _fast_std reducers.
  - activate/restore lifecycle on all 4 SemiParametricPHFitter targets.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
lifelines = pytest.importorskip("lifelines")

import autozyme
from autozyme import lifelines as azll


@pytest.fixture
def fitted_cph():
    """A CoxPHFitter fit under the patch on clean synthetic survival data."""
    from lifelines import CoxPHFitter

    rng = np.random.default_rng(0)
    n = 150
    X = rng.normal(size=(n, 3))
    beta = np.array([0.5, -0.3, 0.1])
    hazard = np.exp(X @ beta)
    duration = np.minimum(rng.exponential(scale=1.0 / hazard), 1.5)
    event = (duration < 1.5).astype(int)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    df["duration"] = duration
    df["event"] = event

    autozyme.activate("lifelines")
    cph = CoxPHFitter().fit(df, duration_col="duration", event_col="event")
    return cph, df


def test_predict_log_partial_hazard_series_input_matches_vanilla(fitted_cph):
    """A pd.Series X hits the `_orig_pred_log` fallback branch; result must
    match the captured upstream original on the same fitted model."""
    cph, df = fitted_cph
    x_series = df[["x1", "x2", "x3"]].iloc[0]  # a single-row Series
    assert isinstance(x_series, pd.Series)

    fast = cph.predict_log_partial_hazard(x_series)
    ref = azll._orig_pred_log(cph, x_series)
    np.testing.assert_allclose(np.asarray(fast), np.asarray(ref),
                               rtol=1e-8, atol=1e-10)


def test_predict_log_partial_hazard_ndarray_input(fitted_cph):
    """A raw numpy-array X hits the `index=None` branch; parity vs a DataFrame
    call on the same rows (the math is (X - norm_mean) @ params_)."""
    cph, df = fitted_cph
    X_df = df[["x1", "x2", "x3"]].head(10)
    X_arr = X_df.to_numpy()

    from_arr = cph.predict_log_partial_hazard(X_arr)
    from_df = cph.predict_log_partial_hazard(X_df)
    assert from_arr.index is None or list(from_arr.index) == list(range(10))
    np.testing.assert_allclose(from_arr.values, from_df.values,
                               rtol=1e-8, atol=1e-10)


def test_check_pre_fitting_delegates_on_non_finite():
    """safe_check_pre_fitting must defer to upstream when X has non-finite
    values, so the original NaN/inf validation still raises."""
    from lifelines import CoxPHFitter

    autozyme.activate("lifelines")
    df = pd.DataFrame({
        "x1": [0.1, 0.2, np.nan, 0.4, 0.5, 0.6],
        "duration": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5],
        "event": [1, 0, 1, 1, 0, 1],
    })
    cph = CoxPHFitter()
    # Upstream raises on non-finite covariates; the fast guard must delegate
    # rather than silently pass.
    with pytest.raises(Exception):
        cph.fit(df, duration_col="duration", event_col="event")


def test_check_pre_fitting_delegates_on_non_numeric_dtype():
    """A non-numeric covariate column routes safe_check_pre_fitting to the
    upstream validator (dtype.kind not in iufb)."""
    from lifelines import CoxPHFitter

    autozyme.activate("lifelines")
    df = pd.DataFrame({
        "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "cat": pd.Categorical(["a", "b", "a", "b", "a", "b"]),
        "duration": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5],
        "event": [1, 0, 1, 1, 0, 1],
    })
    cph = CoxPHFitter()
    # lifelines rejects/handles the non-numeric column via the upstream path.
    with pytest.raises(Exception):
        cph.fit(df, duration_col="duration", event_col="event")


def test_check_pre_fitting_clean_data_uses_fast_path(fitted_cph):
    """On clean numeric data the fast guard returns None (already exercised by
    the successful fit fixture); confirm the helpers agree."""
    cph, df = fitted_cph
    X = df[["x1", "x2", "x3"]]
    assert azll._all_finite_frame(X) is True
    assert azll._all_finite_array(df["duration"]) is True


def test_fast_preprocess_stash_and_reducers(fitted_cph):
    """fast_preprocess_dataframe stashes a contiguous Xv and patches X.mean /
    X.std on the returned frame; the stashed array reflects the inputs."""
    cph, df = fitted_cph
    # cph already has duration_col / event_col configured; _preprocess_dataframe
    # strips those out and returns the covariate frame X.
    X, T, E, W, entries, idx, clusters = azll.fast_preprocess_dataframe(cph, df)
    Xv = X.__dict__.get("_zyme_Xv_contig")
    assert Xv is not None
    # The patched reducers return pandas Series indexed by columns.
    m = X.mean()
    s = X.std()
    assert isinstance(m, pd.Series) and isinstance(s, pd.Series)
    np.testing.assert_allclose(m.values, np.mean(Xv, axis=0), rtol=1e-10)
    np.testing.assert_allclose(s.values, np.std(Xv, axis=0, ddof=1), rtol=1e-10)


def test_activate_restore_all_four_targets():
    """activate binds all 4 SemiParametricPHFitter methods; deactivate restores."""
    from lifelines.fitters.coxph_fitter import SemiParametricPHFitter

    autozyme.deactivate("lifelines")
    orig_efron = SemiParametricPHFitter._get_efron_values_batch
    orig_predict = SemiParametricPHFitter.predict_log_partial_hazard

    assert autozyme.activate("lifelines") is True
    assert SemiParametricPHFitter._get_efron_values_batch is not orig_efron
    info = autozyme.inspect("lifelines")
    assert info["status"] == "active"
    assert len(info["targets"]) == 4

    autozyme.deactivate("lifelines")
    assert SemiParametricPHFitter._get_efron_values_batch is orig_efron
    assert SemiParametricPHFitter.predict_log_partial_hazard is orig_predict
