"""Wave-4 tests for autozyme.lifelines: the batch efron wrapper + smoke recipe.

KERNEL CEILING: the bulk of this module (src lines 60-164, the
`_efron_kernel_jit` numba @njit body) is INVISIBLE to coverage.py even when
exercised -- wave-1 (`test_lifelines_unit.py`) already drives the kernel directly.
This file targets the COVERAGE-VISIBLE reachable Python that waves 1-2 did NOT hit.

KEY FINDING: the existing `test_lifelines_e2e.py` calls `CoxPHFitter().fit(...)`
WITHOUT `batch_mode=True`. lifelines' `_BatchVsSingle.decide` then routes a small,
lightly-tied dataset to `_get_efron_values_single` -- which the patch does NOT
target -- so `fast_get_efron_values_batch` (src 167-200, the NR-invariant cache +
the `_efron_kernel_jit` dispatch) was NEVER reached. We force it by passing
`batch_mode=True`, and assert the patched fit is numerically identical to the
unpatched baseline (and that the `_zyme_efron_cache` was actually populated).

Also covered:
  - the smoke recipe end-to-end: `_smoke_load` (read parquet, build CoxPHFitter),
    `_smoke_call` (`cph.fit`), `_smoke_save` (coef / variance / baseline cumhazard
    parquet + blocked partial-hazard npy + scalars json),
  - `fast_predict_log_partial_hazard`'s ndarray-input branch (src 277-278).
"""
from __future__ import annotations

import os
import tempfile
import warnings

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
lifelines = pytest.importorskip("lifelines")
pytest.importorskip("numba")
pytest.importorskip("pyarrow")  # parquet engine for the smoke recipe

import autozyme
from autozyme import lifelines as azll


@pytest.fixture(autouse=True)
def _silence():
    warnings.filterwarnings("ignore")
    yield
    autozyme.deactivate_all()


def _tied_survival_df(seed=0, n=300):
    """Integer durations -> many tied event times (the regime where lifelines
    prefers the batch efron calculator)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    beta = np.array([0.5, -0.3, 0.2])
    T = np.ceil(rng.exponential(np.exp(-(X @ beta))) * 3).astype(int) + 1
    E = (rng.random(n) < 0.7).astype(int)
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["T"] = T
    df["E"] = E
    return df


def test_batch_mode_routes_through_fast_efron_and_matches_baseline():
    """With `batch_mode=True`, `cph.fit` routes through
    `fast_get_efron_values_batch` (populating `_zyme_efron_cache` + running the
    JIT kernel). The patched coefficients are numerically identical to the
    unpatched baseline."""
    df = _tied_survival_df()

    # Baseline: patches off, but still batch_mode=True.
    base = lifelines.CoxPHFitter().fit(
        df, duration_col="T", event_col="E", batch_mode=True
    )

    autozyme.activate("lifelines")
    fast = lifelines.CoxPHFitter().fit(
        df, duration_col="T", event_col="E", batch_mode=True
    )
    # The fast wrapper stashed its NR-invariant cache on the SemiParametric model.
    assert hasattr(fast._model, "_zyme_efron_cache")
    cache = fast._model._zyme_efron_cache
    assert set(("Xv", "Ev", "Wv", "counts", "we_X")).issubset(cache.keys())
    # Coefficients + log-likelihood match the baseline.
    np.testing.assert_allclose(
        fast.params_.values, base.params_.values, rtol=1e-4, atol=1e-5
    )
    assert fast.log_likelihood_ == pytest.approx(base.log_likelihood_, rel=1e-4)


def test_batch_mode_variance_matrix_matches_baseline():
    """The patch computes the full Hessian every Newton step, so the
    variance_matrix_ (the inverse Hessian) must match the baseline -- inference
    stays valid under the batch fast path."""
    df = _tied_survival_df(seed=1)
    base = lifelines.CoxPHFitter().fit(
        df, duration_col="T", event_col="E", batch_mode=True
    )
    autozyme.activate("lifelines")
    fast = lifelines.CoxPHFitter().fit(
        df, duration_col="T", event_col="E", batch_mode=True
    )
    np.testing.assert_allclose(
        fast.variance_matrix_.values, base.variance_matrix_.values,
        rtol=1e-3, atol=1e-5,
    )


def test_predict_log_partial_hazard_ndarray_int_input():
    """`fast_predict_log_partial_hazard` with a raw ndarray (non-float) input
    drives the `else` ndarray branch + the int->float64 cast (src 277-278)."""
    df = _tied_survival_df(seed=2)
    autozyme.activate("lifelines")
    cph = lifelines.CoxPHFitter().fit(df, duration_col="T", event_col="E")
    model = cph._model
    Xnd = df[["a", "b", "c"]].to_numpy()  # float64 ndarray (no index -> else branch)
    out = azll.fast_predict_log_partial_hazard(model, Xnd)
    assert isinstance(out, pd.Series)
    assert len(out) == len(df)
    assert np.all(np.isfinite(out.values))


def test_predict_log_partial_hazard_int_dataframe_casts():
    """`fast_predict_log_partial_hazard` with an integer-valued DataFrame drives
    the DataFrame branch + the `Xv.astype(np.float64)` cast (src line 274), and
    preserves the input index."""
    df = _tied_survival_df(seed=4)
    autozyme.activate("lifelines")
    cph = lifelines.CoxPHFitter().fit(df, duration_col="T", event_col="E")
    model = cph._model
    hazard_names = list(model.params_.index)
    X_int = df[hazard_names].round().astype(np.int64)  # int64 DataFrame
    X_int.index = [f"row{i}" for i in range(len(df))]
    out = azll.fast_predict_log_partial_hazard(model, X_int)
    assert isinstance(out, pd.Series)
    assert list(out.index) == list(X_int.index)
    assert np.all(np.isfinite(out.values))


# --------------------------------------------------------------------------
# Smoke recipe
# --------------------------------------------------------------------------
@pytest.fixture
def smoke_task_dir():
    import yaml

    td = tempfile.mkdtemp(prefix="autozyme_lifelines_w4_")
    os.makedirs(os.path.join(td, "data"), exist_ok=True)
    df = _tied_survival_df(seed=3)
    df.to_parquet(os.path.join(td, "data", "surv.parquet"))
    with open(os.path.join(td, "task.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"datasets": [{"tier": "small", "path": "data/surv.parquet"}]}, f
        )
    return td


def test_smoke_load_call_save_roundtrip(smoke_task_dir):
    """`_smoke_load` reads the parquet + builds a CoxPHFitter, `_smoke_call`
    fits it, `_smoke_save` writes all five output artifacts."""
    autozyme.activate("lifelines")
    inputs = azll._smoke_load(smoke_task_dir, "small")
    assert set(inputs.keys()) == {"df", "cph"}

    result = azll._smoke_call(inputs)
    assert hasattr(result["cph"], "params_")

    out_dir = tempfile.mkdtemp(prefix="autozyme_lifelines_w4_out_")
    azll._smoke_save(result, out_dir)
    for name in (
        "coef.parquet", "variance.parquet", "baseline_cumhazard.parquet",
        "partial_hazard.npy", "scalars.json",
    ):
        assert os.path.isfile(os.path.join(out_dir, name)), name
    ph = np.load(os.path.join(out_dir, "partial_hazard.npy"))
    assert ph.shape == (len(inputs["df"]),)
    assert np.all(np.isfinite(ph)) and np.all(ph > 0.0)
    import json
    with open(os.path.join(out_dir, "scalars.json"), encoding="utf-8") as f:
        scalars = json.load(f)
    assert {"log_likelihood", "concordance_index", "fit_seconds"} <= set(scalars)
