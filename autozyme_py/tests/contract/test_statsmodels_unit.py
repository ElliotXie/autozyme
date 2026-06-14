"""Unit tests for the pure helpers in autozyme.statsmodels.

The contract test (test_statsmodels.py) drives GLM.fit end to end. Here we test
the self-contained logic helpers directly:

  - _all_ones / _all_zeros          predicate helpers
  - _new_fast_poisson_state         scratch-state factory
  - _normalized_cov_from_state      inv/pinv of cached X'WX
  - _is_poisson_log_model / _can_fast_poisson_irls   model gating (statsmodels)
  - fast_handle_constant            intercept-column detection (via a fake self)

statsmodels must import for the module to load.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
sm = pytest.importorskip("statsmodels.api")

from autozyme import statsmodels as azsm


# --------------------------------------------------------------------------
# _all_ones / _all_zeros
# --------------------------------------------------------------------------
def test_all_ones():
    assert azsm._all_ones(np.ones(5)) is True
    assert azsm._all_ones([1, 1, 1]) is True
    assert azsm._all_ones(np.array([1, 1, 0])) is False
    assert azsm._all_ones(1.0) is True
    # empty -> False (explicitly guarded in the helper)
    assert azsm._all_ones(np.array([])) is False


def test_all_zeros():
    assert azsm._all_zeros(np.zeros(4)) is True
    assert azsm._all_zeros(0.0) is True
    assert azsm._all_zeros(np.array([0, 0, 1])) is False
    assert azsm._all_zeros(np.array([])) is False


# --------------------------------------------------------------------------
# _new_fast_poisson_state
# --------------------------------------------------------------------------
def test_new_fast_poisson_state_keys_all_none():
    st = azsm._new_fast_poisson_state()
    assert set(st) == {
        "XtX", "params", "normalized_cov_params", "wexog", "wendog", "w_half"
    }
    assert all(v is None for v in st.values())


# --------------------------------------------------------------------------
# _normalized_cov_from_state
# --------------------------------------------------------------------------
def test_normalized_cov_from_state_inverts_xtx():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    XtX = A @ A.T + 5 * np.eye(5)  # SPD
    state = azsm._new_fast_poisson_state()
    state["XtX"] = XtX
    cov = azsm._normalized_cov_from_state(state)
    np.testing.assert_allclose(cov, np.linalg.inv(XtX), rtol=1e-9, atol=1e-11)
    # second call returns the cached value (same object).
    assert azsm._normalized_cov_from_state(state) is cov


def test_normalized_cov_from_state_no_xtx_returns_none():
    state = azsm._new_fast_poisson_state()
    assert azsm._normalized_cov_from_state(state) is None


def test_normalized_cov_from_state_singular_uses_pinv():
    XtX = np.zeros((3, 3))  # singular -> pinv path
    state = azsm._new_fast_poisson_state()
    state["XtX"] = XtX
    cov = azsm._normalized_cov_from_state(state)
    np.testing.assert_allclose(cov, np.linalg.pinv(XtX), atol=1e-12)


# --------------------------------------------------------------------------
# _is_poisson_log_model / _can_fast_poisson_irls
# --------------------------------------------------------------------------
def _poisson_model():
    rng = np.random.default_rng(1)
    X = sm.add_constant(rng.standard_normal((50, 2)))
    y = rng.poisson(2.0, size=50).astype(float)
    return sm.GLM(y, X, family=sm.families.Poisson())


def test_is_poisson_log_model():
    assert azsm._is_poisson_log_model(_poisson_model()) is True
    rng = np.random.default_rng(2)
    X = sm.add_constant(rng.standard_normal((30, 2)))
    y = rng.standard_normal(30)
    gauss = sm.GLM(y, X, family=sm.families.Gaussian())
    assert azsm._is_poisson_log_model(gauss) is False


def test_can_fast_poisson_irls_default_true():
    assert azsm._can_fast_poisson_irls(_poisson_model()) is True


def test_can_fast_poisson_irls_rejects_robust_cov():
    m = _poisson_model()
    assert azsm._can_fast_poisson_irls(m, cov_type="HC0") is False
    assert azsm._can_fast_poisson_irls(m, scale="X2") is False


def test_can_fast_poisson_irls_rejects_non_poisson():
    rng = np.random.default_rng(3)
    X = sm.add_constant(rng.standard_normal((20, 2)))
    y = rng.standard_normal(20)
    gauss = sm.GLM(y, X, family=sm.families.Gaussian())
    assert azsm._can_fast_poisson_irls(gauss) is False


def test_can_fast_poisson_irls_bad_start_params_shape():
    m = _poisson_model()
    # exog has 3 cols (const + 2); a wrong-length start vector is rejected.
    assert azsm._can_fast_poisson_irls(m, start_params=np.zeros(7)) is False
    assert azsm._can_fast_poisson_irls(m, start_params=np.zeros(3)) is True


# --------------------------------------------------------------------------
# fast_handle_constant  (operates on self.exog / self.k_constant)
# --------------------------------------------------------------------------
def _data_stub(exog):
    return SimpleNamespace(exog=exog, k_constant=None, const_idx=None)


def test_handle_constant_hasconst_false():
    self = _data_stub(np.ones((4, 2)))
    azsm.fast_handle_constant(self, False)
    assert self.k_constant == 0
    assert self.const_idx is None


def test_handle_constant_detects_intercept_column():
    exog = np.column_stack([np.ones(6), np.arange(6.0)])
    self = _data_stub(exog)
    azsm.fast_handle_constant(self, True)
    assert self.k_constant == 1
    assert self.const_idx == 0


def test_handle_constant_no_intercept_column_delegates():
    # Column 0 is not all-ones -> the fast intercept-detection path declines and
    # delegates to upstream. Result must match calling the captured upstream
    # _handle_constant on an identical stub (that's the delegation contract).
    exog = np.column_stack([np.arange(1.0, 6.0), np.arange(5.0)])
    fast_self = _data_stub(exog)
    azsm.fast_handle_constant(fast_self, True)

    ref_self = _data_stub(exog.copy())
    azsm._orig_handle_constant(ref_self, True)

    assert fast_self.k_constant == ref_self.k_constant
    assert fast_self.const_idx == ref_self.const_idx
    # And critically: the fast path did NOT mis-tag column 0 as the intercept.
    assert fast_self.const_idx != 0 or fast_self.k_constant == 0


def test_handle_constant_none_exog():
    self = _data_stub(None)
    azsm.fast_handle_constant(self, True)
    assert self.k_constant == 0
    assert self.const_idx is None
