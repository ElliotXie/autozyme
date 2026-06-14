"""Unit tests for the pure helpers in autozyme.sccoda.

The contract test (test_sccoda.py) drives sample_hmc end to end through TFP.
sccoda is almost entirely TensorFlow/TFP-coupled; the one self-contained,
numpy-computable piece is get_y_hat (the posterior-mean composition). We test it
through a lightweight fake `self`, plus the contextvar leapfrog-step helper.

sccoda + tensorflow must import for the module to load (heavy but present here).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
# tensorflow is imported at sccoda module import; skip cleanly if absent.
tf = pytest.importorskip("tensorflow")
pytest.importorskip("sccoda")

from autozyme import sccoda as azsccoda


# --------------------------------------------------------------------------
# _scoped_num_leapfrog_steps  (contextvar parsing)
# --------------------------------------------------------------------------
def test_scoped_num_leapfrog_steps_default_none():
    # Default contextvar value is None.
    assert azsccoda._scoped_num_leapfrog_steps() is None


def test_scoped_num_leapfrog_steps_reads_int():
    token = azsccoda._sccoda_leapfrog_steps.set(10)
    try:
        assert azsccoda._scoped_num_leapfrog_steps() == 10
    finally:
        azsccoda._sccoda_leapfrog_steps.reset(token)


def test_scoped_num_leapfrog_steps_garbage_returns_none():
    token = azsccoda._sccoda_leapfrog_steps.set("not-int")
    try:
        assert azsccoda._scoped_num_leapfrog_steps() is None
    finally:
        azsccoda._sccoda_leapfrog_steps.reset(token)


# --------------------------------------------------------------------------
# fast_get_y_hat  (posterior-mean composition; numpy math)
# --------------------------------------------------------------------------
def _make_self(N, K, ref_idx, seed=0):
    """Minimal scCODAModel stand-in exposing only what fast_get_y_hat reads."""
    rng = np.random.default_rng(seed)
    D = 2  # number of covariate columns in x
    self = SimpleNamespace()
    self.N = N
    self.K = K
    self.reference_cell_type = ref_idx
    self.x = rng.standard_normal((N, D)).astype(np.float64)
    n_total = rng.integers(500, 2000, size=N).astype(np.float64)
    self.n_total = tf.constant(n_total)
    self._D = D
    self._rng = rng
    return self


def _states(self, num_results, num_burnin, seed=1):
    """Build a states_burnin list with the shapes fast_get_y_hat expects.

    Indices: [0]=sigma_d, [1]=b_offset, [2]=ind_raw, [3]=alphas.
    chain length = num_results - num_burnin.
    """
    rng = np.random.default_rng(seed)
    chain = num_results - num_burnin
    D, K = self._D, self.K
    Km1 = K - 1  # beta has K-1 cols before inserting the reference 0
    sigma_d = rng.standard_normal((chain, D, 1)).astype(np.float64)
    b_offset = rng.standard_normal((chain, D, Km1)).astype(np.float64)
    ind_raw = (rng.standard_normal((chain, D, Km1)) * 0.01).astype(np.float64)
    alphas = rng.standard_normal((chain, K)).astype(np.float64)
    return [sigma_d, b_offset, ind_raw, alphas]


def test_get_y_hat_matches_softmax_reference():
    N, K, ref = 5, 4, 1
    self = _make_self(N, K, ref, seed=2)
    num_results, num_burnin = 8, 3
    states = _states(self, num_results, num_burnin, seed=3)
    states_in = [s.copy() for s in states]

    y_mean = azsccoda.fast_get_y_hat(self, states_in, num_results, num_burnin)
    assert y_mean.shape == (N, K)

    # Reference using the same algebra the function applies, computed
    # independently with numpy.
    sigma_d, b_offset, ind_raw, alphas = states
    alphas_final = alphas.mean(axis=0)
    ind_ = np.exp(ind_raw * 50) / (1 + np.exp(ind_raw * 50))
    b_raw_ = np.einsum("...jk, ...jl->...jk", b_offset, sigma_d)
    beta_temp = ind_ * b_raw_
    beta_ = np.insert(beta_temp, ref, 0.0, axis=2)
    betas_final = beta_.mean(axis=0)
    n_total = self.n_total.numpy()
    concentration = np.exp(self.x @ betas_final + alphas_final)
    ref_y = concentration / concentration.sum(axis=1, keepdims=True) * n_total[:, None]

    np.testing.assert_allclose(y_mean, ref_y, rtol=1e-10, atol=1e-12)


def test_get_y_hat_row_sums_equal_n_total():
    # Each predicted composition row must sum to that sample's n_total
    # (softmax * n_total).
    N, K, ref = 6, 5, 0
    self = _make_self(N, K, ref, seed=4)
    states = _states(self, 10, 4, seed=5)
    y_mean = azsccoda.fast_get_y_hat(self, states, 10, 4)
    np.testing.assert_allclose(
        y_mean.sum(axis=1), self.n_total.numpy(), rtol=1e-9, atol=1e-6
    )


def test_get_y_hat_reference_column_inserted():
    # The reference cell type's beta column is forced to 0 before the softmax;
    # check the insert happened by confirming output shape includes all K types.
    N, K, ref = 4, 3, 2
    self = _make_self(N, K, ref, seed=6)
    y_mean = azsccoda.fast_get_y_hat(self, _states(self, 7, 2, seed=7), 7, 2)
    assert y_mean.shape == (N, K)
    assert np.all(np.isfinite(y_mean))
