"""Unit tests for the torch helpers in autozyme.cell2location.

The contract test (test_cell2location.py) drives Cell2location.train end to end
through pyro. The forward() override needs the full PyroModule and is covered
there. Here we test the self-contained GammaPoisson log-prob accelerators
directly against pyro's reference distribution and torch autograd:

  - fast_gp_log_prob          cached lgamma + reformulated log term, vs pyro
  - _GPLogProbFn / _gp_log_prob_alpha_mu   custom autograd fn, value + gradients

cell2location + torch + pyro must import for the module to load.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pyrodist = pytest.importorskip("pyro.distributions")
pytest.importorskip("cell2location")

from autozyme import cell2location as azc2l


def setup_function(_fn):
    # The lgamma cache keys on tensor data_ptr; clear between tests so a reused
    # address from a freed tensor can't return a stale entry.
    azc2l._gp_lgamma_cache.clear()


# --------------------------------------------------------------------------
# fast_gp_log_prob  vs pyro GammaPoisson.log_prob
# --------------------------------------------------------------------------
def test_fast_gp_log_prob_matches_pyro():
    torch.manual_seed(0)
    n_obs, n_genes = 12, 5
    concentration = torch.rand(n_obs, n_genes).double() + 0.5
    rate = torch.rand(n_obs, n_genes).double() + 0.5
    value = torch.randint(0, 20, (n_obs, n_genes)).double()

    dist = pyrodist.GammaPoisson(concentration, rate)
    ref = dist.log_prob(value)
    got = azc2l.fast_gp_log_prob(dist, value)
    torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-9)


def test_fast_gp_log_prob_row_redundant_concentration():
    # When concentration is row-redundant (every row identical, n_obs>1) the
    # helper takes the c[0:1] fast path; result must still match pyro.
    torch.manual_seed(1)
    n_obs, n_genes = 8, 4
    row = torch.rand(1, n_genes).double() + 0.3
    concentration = row.expand(n_obs, n_genes).contiguous()
    rate = torch.rand(n_obs, n_genes).double() + 0.3
    value = torch.randint(0, 15, (n_obs, n_genes)).double()

    dist = pyrodist.GammaPoisson(concentration, rate)
    ref = dist.log_prob(value)
    got = azc2l.fast_gp_log_prob(dist, value)
    torch.testing.assert_close(got, ref, rtol=1e-9, atol=1e-9)


def test_fast_gp_log_prob_cache_reused():
    torch.manual_seed(2)
    value = torch.randint(0, 10, (6, 3)).double()
    conc = torch.rand(6, 3).double() + 0.5
    rate = torch.rand(6, 3).double() + 0.5
    dist = pyrodist.GammaPoisson(conc, rate)
    azc2l.fast_gp_log_prob(dist, value)
    key = (value.data_ptr(), value.shape, value.dtype, str(value.device))
    assert key in azc2l._gp_lgamma_cache
    # Second call on the same value tensor reuses the cached lgamma(value+1).
    cached_lg = azc2l._gp_lgamma_cache[key]["lg_v1"]
    azc2l.fast_gp_log_prob(dist, value)
    assert azc2l._gp_lgamma_cache[key]["lg_v1"] is cached_lg


# --------------------------------------------------------------------------
# _GPLogProbFn / _gp_log_prob_alpha_mu  (alpha/mu form)
# --------------------------------------------------------------------------
def _ref_alpha_mu_loglik(alpha, mu, value):
    # GammaPoisson in (alpha=concentration, mu=mean) form: rate = alpha / mu.
    rate = alpha / mu
    dist = pyrodist.GammaPoisson(alpha, rate)
    return dist.log_prob(value).sum()


def test_gp_log_prob_alpha_mu_value_matches_pyro():
    torch.manual_seed(3)
    n_obs, n_genes = 10, 4
    alpha = (torch.rand(1, n_genes).double() + 0.5)  # per-gene overdispersion
    mu = (torch.rand(n_obs, n_genes).double() + 1.0)
    value = torch.randint(0, 12, (n_obs, n_genes)).double()
    got = azc2l._gp_log_prob_alpha_mu(alpha, mu, value)
    ref = _ref_alpha_mu_loglik(alpha, mu, value)
    torch.testing.assert_close(got, ref, rtol=1e-8, atol=1e-7)


def test_gp_log_prob_alpha_mu_gradients_match_autograd():
    torch.manual_seed(4)
    n_obs, n_genes = 9, 3
    alpha = (torch.rand(1, n_genes).double() + 0.6).requires_grad_(True)
    mu = (torch.rand(n_obs, n_genes).double() + 1.0).requires_grad_(True)
    value = torch.randint(1, 10, (n_obs, n_genes)).double()

    # Custom autograd.Function path.
    azc2l._gp_lgamma_cache.clear()
    ll_fast = azc2l._gp_log_prob_alpha_mu(alpha, mu, value)
    ll_fast.backward()
    g_alpha_fast = alpha.grad.clone()
    g_mu_fast = mu.grad.clone()

    # Reference: differentiate pyro's log_prob with standard autograd.
    alpha2 = alpha.detach().clone().requires_grad_(True)
    mu2 = mu.detach().clone().requires_grad_(True)
    ll_ref = _ref_alpha_mu_loglik(alpha2, mu2, value)
    ll_ref.backward()

    torch.testing.assert_close(ll_fast, ll_ref, rtol=1e-7, atol=1e-6)
    torch.testing.assert_close(g_alpha_fast, alpha2.grad, rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(g_mu_fast, mu2.grad, rtol=1e-6, atol=1e-5)


def test_gp_log_prob_fn_gradcheck():
    # Formal gradient check of the custom backward against finite differences.
    torch.manual_seed(5)
    n_obs, n_genes = 5, 2
    alpha = (torch.rand(1, n_genes).double() + 0.7).requires_grad_(True)
    mu = (torch.rand(n_obs, n_genes).double() + 1.0).requires_grad_(True)
    value = torch.randint(1, 8, (n_obs, n_genes)).double()
    lg_v1 = (value + 1).lgamma()
    assert torch.autograd.gradcheck(
        azc2l._GPLogProbFn.apply, (alpha, mu, value, lg_v1),
        eps=1e-6, atol=1e-4, rtol=1e-3,
    )
