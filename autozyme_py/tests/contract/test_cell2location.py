"""Contract tests for the cell2location patch (spatial deconvolution).

Patched surface (2 targets):
  - pyro.distributions.conjugate.GammaPoisson.log_prob (fused kernel)
  - cell2location.models._cell2location_module._LocModel.forward

Public training still needs a real spatial transcriptomics fixture, but
the two mathematical hot kernels can be tested directly on tensors.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cell2location")
pytest.importorskip("pyro")
torch = pytest.importorskip("torch")


def test_gamma_poisson_log_prob_matches_vanilla():
    """Patched public GammaPoisson.log_prob matches Pyro on tensor inputs."""
    import autozyme
    import pyro.distributions as dist

    autozyme.activate("cell2location")
    concentration = torch.tensor(
        [[2.5, 3.0, 4.0], [2.5, 3.0, 4.0]], dtype=torch.float64
    )
    rate = torch.tensor(
        [[0.4, 0.7, 0.5], [0.4, 0.7, 0.5]], dtype=torch.float64
    )
    value = torch.tensor([[3.0, 2.0, 1.0], [4.0, 1.0, 5.0]], dtype=torch.float64)

    fast = dist.GammaPoisson(concentration=concentration, rate=rate).log_prob(value)
    with autozyme.disabled():
        ref = dist.GammaPoisson(concentration=concentration, rate=rate).log_prob(value)
    torch.testing.assert_close(fast, ref, rtol=1e-12, atol=1e-12)


def test_alpha_mu_log_prob_backward_matches_vanilla():
    """Custom autograd kernel used by LocModel.forward has correct gradients."""
    import autozyme
    import pyro.distributions as dist

    autozyme.activate("cell2location")
    from autozyme.cell2location import _gp_log_prob_alpha_mu

    alpha = torch.tensor([[2.5, 3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    mu = torch.tensor([[5.0, 4.0, 6.0]], dtype=torch.float64, requires_grad=True)
    value = torch.tensor([[3.0, 2.0, 1.0], [4.0, 1.0, 5.0]], dtype=torch.float64)

    fast = _gp_log_prob_alpha_mu(alpha, mu, value)
    ref = dist.GammaPoisson(concentration=alpha, rate=alpha / mu).log_prob(value).sum()
    torch.testing.assert_close(fast, ref, rtol=1e-12, atol=1e-12)

    fast.backward(retain_graph=True)
    fast_alpha_grad = alpha.grad.detach().clone()
    fast_mu_grad = mu.grad.detach().clone()
    alpha.grad.zero_()
    mu.grad.zero_()
    ref.backward()
    torch.testing.assert_close(fast_alpha_grad, alpha.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(fast_mu_grad, mu.grad, rtol=1e-10, atol=1e-10)
