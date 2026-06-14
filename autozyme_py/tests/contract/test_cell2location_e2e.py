"""End-to-end / wrapper-line tests for autozyme.cell2location.

Wave-1 (`test_cell2location_unit.py`) + the existing `test_cell2location.py`
test fast_gp_log_prob (2-row redundant) and the _gp_log_prob_alpha_mu autograd
fn. This file covers the remaining COVERAGE-VISIBLE branches in fast_gp_log_prob
that those don't reach, driven through the PUBLIC pyro GammaPoisson.log_prob:

  - the 1-D / non-(2-D) concentration branch (c.dim() != 2 -> row_red=False).
  - distinct-row 2-D concentration (row_red=False) vs identical-row (row_red=True).
  - the lgamma cache reuse on a repeated value tensor.
  - activate/restore lifecycle on the 2 cell2location targets.

The big `fast_forward` LocationModel target needs a full spatial AnnData +
signature + pyro guide construction (a multi-second model build) -- genuinely
too heavy for a unit/contract test, so it is NOT driven here (see report note).
"""
from __future__ import annotations

import pytest

pytest.importorskip("cell2location")
pytest.importorskip("pyro")
torch = pytest.importorskip("torch")

import autozyme
from autozyme import cell2location as azc2l


@pytest.fixture(autouse=True)
def _clear_cache():
    azc2l._gp_lgamma_cache.clear()
    autozyme.activate("cell2location")
    yield
    azc2l._gp_lgamma_cache.clear()


def test_gp_log_prob_1d_concentration_matches_vanilla():
    """A 1-D concentration takes the row_red=False branch (c.dim()!=2);
    patched public log_prob matches pyro."""
    import pyro.distributions as dist

    concentration = torch.tensor([2.5, 3.0, 4.0], dtype=torch.float64)
    rate = torch.tensor([0.4, 0.7, 0.5], dtype=torch.float64)
    value = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)

    fast = dist.GammaPoisson(concentration=concentration, rate=rate).log_prob(value)
    with autozyme.disabled():
        ref = dist.GammaPoisson(concentration=concentration, rate=rate).log_prob(value)
    torch.testing.assert_close(fast, ref, rtol=1e-12, atol=1e-12)


def test_gp_log_prob_distinct_rows_branch():
    """A 2-D concentration with DISTINCT rows takes the row_red=False branch of
    the equality check; parity vs pyro."""
    import pyro.distributions as dist

    concentration = torch.tensor(
        [[2.5, 3.0, 4.0], [1.5, 2.0, 5.0]], dtype=torch.float64
    )
    rate = torch.tensor([[0.4, 0.7, 0.5], [0.6, 0.3, 0.8]], dtype=torch.float64)
    value = torch.tensor([[3.0, 2.0, 1.0], [4.0, 1.0, 5.0]], dtype=torch.float64)

    fast = dist.GammaPoisson(concentration=concentration, rate=rate).log_prob(value)
    with autozyme.disabled():
        ref = dist.GammaPoisson(concentration=concentration, rate=rate).log_prob(value)
    torch.testing.assert_close(fast, ref, rtol=1e-12, atol=1e-12)


def test_gp_log_prob_identical_rows_row_red_true():
    """Identical concentration rows take the row_red=True fast branch (uses
    c[0:1].lgamma()); parity vs pyro."""
    import pyro.distributions as dist

    concentration = torch.tensor(
        [[2.5, 3.0, 4.0], [2.5, 3.0, 4.0]], dtype=torch.float64
    )
    rate = torch.tensor([[0.4, 0.7, 0.5], [0.4, 0.7, 0.5]], dtype=torch.float64)
    value = torch.tensor([[3.0, 2.0, 1.0], [4.0, 1.0, 5.0]], dtype=torch.float64)

    fast = dist.GammaPoisson(concentration=concentration, rate=rate).log_prob(value)
    with autozyme.disabled():
        ref = dist.GammaPoisson(concentration=concentration, rate=rate).log_prob(value)
    torch.testing.assert_close(fast, ref, rtol=1e-12, atol=1e-12)


def test_gp_log_prob_caches_lgamma_on_repeated_value():
    """Calling log_prob twice with the SAME value tensor reuses the cached
    lgamma(value+1) (the cache key is value.data_ptr())."""
    import pyro.distributions as dist

    concentration = torch.tensor([2.5, 3.0, 4.0], dtype=torch.float64)
    rate = torch.tensor([0.4, 0.7, 0.5], dtype=torch.float64)
    value = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)

    d = dist.GammaPoisson(concentration=concentration, rate=rate)
    d.log_prob(value)
    key = (value.data_ptr(), value.shape, value.dtype, str(value.device))
    assert key in azc2l._gp_lgamma_cache
    cached_lg = azc2l._gp_lgamma_cache[key]["lg_v1"]
    d.log_prob(value)
    assert azc2l._gp_lgamma_cache[key]["lg_v1"] is cached_lg


def test_activate_restore_lifecycle():
    import pyro.distributions.conjugate as conj

    autozyme.deactivate("cell2location")
    orig_log_prob = conj.GammaPoisson.log_prob
    assert autozyme.activate("cell2location") is True
    assert conj.GammaPoisson.log_prob is not orig_log_prob
    info = autozyme.inspect("cell2location")
    assert info["status"] == "active"
    assert len(info["targets"]) == 2
    autozyme.deactivate("cell2location")
    assert conj.GammaPoisson.log_prob is orig_log_prob
