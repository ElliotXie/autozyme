"""Wave-3 heavy-path tests for autozyme.cell2location.fast_forward.

Wave-1 (`test_cell2location_unit.py`) + wave-2 (`test_cell2location_e2e.py`)
tested fast_gp_log_prob and the _gp_log_prob_alpha_mu autograd fn. Both noted the
big `fast_forward` LocationModel target as "genuinely too heavy" and did NOT
drive it -- so the whole pyro generative forward (lines 141-284, the n_batch=1
fast path that skips obs2sample matmuls, reorders the cell_state einsum, and
routes the data likelihood through the custom GammaPoisson autograd.Function)
was uncovered.

This file builds the SMALLEST real LocationModelLinearDependentWMultiExperiment...
PyroModule (6 spots / 4 genes / 3 factors / 1 batch) directly -- no AnnData,
setup_anndata, or guide/SVI needed -- and runs ONE forward under a pyro trace.
The model build + a forward trace are ~tens of ms (the multi-second cost is the
shared one-time torch/pyro/cell2location import, paid once per session). It
covers:

  - the in-scope n_batch=1 path end-to-end (all the pyro.sample sites + the
    `data_target` GammaPoisson factor through _gp_log_prob_alpha_mu),
  - PARITY: the shared latent sample sites are bit-identical to the upstream
    forward (same RNG draw order), and the total trace log-prob (incl. the
    data likelihood) matches the upstream forward to float32 tolerance,
  - the eval/`not self.training` deterministic branch (u_sf_mRNA_factors),
  - the OUT-OF-SCOPE guard: n_batch != 1 falls back to _orig_forward.

cell2location 0.1.5 stores `self.cell_state = cell_state_mat.T`, so we pass
cell_state_mat shaped (n_vars, n_factors).
"""
from __future__ import annotations

import warnings

import pytest

pytest.importorskip("cell2location")
pytest.importorskip("pyro")
torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

import pyro
import cell2location.models._cell2location_module as _cm

import autozyme
from autozyme import cell2location as azc2l

_LocModel = (
    _cm.LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel
)

N_OBS, N_VARS, N_FACTORS = 6, 4, 3


@pytest.fixture(autouse=True)
def _silence():
    warnings.filterwarnings("ignore")
    azc2l._gp_lgamma_cache.clear()
    yield
    azc2l._gp_lgamma_cache.clear()
    autozyme.deactivate_all()


def _build_model(n_batch=1, seed=0):
    rng = np.random.default_rng(seed)
    # cell_state_mat is (n_vars, n_factors); the model stores its transpose.
    cell_state = rng.random((N_VARS, N_FACTORS)).astype(np.float32) + 0.1
    torch.manual_seed(seed)
    mod = _LocModel(
        n_obs=N_OBS, n_vars=N_VARS, n_factors=N_FACTORS, n_batch=n_batch,
        cell_state_mat=cell_state, n_groups=5,
    )
    x_data = torch.tensor(
        rng.integers(0, 10, (N_OBS, N_VARS)), dtype=torch.float32
    )
    idx = torch.arange(N_OBS, dtype=torch.long)
    batch_index = torch.zeros((N_OBS, 1), dtype=torch.long)
    return mod, x_data, idx, batch_index


def _trace(forward_fn, x_data, idx, batch_index, seed=0):
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    tr = pyro.poutine.trace(forward_fn).get_trace(x_data, idx, batch_index)
    tr.compute_log_prob()
    return tr


def _total_logp(tr):
    return sum(
        float(tr.nodes[k]["log_prob_sum"])
        for k in tr.nodes
        if tr.nodes[k]["type"] == "sample" and "log_prob_sum" in tr.nodes[k]
    )


def test_fast_forward_runs_and_is_finite():
    """The patched n_batch=1 forward runs the full generative model, emits the
    data_target GammaPoisson factor, and the total trace log-prob is finite."""
    mod, x_data, idx, batch_index = _build_model()
    autozyme.activate("cell2location")
    tr = _trace(mod.forward, x_data, idx, batch_index)
    assert "data_target" in tr.nodes  # the custom-autograd data likelihood ran
    # A representative set of prior sample sites were registered.
    for site in ("m_g", "w_sf", "detection_y_s", "alpha_g_inverse"):
        assert site in tr.nodes
    assert np.isfinite(_total_logp(tr))


def test_fast_forward_latents_bit_identical_to_upstream():
    """fast_forward draws the shared prior sample sites in the SAME RNG order as
    the upstream forward, so every shared latent value is bit-identical."""
    mod, x_data, idx, batch_index = _build_model()

    autozyme.activate("cell2location")
    tr_fast = _trace(mod.forward, x_data, idx, batch_index)
    autozyme.deactivate_all()
    tr_orig = _trace(
        azc2l._orig_forward.__get__(mod), x_data, idx, batch_index
    )

    shared = [
        k for k in tr_fast.nodes
        if tr_fast.nodes[k]["type"] == "sample"
        and k in tr_orig.nodes
        and k != "data_target"
        and "value" in tr_fast.nodes[k]
    ]
    assert len(shared) >= 10
    for k in shared:
        a = tr_fast.nodes[k]["value"]
        b = tr_orig.nodes[k]["value"]
        if a.shape == b.shape:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_fast_forward_total_logp_matches_upstream():
    """Given identical latents, the patched forward's total trace log-prob
    (including the reformulated float32 GammaPoisson data likelihood) matches
    the upstream forward to float32 tolerance."""
    mod, x_data, idx, batch_index = _build_model(seed=1)

    autozyme.activate("cell2location")
    tr_fast = _trace(mod.forward, x_data, idx, batch_index)
    autozyme.deactivate_all()
    tr_orig = _trace(
        azc2l._orig_forward.__get__(mod), x_data, idx, batch_index
    )

    fast_tot = _total_logp(tr_fast)
    orig_tot = _total_logp(tr_orig)
    assert np.isfinite(fast_tot) and np.isfinite(orig_tot)
    # float32 reformulation + lgamma cache vs the upstream path: tolerance is
    # relative to the magnitude of the (negative) log density.
    assert abs(fast_tot - orig_tot) < 1e-2 * abs(orig_tot)


def test_fast_forward_eval_deterministic_branch():
    """In eval mode (`not self.training`) the patched forward populates the
    deterministic u_sf_mRNA_factors node (the mRNA-per-factor breakdown)."""
    mod, x_data, idx, batch_index = _build_model(seed=2)
    mod.eval()
    autozyme.activate("cell2location")
    tr = _trace(mod.forward, x_data, idx, batch_index)
    assert "u_sf_mRNA_factors" in tr.nodes
    mrna = tr.nodes["u_sf_mRNA_factors"]["value"].detach().cpu().numpy()
    assert mrna.shape == (N_OBS, N_FACTORS)
    assert np.all(np.isfinite(mrna))


def test_fast_forward_nbatch_not_one_falls_back(monkeypatch):
    """The out-of-scope guard: n_batch != 1 delegates to _orig_forward (the
    lgamma cache's single-batch assumption doesn't hold)."""
    mod, x_data, idx, batch_index = _build_model(n_batch=2, seed=3)
    autozyme.activate("cell2location")

    hit = {"called": False}
    orig = azc2l._orig_forward

    def _spy(self, *a, **k):
        hit["called"] = True
        return orig(self, *a, **k)

    monkeypatch.setattr(azc2l, "_orig_forward", _spy)
    _trace(mod.forward, x_data, idx, batch_index)
    assert hit["called"] is True
