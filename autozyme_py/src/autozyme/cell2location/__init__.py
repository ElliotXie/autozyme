"""Patch for cell2location.train.

Lifted from autozyme task `test_cell2location`. Two co-evolved patches:
  - pyro.distributions.conjugate.GammaPoisson.log_prob
      → fast_gp_log_prob: caches lgamma(value+1) + row-redundant detection
        for n_batch=1 + reformulated trailing log term
  - cell2location.models._cell2location_module.<LocModel>.forward
      → fast_forward: n_batch=1 fast path that skips obs2sample matmuls,
        reorders the `w_sf @ cell_state * m_g` einsum, and routes the data
        likelihood through a custom autograd.Function with explicit backward
        for GammaPoisson.

Both must be active together — fast_forward routes through
_gp_log_prob_alpha_mu (the autograd.Function path), and fast_gp_log_prob is
the top-level `dist.GammaPoisson(...).log_prob(...)` accelerator for the
upstream code path that fast_forward falls back to (n_batch != 1).

Validated scope — FULL-BATCH training only (the standard cell2location usage,
``train(batch_size=None, train_size=1)``). The lgamma cache keys on
``value.data_ptr()``, which is correct only when the ``x_data`` content is
stable for a given storage address across epochs. Under minibatch SVI
(``batch_size`` set, or ``train_size`` < 1) PyTorch may reuse a buffer address
for different minibatch contents, so a cached ``lgamma(value+1)`` could be
applied to the wrong counts — a silently wrong likelihood. Minibatch training
is therefore OUT OF SCOPE for this patch: use it only with full-batch
cell2location training, or deactivate ``autozyme`` for minibatch runs.
"""
from __future__ import annotations

import os

import torch
import pyro
import pyro.distributions

import cell2location
import cell2location.models._cell2location_module as _cm

from autozyme._core import register_patch
from autozyme._utils import resolve_dataset_path

# Validation-args flag is scoped to the patched forward via a
# save/restore guard — see fast_forward. No process-global side effect.

# Resolve the long PyroModule class once, capture the upstream forward.
_LocModel = _cm.LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel
_orig_forward = _LocModel.forward

# Cache for lgamma(value+1) keyed by tensor identity — value=x_data is
# constant across all training epochs.
_gp_lgamma_cache: dict = {}


def fast_gp_log_prob(self, value):
    post_value = self.concentration + value
    key = (value.data_ptr(), value.shape, value.dtype, str(value.device))
    cache = _gp_lgamma_cache.get(key)
    if cache is None:
        cache = {"lg_v1": (value + 1).lgamma()}
        _gp_lgamma_cache[key] = cache

    c = self.concentration
    if "row_red" not in cache:
        if c.dim() == 2 and c.shape[0] == value.shape[0] and value.shape[0] > 1:
            cache["row_red"] = bool(torch.equal(c[0:1].expand_as(c), c))
        else:
            cache["row_red"] = False

    if cache["row_red"]:
        c_lgamma = c[0:1].lgamma()
    else:
        c_lgamma = c.lgamma()

    return (
        -c_lgamma
        - cache["lg_v1"]
        + post_value.lgamma()
        + self.concentration * self.rate.log()
        - post_value * (1 + self.rate).log()
    )


class _GPLogProbFn(torch.autograd.Function):
    """Custom autograd.Function for the (alpha, mu) form of GammaPoisson
    log-prob — bypasses autograd graph traversal of 10+ ops with one
    explicit backward.
    """
    @staticmethod
    def forward(ctx, alpha, mu, value, lg_v1):
        post_value = alpha + value
        mu_plus_alpha = mu + alpha
        log_alpha = alpha.log()
        log_mu = mu.log()
        log_mu_plus_alpha = mu_plus_alpha.log()
        n_obs = value.shape[0]
        ll = (
            -alpha.lgamma().sum() * n_obs
            - lg_v1.sum()
            + post_value.lgamma().sum()
            + (alpha * log_alpha).sum() * n_obs
            + (value * log_mu).sum()
            - (post_value * log_mu_plus_alpha).sum()
        )
        ctx.save_for_backward(
            alpha, mu, value, post_value, mu_plus_alpha,
            log_alpha, log_mu_plus_alpha,
        )
        ctx.n_obs = n_obs
        return ll

    @staticmethod
    def backward(ctx, grad_output):
        (
            alpha, mu, value, post_value, mu_plus_alpha,
            log_alpha, log_mu_plus_alpha,
        ) = ctx.saved_tensors
        n_obs = ctx.n_obs
        digamma_post = torch.digamma(post_value)
        digamma_post.sub_(log_mu_plus_alpha)
        ratio = post_value / mu_plus_alpha
        digamma_post.sub_(ratio)
        sum_per_gene = digamma_post.sum(dim=0, keepdim=True)
        d_alpha = grad_output * (
            n_obs * (-torch.digamma(alpha) + log_alpha + 1.0) + sum_per_gene
        )
        d_mu = value / mu
        d_mu.sub_(ratio)
        d_mu.mul_(grad_output)
        return d_alpha, d_mu, None, None


def _gp_log_prob_alpha_mu(alpha, mu, value):
    key = (value.data_ptr(), value.shape, value.dtype, str(value.device))
    cache = _gp_lgamma_cache.get(key)
    if cache is None:
        cache = {"lg_v1": (value + 1).lgamma()}
        _gp_lgamma_cache[key] = cache
    return _GPLogProbFn.apply(alpha, mu, value, cache["lg_v1"])


def fast_forward(self, x_data, idx, batch_index):
    # Out-of-scope guards — fall back to upstream (no silent-wrong path):
    #   n_batch != 1   : multi-sample data; the lgamma cache assumes one batch.
    #   dropout_p != 0 : self.dropout(x_data) makes a fresh tensor each step, so
    #                    the data_ptr-keyed lgamma cache would reuse
    #                    lgamma(dropout(x)+1) against differently-dropped counts.
    # Benchmark/default is dropout_p == 0, so this never fires in-scope.
    if self.n_batch != 1 or self.dropout_p != 0:
        return _orig_forward(self, x_data, idx, batch_index)

    _prev_validate = torch.distributions.Distribution._validate_args
    torch.distributions.Distribution.set_default_validate_args(False)

    def _restore_validate():
        torch.distributions.Distribution._validate_args = _prev_validate

    obs_plate = self.create_plates(x_data, idx, batch_index)

    m_g_mean = pyro.sample(
        "m_g_mean",
        _cm.dist.Gamma(
            self.m_g_mu_mean_var_ratio_hyp * self.m_g_mu_hyp,
            self.m_g_mu_mean_var_ratio_hyp,
        ).expand([1, 1]).to_event(2),
    )
    m_g_alpha_e_inv = pyro.sample(
        "m_g_alpha_e_inv",
        _cm.dist.Exponential(self.m_g_alpha_hyp_mean).expand([1, 1]).to_event(2),
    )
    m_g_alpha_e = self.ones / m_g_alpha_e_inv.pow(2)
    m_g = pyro.sample(
        "m_g",
        _cm.dist.Gamma(m_g_alpha_e, m_g_alpha_e / m_g_mean)
        .expand([1, self.n_vars]).to_event(2),
    )

    with obs_plate:
        n_s_cells_per_location = pyro.sample(
            "n_s_cells_per_location",
            _cm.dist.Gamma(
                self.N_cells_per_location * self.N_cells_mean_var_ratio,
                self.N_cells_mean_var_ratio,
            ),
        )
        b_s_groups_per_location = pyro.sample(
            "b_s_groups_per_location",
            _cm.dist.Gamma(self.B_groups_per_location, self.ones),
        )
    shape = self.ones_1_n_groups * b_s_groups_per_location / self.n_groups_tensor
    rate = self.ones_1_n_groups / (n_s_cells_per_location / b_s_groups_per_location)
    with obs_plate:
        z_sr_groups_factors = pyro.sample(
            "z_sr_groups_factors", _cm.dist.Gamma(shape, rate)
        )
    k_r_factors_per_groups = pyro.sample(
        "k_r_factors_per_groups",
        _cm.dist.Gamma(self.factors_per_groups, self.ones)
        .expand([self.n_groups, 1]).to_event(2),
    )
    c2f_shape = k_r_factors_per_groups / self.n_factors_tensor
    x_fr_group2fact = pyro.sample(
        "x_fr_group2fact",
        _cm.dist.Gamma(c2f_shape, k_r_factors_per_groups)
        .expand([self.n_groups, self.n_factors]).to_event(2),
    )
    with obs_plate:
        w_sf_mu = z_sr_groups_factors @ x_fr_group2fact
        w_sf = pyro.sample(
            "w_sf",
            _cm.dist.Gamma(
                w_sf_mu * self.w_sf_mean_var_ratio_tensor,
                self.w_sf_mean_var_ratio_tensor,
            ),
        )

    detection_mean_y_e = pyro.sample(
        "detection_mean_y_e",
        _cm.dist.Gamma(
            self.ones * self.detection_mean_hyp_prior_alpha,
            self.ones * self.detection_mean_hyp_prior_beta,
        ).expand([self.n_batch, 1]).to_event(2),
    )
    detection_hyp_prior_alpha = pyro.deterministic(
        "detection_hyp_prior_alpha",
        self.ones_n_batch_1 * self.detection_hyp_prior_alpha,
    )
    beta = detection_hyp_prior_alpha / detection_mean_y_e
    with obs_plate:
        detection_y_s = pyro.sample(
            "detection_y_s", _cm.dist.Gamma(detection_hyp_prior_alpha, beta),
        )

    s_g_gene_add_alpha_hyp = pyro.sample(
        "s_g_gene_add_alpha_hyp",
        _cm.dist.Gamma(
            self.ones * self.gene_add_alpha_hyp_prior_alpha,
            self.ones * self.gene_add_alpha_hyp_prior_beta,
        ),
    )
    s_g_gene_add_mean = pyro.sample(
        "s_g_gene_add_mean",
        _cm.dist.Gamma(
            self.gene_add_mean_hyp_prior_alpha,
            self.gene_add_mean_hyp_prior_beta,
        ).expand([self.n_batch, 1]).to_event(2),
    )
    s_g_gene_add_alpha_e_inv = pyro.sample(
        "s_g_gene_add_alpha_e_inv",
        _cm.dist.Exponential(s_g_gene_add_alpha_hyp).expand([self.n_batch, 1]).to_event(2),
    )
    s_g_gene_add_alpha_e = self.ones / s_g_gene_add_alpha_e_inv.pow(2)
    s_g_gene_add = pyro.sample(
        "s_g_gene_add",
        _cm.dist.Gamma(s_g_gene_add_alpha_e, s_g_gene_add_alpha_e / s_g_gene_add_mean)
        .expand([self.n_batch, self.n_vars]).to_event(2),
    )

    alpha_g_phi_hyp = pyro.sample(
        "alpha_g_phi_hyp",
        _cm.dist.Gamma(
            self.ones * self.alpha_g_phi_hyp_prior_alpha,
            self.ones * self.alpha_g_phi_hyp_prior_beta,
        ),
    )
    alpha_g_inverse = pyro.sample(
        "alpha_g_inverse",
        _cm.dist.Exponential(alpha_g_phi_hyp)
        .expand([self.n_batch, self.n_vars]).to_event(2),
    )

    if not self.training_wo_observed:
        mu = (w_sf @ (self.cell_state * m_g) + s_g_gene_add) * detection_y_s
        alpha = self.ones / alpha_g_inverse.pow(2)
        if self.dropout_p != 0:
            x_data = self.dropout(x_data)
        ll = _gp_log_prob_alpha_mu(alpha, mu, x_data)
        pyro.factor("data_target", ll)

    if not self.training:
        with obs_plate:
            mRNA = w_sf * (self.cell_state * m_g).sum(-1)
            pyro.deterministic("u_sf_mRNA_factors", mRNA)

    _restore_validate()


# Stable __module__ for the patched function so introspection isn't confusing.
fast_forward.__module__ = _orig_forward.__module__


# ---------- smoke recipe ----------

def _smoke_load(task_dir, tier):
    import yaml
    import scanpy as sc
    import pandas as pd
    import numpy as np
    import lightning.pytorch as pl
    import pyro
    with open(os.path.join(task_dir, "task.yaml"), encoding="utf-8") as f:
        task = yaml.safe_load(f)
    ds = next(d for d in task["datasets"] if d["tier"] == tier)
    adata = sc.read_h5ad(resolve_dataset_path(task_dir, ds["path"]))
    factor_names = list(adata.uns["c2l_factor_names"])
    inf_aver = pd.DataFrame(
        np.asarray(adata.varm["c2l_signature"]),
        index=adata.var_names,
        columns=factor_names,
    )
    # User-side prep that the patch doesn't accelerate (setup_anndata +
    # Cell2location(...) construction) lives here. Under the subprocess
    # verify_patch protocol every measurement runs in a fresh Python
    # interpreter, so there's no state contamination between reps and we can
    # build the model directly (no template + deepcopy ritual). Seed and
    # clear param store BEFORE construction so the guide / AutoNormal
    # initialization is deterministic per subprocess.
    cell2location.models.Cell2location.setup_anndata(
        adata=adata, batch_key="sample"
    )
    pyro.clear_param_store()
    pl.seed_everything(42, workers=True)
    mod = cell2location.models.Cell2location(
        adata,
        cell_state_df=inf_aver,
        N_cells_per_location=30,
        detection_alpha=20,
    )
    return {"mod": mod}


def _smoke_call(inputs):
    # One-shot per fresh Python process: nothing to reset, nothing to deepcopy.
    # The patch targets train(); only train() is timed.
    inputs["mod"].train(
        max_epochs=300,
        batch_size=None,
        train_size=1,
        lr=0.002,
        accelerator="cpu",
        enable_progress_bar=False,
    )
    return inputs["mod"]


def _smoke_save(mod, dir, **kwargs):
    import numpy as np
    loss_history = np.asarray(mod.history["elbo_train"]).ravel().astype(np.float64)
    with torch.no_grad():
        guide_medians = mod.module.guide.median()
    w_sf = guide_medians["w_sf"].detach().cpu().numpy().astype(np.float64)
    np.savez_compressed(
        os.path.join(dir, "outputs.npz"),
        loss_history=loss_history,
        w_sf=w_sf,
    )


register_patch(
    name="cell2location",
    targets=[
        ("pyro.distributions.conjugate.GammaPoisson", "log_prob", fast_gp_log_prob),
        (f"cell2location.models._cell2location_module.{_LocModel.__name__}",
         "forward", fast_forward),
    ],
    smoke={"load": _smoke_load, "call": _smoke_call, "save": _smoke_save},
    tested_against="cell2location 0.1.5",
    tested_upstream_versions={"cell2location": ["0.1.5"]},
)
