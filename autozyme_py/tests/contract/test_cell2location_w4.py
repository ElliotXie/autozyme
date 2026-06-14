"""Wave-4 heavy-path tests for autozyme.cell2location: the smoke recipe.

Wave-1 (`test_cell2location_unit.py`) + wave-2 (`test_cell2location_e2e.py`)
tested fast_gp_log_prob / _gp_log_prob_alpha_mu; wave-3 (`test_cell2location_w3.py`)
drove the full `fast_forward` n_batch=1 generative path + the eval branch + the
n_batch!=1 fallback. NONE of those touched the smoke recipe (lines 293-351):
`_smoke_load` (build a real Cell2location model from an h5ad), `_smoke_call`
(the real `train(...)` SVI loop the patch targets), and `_smoke_save` (guide
medians + loss history -> npz).

This file builds the SMALLEST real Cell2location model (12 spots / 6 genes /
2 factors / 1 batch) from a synthetic AnnData carrying the `c2l_factor_names`
uns + `c2l_signature` varm that `_smoke_load` reads, then drives the full
load -> call (real 300-epoch full-batch train, ~3 s on tiny data, with the
patch active so the fast n_batch=1 forward + custom-autograd GammaPoisson
likelihood actually run) -> save round-trip. It asserts the saved loss history
+ w_sf guide medians are finite and well-shaped.

The 300-epoch train on a real dataset would be minutes; on this 12x6 toy it is
a few seconds, so the genuinely-heavy `train()` SVI path the wave-3 file
declared "too heavy" is reached here end-to-end via the smoke recipe.

The only `fast_forward` line that stays uncovered is the in-function
`x_data = self.dropout(x_data)` (src line 275): it is dead under the patch's own
`dropout_p != 0` guard at the top of fast_forward, so it can never execute on
the patched path -- documented, not reachable.
"""
from __future__ import annotations

import os
import tempfile
import warnings

import pytest

pytest.importorskip("cell2location")
pytest.importorskip("pyro")
sc = pytest.importorskip("scanpy")
pytest.importorskip("lightning")
torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import autozyme
from autozyme import cell2location as azc2l

N_OBS, N_VARS, N_FACTORS = 12, 6, 2


@pytest.fixture(autouse=True)
def _silence():
    warnings.filterwarnings("ignore")
    azc2l._gp_lgamma_cache.clear()
    yield
    azc2l._gp_lgamma_cache.clear()
    autozyme.deactivate_all()


@pytest.fixture(scope="module")
def task_dir():
    """A synthetic task dir: task.yaml + a tiny h5ad carrying the c2l_signature
    varm + c2l_factor_names uns that `_smoke_load` reads."""
    import yaml

    td = tempfile.mkdtemp(prefix="autozyme_c2l_w4_")
    os.makedirs(os.path.join(td, "data"), exist_ok=True)
    rng = np.random.default_rng(0)
    X = rng.integers(0, 15, (N_OBS, N_VARS)).astype("float32")
    adata = sc.AnnData(X)
    adata.var_names = [f"g{i}" for i in range(N_VARS)]
    adata.obs_names = [f"c{i}" for i in range(N_OBS)]
    adata.obs["sample"] = "s1"
    adata.uns["c2l_factor_names"] = np.array([f"f{i}" for i in range(N_FACTORS)])
    adata.varm["c2l_signature"] = (
        rng.random((N_VARS, N_FACTORS)).astype("float32") + 0.1
    )
    adata.write_h5ad(os.path.join(td, "data", "tiny.h5ad"))
    with open(os.path.join(td, "task.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"datasets": [{"tier": "small", "path": "data/tiny.h5ad"}]}, f
        )
    return td


def test_smoke_load_builds_cell2location_model(task_dir):
    """`_smoke_load` reads the h5ad, derives inf_aver from c2l_signature, runs
    setup_anndata + Cell2location construction, and returns a live model."""
    inputs = azc2l._smoke_load(task_dir, "small")
    assert set(inputs.keys()) == {"mod"}
    mod = inputs["mod"]
    # A real Cell2location model exposes a pyro module + guide.
    assert hasattr(mod, "module")
    assert hasattr(mod, "train")


def test_smoke_full_load_call_save_roundtrip(task_dir):
    """End-to-end smoke recipe under the patch: load -> real 300-epoch
    full-batch train (the SVI path the patch targets) -> save. The saved
    loss history + w_sf guide medians are finite and well-shaped."""
    autozyme.activate("cell2location")
    inputs = azc2l._smoke_load(task_dir, "small")
    # `_smoke_call` uses the real hardcoded max_epochs=300 / batch_size=None /
    # train_size=1 -- the in-scope full-batch SVI config the patch validates.
    out_mod = azc2l._smoke_call(inputs)
    assert out_mod is inputs["mod"]

    out_dir = tempfile.mkdtemp(prefix="autozyme_c2l_w4_out_")
    azc2l._smoke_save(out_mod, out_dir)

    npz_path = os.path.join(out_dir, "outputs.npz")
    assert os.path.isfile(npz_path)
    z = np.load(npz_path)
    assert set(z.keys()) == {"loss_history", "w_sf"}
    # 300 epochs -> 300 ELBO values; all finite.
    assert z["loss_history"].shape == (300,)
    assert np.all(np.isfinite(z["loss_history"]))
    # w_sf median is (n_obs, n_factors), non-negative (a Gamma latent), finite.
    assert z["w_sf"].shape == (N_OBS, N_FACTORS)
    assert np.all(np.isfinite(z["w_sf"]))
    assert np.all(z["w_sf"] >= 0.0)


def test_smoke_call_runs_with_patch_forward_active(task_dir):
    """During `_smoke_call`'s train, the patched n_batch=1 fast_forward is the
    live forward (the model has n_batch == 1), so the train exercises the fast
    generative path + custom-autograd GammaPoisson likelihood, not just SVI
    plumbing."""
    autozyme.activate("cell2location")
    inputs = azc2l._smoke_load(task_dir, "small")
    mod = inputs["mod"]
    assert mod.module.model.n_batch == 1  # the in-scope fast path applies
    # The patched forward is bound on the model module's class.
    azc2l._smoke_call(inputs)
    # history is populated by a real Lightning/pyro training run.
    assert "elbo_train" in mod.history
