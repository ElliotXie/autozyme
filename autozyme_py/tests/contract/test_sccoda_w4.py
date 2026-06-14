"""Wave-4 fuller-HMC-orchestration tests for autozyme.sccoda.

Wave-3 (`test_sccoda_w3.py`) drove the unrolled `fast_lfi_call` leapfrog kernel
+ the kinetic_energy_fn path + the out-of-range fallback + the `zyme=False`
dispatch -- but it NEVER ran a real `fast_sample_hmc` end-to-end. So the verbose
ProgressBar orchestration (src lines 274-292, 325-326), the SimpleStepSizeAdaptation
+ TransformedTransitionKernel + sample_chain plumbing, the real
`fast_get_y_hat` consumption, and the instance-tlpf restore (src 339-340) were
all uncovered, as was the smoke recipe.

This file builds the SMALLEST real scCODA `CompositionalAnalysis` (6 samples /
4 cell types / 1 binary covariate) and runs a TINY real HMC (num_results=200,
num_burnin=50) -- ~1.3 s -- under the patch, with `verbose=True`, covering:

  - the full `fast_sample_hmc` verbose orchestration end-to-end (the patched
    leapfrog + flat-rolled target_log_prob_fn + retained-only chain),
  - the instance-tlpf save/restore (CompositionalAnalysis sets
    `target_log_prob_fn` at construction -> the `had_instance_tlpf` restore branch),
  - the smoke recipe: `_smoke_load` (read CSV, build model, tier config) +
    `_smoke_save` (posterior/stats npz + summary json + result pkl).

`_smoke_call` itself uses the tier's hardcoded `num_results=20000` (too heavy for
a contract test); we instead shrink `inputs["num_results"]`/`num_burnin` and let
`_smoke_call` drive the tiny fit (so its body -- src line 403 -- is covered).
"""
from __future__ import annotations

import os
import tempfile
import warnings

import pytest

# TF env knobs must be set before TF import (mirrors the patch module).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
tf = pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_probability")
pytest.importorskip("sccoda")

import autozyme
from autozyme import sccoda as azsccoda


@pytest.fixture(autouse=True)
def _silence():
    warnings.filterwarnings("ignore")
    yield
    autozyme.deactivate_all()


def _tiny_comp_df():
    rng = np.random.default_rng(0)
    n_samp, n_ct = 6, 4
    counts = rng.integers(5, 50, (n_samp, n_ct))
    df = pd.DataFrame(counts, columns=[f"ct{i}" for i in range(n_ct)])
    df["sample_id"] = [f"s{i}" for i in range(n_samp)]
    df["cond"] = ["A", "A", "A", "B", "B", "B"]
    return df


def _build_model(df):
    from sccoda.util import comp_ana as ca
    from sccoda.util import cell_composition_data as scd

    tf.random.set_seed(42)
    np.random.seed(42)
    data = scd.from_pandas(df.set_index("sample_id"), covariate_columns=["cond"])
    return ca.CompositionalAnalysis(
        data, formula="cond", reference_cell_type="automatic"
    )


def test_fast_sample_hmc_verbose_end_to_end():
    """A tiny real HMC under the patch with verbose=True drives the full
    `fast_sample_hmc` orchestration (verbose ProgressBarReducer + WithReductions
    trace_fn, the patched leapfrog, flat-rolled tlpf, retained-only chain, and
    `fast_get_y_hat`). The result carries a posterior + a sane acceptance rate."""
    df = _tiny_comp_df()
    model = _build_model(df)
    autozyme.activate("sccoda")
    res = model.sample_hmc(num_results=200, num_burnin=50, verbose=True)
    assert hasattr(res, "posterior")
    acc = float(res.sampling_stats["acc_rate"])
    assert 0.0 <= acc <= 1.0
    # y_hat (from fast_get_y_hat) is part of sampling_stats.
    assert "y_hat" in res.sampling_stats


def test_fast_sample_hmc_restores_instance_tlpf():
    """CompositionalAnalysis sets `target_log_prob_fn` at construction, so
    `had_instance_tlpf` is True and `fast_sample_hmc` RESTORES the original
    instance tlpf in its finally block (src 339-340) rather than deleting it."""
    df = _tiny_comp_df()
    model = _build_model(df)
    original_tlpf = model.__dict__.get("target_log_prob_fn")
    assert "target_log_prob_fn" in model.__dict__  # set at construction

    autozyme.activate("sccoda")
    model.sample_hmc(num_results=120, num_burnin=30, verbose=False)
    # After the patched call, the instance tlpf is restored to the original.
    assert model.__dict__.get("target_log_prob_fn") is original_tlpf


@pytest.fixture
def smoke_task_dir():
    import yaml

    td = tempfile.mkdtemp(prefix="autozyme_sccoda_w4_")
    os.makedirs(os.path.join(td, "data"), exist_ok=True)
    df = _tiny_comp_df()
    df.to_csv(os.path.join(td, "data", "comp.csv"), index=False)
    with open(os.path.join(td, "task.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"datasets": [{"tier": "small", "path": "data/comp.csv"}]}, f
        )
    return td


def test_smoke_load_builds_model_and_tier_config(smoke_task_dir):
    """`_smoke_load` reads the CSV, infers the single covariate column, builds a
    real CompositionalAnalysis, and returns the tier's num_results/num_burnin."""
    autozyme.activate("sccoda")
    inputs = azsccoda._smoke_load(smoke_task_dir, "small")
    assert set(inputs.keys()) == {"model", "num_results", "num_burnin"}
    assert inputs["num_results"] == 20000
    assert inputs["num_burnin"] == 5000
    assert hasattr(inputs["model"], "sample_hmc")


def test_smoke_load_rejects_multiple_covariates():
    """`_smoke_load` raises when the CSV has more than one covariate (object)
    column -- the `len(covariate_cols) != 1` guard (src lines 363-364)."""
    import yaml

    td = tempfile.mkdtemp(prefix="autozyme_sccoda_w4_multi_")
    os.makedirs(os.path.join(td, "data"), exist_ok=True)
    df = pd.DataFrame({"ct0": [1, 2, 3, 4], "ct1": [5, 6, 7, 8]})
    df["sample_id"] = ["s0", "s1", "s2", "s3"]
    df["cond1"] = ["A", "A", "B", "B"]
    df["cond2"] = ["X", "Y", "X", "Y"]  # second covariate column
    df.to_csv(os.path.join(td, "data", "comp.csv"), index=False)
    with open(os.path.join(td, "task.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"datasets": [{"tier": "small", "path": "data/comp.csv"}]}, f
        )
    with pytest.raises(RuntimeError, match="expected 1 covariate column"):
        azsccoda._smoke_load(td, "small")


def test_smoke_call_and_save_roundtrip(smoke_task_dir):
    """Drive `_smoke_call` (shrunk to a tiny fit) + `_smoke_save`: the posterior
    / stats npz + summary json + result pkl are all written and readable."""
    autozyme.activate("sccoda")
    inputs = azsccoda._smoke_load(smoke_task_dir, "small")
    # Shrink the tier-default 20000/5000 so _smoke_call's real fit is fast.
    inputs["num_results"] = 200
    inputs["num_burnin"] = 50
    result = azsccoda._smoke_call(inputs)
    assert hasattr(result, "posterior")

    out_dir = tempfile.mkdtemp(prefix="autozyme_sccoda_w4_out_")
    azsccoda._smoke_save(result, out_dir)
    for name in ("posterior.npz", "stats.npz", "summary.json", "result.pkl"):
        assert os.path.isfile(os.path.join(out_dir, name)), name
    posterior = np.load(os.path.join(out_dir, "posterior.npz"))
    assert len(posterior.files) > 0
    import json
    with open(os.path.join(out_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    assert "acc_rate" in summary and "cell_types" in summary
