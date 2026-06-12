"""Contract tests for the sccoda patch (HMC compositional analysis).

Patched surface (3 targets):
  - scCODAModel.get_y_hat   (per-iteration posterior helper)
  - SimpleLeapfrogIntegrator.__call__ (TFP MCMC inner loop)
  - scCODAModel.sample_hmc  (the public HMC sampling entry)

Default sample_hmc draws 20000 results / 5000 burn-in. For a contract
test we override to ~100/20 -- enough for the patched code paths to be
exercised end-to-end without paying the full sampling cost.
"""
from __future__ import annotations

import os
import pytest

# Quiet TF noise before sccoda imports it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sccoda")
pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_probability")
pytest.importorskip("tf_keras")


@pytest.fixture(scope="module")
def sccoda_model():
    """Minimal compositional dataset + fitted CompositionalAnalysis model.

    8 samples (4 control, 4 treated) x 4 cell types -- the smallest
    fixture sccoda's reference-cell-type selection won't reject. Module-
    scoped because model setup is the heavy part (~5s); HMC is the bit
    we're contract-testing per call."""
    import sccoda.util.cell_composition_data as dat
    import sccoda.util.comp_ana as ca

    rng = np.random.default_rng(0)
    n_samples = 8
    n_types = 4
    # Composition counts with deliberate shift in cell-type 0 under treatment.
    counts = rng.integers(50, 200, size=(n_samples, n_types))
    counts[4:, 0] = counts[4:, 0] * 2  # treatment doubles type-0
    df = pd.DataFrame(counts, columns=[f"T{i}" for i in range(n_types)])
    df["condition"] = ["ctrl"] * 4 + ["treat"] * 4
    df["sample"] = [f"s{i}" for i in range(n_samples)]
    data = dat.from_pandas(df, covariate_columns=["condition", "sample"])
    model = ca.CompositionalAnalysis(
        data, formula="condition", reference_cell_type="automatic"
    )
    return model


def test_sample_hmc_returns_result_object(sccoda_model):
    """sample_hmc with tiny iters must return a CompAnaResult / similar
    -- the model's documented sampling output type."""
    import autozyme
    autozyme.activate("sccoda")

    # 100 results / 20 burnin is the smallest viable HMC chain that lets
    # the patched code path complete; below this, TF graph trace +
    # warm-up dominates and assertion timing gets flaky.
    out = sccoda_model.sample_hmc(num_results=100, num_burnin=20,
                                  verbose=False)
    assert out is not None
    # Result has a .posterior / .summary() method depending on version.
    has_posterior = hasattr(out, "posterior") or hasattr(out, "summary")
    assert has_posterior, (
        f"sample_hmc result lacks .posterior / .summary -- got "
        f"{type(out).__name__}"
    )


def test_sample_hmc_zyme_false_returns_same_type(sccoda_model):
    """zyme=False bypass must return the same result type as patched.

    Numerical parity isn't checked: HMC is stochastic and exact bit-
    parity between fast and vanilla TFP kernels isn't a sustainable
    contract on tiny chains. The TYPE contract is what matters: a user
    can swap patched <-> vanilla and downstream code (e.g. .summary())
    keeps working."""
    import autozyme
    autozyme.activate("sccoda")

    fast = sccoda_model.sample_hmc(num_results=100, num_burnin=20,
                                   verbose=False)
    with autozyme.disabled():
        ref = sccoda_model.sample_hmc(num_results=100, num_burnin=20,
                                      verbose=False)
    assert type(fast) is type(ref), (
        f"patched={type(fast).__name__}, vanilla={type(ref).__name__}"
    )
