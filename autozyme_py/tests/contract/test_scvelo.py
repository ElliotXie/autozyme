"""Contract tests for the scvelo patch.

Patched surface (5 targets):
  - assign_tau         (per-cell EM tau assignment for recover_dynamics)
  - get_n_jobs         (n_jobs auto-resolution)
  - _fit_recovery      (worker-side patch re-installer for loky)
  - get_connectivities (neighbor-graph accessor)
  - SplicingDynamics.get_solution (per-time trajectory closed-form)

The end-to-end ``scv.tl.recover_dynamics`` path is too heavy for a
contract test on a synthetic fixture; we exercise the
contract-checkable units directly: get_n_jobs (deterministic) and
SplicingDynamics.get_solution (closed-form, no RNG).
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
scvelo = pytest.importorskip("scvelo")


def test_get_n_jobs_returns_positive_int():
    """get_n_jobs(None) -> auto-resolved CPU count >= 1."""
    import autozyme
    autozyme.activate("scvelo")
    from scvelo.tools._em_model_core import get_n_jobs

    out = get_n_jobs(None)
    assert isinstance(out, int) or hasattr(out, "__int__")
    assert int(out) >= 1


def test_get_n_jobs_one_passes_through():
    """get_n_jobs(1) must stay 1 -- standard 'serial' signal."""
    import autozyme
    autozyme.activate("scvelo")
    from scvelo.tools._em_model_core import get_n_jobs

    assert get_n_jobs(1) == 1


def test_get_n_jobs_caps_above_cpu():
    """Requesting more than available CPUs caps at CPU count."""
    import autozyme
    import os
    autozyme.activate("scvelo")
    from scvelo.tools._em_model_core import get_n_jobs

    cpu = os.cpu_count() or 1
    # Asking for absurd n_jobs should cap at cpu.
    out = get_n_jobs(cpu * 100)
    assert int(out) <= cpu


def test_splicing_dynamics_get_solution_shape():
    """SplicingDynamics.get_solution(t) returns (len(t), 2) [u, s] trajectory."""
    import autozyme
    autozyme.activate("scvelo")
    from scvelo.core import SplicingDynamics

    sd = SplicingDynamics(alpha=1.0, beta=0.5, gamma=0.3)
    t = np.linspace(0, 5.0, 50)
    sol = np.asarray(sd.get_solution(t))
    assert sol.shape == (50, 2), f"unexpected shape {sol.shape}"
    # u, s both non-negative under standard induction dynamics.
    assert np.all(sol >= -1e-9)


def test_splicing_dynamics_get_solution_zyme_false_matches_vanilla():
    """zyme=False (via context) matches patched closed-form output."""
    import autozyme
    autozyme.activate("scvelo")
    from scvelo.core import SplicingDynamics

    t = np.linspace(0, 5.0, 50)
    sd = SplicingDynamics(alpha=1.0, beta=0.5, gamma=0.3)
    fast = np.asarray(sd.get_solution(t))
    with autozyme.disabled():
        sd_v = SplicingDynamics(alpha=1.0, beta=0.5, gamma=0.3)
        ref = np.asarray(sd_v.get_solution(t))

    np.testing.assert_allclose(fast, ref, rtol=1e-6, atol=1e-9,
                               err_msg="SplicingDynamics drift patched vs vanilla")


def test_get_connectivities_returns_sparse_or_array(scvelo_adata):
    """get_connectivities returns the neighbor connectivity matrix.

    Skips if scvelo doesn't expose get_connectivities at the import path
    we patch (version drift)."""
    import autozyme
    autozyme.activate("scvelo")
    try:
        from scvelo.tools._em_model_core import get_connectivities
    except ImportError:
        pytest.skip("scvelo version does not expose get_connectivities at "
                    "the patched import path")

    conn = get_connectivities(scvelo_adata)
    assert conn is not None
    # Either sparse or dense -- both valid for the connectivity matrix.
    assert hasattr(conn, "shape")


@pytest.fixture
def scvelo_adata():
    """Tiny AnnData with a precomputed neighbor graph (needed for
    get_connectivities). 60-cell PCA + neighbors is the cheapest fixture
    that lets the patched function find a connectivities slot."""
    import scanpy as sc
    import anndata as ad
    from scipy import sparse

    rng = np.random.default_rng(0)
    X = sparse.csr_matrix(
        rng.poisson(2.0, size=(60, 80)).astype(np.float32)
    )
    a = ad.AnnData(X)
    # Vanilla scanpy chain to populate neighbors -- use autozyme.disabled
    # so this fixture doesn't accidentally test scanpy + scvelo at once.
    import autozyme
    with autozyme.disabled():
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.tl.pca(a, n_comps=10)
        sc.pp.neighbors(a, n_neighbors=10)
    return a
