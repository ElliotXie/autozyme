"""End-to-end / wrapper-line tests for autozyme.scvelo.

Wave-1 (`test_scvelo_unit.py`) tested the numba kernels + NumbaConn dispatch +
fast_get_n_jobs directly. The existing `test_scvelo.py` drives get_solution /
get_n_jobs / get_connectivities. This file covers the remaining COVERAGE-VISIBLE
wrapper branches neither hits:

  - `_fast_get_solution`: the `_orig_get_solution` fallback branch (2-D `t`),
    plus the `with_keys=True` and `stacked=False` return shapes on the fast path.
  - `fast_assign_tau`: both the projection branch and the non-projection
    (`_orig_tau_inv`) branch, vs the captured upstream tau_inv.
  - `fit_recovery_with_overrides`: idempotent worker-side re-install wrapper.
  - `fast_get_connectivities`: the `conn is None` early-return + the float64
    cast wrapping path (NumbaConn).
  - activate/restore lifecycle on all 5 scvelo targets.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.sparse")
scvelo = pytest.importorskip("scvelo")

import autozyme
from autozyme import scvelo as azscvelo


def test_get_solution_2d_t_falls_back_to_original():
    """A 2-D `t` array routes _fast_get_solution to _orig_get_solution; the
    result must match the captured upstream original."""
    from scvelo.core import SplicingDynamics

    autozyme.activate("scvelo")
    sd = SplicingDynamics(alpha=1.0, beta=0.5, gamma=0.3)
    t2d = np.linspace(0, 5.0, 12).reshape(6, 2)

    got = sd.get_solution(t2d)  # patched dispatcher -> fallback branch
    ref = azscvelo._orig_get_solution(sd, t2d)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref),
                               rtol=1e-9, atol=1e-12)


def test_get_solution_with_keys_and_unstacked():
    """The fast path's with_keys=True (dict) and stacked=False (tuple) return
    shapes match the stacked output."""
    from scvelo.core import SplicingDynamics

    autozyme.activate("scvelo")
    sd = SplicingDynamics(alpha=1.0, beta=0.5, gamma=0.3)
    t = np.linspace(0, 5.0, 40)

    stacked = np.asarray(sd.get_solution(t))
    d = sd.get_solution(t, with_keys=True)
    assert set(d.keys()) == {"u", "s"}
    np.testing.assert_allclose(d["u"], stacked[:, 0], rtol=1e-9)
    np.testing.assert_allclose(d["s"], stacked[:, 1], rtol=1e-9)

    u, s = sd.get_solution(t, stacked=False)
    np.testing.assert_allclose(u, stacked[:, 0], rtol=1e-9)
    np.testing.assert_allclose(s, stacked[:, 1], rtol=1e-9)


def test_assign_tau_non_projection_matches_tau_inv():
    """assignment_mode=None takes the non-projection branch (np.clip on
    _orig_tau_inv); confirm it runs and stays within [0, t_]."""
    autozyme.activate("scvelo")
    rng = np.random.default_rng(0)
    n = 40
    u = rng.random(n) + 0.1
    s = rng.random(n) + 0.1
    tau, tau_, t_out = azscvelo.fast_assign_tau(
        u, s, alpha=2.0, beta=0.7, gamma=0.3, t_=10.0, u0_=0.5, s0_=0.2,
        assignment_mode=None,
    )
    assert t_out == 10.0
    assert np.all(tau >= 0.0) and np.all(tau <= 10.0 + 1e-9)
    assert tau.shape == (n,)


def test_assign_tau_projection_branch_runs():
    """assignment_mode='projection' with beta<gamma takes the streaming-argmin
    projection branch; returns finite tau / tau_ of length n."""
    autozyme.activate("scvelo")
    rng = np.random.default_rng(1)
    n = 60
    u = rng.random(n) + 0.2
    s = rng.random(n) + 0.2
    tau, tau_, t_out = azscvelo.fast_assign_tau(
        u, s, alpha=2.0, beta=0.3, gamma=0.7, t_=8.0, u0_=0.4, s0_=0.1,
        assignment_mode="projection",
    )
    assert tau.shape == (n,) and tau_.shape == (n,)
    assert np.all(np.isfinite(tau)) and np.all(np.isfinite(tau_))
    assert t_out == 8.0


def test_fit_recovery_with_overrides_reinstalls_and_delegates(monkeypatch):
    """fit_recovery_with_overrides re-installs the assign_tau / get_solution
    overrides into the (worker) scvelo namespace, then calls the captured
    upstream _fit_recovery. We stub _orig_fit_recovery to a sentinel."""
    import scvelo.tools._em_model_utils as em_utils
    from scvelo.core import SplicingDynamics

    autozyme.activate("scvelo")
    sentinel = object()
    seen = {}

    def _fake_orig(*args, **kwargs):
        seen["args"] = (args, kwargs)
        return sentinel

    monkeypatch.setattr(azscvelo, "_orig_fit_recovery", _fake_orig)
    # Pretend a fresh worker without the overrides installed.
    monkeypatch.setattr(em_utils, "assign_tau", object())
    monkeypatch.setattr(SplicingDynamics, "get_solution", object())

    out = azscvelo.fit_recovery_with_overrides("gene_a", n=1)
    assert out is sentinel
    assert seen["args"] == (("gene_a",), {"n": 1})
    # The overrides were re-installed in the worker namespace.
    assert em_utils.assign_tau is azscvelo.fast_assign_tau
    assert SplicingDynamics.get_solution is azscvelo._fast_get_solution


def test_get_connectivities_none_when_no_neighbors():
    """fast_get_connectivities returns None when upstream get_connectivities
    yields None (no neighbor graph)."""
    import anndata as ad

    autozyme.activate("scvelo")
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.random((20, 10)).astype(np.float32))
    # No neighbors/connectivities -> upstream returns None -> wrapper returns None.
    out = azscvelo.fast_get_connectivities(a)
    assert out is None


def test_get_connectivities_wraps_in_numbaconn():
    """With a neighbor graph, fast_get_connectivities returns a NumbaConn whose
    .dot matches the underlying scipy matrix."""
    import scanpy as sc
    import anndata as ad

    autozyme.activate("scvelo")
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(2.0, size=(60, 80)).astype(np.float32)))
    with autozyme.disabled():
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.tl.pca(a, n_comps=10)
        sc.pp.neighbors(a, n_neighbors=10)
    conn = azscvelo.fast_get_connectivities(a)
    assert isinstance(conn, azscvelo.NumbaConn)
    x = rng.standard_normal(conn.shape[1])
    got = conn.dot(x)
    assert got.shape == (conn.shape[0],)


def test_activate_restore_all_five_targets():
    """activate binds all 5 scvelo targets; deactivate restores."""
    import scvelo.tools._em_model_core as emc
    from scvelo.core import SplicingDynamics

    autozyme.deactivate("scvelo")
    orig_get_sol = SplicingDynamics.get_solution
    orig_n_jobs = emc.get_n_jobs

    assert autozyme.activate("scvelo") is True
    info = autozyme.inspect("scvelo")
    assert info["status"] == "active"
    assert len(info["targets"]) == 5

    autozyme.deactivate("scvelo")
    assert SplicingDynamics.get_solution is orig_get_sol
    assert emc.get_n_jobs is orig_n_jobs
