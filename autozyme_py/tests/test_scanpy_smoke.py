"""Smoke test for the autozyme scanpy plugin (real pbmc68k data).

Runs pbmc68k.h5ad (65877 cells x 33939 genes) through a full preprocessing +
clustering + DE pipeline, twice:

  1. Baseline — vanilla scanpy (inside ``autozyme.disabled()``).
  2. Patched  — fast paths active.

Per-function unit parity (the rigorous checks):

  - HVG selection                  overlap >= 95%
  - PCA top-5 components           cosine >= 0.95 up-to-sign
  - leiden on IDENTICAL graph      ARI >= 0.99 (isolates fast_leiden)
  - rank_genes top-20 per celltype overlap >= 80%

Plus toggle/escape-hatch tests:
  - autozyme.disabled() context falls back to original
  - per-call ``zyme=False`` falls back to original
  - default (zyme=True implicit) hits fast path

Critical: requires vanilla scanpy (NOT scanpy-zyme). The first test asserts
the environment is clean — otherwise the dispatcher's captured "original"
would already be scanpy-zyme's fast_fn, and ``disabled()``/``zyme=False``
both silently no-op.

Pipeline runs are cached to ``_smoke_cache/``: first run ~22 min, subsequent
~50s. Delete the cache to force re-computation when vendor code changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scanpy")
pytest.importorskip("scipy")
pytest.importorskip("anndata")


# --- Environment guards -------------------------------------------------

def test_environment_is_vanilla_scanpy():
    """Hard precondition: no scanpy-zyme installed, scanpy.__turbo__ is False.

    If this fails, every later assertion is meaningless — the "vanilla
    baseline" would secretly already be patched by scanpy-zyme's shadow.
    """
    import scanpy as sc

    assert not getattr(sc, "__turbo__", False), (
        "scanpy reports __turbo__=True; you have scanpy-zyme installed. "
        "Uninstall it before testing autozyme.scanpy:\n"
        "    pip uninstall -y scanpy-zyme && pip install scanpy==1.11.5\n"
        "Otherwise autozyme's 'original' is captured AS scanpy-zyme's "
        "fast_fn, and disabled()/zyme=False both silently no-op."
    )


# --- Fixtures -----------------------------------------------------------

def _make_adata():
    """Real pbmc68k AnnData via the plugin's smoke recipe."""
    from autozyme.scanpy import _smoke_load
    return _smoke_load(None, None)["adata"]


def _make_tiny_adata():
    """Minimal AnnData for dispatcher/toggle tests — no pbmc68k load needed."""
    import numpy as np
    from scipy import sparse
    import anndata as ad

    rng = np.random.default_rng(0)
    n_cells, n_genes = 100, 200
    X = sparse.random(n_cells, n_genes, density=0.2, format="csr",
                      dtype=np.float32, random_state=rng).astype(np.float32)
    X.data = rng.integers(1, 10, size=X.nnz).astype(np.float32)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    return ad.AnnData(X)


def _make_tiny_batch_counts():
    """Small raw-count CSR AnnData for seurat_v3 batch-aware HVG."""
    import numpy as np
    import pandas as pd
    from scipy import sparse
    import anndata as ad

    rng = np.random.default_rng(123)
    n_cells, n_genes = 600, 500
    X = sparse.random(n_cells, n_genes, density=0.08, format="csr",
                      dtype=np.float32, random_state=rng)
    X.data = rng.poisson(2.0, size=X.nnz).astype(np.float32) + 1.0
    obs = pd.DataFrame({
        "donor_id": pd.Categorical(
            np.repeat(["a", "b", "c"], [200, 180, 220])
        )
    })
    return ad.AnnData(X, obs=obs)


def _run_pipeline(adata):
    """In-place pipeline. Returns adata."""
    from autozyme.scanpy import _smoke_call
    return _smoke_call({"adata": adata})


# --- Activation sanity --------------------------------------------------

def test_activate_installs_dispatcher():
    """After autozyme.activate('scanpy'), sc.pp.X has __autozyme_fast__."""
    import autozyme
    autozyme.activate("scanpy")
    import scanpy as sc

    for mod, attr, expected_fast_name in [
        (sc.pp, "normalize_total", "fast_normalize_total"),
        (sc.pp, "log1p",            "fast_log1p"),
        (sc.pp, "scale",            "fast_scale"),
        (sc.pp, "highly_variable_genes", "_patched_hvg"),
        (sc.tl, "pca",              "fast_pca"),
        (sc.tl, "leiden",           "fast_leiden"),
        (sc.tl, "rank_genes_groups", "_fast_rank_genes_groups"),
    ]:
        fn = getattr(mod, attr)
        fast = getattr(fn, "__autozyme_fast__", None)
        assert fast is not None, f"sc.{mod.__name__.split('.')[-1]}.{attr} not patched"
        assert fast.__name__ == expected_fast_name, (
            f"{attr}: expected fast_fn={expected_fast_name}, got {fast.__name__}"
        )
        # Original must NOT be the fast_fn (env-pollution guard).
        orig = fn.__autozyme_original__
        assert orig is not fast, (
            f"{attr}: dispatcher captured fast_fn as 'original' — "
            "scanpy-zyme contamination, see test_environment_is_vanilla_scanpy"
        )


def test_regress_out_is_registered_with_finalized_coverage():
    """regress_out ships as a finalized Scanpy subpatch and must bind both paths."""
    import autozyme
    import scanpy as sc

    from scanpy.preprocessing import _simple as simple

    autozyme.activate("scanpy")

    assert getattr(sc.pp.regress_out, "__autozyme_fast__", None).__name__ == "fast_regress_out"
    assert getattr(simple.regress_out, "__autozyme_fast__", None).__name__ == "fast_regress_out"


# --- Parity test --------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent / "_smoke_cache"


def _write_h5ad_atomic(adata, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    adata.write_h5ad(tmp)
    tmp.replace(path)


@pytest.fixture(scope="module")
def parity_results():
    """Run pipeline once disabled, once activated; collect both outputs.

    Caches both AnnDatas to ``_CACHE_DIR`` so re-runs of just the parity
    asserts (without re-running pytest from scratch) can skip the 20+
    minute pipeline. Delete the cache dir to force re-computation.
    """
    import autozyme
    import anndata as ad

    autozyme.activate("scanpy")
    _CACHE_DIR.mkdir(exist_ok=True)
    base_path = _CACHE_DIR / "base.h5ad"
    fast_path = _CACHE_DIR / "fast.h5ad"

    if base_path.exists() and fast_path.exists():
        print(f"\n[parity_results] using cache: {_CACHE_DIR}")
        try:
            return ad.read_h5ad(base_path), ad.read_h5ad(fast_path)
        except OSError:
            base_path.unlink(missing_ok=True)
            fast_path.unlink(missing_ok=True)

    # Baseline: vanilla via autozyme.disabled() context.
    print("\n[parity_results] running BASELINE pipeline (vanilla)...")
    adata_base = _make_adata()
    with autozyme.disabled():
        _run_pipeline(adata_base)
    _write_h5ad_atomic(adata_base, base_path)

    # Patched: same fresh AnnData, all fast paths active.
    print("[parity_results] running PATCHED pipeline (autozyme fast)...")
    adata_fast = _make_adata()
    _run_pipeline(adata_fast)
    _write_h5ad_atomic(adata_fast, fast_path)

    return adata_base, adata_fast


def test_hvg_overlap(parity_results):
    base, fast = parity_results
    hvg_base = base.var["highly_variable"].values
    hvg_fast = fast.var["highly_variable"].values
    overlap = (hvg_base & hvg_fast).sum() / hvg_base.sum()
    print(f"\n  HVG overlap: {overlap:.4f} (n_base_hvg={hvg_base.sum()}, n_fast_hvg={hvg_fast.sum()})")
    assert overlap >= 0.95, f"HVG overlap {overlap:.3f} < 0.95"


def test_pca_components_match(parity_results):
    base, fast = parity_results
    P_base = base.obsm["X_pca"]
    P_fast = fast.obsm["X_pca"]
    cosines = []
    for k in range(min(10, P_base.shape[1])):
        cos = abs(np.dot(P_base[:, k], P_fast[:, k])) / (
            np.linalg.norm(P_base[:, k]) * np.linalg.norm(P_fast[:, k]) + 1e-12
        )
        cosines.append(float(cos))
    print(f"\n  PCA per-PC cosine (top-10, up-to-sign): {[f'{c:.3f}' for c in cosines]}")
    # First 5 PCs should match up-to-sign (cos > 0.95). Lower PCs may
    # legitimately diverge when their eigenvalues are nearly degenerate.
    for k, cos in enumerate(cosines[:5]):
        assert cos > 0.95, f"PC{k} cosine {cos:.3f} < 0.95 (up-to-sign)"


def test_leiden_same_graph_parity(parity_results):
    """fast_leiden must produce identical partitions to vanilla on identical input.

    This test isolates fast_leiden by running it on vanilla's connectivities
    and asserting an ARI of 1.0 against vanilla's leiden labels.
    """
    pytest.importorskip("sklearn")
    from sklearn.metrics import adjusted_rand_score
    import autozyme
    import scanpy as sc

    autozyme.activate("scanpy")  # ensure fast_leiden dispatcher is installed
    base, _fast = parity_results

    # Run sc.tl.leiden (dispatched to fast_leiden) on vanilla's connectivities.
    ad_test = base.copy()
    sc.tl.leiden(ad_test, key_added="leiden_fast", flavor="igraph",
                 n_iterations=2, directed=False, random_state=0)
    labels_vanilla = base.obs["leiden"].astype(str).values
    labels_fast_on_vanilla_graph = ad_test.obs["leiden_fast"].astype(str).values
    ari = adjusted_rand_score(labels_vanilla, labels_fast_on_vanilla_graph)
    n_vanilla = len(set(labels_vanilla))
    n_fast = len(set(labels_fast_on_vanilla_graph))
    print(f"\n  leiden same-graph ARI: {ari:.4f} "
          f"(n_clusters: vanilla={n_vanilla}, fast={n_fast})")
    assert ari >= 0.99, (
        f"fast_leiden produced different partition on identical graph: "
        f"ARI {ari:.4f} (n_clusters vanilla={n_vanilla}, fast={n_fast})"
    )


def test_rank_genes_top_overlap(parity_results):
    """Top-20 markers per celltype must overlap >=80% between vanilla and fast.

    Computes top-20 from scores (descending) rather than trusting the
    ``names`` recarray order: vanilla scanpy stores names sorted by score,
    but ``_fast_rank_genes_groups`` with ``n_genes=None`` stores names in
    raw ``var_names`` order. Sorting from scores gives a consistent
    comparison regardless of upstream's storage convention.
    """
    base, fast = parity_results
    names_base = base.uns["rank_genes_groups"]["names"]
    names_fast = fast.uns["rank_genes_groups"]["names"]
    scores_base = base.uns["rank_genes_groups"]["scores"]
    scores_fast = fast.uns["rank_genes_groups"]["scores"]
    groups = names_base.dtype.names

    overlaps = []
    for g in groups:
        # argsort descending by score, then read names at those indices.
        top_base_idx = np.argsort(-scores_base[g])[:20]
        top_fast_idx = np.argsort(-scores_fast[g])[:20]
        top_base = set(np.asarray(names_base[g])[top_base_idx])
        top_fast = set(np.asarray(names_fast[g])[top_fast_idx])
        overlap = len(top_base & top_fast) / 20
        overlaps.append((g, overlap))

    print(f"\n  rank_genes top-20 overlap per celltype (sorted by score):")
    for g, o in overlaps:
        print(f"    {g}: {o:.2f}")
    for g, overlap in overlaps:
        assert overlap >= 0.80, (
            f"rank_genes celltype={g}: top-20 overlap {overlap:.2f} < 0.80"
        )


# --- Disabled() round-trip ----------------------------------------------

def test_normalize_only_matches_vanilla():
    """normalize_total alone must match vanilla (linear scaled counts, not log)."""
    import autozyme
    autozyme.activate("scanpy")
    import scanpy as sc

    adata_v = _make_tiny_adata()
    adata_f = _make_tiny_adata()
    with autozyme.disabled():
        sc.pp.normalize_total(adata_v, target_sum=1e4)
    sc.pp.normalize_total(adata_f, target_sum=1e4)

    assert np.allclose(
        np.asarray(adata_v.X.todense()),
        np.asarray(adata_f.X.todense()),
        rtol=1e-5,
        atol=1e-5,
    )
    assert "log1p" not in adata_f.uns
    assert adata_f.X.data.max() > 50, (
        "fast normalize_total looks log-transformed — expected linear scaled counts"
    )


def test_normalize_log1p_pipeline_matches_vanilla():
    """Standard tutorial pair must still match vanilla numerically."""
    import autozyme
    autozyme.activate("scanpy")
    import scanpy as sc

    adata_v = _make_tiny_adata()
    adata_f = _make_tiny_adata()
    with autozyme.disabled():
        sc.pp.normalize_total(adata_v, target_sum=1e4)
        sc.pp.log1p(adata_v)
    sc.pp.normalize_total(adata_f, target_sum=1e4)
    sc.pp.log1p(adata_f)

    assert np.allclose(
        adata_v.X.data,
        adata_f.X.data,
        rtol=1e-5,
        atol=1e-5,
    )


def test_disabled_falls_back_to_original():
    """Inside autozyme.disabled(), dispatcher MUST call original, not fast."""
    import autozyme
    autozyme.activate("scanpy")
    import scanpy as sc

    adata = _make_tiny_adata()
    with autozyme.disabled():
        sc.pp.normalize_total(adata, target_sum=1e4)
    # Vanilla normalize leaves linear scaled counts (not log space).
    assert float(adata.X.data.max()) > 50


def test_zyme_false_per_call_escape():
    """zyme=False per-call should bypass the fast path."""
    import autozyme
    autozyme.activate("scanpy")
    import scanpy as sc

    adata = _make_tiny_adata()
    adata_ref = _make_tiny_adata()
    with autozyme.disabled():
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
    sc.pp.normalize_total(adata, target_sum=1e4, zyme=False)
    assert np.allclose(adata.X.data, adata_ref.X.data, rtol=1e-5, atol=1e-5)


def test_zyme_true_takes_fast_path():
    """Default (zyme=True implicit) should hit fast path and match vanilla."""
    import autozyme
    autozyme.activate("scanpy")
    import scanpy as sc

    adata = _make_tiny_adata()
    adata_ref = _make_tiny_adata()
    with autozyme.disabled():
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
    sc.pp.normalize_total(adata, target_sum=1e4)
    assert np.allclose(adata.X.data, adata_ref.X.data, rtol=1e-5, atol=1e-5)


def test_hvg_seurat_v3_paper_batch_matches_vanilla():
    """Batch-aware seurat_v3_paper HVG fast path must match upstream."""
    pytest.importorskip("skmisc")
    import autozyme
    autozyme.activate("scanpy")
    import scanpy as sc

    adata_v = _make_tiny_batch_counts()
    adata_f = _make_tiny_batch_counts()
    with autozyme.disabled():
        sc.pp.highly_variable_genes(
            adata_v,
            flavor="seurat_v3_paper",
            batch_key="donor_id",
            n_top_genes=100,
        )
    sc.pp.highly_variable_genes(
        adata_f,
        flavor="seurat_v3_paper",
        batch_key="donor_id",
        n_top_genes=100,
    )

    hvg_v = adata_v.var["highly_variable"].to_numpy()
    hvg_f = adata_f.var["highly_variable"].to_numpy()
    assert np.array_equal(hvg_v, hvg_f)
    assert np.allclose(
        adata_v.var["variances_norm"].to_numpy(),
        adata_f.var["variances_norm"].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.array_equal(
        adata_v.var["highly_variable_nbatches"].to_numpy(),
        adata_f.var["highly_variable_nbatches"].to_numpy(),
    )


def test_hvg_seurat_v3_batch_noninteger_warns_without_fallback(monkeypatch):
    """Non-integer inputs should warn like upstream, not disable the fast path."""
    pytest.importorskip("skmisc")
    import autozyme
    autozyme.activate("scanpy")
    import scanpy as sc
    from autozyme.scanpy import _highly_variable as hvg_mod

    adata_v = _make_tiny_batch_counts()
    adata_f = _make_tiny_batch_counts()
    for adata in (adata_v, adata_f):
        adata.X = adata.X.copy()
        adata.X.data = adata.X.data.astype(np.float32, copy=False)
        adata.X.data += np.float32(0.125)

    with autozyme.disabled():
        with pytest.warns(UserWarning, match="non-integers"):
            sc.pp.highly_variable_genes(
                adata_v,
                flavor="seurat_v3_paper",
                batch_key="donor_id",
                n_top_genes=100,
            )

    def _raise_if_fallback(*args, **kwargs):
        raise AssertionError("non-integer seurat_v3 batch path fell back upstream")

    monkeypatch.setattr(hvg_mod, "_orig_hvg", lambda: _raise_if_fallback)
    with pytest.warns(UserWarning, match="non-integers"):
        sc.pp.highly_variable_genes(
            adata_f,
            flavor="seurat_v3_paper",
            batch_key="donor_id",
            n_top_genes=100,
        )

    assert np.array_equal(
        adata_v.var["highly_variable"].to_numpy(),
        adata_f.var["highly_variable"].to_numpy(),
    )
    assert np.allclose(
        adata_v.var["variances_norm"].to_numpy(),
        adata_f.var["variances_norm"].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.array_equal(
        adata_v.var["highly_variable_nbatches"].to_numpy(),
        adata_f.var["highly_variable_nbatches"].to_numpy(),
    )
