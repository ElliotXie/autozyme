"""Unit tests for autozyme.scanpy._highly_variable.

Tests the numba kernels directly:
  _hvg_one_pass        — per-column mean/dispersion/normalized-dispersion
  _v3_batch_mean_var   — per-batch + overall mean/variance + count-data check
  _v3_batch_clip_sum   — clipped sums/sumsq

plus the _patched_hvg dispatch logic and (with scanpy) end-to-end parity of
flavor='seurat' against vanilla scanpy on a tiny CSR matrix.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from autozyme.scanpy import _highly_variable as HV


pytestmark = pytest.mark.skipif(not HV.HAS_NUMBA, reason="numba not installed")


def _nt():
    """Active numba thread count.

    The parallel kernels index per-thread buffers by ``numba.get_thread_id()``
    with ``boundscheck=False``. The kernels now size those buffers internally
    by ``numba.get_num_threads()`` (the live pool), so the ``n_threads`` arg is
    no longer load-bearing for buffer safety: passing a smaller value can no
    longer write out of bounds. Production callers still pass the live pool
    count, which this helper returns.
    """
    import numba
    return numba.get_num_threads()


def _csr_lognorm(rows):
    """Build a tiny CSR matrix of log1p-style values (float32)."""
    m = sparse.csr_matrix(np.asarray(rows, dtype=np.float32))
    m.indptr = m.indptr.astype(np.int32)
    m.indices = m.indices.astype(np.int32)
    return m


# ==========================================================================
# _hvg_one_pass — mean + dispersion via single-pass sum / sumsq
# ==========================================================================

def test_hvg_one_pass_mean_matches_expm1_reference():
    # Stock seurat HVG works on expm1(X). Build a small matrix and check the
    # per-column mean of expm1(values) matches the kernel's `mean` output.
    rows = [[0.5, 1.0, 0.0],
            [0.8, 0.0, 1.5],
            [0.2, 0.3, 0.7]]
    X = _csr_lognorm(rows)
    n_rows, n_cols = X.shape
    mean, dispersion, disp_norm = HV._hvg_one_pass(
        X.data, X.indices, X.indptr, n_cols, n_rows, 20, _nt())

    dense = np.asarray(rows, dtype=np.float64)
    expm1 = np.expm1(dense)
    ref_mean = expm1.mean(axis=0)
    # Kernel guards mean==0 -> 1e-12; none of our columns are all-zero.
    np.testing.assert_allclose(mean, ref_mean, rtol=1e-5)


def test_hvg_one_pass_dispersion_matches_var_over_mean():
    rows = [[0.5, 1.0],
            [0.8, 2.0],
            [0.2, 0.3],
            [0.9, 1.1]]
    X = _csr_lognorm(rows)
    n_rows, n_cols = X.shape
    mean, dispersion, _ = HV._hvg_one_pass(
        X.data, X.indices, X.indptr, n_cols, n_rows, 20, _nt())

    expm1 = np.expm1(np.asarray(rows, dtype=np.float64))
    m = expm1.mean(axis=0)
    v = expm1.var(axis=0, ddof=1)  # sample variance (n-1)
    ref_disp = np.log(v / m)
    np.testing.assert_allclose(dispersion, ref_disp, rtol=1e-4, atol=1e-4)


def test_hvg_one_pass_disp_norm_finite_and_centered():
    rng = np.random.default_rng(0)
    dense = (rng.random((20, 12)) * 1.5).astype(np.float32)
    X = _csr_lognorm(dense.tolist())
    mean, dispersion, disp_norm = HV._hvg_one_pass(
        X.data, X.indices, X.indptr, 12, 20, 20, _nt())
    # Normalized dispersions are finite where dispersion is finite.
    finite = ~np.isnan(dispersion)
    assert np.all(np.isfinite(disp_norm[finite]))


def test_hvg_one_pass_thread_reduction_matches_serial_reference():
    # The per-thread buffers reduce to the correct serial result; verify the
    # parallel kernel (at the live thread count) matches a numpy reference.
    rng = np.random.default_rng(1)
    dense = (rng.random((30, 8)) * 1.2).astype(np.float32)
    X = _csr_lognorm(dense.tolist())
    mean, disp, _ = HV._hvg_one_pass(
        X.data, X.indices, X.indptr, 8, 30, 20, _nt())
    expm1 = np.expm1(dense.astype(np.float64))
    np.testing.assert_allclose(mean, expm1.mean(axis=0), rtol=1e-5)
    v = expm1.var(axis=0, ddof=1)
    np.testing.assert_allclose(disp, np.log(v / expm1.mean(axis=0)),
                               rtol=1e-4, atol=1e-4)


# ==========================================================================
# _v3_batch_mean_var
# ==========================================================================

def test_v3_batch_mean_var_single_batch_matches_numpy():
    rows = [[1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [0.0, 8.0]]
    X = _csr_lognorm(rows)
    codes = np.zeros(4, dtype=np.int32)  # all in batch 0
    batch_sizes = np.array([4], dtype=np.int64)
    means, variances, om, ov, bad = HV._v3_batch_mean_var(
        X.data, X.indices, X.indptr, codes, 2, 1, batch_sizes, _nt(), False)

    dense = np.asarray(rows, dtype=np.float64)
    np.testing.assert_allclose(means[0], dense.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(variances[0], dense.var(axis=0, ddof=1), rtol=1e-6)
    # Single batch -> overall == batch.
    np.testing.assert_allclose(om, dense.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(ov, dense.var(axis=0, ddof=1), rtol=1e-6)
    assert bad is False


def test_v3_batch_mean_var_two_batches():
    rows = [[2.0, 0.0],   # batch 0
            [4.0, 6.0],   # batch 0
            [1.0, 3.0],   # batch 1
            [3.0, 5.0]]   # batch 1
    X = _csr_lognorm(rows)
    codes = np.array([0, 0, 1, 1], dtype=np.int32)
    batch_sizes = np.array([2, 2], dtype=np.int64)
    means, variances, om, ov, bad = HV._v3_batch_mean_var(
        X.data, X.indices, X.indptr, codes, 2, 2, batch_sizes, _nt(), False)
    dense = np.asarray(rows, dtype=np.float64)
    np.testing.assert_allclose(means[0], dense[:2].mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(means[1], dense[2:].mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(om, dense.mean(axis=0), rtol=1e-9)


def test_v3_batch_mean_var_count_check_flags_noninteger():
    rows = [[1.0, 2.5],   # 2.5 is non-integer -> bad
            [3.0, 4.0]]
    X = _csr_lognorm(rows)
    codes = np.zeros(2, dtype=np.int32)
    batch_sizes = np.array([2], dtype=np.int64)
    *_, bad = HV._v3_batch_mean_var(
        X.data, X.indices, X.indptr, codes, 2, 1, batch_sizes, _nt(), True)
    assert bad is True


def test_v3_batch_mean_var_count_check_passes_integers():
    rows = [[1.0, 2.0], [3.0, 4.0]]
    X = _csr_lognorm(rows)
    codes = np.zeros(2, dtype=np.int32)
    batch_sizes = np.array([2], dtype=np.int64)
    *_, bad = HV._v3_batch_mean_var(
        X.data, X.indices, X.indptr, codes, 2, 1, batch_sizes, _nt(), True)
    assert bad is False


# ==========================================================================
# _v3_batch_clip_sum
# ==========================================================================

def test_v3_batch_clip_sum_no_clip():
    rows = [[1.0, 2.0], [3.0, 4.0]]
    X = _csr_lognorm(rows)
    codes = np.zeros(2, dtype=np.int32)
    clip_vals = np.full((1, 2), 1e9, dtype=np.float64)  # huge cap -> no clip
    out_sum, out_sumsq = HV._v3_batch_clip_sum(
        X.data, X.indices, X.indptr, codes, 2, 1, clip_vals, _nt())
    dense = np.asarray(rows, dtype=np.float64)
    np.testing.assert_allclose(out_sum[0], dense.sum(axis=0))
    np.testing.assert_allclose(out_sumsq[0], (dense * dense).sum(axis=0))


def test_v3_batch_clip_sum_applies_cap():
    rows = [[10.0, 1.0], [20.0, 2.0]]
    X = _csr_lognorm(rows)
    codes = np.zeros(2, dtype=np.int32)
    clip_vals = np.array([[5.0, 1e9]], dtype=np.float64)  # cap col 0 at 5
    out_sum, out_sumsq = HV._v3_batch_clip_sum(
        X.data, X.indices, X.indptr, codes, 2, 1, clip_vals, _nt())
    # col 0: both 10 and 20 clipped to 5 -> sum 10, sumsq 50
    assert out_sum[0, 0] == pytest.approx(10.0)
    assert out_sumsq[0, 0] == pytest.approx(50.0)
    # col 1 uncapped: 1 + 2 = 3, 1 + 4 = 5
    assert out_sum[0, 1] == pytest.approx(3.0)
    assert out_sumsq[0, 1] == pytest.approx(5.0)


# ==========================================================================
# _patched_hvg dispatch
# ==========================================================================

def test_patched_hvg_zyme_false_delegates(monkeypatch):
    monkeypatch.setattr(HV, "_orig_hvg", lambda: (lambda adata, **kw: "ORIG"))
    assert HV._patched_hvg("ADATA", zyme=False) == "ORIG"


def test_patched_hvg_batch_key_non_v3_delegates(monkeypatch):
    monkeypatch.setattr(HV, "_orig_hvg", lambda: (lambda adata, **kw: "ORIG"))
    # flavor='seurat' (not v3) + batch_key -> upstream.
    assert HV._patched_hvg("ADATA", flavor="seurat", batch_key="b") == "ORIG"


def test_patched_hvg_unknown_flavor_delegates(monkeypatch):
    monkeypatch.setattr(HV, "_orig_hvg", lambda: (lambda adata, **kw: "ORIG"))
    assert HV._patched_hvg("ADATA", flavor="cell_ranger") == "ORIG"


def test_patched_hvg_seurat_v3_batch_routes(monkeypatch):
    routed = {}
    monkeypatch.setattr(
        HV, "_fast_hvg_seurat_v3_batch",
        lambda adata, **kw: routed.setdefault("v3", True))
    HV._patched_hvg("ADATA", flavor="seurat_v3", batch_key="b")
    assert routed.get("v3")


def test_patched_hvg_seurat_routes_to_fast(monkeypatch):
    routed = {}
    monkeypatch.setattr(
        HV, "_fast_hvg_seurat",
        lambda adata, **kw: routed.setdefault("seurat", True))
    HV._patched_hvg("ADATA", flavor="seurat")
    assert routed.get("seurat")


# ==========================================================================
# _fast_hvg_seurat — end-to-end parity vs vanilla scanpy
# ==========================================================================

def _adata_lognorm(n_cells=60, n_genes=30, seed=0):
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.6, size=(n_cells, n_genes)).astype(np.float32)
    X = sparse.csr_matrix(counts)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    a = ad.AnnData(X)
    import scanpy as sc
    with __import__("autozyme").disabled():
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    return a


def test_fast_hvg_seurat_selection_matches_vanilla_top_genes():
    sc = pytest.importorskip("scanpy")
    import autozyme
    a = _adata_lognorm(seed=2)
    a_van = a.copy()
    with autozyme.disabled():
        sc.pp.highly_variable_genes(a_van, flavor="seurat", n_top_genes=10)
    # Fast path (direct call, autozyme need not be active for the helper).
    HV._fast_hvg_seurat(a, n_top_genes=10)
    # The set of selected HVGs should match vanilla exactly (deterministic
    # dispersion ranking on identical data).
    fast_sel = set(np.where(a.var["highly_variable"].values)[0])
    van_sel = set(np.where(a_van.var["highly_variable"].values)[0])
    assert fast_sel == van_sel


def test_fast_hvg_seurat_inplace_writes_columns():
    pytest.importorskip("scanpy")
    a = _adata_lognorm(seed=3)
    out = HV._fast_hvg_seurat(a, n_top_genes=8)
    assert out is None
    for col in ("highly_variable", "means", "dispersions", "dispersions_norm"):
        assert col in a.var.columns
    assert a.uns["hvg"] == {"flavor": "seurat"}
    # n_top_genes selects at least n_top; ties at the cutoff can include a few
    # extra (matches scanpy's `disp_norm >= cutoff` semantics).
    assert int(a.var["highly_variable"].sum()) >= 8


def test_fast_hvg_seurat_not_inplace_returns_dataframe():
    pytest.importorskip("scanpy")
    pd = pytest.importorskip("pandas")
    a = _adata_lognorm(seed=4)
    df = HV._fast_hvg_seurat(a, n_top_genes=8, inplace=False)
    assert isinstance(df, pd.DataFrame)
    assert "highly_variable" in df.columns
    assert int(df["highly_variable"].sum()) >= 8


def test_fast_hvg_seurat_dense_delegates(monkeypatch):
    ad = pytest.importorskip("anndata")
    monkeypatch.setattr(HV, "_orig_hvg", lambda: (lambda adata, **kw: "ORIG"))
    a = ad.AnnData(np.ones((10, 5), dtype=np.float32))  # dense .X
    assert HV._fast_hvg_seurat(a) == "ORIG"
