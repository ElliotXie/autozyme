"""End-to-end dispatch / activate-restore / prepare coverage for the scanpy patch.

Covers the pure-python wrapper/dispatch lines coverage.py CAN see:
  * ``autozyme.activate("scanpy")`` / ``deactivate`` / ``deactivate_all``
    round-trip (the dispatcher install + restore in autozyme/_core.py and the
    ``__autozyme_original__`` lookup helpers in each submodule).
  * The single registered patch name is ``"scanpy"`` (NOT ``scanpy_<x>``);
    this asserts that and exercises ``list_patches`` / ``status`` / ``inspect``.
  * ``zyme_prepare`` dtype coercion (``_prepare.py``) including the
    copy=True/False and layers branches and the already-canonical no-op path.

The conftest autouse fixture deactivates between tests, so each test activates.

Env recipe: KMP_DUPLICATE_LIB_OK=TRUE NUMBA_THREADING_LAYER=workqueue
NUMBA_NUM_THREADS=14 OMP_NUM_THREADS=2 (see BRIEFING2).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

sc = pytest.importorskip("scanpy")
ad = pytest.importorskip("anndata")
import autozyme  # noqa: E402
from autozyme.scanpy import zyme_prepare  # noqa: E402


# --------------------------------------------------------------------------
# tiny deterministic AnnData builders
# --------------------------------------------------------------------------

def _counts(n_obs=40, n_vars=60, seed=0, dtype=np.float64, int_idx=True):
    """Deterministic sparse CSR count matrix wrapped in AnnData."""
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.4, size=(n_obs, n_vars)).astype(dtype)
    X = sparse.csr_matrix(dense)
    if not int_idx:
        X.indptr = X.indptr.astype(np.int64)
        X.indices = X.indices.astype(np.int64)
    adata = ad.AnnData(X)
    adata.obs_names = [f"c{i}" for i in range(n_obs)]
    adata.var_names = [f"g{j}" for j in range(n_vars)]
    return adata


# --------------------------------------------------------------------------
# patch registration / discovery
# --------------------------------------------------------------------------

def test_scanpy_is_a_listed_patch():
    # The patch is registered under the single name "scanpy".
    assert "scanpy" in autozyme.list_patches()


def test_scanpy_installed_probe_true():
    # scanpy is importable here, so the installed-filter should keep it.
    assert "scanpy" in autozyme.list_patches(installed=True)


def test_status_inactive_before_activate():
    st = autozyme.status()
    assert st.get("scanpy") == "inactive"


# --------------------------------------------------------------------------
# activate / deactivate round-trip + dispatcher attributes
# --------------------------------------------------------------------------

def test_activate_returns_true_and_binds_dispatcher():
    assert autozyme.activate("scanpy") is True
    # sc.pp.normalize_total is now the autozyme dispatcher wrapper.
    assert hasattr(sc.pp.normalize_total, "__autozyme_fast__")
    assert hasattr(sc.pp.normalize_total, "__autozyme_original__")
    assert autozyme.status()["scanpy"] == "active"


def test_activate_idempotent_returns_true_when_already_active():
    assert autozyme.activate("scanpy") is True
    # Second activate hits the `if p.injected: return True` short-circuit.
    assert autozyme.activate("scanpy") is True


def test_deactivate_restores_originals():
    orig = sc.pp.normalize_total
    autozyme.activate("scanpy")
    assert sc.pp.normalize_total is not orig
    autozyme.deactivate("scanpy")
    # The original callable is restored (no dispatcher attribute).
    assert not hasattr(sc.pp.normalize_total, "__autozyme_fast__")
    assert autozyme.status()["scanpy"] == "inactive"


def test_deactivate_idempotent_when_not_active():
    # Never activated in this test → deactivate is a silent no-op.
    autozyme.deactivate("scanpy")
    assert autozyme.status()["scanpy"] == "inactive"


def test_deactivate_all_clears_everything():
    autozyme.activate("scanpy")
    autozyme.deactivate_all()
    assert autozyme.status()["scanpy"] == "inactive"


def test_unknown_patch_name_raises_keyerror():
    with pytest.raises(KeyError):
        autozyme.activate("scanpy_normalize")  # not a real patch name


def test_inspect_reports_targets_bound_to_fast():
    autozyme.activate("scanpy")
    info = autozyme.inspect("scanpy")
    assert info["status"] == "active"
    assert info["name"] == "scanpy"
    bound = {(t["upstream"], t["attr"]): t["currently_bound_to_fast"]
             for t in info["targets"]}
    # The public namespace normalize_total target is bound to its fast fn.
    assert bound[("scanpy.preprocessing", "normalize_total")] is True


def test_env_snapshot_includes_scanpy():
    snap = autozyme.env_snapshot()
    names = {p["name"] for p in snap["patches"]}
    assert "scanpy" in names


# --------------------------------------------------------------------------
# disabled() context manager → dispatcher short-circuits to original
# --------------------------------------------------------------------------

def test_disabled_block_uses_original_path():
    autozyme.activate("scanpy")
    adata = _counts()
    # Inside disabled(): dispatcher forwards to upstream normalize_total.
    with autozyme.disabled():
        assert autozyme.is_disabled() is True
        out = sc.pp.normalize_total(adata, target_sum=1e4, copy=True)
    assert out is not None
    assert autozyme.is_disabled() is False


# --------------------------------------------------------------------------
# zyme_prepare (_prepare.py) dtype coercion
# --------------------------------------------------------------------------

def test_zyme_prepare_coerces_float64_int64_inplace():
    adata = _counts(dtype=np.float64, int_idx=False)
    assert adata.X.dtype == np.float64
    assert adata.X.indptr.dtype == np.int64
    ret = zyme_prepare(adata)
    assert ret is None  # in-place returns None
    assert adata.X.dtype == np.float32
    assert adata.X.indptr.dtype == np.int32
    assert adata.X.indices.dtype == np.int32


def test_zyme_prepare_copy_true_leaves_original_untouched():
    adata = _counts(dtype=np.float64, int_idx=False)
    out = zyme_prepare(adata, copy=True)
    assert out is not adata
    assert out.X.dtype == np.float32
    # Original is unchanged.
    assert adata.X.dtype == np.float64


def test_zyme_prepare_already_canonical_is_noop():
    adata = _counts(dtype=np.float32, int_idx=True)
    x_before = adata.X
    zyme_prepare(adata)
    # No new matrix object created when already canonical.
    assert adata.X is x_before


def test_zyme_prepare_coerces_layers():
    adata = _counts(dtype=np.float64, int_idx=False)
    adata.layers["raw"] = adata.X.copy()
    zyme_prepare(adata, layers=True)
    assert adata.layers["raw"].dtype == np.float32
    assert adata.layers["raw"].indptr.dtype == np.int32


def test_zyme_prepare_layers_false_skips_layers():
    adata = _counts(dtype=np.float64, int_idx=False)
    adata.layers["raw"] = adata.X.copy()
    zyme_prepare(adata, layers=False)
    # .X coerced, layer left alone.
    assert adata.X.dtype == np.float32
    assert adata.layers["raw"].dtype == np.float64


def test_zyme_prepare_dense_x_to_float32():
    adata = _counts(dtype=np.float64)
    adata.X = adata.X.toarray().astype(np.float64)  # dense float64
    zyme_prepare(adata)
    assert adata.X.dtype == np.float32


def test_zyme_prepare_non_csr_sparse_converted_to_csr():
    adata = _counts(dtype=np.float64)
    adata.X = adata.X.tocsc().astype(np.float64)  # CSC, not CSR
    zyme_prepare(adata)
    assert sparse.isspmatrix_csr(adata.X)
    assert adata.X.dtype == np.float32
