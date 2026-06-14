"""Unit tests for autozyme.scanpy._prepare and autozyme.scanpy.__init__.

_prepare: dtype coercion helpers (_coerce_matrix, _coerce_x, _coerce_layer,
zyme_prepare). __init__: the smoke recipe helpers (_smoke_load path resolution
+ error branches, _smoke_save) and the module's exported surface.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from scipy import sparse

from autozyme.scanpy import _prepare as P
import autozyme.scanpy as SC


# ==========================================================================
# _prepare._coerce_matrix
# ==========================================================================

def test_coerce_matrix_dense_float64_to_float32():
    X = np.array([[1.0, 2.0]], dtype=np.float64)
    out = P._coerce_matrix(X)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, X)


def test_coerce_matrix_dense_float32_returns_same_object():
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    out = P._coerce_matrix(X)
    assert out is X  # already canonical, no copy


def test_coerce_matrix_csr_float64_int64_coerced():
    X = sparse.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.float64))
    X.indptr = X.indptr.astype(np.int64)
    X.indices = X.indices.astype(np.int64)
    out = P._coerce_matrix(X)
    assert sparse.isspmatrix_csr(out)
    assert out.dtype == np.float32
    assert out.indptr.dtype == np.int32
    assert out.indices.dtype == np.int32


def test_coerce_matrix_csc_to_csr():
    X = sparse.csc_matrix(np.array([[1, 0], [0, 2]], dtype=np.float32))
    out = P._coerce_matrix(X)
    assert sparse.isspmatrix_csr(out)


def test_coerce_matrix_already_canonical_csr_returns_same():
    X = sparse.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.float32))
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    out = P._coerce_matrix(X)
    assert out is X


# ==========================================================================
# zyme_prepare end-to-end (needs AnnData)
# ==========================================================================

def _adata_f64():
    ad = pytest.importorskip("anndata")
    X = sparse.csr_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float64))
    X.indptr = X.indptr.astype(np.int64)
    X.indices = X.indices.astype(np.int64)
    return ad.AnnData(X)


def test_zyme_prepare_inplace_coerces_X():
    pytest.importorskip("anndata")
    a = _adata_f64()
    out = P.zyme_prepare(a)
    assert out is None  # inplace -> None
    assert a.X.dtype == np.float32
    assert a.X.indptr.dtype == np.int32
    assert a.X.indices.dtype == np.int32


def test_zyme_prepare_copy_returns_new():
    pytest.importorskip("anndata")
    a = _adata_f64()
    out = P.zyme_prepare(a, copy=True)
    assert out is not None and out is not a
    assert out.X.dtype == np.float32
    # Original unchanged.
    assert a.X.dtype == np.float64


def test_zyme_prepare_coerces_layers():
    ad = pytest.importorskip("anndata")
    X = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    a = ad.AnnData(X)
    a.layers["counts"] = sparse.csr_matrix(
        np.array([[1, 0], [0, 2]], dtype=np.float64))
    P.zyme_prepare(a, layers=True)
    assert a.layers["counts"].dtype == np.float32


def test_zyme_prepare_layers_false_leaves_layers():
    ad = pytest.importorskip("anndata")
    X = sparse.csr_matrix(np.eye(2, dtype=np.float32))
    a = ad.AnnData(X)
    a.layers["counts"] = sparse.csr_matrix(np.eye(2, dtype=np.float64))
    P.zyme_prepare(a, layers=False)
    assert a.layers["counts"].dtype == np.float64


# ==========================================================================
# __init__ exported surface
# ==========================================================================

def test_zyme_prepare_exported():
    assert hasattr(SC, "zyme_prepare")
    assert SC.zyme_prepare is P.zyme_prepare
    assert "zyme_prepare" in SC.__all__


def test_init_imports_all_fast_funcs():
    # Every patched fast function must be importable off the package module.
    for name in ("fast_normalize_total", "fast_log1p", "fast_scale",
                 "fast_pca", "_patched_hvg", "fast_leiden",
                 "_fast_rank_genes_groups", "fast_regress_out"):
        assert hasattr(SC, name), name


# ==========================================================================
# __init__._smoke_load — task.yaml / env-var path resolution + error branches
# ==========================================================================

def test_smoke_load_no_taskdir_no_env_raises(monkeypatch):
    pytest.importorskip("scanpy")
    monkeypatch.delenv("AUTOZYME_SCANPY_SMOKE_DATA", raising=False)
    with pytest.raises(FileNotFoundError, match="no task_dir"):
        SC._smoke_load(None, "small")


def test_smoke_load_env_points_to_missing_file(monkeypatch, tmp_path):
    pytest.importorskip("scanpy")
    missing = str(tmp_path / "nope.h5ad")
    monkeypatch.setenv("AUTOZYME_SCANPY_SMOKE_DATA", missing)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        SC._smoke_load(None, "small")


def test_smoke_load_taskdir_missing_tier_raises(monkeypatch, tmp_path):
    pytest.importorskip("scanpy")
    yaml = pytest.importorskip("yaml")
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(yaml.safe_dump(
        {"datasets": [{"tier": "large", "path": "./x.h5ad"}]}))
    monkeypatch.delenv("AUTOZYME_SCANPY_SMOKE_DATA", raising=False)
    with pytest.raises(FileNotFoundError, match="no dataset for tier"):
        SC._smoke_load(str(tmp_path), "small")


def test_smoke_load_taskdir_tier_path_missing_file(monkeypatch, tmp_path):
    pytest.importorskip("scanpy")
    yaml = pytest.importorskip("yaml")
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(yaml.safe_dump(
        {"datasets": [{"tier": "small", "path": "./missing.h5ad"}]}))
    with pytest.raises(FileNotFoundError, match="not found at"):
        SC._smoke_load(str(tmp_path), "small")


def test_smoke_load_reads_relative_and_abs_paths(monkeypatch, tmp_path):
    """task.yaml path resolution: ./rel resolves against task_dir; absolute kept.

    We stub sc.read_h5ad so no real .h5ad is needed; the assertion is on
    which path _smoke_load resolved and passed to read_h5ad.
    """
    sc = pytest.importorskip("scanpy")
    yaml = pytest.importorskip("yaml")

    # Create the file so the os.path.isfile gate passes.
    data_file = tmp_path / "rel.h5ad"
    data_file.write_bytes(b"")
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(yaml.safe_dump(
        {"datasets": [{"tier": "small", "path": "./rel.h5ad"}]}))

    seen = {}

    def fake_read(p):
        seen["path"] = p
        return "ADATA"

    monkeypatch.setattr(sc, "read_h5ad", fake_read)
    out = SC._smoke_load(str(tmp_path), "small")
    assert out == {"adata": "ADATA"}
    assert seen["path"] == str(data_file)


def test_smoke_load_absolute_path_in_yaml(monkeypatch, tmp_path):
    sc = pytest.importorskip("scanpy")
    yaml = pytest.importorskip("yaml")
    data_file = tmp_path / "abs.h5ad"
    data_file.write_bytes(b"")
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(yaml.safe_dump(
        {"datasets": [{"tier": "small", "path": str(data_file)}]}))
    seen = {}
    monkeypatch.setattr(sc, "read_h5ad", lambda p: seen.setdefault("path", p) or "A")
    SC._smoke_load(str(tmp_path), "small")
    assert seen["path"] == str(data_file)


def test_smoke_load_env_fallback_reads(monkeypatch, tmp_path):
    sc = pytest.importorskip("scanpy")
    data_file = tmp_path / "env.h5ad"
    data_file.write_bytes(b"")
    monkeypatch.setenv("AUTOZYME_SCANPY_SMOKE_DATA", str(data_file))
    seen = {}
    monkeypatch.setattr(sc, "read_h5ad", lambda p: seen.setdefault("path", p) or "A")
    # task_dir=None -> falls through to env var.
    SC._smoke_load(None, "small")
    assert seen["path"] == str(data_file)


# ==========================================================================
# __init__._smoke_save
# ==========================================================================

def test_smoke_save_writes_h5ad(tmp_path):
    class FakeResult:
        def __init__(self):
            self.written_to = None

        def write_h5ad(self, path):
            self.written_to = path

    r = FakeResult()
    SC._smoke_save(r, str(tmp_path))
    assert r.written_to == os.path.join(str(tmp_path), "smoke.h5ad")
