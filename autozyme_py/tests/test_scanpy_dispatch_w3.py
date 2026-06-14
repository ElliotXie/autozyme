"""Wave-3 dispatch / smoke-recipe coverage for the scanpy patch.

Targets the pure-python lines coverage.py can see that wave-1/wave-2 left
uncovered in ``autozyme/scanpy/__init__.py``:

  * ``_smoke_load`` — task.yaml dataset resolution (./relative, absolute, and
    plain-relative path forms), the missing-tier / missing-file errors, and
    the ``AUTOZYME_SCANPY_SMOKE_DATA`` env fallback (set / unset / bad-path).
  * ``_smoke_call`` end-to-end mini pipeline (normalize -> log1p -> hvg ->
    scale -> pca -> neighbors -> leiden -> rank_genes) on a tiny synthetic
    ``celltype`` AnnData, driven through the *active* dispatcher.
  * ``_smoke_save`` round-trip.

These are NOT duplicated by ``test_scanpy_dispatch_e2e.py`` (wave-2), which
only covers register/activate/restore/inspect and ``zyme_prepare``.

Env recipe: KMP_DUPLICATE_LIB_OK=TRUE NUMBA_THREADING_LAYER=workqueue
NUMBA_NUM_THREADS=14 OMP_NUM_THREADS=2.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from scipy import sparse

sc = pytest.importorskip("scanpy")
ad = pytest.importorskip("anndata")
yaml = pytest.importorskip("yaml")
import autozyme  # noqa: E402
import autozyme.scanpy as SC  # noqa: E402


def _celltype_adata(n_obs=60, n_vars=40, seed=0):
    """Tiny raw-count AnnData with a 3-level ``celltype`` obs column.

    Each celltype over-expresses a disjoint marker block so the downstream
    leiden / rank_genes_groups steps in ``_smoke_call`` are non-degenerate.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.8, size=(n_obs, n_vars)).astype(np.float32)
    ct = np.array(["A", "B", "C"] * (n_obs // 3))
    for i, label in enumerate(["A", "B", "C"]):
        counts[ct == label, i * 5:(i + 1) * 5] += 6
    X = sparse.csr_matrix(counts)
    X.indptr = X.indptr.astype(np.int32)
    X.indices = X.indices.astype(np.int32)
    a = ad.AnnData(X)
    a.obs["celltype"] = pd.Categorical(ct)
    a.var_names = [f"g{j}" for j in range(n_vars)]
    return a


def _write_task(tmp_path, *, path_value, tier="small"):
    """Write a tiny task.yaml + data.h5ad and return the task_dir."""
    a = _celltype_adata()
    h5 = tmp_path / "data.h5ad"
    a.write_h5ad(str(h5))
    task = {"datasets": [{"tier": tier, "path": path_value}]}
    with open(tmp_path / "task.yaml", "w", encoding="utf-8") as fp:
        yaml.safe_dump(task, fp)
    return str(tmp_path), str(h5)


# --------------------------------------------------------------------------
# _smoke_load — task.yaml path resolution
# --------------------------------------------------------------------------

def test_smoke_load_relative_dot_slash_path(tmp_path):
    # "./data.h5ad" → joined against task_dir with the leading "./" stripped.
    task_dir, _ = _write_task(tmp_path, path_value="./data.h5ad")
    inp = SC._smoke_load(task_dir, "small")
    assert isinstance(inp, dict)
    assert inp["adata"].shape == (60, 40)
    assert "celltype" in inp["adata"].obs


def test_smoke_load_absolute_path(tmp_path):
    # An absolute path in task.yaml is used verbatim.
    task_dir, h5 = _write_task(tmp_path, path_value="placeholder")
    # Rewrite the yaml with the absolute path.
    task = {"datasets": [{"tier": "small", "path": h5}]}
    with open(os.path.join(task_dir, "task.yaml"), "w", encoding="utf-8") as fp:
        yaml.safe_dump(task, fp)
    inp = SC._smoke_load(task_dir, "small")
    assert inp["adata"].shape == (60, 40)


def test_smoke_load_plain_relative_path(tmp_path):
    # A bare "data.h5ad" (no "./") hits the else branch (join verbatim).
    task_dir, _ = _write_task(tmp_path, path_value="data.h5ad")
    inp = SC._smoke_load(task_dir, "small")
    assert inp["adata"].shape == (60, 40)


def test_smoke_load_unknown_tier_raises(tmp_path):
    task_dir, _ = _write_task(tmp_path, path_value="./data.h5ad", tier="small")
    with pytest.raises(FileNotFoundError, match="no dataset for tier"):
        SC._smoke_load(task_dir, "nonexistent_tier")


def test_smoke_load_missing_data_file_raises(tmp_path):
    # task.yaml points at a file that does not exist on disk.
    task = {"datasets": [{"tier": "small", "path": "./absent.h5ad"}]}
    with open(tmp_path / "task.yaml", "w", encoding="utf-8") as fp:
        yaml.safe_dump(task, fp)
    with pytest.raises(FileNotFoundError, match="dataset not found"):
        SC._smoke_load(str(tmp_path), "small")


def test_smoke_load_env_fallback(tmp_path, monkeypatch):
    # No task_dir → AUTOZYME_SCANPY_SMOKE_DATA env var is used.
    a = _celltype_adata()
    h5 = tmp_path / "env_data.h5ad"
    a.write_h5ad(str(h5))
    monkeypatch.setenv("AUTOZYME_SCANPY_SMOKE_DATA", str(h5))
    inp = SC._smoke_load(None, "small")
    assert inp["adata"].shape == (60, 40)


def test_smoke_load_env_unset_raises(monkeypatch):
    monkeypatch.delenv("AUTOZYME_SCANPY_SMOKE_DATA", raising=False)
    with pytest.raises(FileNotFoundError, match="AUTOZYME_SCANPY_SMOKE_DATA is unset"):
        SC._smoke_load(None, "small")


def test_smoke_load_env_bad_path_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOZYME_SCANPY_SMOKE_DATA", str(tmp_path / "missing.h5ad"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        SC._smoke_load(None, "small")


def test_smoke_load_task_dir_without_yaml_uses_env(tmp_path, monkeypatch):
    # task_dir given but it has no task.yaml → fall through to the env var.
    a = _celltype_adata()
    h5 = tmp_path / "data.h5ad"
    a.write_h5ad(str(h5))
    no_yaml_dir = tmp_path / "empty"
    no_yaml_dir.mkdir()
    monkeypatch.setenv("AUTOZYME_SCANPY_SMOKE_DATA", str(h5))
    inp = SC._smoke_load(str(no_yaml_dir), "small")
    assert inp["adata"].shape == (60, 40)


# --------------------------------------------------------------------------
# _smoke_call + _smoke_save — full mini pipeline through the active patch
# --------------------------------------------------------------------------

def test_smoke_call_runs_full_pipeline():
    autozyme.activate("scanpy")
    inp = {"adata": _celltype_adata(seed=1)}
    result = SC._smoke_call(inp)
    # The pipeline must produce all downstream annotations.
    assert "X_pca" in result.obsm
    assert "leiden" in result.obs
    assert "rank_genes_groups" in result.uns
    assert "highly_variable" in result.var.columns
    assert result.raw is not None  # .raw snapshot was taken


def test_smoke_save_writes_h5ad(tmp_path):
    autozyme.activate("scanpy")
    inp = {"adata": _celltype_adata(seed=2)}
    result = SC._smoke_call(inp)
    SC._smoke_save(result, str(tmp_path))
    out = tmp_path / "smoke.h5ad"
    assert out.exists()
    # Round-trips back to an AnnData with the same shape.
    back = sc.read_h5ad(str(out))
    assert back.n_obs == result.n_obs


def test_smoke_call_via_registered_patch_recipe():
    # Drive through the registered patch's smoke dict (not the module attrs)
    # to confirm register_patch wired the recipe in.
    autozyme.activate("scanpy")
    info = autozyme.inspect("scanpy")
    assert info["name"] == "scanpy"
    inp = {"adata": _celltype_adata(seed=3)}
    result = SC._smoke_call(inp)
    assert "leiden" in result.obs
