"""AutoZyme patch for Scanpy v1.11.5 — vendored from scanpy-turbo.

Fast paths registered through ``register_patch``:

  ``sc.pp.normalize_total`` + ``sc.pp.log1p`` (parallel numba CSR kernels)
  ``sc.pp.scale``                            (fused numba kernel)
  ``sc.pp.highly_variable_genes``            (single-pass numba kernels for seurat / seurat_v3 batch)
  ``sc.tl.pca``                              (Gram-matrix BLAS + partial LAPACK)
  ``sc.tl.leiden``                           (simple-graph igraph C, Unix fork)
  ``sc.tl.rank_genes_groups``                (fused CSC kernels for wilcoxon)

Activation:

    >>> import autozyme
    >>> autozyme.activate("scanpy")

Per-call escape:

    >>> sc.pp.normalize_total(adata, zyme=False)   # this one call uses vanilla
    >>> with autozyme.disabled():                  # all patches off in block
    ...     sc.pp.normalize_total(adata)

Bonus utility (not a replacement — a new attr):

    >>> from autozyme.scanpy import zyme_prepare
    >>> zyme_prepare(adata)   # one-shot float32 + int32 CSR coerce
"""
from __future__ import annotations

from autozyme._core import register_patch

from ._normalize import fast_normalize_total, fast_log1p
from ._scale import fast_scale
from ._pca import fast_pca
from ._highly_variable import _patched_hvg
from ._leiden import fast_leiden
from ._rank_genes import _fast_rank_genes_groups
# regress_out patch temporarily disabled — see register_patch() below.
# from ._regress_out import fast_regress_out
from ._prepare import zyme_prepare


# --- Smoke recipe (used by autozyme.verify_patch) -----------------------


def _smoke_load(task_dir, tier):
    """Load the AnnData for the requested tier from task.yaml.

    Mirrors vegan/seurat: parses ``task_dir/task.yaml``, finds the dataset
    row matching ``tier``, resolves a relative path against ``task_dir``,
    and reads the .h5ad. When ``task_dir`` is None or has no task.yaml,
    falls back to the ``AUTOZYME_SCANPY_SMOKE_DATA`` env var (legacy
    standalone recipe used by ``tests/test_scanpy_smoke.py``); raises a
    clear FileNotFoundError when neither is provided.
    """
    import os
    import scanpy as sc

    data_path = None
    if task_dir:
        yaml_path = os.path.join(task_dir, "task.yaml")
        if os.path.isfile(yaml_path):
            import yaml as _yaml
            with open(yaml_path, "r", encoding="utf-8") as fp:
                task = _yaml.safe_load(fp) or {}
            for ds in task.get("datasets") or []:
                if ds.get("tier") == tier:
                    raw = ds.get("path") or ""
                    if raw.startswith("./"):
                        data_path = os.path.join(task_dir, raw[2:])
                    elif os.path.isabs(raw):
                        data_path = raw
                    else:
                        data_path = os.path.join(task_dir, raw)
                    break
            if data_path is None:
                raise FileNotFoundError(
                    f"no dataset for tier {tier!r} in {yaml_path}"
                )
            if not os.path.isfile(data_path):
                raise FileNotFoundError(
                    f"tier {tier!r} dataset not found at {data_path}"
                )
    if data_path is None:
        data_path = os.environ.get("AUTOZYME_SCANPY_SMOKE_DATA")
        if not data_path:
            raise FileNotFoundError(
                "autozyme.scanpy._smoke_load: no task_dir/task.yaml provided "
                "and AUTOZYME_SCANPY_SMOKE_DATA is unset. Either pass a "
                "task_dir whose task.yaml lists this tier's dataset, or set "
                "AUTOZYME_SCANPY_SMOKE_DATA to an .h5ad file (e.g. pbmc68k)."
            )
        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                f"AUTOZYME_SCANPY_SMOKE_DATA points to {data_path!r} which "
                f"does not exist"
            )

    adata = sc.read_h5ad(data_path)
    return {"adata": adata}


def _smoke_call(inputs):
    """Mini end-to-end scanpy pipeline. Exercises the patched fns.

    Follows the standard scanpy idiom of saving the log-normalized matrix to
    ``.raw`` before scaling, so ``rank_genes_groups`` runs on sparse counts
    (not the dense scaled matrix). This keeps the fast path active end-to-end.
    """
    import scanpy as sc
    adata = inputs["adata"]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=500)
    # Snapshot log-normalized sparse counts BEFORE scaling densifies them.
    adata.raw = adata
    sc.pp.scale(adata, max_value=10)
    # use_highly_variable=False so vanilla and fast_pca operate on the same
    # gene set (fast_pca always uses full .X regardless of HVG annotation).
    sc.tl.pca(adata, n_comps=20, use_highly_variable=False)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False, random_state=0)
    # use_raw=True restores the sparse pre-scale matrix so the fused fast path
    # in _fast_rank_genes_groups is reachable. groupby='celltype' uses the
    # real biological labels in pbmc68k.h5ad.
    sc.tl.rank_genes_groups(adata, "celltype", method="wilcoxon", use_raw=True)
    return adata


def _smoke_save(result, out_dir, **kwargs):
    """Persist the AnnData for downstream parity comparison."""
    import os
    result.write_h5ad(os.path.join(out_dir, "smoke.h5ad"))


# --- Register patches ---------------------------------------------------
# Targets mirror scanpy-turbo's patch() coverage: both the public namespace
# attribute (``scanpy.preprocessing.X``) and the canonical module attribute
# (``scanpy.preprocessing._foo.X``) are patched for HVG and leiden, since
# scanpy's own internal code imports from the canonical locations.

register_patch(
    name="scanpy",
    targets=[
        # Public namespace patches (sc.pp.* / sc.tl.*)
        ("scanpy.preprocessing", "normalize_total",       fast_normalize_total),
        ("scanpy.preprocessing", "log1p",                  fast_log1p),
        ("scanpy.preprocessing", "scale",                  fast_scale),
        # regress_out temporarily disabled:
        # ("scanpy.preprocessing", "regress_out",            fast_regress_out),
        ("scanpy.preprocessing", "highly_variable_genes",  _patched_hvg),
        ("scanpy.tools",         "pca",                    fast_pca),
        # sc.pp.pca is the modern canonical PCA call; sc.tl.pca is its
        # deprecated alias. They are distinct name bindings, so patching
        # tools.pca alone left sc.pp.pca (the common path) on the slow original.
        # Same fast_pca, just the other public alias.
        ("scanpy.preprocessing", "pca",                    fast_pca),
        ("scanpy.tools",         "leiden",                 fast_leiden),
        ("scanpy.tools",         "rank_genes_groups",      _fast_rank_genes_groups),

        # Canonical module attrs for the funcs that scanpy itself imports internally
        # regress_out temporarily disabled:
        # ("scanpy.preprocessing._simple",                 "regress_out",          fast_regress_out),
        ("scanpy.preprocessing._highly_variable_genes", "highly_variable_genes", _patched_hvg),
        ("scanpy.preprocessing._pca",                    "pca",                   fast_pca),
        ("scanpy.tools._leiden",                         "leiden",               fast_leiden),

        # _RankGenes class-method patches: scanpy-turbo had three patches
        # for fallback paths (tie_correct=True, reference!=rest, dense). They
        # set self._results (legacy API) which breaks scanpy 1.11.5 (expects
        # self.stats DataFrame). Removed entirely from this vendor — top-level
        # fast path handles the common case; fallback delegates to vanilla.
    ],
    smoke={"load": _smoke_load, "call": _smoke_call, "save": _smoke_save},
    tested_against="scanpy 1.11.5",
    tested_upstream_versions={"scanpy": ["1.11.5"]},
)


# ``zyme_prepare`` is a NEW attribute (not a replacement of an existing one),
# so it cannot go through ``register_patch`` (which captures an "original" via
# getattr). Exposed at module top-level for explicit import:
#
#     from autozyme.scanpy import zyme_prepare
__all__ = ["zyme_prepare"]
