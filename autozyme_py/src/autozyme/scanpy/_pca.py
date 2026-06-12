"""Fast PCA: Gram matrix via BLAS + partial eigendecomposition via LAPACK.

Vendored from scanpy-turbo/_turbo/pca.py. ``zyme=False`` looks up the
upstream original via the autozyme dispatcher's ``__autozyme_original__``
attribute (set on ``sc.tl.pca`` after register_patch).

On macOS, uses Apple Accelerate's cblas_sgemm directly via ctypes for the
Gram matrix and projection (AMX coprocessor, fused alpha scaling).
On other platforms, uses numpy BLAS (OpenBLAS/MKL).
"""

from __future__ import annotations

import ctypes
import platform

import numpy as np
import scipy.linalg
import scipy.sparse as sp


def _orig_pca():
    """Fetch upstream sc.tl.pca via the autozyme dispatcher attribute."""
    import scanpy as sc
    return getattr(sc.tl.pca, "__autozyme_original__", sc.tl.pca)


# --- Apple Accelerate detection and helpers ---
_acc = None
_USE_ACCELERATE = False

if platform.system() == "Darwin":
    try:
        _acc = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
        _USE_ACCELERATE = True
    except OSError:
        pass


def _accelerate_sgemm_gram(X, alpha=1.0):
    """Compute alpha * X^T @ X using Apple Accelerate's cblas_sgemm."""
    n_cells, n_genes = X.shape
    out = np.empty((n_genes, n_genes), dtype=np.float32)
    _acc.cblas_sgemm(
        ctypes.c_int(101),   # CblasRowMajor
        ctypes.c_int(112),   # CblasTrans
        ctypes.c_int(111),   # CblasNoTrans
        ctypes.c_int(n_genes),
        ctypes.c_int(n_genes),
        ctypes.c_int(n_cells),
        ctypes.c_float(alpha),
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n_genes),
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n_genes),
        ctypes.c_float(0.0),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n_genes),
    )
    return out


def _accelerate_sgemm(A, B):
    """Compute A @ B using Apple Accelerate where A is (m,k) and B is (k,n), both float32."""
    m, k = A.shape
    n = B.shape[1]
    out = np.empty((m, n), dtype=np.float32)
    _acc.cblas_sgemm(
        ctypes.c_int(101),   # CblasRowMajor
        ctypes.c_int(111),   # CblasNoTrans
        ctypes.c_int(111),   # CblasNoTrans
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.c_float(1.0),
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(k),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n),
        ctypes.c_float(0.0),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(n),
    )
    return out


# Algorithmic sweet-spot cap. The Gram-matrix algorithm requires `eigh` on
# an n_genes × n_genes matrix; LAPACK 'evr' with subset_by_index still does
# full O(n_genes^3) Householder tridiagonalization (subset only saves the
# inverse-iteration step at the end). Beyond this many genes, vanilla ARPACK
# on the original X dominates by skipping Gram entirely.
#
# Empirical crossover at 18,161 cells (Windows + OpenBLAS sgemm/eigh, May 2026):
#   n_genes=5000   fast_pca 6.4x faster than ARPACK
#   n_genes=7500   fast_pca 2.9x faster
#   n_genes=10000  fast_pca 1.7x  (marginal)
#   n_genes=15000  fast_pca 0.68x (LOSES)
# 8000 keeps fast_pca on the >=2x side with margin. For larger n_cells the
# crossover shifts up (ARPACK cost grows linearly in n_cells but eigh does
# not), so 8000 is conservative across the range we care about.
_GENE_LIMIT = 8000


def fast_pca(adata, n_comps=50, *, copy=False, zyme=True, **kwargs):
    if not zyme:
        return _orig_pca()(adata, n_comps=n_comps, copy=copy, **kwargs)

    # Scope guard (additive safety net — does NOT touch the in-scope fast path).
    # The Gram+eigh kernel is validated only for the default-configuration call:
    # PCA on the (already-scaled) adata.X, mean-centered, full result written to
    # the standard X_pca / PCs / uns['pca'] keys as float32. Any caller arg that
    # changes WHAT is decomposed, WHETHER it is centered, the RETURN type, or
    # WHERE results land is not reproduced here, so defer to upstream rather than
    # silently ignore it. For the benchmarked/default call every check below is
    # False and execution proceeds exactly as before.
    if (
        kwargs.get("layer") is not None
        or not kwargs.get("zero_center", True)
        or ("mask_var" in kwargs and kwargs["mask_var"] is not None)
        or kwargs.get("chunked", False)
        or kwargs.get("return_info", False)
        or kwargs.get("key_added") is not None
        or kwargs.get("dtype", "float32") not in ("float32", np.float32)
    ):
        return _orig_pca()(adata, n_comps=n_comps, copy=copy, **kwargs)

    if copy:
        adata = adata.copy()

    # Honor `use_highly_variable` like upstream scanpy. Default mirrors scanpy:
    # use HVG when the annotation is present. Without this, a real-user call
    # like `sc.tl.pca(adata)` on a 30k-gene AnnData with HVG already computed
    # would hit the O(n_genes^3) eigh path even though only the 2000 HVG were
    # intended downstream — silently turning a 0.5-sec scanpy call into a
    # 10-minute stall.
    #
    # `.get` (not `.pop`) so the kwarg survives into a downstream fall-through:
    # users who explicitly pass `use_highly_variable=False` deserve the full-
    # gene result they asked for, not a silent HVG subset done by vanilla.
    use_hvg_raw = kwargs.get("use_highly_variable", None)
    use_hvg = (use_hvg_raw if use_hvg_raw is not None
               else ("highly_variable" in adata.var.columns))
    if use_hvg and "highly_variable" in adata.var.columns:
        mask = adata.var["highly_variable"].values
        if mask.sum() < adata.n_vars:
            # We're handling HVG subset ourselves now; strip the kwarg so the
            # recursive call doesn't recurse on the already-subset matrix.
            sub_kwargs = {k: v for k, v in kwargs.items() if k != "use_highly_variable"}
            adata_sub = adata[:, mask].copy()
            fast_pca(adata_sub, n_comps=n_comps, zyme=True,
                     use_highly_variable=False, **sub_kwargs)
            # Lift HVG-space PCs back into full var space — zeros for non-HVG
            # genes so downstream code that indexes adata.varm["PCs"] by full
            # gene set still works.
            n_comps_eff = adata_sub.varm["PCs"].shape[1]
            full_pcs = np.zeros((adata.n_vars, n_comps_eff),
                                 dtype=adata_sub.varm["PCs"].dtype)
            full_pcs[mask] = adata_sub.varm["PCs"]
            adata.obsm["X_pca"] = adata_sub.obsm["X_pca"]
            adata.varm["PCs"] = full_pcs
            adata.uns["pca"] = adata_sub.uns["pca"]
            adata.uns["pca"]["params"]["use_highly_variable"] = True
            adata.uns["pca"]["params"]["mask_var"] = "highly_variable"
            return adata if copy else None

    X = adata.X
    n_cells, n_genes = X.shape

    # Out-of-sweet-spot guard: fall through to vanilla ARPACK. See _GENE_LIMIT
    # comment above for the empirical justification. We've already copied
    # adata above when copy=True, so call vanilla in-place on the local
    # handle and return it.
    if n_genes > _GENE_LIMIT:
        _orig_pca()(adata, n_comps=n_comps, copy=False, **kwargs)
        return adata if copy else None

    if sp.issparse(X):
        X_d = X.toarray()
    else:
        X_d = np.asarray(X)

    # Convert to float32 and center
    X_f32 = np.ascontiguousarray(X_d, dtype=np.float32)
    mean = X_f32.mean(axis=0)
    X_f32 -= mean

    if _USE_ACCELERATE:
        # Covariance via Gram matrix using Apple Accelerate (AMX — 5x faster)
        cov = _accelerate_sgemm_gram(X_f32, alpha=1.0 / (n_cells - 1))
    else:
        # Gram matrix via numpy BLAS (uses OpenBLAS/MKL sgemm internally)
        cov = (X_f32.T @ X_f32).astype(np.float32) / (n_cells - 1)

    total_var = float(np.trace(cov))

    # Eigendecomposition via LAPACK.
    # macOS branch: scipy.linalg.eigh(... subset_by_index, driver="evr") segfaults
    # on Apple Accelerate's LAPACK for some matrix conditionings. Fall back to
    # numpy.linalg.eigh, which routes through Accelerate's safer ssyevd path,
    # then take the top n_comps.
    if platform.system() == "Darwin":
        all_eigvals, all_eigvecs = np.linalg.eigh(cov)
        eigvals = all_eigvals[-n_comps:][::-1].astype(np.float64)
        eigvecs_f32 = np.ascontiguousarray(
            all_eigvecs[:, -n_comps:][:, ::-1], dtype=np.float32,
        )
        eigvecs = eigvecs_f32.astype(np.float64)
    else:
        eigvals, eigvecs = scipy.linalg.eigh(
            cov, subset_by_index=[n_genes - n_comps, n_genes - 1],
            overwrite_a=True, driver="evr",
        )
        eigvals = eigvals[::-1].astype(np.float64)
        eigvecs_f32 = np.ascontiguousarray(eigvecs[:, ::-1], dtype=np.float32)
        eigvecs = eigvecs_f32.astype(np.float64)

    if _USE_ACCELERATE:
        X_pca = _accelerate_sgemm(X_f32, eigvecs_f32)
    else:
        X_pca = X_f32 @ eigvecs_f32

    variance_ratio = eigvals / total_var

    adata.obsm["X_pca"] = X_pca
    adata.varm["PCs"] = eigvecs
    adata.uns["pca"] = dict(
        params=dict(zero_center=True, use_highly_variable=False, mask_var=None),
        variance=eigvals.astype(np.float32),
        variance_ratio=variance_ratio.astype(np.float32),
    )

    return adata if copy else None
