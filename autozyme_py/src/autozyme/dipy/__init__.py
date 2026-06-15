"""Patch for dipy's DTI weighted-least-squares tensor fit.

Lifted from autozyme task ``test_dipy_dti``. Two coordinated patches in
``dipy.reconst.dti``:

  - ``wls_fit_tensor`` — batched 7x7 normal-equations solve for the regular
    full-rank case, with an upstream-equivalent SVD-pinv fallback for any voxel
    whose normal matrix is not safely positive definite. The hot inner kernel
    (``_fast_wls_fit_tensor_inner``)
    forms the WLS normal matrix via ``np.einsum`` ("gi,gj->gij") and solves
    each voxel's 7x7 system via a guarded hand-rolled batched Cholesky
    (``_chol7_solve``). The fitted lower-triangular tensor is then passed
    through dipy's ``eig_from_lo_tri`` so the output preserves the full
    upstream ``(..., 12)`` contract: 3 eigenvalues plus 9 eigenvector
    components. Wrapped with ``dipy.reconst.dti.iter_fit_tensor`` so it
    streams over the same chunked voxel batches the upstream driver expects.
  - ``TensorModel.fit`` — chunked masked-voxel path: only voxels inside the
    brain mask are unpacked from the (X,Y,Z,G) volume and fed to the
    WLS kernel, in steps of ``step=1250`` (matches upstream's chunk
    granularity). Avoids materializing a contiguous ``data[mask]`` copy of
    the full masked block when the input volume is large. The unmasked
    fallback (rare in DTI workflows but supported by upstream) defers to
    upstream's reshape path with the patched ``wls_fit_tensor`` underneath.

Threads: WLS is a per-voxel independent solve; the heavy numerics are
``@`` (matmul) and ``np.einsum``, both of which route to OpenBLAS / MKL and
already use multiple threads via ``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS``.
The patch doesn't set thread env vars itself — autozyme's standard
``ZYME_THREADS`` sweep (1, 4, 8) controls them via the calling environment.

The ``dti.common_fit_methods["WLS"]`` dict is a method registry the upstream
``TensorModel.__init__`` consults: ``self.fit_method = common_fit_methods[fit_method]``.
Activating the patch rebinds ``dti.wls_fit_tensor`` (and ``WLLS`` alias) so
that any ``TensorModel`` constructed AFTER activation picks up the fast fn;
we mirror that into the dict inside ``_smoke_load`` so baseline (untouched)
and patched (rebound) both see the matching dispatch entry.
"""
from __future__ import annotations

import os

import numpy as np

from dipy.reconst import dti
from dipy.reconst.dti import eig_from_lo_tri

from autozyme._core import register_patch
from autozyme._utils import resolve_dataset_path


# ---- file-scope captures of upstream originals (for reference / inspection) ----
_orig_wls_fit_tensor = dti.wls_fit_tensor
_orig_tensor_model_fit = dti.TensorModel.fit


# ============================================================
# Optimized WLS tensor-fit kernel
# ============================================================

def _chol7_solve(lhs, rhs):
    """Batched Cholesky solve for safely SPD 7x7 systems.

    Forms L (lower triangular) per-voxel, then forward/back-substitutes.
    All loops are over the FIXED dimension (7), so the per-voxel cost is
    O(1) and the batch dimension is vectorized natively by numpy.

    Returns the solution and a boolean mask for voxels that need the SVD-pinv
    fallback because the Cholesky factorization was not numerically safe.
    """
    batch_shape = rhs.shape[:-1]
    a = lhs.reshape((-1, 7, 7))
    b = rhs.reshape((-1, 7))
    l_mat = np.zeros_like(a)
    ok = np.ones(a.shape[0], dtype=bool)
    diag_scale = np.maximum(
        np.max(np.abs(np.diagonal(a, axis1=1, axis2=2)), axis=1),
        1.0,
    )
    min_diag = np.finfo(a.dtype).eps * diag_scale

    for i in range(7):
        diag = a[:, i, i].copy()
        for k in range(i):
            diag -= l_mat[:, i, k] * l_mat[:, i, k]
        ok &= np.isfinite(diag) & (diag > min_diag)
        l_mat[:, i, i] = np.sqrt(np.where(ok, diag, 1.0))
        for j in range(i + 1, 7):
            val = a[:, j, i].copy()
            for k in range(i):
                val -= l_mat[:, j, k] * l_mat[:, i, k]
            l_mat[:, j, i] = np.where(ok, val / l_mat[:, i, i], 0.0)

    y = np.empty_like(b)
    for i in range(7):
        val = b[:, i].copy()
        for k in range(i):
            val -= l_mat[:, i, k] * y[:, k]
        y[:, i] = np.where(ok, val / l_mat[:, i, i], 0.0)

    x = np.empty_like(b)
    for i in range(6, -1, -1):
        val = y[:, i].copy()
        for k in range(i + 1, 7):
            val -= l_mat[:, k, i] * x[:, k]
        x[:, i] = np.where(ok, val / l_mat[:, i, i], 0.0)

    bad = (~ok) | (~np.all(np.isfinite(x), axis=1))
    return x.reshape(batch_shape + (7,)), bad.reshape(batch_shape)


def _pinv_wls_fit(design_matrix, log_s, w):
    """Upstream-equivalent SVD-pinv WLS solve for fallback voxels."""
    return np.einsum(
        "...ij,...j->...i",
        np.linalg.pinv(design_matrix * w[..., None]),
        w * log_s,
    )


def _fast_wls_fit_tensor_inner(
    design_matrix,
    data,
    *,
    weights=None,
    return_S0_hat=False,
    return_lower_triangular=False,
    return_leverages=False,
    min_signal=None,
):
    """WLS tensor fit using batched 7x7 normal equations instead of SVD pinv."""
    tol = 1e-6
    design_matrix = np.asarray(design_matrix, dtype=np.float64)
    data = np.asarray(data)
    if min_signal is None:
        log_s = np.log(data).astype(np.float64, copy=False)
    else:
        log_s = np.array(data, dtype=np.float64, copy=True)
        np.maximum(log_s, min_signal, out=log_s)
        np.log(log_s, out=log_s)

    if weights is None:
        fit_result = log_s @ np.linalg.pinv(design_matrix).T
        w = np.exp(fit_result @ design_matrix.T)
    else:
        w = np.sqrt(np.asarray(weights, dtype=np.float64))

    weight_sq = w * w
    design_outer = np.einsum("gi,gj->gij", design_matrix, design_matrix)
    lhs = (weight_sq @ design_outer.reshape(design_matrix.shape[0], -1)).reshape(
        weight_sq.shape[:-1] + (design_matrix.shape[1], design_matrix.shape[1])
    )

    if return_leverages is False:
        rhs = (weight_sq * log_s) @ design_matrix
        fit_result, bad_chol = _chol7_solve(lhs, rhs)
        if np.any(bad_chol):
            fit_flat = fit_result.reshape((-1, 7))
            log_s_flat = log_s.reshape((-1, design_matrix.shape[0]))
            w_flat = w.reshape((-1, design_matrix.shape[0]))
            bad_flat = bad_chol.reshape(-1)
            fit_flat[bad_flat] = _pinv_wls_fit(
                design_matrix, log_s_flat[bad_flat], w_flat[bad_flat]
            )
        leverages = None
    else:
        tmp = np.einsum(
            "...ij,...j->...ij",
            np.linalg.pinv(design_matrix * w[..., None]),
            w,
        )
        fit_result = np.einsum("...ig,...g->...i", tmp, log_s)
        leverages = np.einsum("gi,...ig->...g", design_matrix, tmp)

    if leverages is not None:
        leverages = {"leverages": leverages}

    if return_lower_triangular:
        return fit_result, leverages

    dti_params = eig_from_lo_tri(
        fit_result[..., :6], min_diffusivity=tol / -design_matrix.min()
    )
    if return_S0_hat:
        return (dti_params, np.exp(-fit_result[..., -1])), leverages
    return dti_params, leverages


# `iter_fit_tensor` is dipy's chunking decorator; matches upstream signature
# so the fast fn is a drop-in replacement for dti.wls_fit_tensor.
fast_wls_fit_tensor = dti.iter_fit_tensor()(_fast_wls_fit_tensor_inner)


# ============================================================
# Optimized TensorModel.fit (chunked masked-voxel path)
# ============================================================

def fast_tensor_model_fit(self, data, *, mask=None):
    """Chunked masked-voxel WLS fit.

    Only voxels inside ``mask`` are unpacked and fed to the WLS kernel, in
    chunks of ``step`` (default 1250 from upstream's fit_kwargs convention).
    Avoids the contiguous ``data[mask]`` copy that upstream does for the
    masked path, which can be large at high tile factors.
    """
    S0_params = None

    img_shape = data.shape[:-1]
    if mask is not None:
        if mask.shape != img_shape:
            raise ValueError("Mask is not the same shape as data.")
        mask = np.asarray(mask, dtype=bool)
    if self.min_signal is None:
        min_signal = dti.MIN_POSITIVE_SIGNAL
    else:
        min_signal = self.min_signal

    fit_kwargs = dict(self.kwargs)
    # TensorModel snapshots common_fit_methods["WLS"] at construction time.
    # Public activate() rebinds dti.wls_fit_tensor, but not that registry, so
    # freshly constructed WLS models can still hold the original function.
    # Treat that case as WLS too and call the live patched function below.
    live_wls_is_patched = dti.wls_fit_tensor is not _orig_wls_fit_tensor
    is_fast_wls = live_wls_is_patched and (
        self.fit_method is dti.wls_fit_tensor
        or self.fit_method is _orig_wls_fit_tensor
    )
    fit_method = dti.wls_fit_tensor if is_fast_wls else self.fit_method
    # Explicit per-voxel `weights` must be sliced per chunk; this masked fast loop
    # forwards the whole `weights` array to every chunk (chunk-2+ voxels then get
    # mis-aligned weights or a shape mismatch). Defer weighted WLS fits to the
    # upstream-style reshape fallback below, which fits the full masked block where
    # the patched wls_fit_tensor's own iter_fit_tensor slices weights correctly.
    if is_fast_wls and mask is not None and "weights" not in fit_kwargs:
        fit_kwargs["min_signal"] = min_signal
        chunk_step = int(fit_kwargs.get("step", 1250)) or int(mask.sum())
        mask_indices = np.flatnonzero(mask.ravel())
        flat_data = data.reshape((-1, data.shape[-1]))
        dti_params = np.zeros(data.shape[:-1] + (12,), dtype=np.float64)
        dti_params_flat = dti_params.reshape((-1, 12))
        if self.return_S0_hat:
            S0_params = np.zeros(data.shape[:-1], dtype=np.float64)
            S0_params_flat = S0_params.ravel()

        for start in range(0, mask_indices.size, chunk_step):
            idx = mask_indices[start : start + chunk_step]
            fit_out, extra_i = fit_method(
                self.design_matrix,
                flat_data[idx],
                *self.args,
                return_S0_hat=self.return_S0_hat,
                **fit_kwargs,
            )
            if self.return_S0_hat:
                params_i, model_S0_i = fit_out
                S0_params_flat[idx] = model_S0_i.reshape(-1)
            else:
                params_i = fit_out
            dti_params_flat[idx, :] = params_i

            if extra_i is not None:
                for key in extra_i:
                    if key not in self.extra:
                        self.extra[key] = np.zeros(data.shape)
                    self.extra[key].reshape((-1, data.shape[-1]))[idx, :] = extra_i[key]

        return dti.TensorFit(self, dti_params, model_S0=S0_params)

    # Unmasked or non-WLS fallback: defer to the upstream-style reshape path
    # with whatever fit_method is currently bound (patched WLS or upstream NLLS).
    data_in_mask = np.reshape(data[mask], (-1, data.shape[-1])) if mask is not None else \
        np.reshape(data, (-1, data.shape[-1]))
    if is_fast_wls:
        fit_kwargs["min_signal"] = min_signal
    else:
        data_in_mask = np.maximum(data_in_mask, min_signal)

    params_in_mask, extra = fit_method(
        self.design_matrix,
        data_in_mask,
        *self.args,
        return_S0_hat=self.return_S0_hat,
        **fit_kwargs,
    )

    if self.return_S0_hat:
        params_in_mask, model_S0 = params_in_mask

    if mask is None:
        out_shape = data.shape[:-1] + (-1,)
        dti_params = params_in_mask.reshape(out_shape)
        if self.return_S0_hat:
            S0_params = model_S0.reshape(out_shape[:-1])
        if extra is not None:
            for key in extra:
                self.extra[key] = extra[key].reshape(data.shape)
    else:
        dti_params = np.zeros(data.shape[:-1] + (12,), dtype=params_in_mask.dtype)
        dti_params[mask, :] = params_in_mask
        if self.return_S0_hat:
            S0_params = np.zeros(data.shape[:-1], dtype=model_S0.dtype)
            try:
                S0_params[mask] = model_S0.squeeze(axis=-1)
            except ValueError:
                S0_params[mask] = model_S0
        if extra is not None:
            for key in extra:
                self.extra[key] = np.zeros(data.shape)
                self.extra[key][mask, :] = extra[key]

    return dti.TensorFit(self, dti_params, model_S0=S0_params)


# ============================================================
# Smoke recipe
# ============================================================

# Mirrors reference.py / pipeline/run.py. Tile factors keyed by tier.
_TILE_FACTORS = {
    # `tiny` was renamed to `small` in the 2026-05 task.yaml refactor; keep
    # both keys so .get() falls back consistently. Without the alias the
    # `small` lookup silently degraded to (1,1,1) = no tiling, producing
    # 1x measurements that look ~6x faster than the 8x baseline.
    "tiny":       (2, 2, 2),
    "small":      (2, 2, 2),
    "medium":     (5, 3, 2),
    "large":      (5, 4, 3),
    "ood_large":  (5, 4, 3),
    "ood_xlarge": (9, 5, 4),
}


def _smoke_load(task_dir, tier):
    """User-side prep: read DWI volume, tile per-tier, compute brain mask,
    construct TensorModel(WLS). NONE of this is patch-accelerated — the patch
    targets ``wls_fit_tensor`` / ``TensorModel.fit``, so only the fit call
    goes in the timed window. Mirrors pipeline/run.py's pre-fit block.

    We also (re)bind ``dti.common_fit_methods["WLS"]`` to the currently-live
    ``dti.wls_fit_tensor`` — at this point the patch (if any) has already been
    activated by the verify worker, so the dict picks up the dispatcher under
    activation and the original under baseline. ``TensorModel.__init__``
    snapshots this entry into ``self.fit_method``, so the dict must reflect
    the live binding BEFORE construction.
    """
    import yaml
    import nibabel as nib
    from dipy.core.gradients import gradient_table
    from dipy.io import read_bvals_bvecs
    from dipy.segment.mask import median_otsu

    with open(os.path.join(task_dir, "task.yaml"), encoding="utf-8") as f:
        task = yaml.safe_load(f)
    ds = next(d for d in task["datasets"] if d["tier"] == tier)
    data_path = resolve_dataset_path(task_dir, ds["path"])

    base = data_path[:-len(".nii.gz")] if data_path.endswith(".nii.gz") else os.path.splitext(data_path)[0]
    bval_path = base + ".bval"
    bvec_path = base + ".bvec"

    img = nib.load(data_path)
    data = np.asarray(img.dataobj)
    factors = _TILE_FACTORS.get(tier, (1, 1, 1))
    if factors != (1, 1, 1):
        data = np.tile(data, (factors[0], factors[1], factors[2], 1))
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = gradient_table(bvals, bvecs=bvecs)

    _, mask = median_otsu(data, vol_idx=np.where(bvals < 50)[0],
                          median_radius=3, numpass=1)

    # Pick up the live `wls_fit_tensor` (dispatcher under activation, original
    # under baseline). Done AFTER `from dipy.reconst import dti` at module top
    # and AFTER the verify worker's optional activate() — this is the latest
    # binding for the subprocess.
    dti.common_fit_methods["WLS"] = dti.wls_fit_tensor
    dti.common_fit_methods["WLLS"] = dti.wls_fit_tensor

    tenmodel = dti.TensorModel(gtab, fit_method="WLS", return_S0_hat=True)
    return {"tenmodel": tenmodel, "data": data, "mask": mask}


def _smoke_call(inputs):
    """ONLY the upstream API the patch targets — ``tenmodel.fit(data, mask)``.
    Mirrors pipeline/run.py's ``time.perf_counter`` window exactly.
    """
    tenfit = inputs["tenmodel"].fit(inputs["data"], mask=inputs["mask"])
    return {"tenfit": tenfit, "mask": inputs["mask"]}


def _smoke_save(result, dir, **kwargs):
    """Write the npz the task's evaluate.py reads: model_params, model_S0, mask.
    """
    tenfit = result["tenfit"]
    mask = result["mask"]
    np.savez_compressed(
        os.path.join(dir, "params.npz"),
        model_params=tenfit.model_params,
        model_S0=tenfit.model_S0,
        mask=mask,
    )


register_patch(
    name="dipy",
    targets=[
        ("dipy.reconst.dti", "wls_fit_tensor", fast_wls_fit_tensor),
        ("dipy.reconst.dti.TensorModel", "fit", fast_tensor_model_fit),
    ],
    smoke={"load": _smoke_load, "call": _smoke_call, "save": _smoke_save},
    tested_against="dipy 1.12.1",
    tested_upstream_versions={"dipy": ["1.12.1"]},
)
