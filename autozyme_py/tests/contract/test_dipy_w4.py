"""Wave-4 wrapper/dispatch-line tests for autozyme.dipy.

Wave-1 (`_unit`) tested the WLS kernels; wave-2 (`_e2e`) covered return_S0_hat,
the mask-shape guard, and the lower-triangular / leverages branches. The
COVERAGE-VISIBLE lines still missing are:

  - lines 157-161: the ``bad_chol`` SVD-pinv fallback inside
    ``_fast_wls_fit_tensor_inner`` (a voxel whose 7x7 normal matrix is not
    safely SPD, so the batched Cholesky declines and the upstream-equivalent
    pinv path runs).
  - line 215: the explicit ``self.min_signal`` branch of
    ``fast_tensor_model_fit``.
  - line 270 + 289-303: the NON-fast-WLS fallback paths (a non-WLS fit method,
    masked + unmasked) where the patch defers to upstream's reshape path.
  - lines 256-259: the ``extra`` (leverages) accumulation in the chunked masked
    fast-WLS path.
  - the smoke recipe ``_smoke_load`` / ``_smoke_call`` / ``_smoke_save``
    (340-390) via a tiny synthetic NIfTI + bval/bvec.

(The batched Cholesky / einsum numba-free kernel interiors are pure numpy and
already covered by wave-1; here we drive the python dispatch around them.)
"""
from __future__ import annotations

import os

import numpy as np
import pytest

dipy = pytest.importorskip("dipy")
pytest.importorskip("dipy.reconst.dti")
yaml = pytest.importorskip("yaml")

import autozyme
from autozyme import dipy as azdipy


@pytest.fixture
def gtab_data():
    from dipy.core.gradients import gradient_table

    n_grads = 10
    bvals = np.concatenate([[0], np.full(n_grads - 1, 1000)])
    rng = np.random.default_rng(0)
    bvecs = np.zeros((n_grads, 3))
    bvecs[1:] = rng.normal(size=(n_grads - 1, 3))
    bvecs[1:] /= np.linalg.norm(bvecs[1:], axis=1)[:, None]
    gtab = gradient_table(bvals, bvecs=bvecs)
    data = (rng.random((4, 4, 4, n_grads), dtype=np.float32) * 100.0)
    data[..., 0] *= 5.0
    return gtab, data


# --------------------------------------------------------------------------
# lines 157-161: bad_chol SVD-pinv fallback
# --------------------------------------------------------------------------
def test_inner_wls_bad_chol_falls_back_to_pinv(gtab_data):
    """A voxel whose weighted 7x7 normal matrix is rank-deficient (explicit
    weights that zero out enough gradients to leave < 7 effective measurements)
    trips the not-SPD guard in the batched Cholesky, forcing the SVD-pinv
    fallback branch (157-161). The fallback voxel must match an explicit
    _pinv_wls_fit on the same weights."""
    from dipy.reconst.dti import TensorModel

    autozyme.activate("dipy")
    gtab, data = gtab_data
    model = TensorModel(gtab)
    design = model.design_matrix
    n_grad = design.shape[0]

    flat = np.maximum(
        data.reshape(-1, data.shape[-1])[:3], 1.0
    ).astype(np.float64)
    # Explicit weights: voxel 1 keeps only 5 (< 7) nonzero gradient weights ->
    # the weighted normal matrix is rank-deficient -> bad_chol True for it.
    weights = np.ones((3, n_grad), dtype=np.float64)
    weights[1, 5:] = 0.0

    fit, leverages = azdipy._fast_wls_fit_tensor_inner(
        design, flat, weights=weights, return_lower_triangular=True,
    )
    assert fit.shape == (3, 7)
    assert np.all(np.isfinite(fit))
    # The degenerate voxel must equal the explicit SVD-pinv WLS solve.
    log_s = np.log(flat)
    w = np.sqrt(weights)
    ref = azdipy._pinv_wls_fit(design, log_s[1], w[1])
    np.testing.assert_allclose(fit[1], ref, rtol=1e-6, atol=1e-8)


def test_chunked_masked_leverages_extra_accumulation(gtab_data):
    """A WLS TensorModel built with return_leverages=True passes that kwarg into
    the chunked masked fit path, so each chunk returns a non-None ``extra``
    (leverages), exercising the per-key ``self.extra`` accumulation at lines
    256-259."""
    from dipy.reconst.dti import TensorModel

    autozyme.activate("dipy")
    gtab, data = gtab_data
    mask = np.zeros(data.shape[:3], dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    model = TensorModel(gtab, fit_method="WLS", return_leverages=True)
    model.fit(data, mask=mask)
    assert "leverages" in model.extra
    assert model.extra["leverages"].shape == data.shape
    # Leverages are populated only inside the mask; outside stays zero.
    assert np.any(model.extra["leverages"][mask])


# --------------------------------------------------------------------------
# line 215: explicit min_signal on the model
# --------------------------------------------------------------------------
def test_fit_explicit_min_signal_branch(gtab_data):
    """A TensorModel constructed with min_signal != None takes the
    ``min_signal = self.min_signal`` branch (line 215) and still matches a
    vanilla fit using the same min_signal."""
    from dipy.reconst.dti import TensorModel

    autozyme.activate("dipy")
    gtab, data = gtab_data
    mask = np.zeros(data.shape[:3], dtype=bool)
    mask[1:3, 1:3, 1:3] = True

    fast = TensorModel(gtab, min_signal=2.0).fit(data, mask=mask)
    with autozyme.disabled():
        ref = TensorModel(gtab, min_signal=2.0).fit(data, mask=mask)
    np.testing.assert_allclose(fast.evals, ref.evals, rtol=1e-4, atol=1e-6)


# --------------------------------------------------------------------------
# line 270 + 289-303: non-fast-WLS fallback (masked + unmasked) via NLLS method
# --------------------------------------------------------------------------
def test_fit_non_wls_method_unmasked_fallback(gtab_data):
    """An NLLS TensorModel is not the fast-WLS function, so fast_tensor_model_fit
    takes the upstream-style reshape fallback (line 270, unmasked). Must match
    vanilla NLLS."""
    from dipy.reconst.dti import TensorModel

    autozyme.activate("dipy")
    gtab, data = gtab_data
    fast = TensorModel(gtab, fit_method="NLLS").fit(data)
    with autozyme.disabled():
        ref = TensorModel(gtab, fit_method="NLLS").fit(data)
    np.testing.assert_allclose(fast.evals, ref.evals, rtol=1e-3, atol=1e-5)


def test_fit_non_wls_method_masked_fallback_with_s0(gtab_data):
    """An NLLS TensorModel with a mask AND return_S0_hat=True takes the masked
    non-fast fallback including the S0 scatter (lines 292-299) and matches
    vanilla NLLS on the masked voxels."""
    from dipy.reconst.dti import TensorModel

    autozyme.activate("dipy")
    gtab, data = gtab_data
    mask = np.zeros(data.shape[:3], dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    fast = TensorModel(gtab, fit_method="NLLS", return_S0_hat=True).fit(
        data, mask=mask)
    with autozyme.disabled():
        ref = TensorModel(gtab, fit_method="NLLS", return_S0_hat=True).fit(
            data, mask=mask)
    assert fast.model_S0 is not None
    np.testing.assert_allclose(fast.evals[mask], ref.evals[mask],
                               rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(fast.model_S0)[mask], np.asarray(ref.model_S0)[mask],
        rtol=1e-3, atol=1e-4,
    )


# --------------------------------------------------------------------------
# smoke recipe (340-390) via a tiny synthetic NIfTI
# --------------------------------------------------------------------------
def _make_task_dir(tmp_path):
    nib = pytest.importorskip("nibabel")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    rng = np.random.default_rng(0)
    n_grads = 10
    # Small volume; the smoke loader tiles 'small' by (2,2,2).
    vol = (rng.random((6, 6, 4, n_grads), dtype=np.float32) * 100.0 + 10.0)
    vol[..., 0] *= 6.0  # strong b0 so median_otsu finds a brain mask
    img = nib.Nifti1Image(vol, affine=np.eye(4))
    nib.save(img, str(data_dir / "dwi.nii.gz"))

    bvals = np.concatenate([[0], np.full(n_grads - 1, 1000)])
    bvecs = np.zeros((n_grads, 3))
    bvecs[1:] = rng.normal(size=(n_grads - 1, 3))
    bvecs[1:] /= np.linalg.norm(bvecs[1:], axis=1)[:, None]
    np.savetxt(data_dir / "dwi.bval", bvals[None, :], fmt="%d")
    np.savetxt(data_dir / "dwi.bvec", bvecs.T, fmt="%.6f")

    task = {"datasets": [{"tier": "small", "name": "synth_dwi",
                          "path": "./data/dwi.nii.gz"}]}
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    return str(tmp_path)


def test_smoke_load_call_save_roundtrip(tmp_path):
    """_smoke_load builds the tiled volume + mask + TensorModel; _smoke_call
    fits; _smoke_save writes params.npz (covers 340-390)."""
    autozyme.activate("dipy")
    task_dir = _make_task_dir(tmp_path)
    inputs = azdipy._smoke_load(task_dir, "small")
    assert "tenmodel" in inputs and "data" in inputs and "mask" in inputs
    # 'small' tiles by (2,2,2): (6,6,4) -> (12,12,8).
    assert inputs["data"].shape[:3] == (12, 12, 8)

    result = azdipy._smoke_call(inputs)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    azdipy._smoke_save(result, str(out_dir))
    saved = np.load(out_dir / "params.npz")
    assert set(saved.files) == {"model_params", "model_S0", "mask"}
    assert saved["model_params"].shape[-1] == 12
    assert np.all(np.isfinite(saved["model_params"]))
