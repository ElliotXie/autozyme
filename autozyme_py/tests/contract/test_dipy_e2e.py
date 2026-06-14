"""End-to-end / wrapper-line tests for autozyme.dipy.

Wave-1 (`test_dipy_unit.py`) tested the WLS kernels. The existing `test_dipy.py`
drives TensorModel.fit on the unmasked + masked paths with return_S0_hat=False.
This file covers the remaining COVERAGE-VISIBLE branches in fast_tensor_model_fit
and _fast_wls_fit_tensor_inner that neither hits:

  - return_S0_hat=True: the chunked masked S0 path AND the unmasked S0 path.
  - the mask-shape-mismatch ValueError guard.
  - return_lower_triangular / return_leverages branches of the inner WLS fn.
  - explicit min_signal handling.
  - activate/restore lifecycle.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
dipy = pytest.importorskip("dipy")
pytest.importorskip("dipy.reconst.dti")

import autozyme
from autozyme import dipy as azdipy


@pytest.fixture
def dwi():
    from dipy.core.gradients import gradient_table

    n_grads = 10
    bvals = np.concatenate([[0], np.full(n_grads - 1, 1000)])
    rng = np.random.default_rng(0)
    bvecs = np.zeros((n_grads, 3))
    bvecs[1:] = rng.normal(size=(n_grads - 1, 3))
    bvecs[1:] /= np.linalg.norm(bvecs[1:], axis=1)[:, None]
    gtab = gradient_table(bvals, bvecs=bvecs)
    data = rng.random((4, 4, 4, n_grads), dtype=np.float32) * 100.0
    data[..., 0] *= 5.0
    return gtab, data


def test_fit_return_s0_hat_masked_matches_vanilla(dwi):
    """return_S0_hat=True on the chunked masked path: params + S0 match vanilla."""
    from dipy.reconst.dti import TensorModel

    autozyme.activate("dipy")
    gtab, data = dwi
    mask = np.zeros(data.shape[:3], dtype=bool)
    mask[1:3, 1:3, 1:3] = True

    fast = TensorModel(gtab, return_S0_hat=True).fit(data, mask=mask)
    with autozyme.disabled():
        ref = TensorModel(gtab, return_S0_hat=True).fit(data, mask=mask)

    assert fast.model_S0 is not None
    np.testing.assert_allclose(fast.evals, ref.evals, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(fast.model_S0), np.asarray(ref.model_S0),
        rtol=1e-4, atol=1e-5,
    )


def test_fit_return_s0_hat_unmasked_matches_vanilla(dwi):
    """return_S0_hat=True on the UNMASKED fallback path."""
    from dipy.reconst.dti import TensorModel

    autozyme.activate("dipy")
    gtab, data = dwi
    fast = TensorModel(gtab, return_S0_hat=True).fit(data)
    with autozyme.disabled():
        ref = TensorModel(gtab, return_S0_hat=True).fit(data)
    np.testing.assert_allclose(fast.evals, ref.evals, rtol=1e-4, atol=1e-6)
    assert fast.model_S0 is not None
    np.testing.assert_allclose(
        np.asarray(fast.model_S0), np.asarray(ref.model_S0),
        rtol=1e-4, atol=1e-5,
    )


def test_fit_mask_shape_mismatch_raises(dwi):
    """A mask whose shape != data.shape[:-1] hits the ValueError guard."""
    from dipy.reconst.dti import TensorModel

    autozyme.activate("dipy")
    gtab, data = dwi
    bad_mask = np.ones((2, 2, 2), dtype=bool)  # wrong shape
    with pytest.raises(ValueError, match="Mask is not the same shape"):
        TensorModel(gtab).fit(data, mask=bad_mask)


def test_inner_wls_lower_triangular_branch(dwi):
    """_fast_wls_fit_tensor_inner with return_lower_triangular=True returns the
    (..., 7) lower-triangular fit + None leverages."""
    from dipy.reconst.dti import TensorModel

    autozyme.activate("dipy")
    gtab, data = dwi
    model = TensorModel(gtab)
    design = model.design_matrix
    flat = data.reshape(-1, data.shape[-1])[:8]
    flat = np.maximum(flat, 1.0)  # avoid log(0)

    fit_lt, leverages = azdipy._fast_wls_fit_tensor_inner(
        design, flat, return_lower_triangular=True, min_signal=1.0,
    )
    assert fit_lt.shape[-1] == 7
    assert leverages is None


def test_inner_wls_leverages_branch(dwi):
    """return_leverages=True takes the pinv leverages branch."""
    from dipy.reconst.dti import TensorModel

    autozyme.activate("dipy")
    gtab, data = dwi
    model = TensorModel(gtab)
    design = model.design_matrix
    flat = np.maximum(data.reshape(-1, data.shape[-1])[:6], 1.0)
    fit_res, leverages = azdipy._fast_wls_fit_tensor_inner(
        design, flat, return_leverages=True, return_lower_triangular=True,
        min_signal=1.0,
    )
    assert leverages is not None and "leverages" in leverages


def test_activate_restore_lifecycle():
    from dipy.reconst import dti

    autozyme.deactivate("dipy")
    orig_wls = dti.wls_fit_tensor
    orig_fit = dti.TensorModel.fit
    assert autozyme.activate("dipy") is True
    assert dti.wls_fit_tensor is not orig_wls
    assert dti.TensorModel.fit is not orig_fit
    info = autozyme.inspect("dipy")
    assert info["status"] == "active"
    assert len(info["targets"]) == 2
    autozyme.deactivate("dipy")
    assert dti.wls_fit_tensor is orig_wls
    assert dti.TensorModel.fit is orig_fit
