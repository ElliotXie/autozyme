"""Contract tests for the dipy patch (DTI tensor fitting).

Patched surface (2 targets):
  - dipy.reconst.dti.wls_fit_tensor   (module-level WLS fitter)
  - dipy.reconst.dti.TensorModel.fit  (the user-facing entry)

Fixture: minimal synthetic DWI volume (5x5x5x10) with one b=0 + 9
gradient directions. Contract pins:
  - .fit returns a TensorFit with evals.shape == (X,Y,Z,3) + evecs
  - zyme=False matches patched evals numerically
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
dipy = pytest.importorskip("dipy")
pytest.importorskip("dipy.reconst.dti")


@pytest.fixture
def dwi_fixture():
    """5x5x5 voxels, 10-direction DWI (1 b=0 + 9 b=1000)."""
    from dipy.core.gradients import gradient_table

    n_grads = 10
    bvals = np.concatenate([[0], np.full(n_grads - 1, 1000)])
    bvecs = np.zeros((n_grads, 3))
    rng = np.random.default_rng(0)
    bvecs[1:] = rng.normal(size=(n_grads - 1, 3))
    bvecs[1:] /= np.linalg.norm(bvecs[1:], axis=1)[:, None]
    gtab = gradient_table(bvals, bvecs=bvecs)

    # Synthetic DWI with brighter b=0 frame (canonical contrast).
    data = rng.random((5, 5, 5, n_grads), dtype=np.float32) * 100.0
    data[..., 0] *= 5.0
    return gtab, data


def test_tensor_model_fit_returns_tensorfit(dwi_fixture):
    """TensorModel.fit(data) returns TensorFit with evals + evecs."""
    import autozyme
    autozyme.activate("dipy")
    from dipy.reconst.dti import TensorModel

    gtab, data = dwi_fixture
    fit = TensorModel(gtab).fit(data)
    assert hasattr(fit, "evals"), "TensorFit missing .evals"
    assert hasattr(fit, "evecs"), "TensorFit missing .evecs"
    assert fit.evals.shape == (5, 5, 5, 3), (
        f"unexpected evals shape: {fit.evals.shape}"
    )


def test_tensor_model_fit_zyme_false_matches_vanilla(dwi_fixture):
    """Patched evals match vanilla within float tolerance."""
    import autozyme
    autozyme.activate("dipy")
    from dipy.reconst.dti import TensorModel

    gtab, data = dwi_fixture
    fast = TensorModel(gtab).fit(data).evals
    with autozyme.disabled():
        ref = TensorModel(gtab).fit(data).evals

    np.testing.assert_allclose(fast, ref, rtol=1e-4, atol=1e-6,
                               err_msg="DTI evals drift patched vs vanilla")


def test_tensor_model_fit_mask_respected(dwi_fixture):
    """Mask kwarg restricts the fit; un-masked voxels stay zero."""
    import autozyme
    autozyme.activate("dipy")
    from dipy.reconst.dti import TensorModel

    gtab, data = dwi_fixture
    mask = np.zeros(data.shape[:3], dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    fit = TensorModel(gtab).fit(data, mask=mask)
    # Voxels outside the mask should have zero evals.
    outside = ~mask
    if outside.any():
        outside_evals = fit.evals[outside]
        assert np.all(outside_evals == 0), "fit leaked outside mask"
