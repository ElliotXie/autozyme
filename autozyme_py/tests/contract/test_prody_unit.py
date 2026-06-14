"""Unit tests for the pure numpy/scipy helpers in autozyme.prody.

The contract test (test_prody.py) drives ANM build + eigensolve end to end. Here
we test the self-contained linear-algebra helpers directly:

  - _rigid_body_basis       6 orthonormal rigid-body modes (3 trans + 3 rot)
  - _seeded_lobpcg_guess    deterministic seeded initial guess
  - _lobpcg_residuals_ok    residual acceptance logic
  - fast_build_hessian      sparse ANM Hessian, vs a dense reference build

prody must import for the module to load (the Hessian builder uses only
prody.checkENMParameters for validation; the numerics are pure numpy/scipy).
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("prody")

from autozyme import prody as azprody


def _coords(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) * 10.0).astype(np.float64)


# --------------------------------------------------------------------------
# _rigid_body_basis
# --------------------------------------------------------------------------
def test_rigid_body_basis_orthonormal():
    coords = _coords(10, seed=1)
    Q = azprody._rigid_body_basis(coords)
    assert Q.shape == (30, 6)
    # Columns are orthonormal (QR reduced).
    np.testing.assert_allclose(Q.T @ Q, np.eye(6), rtol=1e-10, atol=1e-10)


def test_rigid_body_basis_spans_translations_and_rotations():
    coords = _coords(6, seed=2)
    n = coords.shape[0]
    Q = azprody._rigid_body_basis(coords)
    # The pre-QR rigid-body modes must lie in span(Q): translations along each
    # axis and infinitesimal rotations about the centroid.
    centered = coords - coords.mean(axis=0)
    trans = []
    for axis in range(3):
        v = np.zeros(n * 3)
        v[axis::3] = 1.0
        trans.append(v)
    # rotation about z: (-y, x, 0) per atom
    rot_z = np.zeros(n * 3)
    rot_z[0::3] = -centered[:, 1]
    rot_z[1::3] = centered[:, 0]
    P = Q @ Q.T  # projector onto the rigid-body subspace
    for v in trans + [rot_z]:
        np.testing.assert_allclose(P @ v, v, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------
# _seeded_lobpcg_guess
# --------------------------------------------------------------------------
def test_seeded_guess_shape_dtype_and_determinism():
    coords = _coords(12, seed=3)
    g1 = azprody._seeded_lobpcg_guess(
        coords, 8, np.random.default_rng(0), np.float64
    )
    g2 = azprody._seeded_lobpcg_guess(
        coords, 8, np.random.default_rng(0), np.float64
    )
    assert g1.shape == (36, 8)
    assert g1.dtype == np.float64
    # Same seed -> identical guess.
    np.testing.assert_array_equal(g1, g2)


def test_seeded_guess_float32_dtype():
    coords = _coords(9, seed=4)
    g = azprody._seeded_lobpcg_guess(
        coords, 4, np.random.default_rng(1), np.float32
    )
    assert g.dtype == np.float32
    assert g.shape == (27, 4)
    assert np.all(np.isfinite(g))


# --------------------------------------------------------------------------
# _lobpcg_residuals_ok
# --------------------------------------------------------------------------
def test_residuals_ok_empty_history_false():
    assert azprody._lobpcg_residuals_ok(
        [], np.array([0, 1]), np.array([True, True]), 1, np.array([1.0])
    ) is False


def test_residuals_ok_accepts_small_residuals():
    order = np.array([0, 1, 2])
    keep = np.array([True, True, True])
    eigvals = np.array([1.0, 2.0])
    # Residuals well below 1e-3 * |eigval|.
    residual_history = [np.array([1e-8, 1e-8, 1e-8])]
    assert azprody._lobpcg_residuals_ok(
        residual_history, order, keep, 2, eigvals
    ) is True


def test_residuals_ok_rejects_large_residuals():
    order = np.array([0, 1, 2])
    keep = np.array([True, True, True])
    eigvals = np.array([1.0, 2.0])
    residual_history = [np.array([1.0, 1.0, 1.0])]  # >> 1e-3 threshold
    assert azprody._lobpcg_residuals_ok(
        residual_history, order, keep, 2, eigvals
    ) is False


def test_residuals_ok_shape_mismatch_false():
    order = np.array([0, 1, 2])
    keep = np.array([True, True, True])
    # final residual has wrong length vs `order`.
    residual_history = [np.array([1e-9, 1e-9])]
    assert azprody._lobpcg_residuals_ok(
        residual_history, order, keep, 2, np.array([1.0, 2.0])
    ) is False


# --------------------------------------------------------------------------
# fast_build_hessian vs dense reference ANM build
# --------------------------------------------------------------------------
def _dense_anm_reference(coords, cutoff, gamma):
    n = coords.shape[0]
    dof = n * 3
    H = np.zeros((dof, dof))
    K = np.zeros((n, n))
    cutoff2 = cutoff * cutoff
    for i in range(n):
        for j in range(i + 1, n):
            diff = coords[i] - coords[j]
            d2 = float(diff @ diff)
            if d2 > cutoff2:
                continue
            block = gamma * np.outer(diff, diff) / d2
            i3, j3 = i * 3, j * 3
            H[i3:i3 + 3, j3:j3 + 3] -= block
            H[j3:j3 + 3, i3:i3 + 3] -= block
            H[i3:i3 + 3, i3:i3 + 3] += block
            H[j3:j3 + 3, j3:j3 + 3] += block
            K[i, j] -= gamma
            K[j, i] -= gamma
            K[i, i] += gamma
            K[j, j] += gamma
    return H, K


class _FakeANM:
    """Minimal stand-in for ANMBase exposing the attributes fast_build_hessian
    reads/writes. fast_build_hessian only calls self._reset() and assigns a
    handful of attributes — no other prody ANM behaviour is touched."""
    def _reset(self):
        self._hessian = None
        self._kirchhoff = None


def test_fast_build_hessian_matches_dense_reference():
    coords = _coords(10, seed=7)
    cutoff, gamma = 12.0, 1.0
    anm = _FakeANM()
    azprody.fast_build_hessian(anm, coords, cutoff=cutoff, gamma=gamma)

    H_ref, K_ref = _dense_anm_reference(coords, cutoff, gamma)
    H_got = anm._hessian.toarray()
    K_got = anm._kirchhoff.toarray()
    np.testing.assert_allclose(H_got, H_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(K_got, K_ref, rtol=1e-10, atol=1e-10)


def test_fast_build_hessian_row_sums_zero():
    # An ANM Hessian and Kirchhoff matrix have zero row sums (rigid-body
    # translation invariance / graph Laplacian property).
    coords = _coords(9, seed=8)
    anm = _FakeANM()
    azprody.fast_build_hessian(anm, coords, cutoff=15.0, gamma=1.0)
    K = anm._kirchhoff.toarray()
    np.testing.assert_allclose(K.sum(axis=1), 0.0, atol=1e-10)
    H = anm._hessian.toarray()
    # Hessian: summing the 3 Cartesian blocks per atom-row also cancels.
    n = coords.shape[0]
    H_blocks = H.reshape(n, 3, n, 3).sum(axis=2)  # (n, 3, 3) sum over atom cols
    np.testing.assert_allclose(H_blocks, 0.0, atol=1e-10)


def test_fast_build_hessian_attaches_coords():
    coords = _coords(7, seed=9)
    anm = _FakeANM()
    azprody.fast_build_hessian(anm, coords, cutoff=15.0, gamma=1.0)
    # fast_solve_eig reads back the build coords from the sparse Hessian.
    attached = anm._hessian._autozyme_anm_coords
    np.testing.assert_allclose(attached, coords)
    assert anm._n_atoms == 7
    assert anm._dof == 21
