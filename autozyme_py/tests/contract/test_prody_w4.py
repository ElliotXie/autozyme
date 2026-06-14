"""Wave-4 wrapper/dispatch-line tests for autozyme.prody.

Wave-1 (`_unit`) tested the pure helpers (_rigid_body_basis, _seeded_lobpcg_guess,
_lobpcg_residuals_ok, fast_build_hessian-vs-dense); the existing `test_prody.py`
drives ANM build + calcModes through the LOBPCG fast path. The COVERAGE-VISIBLE
lines still missing are the validation/dispatch/fallback branches that the happy
path skips:

  - line 70: the zero-norm ``continue`` in ``_seeded_lobpcg_guess``.
  - line 95: ``_lobpcg_residuals_ok`` returning False when fewer than n_modes
    residuals survive the keep mask.
  - lines 110-111: ``fast_build_hessian``'s no-getCoords / checkCoords-TypeError
    re-raise.
  - line 158: the callable-``gamma`` ``np.fromiter`` branch of fast_build_hessian.
  - line 229: ``fast_solve_eig``'s upstream delegate for non-sparse / reverse /
    n_modes>=dof / n_modes='all'.
  - lines 234-235: the ``expct_n_zeros is None`` default.
  - lines 286-309: the deterministic shift-invert ``eigsh`` fallback path (taken
    when expct_n_zeros != 6 or zeros=True, i.e. the LOBPCG fast path is skipped).
  - the smoke recipe (320-345) via a tiny synthetic PDB.

(The coords>=5000 LOBPCG branch at 262-266 needs a 15k-dof sparse solve and is
omitted to stay under the per-test time budget; it is pure-python dispatch, not
a numba kernel.)
"""
from __future__ import annotations

import os

import numpy as np
import pytest

prody = pytest.importorskip("prody")
pytest.importorskip("scipy")
yaml = pytest.importorskip("yaml")

import autozyme
from autozyme import prody as azprody


def _coords(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) * 10.0).astype(np.float64)


# --------------------------------------------------------------------------
# line 70: zero-norm field -> continue in _seeded_lobpcg_guess
# --------------------------------------------------------------------------
def test_seeded_guess_skips_zero_norm_field():
    """Collinear coords on the x-axis make several derived fields constant (zero
    norm after mean-subtraction), exercising the ``if norm == 0: continue``
    branch (line 70). The guess stays finite + correctly shaped."""
    n = 10
    coords = np.zeros((n, 3), dtype=np.float64)
    coords[:, 0] = np.arange(n, dtype=np.float64)  # y, z constant -> zero-norm fields
    guess = azprody._seeded_lobpcg_guess(coords, 8, np.random.default_rng(0),
                                         np.float64)
    assert guess.shape == (n * 3, 8)
    assert np.all(np.isfinite(guess))


# --------------------------------------------------------------------------
# line 95: residuals_ok returns False when fewer than n_modes survive
# --------------------------------------------------------------------------
def test_residuals_ok_too_few_selected_false():
    """If the keep mask selects fewer than n_modes residuals, the guard at line
    95 returns False (distinct from the shape-mismatch and large-residual
    rejections already covered in wave-1)."""
    order = np.array([0, 1, 2])
    keep = np.array([True, False, False])  # only 1 survives
    residual_history = [np.array([1e-9, 1e-9, 1e-9])]
    # request 2 modes but only 1 kept -> selected.shape[0] (1) < n_modes (2)
    assert azprody._lobpcg_residuals_ok(
        residual_history, order, keep, 2, np.array([1.0, 2.0])
    ) is False


# --------------------------------------------------------------------------
# lines 110-111: fast_build_hessian rejects coords with no getCoords + bad type
# --------------------------------------------------------------------------
def test_build_hessian_rejects_non_coords_object():
    """An object that is neither an ndarray nor has getCoords, and that
    prody.checkCoords rejects with TypeError, must surface the explicit
    TypeError re-raise (lines 109-113)."""

    class _NotCoords:
        pass

    anm = _FakeANM()
    with pytest.raises(TypeError, match="Numpy array or an object"):
        azprody.fast_build_hessian(anm, _NotCoords(), cutoff=15.0, gamma=1.0)


class _FakeANM:
    def _reset(self):
        self._hessian = None
        self._kirchhoff = None


# --------------------------------------------------------------------------
# line 158: callable gamma -> np.fromiter branch
# --------------------------------------------------------------------------
def test_build_hessian_callable_gamma():
    """Passing a callable gamma(dist2, i, j) routes through the per-pair
    np.fromiter branch (line 158). A constant-callable gamma must reproduce the
    scalar-gamma Hessian exactly."""
    coords = _coords(9, seed=3)

    def gamma_fn(dist2, i, j):
        return 1.0

    anm_cb = _FakeANM()
    azprody.fast_build_hessian(anm_cb, coords, cutoff=15.0, gamma=gamma_fn)
    anm_const = _FakeANM()
    azprody.fast_build_hessian(anm_const, coords, cutoff=15.0, gamma=1.0)
    np.testing.assert_allclose(
        anm_cb._hessian.toarray(), anm_const._hessian.toarray(),
        rtol=1e-12, atol=1e-14,
    )


# --------------------------------------------------------------------------
# line 229: fast_solve_eig upstream delegate (non-sparse / n_modes='all')
# --------------------------------------------------------------------------
def test_solve_eig_dense_matrix_delegates_to_upstream():
    """A dense (non-sparse) M is not eligible for the sparse fast path, so
    fast_solve_eig delegates to the captured upstream solveEig (line 229).
    Compare eigenvalues against the upstream result directly."""
    autozyme.activate("prody")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((12, 12))
    M = A @ A.T + 12 * np.eye(12)  # dense SPD
    vals_fast, vecs_fast, inv_fast = azprody.fast_solve_eig(
        M, n_modes=3, zeros=False, expct_n_zeros=0,
    )
    vals_ref, vecs_ref, inv_ref = azprody._orig_solve_eig(
        M, n_modes=3, zeros=False, expct_n_zeros=0,
    )
    np.testing.assert_allclose(np.sort(vals_fast), np.sort(vals_ref),
                               rtol=1e-8, atol=1e-10)


def test_solve_eig_n_modes_ge_dof_delegates():
    """A sparse M with n_modes >= dof is not eligible for the truncated fast
    path, so fast_solve_eig delegates to upstream (line 227-232). Eigenvalues
    match the upstream solveEig."""
    from scipy import sparse

    autozyme.activate("prody")
    rng = np.random.default_rng(1)
    A = rng.standard_normal((10, 10))
    dense = A @ A.T + 10 * np.eye(10)
    M = sparse.csr_matrix(dense)
    # n_modes == dof -> the `n_modes >= dof` guard fires -> upstream delegate.
    vals, vecs, inv = azprody.fast_solve_eig(M, n_modes=10, expct_n_zeros=0)
    ref_vals, _, _ = azprody._orig_solve_eig(M, n_modes=10, expct_n_zeros=0)
    np.testing.assert_allclose(np.sort(vals), np.sort(ref_vals),
                               rtol=1e-8, atol=1e-10)


# --------------------------------------------------------------------------
# lines 234-235 (expct_n_zeros None default) + 286-309 (eigsh fallback)
# --------------------------------------------------------------------------
def test_solve_eig_eigsh_fallback_path():
    """With ``expct_n_zeros=6`` but the build coords stripped, fast_solve_eig
    cannot take the LOBPCG fast path (coords guard False) and instead runs the
    deterministic shift-invert ``eigsh`` path (286-309). The non-zero
    eigenvalues match the upstream solveEig on the same sparse Hessian."""
    autozyme.activate("prody")
    coords = _coords(40, seed=11)
    anm = _FakeANM()
    azprody.fast_build_hessian(anm, coords, cutoff=15.0, gamma=1.0)
    H = anm._hessian
    # Strip the attached coords so the LOBPCG branch's coords-guard is False ->
    # the eigsh fallback runs.
    if hasattr(H, "_autozyme_anm_coords"):
        del H._autozyme_anm_coords

    vals_fast, vecs_fast, inv_fast = azprody.fast_solve_eig(
        H, n_modes=6, zeros=False, expct_n_zeros=6,
    )
    vals_ref, vecs_ref, inv_ref = azprody._orig_solve_eig(
        H, n_modes=6, zeros=False, expct_n_zeros=6,
    )
    assert len(vals_fast) == 6
    # Both paths are iterative shift-invert eigsh solves; near the smallest
    # (rigid-body-adjacent) modes the solver tolerance leaves ~1e-3 relative
    # noise, which is below the task's calcModes concordance budget.
    np.testing.assert_allclose(np.sort(vals_fast), np.sort(vals_ref),
                               rtol=2e-3, atol=1e-4)


def test_solve_eig_expct_n_zeros_none_default():
    """expct_n_zeros omitted -> None -> defaults to 0 (lines 234-235). With
    coords stripped, the eigsh path runs and returns the requested modes,
    finite. (n_zeros is auto-detected from the spectrum, so the non-rigid modes
    are still resolved.)"""
    autozyme.activate("prody")
    coords = _coords(40, seed=13)
    anm = _FakeANM()
    azprody.fast_build_hessian(anm, coords, cutoff=15.0, gamma=1.0)
    H = anm._hessian
    if hasattr(H, "_autozyme_anm_coords"):
        del H._autozyme_anm_coords
    vals, vecs, inv = azprody.fast_solve_eig(H, n_modes=4, zeros=False)
    assert len(vals) == 4
    assert np.all(np.isfinite(vals)) and np.all(np.isfinite(inv))


def test_solve_eig_zeros_true_branch():
    """zeros=True takes the eigsh fallback (LOBPCG fast path requires zeros=False)
    and the ``else`` invvals-with-zero-guard branch (303-308)."""
    autozyme.activate("prody")
    coords = _coords(30, seed=12)
    anm = _FakeANM()
    azprody.fast_build_hessian(anm, coords, cutoff=15.0, gamma=1.0)
    H = anm._hessian
    vals, vecs, invvals = azprody.fast_solve_eig(
        H, n_modes=6, zeros=True, expct_n_zeros=6,
    )
    assert len(vals) == 6
    assert vecs.shape == (H.shape[0], 6)
    # invvals zero-guards: where eigval==0, inv is 0 (not inf).
    assert np.all(np.isfinite(invvals))


# --------------------------------------------------------------------------
# smoke recipe (320-345) via a tiny synthetic PDB
# --------------------------------------------------------------------------
def _write_tiny_pdb(path, n=30, seed=0):
    """Write a minimal CA-only PDB the prody smoke loader can parse."""
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 3)) * 20.0
    lines = []
    for i in range(n):
        x, y, z = coords[i]
        # PDB ATOM record, fixed-column format; CA of ALA chain A.
        lines.append(
            f"ATOM  {i + 1:>5} CA   ALA A{i + 1:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


_SMOKE_N_ATOMS = 90  # >> _N_MODES(20)+6 block size so LOBPCG runs (no dense fallback)


def _make_task_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_tiny_pdb(str(data_dir / "tiny.pdb"), n=_SMOKE_N_ATOMS)
    task = {"datasets": [{"tier": "small", "name": "tiny",
                          "path": "./data/tiny.pdb"}]}
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    return str(tmp_path)


def test_smoke_load_call_save_roundtrip(tmp_path):
    """_smoke_load parses the PDB + selects CA; _smoke_call builds the ANM +
    modes; _smoke_save writes anm.npz (covers 320-345)."""
    autozyme.activate("prody")
    task_dir = _make_task_dir(tmp_path)
    inputs = azprody._smoke_load(task_dir, "small")
    assert inputs["sel"].numAtoms() == _SMOKE_N_ATOMS

    result = azprody._smoke_call(inputs)
    assert "eigvals" in result and "eigvecs" in result
    assert len(result["eigvals"]) == azprody._N_MODES

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    azprody._smoke_save(result, str(out_dir))
    saved = np.load(out_dir / "anm.npz")
    assert set(saved.files) >= {"eigvals", "eigvecs", "n_atoms", "cutoff",
                                "n_modes"}
    assert int(saved["n_atoms"]) == _SMOKE_N_ATOMS
    assert saved["eigvecs"].shape[0] == _SMOKE_N_ATOMS * 3
