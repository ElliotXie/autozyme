"""Patch for prody.calcANM (ANM build + eigensolve).

Lifted from autozyme task `test_prody`. Two coordinated patches:
  - prody.dynamics.anm.ANMBase.buildHessian (class method)
      → vectorized pair extraction + sparse COO assembly in float64. The
        build coords are attached to the sparse Hessian instance so solveEig
        can build the matching rigid-body constraints without global state.
  - prody.dynamics.anm.solveEig (module function)
      → LOBPCG with rigid-body constraints + diagonal preconditioner for
        the n_modes < dof + 6 zero-eigenvalues case (typical ANM use).
        Falls back to the upstream eigsh path for general matrices.
"""
from __future__ import annotations

import os

import numpy as np

import prody
import prody.dynamics.anm as prody_anm
from prody.utilities.eigtools import ZERO

from autozyme._core import register_patch
from autozyme._utils import resolve_dataset_path

# Capture upstream originals BEFORE register_patch runs.
_orig_build_hessian = prody_anm.ANMBase.buildHessian
_orig_solve_eig = prody_anm.solveEig


def _rigid_body_basis(coords):
    centered = coords - coords.mean(axis=0)
    x = centered[:, 0]
    y = centered[:, 1]
    z = centered[:, 2]
    basis = np.zeros((coords.shape[0] * 3, 6), dtype=coords.dtype)
    basis[0::3, 0] = 1.0
    basis[1::3, 1] = 1.0
    basis[2::3, 2] = 1.0
    basis[1::3, 3] = -z
    basis[2::3, 3] = y
    basis[0::3, 4] = z
    basis[2::3, 4] = -x
    basis[0::3, 5] = -y
    basis[1::3, 5] = x
    q, _ = np.linalg.qr(basis, mode="reduced")
    return q


def _seeded_lobpcg_guess(coords, n_cols, rng, dtype):
    dof = coords.shape[0] * 3
    guess = (0.05 * rng.standard_normal((dof, n_cols))).astype(dtype)
    centered = coords - coords.mean(axis=0)
    scale = np.std(centered, axis=0)
    scale[scale == 0] = 1.0
    normed = centered / scale
    x = normed[:, 0]
    y = normed[:, 1]
    z = normed[:, 2]
    fields = (x, y, z, x * y, x * z, y * z,
              x * x - y * y, y * y - z * z, z * z - x * x,
              x * x + y * y + z * z)
    col = 0
    for field in fields:
        if col >= n_cols:
            break
        field = field - field.mean()
        norm = np.linalg.norm(field)
        if norm == 0:
            continue
        field = (field / norm).astype(dtype)
        for axis in range(3):
            if col >= n_cols:
                break
            guess[axis::3, col] += field
            col += 1
    return guess


def _lobpcg_residuals_ok(residual_history, order, keep, n_modes, eigvals):
    """Accept LOBPCG only when the returned physical modes are well-resolved.

    SciPy may return vectors after hitting maxiter even when it emits a
    convergence warning. That can be fine, but it should not be a correctness
    assumption. Treat LOBPCG as an opportunistic fast path; if final residuals
    are too large, fall through to the deterministic shift-invert eigsh path.
    """
    if not residual_history:
        return False
    final_residuals = np.asarray(residual_history[-1], dtype=np.float64)
    if final_residuals.shape[0] != order.shape[0]:
        return False
    selected = final_residuals[order][keep][:n_modes]
    if selected.shape[0] < n_modes or not np.all(np.isfinite(selected)):
        return False
    scaled = selected / np.maximum(np.abs(eigvals), 1.0)
    return bool(np.max(scaled) <= 1.0e-3)


def fast_build_hessian(self, coords, cutoff=15.0, gamma=1.0, **kwargs):
    from scipy import sparse as scipy_sparse

    try:
        coords = (
            coords._getCoords() if hasattr(coords, "_getCoords") else coords.getCoords()
        )
    except AttributeError:
        try:
            prody_anm.checkCoords(coords)
        except TypeError:
            raise TypeError(
                "coords must be a Numpy array or an object with `getCoords` method"
            )

    cutoff, g, gamma = prody_anm.checkENMParameters(cutoff, gamma)
    # Final audit found that the old float32 threshold was not a safe
    # correctness boundary: the 13k-C-alpha OOD structure produced
    # catastrophic slow-mode errors before it was special-cased to float64.
    # Keep the sparse/iterative algorithmic wins, but build the matrix in the
    # same precision class as upstream by default.
    hessian_dtype = np.float64
    coords = np.asarray(coords, dtype=hessian_dtype)
    self._reset()
    self._cutoff = cutoff
    self._gamma = g
    n_atoms = coords.shape[0]
    dof = n_atoms * 3
    cutoff2 = cutoff * cutoff

    n_pairs = 0
    for i in range(n_atoms):
        diffs = coords[i + 1:, :] - coords[i]
        dist2 = np.einsum("ij,ij->i", diffs, diffs)
        n_pairs += int(np.count_nonzero(dist2 <= cutoff2))

    pair_i = np.empty(n_pairs, dtype=np.int32)
    pair_j = np.empty(n_pairs, dtype=np.int32)
    pair_diffs = np.empty((n_pairs, 3), dtype=hessian_dtype)
    pair_dist2 = np.empty(n_pairs, dtype=hessian_dtype)

    p = 0
    for i in range(n_atoms):
        diffs = coords[i + 1:, :] - coords[i]
        dist2_all = np.einsum("ij,ij->i", diffs, diffs)
        hits = np.nonzero(dist2_all <= cutoff2)[0]
        n_hits = len(hits)
        if n_hits:
            sl = slice(p, p + n_hits)
            pair_i[sl] = i
            pair_j[sl] = i + 1 + hits
            pair_diffs[sl] = diffs[hits]
            pair_dist2[sl] = dist2_all[hits]
            p += n_hits

    if isinstance(g, (float, int, np.floating, np.integer)):
        g_values = np.full(n_pairs, float(g), dtype=hessian_dtype)
    else:
        g_values = np.fromiter(
            (gamma(float(dist2), int(i), int(j))
             for dist2, i, j in zip(pair_dist2, pair_i, pair_j)),
            dtype=hessian_dtype, count=n_pairs,
        )

    a_idx = np.repeat(np.arange(3, dtype=np.int32), 3)
    b_idx = np.tile(np.arange(3, dtype=np.int32), 3)
    pair_i3 = pair_i * 3
    pair_j3 = pair_j * 3
    block = (pair_diffs[:, a_idx] * pair_diffs[:, b_idx]
             * (-g_values / pair_dist2)[:, None])

    h_rows = np.empty((n_pairs, 36), dtype=np.int32)
    h_cols = np.empty((n_pairs, 36), dtype=np.int32)
    h_data = np.empty((n_pairs, 36), dtype=hessian_dtype)

    h_rows[:, 0:9] = pair_i3[:, None] + a_idx
    h_cols[:, 0:9] = pair_j3[:, None] + b_idx
    h_data[:, 0:9] = block
    h_rows[:, 9:18] = pair_j3[:, None] + a_idx
    h_cols[:, 9:18] = pair_i3[:, None] + b_idx
    h_data[:, 9:18] = block
    h_rows[:, 18:27] = pair_i3[:, None] + a_idx
    h_cols[:, 18:27] = pair_i3[:, None] + b_idx
    h_data[:, 18:27] = -block
    h_rows[:, 27:36] = pair_j3[:, None] + a_idx
    h_cols[:, 27:36] = pair_j3[:, None] + b_idx
    h_data[:, 27:36] = -block

    hessian = scipy_sparse.coo_matrix(
        (h_data.ravel(), (h_rows.ravel(), h_cols.ravel())), shape=(dof, dof)
    ).tocsr()
    hessian.sum_duplicates()

    k_rows = np.empty((n_pairs, 4), dtype=np.int32)
    k_cols = np.empty((n_pairs, 4), dtype=np.int32)
    k_data = np.empty((n_pairs, 4), dtype=hessian_dtype)
    k_rows[:, 0] = pair_i
    k_cols[:, 0] = pair_j
    k_data[:, 0] = -g_values
    k_rows[:, 1] = pair_j
    k_cols[:, 1] = pair_i
    k_data[:, 1] = -g_values
    k_rows[:, 2] = pair_i
    k_cols[:, 2] = pair_i
    k_data[:, 2] = g_values
    k_rows[:, 3] = pair_j
    k_cols[:, 3] = pair_j
    k_data[:, 3] = g_values
    kirchhoff = scipy_sparse.coo_matrix(
        (k_data.ravel(), (k_rows.ravel(), k_cols.ravel())),
        shape=(n_atoms, n_atoms),
    ).tocsr()
    kirchhoff.sum_duplicates()

    hessian._autozyme_anm_coords = coords
    self._kirchhoff = kirchhoff
    self._hessian = hessian
    self._n_atoms = n_atoms
    self._dof = dof


def fast_solve_eig(M, n_modes=None, zeros=False, turbo=True, expct_n_zeros=None,
                   reverse=False, **kwargs):
    from scipy.sparse import issparse
    from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg

    dof = M.shape[0]
    if (not issparse(M) or reverse or n_modes is None
            or str(n_modes).lower() == "all" or n_modes >= dof):
        return _orig_solve_eig(
            M, n_modes=n_modes, zeros=zeros, turbo=turbo,
            expct_n_zeros=expct_n_zeros, reverse=reverse, **kwargs,
        )

    if expct_n_zeros is None:
        expct_n_zeros = 0

    coords = getattr(M, "_autozyme_anm_coords", None)
    if (not zeros and expct_n_zeros == 6
            and coords is not None
            and coords.shape[0] * 3 == dof
            and coords.shape[0] < 12000):
        diag = M.diagonal()
        inv_diag = np.zeros_like(diag)
        nonzero_diag = np.abs(diag) > 0
        inv_diag[nonzero_diag] = 1.0 / diag[nonzero_diag]

        def prec_matvec(x):
            return inv_diag * x

        def prec_matmat(x):
            return inv_diag[:, None] * x

        precond = LinearOperator(
            M.shape, matvec=prec_matvec, matmat=prec_matmat, dtype=M.dtype
        )
        lobpcg_modes = min(dof - 7, n_modes + 6)
        rng = np.random.default_rng(0)
        if coords.shape[0] < 5000:
            guess = _seeded_lobpcg_guess(coords, lobpcg_modes, rng, M.dtype)
            maxiter = 65
        else:
            guess = rng.standard_normal((dof, lobpcg_modes)).astype(M.dtype)
            # 5k-10k C-alpha structures are still cheap enough for a few more
            # LOBPCG iterations. At 80 iterations, single-thread OpenBLAS
            # occasionally stopped just above the 1e-3 eigvalue audit gate.
            maxiter = 140
        constraints = _rigid_body_basis(coords)
        values, vectors, residual_history = lobpcg(
            M, guess, M=precond, Y=constraints,
            tol=1e-8, maxiter=maxiter, largest=False,
            retResidualNormsHistory=True,
        )
        order = np.argsort(values)
        values = values[order]
        vectors = vectors[:, order]
        keep = values >= ZERO
        if int(np.count_nonzero(keep)) >= n_modes:
            eigvals = values[keep][:n_modes]
            if _lobpcg_residuals_ok(
                residual_history, order, keep, n_modes, eigvals
            ):
                eigvecs = vectors[:, keep][:, :n_modes]
                invvals = 1.0 / eigvals
                return eigvals, eigvecs, invvals

    k = min(dof - 2, max(n_modes + expct_n_zeros, n_modes + 2))
    values, vectors = eigsh(M, k=k, sigma=-1e-8, which="LM", tol=1e-10)
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]

    n_zeros = int(np.sum(values < ZERO))
    if not zeros:
        final_n_modes = n_zeros + n_modes
        if final_n_modes > len(values):
            return _orig_solve_eig(
                M, n_modes=n_modes, zeros=zeros, turbo=turbo,
                expct_n_zeros=expct_n_zeros, reverse=reverse, **kwargs,
            )
        eigvals = values[n_zeros:final_n_modes]
        eigvecs = vectors[:, n_zeros:final_n_modes]
        invvals = 1.0 / eigvals
    else:
        eigvals = values[:n_modes]
        eigvecs = vectors[:, :n_modes]
        invvals = np.zeros_like(eigvals)
        nz = np.abs(eigvals) > 0
        invvals[nz] = 1.0 / eigvals[nz]
    return eigvals, eigvecs, invvals


# ---------- smoke recipe ----------

_CUTOFF = 15.0
_GAMMA = 1.0
_N_MODES = 20


def _smoke_load(task_dir, tier):
    import yaml
    with open(os.path.join(task_dir, "task.yaml"), encoding="utf-8") as f:
        task = yaml.safe_load(f)
    ds = next(d for d in task["datasets"] if d["tier"] == tier)
    pdb_path = resolve_dataset_path(task_dir, ds["path"])
    if pdb_path.lower().endswith(".cif"):
        ag = prody.parseMMCIF(pdb_path)
    else:
        ag = prody.parsePDB(pdb_path)
    sel = ag.select("calpha")
    return {"sel": sel, "name": os.path.basename(pdb_path)}


def _smoke_call(inputs):
    anm = prody.ANM(name=inputs["name"])
    anm.buildHessian(inputs["sel"], cutoff=_CUTOFF, gamma=_GAMMA)
    anm.calcModes(n_modes=_N_MODES, zeros=False, turbo=True)
    return {
        "eigvals": anm.getEigvals(),
        "eigvecs": anm.getEigvecs(),
        "n_atoms": inputs["sel"].numAtoms(),
    }


def _smoke_save(result, dir, **kwargs):
    np.savez_compressed(
        os.path.join(dir, "anm.npz"),
        eigvals=result["eigvals"].astype(np.float64),
        eigvecs=result["eigvecs"].astype(np.float64),
        n_atoms=np.int64(result["n_atoms"]),
        cutoff=np.float64(_CUTOFF),
        n_modes=np.int64(_N_MODES),
    )


register_patch(
    name="prody",
    targets=[
        ("prody.dynamics.anm.ANMBase", "buildHessian", fast_build_hessian),
        ("prody.dynamics.anm", "solveEig", fast_solve_eig),
    ],
    smoke={"load": _smoke_load, "call": _smoke_call, "save": _smoke_save},
    tested_against="prody 2.6.1",
    tested_upstream_versions={"prody": ["2.6.1"]},
)
