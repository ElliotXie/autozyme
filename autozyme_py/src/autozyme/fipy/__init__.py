"""Patch for FiPy's ``Term.solve`` — caches the assembled linear system
across timesteps when the equation + mesh + dt are stable.

Lifted from autozyme task ``test_fipy`` (round 6-8 keep). Targets the
canonical FV time-stepping pattern:

    eqn = TransientTerm() == DiffusionTerm(coeff=const)
    for _ in range(n_steps):
        eqn.solve(var=v, dt=dt)

For this pattern on a fixed Grid2D + constant diffusion coefficient + fixed
dt, the assembled matrix ``M/dt - L_diff`` and its LU factor are
**step-invariant** — only the right-hand side depends on the previous
timestep's ``var.value``. Upstream rebuilds and refactors both every step
(~30+ Python method dispatches: ``_prepareLinearSystem`` + ``_buildAndAddMatrices``
+ ``getDefaultSolver`` + ``_verifyVar`` + ``_checkVar`` + ``_storeMatrix``
+ ``solver._solve(_Lxb)`` + ``_scatterGhosts`` + ``_cleanup``).

Three coordinated layered patches (each one a strict subset of upstream's
work — same numerics, fewer dispatches):

  1. ``LinearLUSolver._solve_`` — content-keyed ``scipy.sparse.linalg.splu``
     cache. CSR ``data``/``indices`` hashed via ``blake2b`` (digest_size=16);
     identity fast-path skips the hash when the same Python matrix object
     is passed back-to-back (the round-3 stable-matrix path). The iterative
     refinement loop runs as upstream — only ``splu`` is reused.
  2. ``_BinaryTerm._buildAndAddMatrices`` — caches the FULL combined sparse
     matrix and the diffusion sub-term's RHS contribution across timesteps.
     On the first call, runs upstream and snapshots: the combined matrix
     (deep-copied because upstream's next ``_getMatrix`` zeroes the storage),
     the diff sub-term's tmpRHSvector, and the transient sub-term's
     ``coeffVectors[old value]`` (per-cell ``cell_volume / dt``, named
     ``mass_per_dt`` here). On subsequent calls, compute ``b_trans`` directly
     from ``var.old.value`` (one O(N) numpy op) and add to the cached
     ``b_diff`` — no zeroEntries, no ``L.addAt`` calls, no sparse adds.
  3. ``Term.solve`` (registered target per round 6) — bypasses the entire
     upstream ``Term.solve`` machinery on cache hit. First call goes through
     the upstream path (which populates the two lower-level caches above),
     then we fold the LU's maxdiag scaling once and stash
     ``(LU, mass_per_dt_scaled, b_diff_scaled)`` keyed by ``(id(self),
     id(var), dt, state_sig)``. ``state_sig`` fingerprints constraints,
     boundary conditions, and coefficient object identity. Per call on hit,
     three numpy ops + one LU.solve:

         b_scaled = var.old.value * mass_per_dt_scaled + b_diff_scaled
         x = LU.solve(b_scaled)
         var.value = x.reshape(var.shape)

     Skips ``_prepareLinearSystem`` + ``getDefaultSolver`` + ``_verifyVar`` +
     ``_checkVar`` + ``_buildAndAddMatrices`` wrapper + ``_buildCache`` +
     ``_storeMatrix`` + ``solver._solve(_Lxb)`` + ``_scatterGhosts`` +
     unit-factor scaling + ``_cleanup``.

Why ``Term.solve`` is the registered (user-facing) target and not the
lower-level ``LinearLUSolver._solve_``: FiPy's ``Term.solve`` is the public
API a typical FV user code calls (``eqn.solve(var=v, dt=dt)``). Patching it
at the top of the dispatch chain means each user-loop iteration pays only
the cache-hit fast-path cost — without this layer the per-call savings on
the inner two layers are still gated behind the outer Python dispatch.

Caveat (algorithmic kind): on cache miss (first step of a new equation),
the patch runs the full upstream path; concordance metrics are bit-level
under the standard LinearLUSolver (which is deterministic). On cache hit
the LU factor is reused — ``rel_l2_err`` is at FP noise (~1e-15) as long as
the underlying matrix is genuinely step-invariant, which it is for
``TransientTerm() == DiffusionTerm(coeff=const)`` on a fixed mesh.

Cache key: ``(id(self), id(var), float(dt), state_sig)``. Changing ``dt``,
swapping the var, changing constraint values/masks, changing explicit
boundary-condition values/faces, or reassigning a coefficient object triggers a
cache miss and a fresh upstream path; this is correct when the matrix or RHS
genuinely changes.
"""
from __future__ import annotations

import copy
import hashlib
import os

import numpy as np

from fipy.solvers.scipy.linearLUSolver import LinearLUSolver
from fipy.terms.abstractDiffusionTerm import _AbstractDiffusionTerm
from fipy.terms.binaryTerm import _BinaryTerm
from fipy.terms.term import Term
from fipy.tools import numerix
from scipy.sparse.linalg import splu

from autozyme._core import register_patch


# ============================================================
# Capture upstream originals BEFORE register_patch runs.
# ============================================================
_orig_solve_ = LinearLUSolver._solve_
_orig_binary_build = _BinaryTerm._buildAndAddMatrices
_orig_term_solve = Term.solve


# ============================================================
# Layer 1: content-keyed splu cache for LinearLUSolver._solve_
# ============================================================
# sig (blake2b digest) -> (LU, maxdiag, L_scaled)
_LU_CACHE: dict[bytes, tuple] = {}
# Identity-shortcut: skip the blake2b on back-to-back calls with the same L object.
_LU_LAST_OBJ: list = [None, None]  # [last L id, last cache entry]


def _matrix_signature(L_csr) -> bytes:
    """Cheap content-based signature for a CSR sparse matrix.

    Hashes shape + data + indices via blake2b. For n=240 the assembled FV
    Laplacian has ~290k nonzeros (~2.3 MB doubles); blake2b runs at multi-GB/s
    so hashing cost is <1 ms vs ~250 ms splu — a >100x margin.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(L_csr.shape[0].to_bytes(8, "little"))
    h.update(L_csr.shape[1].to_bytes(8, "little"))
    h.update(L_csr.data.tobytes())
    h.update(L_csr.indices.tobytes())
    return h.digest()


def fast_solve_(self, L, x, b):
    """Drop-in for ``LinearLUSolver._solve_`` that caches splu across timesteps.

    Cache miss: identical to upstream — scale by 1/maxdiag, splu factor, run
    iterative refinement against the scaled matrix.

    Cache hit: reuse stored ``LU`` + ``maxdiag``, run the same iterative
    refinement loop on the cached scaled matrix. Identity fast-path skips
    the blake2b hash when the same Python object L is passed back-to-back.
    """
    cached = None
    if _LU_LAST_OBJ[0] is L:
        cached = _LU_LAST_OBJ[1]
    if cached is None:
        sig = _matrix_signature(L)
        cached = _LU_CACHE.get(sig)
        if cached is not None:
            _LU_LAST_OBJ[0] = L
            _LU_LAST_OBJ[1] = cached
    if cached is not None:
        LU, maxdiag, L_scaled = cached
        b_scaled = b * (1.0 / maxdiag)
    else:
        diag = L.diagonal()
        maxdiag = max(numerix.absolute(diag))
        L_scaled = L * (1.0 / maxdiag)
        b_scaled = b * (1.0 / maxdiag)
        LU = splu(
            L_scaled.asformat("csc"),
            diag_pivot_thresh=maxdiag,
            relax=1,
            panel_size=10,
            permc_spec=3,
        )
        entry = (LU, maxdiag, L_scaled)
        sig = _matrix_signature(L)
        _LU_CACHE[sig] = entry
        _LU_LAST_OBJ[0] = L
        _LU_LAST_OBJ[1] = entry

    tolerance_scale, _ = self._adaptTolerance(L_scaled, x, b_scaled)

    iteration = 0
    residual = float("inf")
    for iteration in range(min(self.iterations, 10)):
        residualVector, residual = self._residualVectorAndNorm(L_scaled, x, b_scaled)
        if residual <= self.tolerance * tolerance_scale:
            break
        xError = LU.solve(residualVector)
        x[:] = x - xError

    self._setConvergence(suite="scipy", code=0,
                         iterations=iteration + 1, residual=residual)
    self.convergence.warn()
    return x


# ============================================================
# Cache content-fingerprint helpers (2026-05-23)
# ============================================================
# Earlier versions of this patch keyed _BINARY_FULL_CACHE and _TERM_SOLVE_CACHE
# on `(id(self), id(var), float(dt))` alone — pure Python object identity.
# Cat 4 audit flag (2026-05-20): any user mutating boundary conditions,
# `var.constrain(...)`, or reassigning equation coefficients (`term.coeff =
# new_coeff`) between solves on the same eqn/var objects gets a cache hit
# returning the stale combined matrix → wrong solution. The benchmark only
# solves once per (eqn, var) so concordance never tripped.
#
# Fix: extend cache key with cheap content signals — attached constraint
# value/where fingerprints (including CellVariable.faceConstraints), BC
# value/face fingerprints, and a coefficient CONTENT fingerprint (catches both
# reassignment and in-place value mutation). Constraint/BC masks are O(N_faces) to hash, but they are tiny
# compared with sparse assembly / LU and are the state that determines whether
# the cached operator and RHS are still valid.
#
# Closed (2026-06-07): the coefficient is now fingerprinted by CONTENT via
# _array_state_sig (below), not object id, so in-place mutation of a
# coefficient array's values (e.g. `eqn.term.coeff.value *= 2` without
# reassigning `.coeff`) changes the key -> cache miss -> correct upstream
# recompute. For a constant scalar coeff (the target pattern) the fingerprint
# is a hash of a few bytes per solve, so the fast-path speedup is preserved.
def _array_state_sig(value):
    """Content fingerprint for numeric FiPy/numpy/scalar state."""
    try:
        arr = np.ascontiguousarray(numerix.asarray(value))
    except Exception:
        return ("object", type(value).__module__, type(value).__name__, id(value))

    if arr.dtype == object:
        return (
            "object-array",
            arr.shape,
            tuple(repr(x) for x in arr.ravel()),
        )

    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.dtype).encode("ascii", errors="ignore"))
    h.update(len(arr.shape).to_bytes(2, "little"))
    for dim in arr.shape:
        h.update(int(dim).to_bytes(8, "little", signed=False))
    h.update(arr.tobytes())
    return (arr.shape, str(arr.dtype), h.digest())


def _iter_constraints(var):
    seen: set[int] = set()
    for attr in ("constraints", "_constraints", "faceConstraints"):
        for constraint in getattr(var, attr, ()) or ():
            ident = id(constraint)
            if ident in seen:
                continue
            seen.add(ident)
            yield constraint


def _constraint_state_sig(var):
    return tuple(
        (
            type(constraint).__module__,
            type(constraint).__name__,
            _array_state_sig(getattr(constraint, "value", None)),
            _array_state_sig(getattr(constraint, "where", None)),
        )
        for constraint in _iter_constraints(var)
    )


def _boundary_conditions_sig(boundaryConditions):
    if not boundaryConditions:
        return ()
    return tuple(
        (
            type(bc).__module__,
            type(bc).__name__,
            _array_state_sig(getattr(bc, "value", None)),
            _array_state_sig(getattr(bc, "faces", None)),
        )
        for bc in boundaryConditions
    )


def _binary_state_sig(self, var, boundaryConditions):
    var_csig = _constraint_state_sig(var)
    bc_csig = _boundary_conditions_sig(boundaryConditions)
    term = getattr(self, "term", None)
    other = getattr(self, "other", None)
    # Content fingerprint (not id()) so in-place coefficient mutation
    # (`term.coeff.value *= 2`) invalidates the cache, not just reassignment.
    term_coeff_sig = _array_state_sig(getattr(term, "coeff", None)) if term is not None else 0
    other_coeff_sig = _array_state_sig(getattr(other, "coeff", None)) if other is not None else 0
    return (var_csig, bc_csig, term_coeff_sig, other_coeff_sig)


def _term_state_sig(self, var, boundaryConditions):
    var_csig = _constraint_state_sig(var)
    bc_csig = _boundary_conditions_sig(boundaryConditions)
    # Content fingerprint (not id()) so in-place coefficient mutation invalidates.
    # A composite/binary term's own `.coeff` does NOT reflect the operative
    # coefficients (e.g. for `TransientTerm() == DiffusionTerm(coeff)` the
    # DiffusionTerm coeff lives on self.other.coeff), so fold the sub-term
    # coefficient fingerprints into the top-level Term.solve cache key too —
    # otherwise an in-place coeff mutation hits the stale cached LU.
    self_coeff_sig = _array_state_sig(getattr(self, "coeff", None))
    sub_coeff_sig = tuple(
        _array_state_sig(getattr(sub, "coeff", None))
        for sub in (getattr(self, "term", None), getattr(self, "other", None))
        if sub is not None
    )
    return (var_csig, bc_csig, self_coeff_sig, sub_coeff_sig)


# ============================================================
# Layer 2: _BinaryTerm._buildAndAddMatrices — cache combined matrix + diff RHS
# ============================================================
# (id(self), id(var), dt, state_sig) -> (combined_matrix, b_diff,
#                             tmpMatrix_diff_cached, mass_per_dt, b_trans_const)
_BINARY_FULL_CACHE: dict[tuple, tuple] = {}


def fast_binary_buildAndAddMatrices(self, var, SparseMatrix, boundaryConditions=(),
                                    dt=None, transientGeomCoeff=None,
                                    diffusionGeomCoeff=None,
                                    buildExplicitIfOther=True):
    """Drop-in for ``_BinaryTerm._buildAndAddMatrices`` that caches the FULL
    combined matrix and the diffusion sub-term's RHS contribution.

    First call: run upstream's loop. After accumulation, copy the combined
    matrix into a freestanding wrapper (because ``self._sparsematrix`` is
    zeroed by the next call's ``_getMatrix``). Also snapshot the transient
    sub-term's ``coeffVectors`` so subsequent calls can rebuild b_trans
    directly from ``var.old.value`` — bypassing the trans-term's own
    ``_buildAndAddMatrices``.

    Subsequent calls: compute ``b_trans = var.old.value * mass_per_dt
    [+ b_trans_const]`` in one O(N) op, add to cached ``b_diff``, return
    the cached combined matrix. Skips zeroEntries + the two L.addAt calls
    + the diff face-to-cell scatter chain.
    """
    cache_key = (
        id(self), id(var), float(dt) if dt is not None else None,
        _binary_state_sig(self, var, boundaryConditions),
    )
    cached_full = _BINARY_FULL_CACHE.get(cache_key)

    if cached_full is not None:
        (combined_matrix, b_diff, _tmpMatrix_diff_cached,
         mass_per_dt, b_trans_const) = cached_full
        oldArray_val = numerix.asarray(var.old.value).ravel()
        b_trans = oldArray_val * mass_per_dt
        if b_trans_const is not None:
            b_trans = b_trans + b_trans_const
        RHSvector = b_diff + b_trans
        return (var, combined_matrix, RHSvector)

    # First call: full upstream path with intermediates captured for the cache.
    matrix = self._getMatrix(SparseMatrix=SparseMatrix, mesh=var.mesh, var=var)
    RHSvector = 0
    tmpMatrix_diff_for_cache = None
    tmpRHSvector_diff_for_cache = None

    for term in (self.term, self.other):
        is_diff = isinstance(term, _AbstractDiffusionTerm)
        tmpVar, tmpMatrix, tmpRHSvector = term._buildAndAddMatrices(
            var, SparseMatrix,
            boundaryConditions=boundaryConditions,
            dt=dt,
            transientGeomCoeff=transientGeomCoeff,
            diffusionGeomCoeff=diffusionGeomCoeff,
            buildExplicitIfOther=buildExplicitIfOther,
        )

        if is_diff:
            tmpMatrix_diff_for_cache = tmpMatrix
            tmpRHSvector_diff_for_cache = tmpRHSvector

        matrix += tmpMatrix
        RHSvector += tmpRHSvector

        term._buildCache(tmpMatrix, tmpRHSvector)

    # Snapshot the combined matrix into a free-standing wrapper so the next
    # call's `_getMatrix` (which zeroes `self._sparsematrix`) can't trash it.
    combined_matrix_snapshot = copy.copy(matrix)
    combined_matrix_snapshot.matrix = matrix.matrix.copy()

    # Capture trans-term coeffs so subsequent calls can compute
    # b_trans = oldArray.value * mass_per_dt + b_const directly.
    mass_per_dt = None
    b_trans_const = None
    for term in (self.term, self.other):
        if isinstance(term, _AbstractDiffusionTerm):
            continue
        try:
            coeffVectors = term._getCoeffVectors_(
                var=var,
                transientGeomCoeff=transientGeomCoeff,
                diffusionGeomCoeff=diffusionGeomCoeff,
            )
        except Exception:
            # Non-transient or unusual sub-term — leave caches empty so the
            # Term.solve layer falls back to the slow path.
            break
        mass_per_dt = numerix.asarray(coeffVectors['old value']).ravel() / float(dt)
        bvec = numerix.asarray(coeffVectors['b vector']).ravel()
        b_trans_const = bvec if numerix.any(bvec) else None
        break

    _BINARY_FULL_CACHE[cache_key] = (
        combined_matrix_snapshot,
        tmpRHSvector_diff_for_cache,
        tmpMatrix_diff_for_cache,
        mass_per_dt,
        b_trans_const,
    )

    return (var, matrix, RHSvector)


# ============================================================
# Layer 3: Term.solve — top-level bypass on cache hit (round 6 keep)
# ============================================================
# (id(self), id(var), dt) -> (LU, mass_per_dt_scaled, b_diff_scaled,
#                             b_const_scaled, var_shape, mass_scalar)
_TERM_SOLVE_CACHE: dict[tuple, tuple] = {}


def fast_term_solve(self, var=None, solver=None, boundaryConditions=(), dt=None):
    """Drop-in for ``Term.solve`` that short-circuits the upstream machinery
    on cache hit. First call runs upstream (populates ``_BINARY_FULL_CACHE``
    and ``_LU_CACHE``); afterwards on hit:

        b_scaled = var.value * mass_per_dt_scaled + b_diff_scaled
        x = LU.solve(b_scaled)
        var.value = x.reshape(var.shape)
    """
    # Delegate guard: a user-supplied solver is never reflected in the cache key
    # nor honored on the cache-hit path (which always reuses the cached direct-LU
    # factor), so any non-default solver must fall back to upstream rather than
    # be silently ignored. The default (solver=None) fast path is untouched.
    if solver is not None:
        return _orig_term_solve(self, var=var, solver=solver,
                                boundaryConditions=boundaryConditions, dt=dt)
    cache_key = (
        id(self), id(var), float(dt) if dt is not None else None,
        _term_state_sig(self, var, boundaryConditions),
    )
    cached = _TERM_SOLVE_CACHE.get(cache_key)

    if cached is not None:
        LU, mass_per_dt_scaled, b_diff_scaled, b_const_scaled, var_shape, mass_scalar = cached
        # Upstream `CellTerm._buildMatrix` forms the transient RHS from
        # `var.old.value` (the previous time-step's solution), not the
        # current `var.value`. Earlier versions of this patch used
        # `var.value` and only matched on benchmark loops where the two
        # happen to coincide — Cat 2 output-contract gap audited 2026-05-20,
        # fixed 2026-05-23.
        var_old = var.old if hasattr(var, "old") else var
        oldArray_val = numerix.asarray(var_old.value).ravel()
        # Scalar-mass fast path (round 8): when mass_per_dt is uniform (true
        # for regular Grid2D + default coeffs), use scalar multiply — half
        # the memory traffic of an N-vector multiply.
        if mass_scalar is not None:
            b_scaled = oldArray_val * mass_scalar
        else:
            b_scaled = oldArray_val * mass_per_dt_scaled
        b_scaled = b_scaled + b_diff_scaled
        if b_const_scaled is not None:
            b_scaled = b_scaled + b_const_scaled
        x = LU.solve(b_scaled)
        var.value = x.reshape(var_shape)
        return

    # First call: full upstream path — populates _BINARY_FULL_CACHE + _LU_CACHE.
    _orig_term_solve(self, var=var, solver=solver,
                     boundaryConditions=boundaryConditions, dt=dt)

    # Pull state from the two lower-level caches. Binary cache uses the
    # SAME state_sig — boundaryConditions passes through to its
    # _buildAndAddMatrices, so the composition is identical here.
    binary_key = (
        id(self), id(var), float(dt) if dt is not None else None,
        _binary_state_sig(self, var, boundaryConditions),
    )
    binary_entry = _BINARY_FULL_CACHE.get(binary_key)
    if binary_entry is None:
        return  # Unsupported equation shape — stay on slow path next call too.
    (_combined_matrix, b_diff, _tmpMatrix_diff_cached,
     mass_per_dt, b_trans_const) = binary_entry

    if _LU_LAST_OBJ[1] is None or mass_per_dt is None or b_diff is None:
        return
    LU, maxdiag, _L_scaled = _LU_LAST_OBJ[1]

    inv_maxdiag = 1.0 / maxdiag
    mass_per_dt_scaled = mass_per_dt * inv_maxdiag
    b_diff_scaled = numerix.asarray(b_diff).ravel() * inv_maxdiag
    b_const_scaled = (
        b_trans_const * inv_maxdiag if b_trans_const is not None else None
    )

    # Scalar-mass detection: collapse N-vector multiply to a scalar when uniform.
    if (mass_per_dt_scaled.ndim == 1 and mass_per_dt_scaled.size > 0
            and np.all(mass_per_dt_scaled == mass_per_dt_scaled[0])):
        mass_scalar = float(mass_per_dt_scaled[0])
    else:
        mass_scalar = None

    _TERM_SOLVE_CACHE[cache_key] = (
        LU, mass_per_dt_scaled, b_diff_scaled, b_const_scaled,
        var.shape, mass_scalar,
    )


# ============================================================
# Smoke recipe — fair comparison: load builds mesh / variable / equation
# (user-side prep, untimed); call runs the timestep loop calling eq.solve()
# repeatedly (= the user's natural loop; each solve goes through the patched
# Term.solve). Reference matches reference.py exactly.
# ============================================================

def _smoke_load(task_dir, tier):
    """User-side prep: read task.yaml + the tier's heat2d spec, build the
    Grid2D mesh, CellVariable, and TransientTerm == DiffusionTerm equation.
    NONE of this is patch-accelerated — the patch targets Term.solve, so only
    the time-stepping loop goes in the timed window. Mirrors reference.py's
    pre-loop block exactly.
    """
    import json
    import yaml

    from fipy import Grid2D, CellVariable, TransientTerm, DiffusionTerm

    with open(os.path.join(task_dir, "task.yaml"), encoding="utf-8") as f:
        task = yaml.safe_load(f)
    ds = next(d for d in task["datasets"] if d["tier"] == tier)
    data_path = ds["path"]
    if not os.path.isabs(data_path):
        data_path = os.path.normpath(os.path.join(task_dir, data_path))

    with open(data_path, encoding="utf-8") as f:
        spec = json.load(f)

    nx, ny = spec["nx"], spec["ny"]
    n_steps = spec["n_steps"]
    L = spec["L"]
    dx = L / nx
    dy = L / ny
    dt = 0.5 * dx ** 2
    coeff = spec["diffusion_coeff"]

    mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
    v = CellVariable(mesh=mesh, value=spec["init_value"])
    v.constrain(spec["bc_left"], mesh.facesLeft)
    v.constrain(spec["bc_right"], mesh.facesRight)
    eqn = TransientTerm() == DiffusionTerm(coeff=coeff)

    return {"var": v, "eq": eqn, "dt": dt, "n_steps": n_steps,
            "nx": nx, "ny": ny}


def _smoke_call(inputs):
    """The timed region: only the timestep loop calling eq.solve() — exactly
    the user's natural call pattern, each iteration going through the patched
    Term.solve. Mirrors reference.py's perf_counter window.
    """
    eq = inputs["eq"]
    var = inputs["var"]
    dt = inputs["dt"]
    n_steps = inputs["n_steps"]
    for _ in range(n_steps):
        eq.solve(var=var, dt=dt)
    return {"var": var, "nx": inputs["nx"], "ny": inputs["ny"],
            "n_steps": n_steps, "dt": dt}


def _smoke_save(result, dir, **kwargs):
    """Write the npz the task's evaluate.py reads: solution shaped (ny, nx)."""
    var = result["var"]
    nx, ny = result["nx"], result["ny"]
    solution = np.asarray(var.value, dtype=np.float64).reshape(ny, nx)
    np.savez(
        os.path.join(dir, "solution.npz"),
        solution=solution,
        nx=nx,
        ny=ny,
        n_steps=result["n_steps"],
        dt=result["dt"],
    )


register_patch(
    name="fipy",
    targets=[
        # Top-level user-facing entry — the round-6 keep target. Patching here
        # means each `eqn.solve(var=v, dt=dt)` call short-circuits to the
        # cached-LU + cached-RHS path on hit. The two layers below are the
        # cache-population path that runs on the first call.
        ("fipy.terms.term.Term", "solve", fast_term_solve),
        # Layer 2: cache combined matrix + diff RHS at the binary-term level.
        ("fipy.terms.binaryTerm._BinaryTerm", "_buildAndAddMatrices",
         fast_binary_buildAndAddMatrices),
        # Layer 1: content-keyed splu cache.
        ("fipy.solvers.scipy.linearLUSolver.LinearLUSolver", "_solve_",
         fast_solve_),
    ],
    smoke={"load": _smoke_load, "call": _smoke_call, "save": _smoke_save},
    tested_against="fipy 4.0.2+4.gb847e2c",
    tested_upstream_versions={"fipy": ["4.0.2+4.gb847e2c"]},
)
