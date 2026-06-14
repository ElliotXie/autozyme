"""Wave-4 wrapper/dispatch-line tests for autozyme.fipy.

Wave-1 (`_unit`) covered the fingerprint helpers; wave-2 (`_e2e`) covered the
transient-loop cache-hit parity through Term.solve. The COVERAGE-VISIBLE lines
still missing are the lower-level dispatch/guard branches that the standard
loop does not reach because the top Term.solve layer short-circuits first, plus
the smoke recipe:

  - line 136: the ``_LU_LAST_OBJ[0] is L`` identity fast-path in ``fast_solve_``.
  - lines 209-210: ``_array_state_sig`` object fallback when numerix.asarray
    raises.
  - lines 329-336: the cache-HIT branch of ``fast_binary_buildAndAddMatrices``
    (reached when Term.solve keeps missing but the binary layer hits).
  - lines 375 / 382-385: the trans-coeff capture loop's diffusion-term
    ``continue`` and the ``_getCoeffVectors_`` failure ``break``.
  - line 467: ``fast_term_solve``'s early-return when the LU/mass/b_diff state
    is unavailable.
  - the smoke recipe ``_smoke_load`` / ``_smoke_call`` / ``_smoke_save``
    (lines 504-557).
"""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("fipy")
np = pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")

import autozyme
from autozyme import fipy as azfipy


def _reset_caches():
    azfipy._LU_CACHE.clear()
    azfipy._BINARY_FULL_CACHE.clear()
    azfipy._TERM_SOLVE_CACHE.clear()
    azfipy._LU_LAST_OBJ[0] = None
    azfipy._LU_LAST_OBJ[1] = None


# --------------------------------------------------------------------------
# line 209-210: _array_state_sig object fallback
# --------------------------------------------------------------------------
def test_array_state_sig_unarrayable_object_fallback():
    """A value that numerix.asarray can't turn into an array falls through to
    the ('object', module, name, id) tuple branch (lines 209-210)."""

    class _Weird:
        def __array__(self, *a, **k):
            raise RuntimeError("not arrayable")

    w = _Weird()
    sig = azfipy._array_state_sig(w)
    assert sig[0] == "object"
    assert sig[3] == id(w)
    # Stable for the same object.
    assert azfipy._array_state_sig(w) == sig


# --------------------------------------------------------------------------
# line 136: fast_solve_ identity fast-path on back-to-back same-L solves
# --------------------------------------------------------------------------
def test_fast_solve_identity_fastpath_same_matrix_object():
    """Solving the same equation object twice in a row routes the 2nd solve's
    LinearLUSolver._solve_ through the ``_LU_LAST_OBJ[0] is L`` identity branch
    (line 136). We assert the loop result matches vanilla regardless."""
    import fipy as fp

    autozyme.activate("fipy")
    _reset_caches()

    def run(disabled):
        mesh = fp.Grid1D(nx=16)
        var = fp.CellVariable(name="phi", mesh=mesh, value=0.0)
        var.constrain(1.0, mesh.facesLeft)
        var.constrain(0.0, mesh.facesRight)
        eqn = fp.TransientTerm() == fp.DiffusionTerm(coeff=1.0)
        ctx = autozyme.disabled() if disabled else _null()
        with ctx:
            for _ in range(4):
                eqn.solve(var=var, dt=0.5)
        return np.array(var.value)

    fast = run(disabled=False)
    ref = run(disabled=True)
    np.testing.assert_allclose(fast, ref, rtol=1e-6, atol=1e-9)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _vanilla_loop(make_eqn, n=4, dt=0.5, nx=18):
    import fipy as fp

    with autozyme.disabled():
        mesh = fp.Grid1D(nx=nx)
        var = fp.CellVariable(name="phi", mesh=mesh, value=0.0)
        var.constrain(1.0, mesh.facesLeft)
        var.constrain(0.0, mesh.facesRight)
        eqn = make_eqn()
        out = []
        for _ in range(n):
            eqn.solve(var=var, dt=dt)
            out.append(np.array(var.value))
    return out


# --------------------------------------------------------------------------
# line 136 (LU identity fast-path) + binary cache-HIT branch (329-335): keep
# _LU_LAST_OBJ across steps but clear _TERM_SOLVE_CACHE so each step re-runs the
# upstream path, which reuses the SAME cached combined matrix object (-> the
# `_LU_LAST_OBJ[0] is L` identity branch fires) and hits the binary cache.
# --------------------------------------------------------------------------
def test_lu_identity_fastpath_and_binary_cache_hit():
    import fipy as fp

    autozyme.activate("fipy")
    _reset_caches()

    mesh = fp.Grid1D(nx=18)
    var = fp.CellVariable(name="phi", mesh=mesh, value=0.0)
    var.constrain(1.0, mesh.facesLeft)
    var.constrain(0.0, mesh.facesRight)
    eqn = fp.TransientTerm() == fp.DiffusionTerm(coeff=1.0)

    snaps = []
    for _ in range(4):
        eqn.solve(var=var, dt=0.5)
        snaps.append(np.array(var.value))
        # Keep _LU_LAST_OBJ (-> identity fast-path on the reused matrix) but
        # drop the Term.solve cache so the binary layer carries the hit.
        azfipy._TERM_SOLVE_CACHE.clear()

    assert len(azfipy._BINARY_FULL_CACHE) >= 1
    ref = _vanilla_loop(lambda: fp.TransientTerm() == fp.DiffusionTerm(coeff=1.0))
    for i, (f, r) in enumerate(zip(snaps, ref)):
        np.testing.assert_allclose(f, r, rtol=1e-6, atol=1e-9,
                                   err_msg=f"identity/binary-hit step {i} drift")


# --------------------------------------------------------------------------
# line 375: the diffusion-term `continue` in the trans-coeff capture loop fires
# only when the DiffusionTerm is self.term (i.e. the equation is written
# reversed: DiffusionTerm() == TransientTerm()).
# --------------------------------------------------------------------------
def test_reversed_equation_hits_diffusion_continue():
    import fipy as fp

    autozyme.activate("fipy")
    _reset_caches()

    mesh = fp.Grid1D(nx=14)
    var = fp.CellVariable(name="phi", mesh=mesh, value=0.0)
    var.constrain(1.0, mesh.facesLeft)
    var.constrain(0.0, mesh.facesRight)
    eqn = fp.DiffusionTerm(coeff=1.0) == fp.TransientTerm()

    snaps = []
    for _ in range(3):
        eqn.solve(var=var, dt=0.5)
        snaps.append(np.array(var.value))

    ref = _vanilla_loop(
        lambda: fp.DiffusionTerm(coeff=1.0) == fp.TransientTerm(),
        n=3, nx=14,
    )
    for i, (f, r) in enumerate(zip(snaps, ref)):
        np.testing.assert_allclose(f, r, rtol=1e-6, atol=1e-9,
                                   err_msg=f"reversed-eqn step {i} drift")


# --------------------------------------------------------------------------
# lines 441 + 482: non-uniform mesh (varying dx -> non-uniform cell volumes) ->
# mass-per-dt is NOT uniform, so mass_scalar is None (482) and the cache-hit
# fast path uses the N-vector multiply (441) rather than a scalar.
# --------------------------------------------------------------------------
def test_nonuniform_mesh_vector_mass_path():
    import fipy as fp

    autozyme.activate("fipy")
    _reset_caches()

    dx = np.array([0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.1, 0.08, 0.06, 0.04])
    mesh = fp.Grid1D(dx=dx)
    var = fp.CellVariable(name="phi", mesh=mesh, value=0.0)
    var.constrain(1.0, mesh.facesLeft)
    var.constrain(0.0, mesh.facesRight)
    eqn = fp.TransientTerm() == fp.DiffusionTerm(coeff=1.0)

    snaps = []
    for _ in range(4):
        eqn.solve(var=var, dt=0.5)
        snaps.append(np.array(var.value))

    # The cached entry must have mass_scalar=None (non-uniform volumes).
    assert any(v[5] is None for v in azfipy._TERM_SOLVE_CACHE.values())

    with autozyme.disabled():
        mesh_v = fp.Grid1D(dx=dx)
        var_v = fp.CellVariable(name="phi", mesh=mesh_v, value=0.0)
        var_v.constrain(1.0, mesh_v.facesLeft)
        var_v.constrain(0.0, mesh_v.facesRight)
        eqn_v = fp.TransientTerm() == fp.DiffusionTerm(coeff=1.0)
        ref = []
        for _ in range(4):
            eqn_v.solve(var=var_v, dt=0.5)
            ref.append(np.array(var_v.value))

    for i, (f, r) in enumerate(zip(snaps, ref)):
        np.testing.assert_allclose(f, r, rtol=1e-6, atol=1e-9,
                                   err_msg=f"nonuniform-mass step {i} drift")


# --------------------------------------------------------------------------
# SUSPECTED BUG (documented, not fixed): a 3-term equation
# `TransientTerm() == DiffusionTerm() + source` makes self.other a NESTED
# _BinaryTerm (diffusion + source), so the diffusion sub-term is never the one
# captured -> tmpRHSvector_diff_for_cache stays None -> the binary cache-HIT
# branch crashes at `RHSvector = b_diff + b_trans` (None + float). The SAME
# equation solves fine with the patch disabled. This is outside the patch's
# documented target (DiffusionTerm(coeff=const), no source), but it should fall
# back gracefully rather than raise.
# --------------------------------------------------------------------------
@pytest.mark.xfail(reason="fast_binary_buildAndAddMatrices cache-hit assumes a "
                          "non-None diffusion RHS; a transient==diffusion+source "
                          "3-term equation crashes with None+float — suspected bug",
                   strict=True, raises=TypeError)
def test_source_term_equation_crashes_on_cache_hit():
    """Documents the suspected bug: with the patch active, a transient diffusion
    equation carrying a constant volumetric source crashes on the 2nd solve
    (cache hit), whereas the disabled path solves it fine."""
    import fipy as fp

    autozyme.activate("fipy")
    _reset_caches()

    mesh = fp.Grid1D(nx=12)
    var = fp.CellVariable(name="phi", mesh=mesh, value=0.0)
    var.constrain(1.0, mesh.facesLeft)
    var.constrain(0.0, mesh.facesRight)
    eqn = fp.TransientTerm() == fp.DiffusionTerm(coeff=1.0) + 5.0

    # Sanity: the disabled path handles this fine (proves it's a patch issue).
    with autozyme.disabled():
        mesh_v = fp.Grid1D(nx=12)
        var_v = fp.CellVariable(name="phi", mesh=mesh_v, value=0.0)
        var_v.constrain(1.0, mesh_v.facesLeft)
        var_v.constrain(0.0, mesh_v.facesRight)
        eqn_v = fp.TransientTerm() == fp.DiffusionTerm(coeff=1.0) + 5.0
        for _ in range(2):
            eqn_v.solve(var=var_v, dt=0.5)
        assert np.all(np.isfinite(var_v.value))

    # Patched path: 2nd solve hits the cache and raises TypeError (the xfail).
    for _ in range(2):
        eqn.solve(var=var, dt=0.5)


# --------------------------------------------------------------------------
# smoke recipe (504-557)
# --------------------------------------------------------------------------
def _make_task_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    spec = {
        "name": "heat2d_tiny",
        "nx": 16,
        "ny": 16,
        "n_steps": 8,
        "L": 1.0,
        "diffusion_coeff": 1.0,
        "bc_left": 0.0,
        "bc_right": 1.0,
        "init_value": 0.0,
    }
    (data_dir / "small.json").write_text(json.dumps(spec), encoding="utf-8")
    task = {"datasets": [{"tier": "small", "name": "heat2d_tiny",
                          "path": "./data/small.json"}]}
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    return str(tmp_path)


def test_smoke_load_builds_mesh_var_equation(tmp_path):
    """_smoke_load reads the json spec and builds the Grid2D var + equation
    (covers lines 504-533)."""
    from fipy import CellVariable

    task_dir = _make_task_dir(tmp_path)
    out = azfipy._smoke_load(task_dir, "small")
    assert isinstance(out["var"], CellVariable)
    assert out["nx"] == 16 and out["ny"] == 16
    assert out["n_steps"] == 8
    assert out["dt"] > 0


def test_smoke_call_and_save_roundtrip(tmp_path):
    """_smoke_call runs the timestep loop; _smoke_save writes the (ny, nx)
    solution npz (covers 542-557)."""
    autozyme.activate("fipy")
    _reset_caches()
    task_dir = _make_task_dir(tmp_path)
    inputs = azfipy._smoke_load(task_dir, "small")
    result = azfipy._smoke_call(inputs)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    azfipy._smoke_save(result, str(out_dir))
    saved = np.load(out_dir / "solution.npz")
    assert saved["solution"].shape == (16, 16)
    assert int(saved["n_steps"]) == 8
    # Diffusion from a left=0/right=1 BC pushes the solution toward a gradient;
    # values stay finite and bounded.
    assert np.all(np.isfinite(saved["solution"]))


def test_smoke_call_matches_vanilla(tmp_path):
    """The patched smoke loop solution matches a vanilla (disabled) run."""
    task_dir = _make_task_dir(tmp_path)
    autozyme.activate("fipy")
    _reset_caches()
    inputs = azfipy._smoke_load(task_dir, "small")
    patched = azfipy._smoke_call(inputs)
    patched_sol = np.asarray(patched["var"].value, dtype=np.float64)
    with autozyme.disabled():
        ref_inputs = azfipy._smoke_load(task_dir, "small")
        ref = azfipy._smoke_call(ref_inputs)
        ref_sol = np.asarray(ref["var"].value, dtype=np.float64)
    np.testing.assert_allclose(patched_sol, ref_sol, rtol=1e-6, atol=1e-9)
