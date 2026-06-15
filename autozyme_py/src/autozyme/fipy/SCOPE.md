# autozyme `fipy` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `fipy.Term.solve (registered as fipy.terms.term.Term.solve, plus two helper layers: fipy.terms.binaryTerm._BinaryTerm._buildAndAddMatrices and fipy.solvers.scipy.linearLUSolver.LinearLUSolver._solve_)`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `eqn = TransientTerm() == DiffusionTerm(coeff=<const>); for _ in range(n_steps): eqn.solve(var=v, dt=dt) — where solver=None (default scipy LinearLUSolver), boundaryConditions=() (default; Dirichlet BCs applied via v.constrain on facesLeft/facesRight), dt = 0.5*dx**2 (constant across all steps), on a fixed Grid2D(nx,ny,dx,dy) unit-square mesh. Per tier (small n80, medium n160, large n240, ood_large 180x360 anisotropic, ood_xlarge n320): nx/ny/n_steps/L/diffusion_coeff/init_value/bc_left/bc_right from <data>.`
- **Supported scope:** The fast path is correct for the canonical FV time-stepping pattern it was built for: eqn = TransientTerm() == DiffusionTerm(coeff=const) solved repeatedly with eqn.solve(var=v, dt=dt) on a FIXED mesh, FIXED dt, and CONSTANT diffusion coefficient, using the default scipy LinearLUSolver (solver=None). Under these conditions the assembled matrix M/dt - L_diff and its LU factor are genuinely step-invariant; only the RHS depends on var.old.value, so the patch reuses the cached LU and cached diffusion-RHS and computes b = var.old.value*mass_per_dt_scaled + b_diff_scaled, then x = LU.solve(b). First call per (eqn,var,dt,state) goes through the upstream path (populating caches), so numerics are bit-exact on the cold call and at FP noise (~1e-15) on hits. The cache key (id(self), id(var), float(dt), state_sig) correctly forces a fresh upstream rebuild when: dt changes, the var object is swapped, attached constraint value/where masks change (incl. CellVariable.faceConstraints), explicit boundaryConditions value/faces change, or a coefficient OBJECT is reassigned (term.coeff = new_coeff) — state_sig fingerprints these via content hash or object id. Layer 1 (LinearLUSolver._solve_) content-keys splu by a blake2b hash of the CSR shape/data/indices, so it stays correct even for matrices it has not memoized (it falls back to a fresh splu on hash miss).
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

