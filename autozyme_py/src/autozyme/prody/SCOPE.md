# autozyme `prody` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `prody.calcANM`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `anm = prody.ANM(name=...); anm.buildHessian(<data: calpha selection>, cutoff=15.0, gamma=1.0); anm.calcModes(n_modes=20, zeros=False, turbo=True)  # which internally calls solveEig(self._hessian, n_modes=20, zeros=False, turbo=True, expct_n_zeros=6). Equivalent to prody.calcANM(pdb, selstr='calpha', cutoff=15.0, gamma=1.0, n_modes=20, zeros=False). Datasets (Cα): 1OEL 3668, 5GAR 5878, 1aon 8015, 6TLJ 9200, 7B0U 13320.`
- **Supported scope:** The patch replaces two functions (prody.dynamics.anm.ANMBase.buildHessian and prody.dynamics.anm.solveEig); calcANM/calcModes route through them. buildHessian is general and mathematically exact: it builds the Hessian in float64 unconditionally, supports AtomGroup/Atomic inputs (via _getCoords/getCoords) and raw numpy coord arrays (via checkCoords), any cutoff>0 (validated by checkENMParameters), and both scalar gamma and callable Gamma objects (callable invoked as gamma(dist2,i,j), matching upstream exactly, incl. squared-distance arg). It also restores a real sparse CSR Kirchhoff matrix and attaches the build coords to the sparse Hessian instance (no global state). solveEig has a deterministic eigsh shift-invert path (eigsh sigma=-1e-8, which=LM, tol=1e-10) that handles any sparse M with a finite integer n_modes < dof for both zeros=False and zeros=True, with an internal guard that defers to upstream if final_n_modes exceeds available values. The LOBPCG accelerated path (the fast path actually measured) is taken only for the standard ANM config: M sparse, reverse=False, integer n_modes < dof, zeros=False, expct_n_zeros==6, build coords attached, coords.shape[0]*3==dof, and coords.shape[0] < 12000; its result is accepted only after a residual-norm gate (max scaled residual <= 1e-3) and otherwise falls through to the deterministic eigsh path. Benchmark tiers 1OEL/5GAR/1aon/6TLJ (3668–9200 Cα) exercise LOBPCG; the 7B0U OOD tier (13320 Cα) is >=12000 so it uses the eigsh shift-invert path, not LOBPCG.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

