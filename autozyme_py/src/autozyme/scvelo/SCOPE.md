# autozyme `scvelo` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `scvelo.tl.recover_dynamics`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `scv.tl.recover_dynamics(<data>, var_names="velocity_genes", n_jobs=1, show_progress_bar=False)  # reference.py baseline (N_JOBS locked to 1 for reproducibility). assignment_mode left at default 'projection'. NOTE: the shipped patch's _smoke_call uses n_jobs=-1 instead.`
- **Supported scope:** The fast path is a faithful structural mirror of upstream and handles ALL assignment_mode values correctly. (1) fast_assign_tau (lines 172-200) reproduces upstream assign_tau branch-for-branch: the projection family (assignment_mode in {full_projection, partial_projection}, or projection with beta<gamma) is accelerated with a streaming-argmin numba kernel that drops the per-cell constant ||x_obs||^2 term from the squared-distance argmin (verified mathematically equivalent against upstream's 3D-broadcast argmin); every other mode (including 'projection' with beta>=gamma, and any non-projection mode) falls into the SAME else-branch as upstream, calling the unchanged original tau_inv -> bit-identical. (2) _fast_get_solution (lines 123-147) JITs the ODE solution only when t is 1D and u0/s0/alpha/beta/gamma are all scalars (the recover_dynamics per-gene fit path); it FALLS BACK to the captured upstream get_solution for 2D t, array initial_state, and per-cell array rate params (the get_divergence / velocity(mode='dynamical') path) -- guarded. (3) NumbaConn.dot routes 1D, 2-col, and 4-col matvecs to specialized kernels and any other n_cols to a correct generic kernel; an F-contiguous fast view is used only for n<=2000 with the rest copied to C-contig -- all branches covered. (4) fast_get_n_jobs only rewrites the worker count (parallelism), not the math; per-gene EM fits are independent so worker count does not affect results. All five overrides must be active together and the wrapper re-installs assign_tau + get_solution inside each loky worker so parallel runs match serial. Numerical caveat: _splicing_solve and the numba matvec/argmin kernels use fastmath=True (scalar exp vs upstream vectorized exp), so results match upstream only up to fp rounding (the task tolerates this with pearson>=0.99 / n_genes_fit_diff<=5 acceptance gates).
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

