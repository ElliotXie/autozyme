# autozyme `statsmodels` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `statsmodels.genmod.generalized_linear_model.GLM.fit`

- **In-scope output equivalence:** tolerance
- **Validated at:** `sm.GLM(<data y>, <data X>, family=sm.families.Poisson()).fit()   # X is asfortranarray float64 from glm_poisson_*.npz; fit() called with ALL defaults: start_params=None, maxiter=100, method='IRLS', tol=1e-8, scale=None, cov_type='nonrobust'. Poisson uses its default Log link, default unit freq_weights/var_weights, no offset/exposure.`
- **Supported scope:** Fast Poisson/log-link IRLS path is gated by _can_fast_poisson_irls (__init__.py:135-173) and activates ONLY for: family is exactly sm.families.Poisson with sm.families.links.Log link (default); method='IRLS'; scale is None; cov_type='nonrobust'; cov_kwds is None; kwargs attach_wls=False, wls_method='lstsq', tol_criterion='deviance', rtol in (0,0.0,None); _offset_exposure all-zero (no offset/exposure); freq_weights, var_weights, iweights, n_trials all all-ones (unit weights, no binomial trials); start_params either None or shape[0]==exog.shape[1]; and design matrix is FULL RANK (implicit — fast_minimal_wls_fit at :285-287 uses np.linalg.solve(wexog.T@wexog, wexog.T@wendog) normal equations, and fast_glm_initialize at :210-219 sets df_model=p-1 / df_resid=n-p directly, skipping the matrix_rank SVD). Convergence uses abs(dev[i-1]-dev[i])<=atol (atol=tol), which is mathematically identical to upstream _check_convergence's np.allclose(...,rtol=0) on the deviance criterion. fast_handle_constant (:179-200) assumes the first all-ones finite column (as produced by sm.add_constant) is the intercept. For the benchmarked default Poisson fit() the produced params/llf/scale/converged/n_iter match upstream within max_abs/rel_diff 1e-6 and rel_diff_llf 1e-8 (task.yaml metrics).
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

