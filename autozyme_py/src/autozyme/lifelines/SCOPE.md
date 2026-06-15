# autozyme `lifelines` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `lifelines.CoxPHFitter.fit`

- **In-scope output equivalence:** tolerance
- **Validated at:** `CoxPHFitter().fit(<data>, duration_col="T", event_col="E")  — default constructor (penalizer=0.0, l1_ratio=0.0, strata=None, baseline_estimation_method="breslow", batch_mode=None), fit() with no weights_col / entry_col / cluster_col / formula / robust; data = cox_synth_*.parquet (plain float64 covariates x00..xNN + int T + int E), tiers small(1M×30)/medium(4M×50)/large(4M×90)/ood_large(1M×150)/ood_xlarge(8M×100), threads=1, lifelines 0.30.3`
- **Supported scope:** The patch replaces four SemiParametricPHFitter methods (the breslow/semi-parametric Cox path used by default CoxPHFitter). It is mathematically faithful to upstream for the default unweighted, unstratified, non-left-truncated Cox fit on a clean dense numeric DataFrame with the identity (default) formula. (1) fast_get_efron_values_batch / _efron_kernel_jit reproduces upstream _get_efron_values_batch exactly (Efron tied-time partial likelihood, per-observation weights, full d×d Hessian + gradient + log-lik every Newton step), so variance_matrix_/SE/p-values/CI inference stays valid. The kernel is ONLY invoked when lifelines' own _BatchVsSingle().decide() chooses 'batch' (or batch_mode=True); when 'single' is chosen the unpatched upstream _get_efron_values_single runs (correct). The benchmark datasets have many tied integer durations, forcing the batch path. (2) penalizer (L1/L2 elastic-net) and strata are applied OUTSIDE the kernel by the unpatched _newton_raphson_for_efron_model / _partition_by_strata_and_apply, so penalized and stratified fits remain correct (kernel just returns per-stratum h/g/ll). (3) fast_preprocess_dataframe stashes a contiguous float64 view and overrides X.mean/X.std with numpy equivalents using ddof=1 (matches pandas std). (4) safe_check_pre_fitting falls back to full upstream validation whenever weights_col or entry_col is set, any non-numeric dtype, any non-finite value, or any exception; only fully-clean numeric unweighted/non-truncated frames take the cheap finite-check path. (5) fast_predict_log_partial_hazard runs only post-fit (outside the timed window) and falls back to upstream for pandas Series input.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

