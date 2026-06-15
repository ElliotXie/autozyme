# autozyme `squidpy_cooccurrence` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `squidpy.gr.co_occurrence`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `sq.gr.co_occurrence(<data>, cluster_key=<CLUSTER_KEY>, copy=True, show_progress_bar=False)  # plus n_jobs=N added only when ZYME_THREADS in {4,8}; all other algorithmic args (spatial_key="spatial", interval=50, n_splits=None auto, backend="loky") left at upstream default`
- **Supported scope:** The patch replaces the inner helper squidpy.gr._ppatterns._co_occurrence_helper (register_patch targets=[("squidpy.gr._ppatterns","_co_occurrence_helper", fast_co_occurrence_helper)]), so it is invoked for EVERY co_occurrence call regardless of public-API arguments. By binding below the public entry point it correctly supports: any cluster_key, any spatial_key, copy=True/False, any interval (int->linspace or array — upstream builds the float interval array before the helper is reached; the helper only sees interval as a numeric array and bisects it, so non-default interval sizes/values work), any n_splits (auto or explicit — the helper consumes whatever tile collection co_occurrence built), and any n_jobs/backend (squidpy.parallelize chunks idx_splits across joblib workers; the patched helper handles arbitrary sub-lists of triu pairs per chunk, and re-derives its own numba thread count at call time via auto_threads). Per-pair same_split symmetry and the divide-by-zero / zero-marginal NaN guards mirror upstream (the "if rs==0.0: continue" / "if m==0.0: continue" branches reproduce upstream's "np.sum==0 -> zeros" behavior). Two internal kernels: a fused 2D distance+bin+histogram path (spatial.shape[1]==2) and a runtime-dim nd fallback (spatial.shape[1]!=2). An all-tiles parallel-over-tiles kernel is used only when is_2d AND len(idx_splits) >= _ALL_TILES_THRESHOLD (default 2000, tunable via AUTOZYME_COOCCURRENCE_ALL_TILES env at import). Concordance verified pearson_occ=1.0 and q99_abs_diff_occ <= 2e-6 across all five tiers (small/medium/large/ood_large/ood_xlarge), thread 1/4/14, pass_rate=1.0. Tested against squidpy 1.6.5.
- **Out-of-scope behavior:** Handles the **full parameter signature**.

