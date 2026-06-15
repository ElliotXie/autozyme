# autozyme `decontx` (R) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `celda::decontX`

- **In-scope output equivalence:** tolerance
- **Validated at:** `celda::decontX(<data>, z = NULL, batch = NULL, maxIter = 500, delta = c(10,10), estimateDelta = TRUE, varGenes = 5000, dbscanEps = 1, seed = 12345, verbose = FALSE)`
- **Supported scope:** The patch overrides four celda namespace functions that decontX calls internally; the upstream decontX driver (argument parsing, the per-batch loop over .decontXoneBatch, the z=NULL vs user-z dispatch, the EM convergence test on max|Δθ|) is left fully intact, so the fast path runs for the SAME parameter space decontX itself accepts. Specifically: (1) decontXLogLik is restored to a passthrough of celda's original C++ LL (the header comment calling it a no-op is stale — it computes the real LL), so LL-based diagnostics stay correct. (2) .decontxInitializeZ reproduces celda's UMAP+dbscan(+kmeans fallback) initialization on a raw counts matrix or SingleCellExperiment, differing only by passing auto_threads(cap=8) to scater::calculateUMAP; it is reached only when z=NULL (celda skips init when the user supplies z), and respects varGenes, dbscanEps, estimateCellTypes, and seed. (3) calculateNativeMatrix is a sparse R reimplementation of celda's normp-weighted native-count formula, keeping res$decontXcounts a real output. (4) decontXEM is an RcppEigen + std::thread reimplementation of one EM iteration, math-equivalent to upstream and reported bit-exact at all tested tiers; it honors estimate_eta, estimate_delta, delta (2-element prior), pseudocount, theta, and per-cell counts. Parallelism engages only when nC >= 500 AND requested threads > 1 (otherwise an identical serial path runs), and estimate_delta delegates to MCMCprecision::fit_dirichlet exactly as upstream. Because the patch sits below the batch loop (decontXEM is invoked once per batch by celda), batch != NULL is handled correctly via upstream's preserved dispatch. Benchmarked only at the default config (delta=c(10,10), estimateDelta=TRUE, varGenes=5000, dbscanEps=1, maxIter=500, seed=12345) across tiny/small/medium/large + two OOD tiers; honest speedups ~1.1-1.5x at 1 thread, up to ~2.85x at 4 threads.
- **Out-of-scope behavior:** ⚠ **Documented approximation.** Correct for the validated configuration below; results may differ outside it and there is no automatic fall-back, so stay within the stated scope (or deactivate the patch).
- **Approximation details:** see above

