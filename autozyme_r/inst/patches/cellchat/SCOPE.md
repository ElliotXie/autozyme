# autozyme `cellchat` (R) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `CellChat::computeCommunProb`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `computeCommunProb(object = <data>, type = "triMean", raw.use = TRUE, population.size = FALSE, nboot = 100L, seed.use = 1L)  # all other args left at signature defaults: trim=0.1, LR.use=NULL, distance.use=TRUE, interaction.range=250, scale.distance=0.01, k.min=10, contact.dependent=TRUE, contact.range=NULL, contact.knn.k=NULL, contact.dependent.forced=FALSE, do.symmetric=TRUE, Kh=0.5, n=1`
- **Supported scope:** Fast native path handles datatype="RNA" CellChat objects (the only branch the pipeline exercises). type="triMean" gets the full Rcpp acceleration: cpp_aggregate_triMean for the per-group average, batched cpp_aggregate_triMean_boot for the nboot permutation tensor, cpp_outer_Pnull for the Prob outer product, and cpp_unified_inner for the per-LR/per-bootstrap Hill-product p-values. Non-triMean types (truncatedMean, thresholdedMean, median) are also handled correctly but only partly accelerated: the aggregator and per-bootstrap aggregation fall back to R-level stats::aggregate (lines 230-234, 323-330) while the outer/inner kernels still run. raw.use TRUE/FALSE both supported (data.signaling vs data.smooth, lines 158-162). nboot and seed.use are honored (set.seed(seed.use); replicate(nboot,...) lines 314-315). LR.use=NULL and explicit LR.use both supported (lines 163-175). Kh and n flow into the Hill kernels. Per-LR simple-gene and complex-subunit (geometric-mean) ligand/receptor expansion plus coreceptor/agonist/antagonist cofactors are precomputed once off the inner loop. Output prob/pval reported bit-identical (pearson 1.0, max_abs_diff 0.0) vs upstream at the benchmarked config.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

