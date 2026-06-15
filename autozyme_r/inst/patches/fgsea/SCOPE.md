# autozyme `fgsea` (R) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `fgsea::fgsea`

- **In-scope output equivalence:** tolerance
- **Validated at:** `fgsea(pathways = <data: MSigDB pathway list>, stats = <data: per-cluster ranked stats vector>, minSize = 15L, maxSize = 500L) — called once per cluster in a loop with set.seed(42L) before each call. (Patch smoke/run additionally sets eps = 9e-6 and BPPARAM = BiocParallel::SerialParam(); reference baseline leaves both at upstream default.)`
- **Supported scope:** Fast path covers the default fgsea() dispatch route: fgsea() with no `nperm` argument forwards to fgseaMultilevel, which is the only top-level function this patch overrides (along with the internal helpers preparePathwaysAndStats and calcGseaStat). Within fgseaMultilevel the fast path correctly handles: scoreType in {"std","pos","neg"} (branch lifted in both fast_calcGseaStat and the C++ calcEsLeBatchCpp / EsRuler sign handling); arbitrary gseaParam (re-applied as abs(stats)^gseaParam inside preparePathwaysAndStats before the C++ ES kernel, which therefore does not re-apply it); minSize/maxSize filtering (clamped minSize>=1, maxSize<=length(stats)-1); nPermSimple and sampleSize arbitrary (qbeta / multilevelError / trigamma lookup tables keyed and built per (nPermSimple, sampleSize)); eps arbitrary (clamped to [0,1]); the all-stats-zero edge case (NR==0 falls back to R-level fast_calcGseaStat with uniform 1/k increments). Pathway-prep result is cached across clusters and correctly invalidated when names(stats) length/endpoints or the pathways object address/endpoints change. Output is intended bit-exact for ES (no RNG) and seed-deterministic for NES/pval/padj/log2err. Multilevel C++ batch kernel uses per-group RNG seeded from a shared seed so results are dispatch-order-independent.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

