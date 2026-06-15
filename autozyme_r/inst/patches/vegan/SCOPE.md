# autozyme `vegan` (R) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `vegan::adonis2`

- **In-scope output equivalence:** tolerance
- **Validated at:** `adonis2(<data: Bray-Curtis dist of count matrix> ~ group + x, data = meta, permutations = 999, by = "terms") with set.seed(1234). Two-term model: group = 3-level factor, x = continuous covariate. method defaults to "bray". In the headline smoke benchmark the LHS is a PRECOMPUTED dist (vegdist Bray), so fast_adonis2's parallelDist distance-build branch is bypassed and only the dbRDA fit (fast_adonis0) + permutation trace engine (fast_permutest_cca) are timed. reference.R/run.R pass the raw count matrix counts as LHS (counts ~ group + x), which DOES exercise the parallelDist Bray swap.`
- **Supported scope:** The fast path correctly handles the by="terms", model="reduced" dbRDA (distance-based RDA) PERMANOVA path with NO conditioning/partial term (no Condition() / Z block) and NON-classical-CCA (no row weights). Concretely: (1) fast_adonis2 accelerates only method="bray" with no extra ... args and parallelDist installed; it accepts either a precomputed dist LHS or a count matrix/data.frame LHS (built into Bray via parallelDist::parDist). It still honors sqrt.dist, add=lingoes/cailliez, na.action, strata, and permutations by routing them through the original vegan code paths. (2) fast_adonis0 builds the dbRDA fit directly from the doubly-centered Gram + a single qr() only when there is no Z conditioning block and no extra dots. (3) fast_permutest_cca runs the bounded-batch trace engine only when model=="reduced" AND by=="terms" AND first==FALSE AND not partial (no pCCA) AND not classical CCA (no RW weights) AND inherits "dbrda" AND x$CCA$rank>0. Permutation count (default 999), strata, and seed are all respected; results are bit-identical to upstream for the supported branch (changelog reports F0 ~4e-8, F.perm ~8e-13 vs unpatched reference). Memory is bounded to O(batch_perms) via getOption("autozyme.vegan.perm_batch",128L).
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

