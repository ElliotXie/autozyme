# autozyme `clusterprofiler` (R) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `clusterProfiler::compareCluster`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `compareCluster(geneClusters = <data>, fun = "enrichGO", OrgDb = "org.Hs.eg.db" (org.Mm.eg.db for mouse OOD tiers), keyType = "ENTREZID", ont = "ALL", pvalueCutoff = 0.05, qvalueCutoff = 0.2, pAdjustMethod = "BH", minGSSize = 10, maxGSSize = 500)`
- **Supported scope:** Correctly accelerates the GO over-representation (ORA) workflow: compareCluster(geneClusters = <named list of character gene-ID vectors>, fun = "enrichGO", OrgDb, keyType, ont in {ALL,BP,CC,MF}, pvalueCutoff/qvalueCutoff/pAdjustMethod/minGSSize/maxGSSize at any value). Three coordinated overrides: (1) fast_get_GO_data caches the per-(organism, ont, keytype) PATHID2EXTID/EXTID2PATHID/PATHID2NAME/GO2ONT plus ZYME_ALLEXTID/ZYME_TERM_LENGTHS into the shared .Anno_clusterProfiler_Env, building all 4 ont entries from one mapIds + split; (2) fast_enricher_internal vectorizes phyper and uses cached extID/term-length when universe is NULL (the compareCluster default) — this is bound to BOTH clusterProfiler::enricher_internal and DOSE::enricher_internal, so it also correctly handles any other ORA fun (enrichKEGG/enrichDO/etc.) by falling back to .ALLEXTID_fn/lengths() when the GO cache keys are absent; (3) fast_compareCluster fans the independent per-cluster fun() calls across cores via .zyme_mclapply (lapply fallback on Windows or threads<=1), applies the exact pvalue<=cutoff & p.adjust<=cutoff then qvalue<=cutoff filtering that upstream get_enriched/as.data.frame applies, and rebuilds the compareClusterResult. fun may be any character name resolvable in the clusterProfiler namespace or a function object. universe handling is preserved: character universe intersects (or replaces, under options(enrichment_force_universe=TRUE)); non-character universe is ignored with a message, matching upstream. Concordance verified at term_jaccard=1, pearson_logp_shared=1, max_abs_logp_diff=0 for the benchmarked GO-ALL/ENTREZID config (human pbmc + mouse OOD tiers).
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

