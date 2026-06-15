# autozyme `cellphonedb` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `cellphonedb.src.core.methods.cpdb_statistical_analysis_method::call`

- **In-scope output equivalence:** tolerance
- **Validated at:** `cpdb_statistical_analysis_method.call(cpdb_file_path=<data: cellphonedb_v5.0.0.zip>, meta_file_path=<data: meta.tsv>, counts_file_path=<data: counts.tsv>, counts_data="ensembl", output_path=<tmp>, iterations=1000, threshold=0.1, threads=4, result_precision=3, pvalue=0.05, separator="|", output_suffix="ref"/"pipe"/"smoke", score_interactions=False)`
- **Supported scope:** Correct for the statistical analysis method run with counts_data in {ensembl, gene_name, hgnc_symbol} (column-name driven), upstream-default-style args: any threshold (flows into fast_percent_analysis and fast_build_clusters), any separator, any iterations (>= ~50 for the batched matmul to pay off; BATCH=50 hardcoded), any result_precision/pvalue (applied downstream of the patched helpers, unchanged), simple+complex interactions (complex handled via min-over-protein-rows), and any threads value (silently ignored by the fast path, result-identical). The patch rewrites 8 internal cpdb_statistical_analysis_helper functions + 1 no-op file_utils.save_dfs_as_tsv, all activated together as one coupled unit. Outputs are NOT bit-exact vs upstream: means are bit-exact, but np.random.shuffle yields a different permutation sequence than upstream's Categorical-setitem shuffle, so p-values differ (spearman ~0.99, Jaccard ~0.99 — within the task's noise floor, not 1.0).
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

