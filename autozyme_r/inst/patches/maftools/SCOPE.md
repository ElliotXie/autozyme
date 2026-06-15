# autozyme `maftools` (R) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `maftools::read.maf`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `read.maf(maf = <data>, verbose = FALSE)  where <data> is a .maf.gz file path (laml_{small,medium,large}.maf.gz dev tiers; ood_large/ood_xlarge inject ~10% CNV records). All other args at upstream defaults: clinicalData=NULL, rmFlags=FALSE, removeDuplicatedVariants=TRUE, useAll=TRUE, gistic*=NULL, cnLevel='all', cnTable=NULL, isTCGA=FALSE, vc_nonSyn=NULL.`
- **Supported scope:** The fast path (fast_read.maf -> fast_validateMaf + fast_summarizeMaf) handles a single MAF input that is EITHER a file path (.maf or .maf.gz, tab-separated, with a Hugo_Symbol header) OR an in-memory data.frame/data.table, when zyme=TRUE AND gisticAllLesionsFile is NULL AND cnTable is NULL AND isTCGA=FALSE AND useAll=TRUE. Within that envelope it correctly supports: removeDuplicatedVariants TRUE/FALSE (forwarded to validateMaf multi-key duplicated); rmFlags FALSE/TRUE/numeric (FLAG-gene removal at lines 392-402); custom vc_nonSyn (non-synonymous override at 387-391); clinicalData as NULL / data.frame / file path (handled in summarizeMaf 222-237); presence or absence of CNV variants (has_cnv branch 187-199 and Amp/Del + CNV column handling 142-170, exercised by the ood tiers); blank and NA Hugo_Symbol -> 'Unknown' (293-309); single-sample and (via nrow==0 guard) zero-variant inputs. All summary statistics (uniqueN, tabulate, colMeans/median rounded to 3, Rcpp zyme_fill_dcast integer fill) are exact equivalences to upstream; the task reports bit-exact concordance (vps_max_rel_diff=0, all *_match=1) across all 5 tiers x 2 platforms.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

