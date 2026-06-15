# autozyme `tradeseq` (R) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `tradeSeq::fitGAM`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `tradeSeq::fitGAM(counts=<data>, pseudotime=<data>, cellWeights=<data>, nknots=6, verbose=FALSE, parallel=FALSE, sce=TRUE). All other args left at upstream defaults: conditions=NULL, U=NULL, genes=seq_len(nrow(counts)), weights=NULL, offset=NULL, BPPARAM=BiocParallel::bpparam(), control=mgcv::gam.control(), family="nb", gcv=FALSE, aic=FALSE. nknots is read from input$nknots in reference.R; the OOD dataset generator (setup/prepare_ood_inputs.R:51,139) and upstream both default it to 6.`
- **Supported scope:** The fast path (fast_fitGAM_hoist_formula) is entered only when zyme=TRUE AND conditions is NULL. Within that it correctly handles the standard tradeSeq NB-GAM fitting workload: family="nb" with mgcv's log link, weights=NULL, offset with no dim (vector offset, the .get_offset default), sce=TRUE, aic=FALSE, single-lineage (ncol(pseudotime)==1, quantile-based knot placement with duplicate repair) AND multi-lineage (ncol(pseudotime)>=2, delegates knot placement to tradeSeq::.findKnots). Both single-lineage (dev tiers small/medium/large) and 2-lineage (OOD tiers ood_large/ood_xlarge, which force ncol==2 and the .findKnots branch) are benchmarked and pass bit-exact gates (beta/sigma/X/knot max_abs_diff <=1e-8..1e-12, converged_match=1.0). The hot-path acceleration (prefit_G reuse, hoisted formula template, fork-pool via .zyme_mclapply on macOS / PSOCK on Windows, and the mgcv::nb / gam.fit4 crossprod overrides) activates only when use_prefit_G is TRUE: is.null(weights) && is.null(dim(offset)) && family=="nb" && length(id)>0 (patch.R:335-336), and the fork/parallel compact path additionally requires !verbose && sce && !aic && worker_count>1 (patch.R:416). When use_prefit_G is FALSE it falls through to a serial/pbapply per-gene mgcv::gam() with the same formula. The scBLAS NB kernels (fast_dDeta and scblasR routing in fast_nb) are gated OFF by default via .az_feature_enabled('scblas_nb', default=FALSE); the default path uses the inline Rcpp nb_Dd_cpp/linkinv_log_cpp/nb_dev_resids_cpp kept bit-exact. The mgcv gam.fit4 crossprod rewrite is a textual substitution pinned to mgcv 1.9.4 (sentinel warns if patterns miss).
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

