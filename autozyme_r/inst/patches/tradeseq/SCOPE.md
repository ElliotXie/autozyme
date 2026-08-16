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

- **In-scope output equivalence:** tolerance
- **Validated at:** `tradeSeq::fitGAM(counts=<data>, pseudotime=<data>, cellWeights=<data>, nknots=6, verbose=FALSE, parallel=FALSE, sce=TRUE). All other args left at upstream defaults: conditions=NULL, U=NULL, genes=seq_len(nrow(counts)), weights=NULL, offset=NULL, BPPARAM=BiocParallel::bpparam(), control=mgcv::gam.control(), family="nb", gcv=FALSE, aic=FALSE. Concordance is scored vs mgcv::gam(family="nb") ground truth on the fitted linear predictor eta and the contrast standard errors.`
- **Supported scope:** The fast path (fast_fitGAM_v4) is entered only when zyme=TRUE AND conditions is NULL AND family="nb" AND sce=TRUE AND aic=FALSE AND weights=NULL AND offset has no dim (vector offset, the .get_offset default) AND U is NULL (default all-ones intercept) AND length(genes)>0. Within that envelope it handles the standard tradeSeq NB-GAM fitting workload, single-lineage (ncol(pseudotime)==1, quantile-based knot placement with the upstream duplicate->seq repair) AND multi-lineage (ncol(pseudotime)>=2, delegates knot placement to tradeSeq::.findKnots). The per-gene fit runs entirely in the native engine autozyme:::fastgam_fit (src/fastgam.cpp): the shared design X / penalty S / offset are built once via a single mgcv::gam(fit=FALSE) plus one real single-gene fit for the canonical lpmatrix / model frame / knot points; the design is reparameterized into its rank-identifiable subspace (SVD, general to 1..k lineages); each gene's penalized NB Newton + Laplace-REML selection of (lambda, theta) is solved independently in a gene-level OpenMP loop with analytic-gradient damped Newton and a Nelder-Mead fallback. No mgcv namespace is mutated; the two setup fits use stock mgcv so the stored X/dm/knotPoints metadata is byte-identical to upstream. Validated against mgcv::gam(family="nb") on bench_500s, real Paul/Nestorowa, a 1->4 lineage synthetic sweep and the 5 HF autozyme datasets. Thread count is a pure performance knob (each gene is an independent solve; results are thread-invariant).
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

