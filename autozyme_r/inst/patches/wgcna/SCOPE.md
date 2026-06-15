# autozyme `wgcna` (R) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `WGCNA::blockwiseModules`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `WGCNA::blockwiseModules(<data: cells x HVG-genes log-normalized matrix, e.g. 5000x2000 .. 25000x5500>, power=4L, TOMType="signed", networkType="unsigned", maxBlockSize=6000L, minModuleSize=20L, mergeCutHeight=0.20, deepSplit=2L, numericLabels=TRUE, saveTOMs=FALSE, randomSeed=54321L, nThreads=0, verbose=0)  [corType not passed -> default "pearson"; TOMDenom not passed -> default "min"]`
- **Supported scope:** The fast path is correct only for WGCNA's "common case" as explicitly gated in .fast_tom_kernel_dispatch (patch.R lines 359-367): corType="pearson" (CcorType==0), networkType="unsigned" (CnetworkType==0), TOMType="signed" (CTOMType==2), TOMDenom="min" (TOMDenomC==0), no observation weights (weights NULL), cosineCorrelation FALSE, replaceMissingAdjacencies FALSE, suppressTOMForZeroAdjacencies FALSE, suppressNegativeTOM FALSE, useInternalMatrixAlgebra FALSE, and no NAs in the per-block expression submatrix (!anyNA(selExpr)). When ALL those hold the per-block TOM is computed via matrixStats column z-score + BLAS crossprod (Apple Accelerate on macOS, dynamic BLAS on Windows, forked-chunk crossprod on other Unix, direct crossprod fallback) and is claimed bit-perfect vs WGCNA's C kernel. For ANY other combination the dispatch falls through to the original .Call("tomSimilarity_call", PACKAGE="WGCNA"), so non-common-case TOM is handled correctly by upstream. The other three namespace overrides are independently guarded: fast_moduleEigengenes defers to the original when zyme=FALSE and reimplements the upstream eigengene pipeline (irlba truncated SVD, matrixStats row-scale) for arbitrary colors/nPC/align/impute/subHubs; fast_goodSamplesGenes short-circuits to all-TRUE only after verifying no weights, no NAs, and all-finite nonzero column variances, otherwise defers to upstream; fast_collectGarbage is an unconditional no-op. blockwiseModules itself is body-patched (dead scale() skip + TOM .Call redirection) with a guarded fallback to the unmodified original if either string substitution fails to match (e.g. upstream version drift); tested_against WGCNA 1.74. The benchmarked params (power=4, signed TOM, unsigned network, pearson, min denom, no weights, clean HVG matrix) sit squarely inside the common-case gate.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

