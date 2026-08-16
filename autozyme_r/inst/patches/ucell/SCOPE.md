# autozyme `ucell` (R) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

## `UCell::ScoreSignatures_UCell`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `ScoreSignatures_UCell(matrix, features, maxRank=1500, BPPARAM=BiocParallel::SerialParam())` on the attested serial sparse path. `maxRank` and `ties.method='average'` stay at upstream defaults; `precalc.ranks=NULL`; `storeRanks` is upstream-only.
- **Supported scope:** The fast path (`ucell_fast_scores_dgC` in `src/ucell.cpp`) fires only when `matrix` is a finite `Matrix::dgCMatrix` with row names, `precalc.ranks` is NULL, `ties.method` is `"average"`, `BPPARAM` has at most one worker (or `ncores <= 1` when `BPPARAM` is NULL), and `maxRank` / `w_neg` are finite scalars. Finiteness is sample-checked on 20000 evenly-spaced nonzeros. Negative values are scored correctly under descending ranks and do not force a fallback.
- **Out-of-scope behavior:** SingleCellExperiment, dense matrix/data.frame, precomputed ranks, non-finite values, non-average ties, multi-worker BiocParallel, invalid scalar knobs, missing row names, or `zyme=FALSE` **fall back to the upstream implementation** (correct result, no speedup).
