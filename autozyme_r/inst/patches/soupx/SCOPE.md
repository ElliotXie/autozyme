# SoupX Patch Scope

- **Validated call:** `SoupX::adjustCounts(sc, verbose = 0)`.
- **Fast path:** `sc` is a `SoupChannel`, `sc$toc` is a `dgCMatrix`, `method = "subtraction"`, `roundToInt = FALSE`, default `tol`/`pCut`, no extra `...`, and `clusters` is omitted so SoupX derives clusters from `sc$metaData$clusters` or uses no-cluster cell-level correction when that column is absent.
- **Fallbacks:** explicit `clusters`, non-`dgCMatrix` `toc`, `method != "subtraction"`, `roundToInt = TRUE`, non-default `tol`/`pCut`, extra `...`, non-positive or non-finite cell weights, invalid soup profiles, and `zyme = FALSE` delegate to upstream SoupX.
- **Native kernels:** `src/soupx.cpp` implements numerically equivalent bounded water-filling allocation over sparse CSC slots for clustered and no-cluster subtraction.
