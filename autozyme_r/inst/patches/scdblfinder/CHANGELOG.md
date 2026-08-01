# scdblfinder Patch Changelog

## 2026-08-01

- Replaced the delayed PCA bottleneck with the numerically identical generic
  IRLBA algorithm (`fastpath = FALSE`) operating through a non-materialized
  sparse transpose operator, guarded to expanded matrices at or below 50,000
  columns.
- Added release-hash-guarded sparse Poisson resampling in `createDoublets()`;
  zero-count entries are skipped without advancing R's RNG, preserving both
  generated counts and `.Random.seed` exactly while avoiding a dense temporary.
- Revalidated exact outputs on all five tiers, including the 29,033-cell Mair
  near-cap tier, which falls back to upstream PCA above the 50,000-column
  expanded boundary.

- Added the release-locked `scdblfinder` R patch for the attested
  `scDblFinder::scDblFinder(sce, verbose = FALSE, BPPARAM =
  BiocParallel::SerialParam(progressbar = FALSE))` workflow from
  `test_scdblfinder`.
- Registered package smoke coverage that reads the task tier dataset, calls
  the upstream public API, and writes the same `scdblfinder_output.rds` shape
  consumed by `evaluate.R`.
- Scope remains limited to `dgCMatrix` count assays with 1-33,000 cells,
  positive library sizes, default public arguments, and pinned scDblFinder
  1.27.6 source/formal hashes. Unsupported inputs fall back to upstream.
- The package release keeps upstream's `gc()` behavior; the task-level
  post-doublet `gc()` call-site rewrite was not shipped because packaged body
  rewriting corrupted upstream replacement-call forms.
- Lifted the `34b1fc7` memory-balance call-site rewrite: when selected
  features already span all rows, the public driver calls `selFeatures(sce, ...)`
  instead of constructing a redundant full-row `sce[sel_features, ]` subset.
  The transform is exact-match and release-hash guarded.
- Narrowed eager sparse normalization to expanded matrices at or below 35,000
  columns after package large-tier RSS remained above its paired upstream
  baseline; larger expanded inputs retain upstream normalization/PCA behavior
  while the public-driver subset bypass and exact internal fast paths remain
  guarded.
- Memory caveat: the normalized sparse speed claim is publishable only when the
  large tier is exact and RSS is neutral or improved. Campbell large and the
  near-cap Mair tier are retained as exact safety evidence rather than
  normalization speed claims.
