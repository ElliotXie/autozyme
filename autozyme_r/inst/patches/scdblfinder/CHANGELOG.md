# scdblfinder Patch Changelog

## 2026-08-01

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
- Memory caveat: normalized sparse tiers retain the speed claim but can carry
  a higher peak RSS than upstream, especially on large sparse inputs. The
  near-cap Mair tier is retained as exact safety evidence and not as a speed
  claim.
