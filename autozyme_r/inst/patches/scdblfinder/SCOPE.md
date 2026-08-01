# autozyme `scdblfinder` (R) — supported parameter scope

This preparatory patch accelerates the current scDblFinder default workflow.
It is release-locked to scDblFinder 1.27.6 at upstream commit
`2d0e5e4760e3d6d7525ec9a215fbeaab51915e36`.

## `scDblFinder::scDblFinder`

- **In-scope output equivalence:** Fixed-seed public annotations are exact to
  the task's declared numeric tolerances; class, selected features, assays,
  metadata, dimensions, names, and the canonical non-delegated full returned
  `SingleCellExperiment` contract are exact.
- **Validated call:** `scDblFinder(sce, verbose = FALSE, BPPARAM =
  BiocParallel::SerialParam(progressbar = FALSE))`. Matched 4- and 8-worker
  `MulticoreParam` calls on non-Windows are admitted for the scaling contract.
- **Supported input:** A `SingleCellExperiment` with 1–33,000 cells and an
  exact `dgCMatrix` `counts` assay with finite, NA-free stored values and
  positive library sizes. All other public parameters must remain at their
  upstream defaults.
- **Fast path:** Four coordinated namespace targets are active only inside the
  supported public call: eager sparse normalization rewrapped as a delayed
  matrix before unchanged PCA when expanded inputs stay within the package
  RSS-safe boundary, exact full-row feature-selection subset bypass, incremental
  KNN ratio-prefix writes, and skipping sparse NA replacement after proving the
  stored sparse values contain no NA. The package release keeps upstream's
  public-driver `gc()` behavior.
- **Normalization boundary:** The eager normalization path is used only when
  expanded real+artificial columns are at most 35,000 and size factors are
  positive and finite. Expanded inputs above 35,000 retain upstream
  normalization/PCA behavior for RSS neutrality and carry no normalization
  speed claim.
- **Release guard:** The package version must be exactly 1.27.6 and both the
  body and formal-list SHA-256 hashes of all four targets must match the pinned
  source. The public-driver feature-selection call-site transform must match
  exactly once. Any mismatch fails closed.
- **Fallback:** Direct calls to `.defaultProcessing`, `.evaluateKNN`, or
  `cxds2`; dense/delayed/unsupported sparse inputs; NA values; non-positive
  library sizes or size factors; non-default public arguments; unsupported
  thread backends/counts; inputs above 33,000 cells; source/version drift;
  `lgCMatrix` and every other sparse representation; `autozyme::with_disabled()`;
  and runtime fast-path errors all call the captured upstream implementation.
  Error-triggered retries restore `.Random.seed` before calling upstream.
  Unsupported paths receive no speed claim.

Large-tier RSS must be neutral or improved before publishing a speed claim. The
Campbell large tier and the Mair RNA-only 29,033-cell tier exceed the 35,000
expanded-column eager-normalization boundary and are retained as exact
safety/memory evidence rather than headline normalization speed results.
