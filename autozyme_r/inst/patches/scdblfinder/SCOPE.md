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
- **Fast path:** Five coordinated namespace targets are active only inside the
  supported public call: eager sparse normalization followed by the same
  generic IRLBA algorithm through a non-materialized transpose operator,
  sparse-only Poisson resampling for artificial doublets, exact full-row
  feature-selection subset
  bypass, incremental KNN ratio-prefix writes, and skipping sparse NA
  replacement after proving the stored sparse values contain no NA. The
  package release keeps upstream's public-driver `gc()` behavior.
- **Normalization boundary:** The direct sparse PCA path is used only when
  expanded real+artificial columns are at most 50,000 and size factors are
  positive and finite. Expanded inputs above 50,000 retain upstream
  normalization/PCA behavior for exactness and carry no PCA speed claim.
- **Release guard:** The package version must be exactly 1.27.6 and both the
  body and formal-list SHA-256 hashes of all five targets must match the pinned
  source. The public-driver feature-selection call-site transform must match
  exactly once. Any mismatch fails closed.
- **Fallback:** Direct calls to `.defaultProcessing`, `.evaluateKNN`,
  `createDoublets`, or `cxds2`; dense/delayed/unsupported sparse inputs; NA
  values; non-positive
  library sizes or size factors; non-default public arguments; unsupported
  thread backends/counts; inputs above 33,000 cells; source/version drift;
  `lgCMatrix` and every other sparse representation; `autozyme::with_disabled()`;
  and runtime fast-path errors all call the captured upstream implementation.
  Error-triggered retries restore `.Random.seed` before calling upstream.
  Unsupported paths receive no speed claim.

Large-tier RSS must be neutral or improved before publishing a speed claim. The
The Mair RNA-only 29,033-cell tier exceeds the 50,000 expanded-column boundary
and is retained as exact safety/memory evidence rather than a direct-PCA speed
result.
