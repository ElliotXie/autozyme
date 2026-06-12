# Contract: Seurat S3-method stragglers
#
# Five S3 methods patched as part of the seurat patch group that don't
# get their own per-API test files. Documented here so the coverage
# audit is complete:
#
#   - VST.dgCMatrix
#       Sparse dgCMatrix vst dispatch -- called INTERNALLY by
#       FindVariableFeatures.{Seurat,StdAssay}. Implicitly exercised
#       through test-contract-FindVariableFeatures.R + SCTransform.R.
#
#   - FindVariableFeatures.StdAssay
#       Seurat 5's Assay5/StdAssay dispatch path. CreateSeuratObject in
#       Seurat 5 produces Assay5 by default, so test-contract-
#       FindVariableFeatures.R already hits this S3 method (not the
#       legacy .Seurat path). Confirm by reading the class -- our
#       .make_tiny_seurat returns a v5 object.
#
#   - RunPCA.StdAssay
#       Same story as FindVariableFeatures.StdAssay. test-contract-
#       RunPCA.R exercises this on the v5 fixture.
#
#   - FindWeights
#       Integration helper called from IntegrateData / IntegrateLayers.
#       Needs a full FindIntegrationAnchors workflow output, which is
#       itself deferred in test-contract-FindIntegrationAnchors.R for
#       lack of a biology-shaped cross-batch fixture. Deferred.
#
#   - CCAIntegration
#       The v5 layer-integration wrapper. Same fixture requirement as
#       FindWeights -- deferred until a small cross-batch fixture lands.

test_that(".make_tiny_seurat produces a v5 Assay5/StdAssay object", {
  .skip_if_no_seurat()
  obj <- .make_tiny_seurat()
  # In Seurat 5, default assay class is Assay5 which inherits from
  # StdAssay. Confirm so the FindVariableFeatures.StdAssay + RunPCA
  # .StdAssay dispatch paths get tested through our existing contract
  # files rather than the legacy .Seurat S3 methods.
  assay_class <- class(obj[["RNA"]])
  expect_true("Assay5" %in% assay_class || "StdAssay" %in% assay_class,
              info = paste("Tiny Seurat assay class:", paste(assay_class,
                                                             collapse = ", ")))
})

test_that("FindWeights + CCAIntegration contracts deferred to integration", {
  .skip_if_no_seurat()
  info <- autozyme::inspect("seurat")
  targets <- vapply(info$targets, function(x) x$fn_name, character(1))
  bound <- vapply(info$targets, function(x) isTRUE(x$currently_bound),
                  logical(1))
  for (fn in c("FindWeights", "CCAIntegration")) {
    expect_true(fn %in% targets)
    expect_true(bound[match(fn, targets)])
  }
})
