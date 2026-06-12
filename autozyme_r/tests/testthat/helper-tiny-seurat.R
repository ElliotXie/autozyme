# Shared fixtures for Seurat-API contract tests.
#
# testthat auto-sources helper-*.R before any test- file. Each contract
# file (test-contract-NormalizeData.R etc.) builds on these helpers so
# new test files don't reinvent the tiny-object setup.

.skip_if_no_seurat <- function() {
  testthat::skip_if_not_installed("Seurat")
  testthat::skip_if_not_installed("SeuratObject")
}

#' Tiny CSR-counts Seurat object — under a second to build.
.make_tiny_seurat <- function(n_cells = 60, n_genes = 120, seed = 0) {
  .skip_if_no_seurat()
  set.seed(seed)
  counts <- matrix(stats::rpois(n_cells * n_genes, lambda = 2),
                   nrow = n_genes, ncol = n_cells)
  rownames(counts) <- paste0("g", seq_len(n_genes))
  colnames(counts) <- paste0("c", seq_len(n_cells))
  suppressWarnings(
    Seurat::CreateSeuratObject(counts = counts, min.cells = 0,
                               min.features = 0)
  )
}

#' Tiny normalized Seurat — uses vanilla NormalizeData via zyme=FALSE so
#' downstream contract tests don't depend on the patched NormalizeData
#' being correct (that's NormalizeData's own contract test's job).
.make_normalized_seurat <- function() {
  obj <- .make_tiny_seurat()
  suppressWarnings(Seurat::NormalizeData(obj, verbose = FALSE, zyme = FALSE))
}

.make_hvg_seurat <- function() {
  obj <- .make_normalized_seurat()
  suppressWarnings(
    Seurat::FindVariableFeatures(obj, nfeatures = 50, verbose = FALSE,
                                 zyme = FALSE)
  )
}

.make_scaled_seurat <- function() {
  obj <- .make_hvg_seurat()
  suppressWarnings(
    Seurat::ScaleData(obj, verbose = FALSE, zyme = FALSE)
  )
}

.make_pca_seurat <- function() {
  obj <- .make_scaled_seurat()
  suppressWarnings(
    Seurat::RunPCA(obj, npcs = 10, verbose = FALSE, zyme = FALSE)
  )
}
