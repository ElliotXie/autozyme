# Contract: Seurat::FindNeighbors

test_that("FindNeighbors returns a Seurat with snn graph populated", {
  .skip_if_no_seurat()
  obj <- .make_pca_seurat()
  out <- suppressWarnings(
    Seurat::FindNeighbors(obj, reduction = "pca", dims = 1:10,
                          verbose = FALSE)
  )
  expect_s4_class(out, "Seurat")
  # FindNeighbors adds graphs named "<assay>_nn" and "<assay>_snn".
  graphs <- SeuratObject::Graphs(out)
  expect_true(any(grepl("_snn$", graphs)))
})

test_that("FindNeighbors zyme=FALSE matches vanilla graph", {
  .skip_if_no_seurat()
  obj <- .make_pca_seurat()
  vanilla <- suppressWarnings(
    Seurat::FindNeighbors(obj, reduction = "pca", dims = 1:10,
                          verbose = FALSE, zyme = FALSE)
  )
  patched <- suppressWarnings(
    Seurat::FindNeighbors(obj, reduction = "pca", dims = 1:10,
                          verbose = FALSE)
  )
  graphs_v <- SeuratObject::Graphs(vanilla)
  graphs_p <- SeuratObject::Graphs(patched)
  expect_setequal(graphs_v, graphs_p)
  # SNN graph values should match (up to tiny float drift).
  snn_name <- grep("_snn$", graphs_v, value = TRUE)[1]
  expect_equal(
    as.matrix(vanilla[[snn_name]]),
    as.matrix(patched[[snn_name]]),
    tolerance = 1e-5
  )
})
