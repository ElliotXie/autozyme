# Contract: Seurat::FindNeighbors

test_that("FindNeighbors returns a Seurat with snn graph populated", {
  .skip_if_no_seurat()
  withr::local_envvar(c(AUTOZYME_SEURAT_FINDNEIGHBORS_BACKEND = NA))
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

test_that("FindNeighbors zyme=FALSE still returns upstream graph", {
  .skip_if_no_seurat()
  obj <- .make_pca_seurat()
  out <- suppressWarnings(
    Seurat::FindNeighbors(obj, reduction = "pca", dims = 1:10,
                          verbose = FALSE, zyme = FALSE)
  )
  graphs <- SeuratObject::Graphs(out)
  expect_true(any(grepl("_nn$", graphs)))
  expect_true(any(grepl("_snn$", graphs)))
})

test_that("FindNeighbors exact native kNN matches brute force distances", {
  set.seed(42)
  x <- matrix(stats::rnorm(360), nrow = 60, ncol = 6)
  k <- 10L
  idx <- autozyme:::seurat_exact_knn_f32(x, k, 2L)
  dist2 <- as.matrix(stats::dist(x))^2
  ref <- t(apply(dist2, 1L, function(z) order(z)[seq_len(k)]))
  expect_true(all(idx == ref))
})

test_that("FindNeighbors Annoy backend switch remains available", {
  .skip_if_no_seurat()
  withr::local_envvar(c(AUTOZYME_SEURAT_FINDNEIGHBORS_BACKEND = "annoy"))
  obj <- .make_pca_seurat()
  out <- suppressWarnings(
    Seurat::FindNeighbors(obj, reduction = "pca", dims = 1:10,
                          verbose = FALSE)
  )
  graphs <- SeuratObject::Graphs(out)
  expect_true(any(grepl("_nn$", graphs)))
  expect_true(any(grepl("_snn$", graphs)))
})
