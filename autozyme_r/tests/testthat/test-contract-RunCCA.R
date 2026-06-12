# Contract: Seurat::RunCCA (S3 methods on Seurat + .default)
#
# Patched signature has explicit `zyme = TRUE, turbo = NULL`. The fast
# path calls a Python helper for the CCA SVD; falls through to vanilla
# when Python isn't available or zyme=FALSE.

.make_cca_seurat <- function(seed, prefix) {
  .skip_if_no_seurat()
  set.seed(seed)
  n_cells <- 80
  n_genes <- 200
  counts <- matrix(stats::rpois(n_cells * n_genes, lambda = 2),
                   nrow = n_genes)
  rownames(counts) <- paste0("g", seq_len(n_genes))
  colnames(counts) <- paste0(prefix, seq_len(n_cells))
  obj <- Seurat::CreateSeuratObject(counts = counts, min.cells = 0,
                                    min.features = 0)
  obj <- Seurat::NormalizeData(obj, verbose = FALSE, zyme = FALSE)
  obj <- Seurat::FindVariableFeatures(obj, nfeatures = 120, verbose = FALSE,
                                      zyme = FALSE)
  Seurat::ScaleData(obj, features = rownames(obj),
                    verbose = FALSE, zyme = FALSE)
}

test_that("RunCCA.default returns canonical cca matrix dimensions", {
  .skip_if_no_seurat()
  set.seed(1)
  x <- matrix(stats::rnorm(160 * 64), nrow = 160)
  y <- matrix(stats::rnorm(160 * 60), nrow = 160)
  rownames(x) <- rownames(y) <- paste0("g", seq_len(160))
  colnames(x) <- paste0("a", seq_len(64))
  colnames(y) <- paste0("b", seq_len(60))

  out <- suppressWarnings(Seurat::RunCCA(x, y, num.cc = 20, verbose = FALSE))
  expect_type(out, "list")
  expect_equal(dim(out$ccv), c(124L, 20L))
  expect_equal(length(out$d), 20L)
})

test_that("RunCCA.Seurat returns cca reduction and zyme=FALSE dimensions match", {
  .skip_if_no_seurat()
  obj_a <- .make_cca_seurat(1, "a")
  obj_b <- .make_cca_seurat(2, "b")
  features <- rownames(obj_a)[seq_len(120)]

  patched <- suppressWarnings(suppressMessages(
    Seurat::RunCCA(obj_a, obj_b, features = features, num.cc = 20,
                   verbose = FALSE)
  ))
  vanilla <- suppressWarnings(suppressMessages(autozyme::with_disabled(
    Seurat::RunCCA(obj_a, obj_b, features = features, num.cc = 20,
                   verbose = FALSE)
  )))

  expect_s4_class(patched, "Seurat")
  expect_true("cca" %in% SeuratObject::Reductions(patched))
  expect_equal(
    dim(SeuratObject::Embeddings(patched, reduction = "cca")),
    dim(SeuratObject::Embeddings(vanilla, reduction = "cca"))
  )
})

test_that("RunCCA actually exercises the Python fast path when numpy is available", {
  # As with RunPCA: parity passes even on a silent fallback, so assert the
  # Python path *ran*. Skipped where no numpy/scipy Python can be bound.
  .skip_if_no_seurat()
  skip_if_not_installed("reticulate")
  skip_if_not(isTRUE(.az_py_bind()), "no Python with numpy/scipy available")

  obj_a <- .make_cca_seurat(1, "a")
  obj_b <- .make_cca_seurat(2, "b")
  features <- rownames(obj_a)[seq_len(120)]

  # The fast RunCCA SVD runs through reticulate (numpy + scipy.sparse.linalg);
  # an import of either during the call proves the Python branch executed
  # rather than the upstream irlba fallback.
  orig_import <- reticulate::import
  used_python <- FALSE
  testthat::local_mocked_bindings(
    import = function(module, ...) {
      if (grepl("^(numpy|scipy)", module)) used_python <<- TRUE
      orig_import(module, ...)
    },
    .package = "reticulate")

  patched <- suppressWarnings(suppressMessages(
    Seurat::RunCCA(obj_a, obj_b, features = features, num.cc = 20,
                   verbose = FALSE)))
  expect_true(used_python)
  expect_true("cca" %in% SeuratObject::Reductions(patched))
})
