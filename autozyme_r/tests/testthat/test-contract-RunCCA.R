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

test_that("RunCCA exercises a fast path (native or scipy) and not upstream svd", {
  # Same idea as the RunPCA contract: branch on native_pca_available().
  #   native available -> CCA SVD = form_A on a fast BLAS + irlba(mult=
  #     native_matmul); never imports numpy/scipy.
  #   native missing   -> CCA SVD = scipy.sparse.linalg.svds via reticulate.
  # Either way the test rejects a silent fallback to upstream base::svd.
  .skip_if_no_seurat()
  skip_if_not_installed("reticulate")
  native_ok <- isTRUE(autozyme:::native_pca_available())
  if (!native_ok) {
    skip_if_not(isTRUE(.az_py_bind()),
                "no native BLAS and no Python with numpy/scipy")
  }

  obj_a <- .make_cca_seurat(1, "a")
  obj_b <- .make_cca_seurat(2, "b")
  features <- rownames(obj_a)[seq_len(120)]

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
  if (native_ok) {
    expect_false(used_python)                            # native ran
  } else {
    expect_true(used_python)                             # scipy ran
  }
  expect_true("cca" %in% SeuratObject::Reductions(patched))
})
