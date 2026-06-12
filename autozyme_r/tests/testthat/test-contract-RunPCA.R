# Contract: Seurat::RunPCA

test_that("RunPCA returns a Seurat with pca reduction populated", {
  .skip_if_no_seurat()
  obj <- .make_scaled_seurat()
  out <- suppressWarnings(Seurat::RunPCA(obj, npcs = 10, verbose = FALSE))
  expect_s4_class(out, "Seurat")
  expect_true("pca" %in% SeuratObject::Reductions(out))
  emb <- SeuratObject::Embeddings(out, reduction = "pca")
  expect_equal(ncol(emb), 10)
  expect_equal(nrow(emb), ncol(obj))  # one row per cell
})

test_that("RunPCA zyme=FALSE matches vanilla embeddings up to sign", {
  .skip_if_no_seurat()
  obj <- .make_scaled_seurat()
  vanilla <- suppressWarnings(Seurat::RunPCA(obj, npcs = 10, verbose = FALSE,
                                             zyme = FALSE))
  patched <- suppressWarnings(Seurat::RunPCA(obj, npcs = 10, verbose = FALSE))
  emb_v <- SeuratObject::Embeddings(vanilla, reduction = "pca")
  emb_p <- SeuratObject::Embeddings(patched, reduction = "pca")
  # PCs are sign-ambiguous; compare per-PC up to sign via abs cosine.
  for (k in seq_len(min(5, ncol(emb_v)))) {
    cos_k <- abs(sum(emb_v[, k] * emb_p[, k])) /
      (sqrt(sum(emb_v[, k] ^ 2)) * sqrt(sum(emb_p[, k] ^ 2)) + 1e-12)
    expect_gt(cos_k, 0.99)
  }
})

test_that("RunPCA actually exercises the Python fast path when numpy is available", {
  # Parity tests above pass even if the patch silently falls back to upstream
  # (no Python) -- so they can't catch a "we shipped a no-op" regression. This
  # test asserts the Python path *ran*. It only runs where a numpy/scipy Python
  # can be bound; elsewhere the fallback is correct and there's nothing to check.
  .skip_if_no_seurat()
  skip_if_not_installed("reticulate")
  skip_if_not(isTRUE(.az_py_bind()), "no Python with numpy/scipy available")

  obj <- .make_scaled_seurat()

  # Both fast RunPCA paths (default + StdAssay) call reticulate::import("numpy")
  # inside the function body, so a numpy import *during the call* proves the
  # Python branch ran rather than the upstream irlba fallback. Spy on import
  # while delegating to the real one so the computation still happens.
  orig_import <- reticulate::import
  used_python <- FALSE
  testthat::local_mocked_bindings(
    import = function(module, ...) {
      if (identical(module, "numpy")) used_python <<- TRUE
      orig_import(module, ...)
    },
    .package = "reticulate")

  out <- suppressWarnings(Seurat::RunPCA(obj, npcs = 10, verbose = FALSE))
  expect_true(used_python)                                # fast path was taken
  expect_true("pca" %in% SeuratObject::Reductions(out))  # and produced a result
})
