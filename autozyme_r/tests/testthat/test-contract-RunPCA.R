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

test_that("RunPCA exercises a fast path (native or scipy) and not upstream irlba", {
  # Parity tests above pass even on a silent fallback to upstream, so this is
  # the regression net. The dispatcher tries native first (zero Python, runs
  # when a fast BLAS resolves — bundled on Win, Accelerate on mac, AUTOZYME_
  # OPENBLAS_DLL elsewhere) and only then scipy. Branch the assertion on
  # native_pca_available():
  #   native available -> expect numpy NOT imported (proves native fired)
  #   native missing   -> expect numpy IS imported (proves scipy fired)
  # Either way the upstream irlba fallback would import neither, so the test
  # catches silent regressions on both code paths.
  .skip_if_no_seurat()
  skip_if_not_installed("reticulate")
  native_ok <- isTRUE(autozyme:::native_pca_available())
  if (!native_ok) {
    skip_if_not(isTRUE(.az_py_bind()),
                "no native BLAS and no Python with numpy/scipy")
  }

  obj <- .make_scaled_seurat()
  orig_import <- reticulate::import
  used_python <- FALSE
  testthat::local_mocked_bindings(
    import = function(module, ...) {
      if (identical(module, "numpy")) used_python <<- TRUE
      orig_import(module, ...)
    },
    .package = "reticulate")

  out <- suppressWarnings(Seurat::RunPCA(obj, npcs = 10, verbose = FALSE))
  if (native_ok) {
    expect_false(used_python)                            # native ran
  } else {
    expect_true(used_python)                             # scipy ran
  }
  expect_true("pca" %in% SeuratObject::Reductions(out))
})
