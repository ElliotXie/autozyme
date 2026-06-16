test_that("Seurat integration fast paths preserve upstream parameter policy", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")

  expect_true(autozyme:::.ensure_registered("seurat"))
  targets <- autozyme:::.zyme_registry[["seurat"]]$targets

  expect_equal(formals(targets$FindIntegrationAnchors)$n.trees, 50)
  expect_equal(formals(targets$FindWeights)$n.trees, 50)

  cca_body <- paste(deparse(body(targets$CCAIntegration)), collapse = "\n")
  expect_match(cca_body, "dims = dims", fixed = TRUE)
  expect_match(cca_body, "k.weight = k.weight", fixed = TRUE)
  expect_false(grepl("\\.seurat_orig_CCAIntegration\\([\\s\\S]*dims = 1:30",
                     cca_body, perl = TRUE))
  expect_false(grepl("\\.seurat_orig_CCAIntegration\\([\\s\\S]*k\\.weight = 100",
                     cca_body, perl = TRUE))
})

test_that("RunUMAP is part of the Seurat package patch", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("SeuratObject")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("uwot")

  expect_true(autozyme:::.ensure_registered("seurat"))
  targets <- names(autozyme:::.zyme_registry[["seurat"]]$targets)

  expect_true("RunUMAP.Seurat" %in% targets)
  expect_false("RunUMAP.default" %in% targets)
  expect_false("seurat_runumap" %in% autozyme::list_patches())

  on.exit(try(autozyme::activate("seurat"), silent = TRUE), add = TRUE)
  autozyme::deactivate("seurat")
  expect_equal(unname(autozyme::status()[["seurat"]]), "inactive")

  expect_true(autozyme::activate("seurat"))
  expect_equal(unname(autozyme::status()[["seurat"]]), "active")

  autozyme::deactivate("seurat")
  expect_equal(unname(autozyme::status()[["seurat"]]), "inactive")
})
