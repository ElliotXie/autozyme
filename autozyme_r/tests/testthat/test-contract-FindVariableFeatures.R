# Contract: Seurat::FindVariableFeatures

test_that("FindVariableFeatures returns a Seurat with HVG annotated", {
  .skip_if_no_seurat()
  obj <- .make_normalized_seurat()
  out <- suppressWarnings(
    Seurat::FindVariableFeatures(obj, nfeatures = 50, verbose = FALSE)
  )
  expect_s4_class(out, "Seurat")
  expect_equal(length(SeuratObject::VariableFeatures(out)), 50)
})

test_that("FindVariableFeatures zyme=FALSE matches vanilla HVG set", {
  .skip_if_no_seurat()
  obj <- .make_normalized_seurat()
  vanilla <- suppressWarnings(
    Seurat::FindVariableFeatures(obj, nfeatures = 50, verbose = FALSE,
                                 zyme = FALSE)
  )
  patched <- suppressWarnings(
    Seurat::FindVariableFeatures(obj, nfeatures = 50, verbose = FALSE)
  )
  expect_setequal(
    SeuratObject::VariableFeatures(patched),
    SeuratObject::VariableFeatures(vanilla)
  )
})
