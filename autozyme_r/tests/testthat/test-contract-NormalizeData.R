# Contract: Seurat::NormalizeData
#
# Fast path requires LogNormalize + scale.factor scalar + margin == 1.
# Anything else delegates to the vanilla implementation. Contract test
# pins:
#   - Patched call returns a Seurat object (same class as vanilla)
#   - "data" layer is populated
#   - zyme = FALSE per-call escape produces vanilla-identical output
#   - Non-LogNormalize method delegates cleanly

test_that("NormalizeData returns a Seurat object", {
  .skip_if_no_seurat()
  obj <- .make_tiny_seurat()
  out <- suppressWarnings(Seurat::NormalizeData(obj, verbose = FALSE))
  expect_s4_class(out, "Seurat")
  expect_true("data" %in% SeuratObject::Layers(out, search = "data"))
})

test_that("NormalizeData zyme=FALSE matches vanilla numerically", {
  .skip_if_no_seurat()
  obj <- .make_tiny_seurat()
  vanilla <- suppressWarnings(
    Seurat::NormalizeData(obj, verbose = FALSE, zyme = FALSE)
  )
  patched <- suppressWarnings(Seurat::NormalizeData(obj, verbose = FALSE))
  expect_equal(
    as.matrix(SeuratObject::GetAssayData(patched, layer = "data")),
    as.matrix(SeuratObject::GetAssayData(vanilla, layer = "data")),
    tolerance = 1e-6
  )
})

test_that("NormalizeData CLR delegates to vanilla (not on fast path)", {
  .skip_if_no_seurat()
  obj <- .make_tiny_seurat()
  vanilla <- suppressWarnings(
    Seurat::NormalizeData(obj, normalization.method = "CLR",
                          verbose = FALSE, zyme = FALSE)
  )
  patched <- suppressWarnings(
    Seurat::NormalizeData(obj, normalization.method = "CLR",
                          verbose = FALSE)
  )
  expect_s4_class(patched, "Seurat")
  expect_equal(
    as.matrix(SeuratObject::GetAssayData(patched, layer = "data")),
    as.matrix(SeuratObject::GetAssayData(vanilla, layer = "data")),
    tolerance = 1e-6
  )
})
