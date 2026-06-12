# Contract: Seurat::ScaleData

test_that("ScaleData returns a Seurat with scale.data layer populated", {
  .skip_if_no_seurat()
  obj <- .make_hvg_seurat()
  out <- suppressWarnings(Seurat::ScaleData(obj, verbose = FALSE))
  expect_s4_class(out, "Seurat")
  expect_true("scale.data" %in% SeuratObject::Layers(out))
})

test_that("ScaleData zyme=FALSE matches vanilla scale.data", {
  .skip_if_no_seurat()
  obj <- .make_hvg_seurat()
  vanilla <- suppressWarnings(Seurat::ScaleData(obj, verbose = FALSE,
                                                zyme = FALSE))
  patched <- suppressWarnings(Seurat::ScaleData(obj, verbose = FALSE))
  expect_equal(
    SeuratObject::GetAssayData(patched, layer = "scale.data"),
    SeuratObject::GetAssayData(vanilla, layer = "scale.data"),
    tolerance = 1e-4
  )
})
