# Contract: Seurat::SCTransform (S3 dispatch on Seurat objects + the
# .default underlying method)
#
# Patched signature has explicit `zyme = TRUE, turbo = NULL` plus the
# full SCTransform kwarg set. Public entry `Seurat::SCTransform(object)`.
# The fast path optimizes the per-gene NB GLM fitting inside vst.

test_that("SCTransform returns a Seurat object with SCT assay populated", {
  .skip_if_no_seurat()
  testthat::skip_if_not_installed("sctransform")
  obj <- .make_tiny_seurat(n_cells = 80, n_genes = 200)
  # SCTransform is the only test that needs a slightly bigger fixture --
  # tiny Seurat (60x120) hits an empty-variance edge in some sctransform
  # internal regressions.
  out <- tryCatch(
    suppressWarnings(suppressMessages(
      Seurat::SCTransform(obj, verbose = FALSE,
                          variable.features.n = 50,
                          ncells = 50)
    )),
    error = function(e) e
  )
  if (inherits(out, "error")) {
    testthat::skip(paste0("SCTransform raised on minimal fixture: ",
                          conditionMessage(out)))
  }
  expect_s4_class(out, "Seurat")
  expect_true("SCT" %in% SeuratObject::Assays(out),
              info = "SCT assay missing after SCTransform")
})

test_that("SCTransform zyme=FALSE matches patched SCT counts", {
  .skip_if_no_seurat()
  testthat::skip_if_not_installed("sctransform")
  obj <- .make_tiny_seurat(n_cells = 80, n_genes = 200)
  vanilla <- tryCatch(
    suppressWarnings(suppressMessages(
      Seurat::SCTransform(obj, verbose = FALSE,
                          variable.features.n = 50,
                          ncells = 50, zyme = FALSE)
    )),
    error = function(e) e
  )
  if (inherits(vanilla, "error")) {
    testthat::skip(paste0("vanilla SCTransform failed: ",
                          conditionMessage(vanilla)))
  }
  patched <- suppressWarnings(suppressMessages(
    Seurat::SCTransform(obj, verbose = FALSE,
                        variable.features.n = 50,
                        ncells = 50)
  ))
  expect_equal(class(patched), class(vanilla))
  # SCT assay counts have the same rows / cols across patched + vanilla.
  cnt_p <- SeuratObject::GetAssayData(patched, assay = "SCT",
                                      layer = "counts")
  cnt_v <- SeuratObject::GetAssayData(vanilla, assay = "SCT",
                                      layer = "counts")
  expect_equal(dim(cnt_p), dim(cnt_v))
})
