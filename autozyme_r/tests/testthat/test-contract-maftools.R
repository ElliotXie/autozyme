# Contract: maftools::read.maf (+ validateMaf, summarizeMaf internals)
#
# Patched surface: 3 targets (read.maf, validateMaf, summarizeMaf).
# Public entry is `maftools::read.maf(maf, ...)`. fast_read.maf has an
# explicit `zyme = TRUE` arg (per the patch survey).
#
# Fixture: bundled tcga_laml.maf.gz from maftools' own extdata — a real
# small TCGA MAF (~190 samples) that the upstream uses for examples.

.skip_if_no_maftools <- function() {
  testthat::skip_if_not_installed("maftools")
}

.tcga_laml_path <- function() {
  p <- system.file("extdata", "tcga_laml.maf.gz", package = "maftools")
  if (!nzchar(p)) {
    testthat::skip("tcga_laml.maf.gz not found in maftools extdata")
  }
  p
}

test_that("read.maf returns an MAF object with expected slots", {
  .skip_if_no_maftools()
  maf_path <- .tcga_laml_path()
  out <- suppressWarnings(suppressMessages(
    maftools::read.maf(maf = maf_path, verbose = FALSE)
  ))
  expect_s4_class(out, "MAF")
  # MAF S4 has @data (mutation table) + @gene.summary + @maf.silent.
  expect_true("data" %in% methods::slotNames(out))
  expect_gt(nrow(out@data), 0L)
})

test_that("read.maf with_disabled() matches patched mutation count", {
  .skip_if_no_maftools()
  maf_path <- .tcga_laml_path()
  vanilla <- autozyme::with_disabled(
    suppressWarnings(suppressMessages(
      maftools::read.maf(maf = maf_path, verbose = FALSE)
    ))
  )
  patched <- suppressWarnings(suppressMessages(
    maftools::read.maf(maf = maf_path, verbose = FALSE)
  ))
  expect_equal(class(patched), class(vanilla))
  # The mutation count + sample count must match -- core read.maf contract.
  expect_equal(nrow(patched@data), nrow(vanilla@data))
  expect_setequal(
    unique(patched@data$Tumor_Sample_Barcode),
    unique(vanilla@data$Tumor_Sample_Barcode)
  )
})

test_that("read.maf zyme=FALSE per-call delegates cleanly", {
  .skip_if_no_maftools()
  maf_path <- .tcga_laml_path()
  out <- suppressWarnings(suppressMessages(
    maftools::read.maf(maf = maf_path, verbose = FALSE, zyme = FALSE)
  ))
  expect_s4_class(out, "MAF")
  expect_gt(nrow(out@data), 0L)
})
