# Contract: scriabin::GenerateCCIM (cell-cell interaction matrix)
#
# Single patched target. Public entry needs a Seurat object with SCT
# normalization + a CellChat-style ligand-receptor database to compute
# the per-cell-pair interaction matrix.

.skip_if_no_scriabin <- function() {
  testthat::skip_if_not_installed("scriabin")
}

test_that("scriabin GenerateCCIM target is registered and bound", {
  .skip_if_no_scriabin()
  info <- autozyme::inspect("scriabin")
  targets <- vapply(info$targets, function(x) x$fn_name, character(1))
  bound <- vapply(info$targets, function(x) isTRUE(x$currently_bound),
                  logical(1))
  expect_equal(info$upstream, "scriabin")
  expect_true("GenerateCCIM" %in% targets)
  expect_true(bound[match("GenerateCCIM", targets)])
})
