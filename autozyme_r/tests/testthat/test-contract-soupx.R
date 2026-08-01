# Contract: SoupX::adjustCounts
#
# The accelerated path is intentionally narrow: sparse SoupChannel input,
# subtraction, fractional output, default tolerances, and implicit clusters.
# Unsupported branches must continue to delegate to upstream SoupX.

.make_soupx_sc_toy <- function() {
  testthat::skip_if_not_installed("SoupX")
  testthat::skip_if_not_installed("Matrix")

  data_env <- new.env(parent = emptyenv())
  utils::data("scToy", package = "SoupX", envir = data_env)
  sc <- data_env$scToy
  sc$toc <- methods::as(sc$toc, "CsparseMatrix")
  sc
}

test_that("SoupX fractional subtraction is numerically equivalent", {
  sc <- .make_soupx_sc_toy()
  on.exit(autozyme::activate("soupx"), add = TRUE)

  upstream <- autozyme::with_disabled(
    suppressWarnings(SoupX::adjustCounts(
      sc,
      roundToInt = FALSE,
      verbose = 0
    ))
  )
  autozyme::activate("soupx")
  accelerated <- suppressWarnings(SoupX::adjustCounts(
    sc,
    roundToInt = FALSE,
    verbose = 0
  ))

  expect_identical(dim(accelerated), dim(upstream))
  expect_identical(dimnames(accelerated), dimnames(upstream))
  delta <- as.matrix(accelerated) - as.matrix(upstream)
  expect_lte(max(abs(delta)), 1e-12)
  expect_lte(
    sqrt(sum(delta^2)) / sqrt(sum(as.matrix(upstream)^2)),
    1e-12
  )
})

test_that("SoupX integer rounding delegates exactly to upstream", {
  sc <- .make_soupx_sc_toy()
  on.exit(autozyme::activate("soupx"), add = TRUE)

  set.seed(717)
  upstream <- autozyme::with_disabled(
    suppressWarnings(SoupX::adjustCounts(
      sc,
      roundToInt = TRUE,
      verbose = 0
    ))
  )
  autozyme::activate("soupx")
  set.seed(717)
  delegated <- suppressWarnings(SoupX::adjustCounts(
    sc,
    roundToInt = TRUE,
    verbose = 0
  ))

  expect_identical(delegated, upstream)
})
