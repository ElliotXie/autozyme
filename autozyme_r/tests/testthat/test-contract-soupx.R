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

test_that("SoupX fast path aligns metadata weights by cell name", {
  sc <- .make_soupx_sc_toy()
  on.exit(autozyme::activate("soupx"), add = TRUE)
  autozyme::activate("soupx")

  canonical <- suppressWarnings(SoupX::adjustCounts(
    sc,
    roundToInt = FALSE,
    verbose = 0
  ))
  reordered <- sc
  reordered$metaData <- reordered$metaData[
    rev(rownames(reordered$metaData)),
    ,
    drop = FALSE
  ]
  actual <- suppressWarnings(SoupX::adjustCounts(
    reordered,
    roundToInt = FALSE,
    verbose = 0
  ))

  delta <- as.matrix(actual) - as.matrix(canonical)
  expect_lte(max(abs(delta)), 1e-12)
})

test_that("SoupX fallback preserves unsupported argument behavior", {
  sc <- .make_soupx_sc_toy()
  on.exit(autozyme::activate("soupx"), add = TRUE)
  autozyme::activate("soupx")

  observed <- sc$toc[, seq_len(2), drop = FALSE]
  clusters <- stats::setNames(rep("a", ncol(observed)), colnames(observed))
  clustered <- Matrix::Matrix(
    matrix(
      0,
      nrow = nrow(observed),
      ncol = 1,
      dimnames = list(rownames(observed), "a")
    ),
    sparse = TRUE
  )
  expand_clusters <- utils::getFromNamespace("expandClusters", "SoupX")

  expect_error(
    expand_clusters(
      clustered,
      observed,
      clusters,
      rep(1, ncol(observed)),
      verbose = 0,
      unsupported_argument = TRUE
    ),
    "unused argument"
  )
})

test_that("SoupX zero cell weights use the upstream fallback", {
  sc <- .make_soupx_sc_toy()
  on.exit(autozyme::activate("soupx"), add = TRUE)
  sc$metaData$rho[[1]] <- 0

  upstream <- tryCatch(
    autozyme::with_disabled(
      suppressWarnings(SoupX::adjustCounts(
        sc,
        roundToInt = FALSE,
        verbose = 0
      ))
    ),
    error = identity
  )
  autozyme::activate("soupx")
  delegated <- tryCatch(
    suppressWarnings(SoupX::adjustCounts(
      sc,
      roundToInt = FALSE,
      verbose = 0
    )),
    error = identity
  )

  expect_s3_class(upstream, "error")
  expect_s3_class(delegated, "error")
  expect_identical(conditionMessage(delegated), conditionMessage(upstream))
})

test_that("SoupX native water filling stays finite near saturation", {
  p <- as.integer(c(0, 2))
  row_i <- as.integer(c(0, 1))
  counts <- c(0.1, 1)
  target <- 0.5
  nearly_degenerate_weights <- c(1, 1e-17)

  clustered <- autozyme:::soupx_cluster_soup_from_cells_cpp(
    p,
    row_i,
    counts,
    1L,
    target,
    nearly_degenerate_weights,
    2L,
    1L
  )
  corrected <- autozyme:::soupx_adjust_counts_no_cluster_x_cpp(
    p,
    row_i,
    counts,
    target,
    nearly_degenerate_weights,
    2L
  )

  expect_true(all(is.finite(clustered)))
  expect_equal(sum(clustered), target, tolerance = 1e-12)
  expect_true(all(is.finite(corrected)))
  expect_true(all(corrected >= 0))
  expect_equal(sum(counts - corrected), target, tolerance = 1e-12)
})

test_that("SoupX native expansion rejects an empty weight support", {
  expect_error(
    autozyme:::soupx_expand_corrected_x_cpp(
      as.integer(c(0, 1)),
      0L,
      1,
      matrix(0.5, nrow = 1, ncol = 1),
      1L,
      0
    ),
    "positive finite weight sum"
  )
})
