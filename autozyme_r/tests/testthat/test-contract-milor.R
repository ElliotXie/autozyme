# Contract: miloR::calcNhoodDistance (+ internal miloR:::.calc_distance)
#
# Patched surface: 2 targets.
#   1. calcNhoodDistance -- public API; exposes a per-call `zyme=` escape.
#      Batches all per-nhood pairwise Euclidean distances into one C++ call
#      and writes the result via direct slot assignment `x@nhoodDistances`
#      to bypass miloR's broken `nhoodDistances<-,Milo-method` setter.
#   2. .calc_distance -- internal single-matrix helper; no `zyme=` arg, so
#      the bypass for it goes through autozyme::with_disabled().
#
# IMPORTANT FIXTURE NOTE: upstream `calcNhoodDistance(zyme = FALSE)` leaves
# `nhoodDistances(x)` EMPTY (length 0) because miloR's public setter only
# assigns on the dgT->dgC conversion path (documented in the patch header).
# So patched-vs-vanilla parity for the PUBLIC API cannot be read back
# through `nhoodDistances()`. We instead (a) prove the internal
# `.calc_distance` target is bit-exact patched-vs-disabled, and (b) prove
# the public path's output equals a manual `dist()` reference. Both nail
# numeric correctness without depending on the broken accessor.

.skip_if_no_milor <- function() {
  testthat::skip_if_not_installed("miloR")
  testthat::skip_if_not_installed("SingleCellExperiment")
  testthat::skip_if_not_installed("SummarizedExperiment")
  testthat::skip_if_not_installed("Matrix")
  testthat::skip_if_not_installed("irlba")
}

# Tiny Milo object with a graph + neighbourhoods + a PCA reducedDim, built
# end-to-end through miloR's own API. ~0.2s; nhoods come out ~120 x 17.
.make_tiny_milo <- function(seed = 1) {
  .skip_if_no_milor()
  suppressPackageStartupMessages({
    requireNamespace("miloR", quietly = TRUE)
    requireNamespace("SingleCellExperiment", quietly = TRUE)
    requireNamespace("SummarizedExperiment", quietly = TRUE)
    requireNamespace("irlba", quietly = TRUE)
  })
  set.seed(seed)
  n_cells <- 120L
  n_genes <- 80L
  counts <- matrix(stats::rpois(n_genes * n_cells, lambda = 3),
                   nrow = n_genes, ncol = n_cells)
  rownames(counts) <- paste0("g", seq_len(n_genes))
  colnames(counts) <- paste0("c", seq_len(n_cells))
  logc <- log1p(counts)
  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = counts, logcounts = logc))
  pca <- irlba::prcomp_irlba(t(logc), n = 10, scale. = TRUE, center = TRUE)
  SingleCellExperiment::reducedDim(sce, "PCA") <- pca$x
  milo <- miloR::Milo(sce)
  milo <- suppressWarnings(suppressMessages(
    miloR::buildGraph(milo, k = 10, d = 10, reduced.dim = "PCA")))
  milo <- suppressWarnings(suppressMessages(
    miloR::makeNhoods(milo, prop = 0.2, k = 10, d = 10, refined = TRUE,
                      reduced_dims = "PCA")))
  milo
}

test_that("milor registers calcNhoodDistance + .calc_distance and binds them", {
  .skip_if_no_milor()
  info <- autozyme::inspect("milor")
  expect_equal(info$upstream, "miloR")
  targets <- vapply(info$targets, function(x) x$fn_name, character(1))
  bound <- vapply(info$targets, function(x) isTRUE(x$currently_bound),
                  logical(1))
  expect_true("calcNhoodDistance" %in% targets)
  expect_true(".calc_distance" %in% targets)
  expect_true(bound[match("calcNhoodDistance", targets)])
  expect_true(bound[match(".calc_distance", targets)])
})

test_that(".calc_distance is bit-exact patched vs with_disabled", {
  .skip_if_no_milor()
  fast <- utils::getFromNamespace(".calc_distance", "miloR")
  set.seed(3)
  # .calc_distance takes a (cells x dims) reducedDim subset and returns a
  # sparse pairwise-Euclidean distance matrix.
  m <- matrix(stats::rnorm(8 * 5), nrow = 8, ncol = 5)
  rownames(m) <- paste0("c", seq_len(8))
  patched <- fast(m)
  vanilla <- autozyme::with_disabled(fast(m))
  # Both are sparse Matrix objects; the patched C++ kernel emits CSC
  # (dgCMatrix) where upstream returns dgTMatrix -- a benign storage-class
  # difference. The numeric content is what the contract pins.
  expect_true(methods::is(patched, "sparseMatrix"))
  expect_true(methods::is(vanilla, "sparseMatrix"))
  expect_equal(dim(patched), dim(vanilla))
  expect_equal(rownames(as.matrix(patched)), rownames(as.matrix(vanilla)))
  # No cross-row interaction in pairwise distances -> kernel must be exact.
  expect_equal(max(abs(as.matrix(patched) - as.matrix(vanilla))), 0,
               tolerance = 1e-12)
})

test_that(".calc_distance coerces non-matrix / integer storage cleanly", {
  .skip_if_no_milor()
  fast <- utils::getFromNamespace(".calc_distance", "miloR")
  set.seed(4)
  # integer-storage input (storage.mode coercion branch in fast_calc_distance)
  mi <- matrix(as.integer(sample.int(5, 6 * 4, replace = TRUE)),
               nrow = 6, ncol = 4)
  rownames(mi) <- paste0("c", seq_len(6))
  out <- fast(mi)
  ref <- autozyme::with_disabled(fast(mi))
  expect_equal(max(abs(as.matrix(out) - as.matrix(ref))), 0,
               tolerance = 1e-12)
})

test_that("public calcNhoodDistance populates nhoodDistances correctly", {
  .skip_if_no_milor()
  milo <- .make_tiny_milo()
  res <- suppressWarnings(suppressMessages(
    miloR::calcNhoodDistance(milo, d = 10, reduced.dim = "PCA")))
  nd <- res@nhoodDistances  # read slot directly; accessor is upstream-broken
  expect_true(length(nd) > 0L)
  expect_equal(length(nd), ncol(miloR::nhoods(milo)))
  expect_s4_class(nd[[1]], "dgCMatrix")

  # Numeric correctness: the patched batch kernel must reproduce a plain
  # column-subset `dist()` for each neighbourhood.
  nz <- Matrix::which(miloR::nhoods(milo) != 0, arr.ind = TRUE)
  rd <- SingleCellExperiment::reducedDim(milo, "PCA")[, seq_len(10),
                                                      drop = FALSE]
  max_diff <- 0
  for (j in seq_len(min(5L, length(nd)))) {
    rows_j <- nz[nz[, 2] == j, 1]
    if (length(rows_j) < 2L) next
    expected <- as.matrix(stats::dist(rd[rows_j, , drop = FALSE]))
    got <- as.matrix(nd[[j]])
    d <- max(abs(expected - got))
    if (is.finite(d) && d > max_diff) max_diff <- d
  }
  expect_lt(max_diff, 1e-8)
})

test_that("calcNhoodDistance(zyme = FALSE) runs the original path", {
  .skip_if_no_milor()
  milo <- .make_tiny_milo()
  # The escape hatch must dispatch to the captured original. Upstream
  # leaves @nhoodDistances empty (broken setter) -- that is the documented
  # vanilla behaviour, and the contract here is "does not error + returns a
  # Milo whose distances are NOT written through the public setter".
  res_v <- suppressWarnings(suppressMessages(
    miloR::calcNhoodDistance(milo, d = 10, reduced.dim = "PCA",
                             zyme = FALSE)))
  expect_s4_class(res_v, "Milo")
  # vanilla slot is empty (length 0) -- this asserts the documented upstream
  # broken-setter behaviour, which is precisely why the patch bypasses it.
  expect_equal(length(res_v@nhoodDistances), 0L)
})

test_that("milor deactivate/activate round-trip restores the original API", {
  .skip_if_no_milor()
  # Make sure we leave milor active for the rest of the suite no matter what.
  on.exit(try(autozyme::activate("milor"), silent = TRUE), add = TRUE)

  autozyme::activate("milor")
  expect_equal(autozyme::inspect("milor")$status, "active")
  patched <- utils::getFromNamespace(".calc_distance", "miloR")

  autozyme::deactivate("milor")
  expect_equal(autozyme::inspect("milor")$status, "inactive")
  # After deactivation the public symbol is the pristine original: its
  # formals must NOT carry the patch's `zyme` argument.
  restored_formals <- names(formals(miloR::calcNhoodDistance))
  expect_false("zyme" %in% restored_formals)
  expect_true(all(c("x", "d", "reduced.dim", "use.assay") %in%
                    restored_formals))

  # Re-activate and confirm the binding flips back on.
  autozyme::activate("milor")
  expect_equal(autozyme::inspect("milor")$status, "active")
  reactivated <- utils::getFromNamespace(".calc_distance", "miloR")
  expect_identical(reactivated, patched)
})
