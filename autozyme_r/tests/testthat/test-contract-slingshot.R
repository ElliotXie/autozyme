# Contract: slingshot::getCurves
#
# Patches the per-lineage curve-fitting that runs after getLineages has
# laid down the MST topology. The patched function expects a
# PseudotimeOrdering carrying lineages/mst/slingParams metadata.
#
# Fixture: 4-cluster 2D embedding with a clean linear trajectory
# (clusters laid out left-to-right). getLineages auto-builds an MST
# over the cluster centroids, then getCurves smooths a principal curve
# along each lineage.

.skip_if_no_slingshot <- function() {
  testthat::skip_if_not_installed("slingshot")
  testthat::skip_if_not_installed("S4Vectors")
}

.make_tiny_pto <- function(seed = 0) {
  set.seed(seed)
  k <- 4
  per_cluster <- 25
  centers <- rbind(c(0, 0), c(2, 1), c(4, 0), c(6, 1))
  X <- do.call(rbind, lapply(seq_len(k), function(i) {
    pts <- matrix(stats::rnorm(per_cluster * 2, sd = 0.3),
                  nrow = per_cluster, ncol = 2)
    pts + matrix(rep(centers[i, ], each = per_cluster), nrow = per_cluster)
  }))
  colnames(X) <- c("D1", "D2")
  rownames(X) <- paste0("c", seq_len(nrow(X)))
  cl <- rep(seq_len(k), each = per_cluster)
  suppressWarnings(slingshot::getLineages(data = X, clusterLabels = cl))
}

test_that("getCurves returns a PseudotimeOrdering with curves populated", {
  .skip_if_no_slingshot()
  pto <- tryCatch(.make_tiny_pto(), error = function(e) e)
  if (inherits(pto, "error")) {
    testthat::skip(paste0("slingshot fixture build failed: ",
                          conditionMessage(pto)))
  }
  out <- suppressWarnings(slingshot::getCurves(pto))
  expect_s4_class(out, "PseudotimeOrdering")
  meta <- S4Vectors::metadata(out)
  expect_true("curves" %in% names(meta),
              info = "expected 'curves' in PTO metadata after getCurves()")
})

test_that("getCurves with_disabled() matches patched curve count", {
  .skip_if_no_slingshot()
  pto <- tryCatch(.make_tiny_pto(), error = function(e) e)
  if (inherits(pto, "error")) {
    testthat::skip(paste0("slingshot fixture build failed: ",
                          conditionMessage(pto)))
  }
  # The slingshot patch uses the framework's `is_disabled()` toggle for
  # bypass rather than a per-call `zyme =` kwarg (the patch signature
  # has no `zyme` arg). Use `with_disabled({...})` instead.
  vanilla <- autozyme::with_disabled(
    suppressWarnings(slingshot::getCurves(pto))
  )
  patched <- suppressWarnings(slingshot::getCurves(pto))
  expect_equal(
    length(S4Vectors::metadata(vanilla)$curves),
    length(S4Vectors::metadata(patched)$curves)
  )
})
