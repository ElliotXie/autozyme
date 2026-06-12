# Contract: BayesSpace::iterate_t (Gibbs sampling step for spatial clustering)
#
# Single patched target wrapping a C++ kernel (fast_iterate_t_impl).
# Building a real BayesSpace-compatible input requires a SingleCellExperiment
# with spatial coords + cluster init + Gibbs hyperparameters -- non-trivial.
# Universal activation smoke covers the load contract.

.skip_if_no_bayesspace <- function() {
  testthat::skip_if_not_installed("BayesSpace")
}

test_that("bayesspace iterate_t target is registered and bound", {
  .skip_if_no_bayesspace()
  info <- autozyme::inspect("bayesspace")
  targets <- vapply(info$targets, function(x) x$fn_name, character(1))
  bound <- vapply(info$targets, function(x) isTRUE(x$currently_bound),
                  logical(1))
  expect_equal(info$upstream, "BayesSpace")
  expect_true("iterate_t" %in% targets)
  expect_true(bound[match("iterate_t", targets)])
})
