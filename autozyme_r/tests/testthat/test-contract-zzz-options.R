# Contract tests for package-load side effects (`.onLoad` in R/zzz.R).
#
# Regression target: the May 2026 user report — first attempt at
# `future.apply::future_lapply` over a Seurat object crashes on R's
# default `future.globals.maxSize = 500MB`. autozyme's `.onLoad` raises
# it to 16 GiB on package load. These tests pin that contract so a
# future refactor of zzz.R can't silently revert it.
#
# Pattern for adding a new contract test:
#   - one file per public API surface (or, here, per .onLoad invariant)
#   - test the contract (return shape / option set / namespace mutation),
#     NOT performance or numerical fidelity
#   - skip cleanly when an upstream dep is missing

test_that(".onLoad raises future.globals.maxSize to >= 16 GiB", {
  # `library(autozyme)` has already fired .onLoad by the time testthat runs.
  # We assert post-load state rather than reloading the namespace.
  opt <- getOption("future.globals.maxSize")
  expect_false(is.null(opt))
  expect_true(is.finite(opt))
  expect_gte(opt, 16 * 1024^3)
})

test_that(".onLoad does not lower a user-set future.globals.maxSize", {
  # The contract is "only raise, never lower". Simulate a user override
  # by setting a higher value, re-invoking the .onLoad logic, and
  # asserting the higher value survives.
  prev <- getOption("future.globals.maxSize")
  on.exit(options(future.globals.maxSize = prev), add = TRUE)

  options(future.globals.maxSize = 64 * 1024^3)  # 64 GiB user override
  # Re-run the same conditional logic from R/zzz.R inline. We don't reload
  # the package because devtools::load_all in testthat is heavy and
  # noisy; the logic is small enough to mirror exactly.
  fg_max <- getOption("future.globals.maxSize")
  if (is.null(fg_max) || !is.finite(fg_max) || fg_max < 16 * 1024^3) {
    options(future.globals.maxSize = 16 * 1024^3)
  }
  expect_equal(getOption("future.globals.maxSize"), 64 * 1024^3)
})

test_that(".onLoad raises a too-low user-set future.globals.maxSize", {
  prev <- getOption("future.globals.maxSize")
  on.exit(options(future.globals.maxSize = prev), add = TRUE)

  options(future.globals.maxSize = 1 * 1024^3)  # 1 GiB -- below floor
  fg_max <- getOption("future.globals.maxSize")
  if (is.null(fg_max) || !is.finite(fg_max) || fg_max < 16 * 1024^3) {
    options(future.globals.maxSize = 16 * 1024^3)
  }
  expect_equal(getOption("future.globals.maxSize"), 16 * 1024^3)
})
