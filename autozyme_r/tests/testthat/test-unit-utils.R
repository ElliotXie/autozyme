# Unit tests for R/utils.R pure helpers: .zyme_mclapply (lapply fallback path)
# and resolve_dataset_path. Neither needs an upstream package.

test_that(".zyme_mclapply falls back to lapply when mc.cores <= 1", {
  ns <- asNamespace("autozyme")
  expect_equal(ns$.zyme_mclapply(1:5, function(x) x * 2L, mc.cores = 1L),
               lapply(1:5, function(x) x * 2L))
  # mc.cores = 1 default
  expect_equal(ns$.zyme_mclapply(letters[1:3], toupper),
               lapply(letters[1:3], toupper))
})

test_that(".zyme_mclapply treats non-integer mc.cores as serial", {
  ns <- asNamespace("autozyme")
  # NA-coercible mc.cores -> NA_integer_ -> serial lapply (no crash).
  expect_equal(suppressWarnings(ns$.zyme_mclapply(1:3, identity,
                                                  mc.cores = "bogus")),
               lapply(1:3, identity))
})

test_that(".zyme_mclapply forwards extra args to FUN", {
  ns <- asNamespace("autozyme")
  expect_equal(ns$.zyme_mclapply(list(c(1, NA, 3)), sum, na.rm = TRUE,
                                 mc.cores = 1L),
               list(4))
})

test_that(".zyme_mclapply on Unix with mc.cores>1 still matches lapply", {
  skip_on_os("windows")
  ns <- asNamespace("autozyme")
  # Forking path; result must be order-preserving and equal to serial.
  expect_equal(ns$.zyme_mclapply(1:6, function(x) x + 100L, mc.cores = 2L),
               lapply(1:6, function(x) x + 100L))
})

test_that("resolve_dataset_path finds a relative path under task_dir", {
  td <- tempfile("az_task_")
  dir.create(file.path(td, "data"), recursive = TRUE)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  f <- file.path(td, "data", "inputs.rds")
  saveRDS(1, f)

  got <- resolve_dataset_path(td, "data/inputs.rds")
  expect_true(file.exists(got))
  expect_equal(normalizePath(got), normalizePath(f))
})

test_that("resolve_dataset_path strips a leading ./ segment", {
  td <- tempfile("az_task_")
  dir.create(file.path(td, "data"), recursive = TRUE)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  f <- file.path(td, "data", "x.rds")
  saveRDS(1, f)
  got <- resolve_dataset_path(td, "./data/x.rds")
  expect_equal(normalizePath(got), normalizePath(f))
})

test_that("resolve_dataset_path falls back to task_dir/data/<basename>", {
  td <- tempfile("az_task_")
  dir.create(file.path(td, "data"), recursive = TRUE)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  f <- file.path(td, "data", "stale_name.rds")
  saveRDS(1, f)
  # Raw path points to a stale absolute-ish location; basename rescue finds it.
  got <- resolve_dataset_path(td, "some/old/layout/stale_name.rds")
  expect_equal(normalizePath(got), normalizePath(f))
})

test_that("resolve_dataset_path errors when nothing resolves", {
  td <- tempfile("az_task_")
  dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  expect_error(resolve_dataset_path(td, "data/missing.rds"),
               "could not resolve dataset path")
})
