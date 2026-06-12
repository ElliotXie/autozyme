# Save+restore env/option state so tests don't leak.
# Note: an earlier test (set_threads) sets autozyme.threads option;
# we explicitly NULL it at start of each test that exercises the default path.

test_that("auto_threads: AUTOZYME_THREADS env var wins", {
  old_opt <- options(autozyme.threads = NULL)
  Sys.setenv(AUTOZYME_THREADS = "4")
  on.exit({
    Sys.unsetenv("AUTOZYME_THREADS")
    options(old_opt)
  }, add = TRUE)

  expect_equal(auto_threads(), 4L)
  expect_equal(auto_threads(cap = 2L), 4L)  # env wins over cap
  expect_equal(auto_threads(cap = 100L), 4L)
})

test_that("auto_threads: autozyme.threads option wins over cap", {
  Sys.unsetenv("AUTOZYME_THREADS")
  old_opt <- options(autozyme.threads = 5L)
  on.exit(options(old_opt), add = TRUE)

  expect_equal(auto_threads(), 5L)
  expect_equal(auto_threads(cap = 2L), 5L)
})

test_that("auto_threads: env wins over option", {
  Sys.setenv(AUTOZYME_THREADS = "7")
  old_opt <- options(autozyme.threads = 3L)
  on.exit({
    Sys.unsetenv("AUTOZYME_THREADS")
    options(old_opt)
  }, add = TRUE)

  expect_equal(auto_threads(), 7L)
})

test_that("auto_threads: cap applied when no override", {
  Sys.unsetenv("AUTOZYME_THREADS")
  old_opt <- options(autozyme.threads = NULL)
  on.exit(options(old_opt), add = TRUE)

  n <- auto_threads(cap = 2L)
  expect_lte(n, 2L)
  expect_gte(n, 1L)
})

test_that("auto_threads: hardware default with no cap and no override", {
  Sys.unsetenv("AUTOZYME_THREADS")
  old_opt <- options(autozyme.threads = NULL)
  on.exit(options(old_opt), add = TRUE)

  n <- auto_threads()
  expect_true(is.integer(n))
  expect_gte(n, 1L)
  expect_lte(n, 16L)  # hard ceiling
})

test_that("auto_threads: invalid env var falls through to next priority", {
  Sys.setenv(AUTOZYME_THREADS = "not-a-number")
  old_opt <- options(autozyme.threads = 3L)
  on.exit({
    Sys.unsetenv("AUTOZYME_THREADS")
    options(old_opt)
  }, add = TRUE)

  expect_equal(auto_threads(), 3L)  # falls to option
})

test_that("auto_threads: zero/negative env var falls through", {
  Sys.setenv(AUTOZYME_THREADS = "0")
  old_opt <- options(autozyme.threads = 4L)
  on.exit({
    Sys.unsetenv("AUTOZYME_THREADS")
    options(old_opt)
  }, add = TRUE)

  expect_equal(auto_threads(), 4L)
})

test_that("auto_threads: always returns at least 1", {
  Sys.unsetenv("AUTOZYME_THREADS")
  old_opt <- options(autozyme.threads = NULL)
  on.exit(options(old_opt), add = TRUE)

  n <- auto_threads(cap = 0L)   # invalid cap ignored, default kicks in
  expect_gte(n, 1L)

  n <- auto_threads(cap = -5L)  # invalid cap ignored
  expect_gte(n, 1L)
})

test_that("auto_threads: hard ceiling at 16 for default path", {
  Sys.unsetenv("AUTOZYME_THREADS")
  old_opt <- options(autozyme.threads = NULL)
  on.exit(options(old_opt), add = TRUE)

  # Default cannot exceed 16 even on machines with >16 physical cores.
  n <- auto_threads(cap = 1000L)
  expect_lte(n, 16L)
})

test_that("auto_threads: env override BYPASSES the 16-ceiling for testing", {
  # CI thread matrix may want to test with high counts; env should pass through.
  Sys.setenv(AUTOZYME_THREADS = "32")
  old_opt <- options(autozyme.threads = NULL)
  on.exit({
    Sys.unsetenv("AUTOZYME_THREADS")
    options(old_opt)
  }, add = TRUE)

  expect_equal(auto_threads(), 32L)
})
