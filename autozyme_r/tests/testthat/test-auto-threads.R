# Save+restore env/option state so tests don't leak.
# Note: an earlier test (set_threads) sets autozyme.threads option;
# we explicitly NULL it at start of each test that exercises the default path.
#
# auto_threads reads ZYME_THREADS > AUTOZYME_THREADS > OMP_NUM_THREADS (any one
# the attest harness sets), so default-path tests must clear all three to avoid
# a stray shell value pre-empting the resolver.

.thread_env_all <- c("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS")
.clear_thread_env <- function() Sys.unsetenv(.thread_env_all)

test_that("auto_threads: AUTOZYME_THREADS env var wins", {
  old_opt <- options(autozyme.threads = NULL)
  .clear_thread_env()
  Sys.setenv(AUTOZYME_THREADS = "4")
  on.exit({
    .clear_thread_env()
    options(old_opt)
  }, add = TRUE)

  expect_equal(auto_threads(), 4L)
  expect_equal(auto_threads(cap = 2L), 4L)  # env wins over cap
  expect_equal(auto_threads(cap = 100L), 4L)
})

test_that("auto_threads: ZYME_THREADS env var wins (harness primary knob)", {
  old_opt <- options(autozyme.threads = NULL)
  .clear_thread_env()
  Sys.setenv(ZYME_THREADS = "8")
  on.exit({
    .clear_thread_env()
    options(old_opt)
  }, add = TRUE)

  expect_equal(auto_threads(), 8L)
  expect_equal(auto_threads(cap = 2L), 8L)  # env wins over cap
})

test_that("auto_threads: OMP_NUM_THREADS env var honored", {
  old_opt <- options(autozyme.threads = NULL)
  .clear_thread_env()
  Sys.setenv(OMP_NUM_THREADS = "6")
  on.exit({
    .clear_thread_env()
    options(old_opt)
  }, add = TRUE)

  expect_equal(auto_threads(), 6L)
})

test_that("auto_threads: autozyme.threads option wins over cap", {
  .clear_thread_env()
  old_opt <- options(autozyme.threads = 5L)
  on.exit(options(old_opt), add = TRUE)

  expect_equal(auto_threads(), 5L)
  expect_equal(auto_threads(cap = 2L), 5L)
})

test_that("auto_threads: env wins over option", {
  old_opt <- options(autozyme.threads = 3L)
  .clear_thread_env()
  Sys.setenv(AUTOZYME_THREADS = "7")
  on.exit({
    .clear_thread_env()
    options(old_opt)
  }, add = TRUE)

  expect_equal(auto_threads(), 7L)
})

test_that("auto_threads: conservative default is 4 (no override)", {
  .clear_thread_env()
  old_opt <- options(autozyme.threads = NULL)
  on.exit(options(old_opt), add = TRUE)

  cores <- tryCatch(parallel::detectCores(logical = FALSE),
                    error = function(e) NA_integer_)
  if (is.na(cores) || cores < 1L) cores <- 1L
  # The floor of 4, clipped to the machine's physical core count.
  expect_equal(auto_threads(), min(4L, as.integer(cores)))
})

test_that("auto_threads: default=NULL scales to hardware", {
  .clear_thread_env()
  old_opt <- options(autozyme.threads = NULL)
  on.exit(options(old_opt), add = TRUE)

  cores <- tryCatch(parallel::detectCores(logical = FALSE),
                    error = function(e) NA_integer_)
  if (is.na(cores) || cores < 1L) cores <- 1L
  expected <- min(max(1L, as.integer(cores) - 1L), 16L)
  expect_equal(auto_threads(default = NULL), expected)
})

test_that("auto_threads: cap below the floor still bites", {
  .clear_thread_env()
  old_opt <- options(autozyme.threads = NULL)
  on.exit(options(old_opt), add = TRUE)

  n <- auto_threads(cap = 2L)
  expect_lte(n, 2L)
  expect_gte(n, 1L)
})

test_that("auto_threads: hardware default with no cap and no override", {
  .clear_thread_env()
  old_opt <- options(autozyme.threads = NULL)
  on.exit(options(old_opt), add = TRUE)

  n <- auto_threads()
  expect_true(is.integer(n))
  expect_gte(n, 1L)
  expect_lte(n, 16L)  # hard ceiling
})

test_that("auto_threads: invalid env var falls through to next priority", {
  old_opt <- options(autozyme.threads = 3L)
  .clear_thread_env()
  Sys.setenv(AUTOZYME_THREADS = "not-a-number")
  on.exit({
    .clear_thread_env()
    options(old_opt)
  }, add = TRUE)

  expect_equal(auto_threads(), 3L)  # falls to option
})

test_that("auto_threads: zero/negative env var falls through", {
  old_opt <- options(autozyme.threads = 4L)
  .clear_thread_env()
  Sys.setenv(AUTOZYME_THREADS = "0")
  on.exit({
    .clear_thread_env()
    options(old_opt)
  }, add = TRUE)

  expect_equal(auto_threads(), 4L)
})

test_that("auto_threads: always returns at least 1", {
  .clear_thread_env()
  old_opt <- options(autozyme.threads = NULL)
  on.exit(options(old_opt), add = TRUE)

  n <- auto_threads(cap = 0L)   # invalid cap ignored, default kicks in
  expect_gte(n, 1L)

  n <- auto_threads(cap = -5L)  # invalid cap ignored
  expect_gte(n, 1L)
})

test_that("auto_threads: hard ceiling at 16 for default path", {
  .clear_thread_env()
  old_opt <- options(autozyme.threads = NULL)
  on.exit(options(old_opt), add = TRUE)

  # Default cannot exceed 16 even on machines with >16 physical cores.
  n <- auto_threads(cap = 1000L)
  expect_lte(n, 16L)
})

test_that("auto_threads: env override BYPASSES the 16-ceiling for testing", {
  # CI thread matrix may want to test with high counts; env should pass through.
  old_opt <- options(autozyme.threads = NULL)
  .clear_thread_env()
  Sys.setenv(AUTOZYME_THREADS = "32")
  on.exit({
    .clear_thread_env()
    options(old_opt)
  }, add = TRUE)

  expect_equal(auto_threads(), 32L)
})
