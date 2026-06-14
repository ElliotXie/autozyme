# Unit tests for R/threads.R. Complements test-auto-threads.R (which covers the
# auto_threads priority ladder) by exercising set_threads propagation, the
# capture/apply/restore state machine, and the .az_thread_scope helper. We
# assert RELATIONALLY (>=1, <= cap, monotone in cap) — never absolute core
# counts, which are host-dependent and brittle.

# ---- set_threads: env-var + option propagation -----------------------------

test_that("set_threads writes every thread env var and the R option", {
  ns <- asNamespace("autozyme")
  vars <- ns$.thread_env_vars
  old <- Sys.getenv(vars, unset = NA_character_)
  old_opt <- getOption("autozyme.threads")
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
    options(autozyme.threads = old_opt)
  }, add = TRUE)

  ret <- set_threads(4)
  expect_identical(ret, 4L)
  for (v in vars) expect_identical(Sys.getenv(v), "4")
  expect_identical(getOption("autozyme.threads"), 4L)
})

test_that("set_threads returns its integer invisibly", {
  old_opt <- getOption("autozyme.threads")
  on.exit(options(autozyme.threads = old_opt), add = TRUE)
  expect_invisible(set_threads(2))
})

test_that("set_threads coerces numeric input to integer", {
  old_opt <- getOption("autozyme.threads")
  on.exit(options(autozyme.threads = old_opt), add = TRUE)
  set_threads(3.0)
  expect_identical(getOption("autozyme.threads"), 3L)
})

test_that("set_threads rejects non-positive / NA / multi-element input", {
  expect_error(set_threads(0))
  expect_error(set_threads(-2))
  expect_error(set_threads(NA_integer_))
  expect_error(set_threads(c(1L, 2L)))
})

# ---- thread-state capture / apply / restore --------------------------------

test_that(".thread_env_vars covers the BLAS/OpenMP knobs", {
  ns <- asNamespace("autozyme")
  expect_true(all(c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS") %in% ns$.thread_env_vars))
})

test_that(".az_apply_thread_count sets all env vars to n", {
  ns <- asNamespace("autozyme")
  vars <- ns$.thread_env_vars
  old <- Sys.getenv(vars, unset = NA_character_)
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
  }, add = TRUE)
  ns$.az_apply_thread_count(5L)
  for (v in vars) expect_identical(Sys.getenv(v), "5")
})

test_that(".az_capture_thread_state then restore round-trips env + option", {
  ns <- asNamespace("autozyme")
  vars <- ns$.thread_env_vars
  old <- Sys.getenv(vars, unset = NA_character_)
  old_opt <- getOption("autozyme.threads")
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
    options(autozyme.threads = old_opt)
  }, add = TRUE)

  # Establish a known state, capture it, perturb, then restore.
  Sys.setenv(OMP_NUM_THREADS = "6")
  options(autozyme.threads = 6L)
  state <- ns$.az_capture_thread_state()
  ns$.az_apply_thread_count(1L)
  options(autozyme.threads = 1L)
  expect_identical(Sys.getenv("OMP_NUM_THREADS"), "1")

  ns$.az_restore_thread_state(state)
  expect_identical(Sys.getenv("OMP_NUM_THREADS"), "6")
  expect_identical(getOption("autozyme.threads"), 6L)
})

test_that(".az_restore_thread_state unsets vars that were unset at capture", {
  ns <- asNamespace("autozyme")
  vars <- ns$.thread_env_vars
  old <- Sys.getenv(vars, unset = NA_character_)
  old_opt <- getOption("autozyme.threads")
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
    options(autozyme.threads = old_opt)
  }, add = TRUE)

  Sys.unsetenv("OMP_NUM_THREADS")
  state <- ns$.az_capture_thread_state()
  ns$.az_apply_thread_count(2L)
  expect_identical(Sys.getenv("OMP_NUM_THREADS"), "2")
  ns$.az_restore_thread_state(state)
  # Was NA (unset) at capture -> should be unset again ("" from getenv).
  expect_identical(Sys.getenv("OMP_NUM_THREADS", unset = ""), "")
})

test_that(".az_thread_scope restores OMP env after the block", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("OMP_NUM_THREADS", unset = NA_character_)
  old_opt <- getOption("autozyme.threads")
  on.exit({
    if (is.na(old)) Sys.unsetenv("OMP_NUM_THREADS")
    else Sys.setenv(OMP_NUM_THREADS = old)
    options(autozyme.threads = old_opt)
  }, add = TRUE)

  Sys.setenv(OMP_NUM_THREADS = "8")
  ns$.az_thread_scope(1L, {
    expect_identical(Sys.getenv("OMP_NUM_THREADS"), "1")
  })
  expect_identical(Sys.getenv("OMP_NUM_THREADS"), "8")
})

test_that(".az_thread_scope returns the value of expr", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("OMP_NUM_THREADS", unset = NA_character_)
  old_opt <- getOption("autozyme.threads")
  on.exit({
    if (is.na(old)) Sys.unsetenv("OMP_NUM_THREADS")
    else Sys.setenv(OMP_NUM_THREADS = old)
    options(autozyme.threads = old_opt)
  }, add = TRUE)
  expect_equal(ns$.az_thread_scope(1L, 41 + 1), 42)
})

# ---- auto_threads relational invariants (host-independent) ------------------

test_that("auto_threads default path is >=1, <=16, and monotone non-increasing in cap", {
  Sys.unsetenv("AUTOZYME_THREADS")
  old_opt <- options(autozyme.threads = NULL)
  on.exit(options(old_opt), add = TRUE)

  base <- auto_threads()
  expect_true(is.integer(base))
  expect_gte(base, 1L)
  expect_lte(base, 16L)

  # Monotone: a smaller cap never yields more threads than a larger cap.
  prev <- Inf
  for (cap in c(16L, 8L, 4L, 2L, 1L)) {
    n <- auto_threads(cap = cap)
    expect_gte(n, 1L)
    expect_lte(n, cap)
    expect_lte(n, prev)
    prev <- n
  }
})

test_that("auto_threads never exceeds its cap on the default path", {
  Sys.unsetenv("AUTOZYME_THREADS")
  old_opt <- options(autozyme.threads = NULL)
  on.exit(options(old_opt), add = TRUE)
  for (cap in 1:6) {
    expect_lte(auto_threads(cap = cap), cap)
  }
})
