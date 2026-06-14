# Unit tests for R/zzz.R .onLoad side effects. The package is already loaded
# by the time tests run, so we assert on the post-load process state plus
# re-invoke .onLoad in controlled env states to cover its branches.

test_that("future.globals.maxSize was raised by .onLoad to at least 16 GiB", {
  fg <- getOption("future.globals.maxSize")
  expect_false(is.null(fg))
  expect_true(is.finite(fg))
  expect_gte(fg, 16 * 1024^3)
})

test_that(".onLoad is a no-op (returns NULL) when AUTOZYME_DISABLE is set", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("AUTOZYME_DISABLE", unset = NA_character_)
  old_opt <- getOption("future.globals.maxSize")
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_DISABLE")
    else Sys.setenv(AUTOZYME_DISABLE = old)
    options(future.globals.maxSize = old_opt)
  }, add = TRUE)

  Sys.setenv(AUTOZYME_DISABLE = "1")
  # Should short-circuit before touching options / emitting a startup message.
  expect_null(ns$.onLoad("lib", "autozyme"))
})

test_that(".onLoad does not lower an existing larger future.globals.maxSize", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("AUTOZYME_DISABLE", unset = NA_character_)
  old_opt <- getOption("future.globals.maxSize")
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_DISABLE")
    else Sys.setenv(AUTOZYME_DISABLE = old)
    options(future.globals.maxSize = old_opt)
  }, add = TRUE)

  Sys.unsetenv("AUTOZYME_DISABLE")
  big <- 999 * 1024^3
  options(future.globals.maxSize = big)
  suppressMessages(suppressWarnings(ns$.onLoad("lib", "autozyme")))
  # Never lowers a user override.
  expect_equal(getOption("future.globals.maxSize"), big)
})

test_that(".onLoad raises a too-small future.globals.maxSize back to 16 GiB", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("AUTOZYME_DISABLE", unset = NA_character_)
  old_opt <- getOption("future.globals.maxSize")
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_DISABLE")
    else Sys.setenv(AUTOZYME_DISABLE = old)
    options(future.globals.maxSize = old_opt)
  }, add = TRUE)

  Sys.unsetenv("AUTOZYME_DISABLE")
  options(future.globals.maxSize = 100 * 1024^2)  # 100 MB, too small
  suppressMessages(suppressWarnings(ns$.onLoad("lib", "autozyme")))
  expect_gte(getOption("future.globals.maxSize"), 16 * 1024^3)
})
