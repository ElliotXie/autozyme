# Unit tests for R/intercept_probe.R pure pieces: key formatting, counter
# bookkeeping, wrap behavior, JSON emission, and the env-gated
# install_from_env entry point. None of these need an upstream.
#
# NOTE on locked bindings: in the *installed* package the namespace bindings
# `.zyme_intercept` and `.zyme_registry` are locked, so `ns$.zyme_intercept$x
# <- v` (a replacement form on `ns`) errors with "cannot change value of
# locked binding". We work around this by capturing the *environment object*
# into a local variable first and mutating its contents by reference (envs
# are mutable; only the namespace binding NAME is locked).
#
# .zyme_intercept_install_for()'s real-registration path is therefore NOT
# coverable against the installed package: it calls
# `assign(".zyme_registry", registry, envir = asNamespace("autozyme"))`,
# which rebinds the locked `.zyme_registry` symbol and hard-errors. We cover
# its unknown-patch early return instead and document the rest.

test_that(".zyme_intercept_key joins upstream::attr", {
  ns <- asNamespace("autozyme")
  expect_identical(ns$.zyme_intercept_key("Seurat", "NormalizeData"),
                   "Seurat::NormalizeData")
})

test_that(".zyme_intercept_inc increments from 0 and accumulates", {
  ns <- asNamespace("autozyme")
  ic <- ns$.zyme_intercept            # capture env object (mutable by ref)
  old <- ic$counts
  on.exit(ic$counts <- old, add = TRUE)
  ic$counts <- list()

  ns$.zyme_intercept_inc("pkg::fn")
  expect_identical(ic$counts[["pkg::fn"]], 1L)
  ns$.zyme_intercept_inc("pkg::fn")
  ns$.zyme_intercept_inc("pkg::fn")
  expect_identical(ic$counts[["pkg::fn"]], 3L)
  expect_null(ic$counts[["other::fn"]])
})

test_that(".zyme_intercept_wrap counts each call and forwards args/results", {
  ns <- asNamespace("autozyme")
  ic <- ns$.zyme_intercept
  old <- ic$counts
  on.exit(ic$counts <- old, add = TRUE)
  ic$counts <- list()

  base <- function(a, b) a + b
  wrapped <- ns$.zyme_intercept_wrap(base, "math::add")
  expect_identical(attr(wrapped, ".zyme_intercept_key"), "math::add")
  expect_equal(wrapped(2, 3), 5)
  expect_equal(wrapped(10, 1), 11)
  expect_identical(ic$counts[["math::add"]], 2L)
})

test_that(".zyme_intercept_write emits {} for empty counts", {
  ns <- asNamespace("autozyme")
  ic <- ns$.zyme_intercept
  old <- ic$counts
  old_env <- Sys.getenv("ZYME_INTERCEPT_OUT", unset = NA_character_)
  out <- tempfile(fileext = ".json")
  on.exit({
    ic$counts <- old
    if (is.na(old_env)) Sys.unsetenv("ZYME_INTERCEPT_OUT")
    else Sys.setenv(ZYME_INTERCEPT_OUT = old_env)
    unlink(out)
  }, add = TRUE)

  ic$counts <- list()
  Sys.setenv(ZYME_INTERCEPT_OUT = out)
  ns$.zyme_intercept_write()
  expect_identical(trimws(paste(readLines(out, warn = FALSE), collapse = "")), "{}")
})

test_that(".zyme_intercept_write emits sorted JSON-shaped counts", {
  ns <- asNamespace("autozyme")
  ic <- ns$.zyme_intercept
  old <- ic$counts
  old_env <- Sys.getenv("ZYME_INTERCEPT_OUT", unset = NA_character_)
  out <- tempfile(fileext = ".json")
  on.exit({
    ic$counts <- old
    if (is.na(old_env)) Sys.unsetenv("ZYME_INTERCEPT_OUT")
    else Sys.setenv(ZYME_INTERCEPT_OUT = old_env)
    unlink(out)
  }, add = TRUE)

  ic$counts <- list("z::b" = 2L, "a::a" = 5L)
  Sys.setenv(ZYME_INTERCEPT_OUT = out)
  ns$.zyme_intercept_write()
  blob <- paste(readLines(out, warn = FALSE), collapse = "\n")
  # Keys sorted: a::a before z::b.
  expect_lt(regexpr("a::a", blob, fixed = TRUE),
            regexpr("z::b", blob, fixed = TRUE))
  expect_match(blob, '"a::a": 5')
  expect_match(blob, '"z::b": 2')
})

test_that(".zyme_intercept_write escapes embedded double-quotes in keys", {
  ns <- asNamespace("autozyme")
  ic <- ns$.zyme_intercept
  old <- ic$counts
  old_env <- Sys.getenv("ZYME_INTERCEPT_OUT", unset = NA_character_)
  out <- tempfile(fileext = ".json")
  on.exit({
    ic$counts <- old
    if (is.na(old_env)) Sys.unsetenv("ZYME_INTERCEPT_OUT")
    else Sys.setenv(ZYME_INTERCEPT_OUT = old_env)
    unlink(out)
  }, add = TRUE)

  ic$counts <- list(`pkg::weird"name` = 1L)
  Sys.setenv(ZYME_INTERCEPT_OUT = out)
  ns$.zyme_intercept_write()
  blob <- paste(readLines(out, warn = FALSE), collapse = "\n")
  # B11 fix: the embedded `"` is escaped with a SINGLE backslash (\"), which is
  # the valid JSON escape, so the substring `weird\"name` (one backslash + quote)
  # is present and the double-backslash form is NOT.
  expect_true(grepl('weird\\"name', blob, fixed = TRUE))        # weird + \ + " + name
  expect_false(grepl('weird\\\\"name', blob, fixed = TRUE))     # not weird + \\ + "
  # The emitted blob must be valid JSON.
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    parsed <- jsonlite::fromJSON(blob)
    expect_equal(unname(parsed[["pkg::weird\"name"]]), 1L)
  }
})

test_that(".zyme_intercept_write is a no-op when ZYME_INTERCEPT_OUT is unset", {
  ns <- asNamespace("autozyme")
  old_env <- Sys.getenv("ZYME_INTERCEPT_OUT", unset = NA_character_)
  on.exit({
    if (is.na(old_env)) Sys.unsetenv("ZYME_INTERCEPT_OUT")
    else Sys.setenv(ZYME_INTERCEPT_OUT = old_env)
  }, add = TRUE)
  Sys.unsetenv("ZYME_INTERCEPT_OUT")
  expect_silent(ns$.zyme_intercept_write())
})

test_that(".zyme_intercept_install_from_env is FALSE without the gate env", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("ZYME_INSTRUMENT_INTERCEPTS", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("ZYME_INSTRUMENT_INTERCEPTS")
    else Sys.setenv(ZYME_INSTRUMENT_INTERCEPTS = old)
  }, add = TRUE)
  Sys.unsetenv("ZYME_INSTRUMENT_INTERCEPTS")
  expect_false(ns$.zyme_intercept_install_from_env())
})

test_that(".zyme_intercept_install_for is a silent no-op for an unknown patch", {
  ns <- asNamespace("autozyme")
  # No registry entry for this fabricated name -> returns invisibly, no error.
  expect_silent(ns$.zyme_intercept_install_for("definitely_not_a_patch_xyz"))
})
