# Attach the package so test files may call exported helpers (auto_threads,
# set_threads, ...) unqualified. The documented `R CMD check` entry point
# (tests/testthat.R -> test_check) attaches autozyme automatically, but a bare
# `testthat::test_dir("tests/testthat")` only *loads* it (via requireNamespace
# below) without attaching, so those calls would fail with "could not find
# function". Guarded + idempotent: a no-op when autozyme is already attached
# (R CMD check, devtools::test()).
if (requireNamespace("autozyme", quietly = TRUE)) {
  suppressPackageStartupMessages(library(autozyme))
}

# testthat runs setup-*.R once before any test- file. Activate every patch
# whose upstream is installed so per-API contract tests can pass `zyme =
# FALSE` (which only the patched function accepts) without each test file
# having to handle activation itself.
#
# `try(silent = TRUE)` because some patches may legitimately fail to
# activate on a given host (e.g. sarsen's registration-gap bug); the
# universal activation smoke covers those cases separately.

if (!identical(Sys.getenv("AUTOZYME_TEST_SKIP_AUTO_ACTIVATE"), "1") &&
    requireNamespace("autozyme", quietly = TRUE)) {
  .az_available <- tryCatch(autozyme::list_patches(),
                            error = function(e) character(0))
  for (.az_name in .az_available) {
    try(autozyme::activate(.az_name), silent = TRUE)
  }
  rm(.az_available, .az_name)
}
