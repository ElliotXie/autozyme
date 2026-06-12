# Tests for the Python-env discovery layer (R/python_env.R).
#
# The hard part of this code is *which interpreter we pick and when* -- and that
# must be testable without a real Python or numpy on the runner. We drive the
# probe with FAKE interpreter scripts that exit 0 ("imports numpy+scipy fine")
# or 1 ("missing"), and mock reticulate's binders to record the choice. The
# shell-script fakes are unix-only; the logic under test is platform-shared, so
# we skip the fake-driven cases on Windows and rely on Linux/macOS CI for them.

.make_fake_python <- function(dir, name, exit_code) {
  p <- file.path(dir, name)
  writeLines(c("#!/bin/sh", sprintf("exit %d", exit_code)), p)
  Sys.chmod(p, mode = "0755")
  p
}

# ── .az_py_probe ────────────────────────────────────────────────────────────

test_that(".az_py_probe is TRUE only for an interpreter that imports cleanly", {
  skip_on_os("windows")
  d <- file.path(tempfile("azpy")); dir.create(d)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)

  good <- .make_fake_python(d, "py_good", 0L)
  bad  <- .make_fake_python(d, "py_bad",  1L)

  expect_true(.az_py_probe(good))
  expect_false(.az_py_probe(bad))
  expect_false(.az_py_probe(file.path(d, "does_not_exist")))
  expect_false(.az_py_probe(""))
})

# ── .az_py_find_existing ────────────────────────────────────────────────────

test_that(".az_py_find_existing picks an existing usable interpreter (CONDA_PREFIX)", {
  skip_on_os("windows")
  d <- file.path(tempfile("azconda")); dir.create(file.path(d, "bin"), recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  .make_fake_python(file.path(d, "bin"), "python", 0L)  # CONDA_PREFIX/bin/python

  old <- Sys.getenv("CONDA_PREFIX", unset = NA)
  Sys.setenv(CONDA_PREFIX = d)
  on.exit({ if (is.na(old)) Sys.unsetenv("CONDA_PREFIX") else Sys.setenv(CONDA_PREFIX = old) },
          add = TRUE)

  # CONDA_PREFIX/bin/python is the first candidate, so it wins regardless of what
  # else is on PATH / discovered.
  expect_identical(.az_py_find_existing(), file.path(d, "bin", "python"))
})

# ── .az_py_bind: discovery order + never commit to a numpy-less interpreter ──

test_that(".az_py_bind honors AUTOZYME_PYTHON and only binds a probe-passing python", {
  skip_on_os("windows")
  skip_if_not_installed("reticulate")
  d <- file.path(tempfile("azbind")); dir.create(d)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  good <- .make_fake_python(d, "py_good", 0L)

  # Record what gets bound; stub import so no real Python is touched.
  bound <- NULL
  testthat::local_mocked_bindings(
    py_available = function(initialize = FALSE) FALSE,
    use_python   = function(python, required = TRUE) { bound <<- python; invisible() },
    virtualenv_exists = function(...) FALSE,
    import       = function(...) NULL,
    .package = "reticulate")

  # Reset the session memo so bind actually runs.
  .az_py_state$bound <- NA; .az_py_state$err <- NA_character_
  on.exit({ .az_py_state$bound <- NA; .az_py_state$err <- NA_character_ }, add = TRUE)

  old <- Sys.getenv("AUTOZYME_PYTHON", unset = NA)
  Sys.setenv(AUTOZYME_PYTHON = good)
  on.exit({ if (is.na(old)) Sys.unsetenv("AUTOZYME_PYTHON") else Sys.setenv(AUTOZYME_PYTHON = old) },
          add = TRUE)

  expect_true(.az_py_bind())
  expect_identical(bound, good)            # bound the override
  expect_true(isTRUE(.az_py_state$bound))  # memoized
})

test_that(".az_py_bind never binds a numpy-less interpreter and reports failure", {
  skip_on_os("windows")
  skip_if_not_installed("reticulate")
  d <- file.path(tempfile("azbindbad")); dir.create(d)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  bad <- .make_fake_python(d, "py_bad", 1L)  # probe fails

  bound <- NULL
  testthat::local_mocked_bindings(
    py_available = function(initialize = FALSE) FALSE,
    use_python   = function(python, required = TRUE) { bound <<- python; invisible() },
    use_virtualenv = function(...) invisible(),
    virtualenv_exists = function(...) FALSE,
    py_discover_config = function(...) NULL,
    import       = function(...) NULL,
    .package = "reticulate")

  .az_py_state$bound <- NA; .az_py_state$err <- NA_character_
  on.exit({ .az_py_state$bound <- NA; .az_py_state$err <- NA_character_ }, add = TRUE)

  # Isolate so the numpy-less `bad` is the ONLY candidate: point the override at
  # it, and neutralize the ambient discovery (CONDA_PREFIX, PATH, reticulate's
  # py_discover_config is already mocked to NULL above) so find_existing can't
  # pick up a real interpreter on the test machine.
  oldp   <- Sys.getenv("AUTOZYME_PYTHON", unset = NA)
  oldcp  <- Sys.getenv("CONDA_PREFIX",    unset = NA)
  oldpath <- Sys.getenv("PATH",           unset = NA)
  Sys.setenv(AUTOZYME_PYTHON = bad); Sys.unsetenv("CONDA_PREFIX"); Sys.setenv(PATH = "")
  on.exit({
    if (is.na(oldp))   Sys.unsetenv("AUTOZYME_PYTHON") else Sys.setenv(AUTOZYME_PYTHON = oldp)
    if (!is.na(oldcp)) Sys.setenv(CONDA_PREFIX = oldcp)
    if (!is.na(oldpath)) Sys.setenv(PATH = oldpath)
  }, add = TRUE)

  expect_false(.az_py_bind())
  expect_null(bound)                       # never committed to the bad python
  expect_match(.az_py_state$err, "numpy", ignore.case = TRUE)
})

# ── warmup hook: non-interactive must inform, never prompt or error ──────────

test_that(".az_py_warmup_or_notify informs (no prompt) when nothing is usable", {
  skip_if_not_installed("reticulate")
  testthat::local_mocked_bindings(.az_py_bind = function() FALSE)
  old <- Sys.getenv("AUTOZYME_NO_PROMPT", unset = NA)
  Sys.setenv(AUTOZYME_NO_PROMPT = "1")  # belt-and-suspenders; tests are non-interactive
  on.exit({ if (is.na(old)) Sys.unsetenv("AUTOZYME_NO_PROMPT") else Sys.setenv(AUTOZYME_NO_PROMPT = old) },
          add = TRUE)

  expect_message(
    res <- withVisible(.az_py_warmup_or_notify("seurat", "PCA")),
    "install_python_deps")
  expect_false(res$value)
})

# ── install_python_deps: wiring (mock the heavy reticulate calls) ────────────

test_that("install_python_deps creates the env and resets the bind memo", {
  skip_if_not_installed("reticulate")
  created <- NULL
  testthat::local_mocked_bindings(
    py_discover_config = function(...) list(python = "/usr/bin/python3"),
    install_python     = function(...) stop("should not install_python when one exists"),
    virtualenv_create  = function(envname, version = NULL, packages = NULL, ...) {
      created <<- list(envname = envname, packages = packages); invisible() },
    .package = "reticulate")

  .az_py_state$bound <- FALSE  # pretend a prior failed bind
  on.exit({ .az_py_state$bound <- NA }, add = TRUE)

  expect_message(install_python_deps(), "ready")
  expect_identical(created$envname, "r-autozyme")
  expect_true(all(c("numpy", "scipy") %in% created$packages))
  expect_true(is.na(.az_py_state$bound))  # memo reset so new env is picked up
})
