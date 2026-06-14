# Unit tests for R/python_env.R helpers NOT already covered by
# test-python-env.R (which exercises .az_py_probe / .az_py_find_existing /
# .az_py_bind discovery order / warmup / install_python_deps wiring). Here we
# add: env-name + package list resolution, the session bind memo short-circuit,
# reticulate-absent graceful paths, and probe edge cases. Everything stays
# offline; no real Python is initialized.

# ---- .az_py_env_name -------------------------------------------------------

test_that(".az_py_env_name defaults to r-autozyme and honors AUTOZYME_PY_ENV", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("AUTOZYME_PY_ENV", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_PY_ENV")
    else Sys.setenv(AUTOZYME_PY_ENV = old)
  }, add = TRUE)

  Sys.unsetenv("AUTOZYME_PY_ENV")
  expect_identical(ns$.az_py_env_name(), "r-autozyme")
  Sys.setenv(AUTOZYME_PY_ENV = "my-custom-env")
  expect_identical(ns$.az_py_env_name(), "my-custom-env")
})

# ---- .az_py_packages -------------------------------------------------------

test_that(".az_py_packages lists numpy, scipy, threadpoolctl", {
  ns <- asNamespace("autozyme")
  pk <- ns$.az_py_packages()
  expect_type(pk, "character")
  expect_setequal(pk, c("numpy", "scipy", "threadpoolctl"))
})

# ---- .az_py_probe edge cases (no fake scripts needed) ----------------------

test_that(".az_py_probe is FALSE for empty / missing interpreter paths", {
  ns <- asNamespace("autozyme")
  expect_false(ns$.az_py_probe(""))
  expect_false(ns$.az_py_probe(file.path(tempdir(), "no_such_python_xyz")))
})

# ---- .az_py_state memo short-circuit ---------------------------------------

test_that(".az_py_bind returns the cached memo without re-probing", {
  ns <- asNamespace("autozyme")
  st <- ns$.az_py_state                # capture env object (mutable by ref)
  old_bound <- st$bound
  old_err   <- st$err
  on.exit({ st$bound <- old_bound; st$err <- old_err }, add = TRUE)

  st$bound <- TRUE
  expect_true(ns$.az_py_bind())        # short-circuits on the cached TRUE
  st$bound <- FALSE
  expect_false(ns$.az_py_bind())       # short-circuits on the cached FALSE
})

# ---- reticulate-absent graceful behavior -----------------------------------

test_that(".az_py_bind_impl reports failure gracefully when reticulate absent", {
  ns <- asNamespace("autozyme")
  st <- ns$.az_py_state
  old_err <- st$err
  on.exit(st$err <- old_err, add = TRUE)

  # Mock the namespace probe so the function takes the "no reticulate" branch
  # regardless of what's installed on the runner.
  testthat::local_mocked_bindings(
    requireNamespace = function(pkg, ...) {
      if (identical(pkg, "reticulate")) return(FALSE)
      base::requireNamespace(pkg, ...)
    },
    .package = "base")

  expect_false(ns$.az_py_bind_impl())
  expect_match(st$err, "reticulate")
})

test_that("install_python_deps errors with a clear message when reticulate absent", {
  testthat::local_mocked_bindings(
    requireNamespace = function(pkg, ...) {
      if (identical(pkg, "reticulate")) return(FALSE)
      base::requireNamespace(pkg, ...)
    },
    .package = "base")
  expect_error(install_python_deps(), "reticulate")
})

test_that(".az_py_warmup_or_notify nudges (no error) when reticulate absent", {
  ns <- asNamespace("autozyme")
  testthat::local_mocked_bindings(
    requireNamespace = function(pkg, ...) {
      if (identical(pkg, "reticulate")) return(FALSE)
      base::requireNamespace(pkg, ...)
    },
    .package = "base")
  expect_message(
    res <- ns$.az_py_warmup_or_notify("seurat", "PCA"),
    "reticulate")
  expect_false(res)
})

# ---- .az_py_find_existing returns "" when nothing usable -------------------

test_that(".az_py_find_existing returns empty string when no candidate probes pass", {
  ns <- asNamespace("autozyme")
  skip_on_os("windows")
  # Neutralize ambient discovery: no CONDA_PREFIX, empty PATH, and force every
  # probe to fail. reticulate's discovery (if installed) is mocked to NULL.
  oldcp   <- Sys.getenv("CONDA_PREFIX", unset = NA_character_)
  oldpath <- Sys.getenv("PATH", unset = NA_character_)
  on.exit({
    if (is.na(oldcp)) Sys.unsetenv("CONDA_PREFIX") else Sys.setenv(CONDA_PREFIX = oldcp)
    if (!is.na(oldpath)) Sys.setenv(PATH = oldpath)
  }, add = TRUE)
  Sys.unsetenv("CONDA_PREFIX")
  Sys.setenv(PATH = "")

  # Make every probe report "unusable" and (if reticulate present) discovery empty.
  local_mocked_bindings(.az_py_probe = function(python) FALSE,
                        .package = "autozyme")
  if (requireNamespace("reticulate", quietly = TRUE)) {
    local_mocked_bindings(py_discover_config = function(...) NULL,
                          .package = "reticulate")
  }
  expect_identical(ns$.az_py_find_existing(), "")
})
