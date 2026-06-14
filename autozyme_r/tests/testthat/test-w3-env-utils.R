# Wave-3 coverage push for the env/util R modules:
#   R/python_env.R, R/intercept_probe.R, R/native_blas.R, R/utils.R
#
# These tests target the line ranges left uncovered by the wave-1 files
# (test-unit-python-env.R / test-python-env.R / test-unit-intercept-probe.R /
# test-unit-native-blas.R / test-native-blas.R / test-unit-utils.R). They do
# NOT duplicate those; each block notes which gap it closes. Everything stays
# offline (no real Python initialized) and single-threaded-safe (no kernel
# thread-spinning); we drive native GEMM only with threads<=1 on tiny inputs.

ns <- asNamespace("autozyme")

# Restore-an-env-var helper used throughout.
.w3_with_env <- function(vars, code) {
  old <- vapply(names(vars), function(n) Sys.getenv(n, unset = NA_character_),
                character(1))
  on.exit({
    for (n in names(vars)) {
      v <- old[[n]]
      if (is.na(v)) Sys.unsetenv(n) else do.call(Sys.setenv, setNames(list(v), n))
    }
  }, add = TRUE)
  for (n in names(vars)) {
    v <- vars[[n]]
    if (is.na(v)) Sys.unsetenv(n) else do.call(Sys.setenv, setNames(list(v), n))
  }
  force(code)
}

# ============================================================================
# R/python_env.R
# ============================================================================

# ---- .az_py_find_existing: candidate assembly (lines 58-71) ----------------
# Wave-1 covered the CONDA_PREFIX/bin/python first-hit. Here we cover the rest
# of the candidate list: Sys.which(python3/python) + reticulate's discovered
# python + python_versions all flow into the probe loop, and the FIRST that
# probes TRUE wins (not necessarily CONDA_PREFIX).

test_that(".az_py_find_existing assembles PATH + reticulate candidates and first hit wins", {
  skip_on_os("windows")
  d <- tempfile("azfind"); dir.create(file.path(d, "bin"), recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  cprefix_py <- file.path(d, "bin", "python")
  writeLines(c("#!/bin/sh", "exit 0"), cprefix_py); Sys.chmod(cprefix_py, "0755")
  disc_py <- file.path(d, "disc_python")
  writeLines(c("#!/bin/sh", "exit 0"), disc_py); Sys.chmod(disc_py, "0755")

  .w3_with_env(list(CONDA_PREFIX = d), {
    # Mock reticulate discovery so the python/python_versions branches (67-68)
    # contribute candidates; only the disc_py probes TRUE.
    if (requireNamespace("reticulate", quietly = TRUE)) {
      local_mocked_bindings(
        py_discover_config = function(...) list(python = disc_py,
                                                python_versions = c(disc_py)),
        .package = "reticulate")
    }
    # Probe: only the CONDA_PREFIX python passes. Since it is the first
    # candidate, it wins -> exercises the loop's early return (line 71).
    local_mocked_bindings(.az_py_probe = function(python) identical(python, cprefix_py),
                          .package = "autozyme")
    expect_identical(ns$.az_py_find_existing(), cprefix_py)
  })
})

test_that(".az_py_find_existing picks a reticulate-discovered python when CONDA/PATH fail", {
  skip_on_os("windows")
  skip_if_not_installed("reticulate")
  d <- tempfile("azfind2"); dir.create(d)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  disc_py <- file.path(d, "ret_python")
  writeLines(c("#!/bin/sh", "exit 0"), disc_py); Sys.chmod(disc_py, "0755")

  .w3_with_env(list(CONDA_PREFIX = NA, PATH = ""), {
    local_mocked_bindings(
      py_discover_config = function(...) list(python = disc_py),
      .package = "reticulate")
    # Only the discovered python probes TRUE -> covers the python branch (67)
    # and the winning return inside the probe loop.
    local_mocked_bindings(.az_py_probe = function(python) identical(python, disc_py),
                          .package = "autozyme")
    expect_identical(ns$.az_py_find_existing(), disc_py)
  })
})

# ---- .az_py_bind_impl: already-initialized session branch (lines 93-98) -----
# Wave-1 covered the "no reticulate" + override + numpy-less paths. Here: when
# Python is ALREADY initialized this session we cannot switch interpreters; we
# only verify the live one has numpy+scipy. Both have/lacks branches.

test_that(".az_py_bind_impl uses the live session and imports core when numpy/scipy present", {
  skip_if_not_installed("reticulate")
  st <- ns$.az_py_state
  old_b <- st$bound; old_e <- st$err
  on.exit({ st$bound <- old_b; st$err <- old_e }, add = TRUE)
  st$bound <- NA; st$err <- NA_character_

  imported <- character()
  local_mocked_bindings(
    py_available        = function(initialize = FALSE) TRUE,
    py_module_available = function(m) TRUE,
    import              = function(name, ...) { imported <<- c(imported, name); NULL },
    .package = "reticulate")

  expect_true(ns$.az_py_bind_impl())
  # .az_py_import_core() warmed numpy + scipy.* (lines 133-136 / 97-98).
  expect_true("numpy" %in% imported)
  expect_true(any(grepl("scipy", imported)))
})

test_that(".az_py_bind_impl reports the live session lacks numpy/scipy", {
  skip_if_not_installed("reticulate")
  st <- ns$.az_py_state
  old_b <- st$bound; old_e <- st$err
  on.exit({ st$bound <- old_b; st$err <- old_e }, add = TRUE)
  st$bound <- NA; st$err <- NA_character_

  local_mocked_bindings(
    py_available        = function(initialize = FALSE) TRUE,
    py_module_available = function(m) FALSE,
    import              = function(...) stop("must not import when numpy absent"),
    .package = "reticulate")

  expect_false(ns$.az_py_bind_impl())      # line 98 returns have == FALSE
  expect_match(st$err, "numpy/scipy")      # line 96 error set
})

# ---- .az_py_bind_impl: managed virtualenv branch (lines 111-120) -----------
# Override env vars unset, but a managed virtualenv exists and its python
# probes TRUE -> use_virtualenv + import_core.

test_that(".az_py_bind_impl binds the managed virtualenv when it probes clean", {
  skip_on_os("windows")
  skip_if_not_installed("reticulate")
  st <- ns$.az_py_state
  old_b <- st$bound; old_e <- st$err
  on.exit({ st$bound <- old_b; st$err <- old_e }, add = TRUE)
  st$bound <- NA; st$err <- NA_character_

  used <- NULL
  local_mocked_bindings(
    py_available      = function(initialize = FALSE) FALSE,
    virtualenv_exists = function(envname) TRUE,
    virtualenv_python = function(envname) "/fake/venv/bin/python",
    use_virtualenv    = function(envname, required = TRUE) { used <<- envname; invisible() },
    import            = function(...) NULL,
    .package = "reticulate")
  # Force the venv python to probe TRUE; keep overrides + ambient out of the way.
  local_mocked_bindings(.az_py_probe = function(python)
    identical(python, "/fake/venv/bin/python"), .package = "autozyme")

  .w3_with_env(list(AUTOZYME_PYTHON = NA, RETICULATE_PYTHON = NA,
                    AUTOZYME_PY_ENV = "w3-managed"), {
    expect_true(ns$.az_py_bind_impl())
    expect_identical(used, "w3-managed")     # bound the managed env (line 117)
  })
})

# ---- .az_py_bind_impl: explicit AUTOZYME_PYTHON override binds (107-108) ----
# Wave-1's override test lives behind .az_py_bind (memo). Here we hit the impl
# directly so the override use_python + import_core lines are attributed.

test_that(".az_py_bind_impl binds an explicit AUTOZYME_PYTHON that probes clean", {
  skip_on_os("windows")
  skip_if_not_installed("reticulate")
  st <- ns$.az_py_state
  old_b <- st$bound; old_e <- st$err
  on.exit({ st$bound <- old_b; st$err <- old_e }, add = TRUE)
  st$bound <- NA; st$err <- NA_character_

  bound <- NULL
  local_mocked_bindings(
    py_available = function(initialize = FALSE) FALSE,
    use_python   = function(python, required = TRUE) { bound <<- python; invisible() },
    import       = function(...) NULL,
    .package = "reticulate")
  local_mocked_bindings(.az_py_probe = function(python)
    identical(python, "/explicit/py"), .package = "autozyme")

  .w3_with_env(list(AUTOZYME_PYTHON = "/explicit/py", RETICULATE_PYTHON = NA), {
    expect_true(ns$.az_py_bind_impl())
    expect_identical(bound, "/explicit/py")   # lines 107-108 use_python(override)
  })
})

# ---- .az_py_bind_impl: nothing usable anywhere -> error message (126-128) ---
test_that(".az_py_bind_impl reports the install_python_deps nudge when nothing usable", {
  skip_if_not_installed("reticulate")
  st <- ns$.az_py_state
  old_b <- st$bound; old_e <- st$err
  on.exit({ st$bound <- old_b; st$err <- old_e }, add = TRUE)
  st$bound <- NA; st$err <- NA_character_

  local_mocked_bindings(
    py_available      = function(initialize = FALSE) FALSE,
    virtualenv_exists = function(...) FALSE,
    .package = "reticulate")
  local_mocked_bindings(
    .az_py_probe        = function(python) FALSE,   # every override/venv fails
    .az_py_find_existing = function() "",            # and discovery finds nothing
    .package = "autozyme")

  .w3_with_env(list(AUTOZYME_PYTHON = NA, RETICULATE_PYTHON = NA), {
    expect_false(ns$.az_py_bind_impl())             # falls to the error (126-128)
    expect_match(st$err, "install_python_deps")
  })
})

# ---- .az_py_bind_impl: find_existing fallback success (lines 121-125) ------
test_that(".az_py_bind_impl falls through to .az_py_find_existing and binds it", {
  skip_if_not_installed("reticulate")
  st <- ns$.az_py_state
  old_b <- st$bound; old_e <- st$err
  on.exit({ st$bound <- old_b; st$err <- old_e }, add = TRUE)
  st$bound <- NA; st$err <- NA_character_

  bound <- NULL
  local_mocked_bindings(
    py_available      = function(initialize = FALSE) FALSE,
    virtualenv_exists = function(...) FALSE,
    use_python        = function(python, required = TRUE) { bound <<- python; invisible() },
    import            = function(...) NULL,
    .package = "reticulate")
  local_mocked_bindings(
    .az_py_probe       = function(python) FALSE,                 # overrides fail
    .az_py_find_existing = function() "/found/python3",          # but discovery hits
    .package = "autozyme")

  .w3_with_env(list(AUTOZYME_PYTHON = NA, RETICULATE_PYTHON = NA), {
    expect_true(ns$.az_py_bind_impl())
    expect_identical(bound, "/found/python3")   # line 123 use_python(found)
  })
})

# ---- .az_py_bind: error -> memo FALSE wrapper (lines 79-83) -----------------
test_that(".az_py_bind catches an impl error and memoizes FALSE", {
  st <- ns$.az_py_state
  old_b <- st$bound; old_e <- st$err
  on.exit({ st$bound <- old_b; st$err <- old_e }, add = TRUE)
  st$bound <- NA; st$err <- NA_character_

  local_mocked_bindings(.az_py_bind_impl = function() stop("boom-w3"),
                        .package = "autozyme")
  expect_false(ns$.az_py_bind())
  expect_identical(st$err, "boom-w3")           # line 80 captured conditionMessage
  expect_false(st$bound)                         # memoized FALSE (line 82)
})

# ---- install_python_deps: full wiring incl. install_python branch (168-195) -
# Wave-1's test mocked py_discover_config to a present python (skips
# install_python) and used default envname. Here: no system python -> the
# install_python branch (179-183) fires, and we pass explicit envname/packages
# to cover the non-default arg paths (168-169 are the is.null defaults; cover
# the populated side instead).

test_that("install_python_deps installs a standalone python when none is found", {
  skip_if_not_installed("reticulate")
  st <- ns$.az_py_state
  old_b <- st$bound
  on.exit({ st$bound <- old_b }, add = TRUE)

  installed_py <- FALSE; created <- NULL
  local_mocked_bindings(
    py_discover_config = function(...) list(python = ""),   # no system python
    install_python     = function(version = NULL, ...) { installed_py <<- TRUE; invisible() },
    virtualenv_create  = function(envname, version = NULL, packages = NULL, ...) {
      created <<- list(envname = envname, packages = packages, version = version)
      invisible() },
    .package = "reticulate")

  st$bound <- TRUE   # pretend a prior successful bind, must be reset to NA
  expect_message(
    install_python_deps(envname = "w3-env", packages = c("numpy"),
                        python_version = "3.11"),
    "ready")
  expect_true(installed_py)                  # standalone python fetched (line 181)
  expect_identical(created$envname, "w3-env")
  expect_identical(created$packages, "numpy")
  expect_identical(created$version, "3.11")
  expect_true(is.na(st$bound))               # memo reset (line 192)
})

test_that("install_python_deps with NULL args resolves the default env + package list", {
  skip_if_not_installed("reticulate")
  st <- ns$.az_py_state
  old_b <- st$bound
  on.exit({ st$bound <- old_b }, add = TRUE)

  created <- NULL
  local_mocked_bindings(
    py_discover_config = function(...) list(python = "/usr/bin/python3"),  # system py exists
    install_python     = function(...) stop("must not install_python when one exists"),
    virtualenv_create  = function(envname, version = NULL, packages = NULL, ...) {
      created <<- list(envname = envname, packages = packages); invisible() },
    .package = "reticulate")

  .w3_with_env(list(AUTOZYME_PY_ENV = NA), {
    st$bound <- NA
    # envname=NULL, packages=NULL -> the is.null default branches (lines 168-169).
    expect_message(install_python_deps(), "ready")
    expect_identical(created$envname, "r-autozyme")
    expect_setequal(created$packages, c("numpy", "scipy", "threadpoolctl"))
  })
})

# ---- .az_py_warmup_or_notify: interactive prompt accepted (lines 219-232) --
# Non-interactive nudge is wave-1. Here we force the interactive branch and the
# user accepting: install_python_deps + .az_py_bind succeed -> invisible(TRUE).

# DOCUMENTED LIMIT: the interactive setup-prompt body (lines 219-232: askYesNo
# -> install_python_deps -> re-bind) is UNREACHABLE in a batch Rscript session.
# `interactive()` is a .Primitive that returns FALSE here and CANNOT be mocked
# (local_mocked_bindings cannot intercept a primitive call), so `can_prompt` is
# always FALSE and execution drops straight to the final nudge. We therefore
# cover the reachable non-interactive fall-through (the realistic CI behavior):
# bind fails, no prompt possible, one clear actionable note is printed and the
# function returns invisible(FALSE) without erroring.

test_that(".az_py_warmup_or_notify (bind fails, no prompt) prints the actionable nudge", {
  skip_if_not_installed("reticulate")
  local_mocked_bindings(.az_py_bind = function() FALSE, .package = "autozyme")

  # AUTOZYME_NO_PROMPT set OR a non-interactive session both make can_prompt
  # FALSE; either way we land on the final nudge (lines 234-238).
  .w3_with_env(list(AUTOZYME_NO_PROMPT = "1"), {
    res <- NULL
    expect_message(res <- ns$.az_py_warmup_or_notify("seurat", "PCA/CCA"),
                   "acceleration is OFF")
    expect_message(ns$.az_py_warmup_or_notify("seurat", "PCA/CCA"),
                   "install_python_deps")
    expect_false(res)
    expect_invisible(ns$.az_py_warmup_or_notify("seurat", "PCA/CCA"))
  })
})

test_that(".az_py_warmup_or_notify returns invisible(TRUE) when a usable Python binds", {
  skip_if_not_installed("reticulate")
  # Early-return success path (line 216): a usable/managed Python is found.
  local_mocked_bindings(.az_py_bind = function() TRUE, .package = "autozyme")
  expect_true(ns$.az_py_warmup_or_notify("seurat", "PCA"))
  expect_invisible(ns$.az_py_warmup_or_notify("seurat", "PCA"))
})

# ============================================================================
# R/intercept_probe.R
# ============================================================================

# ---- .zyme_intercept_install: first-install path (lines 86-95) -------------
# Wave-1 covered the unknown-patch no-op and install_from_env FALSE. The
# install() body (set installed TRUE + reg.finalizer) was uncovered. We
# snapshot/restore the env so the global state is not perturbed for siblings.

test_that(".zyme_intercept_install runs the first-install path then is idempotent", {
  ic <- ns$.zyme_intercept
  old_installed <- ic$installed
  on.exit(ic$installed <- old_installed, add = TRUE)

  ic$installed <- FALSE
  expect_silent(ns$.zyme_intercept_install())   # first install: sets TRUE + finalizer
  expect_true(ic$installed)
  # Second call: already-installed branch (lines 86-89), still a silent no-op.
  expect_silent(ns$.zyme_intercept_install())
  expect_true(ic$installed)
})

test_that(".zyme_intercept_install (already installed) re-runs install_for for a named patch", {
  ic <- ns$.zyme_intercept
  old_installed <- ic$installed
  on.exit(ic$installed <- old_installed, add = TRUE)
  ic$installed <- TRUE
  # Unknown patch -> install_for is a no-op, so this stays silent (line 87).
  expect_silent(ns$.zyme_intercept_install("definitely_not_a_patch_w3"))
})

test_that(".zyme_intercept_install (first install, named patch) forwards to install_for", {
  ic <- ns$.zyme_intercept
  old_installed <- ic$installed
  on.exit(ic$installed <- old_installed, add = TRUE)
  reg <- get0(".zyme_registry", envir = ns, inherits = FALSE)
  skip_if(is.null(reg), "no .zyme_registry in this build")
  key <- "__w3_first_install__"
  on.exit(if (exists(key, envir = reg, inherits = FALSE)) rm(list = key, envir = reg),
          add = TRUE)
  # A known patch with NULL targets -> install_for returns early (line 45) BEFORE
  # the locked-binding assign, so the first-install path reaches line 94
  # (install_for called from inside the not-yet-installed branch) cleanly.
  assign(key, list(upstream = "U", targets = NULL), envir = reg)
  ic$installed <- FALSE
  expect_silent(ns$.zyme_intercept_install(key))   # first install + install_for(94)
  expect_true(ic$installed)
})

# ---- .zyme_intercept_install_for: the real wrapping loop (lines 48-59) ------
# DOCUMENTED LIMIT (wave-1 note): the final
# `assign(".zyme_registry", ., envir = asNamespace("autozyme"))` (line 60)
# hard-errors because the namespace binding is locked. HOWEVER, .zyme_registry
# is an *environment* whose CONTENTS are mutable by reference, so lines 48-59
# (build keys, wrap function targets AND s4-list targets, mutate the registry
# entry in place) all execute before the locked assign throws. We inject a fake
# patch entry into the live registry env, call install_for inside the expected
# locked-binding error, and assert the wrapping was applied in place.

test_that(".zyme_intercept_install_for wraps both function and s4-list targets in place", {
  reg <- get0(".zyme_registry", envir = ns, inherits = FALSE)
  skip_if(is.null(reg), "no .zyme_registry in this build")
  key <- "__w3_fake_patch__"
  on.exit(if (exists(key, envir = reg, inherits = FALSE)) rm(list = key, envir = reg),
          add = TRUE)

  fn_target <- function(x) x + 1L
  s4_target <- list(kind = "s4", signature = "X", fn = function(y) y * 2L)
  assign(key, list(upstream = "FakeUp",
                   targets = list(opFun = fn_target, opS4 = s4_target)),
         envir = reg)

  # The wrap loop (48-58) + in-place registry[[patch]] <- entry (59) run, THEN
  # the locked-binding assign (60) errors. This is the documented wall.
  expect_error(ns$.zyme_intercept_install_for(key),
               "locked binding")

  # Despite the error, the registry env was mutated in place (line 59) with the
  # wrapped targets -> both target shapes carry the intercept key.
  e2 <- get(key, envir = reg, inherits = FALSE)
  expect_identical(attr(e2$targets$opFun, ".zyme_intercept_key"), "FakeUp::opFun")
  expect_identical(attr(e2$targets$opS4$fn, ".zyme_intercept_key"), "FakeUp::opS4")

  # And the wrapped fn still counts + forwards when invoked.
  oldc <- ic <- ns$.zyme_intercept; old_counts <- ic$counts
  on.exit({ ic$counts <- old_counts }, add = TRUE)
  ic$counts <- list()
  expect_equal(e2$targets$opFun(41L), 42L)
  expect_identical(ic$counts[["FakeUp::opFun"]], 1L)
})

test_that(".zyme_intercept_install_for is a no-op when a known patch has NULL targets", {
  reg <- get0(".zyme_registry", envir = ns, inherits = FALSE)
  skip_if(is.null(reg), "no .zyme_registry in this build")
  key <- "__w3_no_targets__"
  on.exit(if (exists(key, envir = reg, inherits = FALSE)) rm(list = key, envir = reg),
          add = TRUE)
  assign(key, list(upstream = "U", targets = NULL), envir = reg)
  # targets NULL -> early return (line 45) before any locked assign; silent.
  expect_silent(ns$.zyme_intercept_install_for(key))
})

# ---- .zyme_intercept_install_from_env: gated TRUE path (lines 100-103) ------
test_that(".zyme_intercept_install_from_env returns TRUE when gated on", {
  ic <- ns$.zyme_intercept
  old_installed <- ic$installed
  on.exit(ic$installed <- old_installed, add = TRUE)
  ic$installed <- TRUE   # so install() short-circuits, no finalizer churn

  .w3_with_env(list(ZYME_INSTRUMENT_INTERCEPTS = "1",
                    ZYME_INTERCEPT_PATCH = ""), {
    expect_true(ns$.zyme_intercept_install_from_env())  # line 103 returns TRUE
  })
})

test_that(".zyme_intercept_install_from_env forwards a named patch when gated on", {
  ic <- ns$.zyme_intercept
  old_installed <- ic$installed
  on.exit(ic$installed <- old_installed, add = TRUE)
  ic$installed <- TRUE
  # An unknown patch name flows through install() -> install_for() no-op.
  .w3_with_env(list(ZYME_INSTRUMENT_INTERCEPTS = "1",
                    ZYME_INTERCEPT_PATCH = "unknown_patch_w3"), {
    expect_true(ns$.zyme_intercept_install_from_env())
  })
})

# ============================================================================
# R/native_blas.R
# ============================================================================

# ---- .az_conda_roots: CONDA_PREFIX-under-envs detection (lines 63-65) -------
test_that(".az_conda_roots adds the conda root when CONDA_PREFIX is under .../envs/", {
  skip_on_os("windows")
  base_root <- tempfile("azconda"); env_dir <- file.path(base_root, "envs", "myenv")
  dir.create(env_dir, recursive = TRUE)
  on.exit(unlink(base_root, recursive = TRUE), add = TRUE)

  .w3_with_env(list(CONDA_PREFIX = env_dir, MAMBA_ROOT_PREFIX = NA,
                    CONDA_ROOT = NA, PATH = ""), {
    roots <- ns$.az_conda_roots()
    norm_base <- normalizePath(base_root, winslash = "/", mustWork = FALSE)
    # dirname(dirname(env_dir)) == base_root is appended (line 64).
    expect_true(norm_base %in% roots)
  })
})

test_that(".az_conda_roots harvests a conda dir found on PATH", {
  skip_on_os("windows")
  d <- tempfile("miniconda3"); dir.create(file.path(d, "bin"), recursive = TRUE)
  on.exit(unlink(d, recursive = TRUE), add = TRUE)
  binp <- file.path(d, "bin")
  .w3_with_env(list(PATH = binp, CONDA_PREFIX = NA, MAMBA_ROOT_PREFIX = NA,
                    CONDA_ROOT = NA), {
    roots <- ns$.az_conda_roots()
    # The miniconda3 segment on PATH is sliced back to its root (lines 53-61).
    expect_true(any(grepl("miniconda3", roots)))
  })
})

# ---- .az_gemm: native attempt path is reached when BLAS enabled (157-177) --
# Wave-1 forced AUTOZYME_BLAS_DISABLE=1 so can_native is FALSE and the native
# block (165-173) is skipped. Here we leave dynamic BLAS ENABLED so can_native
# is TRUE; az_blas_gemm runs (returns NULL or errors with no usable lib on this
# host) and the tryCatch fallback (168-170) yields the correct base result.
# threads<=1 keeps any kernel single-threaded and safe.

test_that(".az_gemm exercises the native attempt + safe fallback when BLAS enabled", {
  .w3_with_env(list(AUTOZYME_BLAS_DISABLE = NA, AUTOZYME_DYNAMIC_BLAS = NA,
                    AUTOZYME_DISABLE = NA), {
    set.seed(11)
    A <- matrix(rnorm(12), 4, 3); B <- matrix(rnorm(15), 3, 5)
    # threads = 1L: native path entered (can_native TRUE), result must equal base.
    expect_equal(ns$.az_gemm(A, B, threads = 1L), A %*% B, tolerance = 1e-10)
    # transposed combo through the native attempt as well.
    expect_equal(ns$.az_gemm(A, A, transA = TRUE, threads = 1L), crossprod(A),
                 tolerance = 1e-10)
    # vector promotion through .az_as_dense_double -> native attempt -> fallback.
    v <- c(1, 2, 3)
    expect_equal(ns$.az_gemm(A, v, threads = 0L), A %*% matrix(v, ncol = 1),
                 tolerance = 1e-10)
    # Non-coercible threads -> NA -> clamped to 0L (line 157), still correct.
    expect_equal(suppressWarnings(ns$.az_gemm(A, B, threads = "bogus")),
                 A %*% B, tolerance = 1e-10)
  })
})

test_that(".az_gemm with fallback=FALSE re-stops a native error when BLAS enabled", {
  # When fallback=FALSE and the native call errors, the error is re-raised
  # (lines 168-171 stop(e)). Force az_blas_gemm to throw.
  local_mocked_bindings(az_blas_gemm = function(...) stop("native-boom-w3"),
                        .package = "autozyme")
  .w3_with_env(list(AUTOZYME_BLAS_DISABLE = NA, AUTOZYME_DYNAMIC_BLAS = NA,
                    AUTOZYME_DISABLE = NA), {
    A <- matrix(1, 2, 2); B <- matrix(1, 2, 2)
    expect_error(ns$.az_gemm(A, B, fallback = FALSE, threads = 1L),
                 "native-boom-w3")
    # With fallback TRUE the same native error is swallowed -> base result.
    expect_equal(ns$.az_gemm(A, B, fallback = TRUE, threads = 1L), A %*% B)
  })
})

# ---- .az_dynamic_blas_enabled: per-patch OPTION path (lines 218-223) --------
# Wave-1 covered the per-patch ENV override. Here: env unset, fall to the
# per-patch getOption(autozyme.<patch>.dynamic_blas) (lines 218-220) and the
# final .az_feature_enabled default (line 223).

test_that(".az_dynamic_blas_enabled honors the per-patch option then the default", {
  old_opt <- getOption("autozyme.seurat.dynamic_blas")
  on.exit(options(autozyme.seurat.dynamic_blas = old_opt), add = TRUE)

  .w3_with_env(list(AUTOZYME_DYNAMIC_BLAS = NA, AUTOZYME_SEURAT_BLAS = NA,
                    AUTOZYME_BLAS_DISABLE = NA, AUTOZYME_DISABLE = NA), {
    options(autozyme.seurat.dynamic_blas = FALSE)
    expect_false(ns$.az_dynamic_blas_enabled(patch = "seurat"))   # option FALSE (219-220)
    options(autozyme.seurat.dynamic_blas = TRUE)
    expect_true(ns$.az_dynamic_blas_enabled(patch = "seurat"))
    # Option cleared -> falls to .az_feature_enabled(default=TRUE) (line 223).
    options(autozyme.seurat.dynamic_blas = NULL)
    expect_true(ns$.az_dynamic_blas_enabled(patch = "seurat", default = TRUE))
    expect_false(ns$.az_dynamic_blas_enabled(patch = "seurat", default = FALSE))
  })
})

test_that(".az_dynamic_blas_enabled global env FALSE short-circuits before patch", {
  .w3_with_env(list(AUTOZYME_DYNAMIC_BLAS = "0", AUTOZYME_BLAS_DISABLE = NA,
                    AUTOZYME_DISABLE = NA), {
    # Global truthy-but-FALSE -> early return FALSE (line 211), patch never read.
    expect_false(ns$.az_dynamic_blas_enabled(patch = "seurat"))
  })
})

# ---- .az_default_blas_threads: option non-positive falls to auto_threads ----
# Wave-1 covered ZYME_THREADS + option-set + the null fallthrough. Here: a
# zero/negative option is treated as unset (lines 198-201) and resolves >=1.

test_that(".az_default_blas_threads treats a non-positive option as unset", {
  old_opt <- getOption("autozyme.threads")
  on.exit(options(autozyme.threads = old_opt), add = TRUE)
  .w3_with_env(list(ZYME_THREADS = NA), {
    options(autozyme.threads = 0L)            # <=0 -> ignored
    n <- ns$.az_default_blas_threads()
    expect_true(is.numeric(n) && n >= 1L)     # final clamp (line 202)
  })
})

# ============================================================================
# R/utils.R
# ============================================================================

# ---- resolve_dataset_path: absolute-path candidate branch (line 57-62) ------
# Wave-1 covered the relative / ./ / basename-rescue / error paths. Here: an
# ABSOLUTE raw_path that already exists is returned via the absolute branch.

test_that("resolve_dataset_path returns an existing absolute path as-is", {
  skip_on_os("windows")
  td <- tempfile("az_abs_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  f <- file.path(td, "abs_inputs.rds")
  saveRDS(1, f)
  abs_path <- normalizePath(f)              # starts with "/" on unix
  got <- resolve_dataset_path(td, abs_path) # absolute branch (line 57-58)
  expect_equal(normalizePath(got), normalizePath(f))
})

test_that("resolve_dataset_path: absolute that is missing falls through to basename rescue", {
  skip_on_os("windows")
  td <- tempfile("az_abs2_"); dir.create(file.path(td, "data"), recursive = TRUE)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  saveRDS(1, file.path(td, "data", "shared.rds"))
  # Absolute but non-existent first candidate; basename matches data/shared.rds.
  got <- resolve_dataset_path(td, "/nonexistent/stale/shared.rds")
  expect_equal(normalizePath(got), normalizePath(file.path(td, "data", "shared.rds")))
})
