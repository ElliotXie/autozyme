# ── Python environment management ───────────────────────────────────────────
#
# A few patches (currently only `seurat`: RunPCA / RunCCA / integration) offload
# linear algebra to NumPy + SciPy via reticulate. Everything else in autozyme is
# native C++/R and needs no Python.
#
# Design principle: the heavy cost (downloading + building a Python env) is paid
# at an *explicit setup moment* the user perceives as "installing" (slow is OK),
# never at analysis time. Concretely:
#
#   * install_python_deps()  -- one-time, user-facing, builds a dedicated env.
#   * first interactive activate("seurat") -- offers to run it, then warms Python
#     up so the *first* RunPCA pays zero startup cost (runtime must feel fast).
#   * runtime discovery       -- probes candidates for numpy+scipy *before*
#     binding, so we never lock the session onto a numpy-less interpreter.
#
# No machine-specific conda env names are hardcoded: all users converge on one
# reproducible virtualenv, overridable via env vars.

# Dedicated, reproducible env name. Override with AUTOZYME_PY_ENV.
.az_py_env_name <- function() {
  e <- Sys.getenv("AUTOZYME_PY_ENV", unset = "")
  if (nzchar(e)) e else "r-autozyme"
}

# Python packages the fast paths need.
.az_py_packages <- function() c("numpy", "scipy", "threadpoolctl")

# Session memo for the runtime bind. NA = not tried; TRUE/FALSE = result.
.az_py_state <- new.env(parent = emptyenv())
.az_py_state$bound <- NA
.az_py_state$err   <- NA_character_

# Subprocess probe: does this interpreter import numpy + scipy? Runs in a child
# process so it never initializes reticulate (which would lock the session onto
# whatever it touches first). Portable across platforms via a temp script.
.az_py_probe <- function(python) {
  if (!nzchar(python) || !file.exists(python)) return(FALSE)
  tf <- tempfile(fileext = ".py")
  on.exit(unlink(tf), add = TRUE)
  writeLines("import numpy, scipy", tf)
  q <- shQuote(tf, type = if (.Platform$OS.type == "windows") "cmd" else "sh")
  status <- tryCatch(
    suppressWarnings(system2(python, q, stdout = FALSE, stderr = FALSE)),
    error = function(e) 1L)
  identical(as.integer(status), 0L)
}

# Scan for an already-usable Python (numpy + scipy) the user/system provides,
# so anyone with a working scientific Python never has to run
# install_python_deps(). Looks at the active conda/virtualenv (CONDA_PREFIX),
# PATH, and reticulate's own discovery -- then PROBES each for numpy+scipy in a
# child process and returns the first that passes ("" if none). Cheap: a handful
# of subprocess probes, first hit wins.
.az_py_find_existing <- function() {
  cands <- character(0)
  cp <- Sys.getenv("CONDA_PREFIX", unset = "")
  if (nzchar(cp)) cands <- c(cands,
    file.path(cp, "bin", "python"),   # active conda/virtualenv (Unix/macOS)
    file.path(cp, "python.exe"))      # active conda (Windows)
  for (nm in c("python3", "python")) {
    p <- unname(Sys.which(nm))
    if (nzchar(p)) cands <- c(cands, p)
  }
  cfg <- tryCatch(reticulate::py_discover_config(), error = function(e) NULL)
  if (!is.null(cfg)) {
    if (length(cfg$python))          cands <- c(cands, cfg$python)
    if (length(cfg$python_versions)) cands <- c(cands, cfg$python_versions)
  }
  cands <- unique(cands[nzchar(cands)])
  for (p in cands) if (.az_py_probe(p)) return(p)
  ""
}

# Resolve + bind a numpy/scipy-capable Python and import the modules. Memoized:
# subsequent calls return the cached result instantly. Returns TRUE on success.
.az_py_bind <- function() {
  if (!is.na(.az_py_state$bound)) return(.az_py_state$bound)
  ok <- tryCatch(.az_py_bind_impl(), error = function(e) {
    .az_py_state$err <- conditionMessage(e); FALSE
  })
  .az_py_state$bound <- isTRUE(ok)
  isTRUE(ok)
}

.az_py_bind_impl <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    .az_py_state$err <- "reticulate not installed"
    return(FALSE)
  }
  # If Python is already initialized this session we cannot switch interpreters;
  # only verify the live one has numpy+scipy.
  if (reticulate::py_available(initialize = FALSE)) {
    have <- reticulate::py_module_available("numpy") &&
            reticulate::py_module_available("scipy")
    if (!have) .az_py_state$err <- "active Python session lacks numpy/scipy"
    if (have) .az_py_import_core()
    return(have)
  }
  # Ordered candidates, each probed for numpy+scipy *before* we commit.
  #   1. explicit overrides (AUTOZYME_PYTHON / RETICULATE_PYTHON)
  #   2. the managed env (if the user ran install_python_deps())
  #   3. any existing usable Python the user/system already has
  for (v in c("AUTOZYME_PYTHON", "RETICULATE_PYTHON")) {
    p <- Sys.getenv(v, unset = "")
    if (nzchar(p) && .az_py_probe(p)) {
      reticulate::use_python(p, required = TRUE)
      return(.az_py_import_core())
    }
  }
  envname <- .az_py_env_name()
  if (isTRUE(tryCatch(reticulate::virtualenv_exists(envname),
                      error = function(e) FALSE))) {
    py <- tryCatch(reticulate::virtualenv_python(envname),
                   error = function(e) "")
    if (.az_py_probe(py)) {
      reticulate::use_virtualenv(envname, required = TRUE)
      return(.az_py_import_core())
    }
  }
  found <- .az_py_find_existing()
  if (nzchar(found)) {
    reticulate::use_python(found, required = TRUE)
    return(.az_py_import_core())
  }
  .az_py_state$err <- paste0(
    "no Python with numpy+scipy found; run autozyme::install_python_deps()")
  FALSE
}

# Import the core modules so the first real call pays no import cost.
.az_py_import_core <- function() {
  reticulate::import("numpy",               convert = FALSE)
  reticulate::import("scipy.linalg",        convert = FALSE)
  reticulate::import("scipy.sparse.linalg", convert = FALSE)
  TRUE
}

#' Set up the Python environment for autozyme's Python-backed fast paths
#'
#' Builds a dedicated, reproducible virtualenv (default name \code{r-autozyme})
#' containing NumPy + SciPy, used by the Seurat \code{RunPCA} / \code{RunCCA} /
#' integration fast paths. This is a one-time setup step: the download can take
#' a few minutes, but afterwards analysis runs pay no setup cost.
#'
#' If no system Python is available, a standalone interpreter is fetched first
#' via \code{reticulate::install_python()}.
#'
#' You normally do not need to call this directly: the first interactive
#' \code{activate("seurat")} offers to run it for you. Call it explicitly in
#' non-interactive setups (CI, Docker image builds, headless servers).
#'
#' @param envname Virtualenv name. Defaults to \code{getOption} /
#'   \code{AUTOZYME_PY_ENV} or \code{"r-autozyme"}.
#' @param packages Character vector of pip packages to install. Defaults to
#'   numpy, scipy, threadpoolctl.
#' @param python_version Optional Python version string (e.g. \code{"3.11"}).
#' @param ... Passed to \code{reticulate::virtualenv_create()}.
#' @return Invisibly \code{TRUE} on success.
#' @export
install_python_deps <- function(envname = NULL, packages = NULL,
                                 python_version = NULL, ...) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for the Python fast paths. Install ",
         "it with install.packages(\"reticulate\"), then re-run ",
         "autozyme::install_python_deps().", call. = FALSE)
  }
  if (is.null(envname))  envname  <- .az_py_env_name()
  if (is.null(packages)) packages <- .az_py_packages()
  message(sprintf(
    "[autozyme] Setting up Python env '%s' (%s).", envname,
    paste(packages, collapse = ", ")))
  message("[autozyme] This is a one-time download and may take a few minutes...")

  have_python <- tryCatch({
    cfg <- reticulate::py_discover_config()
    !is.null(cfg) && nzchar(cfg$python)
  }, error = function(e) FALSE)
  if (!have_python) {
    message("[autozyme] No system Python found; installing a standalone Python...")
    reticulate::install_python(
      version = if (is.null(python_version)) "3.11:latest" else python_version)
  }

  reticulate::virtualenv_create(
    envname  = envname,
    version  = python_version,
    packages = packages,
    ...)

  # A new env may supersede a prior failed bind this session.
  .az_py_state$bound <- NA
  .az_py_state$err   <- NA_character_
  message(sprintf("[autozyme] Python env '%s' is ready.", envname))
  invisible(TRUE)
}

# Called from a patch's on_activate hook. First tries to bind an existing or
# managed Python (and warms it up so the first fast-path call is instant). Only
# if NOTHING usable is found does it offer to set one up (interactive) or print
# one clear, actionable note (non-interactive). Never errors out of activation;
# never blocks scripts/CI.
.az_py_warmup_or_notify <- function(patch = "seurat",
                                    feature = "PCA/CCA/integration") {
  nudge <- function(msg) { message(msg); invisible(FALSE) }

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    return(nudge(sprintf(
      paste0("[autozyme] '%s' %s acceleration needs Python. Install reticulate ",
             "and run autozyme::install_python_deps() to enable; other ",
             "speedups are active."), patch, feature)))
  }

  # Use whatever the user already has (or the managed env) if it works -- no
  # install needed -- and warm it up now.
  if (isTRUE(.az_py_bind())) return(invisible(TRUE))

  # Nothing usable found anywhere. Offer to set it up if we can ask; else inform.
  can_prompt <- interactive() &&
    !.az_truthy(Sys.getenv("AUTOZYME_NO_PROMPT", unset = ""))
  if (can_prompt) {
    ans <- tryCatch(utils::askYesNo(sprintf(
      paste0("[autozyme] No Python with numpy/scipy found. Set one up now for ",
             "%s acceleration ('%s')? One-time download."), feature, patch),
      default = TRUE), error = function(e) NA)
    if (isTRUE(ans)) {
      ok <- tryCatch({ install_python_deps(); .az_py_bind() },
                     error = function(e) {
                       warning(conditionMessage(e), call. = FALSE); FALSE
                     })
      if (isTRUE(ok)) return(invisible(TRUE))
    }
  }
  nudge(sprintf(
    paste0("[autozyme] '%s' %s acceleration is OFF (no Python with numpy/scipy ",
           "found). Run autozyme::install_python_deps() once to enable, or set ",
           "AUTOZYME_PYTHON to your interpreter; other speedups are active."),
    patch, feature))
}
