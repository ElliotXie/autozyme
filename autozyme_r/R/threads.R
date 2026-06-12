.thread_env_vars <- c(
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "VECLIB_MAXIMUM_THREADS",
  "NUMEXPR_NUM_THREADS"
)

#' Set thread count for parallelism sources patches use
#'
#' Propagates `n` to BLAS/OpenMP env vars and (if installed) RhpcBLASctl.
#' Patches that read per-package thread options should consult these vars
#' or expose their own knob.
#'
#' @param n Positive integer thread count.
#' @return The integer `n`, invisibly.
#' @export
set_threads <- function(n) {
  n <- as.integer(n)
  stopifnot(length(n) == 1L, !is.na(n), n >= 1L)
  .az_apply_thread_count(n)
  options(autozyme.threads = n)
  invisible(n)
}

.az_capture_thread_state <- function() {
  old_blas <- NULL
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    old_blas <- list(
      blas = tryCatch(RhpcBLASctl::blas_get_num_procs(),
                      error = function(e) NA_integer_),
      omp = tryCatch(RhpcBLASctl::omp_get_max_threads(),
                     error = function(e) NA_integer_)
    )
  }
  list(
    env = Sys.getenv(.thread_env_vars, unset = NA_character_),
    option = getOption("autozyme.threads", NULL),
    blas = old_blas
  )
}

.az_apply_thread_count <- function(n) {
  n <- as.integer(n)
  stopifnot(length(n) == 1L, !is.na(n), n >= 1L)
  args <- as.list(rep(as.character(n), length(.thread_env_vars)))
  names(args) <- .thread_env_vars
  do.call(Sys.setenv, args)
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    RhpcBLASctl::blas_set_num_threads(n)
    RhpcBLASctl::omp_set_num_threads(n)
  }
  if (requireNamespace("RcppParallel", quietly = TRUE)) {
    try(RcppParallel::setThreadOptions(numThreads = n), silent = TRUE)
  }
  invisible(n)
}

.az_restore_thread_state <- function(state) {
  restore_env <- as.list(state$env[!is.na(state$env)])
  unset_env <- names(state$env)[is.na(state$env)]
  if (length(restore_env)) do.call(Sys.setenv, restore_env)
  if (length(unset_env)) Sys.unsetenv(unset_env)

  if (is.null(state$option)) {
    options(autozyme.threads = NULL)
  } else {
    options(autozyme.threads = state$option)
  }

  if (!is.null(state$blas) && requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    if (!is.na(state$blas$blas)) RhpcBLASctl::blas_set_num_threads(state$blas$blas)
    if (!is.na(state$blas$omp)) RhpcBLASctl::omp_set_num_threads(state$blas$omp)
  }
  if (requireNamespace("RcppParallel", quietly = TRUE)) {
    restore_n <- suppressWarnings(as.integer(state$option))
    if (length(restore_n) != 1L || is.na(restore_n) || restore_n < 1L) {
      restore_n <- suppressWarnings(as.integer(
        Sys.getenv("AUTOZYME_THREADS", unset = "1")))
    }
    if (length(restore_n) != 1L || is.na(restore_n) || restore_n < 1L) {
      restore_n <- 1L
    }
    try(RcppParallel::setThreadOptions(numThreads = restore_n), silent = TRUE)
  }
  invisible(NULL)
}

.az_thread_enter <- function(n = 1L) {
  state <- .az_capture_thread_state()
  .az_apply_thread_count(n)
  state
}

.az_thread_exit <- function(state) {
  .az_restore_thread_state(state)
}

.az_thread_scope <- function(n = 1L, expr) {
  state <- .az_thread_enter(n)
  on.exit(.az_thread_exit(state), add = TRUE)
  force(expr)
}

#' Pick a sensible thread count for a patch
#'
#' Resolves a worker count using this priority order:
#' \enumerate{
#'   \item \code{AUTOZYME_THREADS} environment variable (explicit user
#'     override; wins over everything, including \code{cap}).
#'   \item \code{getOption("autozyme.threads")} (also wins over \code{cap}).
#'   \item Hardware default: \code{detectCores(logical=FALSE) - 1}, bounded
#'     above by \code{cap} (if given) and a hard ceiling of 16 to prevent
#'     runaway oversubscription on big machines.
#' }
#'
#' Designed for use inside lifted patches as a drop-in replacement for
#' hardcoded thread counts (\code{mc.cores = 12L} -> \code{mc.cores =
#' auto_threads(cap = 12L)}). \code{cap} should be the patch's max sensible
#' worker count — typically what the lift-time \code{pipeline/run.R} used.
#' Tier-aware patches pass \code{cap} from a per-tier dict.
#'
#' @param cap Optional integer upper bound. Caps the hardware default; does
#'   NOT cap the env-var or option override (those represent explicit user
#'   intent and win even when above \code{cap}, e.g. for CI thread sweeps).
#' @return Positive integer thread count, always >= 1.
#' @examples
#' auto_threads()                        # hardware default
#' auto_threads(cap = 8L)                # cap at 8
#' Sys.setenv(AUTOZYME_THREADS = "4")
#' auto_threads(cap = 8L)                # 4 (env wins)
#' Sys.unsetenv("AUTOZYME_THREADS")
#' @export
auto_threads <- function(cap = NULL) {
  # 1. AUTOZYME_THREADS env var — wins over cap
  env <- Sys.getenv("AUTOZYME_THREADS", unset = "")
  if (nzchar(env)) {
    n <- suppressWarnings(as.integer(env))
    if (!is.na(n) && n >= 1L) return(n)
  }

  # 2. autozyme.threads option — also wins over cap
  opt <- getOption("autozyme.threads", default = NULL)
  if (!is.null(opt)) {
    n <- suppressWarnings(as.integer(opt))
    if (!is.na(n) && n >= 1L) return(n)
  }

  # 3. Hardware default, bounded by cap and 16-thread ceiling
  cores <- tryCatch(parallel::detectCores(logical = FALSE),
                    error = function(e) NA_integer_)
  if (is.na(cores) || cores < 1L) cores <- 1L
  default <- max(1L, as.integer(cores) - 1L)

  if (!is.null(cap)) {
    cap_int <- suppressWarnings(as.integer(cap))
    if (!is.na(cap_int) && cap_int >= 1L) {
      default <- min(default, cap_int)
    }
  }

  default <- min(default, 16L)
  max(1L, default)
}
