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
#'   \item The first set of these environment variables, in order:
#'     \code{ZYME_THREADS} > \code{AUTOZYME_THREADS} > \code{OMP_NUM_THREADS}
#'     (any one the attest harness propagates). An explicit env override wins
#'     over everything, including \code{cap} and \code{default}.
#'   \item \code{getOption("autozyme.threads")} (also wins over \code{cap}
#'     and \code{default}).
#'   \item \code{default} threads, bounded above by the machine's core count,
#'     by \code{cap} (if given), and by a hard ceiling of 16.
#' }
#'
#' \code{default} is \strong{4} — the count the finalized speedup sweeps
#' showed is the best single conservative default: it captures the large 1->4
#' jump (~1.9x median wall-clock) while staying clear of the oversubscription
#' cliff that makes >4 threads \emph{slower} on small inputs for many patches.
#' Pass \code{default = NULL} to opt a patch into hardware scaling
#' (\code{detectCores(logical=FALSE) - 1}, still capped at 16) — reserve that
#' for the few patches whose finalized data keeps improving past 4 threads.
#'
#' Designed for use inside lifted patches as a drop-in replacement for
#' hardcoded thread counts (\code{mc.cores = 12L} -> \code{mc.cores =
#' auto_threads(cap = 12L)}). \code{cap} is the patch's max sensible worker
#' count — a ceiling layered on top of \code{default}.
#'
#' @param cap Optional integer upper bound on the resolved count. Does NOT cap
#'   the env-var or option override (those represent explicit user intent and
#'   win even when above \code{cap}, e.g. for CI thread sweeps).
#' @param default Base thread count when no env/option override is present.
#'   Defaults to 4. \code{NULL} means "scale to hardware"
#'   (\code{detectCores(logical=FALSE) - 1}).
#' @return Positive integer thread count, always >= 1.
#' @examples
#' auto_threads()                        # 4 (bounded by core count)
#' auto_threads(cap = 2L)                # 2
#' auto_threads(default = NULL)          # detectCores() - 1, max 16
#' Sys.setenv(AUTOZYME_THREADS = "8")
#' auto_threads()                        # 8 (env wins)
#' Sys.unsetenv("AUTOZYME_THREADS")
#' @export
auto_threads <- function(cap = NULL, default = 4L) {
  # 1. Thread budget from env — wins over cap/default. Honor any of the env
  # vars the attest harness propagates (ZYME_THREADS primary; AUTOZYME_THREADS
  # / OMP_NUM_THREADS are mirrors, matching the Python side). Reading all
  # three keeps benchmark thread-pinning robust regardless of which knob the
  # harness sets, so the conservative default below never perturbs a measured
  # sweep.
  for (var in c("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS")) {
    env <- Sys.getenv(var, unset = "")
    if (nzchar(env)) {
      n <- suppressWarnings(as.integer(env))
      if (!is.na(n) && n >= 1L) return(n)
    }
  }

  # 2. autozyme.threads option — also wins over cap/default
  opt <- getOption("autozyme.threads", default = NULL)
  if (!is.null(opt)) {
    n <- suppressWarnings(as.integer(opt))
    if (!is.na(n) && n >= 1L) return(n)
  }

  # 3. Base target, bounded by hardware, cap, and the 16-thread ceiling.
  cores <- tryCatch(parallel::detectCores(logical = FALSE),
                    error = function(e) NA_integer_)
  if (is.na(cores) || cores < 1L) cores <- 1L

  if (is.null(default)) {
    # "Scale to hardware" — for patches whose finalized sweeps keep speeding
    # up past 4 threads. Leave one core for the OS.
    base <- max(1L, as.integer(cores) - 1L)
  } else {
    base <- suppressWarnings(as.integer(default))
    if (is.na(base) || base < 1L) base <- 4L  # soft fallback to the floor
    # Never hand out more workers than the machine has cores.
    base <- min(base, as.integer(cores))
  }

  if (!is.null(cap)) {
    cap_int <- suppressWarnings(as.integer(cap))
    if (!is.na(cap_int) && cap_int >= 1L) {
      base <- min(base, cap_int)
    }
  }

  base <- min(base, 16L)
  max(1L, base)
}
