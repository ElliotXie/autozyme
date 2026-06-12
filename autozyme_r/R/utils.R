#' Cross-platform drop-in for parallel::mclapply
#'
#' parallel::mclapply uses fork (Unix-only). On Windows it does not support
#' mc.cores > 1 — calling it that way is a hard error. This helper:
#' \itemize{
#'   \item Forwards to parallel::mclapply on Unix when mc.cores > 1.
#'   \item Falls back to lapply on Windows or when mc.cores <= 1.
#' }
#' The Windows fallback sacrifices in-process parallelism for correctness;
#' patches that rely on fork's closure-capture semantics keep working
#' without rewrites. For a future cross-platform parallel speedup, swap
#' this for future.apply::future_lapply.
#'
.zyme_mclapply <- function(X, FUN, ..., mc.cores = 1, mc.preschedule = TRUE,
                           mc.set.seed = TRUE, mc.cleanup = TRUE,
                           mc.allow.recursive = TRUE, mc.silent = FALSE) {
  mc.cores <- tryCatch(as.integer(mc.cores), warning = function(w) NA_integer_)
  if (is.na(mc.cores) || mc.cores <= 1L) {
    return(lapply(X, FUN, ...))
  }
  if (.Platform$OS.type == "windows") {
    backend <- Sys.getenv("AUTOZYME_PARALLEL", unset = "lapply")
    if (identical(tolower(backend), "future") &&
        requireNamespace("future.apply", quietly = TRUE) &&
        requireNamespace("future", quietly = TRUE)) {
      old_plan <- future::plan(future::multisession, workers = mc.cores)
      on.exit(future::plan(old_plan), add = TRUE)
      return(future.apply::future_lapply(X, FUN, ..., future.seed = TRUE))
    }
    return(lapply(X, FUN, ...))
  }
  parallel::mclapply(X, FUN, ..., mc.cores = mc.cores,
                     mc.preschedule = mc.preschedule,
                     mc.set.seed = mc.set.seed, mc.cleanup = mc.cleanup,
                     mc.allow.recursive = mc.allow.recursive, mc.silent = mc.silent)
}

#' Resolve a task.yaml `datasets[i]$path` field to an actual local file
#'
#' task.yaml's `path` may be:
#' \itemize{
#'   \item a clean relative path like \code{"data/inputs_tiny.rds"}
#'   \item a \code{./...} relative path
#'   \item an absolute path (sometimes stale from a previous repo layout)
#'   \item a Windows path (\code{D:/...}) on tasks ported from a different OS
#' }
#'
#' Tries (in order): as-is if absolute, else \code{task_dir/path}; then
#' \code{task_dir/} with a leading \code{./} stripped; then
#' \code{task_dir/data/<basename>}. Errors if nothing exists.
#'
#' @param task_dir path to the autozyme task directory.
#' @param raw_path the value of \code{datasets[[i]]$path} from task.yaml.
#' @return Existing absolute path.
#' @export
resolve_dataset_path <- function(task_dir, raw_path) {
  candidates <- if (.Platform$file.sep == "/" && substr(raw_path, 1, 1) == "/" ||
                    grepl("^[A-Za-z]:[\\/]", raw_path)) {
    raw_path
  } else {
    file.path(task_dir, raw_path)
  }
  candidates <- c(
    candidates,
    file.path(task_dir, sub("^\\./", "", raw_path)),
    file.path(task_dir, "data", basename(raw_path))
  )
  for (c in candidates) {
    if (file.exists(c)) return(normalizePath(c, mustWork = TRUE))
  }
  stop(sprintf(
    "could not resolve dataset path '%s' from task_dir '%s'; tried: %s",
    raw_path, task_dir, paste(candidates, collapse = ", ")
  ))
}
