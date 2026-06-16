# Subprocess worker for verify_patch — one measurement per invocation.
#
# Spawned by autozyme:::.run_worker once for baseline and once for patched,
# per rep. The parent runs:
#
#   Rscript <pkg>/inst/verify_worker.R \
#     --patch <name> --task-dir <abs> --tier <name> \
#     --output-dir <abs> [--activate]
#
# Inside the subprocess we:
#   1. library(autozyme); .ensure_registered(name) — needed even for
#      baseline since baseline still calls patch$smoke[[...]].
#   2. If --activate: autozyme::activate(name).
#   3. Build inputs via patch$smoke$load(task_dir, tier)        — untimed.
#   4. Time only result <- patch$smoke$call(inputs)              — the timed
#      region (Sys.time() difftime in seconds).
#   5. Save outputs via patch$smoke$save(result, output_dir, tier = tier).
#   6. Print a single JSON line on stdout: {"elapsed_sec": <float>}.
#
# Rationale: in-process restore() → baseline call → activate() → patched
# call previously required state-reset code (model reconstructions, seed
# resets) inside _smoke_call, which contaminated the timed region. Fresh
# subprocesses eliminate the contamination — smoke recipes need no reset
# boilerplate.

suppressPackageStartupMessages({
  library(autozyme)
})

autozyme:::.sync_worker_thread_options()

.parse_args <- function(argv) {
  # Rscript -e '...' --args foo ... leaks the literal --args sentinel into
  # commandArgs(trailingOnly = TRUE) on R 4.5.0 (Mac/Linux). The file-mode
  # `Rscript file.R --args foo ...` form does NOT — the script-name boundary
  # consumes it. Filter it out here so the CLI's -e spawn pattern is robust.
  argv <- argv[argv != "--args"]
  out <- list(activate = FALSE)
  i <- 1L
  while (i <= length(argv)) {
    key <- argv[i]
    if (key == "--activate") {
      out$activate <- TRUE
      i <- i + 1L
      next
    }
    if (!startsWith(key, "--")) {
      stop(sprintf("verify_worker: unexpected positional arg '%s'", key))
    }
    if (i + 1L > length(argv)) {
      stop(sprintf("verify_worker: flag '%s' missing value", key))
    }
    val <- argv[i + 1L]
    nm <- gsub("-", "_", sub("^--", "", key), fixed = TRUE)
    out[[nm]] <- val
    i <- i + 2L
  }
  required <- c("patch", "task_dir", "tier", "output_dir")
  missing_keys <- setdiff(required, names(out))
  if (length(missing_keys)) {
    stop(sprintf(
      "verify_worker: missing required flags: %s",
      paste(paste0("--", gsub("_", "-", missing_keys, fixed = TRUE)),
            collapse = ", ")
    ))
  }
  out
}

argv <- commandArgs(trailingOnly = TRUE)
args <- .parse_args(argv)

# Force-register the patch (the registry is keyed off inst/patches/<name>.R
# which is sourced lazily). Baseline still needs smoke$load/call/save.
ok <- autozyme:::.ensure_registered(args$patch)
if (!isTRUE(ok)) {
  stop(sprintf(
    "verify_worker: patch '%s' did not register (upstream missing?)",
    args$patch
  ))
}
patch <- autozyme:::.zyme_registry[[args$patch]]
task_dir <- normalizePath(args$task_dir, mustWork = TRUE)
smoke <- autozyme:::.resolve_smoke(task_dir, patch)
if (is.null(smoke)) {
  stop(sprintf(
    "verify_worker: no smoke recipe for patch '%s' (add attest/smoke.R under task dir)",
    args$patch))
}

# `zyme package check-intercept` opts in via ZYME_INSTRUMENT_INTERCEPTS=1.
# The probe wraps each registered fast fn with a counter and writes JSON to
# ZYME_INTERCEPT_OUT at session exit. Sourced AFTER the patch is registered
# (so targets exist) and BEFORE activate() resolves dispatchers.
if (Sys.getenv("ZYME_INSTRUMENT_INTERCEPTS", "") == "1") {
  install_fn <- tryCatch(
    get(".zyme_intercept_install_from_env", envir = asNamespace("autozyme")),
    error = function(e) NULL
  )
  if (!is.null(install_fn)) install_fn()
}

if (isTRUE(args$activate)) {
  autozyme::activate(args$patch)
}

# Pass `tier` to smoke$call only if its signature accepts it (back-compat
# with smoke recipes that take just (inputs)).
#
# Do not use do.call() here. do.call() embeds argument values into a literal
# call object; Seurat-family code may inspect match.call()/sys.call() downstack
# and then traverse a call containing the full materialized Seurat object. That
# turns small DoubletFinder/Seurat smoke calls into multi-minute stalls.
.call_smoke <- function(input) {
  fmls <- names(formals(smoke$call))
  if ("tier" %in% fmls) {
    smoke$call(input, tier = args$tier)
  } else {
    smoke$call(input)
  }
}

inputs <- smoke$load(task_dir, args$tier)
t0 <- Sys.time()
result <- .call_smoke(inputs)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
smoke$save(result, args$output_dir, tier = args$tier)

# Single JSON line on stdout. Hand-roll to avoid a hard jsonlite dep — the
# payload is just one numeric field. Parent parses the last non-empty stdout
# line that looks like {"elapsed_sec": ...}.
cat(sprintf('{"elapsed_sec": %.9f}\n', elapsed))
flush.console()

# Some Seurat/uwot/sctransform sessions can hang during implicit R shutdown
# on Windows after the output has already been written. Kill this worker after
# flushing JSON; the parent treats a non-zero exit with valid JSON as success.
if (.Platform$OS.type == "windows") {
  system2("taskkill", c("/PID", as.character(Sys.getpid()), "/F"),
          stdout = FALSE, stderr = FALSE, wait = FALSE)
  Sys.sleep(60)
}

quit(save = "no", status = 0L, runLast = FALSE)
