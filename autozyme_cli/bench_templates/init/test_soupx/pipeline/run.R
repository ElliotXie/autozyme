# pipeline/run.R — agent's optimization patch.
#
# This is the ONLY file the main agent edits. Pattern:
#   1. source() the framework helpers for install_override / emit_summary.
#   2. Define optimized replacement(s) (full body OR thin wrapper).
#   3. Call install_override(func_name, package_name, replacement) — handles
#      namespace + package-env + locked-binding + verification + emits
#      `[override active]` runtime marker.
#   4. Run the same call as reference.R with same args.
#   5. Save output in IDENTICAL format/keys as reference.R — evaluate.R
#      compares symmetrically.
#
# Print `speed_sec: <float>` to stdout (or use emit_summary()).
#
# The first round (= round 0) must be unmodified — pipeline = reference,
# no overrides — to confirm baseline parity.

# Framework helpers — locate the framework repo (autozyme/ or autozyme-framework/)
# by walking up from this script. No symlink dependency.
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("--file=", args, fixed = TRUE)
  if (length(m) > 0) {
    return(dirname(normalizePath(sub("--file=", "", args[m[1]], fixed = TRUE))))
  }
  getwd()
}
SCRIPT_DIR <- get_script_dir()
.fw <- normalizePath(SCRIPT_DIR, mustWork = FALSE)
.helpers_path <- NULL
while (is.null(.helpers_path)) {
  for (.name in c("autozyme", "autozyme-framework")) {
    .h <- file.path(.fw, .name, "autozyme_cli", "zyme", "helpers.R")
    if (file.exists(.h)) { .helpers_path <- .h; break }
  }
  if (is.null(.helpers_path)) {
    .parent <- dirname(.fw)
    if (.parent == .fw) stop("autozyme helpers.R not found above ", SCRIPT_DIR)
    .fw <- .parent
  }
}
source(.helpers_path)

TASK_DIR <- dirname(SCRIPT_DIR)

# Dataset path: zyme runner sets ZYME_DATA_PATH per tier. Falls back to a
# task-default path if invoked outside the runner. ZYME_TIER tells you which
# tier you're on (tiny / medium / large) — useful for tier-conditional code.
DATA_PATH <- Sys.getenv("ZYME_DATA_PATH",
                        unset = file.path(TASK_DIR, "<DEFAULT_DATA_FILE>"))
TIER <- Sys.getenv("ZYME_TIER", unset = "tiny")
REFERENCE_DIR <- Sys.getenv("ZYME_REFERENCE_DIR",
                            unset = file.path(TASK_DIR, "reference_output"))

# Threading: route every hardcoded thread / core count through `get_threads()`
# so `zyme verify` actually exercises the matrix axis. Without this, the
# verify probe will abort.
#   N_THREADS <- get_threads(default = 4L)
#   options(mc.cores = N_THREADS)
#   # RcppParallel::setThreadOptions(numThreads = N_THREADS)
#   # Sys.setenv(OMP_NUM_THREADS = N_THREADS, MKL_NUM_THREADS = N_THREADS)

# Optional: tier-specific subset sizes from task.yaml `datasets[].params`.
# Use this in place of a hardcoded TIER_PARAMS list — adding a new tier
# later becomes yaml-only, no code edit.
#   P <- get_tier_params()
#   N_CELLS <- P$n_cells %||% 10000L
#   N_GENES <- P$n_genes %||% 2000L

# === Libraries — same as reference.R ===
# library(Seurat)


# ============================================================
# Optimized implementation
# ============================================================
# Full body replacement:
# fast_<target_function> <- function(object, ...) {
#   # Faster replacement, semantically equivalent within concordance threshold.
# }
#
# Or thin wrapper (preferred for one-parameter changes):
# orig_<target_function> <- getFromNamespace("<target_function>", "<package_name>")
# fast_<target_function> <- function(...) {
#   args <- list(...)
#   args$<param> <- <new_value>   # the only change
#   do.call(orig_<target_function>, args)
# }


# ============================================================
# Install override (uncomment when you have an optimized impl)
# ============================================================
# install_override("<target_function>", "<package_name>", fast_<target_function>)


# ============================================================
# Execute (mirror reference.R exactly — same input, same call, same save format)
# ============================================================
cat("[pipeline] Loading data...\n")
# obj <- readRDS(DATA_PATH)

cat("[pipeline] Running (possibly patched) function...\n")
t0 <- Sys.time()
result <- with_profile({
  # Use with_subprofile("name", { ... }) inside this block if you need named
  # timing for opaque native/library sub-calls during profiling.
  # <target_function>(obj, ...)
  NULL
})
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

# Save output to SCRIPT_DIR in EXACTLY the same format reference.R used,
# with identical keys / filenames. evaluate.R reads from both sides symmetrically.
# Example:
#   saveRDS(result, file.path(SCRIPT_DIR, "output.rds"))

emit_summary(speed_sec = elapsed, peak_mb = peak_memory_mb())
cat(sprintf("[pipeline] Done in %.1fs\n", elapsed))
