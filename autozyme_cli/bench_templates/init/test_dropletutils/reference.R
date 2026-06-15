#!/usr/bin/env Rscript
# reference.R — Generate ground truth for <TASK_NAME>.
#
# Run once per tier: `Rscript reference.R` (the zyme runner sets
# ZYME_TIER / ZYME_DATA_PATH / ZYME_REFERENCE_DIR env vars).
# Saves output to reference_output*/ for evaluate.R to compare against.
#
# This file is READ-ONLY during experiments — the main agent does not edit it.
# Init agent fills in the placeholders below from the upstream source.

suppressPackageStartupMessages({
  # === Imports for the upstream toolkit ===
  # library(Seurat)
  # library(Matrix)
  # library(SingleCellExperiment)
})

# === Locate self ===
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("--file=", args, fixed = TRUE)
  if (length(m) > 0) {
    return(dirname(normalizePath(sub("--file=", "", args[m[1]], fixed = TRUE))))
  }
  getwd()
}
SCRIPT_DIR <- get_script_dir()
TASK_DIR   <- SCRIPT_DIR

# === Helpers from autozyme_cli/zyme/helpers.R ===
# Sourced unconditionally so `peak_memory_mb()` (cross-platform, ps-aware),
# `emit_summary()` (prints `speed_sec:` / `peak_mb:` / `cpu_sec:` in the
# format the zyme runner parses), and `get_tier_params()` are all available.
# Walks up to find the framework repo (autozyme/ or autozyme-framework/).
# Do NOT inline a `gc()`-based peak_mb: R's `gc()` matrix columns are
# version-dependent and easy to mis-index (raw cell counts vs MB).
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

# === Tier-aware paths (zyme runner provides via env vars) ===
TIER       <- Sys.getenv("ZYME_TIER", unset = "tiny")
DATA_PATH  <- Sys.getenv("ZYME_DATA_PATH",
                         unset = file.path(TASK_DIR, "<DATA_FILE>"))
DATA_PATH  <- gsub('^"|"$', '', DATA_PATH)  # strip stray quotes
OUTPUT_DIR <- Sys.getenv("ZYME_REFERENCE_DIR",
  unset = file.path(TASK_DIR, sprintf("reference_output_%s", TIER)))
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
cat(sprintf("[reference] tier=%s data=%s\n", TIER, DATA_PATH))

# === Stochastic algorithm? Read seed from env var ===
# If your target uses an RNG (MCMC / EM / random init / approximate methods),
# read the seed from `ZYME_RANDOM_SEED` (default 42). `zyme baseline noise` sets
# it to calibration values (default 43,44,45) to measure intrinsic noise — without
# this hook, calibration runs the same chain twice and reports zero noise.
# Skip this block for deterministic algorithms.
# SEED <- as.integer(Sys.getenv("ZYME_RANDOM_SEED", "42"))
# set.seed(SEED)
# # framework-specific seed setters (uncomment as needed):
# # RcppParallel::setThreadOptions(numThreads = 1L)  # only for testing reproducibility

# === Upstream threading (uncomment if `task.yaml::upstream_parallelism` is non-empty) ===
# Mirror the threading wiring from pipeline/run.R so `zyme baseline reference
# --thread N` actually engages the upstream knob — without this, baseline stays
# single-threaded even when --thread > 1 is requested.
#   N_THREADS <- get_threads(default = 1L)
#   options(mc.cores = N_THREADS)
#   # Then pass N_THREADS into the upstream knob — `BPPARAM = MulticoreParam(N_THREADS)`,
#   # `parallel = TRUE`, `nthreads = N_THREADS`, etc.


# === 1. Load input data ===
if (!file.exists(DATA_PATH)) {
  stop("Input not found at ", DATA_PATH, ". Did `zyme init` finish?")
}

cat("[reference] Loading data...\n")
# obj <- readRDS(DATA_PATH)
# cat(sprintf("[reference] Dataset: %d cells, %d genes\n", ncol(obj), nrow(obj)))


# === 2. Run upstream baseline ===
cat("[reference] Running upstream baseline...\n")
t0 <- Sys.time()
# result <- upstream_function(obj, ...)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("[reference] Baseline took %.1fs\n", elapsed))


# === 3. Serialize output for evaluate.R ===
# Use whatever format captures everything evaluate.R needs to compare.
# Common patterns:
#   - any R object:  saveRDS(result, file.path(OUTPUT_DIR, "result.rds"))
#   - dense matrix:  Matrix::writeMM(M, file.path(OUTPUT_DIR, "M.mtx"))
#   - data frame:    write.csv(df, file.path(OUTPUT_DIR, "df.csv"), row.names = FALSE)
#   - named list:    saveRDS(list(coef = C, pval = P), file.path(OUTPUT_DIR, "fit.rds"))
#
# `pipeline/run.R` MUST save in IDENTICAL format and key names — evaluate.R
# loads from both sides symmetrically.

# === 4. Print baseline summary ===
# emit_summary() prints `speed_sec: <float>` and `peak_mb: <float>` (and
# `cpu_sec:`) in the exact format `zyme baseline rebench` / `zyme baseline
# record --from-log` parse. Do NOT replace with ad-hoc `[reference] elapsed:`
# lines — they bypass the parser.
cat(sprintf("\n[reference] Saved to %s\n", OUTPUT_DIR))
emit_summary(speed_sec = elapsed)
cat(sprintf("[reference] Baseline speed: %.1f seconds — this is the time to beat.\n", elapsed))
