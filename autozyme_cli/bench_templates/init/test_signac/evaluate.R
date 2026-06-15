#!/usr/bin/env Rscript
# evaluate.R — Compare optimized output against reference.
#
# Called by the zyme runner after each experiment.
# Prints `metric_name: value` lines for the runner to parse into
# results.tsv (`metrics_json` column).
#
# This file is READ-ONLY during experiments — the main agent does not edit it.
# Init agent writes the metrics inline based on the function's mathematical
# nature (deterministic numerical → pearson + max-abs-diff; clustering → ARI/NMI;
# embedding → rotation-invariant; ranked lists → top-K Jaccard).

suppressPackageStartupMessages({
  # === Add metric-specific imports as needed ===
  # library(Matrix)
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
TIER       <- Sys.getenv("ZYME_TIER", unset = "tiny")

# Reference dir: per-tier under task root unless overridden.
REF_DIR  <- Sys.getenv("ZYME_REFERENCE_DIR",
  unset = file.path(TASK_DIR, sprintf("reference_output_%s", TIER)))

# ZYME_TEST_DIR lets `zyme baseline noise` point evaluate at a non-pipeline
# directory so it can compare two reference runs (primary vs calibration seed).
# Default = <task>/pipeline/.
TEST_DIR <- Sys.getenv("ZYME_TEST_DIR", unset = file.path(TASK_DIR, "pipeline"))

if (!dir.exists(REF_DIR)) {
  stop("Reference output not found at ", REF_DIR, ". Run reference.R first.")
}

# Framework helpers for F3 auto-compare (default-on, opt-out).
.fw <- normalizePath(TASK_DIR, mustWork = FALSE)
.helpers_path <- NULL
while (is.null(.helpers_path)) {
  for (.name in c("autozyme", "autozyme-framework")) {
    .h <- file.path(.fw, .name, "autozyme_cli", "zyme", "helpers.R")
    if (file.exists(.h)) { .helpers_path <- .h; break }
  }
  if (is.null(.helpers_path)) {
    .parent <- dirname(.fw)
    if (.parent == .fw) stop("autozyme helpers.R not found above ", TASK_DIR)
    .fw <- .parent
  }
}
source(.helpers_path)


# === 1. Load reference + test outputs ===
# Mirror exactly what reference.R saved. Symmetric load on both sides.
# Example:
#   ref_path  <- file.path(REF_DIR,  "result.rds")
#   test_path <- file.path(TEST_DIR, "result.rds")
#   if (!file.exists(ref_path))  stop("Reference output missing: ",  ref_path)
#   if (!file.exists(test_path)) stop("Pipeline output missing: ",   test_path)
#   ref  <- readRDS(ref_path)
#   test <- readRDS(test_path)


# === 2. Compute metrics declared in task.yaml ===
# Each metric is (ref, opt) -> numeric. Convention: print "score" form (higher
# = better) for `gte` metrics, raw value for `lte` metrics (max_abs_diff, RMSE).
metrics <- list()

# Helper for the common pearson-on-flattened-matrix pattern (NaN-safe):
# pearson_score <- function(a, b) {
#   av <- as.numeric(a); bv <- as.numeric(b)
#   if (length(av) != length(bv)) return(0)
#   ok <- is.finite(av) & is.finite(bv)
#   if (!any(ok)) return(if (identical(av, bv)) 1 else 0)
#   av <- av[ok]; bv <- bv[ok]
#   if (length(av) < 2) return(if (identical(av, bv)) 1 else 0)
#   if (sd(av) == 0 || sd(bv) == 0) return(if (identical(av, bv)) 1 else 0)
#   suppressWarnings(cor(av, bv))
# }
#
# metrics$pearson_X       <- pearson_score(ref$X, test$X)
# metrics$max_abs_diff_X  <- max(abs(as.numeric(ref$X) - as.numeric(test$X)))


# === 3. F3 default-on structural check ===
# Auto-emits THREE boolean metrics per top-level slot in the reference output
# (1.0 = pass, 0.0 = fail):
#   <slot>_present         — test has the slot (catches dropped output)
#   <slot>_shape_match     — same shape / length as ref
#   <slot>_non_degenerate  — if ref has variance, test does too
#                             (catches "stub to constant" — e.g. an override
#                              returning 0 / NA / single value while ref varies)
#
# This is structure-level only, NOT value comparison. Pearson / max_abs_diff /
# Jaccard remain explicit metrics in `metrics` above — declare them for slots
# where you want strict value agreement.
#
# Waiver: append slot names to WAIVED_SLOTS only when a slot legitimately
# differs across runs (random init log, iteration history of variable length).
# Each waiver REQUIRES a one-line `# diagnostic: <reason>` comment — audit
# flags any waiver without one.
WAIVED_SLOTS <- character(0)  # e.g. c("loglikelihood_history")  # diagnostic: ...

struct_metrics <- auto_structure_check_all_slots(ref, test, waived = WAIVED_SLOTS)


# === 4. Print metrics in `key: value` format ===
# The zyme runner greps these lines from stdout and packs into
# results.tsv.metrics_json.
for (name in names(metrics)) {
  v <- metrics[[name]]
  if (is.numeric(v)) {
    cat(sprintf("%s: %.6f\n", name, v))
  } else {
    cat(sprintf("%s: %s\n", name, as.character(v)))
  }
}
# Auto-structure: one summary line when all perfect; individual lines for failures.
emit_auto_structure_summary(struct_metrics)

# Optional: extra diagnostic info (printed but not parsed as a metric).
# cat(sprintf("\n[evaluate] N elements compared: %d\n", length(as.numeric(ref$X))))
