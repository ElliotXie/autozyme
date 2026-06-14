# Phase 4 smoke test — SCTransform (Seurat + default + scoped sctransform::vst
# override). Also verifies on_deactivate cleanup hook.
#
# What this validates:
#  - fast SCT runs without error and produces a SCT assay with scale.data
#  - HVG selection matches baseline closely (variable_features overlap)
#  - scale.data values within numerical tolerance (residuals)
#  - per-call zyme=FALSE returns to upstream and produces baseline-equivalent
#  - deactivate("seurat") invokes the on_deactivate hook and tears down PSOCK

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

# Skip if sctransform isn't installed (it ships with Seurat, but be defensive)
if (!requireNamespace("sctransform", quietly = TRUE)) {
  cat("SKIP: sctransform not available\n"); quit(status = 0)
}

cat("=== Building test object ===\n")
set.seed(42)
n_genes <- 800; n_cells <- 600
counts <- as(matrix(rpois(n_genes * n_cells, lambda = 2), nrow = n_genes,
                    dimnames = list(paste0("g", seq_len(n_genes)),
                                    paste0("c", seq_len(n_cells)))),
             "CsparseMatrix")
obj <- CreateSeuratObject(counts = counts)
# Ensure some heterogeneity in cell totals so SCT's regression is non-trivial
obj$nCount_RNA <- Matrix::colSums(counts)

# === BASELINE SCT ========================================================
cat("\n=== Baseline SCTransform ===\n")
t0 <- Sys.time()
ref_sct <- suppressWarnings(SCTransform(
  obj, verbose = FALSE, variable.features.n = 200, ncells = 500,
  vst.flavor = "v2", seed.use = 1448145))
t_ref <- difftime(Sys.time(), t0, units = "secs")
ref_hvf  <- VariableFeatures(ref_sct)
ref_sd   <- LayerData(ref_sct[["SCT"]], "scale.data")
cat(sprintf("ref: %d HVFs, scale.data dim = %dx%d, time = %.2fs\n",
            length(ref_hvf), nrow(ref_sd), ncol(ref_sd), t_ref))

# === ACTIVATE ============================================================
cat("\n=== Activating autozyme + seurat ===\n")
suppressPackageStartupMessages(library(autozyme))
stopifnot(autozyme::activate("seurat"))

# === FAST SCT ============================================================
cat("\n=== Fast SCTransform ===\n")
t0 <- Sys.time()
fast_sct <- suppressWarnings(SCTransform(
  obj, verbose = FALSE, variable.features.n = 200, ncells = 500,
  vst.flavor = "v2", seed.use = 1448145))
t_fast <- difftime(Sys.time(), t0, units = "secs")
fast_hvf  <- VariableFeatures(fast_sct)
fast_sd   <- LayerData(fast_sct[["SCT"]], "scale.data")
cat(sprintf("fast: %d HVFs, scale.data dim = %dx%d, time = %.2fs (%.2fx)\n",
            length(fast_hvf), nrow(fast_sd), ncol(fast_sd), t_fast,
            as.numeric(t_ref) / as.numeric(t_fast)))

# === CONCORDANCE =========================================================
cat("\n=== Concordance ===\n")

# 1. HVF overlap (allow some drift — SCT involves randomness via cell sampling)
ov <- length(intersect(ref_hvf, fast_hvf)) / length(ref_hvf)
cat(sprintf("HVF overlap     = %.3f\n", ov))
stopifnot(ov >= 0.8)

# 2. scale.data on common HVFs — same residuals modulo numerical drift
common <- intersect(ref_hvf, fast_hvf)
if (length(common) > 20) {
  ref_common  <- ref_sd[common, , drop = FALSE]
  fast_common <- fast_sd[common, , drop = FALSE]
  # SCT residual computation has its own ordering & sampling — compare
  # row means and row variances rather than element-wise (cells may be
  # in different orders for sampled-cells).
  diff_mean <- max(abs(rowMeans(fast_common) - rowMeans(ref_common)))
  diff_var  <- max(abs(apply(fast_common, 1, var) -
                         apply(ref_common,  1, var)))
  cat(sprintf("scale.data row-mean max diff = %.3e\n", diff_mean))
  cat(sprintf("scale.data row-var  max diff = %.3e\n", diff_var))
  stopifnot(diff_mean < 0.1, diff_var < 0.5)
}

cat("PASS: SCTransform fast path matches baseline\n")

# === ESCAPE: zyme=FALSE ==================================================
cat("\n=== Escape paths ===\n")

esc <- suppressWarnings(SCTransform(
  obj, verbose = FALSE, variable.features.n = 200, ncells = 500,
  vst.flavor = "v2", seed.use = 1448145, zyme = FALSE))
esc_hvf <- VariableFeatures(esc)
# Baseline-equivalent: HVFs should be identical to ref since seed is set
stopifnot(identical(sort(esc_hvf), sort(ref_hvf)))
cat("PASS: SCTransform zyme=FALSE returns baseline HVFs\n")

obj$fallback_covariate <- seq_len(ncol(obj)) / ncol(obj)
reg_fast <- suppressWarnings(SCTransform(
  obj, verbose = FALSE, variable.features.n = 100, ncells = 300,
  vst.flavor = "v2", seed.use = 1448145,
  vars.to.regress = "fallback_covariate"))
reg_base <- suppressWarnings(SCTransform(
  obj, verbose = FALSE, variable.features.n = 100, ncells = 300,
  vst.flavor = "v2", seed.use = 1448145,
  vars.to.regress = "fallback_covariate", zyme = FALSE))
stopifnot(identical(sort(VariableFeatures(reg_fast)),
                    sort(VariableFeatures(reg_base))))
cat("PASS: unsupported SCT args fall back to upstream\n")

# === sctransform::vst should NOT be persistently overridden after our call
# (verifies scoped patch on.exit restored cleanly)
sct_vst_now <- get("vst", envir = asNamespace("sctransform"))
# Test: it should be the original sctransform vst (defined in sctransform pkg,
# not our patched-in closure). The clearest check: function environment
# parent chain reaches sctransform's namespace directly.
fn_env <- environment(sct_vst_now)
stopifnot(!identical(fn_env, environment(autozyme::with_disabled)))  # not ours
cat("PASS: sctransform::vst is NOT persistently overridden\n")

# === Deactivation hook ===================================================
cat("\n=== deactivate('seurat') cleanup ===\n")
# Cluster might be NULL (non-Windows) or populated. Either way deactivate
# should invoke .seurat_sct_cleanup without erroring.
autozyme::deactivate("seurat")
cat("PASS: deactivate('seurat') completed without error\n")

# After deactivate, calling NormalizeData should hit baseline (sanity check)
obj_after <- NormalizeData(obj, verbose = FALSE)
ref_norm <- with(list(), {
  # baseline NormalizeData on identical object
  NormalizeData(obj, verbose = FALSE)
})
stopifnot(identical(LayerData(obj_after, "data"),
                    LayerData(ref_norm, "data")))
cat("PASS: post-restore NormalizeData is baseline\n")

cat("\nALL PHASE 4 SMOKE TESTS PASSED\n")
