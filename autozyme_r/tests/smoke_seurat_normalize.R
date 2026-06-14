# Smoke test for Phase 1 of seurat patch (NormalizeData only).
# Run from anywhere with autozyme installed and Seurat available.
#
# Validates:
#   1. autozyme::activate("seurat") rewires NormalizeData.Seurat
#   2. fast path produces values within 1e-5 of upstream LogNormalize
#   3. fast path records a Seurat command log entry
#   4. unsupported NormalizeData options fall back to upstream
#   5. zyme=FALSE escape works (returns upstream result)
#   6. with_disabled() escape works
#   7. deactivate("seurat") un-patches cleanly
#   8. After restore, NormalizeData() == captured upstream original

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

# --- 1. Build a small Seurat object ---------------------------------------
set.seed(42)
n_genes <- 200
n_cells <- 300
counts <- as(matrix(rpois(n_genes * n_cells, lambda = 1), nrow = n_genes,
                    dimnames = list(paste0("g", seq_len(n_genes)),
                                    paste0("c", seq_len(n_cells)))),
             "CsparseMatrix")
obj <- CreateSeuratObject(counts = counts)

# --- 2. Baseline (no autozyme loaded yet) ---------------------------------
obj_ref <- NormalizeData(obj, verbose = FALSE)
ref_x <- LayerData(obj_ref, layer = "data")
cat("baseline NormalizeData done; data layer dim =",
    paste(dim(ref_x), collapse = "x"), "\n")

obj_ref_clr <- NormalizeData(obj, normalization.method = "CLR",
                             margin = 1, verbose = FALSE)
ref_clr_x <- LayerData(obj_ref_clr, layer = "data")

obj_ref_custom <- NormalizeData(obj, layer = "counts",
                                save = "data_custom", verbose = FALSE)
ref_custom_x <- LayerData(obj_ref_custom, layer = "data_custom")

obj_ref_block <- NormalizeData(obj, block.size = 50, verbose = FALSE)
ref_block_x <- LayerData(obj_ref_block, layer = "data")

obj_ref_margin <- NormalizeData(obj, margin = 2, verbose = FALSE)
ref_margin_x <- LayerData(obj_ref_margin, layer = "data")

# --- 3. Activate autozyme + seurat patch ----------------------------------
library(autozyme)
activated <- autozyme::activate("seurat")
stopifnot(isTRUE(activated))
cat("autozyme::activate('seurat') -> TRUE\n")

# --- 4. Fast path concordance ---------------------------------------------
obj_fast <- NormalizeData(obj, verbose = FALSE)
fast_x <- LayerData(obj_fast, layer = "data")
stopifnot(identical(dim(fast_x), dim(ref_x)))
max_abs <- max(abs(fast_x - ref_x))
cat(sprintf("max |fast - ref| = %.3e\n", max_abs))
stopifnot(max_abs < 1e-5)
cat("PASS: fast path within tolerance\n")

# --- 5. Fast path command log ---------------------------------------------
stopifnot(length(SeuratObject::Command(obj_fast)) > 0L)
cat("PASS: fast path records Seurat command log\n")

# --- 6. Unsupported options fall back --------------------------------------
obj_clr <- NormalizeData(obj, normalization.method = "CLR",
                         margin = 1, verbose = FALSE)
clr_x <- LayerData(obj_clr, layer = "data")
stopifnot(max(abs(clr_x - ref_clr_x)) == 0)
cat("PASS: non-LogNormalize falls back to upstream\n")

obj_custom <- NormalizeData(obj, layer = "counts",
                            save = "data_custom", verbose = FALSE)
custom_x <- LayerData(obj_custom, layer = "data_custom")
stopifnot("data_custom" %in% SeuratObject::Layers(obj_custom[["RNA"]]))
stopifnot(max(abs(custom_x - ref_custom_x)) == 0)
cat("PASS: custom layer/save falls back to upstream\n")

obj_block_size <- NormalizeData(obj, block.size = 50, verbose = FALSE)
block_x <- LayerData(obj_block_size, layer = "data")
stopifnot(max(abs(block_x - ref_block_x)) == 0)
cat("PASS: block.size falls back to upstream\n")

obj_margin <- NormalizeData(obj, margin = 2, verbose = FALSE)
margin_x <- LayerData(obj_margin, layer = "data")
stopifnot(max(abs(margin_x - ref_margin_x)) == 0)
cat("PASS: non-default margin falls back to upstream\n")

# --- 7. Per-call zyme=FALSE escape ----------------------------------------
obj_escape <- NormalizeData(obj, verbose = FALSE, zyme = FALSE)
escape_x <- LayerData(obj_escape, layer = "data")
stopifnot(identical(escape_x, ref_x))
cat("PASS: zyme=FALSE returns upstream result\n")

# --- 8. turbo=FALSE alias also works --------------------------------------
obj_escape2 <- NormalizeData(obj, verbose = FALSE, turbo = FALSE)
escape2_x <- LayerData(obj_escape2, layer = "data")
stopifnot(identical(escape2_x, ref_x))
cat("PASS: turbo=FALSE alias also returns upstream result\n")

# --- 9. with_disabled() block ---------------------------------------------
obj_block <- autozyme::with_disabled({
  NormalizeData(obj, verbose = FALSE)
})
disabled_x <- LayerData(obj_block, layer = "data")
stopifnot(identical(disabled_x, ref_x))
cat("PASS: with_disabled() forces upstream\n")

# --- 10. deactivate("seurat") un-patches ----------------------------------
autozyme::deactivate("seurat")
obj_after <- NormalizeData(obj, verbose = FALSE)
after_x <- LayerData(obj_after, layer = "data")
stopifnot(identical(after_x, ref_x))
cat("PASS: deactivate('seurat') yields upstream\n")

cat("\nALL SMOKE TESTS PASSED\n")
