# End-to-end benchmark on pbmc68k.rds.
#
# Runs full Seurat pipeline twice: baseline (with autozyme disabled) then
# fast (with all 18 seurat patches active). Compares per-step timings and
# concordance on key outputs.
#
# Designed for "did Phase 1-4 actually deliver speedups?" — not a unit test.

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

DATA_PATH <- "D:/autosearch/datasets/single_cell/pbmc68k.rds"

cat(sprintf("=== Loading %s ===\n", DATA_PATH))
t0 <- Sys.time()
obj <- readRDS(DATA_PATH)
cat(sprintf("loaded: %d genes x %d cells (%.1fs)\n",
            nrow(obj), ncol(obj),
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))

# Time helper
.t <- function(expr, label) {
  gc(verbose = FALSE)
  t <- Sys.time()
  res <- eval.parent(substitute(expr))
  dt <- as.numeric(difftime(Sys.time(), t, units = "secs"))
  cat(sprintf("  %-22s %7.2fs\n", label, dt))
  list(res = res, t = dt)
}

run_pipeline <- function(obj, label) {
  cat(sprintf("\n=== %s ===\n", label))
  r <- list()
  r$norm   <- .t(NormalizeData(obj, verbose = FALSE),                            "NormalizeData")
  r$hvg    <- .t(FindVariableFeatures(r$norm$res, nfeatures = 2000, verbose=FALSE), "FindVariableFeatures")
  r$scale  <- .t(ScaleData(r$hvg$res, verbose = FALSE),                           "ScaleData")
  r$pca    <- .t(RunPCA(r$scale$res, npcs = 50, verbose = FALSE, seed.use = 42L), "RunPCA")
  r$nn     <- .t(FindNeighbors(r$pca$res, dims = 1:30, verbose = FALSE),          "FindNeighbors")
  r$umap   <- .t(RunUMAP(r$nn$res, dims = 1:30, verbose = FALSE, seed.use = 42L), "RunUMAP")
  # Cluster for FAM
  cl_obj  <- FindClusters(r$nn$res, resolution = 0.5, verbose = FALSE)
  r$fam    <- .t(suppressWarnings(FindAllMarkers(
                   cl_obj, only.pos = TRUE, min.pct = 0.1,
                   logfc.threshold = 0.25, verbose = FALSE)),
                 "FindAllMarkers")
  r$total <- Reduce(`+`, lapply(r, function(x) if (is.list(x)) x$t else 0))
  cat(sprintf("  %-22s %7.2fs\n", "TOTAL", r$total))
  r
}

# === BASELINE ============================================================
suppressPackageStartupMessages(library(autozyme))
# Activate then immediately disable so we use the same library config
# (Matrix loaded, etc.) for both runs; turn off via env var would skip
# patches entirely.
autozyme::activate("seurat")
ref <- autozyme::with_disabled(run_pipeline(obj, "BASELINE (autozyme disabled)"))

# === FAST ================================================================
fast <- run_pipeline(obj, "FAST (autozyme active)")

# === SUMMARY =============================================================
cat("\n=== SPEEDUP ===\n")
steps <- c("norm", "hvg", "scale", "pca", "nn", "umap", "fam")
for (s in steps) {
  rt <- ref[[s]]$t
  ft <- fast[[s]]$t
  cat(sprintf("  %-22s ref=%7.2fs  fast=%7.2fs  %5.2fx\n",
              s, rt, ft, rt / ft))
}
cat(sprintf("  %-22s ref=%7.2fs  fast=%7.2fs  %5.2fx\n",
            "TOTAL", ref$total, fast$total, ref$total / fast$total))

# === CONCORDANCE ==========================================================
cat("\n=== CONCORDANCE ===\n")

# 1. NormalizeData (exact)
nd <- max(abs(LayerData(ref$norm$res,  "data") -
              LayerData(fast$norm$res, "data")))
cat(sprintf("NormalizeData  max abs diff       = %.3e\n", nd))

# 2. HVF overlap
ref_hvf  <- VariableFeatures(ref$hvg$res)
fast_hvf <- VariableFeatures(fast$hvg$res)
ov <- length(intersect(ref_hvf, fast_hvf)) / length(ref_hvf)
cat(sprintf("HVG overlap                       = %.4f  (%d/%d)\n",
            ov, length(intersect(ref_hvf, fast_hvf)), length(ref_hvf)))

# 3. ScaleData on common HVFs
common <- intersect(ref_hvf, fast_hvf)
if (length(common) > 100) {
  sd_diff <- max(abs(
    LayerData(ref$scale$res,  "scale.data")[common, ] -
    LayerData(fast$scale$res, "scale.data")[common, ]))
  cat(sprintf("ScaleData (common HVF) max diff   = %.3e\n", sd_diff))
}

# 4. PCA stdev
ref_sdev  <- Stdev(ref$pca$res,  reduction = "pca")
fast_sdev <- Stdev(fast$pca$res, reduction = "pca")
cat(sprintf("PCA stdev max abs diff            = %.3e\n",
            max(abs(ref_sdev - fast_sdev))))

# 5. FindNeighbors graph density
ref_snn  <- ref$nn$res[["RNA_snn"]]
fast_snn <- fast$nn$res[["RNA_snn"]]
cat(sprintf("FindNeighbors SNN nnz ratio       = %.4f  (fast/ref)\n",
            length(fast_snn@x) / length(ref_snn@x)))

# 6. UMAP shape (stochastic, just shape)
cat(sprintf("UMAP shape                        = %dx%d  (both finite: %s)\n",
            nrow(Embeddings(fast$umap$res, "umap")),
            ncol(Embeddings(fast$umap$res, "umap")),
            all(is.finite(Embeddings(fast$umap$res, "umap")))))

# 7. FindAllMarkers top-50 by cluster overlap
ref_top  <- by(ref$fam$res,  ref$fam$res$cluster,  function(x) head(x$gene, 50))
fast_top <- by(fast$fam$res, fast$fam$res$cluster, function(x) head(x$gene, 50))
clusters <- intersect(names(ref_top), names(fast_top))
if (length(clusters) > 0) {
  ov_per <- sapply(clusters, function(c)
    length(intersect(ref_top[[c]], fast_top[[c]])) /
    max(length(ref_top[[c]]), 1))
  cat(sprintf("FindAllMarkers top-50/cluster ov  = %.3f mean across %d clusters\n",
              mean(ov_per), length(clusters)))
}

cat("\nDONE\n")
