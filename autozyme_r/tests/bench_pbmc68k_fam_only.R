# Minimal FAM-only timing on pbmc68k, after -O3 rebuild.
suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(autozyme)
})
obj <- readRDS("D:/autosearch/datasets/single_cell/pbmc68k.rds")
stopifnot(autozyme::activate("seurat"))

# Quick preprocess to get clusters
obj <- NormalizeData(obj, verbose = FALSE)
obj <- FindVariableFeatures(obj, nfeatures = 2000, verbose = FALSE)
obj <- ScaleData(obj, verbose = FALSE)
obj <- RunPCA(obj, npcs = 50, verbose = FALSE, seed.use = 42L)
obj <- FindNeighbors(obj, dims = 1:30, verbose = FALSE)
obj <- FindClusters(obj, resolution = 0.5, verbose = FALSE)
cat(sprintf("preproc done: %d clusters\n", length(unique(Idents(obj)))))

# === FAM timing (run twice to factor out warm-up) ===
for (i in 1:2) {
  gc(verbose = FALSE)
  t <- Sys.time()
  m <- suppressWarnings(FindAllMarkers(
    obj, only.pos = TRUE, min.pct = 0.1,
    logfc.threshold = 0.25, verbose = FALSE))
  dt <- as.numeric(difftime(Sys.time(), t, units = "secs"))
  cat(sprintf("FAM run %d:  %.2fs  (%d marker rows)\n", i, dt, nrow(m)))
}
