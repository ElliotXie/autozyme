# pbmc68k fast-path-only run. No baseline comparison — just confirms the
# full 18-target Seurat pipeline runs end-to-end on real data and records
# per-step wall-clock timings.

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(autozyme)
})

DATA <- "D:/autosearch/datasets/single_cell/pbmc68k.rds"

cat(sprintf("=== Loading %s ===\n", DATA))
t0 <- Sys.time()
obj <- readRDS(DATA)
cat(sprintf("loaded: %d genes x %d cells (%.1fs)\n",
            nrow(obj), ncol(obj),
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))

stopifnot(autozyme::activate("seurat"))

.t <- function(expr, label) {
  gc(verbose = FALSE)
  t <- Sys.time()
  res <- eval.parent(substitute(expr))
  dt <- as.numeric(difftime(Sys.time(), t, units = "secs"))
  cat(sprintf("  %-22s %7.2fs\n", label, dt))
  res
}

cat("\n=== FAST pipeline ===\n")
obj <- .t(NormalizeData(obj, verbose = FALSE),                            "NormalizeData")
obj <- .t(FindVariableFeatures(obj, nfeatures = 2000, verbose = FALSE),   "FindVariableFeatures")
obj <- .t(ScaleData(obj, verbose = FALSE),                                "ScaleData")
obj <- .t(RunPCA(obj, npcs = 50, verbose = FALSE, seed.use = 42L),        "RunPCA")
obj <- .t(FindNeighbors(obj, dims = 1:30, verbose = FALSE),               "FindNeighbors")
obj <- .t(FindClusters(obj, resolution = 0.5, verbose = FALSE),           "FindClusters")
obj <- .t(RunUMAP(obj, dims = 1:30, verbose = FALSE, seed.use = 42L),     "RunUMAP")
markers <- .t(suppressWarnings(FindAllMarkers(
                obj, only.pos = TRUE, min.pct = 0.1,
                logfc.threshold = 0.25, verbose = FALSE)),
              "FindAllMarkers")

cat(sprintf("\nResult: %d clusters, %d marker rows\n",
            length(unique(Idents(obj))), nrow(markers)))
cat(sprintf("HVGs: %d  PCs: %d  UMAP dim: %dx%d\n",
            length(VariableFeatures(obj)),
            ncol(Embeddings(obj, "pca")),
            nrow(Embeddings(obj, "umap")), ncol(Embeddings(obj, "umap"))))
cat("\nDONE\n")
