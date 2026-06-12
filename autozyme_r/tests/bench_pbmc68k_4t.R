# pbmc68k fast-path benchmark with thread count locked to 4 (matches
# paper's turbo_4t column).
#
# Threads are pinned in three places:
#   1. OMP_NUM_THREADS env var — picked up by OpenMP kernels (NormalizeData,
#      ScaleData, etc.) AND by our R wrappers that consult it
#      (FindNeighbors, RunUMAP, SCT init).
#   2. RhpcBLASctl — controls BLAS threads (matters for RunPCA Python path
#      and any internal lapack calls).
#   3. Python MKL/OpenBLAS via env vars set before reticulate init.

# === Pin threads BEFORE loading anything that uses BLAS / OpenMP ===
N_THREADS <- 4L

Sys.setenv(OMP_NUM_THREADS      = as.character(N_THREADS),
           MKL_NUM_THREADS      = as.character(N_THREADS),
           OPENBLAS_NUM_THREADS = as.character(N_THREADS),
           NUMEXPR_NUM_THREADS  = as.character(N_THREADS),
           VECLIB_MAXIMUM_THREADS = as.character(N_THREADS))

suppressPackageStartupMessages({
  library(Seurat); library(Matrix); library(autozyme)
})

if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
  RhpcBLASctl::blas_set_num_threads(N_THREADS)
  RhpcBLASctl::omp_set_num_threads(N_THREADS)
}

cat(sprintf("[bench] OMP_NUM_THREADS = %s\n",   Sys.getenv("OMP_NUM_THREADS")))
cat(sprintf("[bench] BLAS threads    = %s\n",
            tryCatch(RhpcBLASctl::blas_get_num_procs(), error = function(e) "?")))

obj <- readRDS("D:/autosearch/datasets/single_cell/pbmc68k.rds")
cat(sprintf("[bench] loaded %dx%d\n", nrow(obj), ncol(obj)))
stopifnot(autozyme::activate("seurat"))

.t <- function(expr, label) {
  gc(verbose = FALSE)
  t <- Sys.time()
  res <- eval.parent(substitute(expr))
  dt <- as.numeric(difftime(Sys.time(), t, units = "secs"))
  cat(sprintf("  %-22s %7.2fs\n", label, dt))
  res
}

cat("\n=== FAST pipeline (4 threads) ===\n")
obj <- .t(NormalizeData(obj, verbose = FALSE),                          "NormalizeData")
obj <- .t(FindVariableFeatures(obj, nfeatures = 2000, verbose = FALSE), "FindVariableFeatures")
obj <- .t(ScaleData(obj, verbose = FALSE),                              "ScaleData")
obj <- .t(RunPCA(obj, npcs = 50, verbose = FALSE, seed.use = 42L),      "RunPCA")
obj <- .t(FindNeighbors(obj, dims = 1:30, verbose = FALSE),             "FindNeighbors")
obj <- .t(FindClusters(obj, resolution = 0.5, verbose = FALSE),         "FindClusters")
obj <- .t(RunUMAP(obj, dims = 1:30, verbose = FALSE, seed.use = 42L),   "RunUMAP")
m   <- .t(suppressWarnings(FindAllMarkers(obj, only.pos = TRUE,
                                           min.pct = 0.1,
                                           logfc.threshold = 0.25,
                                           verbose = FALSE)),
          "FindAllMarkers")

cat(sprintf("\nResult: %d clusters, %d markers\n",
            length(unique(Idents(obj))), nrow(m)))
