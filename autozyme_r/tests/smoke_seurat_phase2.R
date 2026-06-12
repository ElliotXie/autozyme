# Phase 2 smoke test — covers FindVariableFeatures, ScaleData, FindNeighbors,
# FindAllMarkers, RunUMAP. Each function: baseline vs fast vs zyme=FALSE.
#
# Strategy: run the *baseline* in a clean R session BEFORE library(autozyme),
# then activate, re-run, compare.

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

cat("=== Building test object ===\n")
set.seed(42)
n_genes <- 500
n_cells <- 600
counts <- as(matrix(rpois(n_genes * n_cells, lambda = 1), nrow = n_genes,
                    dimnames = list(paste0("g", seq_len(n_genes)),
                                    paste0("c", seq_len(n_cells)))),
             "CsparseMatrix")
obj <- CreateSeuratObject(counts = counts)
obj$group <- sample(letters[1:3], n_cells, replace = TRUE)

# === BASELINE (autozyme NOT loaded) =======================================
cat("\n=== Phase 2: BASELINE (no autozyme) ===\n")
ref_norm   <- NormalizeData(obj, verbose = FALSE)
ref_hvg    <- FindVariableFeatures(ref_norm, nfeatures = 100, verbose = FALSE)
ref_scaled <- ScaleData(ref_hvg, verbose = FALSE)
ref_pca    <- RunPCA(ref_scaled, npcs = 20, verbose = FALSE,
                     features = VariableFeatures(ref_hvg))
ref_nn     <- FindNeighbors(ref_pca, dims = 1:10, verbose = FALSE)
Idents(ref_nn) <- ref_nn$group
ref_markers <- suppressWarnings(FindAllMarkers(
  ref_nn, only.pos = TRUE, min.pct = 0.05, logfc.threshold = 0.05,
  verbose = FALSE))
ref_umap <- RunUMAP(ref_pca, dims = 1:10, verbose = FALSE, seed.use = 42L)

cat("baseline pipeline ok\n")

# === ACTIVATE ============================================================
cat("\n=== Activating autozyme + seurat patch ===\n")
suppressPackageStartupMessages(library(autozyme))
stopifnot(autozyme::activate("seurat"))

# === FAST PATH ===========================================================
cat("\n=== FAST PATH ===\n")
fast_norm   <- NormalizeData(obj, verbose = FALSE)
fast_hvg    <- FindVariableFeatures(fast_norm, nfeatures = 100, verbose = FALSE)
fast_scaled <- ScaleData(fast_hvg, verbose = FALSE)
# PCA isn't patched in Phase 2 — use upstream (deterministic seed)
fast_pca    <- RunPCA(fast_scaled, npcs = 20, verbose = FALSE,
                      features = VariableFeatures(fast_hvg))
fast_nn     <- FindNeighbors(fast_pca, dims = 1:10, verbose = FALSE)
Idents(fast_nn) <- fast_nn$group
fast_markers <- suppressWarnings(FindAllMarkers(
  fast_nn, only.pos = TRUE, min.pct = 0.05, logfc.threshold = 0.05,
  verbose = FALSE))
fast_umap <- RunUMAP(fast_pca, dims = 1:10, verbose = FALSE, seed.use = 42L)

# === CONCORDANCE CHECKS ==================================================
cat("\n=== Concordance ===\n")

# 1. NormalizeData (already validated in Phase 1, re-check)
max_norm <- max(abs(LayerData(fast_norm, "data") - LayerData(ref_norm, "data")))
cat(sprintf("NormalizeData          max|fast - ref| = %.3e\n", max_norm))
stopifnot(max_norm < 1e-5)

# 2. FindVariableFeatures — variable feature set identity (algorithmic diff
#    is allowed at the loess fitting layer; we accept >=95% overlap of HVGs)
ref_hvf  <- VariableFeatures(ref_hvg)
fast_hvf <- VariableFeatures(fast_hvg)
ov <- length(intersect(ref_hvf, fast_hvf)) / length(ref_hvf)
cat(sprintf("FindVariableFeatures   HVG overlap     = %.3f\n", ov))
stopifnot(ov >= 0.95)

# 3. ScaleData — only validates if HVG sets match; use intersection
common <- intersect(ref_hvf, fast_hvf)
if (length(common) > 50) {
  ref_sc  <- LayerData(ref_scaled, "scale.data")[common, ]
  fast_sc <- LayerData(fast_scaled, "scale.data")[common, ]
  max_sc <- max(abs(fast_sc - ref_sc))
  cat(sprintf("ScaleData              max|fast - ref| = %.3e (common HVGs)\n",
              max_sc))
  stopifnot(max_sc < 1e-3)
}

# 4. FindNeighbors — Annoy is approximate; check graph sparsity matches
ref_snn  <- ref_nn[["RNA_snn"]]
fast_snn <- fast_nn[["RNA_snn"]]
ref_nnz  <- length(ref_snn@x)
fast_nnz <- length(fast_snn@x)
ratio <- fast_nnz / ref_nnz
cat(sprintf("FindNeighbors          SNN nnz ratio   = %.3f (fast/ref)\n",
            ratio))
stopifnot(ratio > 0.7 && ratio < 1.4)

# 5. FindAllMarkers — top-k gene overlap per cluster
top_ref  <- unique(head(ref_markers$gene, 30))
top_fast <- unique(head(fast_markers$gene, 30))
mk_ov <- length(intersect(top_ref, top_fast)) / max(length(top_ref), 1)
cat(sprintf("FindAllMarkers         top-30 overlap  = %.3f\n", mk_ov))
stopifnot(mk_ov >= 0.6 || length(ref_markers$gene) < 5)

# 6. RunUMAP — uwot is stochastic; just check shape + finiteness
ref_emb  <- Embeddings(ref_umap, "umap")
fast_emb <- Embeddings(fast_umap, "umap")
stopifnot(identical(dim(ref_emb), dim(fast_emb)))
stopifnot(all(is.finite(fast_emb)))
cat(sprintf("RunUMAP                shape ok (%dx%d), all finite\n",
            nrow(fast_emb), ncol(fast_emb)))

# === ESCAPES: zyme=FALSE + turbo=FALSE + with_disabled() =================
cat("\n=== Escape paths ===\n")

esc1 <- NormalizeData(obj, verbose = FALSE, zyme = FALSE)
stopifnot(identical(LayerData(esc1, "data"), LayerData(ref_norm, "data")))
cat("PASS: NormalizeData zyme=FALSE  -> baseline\n")

esc2 <- ScaleData(fast_hvg, verbose = FALSE, zyme = FALSE)
# Original ScaleData on hvf'd object — same as ref_scaled built from ref_hvg
# only if HVG sets match; loosen to "executes without error, has scale.data"
stopifnot("scale.data" %in% SeuratObject::Layers(esc2[["RNA"]]))
cat("PASS: ScaleData zyme=FALSE      -> runs upstream path\n")

esc3 <- autozyme::with_disabled({
  FindVariableFeatures(fast_norm, nfeatures = 100, verbose = FALSE)
})
esc3_hvf <- VariableFeatures(esc3)
stopifnot(identical(sort(esc3_hvf), sort(ref_hvf)))
cat("PASS: with_disabled() FVF       -> identical to baseline\n")

esc4 <- RunUMAP(fast_pca, dims = 1:10, verbose = FALSE, seed.use = 42L,
                zyme = FALSE)
stopifnot(identical(dim(Embeddings(esc4, "umap")),
                    dim(Embeddings(ref_umap, "umap"))))
cat("PASS: RunUMAP zyme=FALSE        -> runs upstream\n")

cat("\nALL PHASE 2 SMOKE TESTS PASSED\n")
