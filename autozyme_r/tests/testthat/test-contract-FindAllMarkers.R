# Contract: Seurat::FindAllMarkers

.make_marker_seurat <- function(n_cells = 48, n_genes = 80, seed = 11) {
  .skip_if_no_seurat()
  set.seed(seed)
  counts <- matrix(stats::rpois(n_cells * n_genes, lambda = 1),
                   nrow = n_genes, ncol = n_cells)
  rownames(counts) <- paste0("g", seq_len(n_genes))
  colnames(counts) <- paste0("c", seq_len(n_cells))
  groups <- rep(c("a", "b"), each = n_cells / 2)
  counts[seq_len(6), groups == "a"] <- counts[seq_len(6), groups == "a"] + 8L
  counts[7:12, groups == "b"] <- counts[7:12, groups == "b"] + 8L
  obj <- suppressWarnings(
    Seurat::CreateSeuratObject(counts = counts, min.cells = 0,
                               min.features = 0)
  )
  obj <- suppressWarnings(Seurat::NormalizeData(obj, verbose = FALSE,
                                                zyme = FALSE))
  SeuratObject::Idents(obj) <- factor(groups)
  obj
}

test_that("FindAllMarkers returns a data.frame with marker columns", {
  .skip_if_no_seurat()
  obj <- .make_marker_seurat()
  out <- suppressWarnings(
    Seurat::FindAllMarkers(obj, only.pos = TRUE, verbose = FALSE,
                           min.pct = 0, logfc.threshold = 0)
  )
  expect_s3_class(out, "data.frame")
  for (col in c("p_val", "avg_log2FC", "cluster", "gene")) {
    expect_true(col %in% colnames(out), info = paste("missing col:", col))
  }
})

test_that("FindAllMarkers zyme=FALSE matches vanilla on the same input", {
  .skip_if_no_seurat()
  obj <- .make_marker_seurat()
  vanilla <- suppressWarnings(
    Seurat::FindAllMarkers(obj, only.pos = TRUE, verbose = FALSE,
                           min.pct = 0, logfc.threshold = 0, zyme = FALSE)
  )
  patched <- suppressWarnings(
    Seurat::FindAllMarkers(obj, only.pos = TRUE, verbose = FALSE,
                           min.pct = 0, logfc.threshold = 0)
  )
  # Gene set per cluster should match between vanilla and patched.
  for (cl in levels(SeuratObject::Idents(obj))) {
    g_v <- sort(vanilla$gene[vanilla$cluster == cl])
    g_p <- sort(patched$gene[patched$cluster == cl])
    expect_setequal(g_v, g_p)
  }
})
