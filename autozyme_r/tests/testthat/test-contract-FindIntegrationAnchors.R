# Contract: Seurat::FindIntegrationAnchors
#
# Patched fn has explicit `zyme = TRUE, turbo = NULL`. The anchor-finding
# pipeline internally calls RunCCA / ScaleData / FindAnchors (all of
# which are also patched). FindIntegrationAnchors needs a LIST of
# normalized + variable-feature-marked Seurat objects.

.make_two_hvg_seurats <- function() {
  # FindIntegrationAnchors needs enough cells per object for anchor
  # filtering (k.filter default 200; we set 30 below). Build slightly
  # larger Seurats specifically for this test rather than re-using
  # .make_hvg_seurat's 60-cell default.
  build_one <- function(seed) {
    set.seed(seed)
    n_cells <- 80
    n_genes <- 150
    counts <- matrix(stats::rpois(n_cells * n_genes, lambda = 2),
                     nrow = n_genes, ncol = n_cells)
    latent <- rep(c(0L, 5L), each = n_cells / 2)
    counts[1:20, ] <- counts[1:20, ] +
      matrix(rep(latent, each = 20), nrow = 20)
    rownames(counts) <- paste0("g", seq_len(n_genes))
    colnames(counts) <- paste0("c", seq_len(n_cells))
    obj <- suppressWarnings(
      Seurat::CreateSeuratObject(counts = counts, min.cells = 0,
                                 min.features = 0)
    )
    obj <- suppressWarnings(Seurat::NormalizeData(obj, verbose = FALSE,
                                                  zyme = FALSE))
    obj <- suppressWarnings(
      Seurat::FindVariableFeatures(obj, nfeatures = 80, verbose = FALSE,
                                   zyme = FALSE)
    )
    obj
  }
  obj_a <- build_one(0)
  obj_b <- build_one(1)
  obj_b <- SeuratObject::RenameCells(obj_b, new.names = paste0("b_", colnames(obj_b)))
  list(obj_a, obj_b)
}

test_that("FindIntegrationAnchors returns AnchorSet on a 2-object list", {
  .skip_if_no_seurat()
  obj_list <- .make_two_hvg_seurats()
  out <- suppressWarnings(suppressMessages(
    Seurat::FindIntegrationAnchors(
      object.list = obj_list, anchor.features = 60, dims = 1:5,
      k.anchor = 3, k.filter = NA, k.score = 10, max.features = 60,
      n.trees = 10, verbose = FALSE)
  ))
  expect_s4_class(out, "IntegrationAnchorSet")
})

test_that("FindIntegrationAnchors zyme=FALSE returns same anchor set class", {
  .skip_if_no_seurat()
  obj_list <- .make_two_hvg_seurats()
  patched <- suppressWarnings(suppressMessages(
    Seurat::FindIntegrationAnchors(
      object.list = obj_list, anchor.features = 60, dims = 1:5,
      k.anchor = 3, k.filter = NA, k.score = 10, max.features = 60,
      n.trees = 10, verbose = FALSE)
  ))
  vanilla <- suppressWarnings(suppressMessages(autozyme::with_disabled(
    Seurat::FindIntegrationAnchors(
      object.list = obj_list, anchor.features = 60, dims = 1:5,
      k.anchor = 3, k.filter = NA, k.score = 10, max.features = 60,
      n.trees = 10, verbose = FALSE)
  )))
  expect_s4_class(patched, "IntegrationAnchorSet")
  expect_s4_class(vanilla, "IntegrationAnchorSet")
})
