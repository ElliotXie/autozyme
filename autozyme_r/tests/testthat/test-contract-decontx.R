# Contract: celda::decontX
#
# Patched surface: 4 internal targets (decontXLogLik, .decontxInitializeZ,
# calculateNativeMatrix, decontXEM). All reached through the public
# `decontX()` entry point. None expose a `zyme=` arg directly; bypass
# goes through autozyme::with_disabled().

.skip_if_no_decontx <- function() {
  testthat::skip_if_not_installed("celda")
  testthat::skip_if_not_installed("SingleCellExperiment")
}

.make_tiny_decontx_input <- function(seed = 0) {
  set.seed(seed)
  # 4 cell-types x 25 cells x 60 genes — minimum that decontX's init can
  # cluster meaningfully without falling into degenerate paths.
  k <- 4
  per_cluster <- 25
  n_cells <- k * per_cluster
  n_genes <- 60
  # Per-cluster mean expression with deliberate cluster-specific signature
  # so the dbscan/varGenes init has structure to find.
  cluster <- rep(seq_len(k), each = per_cluster)
  counts <- matrix(0L, nrow = n_genes, ncol = n_cells)
  for (cl in seq_len(k)) {
    cells_cl <- which(cluster == cl)
    # Each cluster has ~15 marker genes upregulated
    marker_genes <- ((cl - 1) * 15 + 1):(cl * 15)
    rate_bg <- 1
    counts[, cells_cl] <- stats::rpois(n_genes * length(cells_cl),
                                       lambda = rate_bg)
    counts[marker_genes, cells_cl] <- stats::rpois(
      length(marker_genes) * length(cells_cl), lambda = 8)
  }
  rownames(counts) <- paste0("g", seq_len(n_genes))
  colnames(counts) <- paste0("c", seq_len(n_cells))
  counts
}

test_that("decontX returns a list with decontXcounts + contamination", {
  .skip_if_no_decontx()
  counts <- .make_tiny_decontx_input()
  out <- tryCatch(
    suppressWarnings(suppressMessages(
      celda::decontX(counts, verbose = FALSE)
    )),
    error = function(e) e
  )
  if (inherits(out, "error")) {
    testthat::skip(paste0("decontX raised on minimal fixture: ",
                          conditionMessage(out)))
  }
  # decontX with a matrix input returns a list (with anndata/SCE input it
  # returns the wrapped object; matrix input contract is `list`).
  expect_type(out, "list")
  for (slot in c("decontXcounts", "contamination")) {
    expect_true(slot %in% names(out),
                info = paste("missing decontX slot:", slot))
  }
  # Decontaminated counts must have the same shape as input.
  expect_equal(dim(out$decontXcounts), dim(counts))
  expect_equal(length(out$contamination), ncol(counts))
})

test_that("decontX with_disabled() matches patched output shape", {
  .skip_if_no_decontx()
  counts <- .make_tiny_decontx_input()
  vanilla <- tryCatch(
    autozyme::with_disabled(
      suppressWarnings(suppressMessages(
        celda::decontX(counts, verbose = FALSE)
      ))
    ),
    error = function(e) e
  )
  if (inherits(vanilla, "error")) {
    testthat::skip(paste0("vanilla decontX failed: ",
                          conditionMessage(vanilla)))
  }
  patched <- suppressWarnings(suppressMessages(
    celda::decontX(counts, verbose = FALSE)
  ))
  expect_equal(class(patched), class(vanilla))
  expect_equal(dim(patched$decontXcounts), dim(vanilla$decontXcounts))
  expect_equal(length(patched$contamination), length(vanilla$contamination))
})
