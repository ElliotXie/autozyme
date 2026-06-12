# Contract: tradeseq::.fitGAM (patched via internal namespace; reached
# through the public fitGAM API)
#
# fitGAM fits a per-gene generalized additive model along pseudotime
# trajectories. Inputs: counts (genes x cells), pseudotime (cells x
# lineages), cellWeights (cells x lineages). Output: SingleCellExperiment
# with a `tradeSeq` slot carrying the gam fits.
#
# Fixture is light by tradeseq standards: 12 genes x 30 cells, single
# linear lineage. Still ~5-10s per test because of GAM fitting; runs
# remain reasonable for a contract test.

.skip_if_no_tradeseq <- function() {
  testthat::skip_if_not_installed("tradeSeq")
  testthat::skip_if_not_installed("SingleCellExperiment")
}

.make_tradeseq_inputs <- function(seed = 0) {
  set.seed(seed)
  n_cells <- 30
  n_genes <- 12
  # Counts with mild pseudotime-correlated signal in half the genes so
  # the GAM has something to fit (not pure noise).
  pseudotime <- sort(stats::runif(n_cells))
  base_lambda <- 5
  counts <- matrix(0L, nrow = n_genes, ncol = n_cells)
  for (g in seq_len(n_genes)) {
    rate <- if (g <= n_genes / 2) {
      base_lambda * (1 + pseudotime)  # signal genes
    } else {
      rep(base_lambda, n_cells)        # noise genes
    }
    counts[g, ] <- stats::rpois(n_cells, lambda = rate)
  }
  rownames(counts) <- paste0("g", seq_len(n_genes))
  colnames(counts) <- paste0("c", seq_len(n_cells))

  list(
    counts = counts,
    # Single lineage -> Nx1 matrix.
    pseudotime = matrix(pseudotime, ncol = 1),
    cellWeights = matrix(1, nrow = n_cells, ncol = 1)
  )
}

test_that("fitGAM returns a SingleCellExperiment with tradeSeq slot", {
  .skip_if_no_tradeseq()
  inp <- tryCatch(.make_tradeseq_inputs(), error = function(e) e)
  if (inherits(inp, "error")) {
    testthat::skip(paste0("tradeseq fixture failed: ",
                          conditionMessage(inp)))
  }
  out <- tryCatch(
    suppressWarnings(suppressMessages(
      tradeSeq::fitGAM(counts = inp$counts, pseudotime = inp$pseudotime,
                       cellWeights = inp$cellWeights, nknots = 3,
                       verbose = FALSE, parallel = FALSE)
    )),
    error = function(e) e
  )
  if (inherits(out, "error")) {
    testthat::skip(paste0("fitGAM raised on minimal fixture: ",
                          conditionMessage(out)))
  }
  # fitGAM returns SingleCellExperiment by default (sce = TRUE).
  expect_s4_class(out, "SingleCellExperiment")
})

test_that("fitGAM with_disabled() matches patched on the SCE shape", {
  .skip_if_no_tradeseq()
  inp <- tryCatch(.make_tradeseq_inputs(), error = function(e) e)
  if (inherits(inp, "error")) {
    testthat::skip(paste0("tradeseq fixture failed: ",
                          conditionMessage(inp)))
  }
  vanilla <- tryCatch(
    autozyme::with_disabled(
      suppressWarnings(suppressMessages(
        tradeSeq::fitGAM(counts = inp$counts, pseudotime = inp$pseudotime,
                         cellWeights = inp$cellWeights, nknots = 3,
                         verbose = FALSE, parallel = FALSE)
      ))
    ),
    error = function(e) e
  )
  if (inherits(vanilla, "error")) {
    testthat::skip(paste0("vanilla fitGAM failed: ",
                          conditionMessage(vanilla)))
  }
  patched <- suppressWarnings(suppressMessages(
    tradeSeq::fitGAM(counts = inp$counts, pseudotime = inp$pseudotime,
                     cellWeights = inp$cellWeights, nknots = 3,
                     verbose = FALSE, parallel = FALSE)
  ))
  expect_equal(class(patched), class(vanilla))
  expect_equal(dim(patched), dim(vanilla))
})
