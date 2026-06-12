# Contract: infercnv (CNV inference from scRNA-seq)
#
# Patched surface: 7+ internal helpers (smooth_window, smooth_by_chromosome,
# normalize_counts_by_seq_depth, etc.) all called by infercnv::run().
#
# infercnv::run needs:
#   - raw counts matrix
#   - cell annotations
#   - gene order file (chr, start, end)
#   - reference cell types
# Plus internally invokes HMM Gibbs (rjags) which is slow + sequential.
# Heavy fixture; defer.

.skip_if_no_infercnv <- function() {
  testthat::skip_if_not_installed("infercnv")
}

test_that("infercnv .smooth_window fast path matches upstream on dense matrix", {
  .skip_if_no_infercnv()
  smooth_window <- utils::getFromNamespace(".smooth_window", "infercnv")
  set.seed(7)
  x <- matrix(stats::rnorm(18 * 4), nrow = 18, ncol = 4)
  rownames(x) <- paste0("g", seq_len(nrow(x)))
  colnames(x) <- paste0("c", seq_len(ncol(x)))

  patched <- smooth_window(x, window_length = 5L)
  vanilla <- autozyme::with_disabled(smooth_window(x, window_length = 5L))
  expect_equal(patched, vanilla, tolerance = 1e-10)
  expect_equal(dim(patched), dim(x))
  expect_equal(dimnames(patched), dimnames(x))
})
