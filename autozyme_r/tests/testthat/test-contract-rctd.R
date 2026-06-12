# Contract: spacexr::run.RCTD + internal solver helpers (RCTD deconvolution)
#
# Patched surface: 9 internal targets (calc_log_l_vec, get_der_fast,
# solveWLS, solveIRWLS.weights, psd, process_bead_doublet,
# decompose_sparse, gather_results, process_beads_batch).
#
# All reached through `spacexr::run.RCTD(myRCTD, doublet_mode = ...)`.
# Building a real RCTD S4 object needs a reference (annotated scRNA-seq)
# + spatial counts + nUMI -- non-trivial fixture. Universal activation
# smoke proves the patch loads cleanly on every CI runner. Per-API
# deep contract here defers until a small fixture lands.

.skip_if_no_rctd <- function() {
  testthat::skip_if_not_installed("spacexr")
  # RCTD's S4 class needs the package on the search path, not just in
  # the namespace cache -- same trap as MAST/WGCNA we documented in
  # those tests. Attach via library() once.
  suppressPackageStartupMessages(library(spacexr))
}

test_that("rctd psd fast path matches upstream on small Hessians", {
  .skip_if_no_rctd()
  psd <- utils::getFromNamespace("psd", "spacexr")

  for (H in list(
    matrix(0.5, nrow = 1L, ncol = 1L),
    matrix(c(1, 0.2, 0.2, 2), nrow = 2L),
    matrix(c(-1, 0.4, 0.4, 0.2), nrow = 2L)
  )) {
    patched <- psd(H)
    vanilla <- autozyme::with_disabled(psd(H))
    expect_equal(patched, vanilla, tolerance = 1e-10)
    expect_equal(dim(patched), dim(H))
  }
})
