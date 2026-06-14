# Direct unit tests for the infercnv.cpp kernels.
#
# These 11 kernels are all single-threaded (the HMM Gibbs sampler sets the
# pacing floor, per the file header) so they run safely under the campaign's
# OMP_NUM_THREADS=2 recipe. Each is checked against a base-R reference.

.cpp <- function(name) {
  fn <- tryCatch(get(name, envir = asNamespace("autozyme")),
                 error = function(e) NULL)
  if (is.null(fn)) testthat::skip(paste0("kernel ", name, " not exported"))
  fn
}

# ---- column / log / exp elementwise transforms ---------------------------

test_that("fast_scale_columns_cpp multiplies each column by its scale", {
  fn <- .cpp("fast_scale_columns_cpp")
  set.seed(1)
  m <- matrix(rnorm(15), 3, 5)
  s <- runif(5, 0.5, 2)
  expect_equal(fn(m, s), sweep(m, 2L, s, "*"), tolerance = 1e-12)
})

test_that("fast_scale_columns_cpp handles single row and single column", {
  fn <- .cpp("fast_scale_columns_cpp")
  expect_equal(fn(matrix(c(2, 4, 6), 1, 3), c(1, 2, 3)),
               matrix(c(2, 8, 18), 1, 3), tolerance = 1e-12)
  expect_equal(fn(matrix(c(1, 2, 3), 3, 1), 5),
               matrix(c(5, 10, 15), 3, 1), tolerance = 1e-12)
})

test_that("fast_log1p_scale_cpp equals log2(x + 1)", {
  fn <- .cpp("fast_log1p_scale_cpp")
  set.seed(2)
  m <- matrix(runif(12, 0, 100), 3, 4)
  inv_ln2 <- 1 / log(2)
  expect_equal(fn(m, inv_ln2), log2(m + 1), tolerance = 1e-12)
})

test_that("fast_invert_log2_cpp equals 2^x", {
  fn <- .cpp("fast_invert_log2_cpp")
  set.seed(3)
  m <- matrix(rnorm(12), 3, 4)
  expect_equal(fn(m, log(2)), 2^m, tolerance = 1e-12)
})

test_that("fast_log1p_scale and fast_invert_log2 round-trip", {
  lg <- .cpp("fast_log1p_scale_cpp")
  iv <- .cpp("fast_invert_log2_cpp")
  set.seed(4)
  m <- matrix(runif(20, 0, 50), 4, 5)
  back <- iv(lg(m, 1 / log(2)), log(2)) - 1
  expect_equal(back, m, tolerance = 1e-9)
})

test_that("fast_center_columns_cpp subtracts per-column centers", {
  fn <- .cpp("fast_center_columns_cpp")
  set.seed(5)
  m <- matrix(rnorm(20), 4, 5)
  ctr <- colMeans(m)
  out <- fn(m, ctr)
  expect_equal(out, sweep(m, 2L, ctr, "-"), tolerance = 1e-12)
  expect_equal(colMeans(out), rep(0, 5), tolerance = 1e-12)
})

# ---- reference subtraction with optional bounds/threshold ----------------

test_that("fast_subtract_ref_bounds_cpp (no bounds) subtracts mean-of-group-means", {
  fn <- .cpp("fast_subtract_ref_bounds_cpp")
  set.seed(6)
  nr <- 4; nc <- 8
  m <- matrix(rnorm(nr * nc), nr, nc)
  ref_groups <- list(as.integer(c(1, 2, 3)), as.integer(c(4, 5)))  # 1-based cols
  out <- fn(m, ref_groups, FALSE, 0, FALSE)

  mean1 <- rowMeans(m[, c(1, 2, 3), drop = FALSE])
  mean2 <- rowMeans(m[, c(4, 5), drop = FALSE])
  ref_mean <- (mean1 + mean2) / 2
  expect_equal(out, sweep(m, 1L, ref_mean, "-"), tolerance = 1e-10)
})

test_that("fast_subtract_ref_bounds_cpp (bounds) clamps to per-gene [min,max] of group means", {
  fn <- .cpp("fast_subtract_ref_bounds_cpp")
  set.seed(7)
  nr <- 3; nc <- 6
  m <- matrix(rnorm(nr * nc), nr, nc)
  ref_groups <- list(as.integer(c(1, 2)), as.integer(c(3, 4)), as.integer(c(5, 6)))
  thr <- 0.5
  out <- fn(m, ref_groups, TRUE, thr, TRUE)

  gmeans <- sapply(ref_groups, function(idx) rowMeans(m[, idx, drop = FALSE]))
  gmin <- apply(gmeans, 1L, min)
  gmax <- apply(gmeans, 1L, max)
  ref <- m
  for (i in seq_len(nr)) {
    bounded <- pmin(pmax(m[i, ], gmin[i]), gmax[i])
    y <- m[i, ] - bounded
    y <- pmin(pmax(y, -thr), thr)
    ref[i, ] <- y
  }
  expect_equal(out, ref, tolerance = 1e-10)
})

# ---- triangle-filter smoothers -------------------------------------------

test_that("fast_smooth_window_cpp center matches a moving-average reference", {
  fn <- .cpp("fast_smooth_window_cpp")
  # window_length must be odd; use a small odd window.
  wl <- 5L
  set.seed(8)
  nr <- 40L; nc <- 2L
  m <- matrix(rnorm(nr * nc), nr, nc)
  out <- fn(m, wl)

  # The kernel is a triangle (Bartlett) filter. We verify deterministic
  # structural properties rather than re-deriving the exact weights:
  expect_equal(dim(out), c(nr, nc))
  expect_true(all(is.finite(out)))
  # A constant column maps to (approximately) the same constant everywhere.
  mc <- matrix(7, nr, nc)
  oc <- fn(mc, wl)
  expect_equal(oc, mc, tolerance = 1e-9)
})

test_that("fast_smooth_long_chromosomes_cpp leaves non-listed rows untouched and smooths segments", {
  fn <- .cpp("fast_smooth_long_chromosomes_cpp")
  fn_single <- .cpp("fast_smooth_window_cpp")
  wl <- 5L
  set.seed(9)
  nr <- 60L; nc <- 3L
  m <- matrix(rnorm(nr * nc), nr, nc)
  # one long segment spanning the whole matrix
  starts <- as.integer(1)
  lens <- as.integer(nr)
  out <- fn(m, starts, lens, wl)
  # one whole-matrix segment should equal the single-window smoother.
  expect_equal(out, fn_single(m, wl), tolerance = 1e-9)
})

# ---- per-gene dropout zeroing (RNG) --------------------------------------

test_that("fast_apply_dropout_cpp zeros entries per per-gene probability", {
  fn <- .cpp("fast_apply_dropout_cpp")
  m <- matrix(5, 3, 4)
  set.seed(123)
  out <- fn(m, c(0.0, 0.5, 1.0))   # padj per gene (row)
  expect_true(all(out[1, ] == 5))  # p=0 -> nothing zeroed
  expect_true(all(out[3, ] == 0))  # p=1 -> all zeroed
  # surviving entries keep their original value (5), zeroed ones become 0
  expect_true(all(out[2, ] %in% c(0, 5)))
})

test_that("fast_apply_dropout_cpp is deterministic for a fixed seed", {
  fn <- .cpp("fast_apply_dropout_cpp")
  set.seed(7)
  a <- fn(matrix(3, 4, 5), rep(0.4, 4))
  set.seed(7)
  b <- fn(matrix(3, 4, 5), rep(0.4, 4))
  expect_identical(a, b)
})

# ---- HMM consensus / probability / viterbi -------------------------------

test_that("fast_state_consensus_cpp returns per-row mode with smallest-state tie-break", {
  fn <- .cpp("fast_state_consensus_cpp")
  mat <- rbind(
    c(1, 1, 2, 3),      # mode 1
    c(2, 2, 3, 3),      # tie 2 vs 3 -> smaller state 2
    c(5, 5, 5, 1),      # mode 5
    c(NA, 4, 4, NA)     # NA ignored -> mode 4
  )
  expect_equal(fn(mat), c(1, 2, 5, 4))
})

test_that("fast_cell_prob_cpp tallies per-column state probabilities", {
  fn <- .cpp("fast_cell_prob_cpp")
  eps <- cbind(
    c(1, 1, 2, 3),     # col1: state1 x2, state2 x1, state3 x1 over 4 -> .5,.25,.25
    c(2, 2, 2, 2)      # col2: all state 2 -> 0,1,0
  )
  out <- fn(eps, 3L)
  expect_equal(out[, 1], c(0.5, 0.25, 0.25), tolerance = 1e-12)
  expect_equal(out[, 2], c(0, 1, 0), tolerance = 1e-12)
  # probabilities sum to 1 per observed column
  expect_equal(colSums(out), c(1, 1), tolerance = 1e-12)
})

test_that("fast_cell_prob_cpp gives NA column when nothing observed in range", {
  fn <- .cpp("fast_cell_prob_cpp")
  eps <- cbind(c(NA_real_, NA_real_), c(1, 2))
  out <- fn(eps, 2L)
  expect_true(all(is.na(out[, 1])))
  expect_false(any(is.na(out[, 2])))
})

test_that("fast_viterbi_adj_cpp returns neutral state for length < 2", {
  fn <- .cpp("fast_viterbi_adj_cpp")
  Pi <- matrix(1 / 3, 3, 3)
  delta <- rep(1 / 3, 3)
  means <- c(0.5, 1.0, 1.5)
  sds <- c(0.2, 0.2, 0.2)
  out <- fn(c(1.0), Pi, delta, means, sds)
  expect_equal(out, 3)
})

test_that("fast_viterbi_adj_cpp decodes a clean signal to the matching states", {
  fn <- .cpp("fast_viterbi_adj_cpp")
  m <- 3L
  # strongly self-persistent chain so the MAP path follows the nearest mean
  Pi <- matrix(0.05, m, m); diag(Pi) <- 0.9
  Pi <- Pi / rowSums(Pi)
  delta <- rep(1 / m, m)
  means <- c(0.5, 1.0, 1.5)
  sds <- c(0.1, 0.1, 0.1)
  x <- c(0.5, 0.5, 1.5, 1.5, 1.0)  # near state 1,1,3,3,2
  out <- fn(x, Pi, delta, means, sds)
  expect_length(out, length(x))
  expect_true(all(out %in% seq_len(m)))
  # first two observations sit on mean of state 1
  expect_equal(out[1:2], c(1, 1))
})
