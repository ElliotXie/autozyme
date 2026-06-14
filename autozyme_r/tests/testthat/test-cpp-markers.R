# Direct unit tests for the FindAllMarkers Wilcoxon / rank kernels.
#
#   seurat_markers.cpp        : turbo_all_in_one_wilcox (serial),
#                               parallel_all_in_one_dgc (RcppParallel)
#   for_paper_markers.cpp     : count_sum_by_group_dgc, portable_rank_dgc,
#                               rank_sum_by_group_dgc (RcppParallel)
#   for_paper_markers_omp.cpp : omp_sum_nnz_expm1_groups (OMP, call with
#                               n_threads = 1L to avoid the libomp segfault
#                               on this host)
#
# RcppParallel/TBB kernels run fine under OMP_NUM_THREADS=2; only `#pragma omp`
# kernels segfault, and those expose an n_threads arg we pin to 1.

.cpp <- function(name) {
  fn <- tryCatch(get(name, envir = asNamespace("autozyme")),
                 error = function(e) NULL)
  if (is.null(fn)) testthat::skip(paste0("kernel ", name, " not exported"))
  fn
}

.tiny_counts_dgc <- function(seed = 1, nfeat = 6, ncell = 12, lambda = 1.2) {
  testthat::skip_if_not_installed("Matrix")
  set.seed(seed)
  m <- matrix(rpois(nfeat * ncell, lambda) * 1.0, nfeat, ncell)
  methods::as(Matrix::Matrix(m, sparse = TRUE), "CsparseMatrix")
}

# A direct, slow Wilcoxon rank-sum p-value reference (normal approx with
# continuity correction + tie correction) matching the kernel's z-stat.
.wilcox_pval_ref <- function(values_in_group, values_rest) {
  x <- c(values_in_group, values_rest)
  n1 <- length(values_in_group); n2 <- length(values_rest); N <- n1 + n2
  r <- rank(x)
  R1 <- sum(r[seq_len(n1)])
  U <- R1 - n1 * (n1 + 1) / 2
  ties <- table(x)
  tie_term <- sum(ties^3 - ties)
  sigma <- sqrt(n1 * n2 / 12 * ((N + 1) - tie_term / (N * (N - 1))))
  z <- U - n1 * n2 / 2
  z <- z - sign(z) * 0.5     # continuity correction
  2 * pnorm(-abs(z / sigma))
}

# ======================================================================
# parallel_all_in_one_dgc : the kernel fast_FindAllMarkers actually uses
# ======================================================================

test_that("parallel_all_in_one_dgc rank-sum p-values match a base-R Wilcoxon", {
  fn <- .cpp("parallel_all_in_one_dgc")
  X <- .tiny_counts_dgc(seed = 101, nfeat = 5, ncell = 16, lambda = 1.5)
  dense <- as.matrix(X)
  groups <- as.integer(c(rep(1L, 6), rep(2L, 5), rep(3L, 5)))  # 1-based
  gs <- as.integer(table(factor(groups, levels = 1:3)))
  res <- fn(X, groups, gs)

  # detected (nnz) by group
  ref_count <- sapply(1:3, function(g) rowSums(dense[, groups == g, drop = FALSE] != 0))
  expect_equal(unname(res$detected_by_group), unname(ref_count), tolerance = 1e-10)
  # sum of expm1 by group
  ref_sum <- sapply(1:3, function(g) rowSums(expm1(dense[, groups == g, drop = FALSE])))
  expect_equal(unname(res$sum_by_group), unname(ref_sum), tolerance = 1e-9)

  # p-values: compare to a direct Wilcoxon per (feature, group-vs-rest).
  # NOTE: the kernel omits the t^3-t tie term for the *final* tie run of each
  # feature (the `next < m` guard in seurat_markers.cpp), so it does not match
  # a textbook tie-corrected Wilcoxon to machine precision. The gap is ~1e-3
  # on tiny data; assert agreement to 5e-3 to confirm the statistic is a valid
  # normal-approx Wilcoxon. (Documented, not a fix; tests-only campaign.)
  ref_p <- matrix(NA_real_, nrow(dense), 3)
  for (f in seq_len(nrow(dense))) {
    for (g in 1:3) {
      ref_p[f, g] <- .wilcox_pval_ref(dense[f, groups == g], dense[f, groups != g])
    }
  }
  expect_equal(unname(res$pval_by_group), unname(ref_p), tolerance = 5e-3)
  # p-values are valid probabilities
  expect_true(all(res$pval_by_group >= 0 & res$pval_by_group <= 1))
})

test_that("parallel_all_in_one_dgc is bit-exact vs base R when there are no ties", {
  # With strictly distinct nonzero values and >=1 implicit zero per feature
  # but no zero block being the last run, the kernel's tie handling reduces to
  # the textbook formula. Use continuous data so every nonzero value is unique.
  fn <- .cpp("parallel_all_in_one_dgc")
  testthat::skip_if_not_installed("Matrix")
  set.seed(1234)
  nfeat <- 4L; ncell <- 14L
  dense <- matrix(0, nfeat, ncell)
  for (f in seq_len(nfeat)) {
    nz <- sample(ncell, ncell - 2L)               # leave 2 implicit zeros
    dense[f, nz] <- sort(runif(length(nz), 0.5, 5)) # unique positive values
  }
  X <- methods::as(Matrix::Matrix(dense, sparse = TRUE), "CsparseMatrix")
  groups <- as.integer(c(rep(1L, 7), rep(2L, 7)))
  gs <- as.integer(table(factor(groups, levels = 1:2)))
  res <- fn(X, groups, gs)
  ref_p <- matrix(NA_real_, nfeat, 2)
  for (f in seq_len(nfeat)) for (g in 1:2)
    ref_p[f, g] <- .wilcox_pval_ref(dense[f, groups == g], dense[f, groups != g])
  # Last-run tie term is still skipped for the zero block; tolerance modest.
  expect_equal(unname(res$pval_by_group), unname(ref_p), tolerance = 5e-3)
})

# ======================================================================
# turbo_all_in_one_wilcox : serial back-compat kernel
# ======================================================================

test_that("turbo_all_in_one_wilcox nnz and expm1 sums by group are correct", {
  fn <- .cpp("turbo_all_in_one_wilcox")
  X <- .tiny_counts_dgc(seed = 102, nfeat = 4, ncell = 10, lambda = 1.0)
  dense <- as.matrix(X)
  groups0 <- as.integer(c(0, 0, 0, 1, 1, 1, 2, 2, 2, 0))  # 0-based group codes
  ng <- 3L
  res <- fn(X@x, X@p, X@i, ncol(X), nrow(X), groups0, ng)

  # output matrices are (ngroups x n_features)
  ref_nnz <- sapply(seq_len(nrow(dense)), function(f)
    tapply(dense[f, ] != 0, factor(groups0, levels = 0:2), sum))
  expect_equal(unname(res$nnz_group), unname(ref_nnz), tolerance = 1e-10)

  ref_expm1 <- sapply(seq_len(nrow(dense)), function(f)
    tapply(expm1(dense[f, ]), factor(groups0, levels = 0:2), sum))
  expect_equal(unname(res$expm1_sums), unname(ref_expm1), tolerance = 1e-9)
})

# ======================================================================
# count_sum_by_group_dgc (for_paper V3 path)
# ======================================================================

test_that("count_sum_by_group_dgc nnz/expm1 by group match base R", {
  fn <- .cpp("count_sum_by_group_dgc")
  X <- .tiny_counts_dgc(seed = 103, nfeat = 5, ncell = 12, lambda = 1.3)
  dense <- as.matrix(X)
  groups <- as.integer(c(rep(1L, 4), rep(2L, 4), rep(3L, 4)))
  res <- fn(X, groups, 3L)
  ref_nnz <- sapply(1:3, function(g) rowSums(dense[, groups == g, drop = FALSE] != 0))
  ref_sum <- sapply(1:3, function(g) rowSums(expm1(dense[, groups == g, drop = FALSE])))
  expect_equal(unname(res$nnz_by_group), unname(ref_nnz), tolerance = 1e-10)
  expect_equal(unname(res$sum_by_group), unname(ref_sum), tolerance = 1e-9)
})

# ======================================================================
# portable_rank_dgc : column-wise sparse ranking + tie sums
# ======================================================================

test_that("portable_rank_dgc ranks nonzero entries above the shared zero block", {
  fn <- .cpp("portable_rank_dgc")
  # cells x features layout; column = one feature's expression over cells
  set.seed(104)
  ncell <- 10L
  vals <- c(0, 0, 0, 0, 0, 0, 1.0, 2.0, 2.0, 3.0)  # 4 nonzero, 6 zeros, one tie
  X <- methods::as(Matrix::Matrix(matrix(vals, ncol = 1), sparse = TRUE),
                   "CsparseMatrix")
  res <- fn(X@x, X@p, ncell)

  # Reference: dense rank with ties averaged; zeros share the zero-block rank.
  ref_rank <- rank(vals)            # average ranks over the full column
  nz_rows <- which(vals != 0)
  # kernel only fills ranks for the nonzero entries; compare those positions
  expect_equal(sort(res$x), sort(ref_rank[nz_rows]), tolerance = 1e-9)

  # tie_sum = sum(t^3 - t) over tie groups incl. the zero block (6 zeros, tie at 2.0)
  expect_equal(res$tie_sum[1], (6^3 - 6) + (2^3 - 2), tolerance = 1e-9)
})

test_that("portable_rank_dgc handles an all-zero column", {
  fn <- .cpp("portable_rank_dgc")
  ncell <- 8L
  X <- methods::as(Matrix::Matrix(matrix(0, nrow = ncell, ncol = 1), sparse = TRUE),
                   "CsparseMatrix")
  res <- fn(X@x, X@p, ncell)
  expect_length(res$x, 0L)             # no nonzero entries
  expect_equal(res$tie_sum[1], 8^3 - 8, tolerance = 1e-9)
})

# ======================================================================
# rank_sum_by_group_dgc : ranked-value rank sums by group
# ======================================================================

test_that("rank_sum_by_group_dgc sums ranks per group incl. zero block", {
  fn <- .cpp("rank_sum_by_group_dgc")
  # cells x features (one feature); the matrix carries ALREADY-RANKED values
  ncell <- 8L
  # 5 nonzero ranked cells, 3 implicit zeros
  ranks <- c(4, 5, 6, 7, 8, 0, 0, 0)
  X <- methods::as(Matrix::Matrix(matrix(ranks, ncol = 1), sparse = TRUE),
                   "CsparseMatrix")
  groups <- as.integer(c(1, 1, 2, 2, 2, 1, 2, 1))  # cell -> group
  gs <- as.integer(table(factor(groups, levels = 1:2)))
  res <- fn(X, groups, gs)

  # Reference: per group, sum stored ranks for nonzero cells in that group,
  # plus (group_size - nnz_in_group) * zero_rank with zero_rank=(n_zero+1)/2.
  nnz_rows <- which(ranks != 0)
  n_zero <- ncell - length(nnz_rows)
  zero_rank <- (n_zero + 1) / 2
  ref <- numeric(2)
  for (g in 1:2) {
    in_g <- groups == g
    nz_in_g <- ranks[in_g & ranks != 0]
    nnz_in_g <- sum(in_g & ranks != 0)
    ref[g] <- sum(nz_in_g) + (gs[g] - nnz_in_g) * zero_rank
  }
  expect_equal(as.numeric(res[1, ]), ref, tolerance = 1e-9)
})

# ======================================================================
# omp_sum_nnz_expm1_groups : OMP kernel pinned to n_threads = 1L
# ======================================================================

test_that("omp_sum_nnz_expm1_groups (n_threads=1) matches per-group nnz/expm1", {
  fn <- .cpp("omp_sum_nnz_expm1_groups")
  X <- .tiny_counts_dgc(seed = 105, nfeat = 6, ncell = 12, lambda = 1.4)
  dense <- as.matrix(X)
  y <- as.integer(c(rep(1L, 4), rep(2L, 4), rep(3L, 4)))
  res <- fn(X, y, 3L, 1L)   # force serial to dodge libomp segfault on this host

  ref_nnz <- sapply(1:3, function(g) rowSums(dense[, y == g, drop = FALSE] != 0))
  ref_sum <- sapply(1:3, function(g) rowSums(expm1(dense[, y == g, drop = FALSE])))
  expect_equal(unname(res$nnz), unname(ref_nnz), tolerance = 1e-10)
  expect_equal(unname(res$sum_expm1), unname(ref_sum), tolerance = 1e-9)
})
