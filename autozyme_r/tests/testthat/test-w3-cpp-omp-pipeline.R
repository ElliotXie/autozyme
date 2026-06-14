# Wave-3 direct unit tests for the for_paper OMP FindAllMarkers pipeline
# (for_paper_markers_omp.cpp) plus the az_blas_gemm / nb_dDeta_log_cpp
# leftovers wave-1 left on the allowlist.
#
# The OMP kernels chained here replace presto's nnzeroGroups + sumGroups +
# rank_matrix + compute_ustat + compute_pval in the serial all-union path:
#
#   omp_sum_nnz_expm1_groups       per-(feature,group) nnz + sum expm1
#   omp_filter_pct_lfc             pct1/pct2/log-fold-change/pass mask
#   omp_subset_transpose_rank_pval subset features -> Wilcoxon p (CSR transpose)
#   omp_rank_ustat_pval            full-matrix Wilcoxon p (cells x features)
#
# Every kernel carries `#pragma omp ... num_threads(n_threads)`, so passing
# n_threads = 1L keeps them single-threaded and segfault-safe even under
# OMP_NUM_THREADS>1. We always pass n_threads = 1L.
#
# References are plain base-R: rowSums for nnz/sum, the kernel's own rounding
# convention for pct, and wilcox.test(correct=TRUE) for the U-stat p-values
# (the kernels reproduce the normal-approximation two-sided Wilcoxon with tie
# + continuity correction, which is exactly what wilcox.test computes here).

.cpp_omp <- function(name) {
  fn <- tryCatch(get(name, envir = asNamespace("autozyme")),
                 error = function(e) NULL)
  if (is.null(fn)) testthat::skip(paste0("kernel ", name, " not exported"))
  fn
}

# std::round in the kernel rounds half away from zero; base R round() uses
# round-half-to-even, so reproduce the C convention for the pct comparison.
.cround3 <- function(x) trunc(x * 1000 + sign(x) * 0.5) / 1000

# A small features x cells count matrix with partly-broken ties (so the tie
# correction path is exercised but the comparison stays well-defined).
.omp_fixture <- function(nfeat = 6L, ncells = 48L, ngroups = 3L, seed = 33) {
  testthat::skip_if_not_installed("Matrix")
  set.seed(seed)
  M <- matrix(stats::rpois(nfeat * ncells, 0.7) * 1.0, nfeat, ncells)
  M[M > 0] <- M[M > 0] + runif(sum(M > 0))
  spm <- methods::as(methods::as(M, "CsparseMatrix"), "generalMatrix")
  y <- as.integer(rep(seq_len(ngroups), length.out = ncells))
  cs <- as.numeric(table(factor(y, levels = seq_len(ngroups))))
  list(M = M, spm = spm, y = y, cs = cs,
       nfeat = nfeat, ncells = ncells, ngroups = ngroups)
}

# ======================================================================
# omp_sum_nnz_expm1_groups : per-(feature, group) nonzero count + sum expm1
# ======================================================================

test_that("omp_sum_nnz_expm1_groups matches rowSums of nnz and expm1 per group", {
  fn <- .cpp_omp("omp_sum_nnz_expm1_groups")
  fx <- .omp_fixture()
  agg <- fn(fx$spm, fx$y, fx$ngroups, 1L)
  nnz_ref <- sapply(seq_len(fx$ngroups),
                    function(g) rowSums(fx$M[, fx$y == g, drop = FALSE] > 0))
  sum_ref <- sapply(seq_len(fx$ngroups),
                    function(g) rowSums(expm1(fx$M[, fx$y == g, drop = FALSE])))
  expect_equal(dim(agg$nnz), c(fx$nfeat, fx$ngroups))
  expect_equal(agg$nnz, nnz_ref, tolerance = 1e-10)
  expect_equal(agg$sum_expm1, sum_ref, tolerance = 1e-10)
})

# ======================================================================
# omp_filter_pct_lfc : pct1/pct2/lfc/pass + row_any, chained off the aggregator
# ======================================================================

test_that("omp_filter_pct_lfc reproduces the pct / lfc / pass-mask R reference", {
  agg_fn  <- .cpp_omp("omp_sum_nnz_expm1_groups")
  filt_fn <- .cpp_omp("omp_filter_pct_lfc")
  fx <- .omp_fixture()
  agg <- agg_fn(fx$spm, fx$y, fx$ngroups, 1L)

  total_nnz   <- rowSums(fx$M > 0)
  total_expm1 <- rowSums(expm1(fx$M))
  sizes_rest  <- sum(fx$cs) - fx$cs
  valid_mask  <- rep(TRUE, fx$ngroups)
  min_pct <- 0.1; min_diff_pct <- 0.0; logfc_threshold <- 0.0
  only_pos <- FALSE; log_base <- exp(1)

  filt <- filt_fn(agg$nnz, agg$sum_expm1, total_nnz, total_expm1,
                  fx$cs, sizes_rest, valid_mask, min_pct, min_diff_pct,
                  logfc_threshold, only_pos, log_base, 1L)

  pct1_ref <- .cround3(sweep(agg$nnz, 2, fx$cs, "/"))
  pct2_ref <- .cround3(sweep(total_nnz - agg$nnz, 2, sizes_rest, "/"))
  m1 <- log(sweep(agg$sum_expm1 + 1, 2, fx$cs, "/")) / log(log_base)
  m2 <- log(sweep(total_expm1 - agg$sum_expm1 + 1, 2, sizes_rest, "/")) / log(log_base)
  lfc_ref <- m1 - m2
  a_max  <- pmax(pct1_ref, pct2_ref)
  a_diff <- a_max - pmin(pct1_ref, pct2_ref)
  pass_ref <- (a_max >= min_pct) & (a_diff >= min_diff_pct) &
              (abs(lfc_ref) >= logfc_threshold)

  expect_equal(filt$pct1, pct1_ref, tolerance = 1e-10)
  expect_equal(filt$pct2, pct2_ref, tolerance = 1e-10)
  expect_equal(filt$lfc, lfc_ref, tolerance = 1e-10)
  expect_equal(filt$pass, matrix(pass_ref, fx$nfeat, fx$ngroups))
  # row_any: per-feature OR over the group pass flags.
  expect_equal(as.logical(filt$row_any),
               apply(matrix(pass_ref, fx$nfeat, fx$ngroups), 1, any))
})

test_that("omp_filter_pct_lfc only_pos restricts pass to positive log-fold-change", {
  agg_fn  <- .cpp_omp("omp_sum_nnz_expm1_groups")
  filt_fn <- .cpp_omp("omp_filter_pct_lfc")
  fx <- .omp_fixture()
  agg <- agg_fn(fx$spm, fx$y, fx$ngroups, 1L)
  total_nnz <- rowSums(fx$M > 0); total_expm1 <- rowSums(expm1(fx$M))
  sizes_rest <- sum(fx$cs) - fx$cs
  filt <- filt_fn(agg$nnz, agg$sum_expm1, total_nnz, total_expm1,
                  fx$cs, sizes_rest, rep(TRUE, fx$ngroups),
                  0.1, 0.0, 0.25, only_pos = TRUE, exp(1), 1L)
  # Every passing (feature, group) must have lfc >= the threshold.
  passed <- which(filt$pass)
  expect_true(all(filt$lfc[passed] >= 0.25 - 1e-12))
})

# ======================================================================
# omp_rank_ustat_pval : full-matrix Wilcoxon p (input = cells x features)
# ======================================================================

test_that("omp_rank_ustat_pval matches wilcox.test(correct=TRUE) per (feature, group)", {
  fn <- .cpp_omp("omp_rank_ustat_pval")
  fx <- .omp_fixture()
  spm_t <- methods::as(Matrix::t(fx$spm), "CsparseMatrix")   # cells x features
  pv <- fn(spm_t, fx$y, fx$ngroups, fx$cs, 1L)
  expect_equal(dim(pv), c(fx$nfeat, fx$ngroups))

  ref <- matrix(NA_real_, fx$nfeat, fx$ngroups)
  for (f in seq_len(fx$nfeat)) {
    xv <- fx$M[f, ]
    for (g in seq_len(fx$ngroups)) {
      wt <- suppressWarnings(
        stats::wilcox.test(xv[fx$y == g], xv[fx$y != g],
                           correct = TRUE, exact = FALSE))
      ref[f, g] <- wt$p.value
    }
  }
  expect_equal(pv, ref, tolerance = 1e-9)
})

test_that("omp_rank_ustat_pval returns 1 for a degenerate single-group split", {
  fn <- .cpp_omp("omp_rank_ustat_pval")
  testthat::skip_if_not_installed("Matrix")
  set.seed(44)
  nfeat <- 3L; ncells <- 10L
  M <- matrix(stats::rpois(nfeat * ncells, 1) * 1.0, nfeat, ncells)
  spm_t <- methods::as(methods::as(t(M), "CsparseMatrix"), "generalMatrix")
  # one group only -> n2 == 0 for that group -> kernel short-circuits to p = 1.
  y <- rep(1L, ncells)
  cs <- c(ncells)
  pv <- fn(spm_t, y, 1L, cs, 1L)
  expect_true(all(pv == 1))
})

# ======================================================================
# omp_subset_transpose_rank_pval : subset features then Wilcoxon p
# ======================================================================

test_that("omp_subset_transpose_rank_pval matches wilcox.test on the feature subset", {
  fn <- .cpp_omp("omp_subset_transpose_rank_pval")
  fx <- .omp_fixture()
  union_idx <- c(1L, 3L, 5L)          # 1-based feature indices to keep
  pv <- fn(fx$spm, union_idx, fx$y, fx$ngroups, fx$cs, 1L)
  expect_equal(dim(pv), c(length(union_idx), fx$ngroups))

  ref <- matrix(NA_real_, length(union_idx), fx$ngroups)
  for (jj in seq_along(union_idx)) {
    xv <- fx$M[union_idx[jj], ]
    for (g in seq_len(fx$ngroups)) {
      wt <- suppressWarnings(
        stats::wilcox.test(xv[fx$y == g], xv[fx$y != g],
                           correct = TRUE, exact = FALSE))
      ref[jj, g] <- wt$p.value
    }
  }
  expect_equal(pv, ref, tolerance = 1e-9)
})

test_that("omp_subset_transpose_rank_pval agrees with omp_rank_ustat_pval on the same rows", {
  sub_fn  <- .cpp_omp("omp_subset_transpose_rank_pval")
  full_fn <- .cpp_omp("omp_rank_ustat_pval")
  fx <- .omp_fixture()
  union_idx <- c(2L, 4L, 6L)
  pv_sub <- sub_fn(fx$spm, union_idx, fx$y, fx$ngroups, fx$cs, 1L)
  spm_t <- methods::as(Matrix::t(fx$spm), "CsparseMatrix")
  pv_full <- full_fn(spm_t, fx$y, fx$ngroups, fx$cs, 1L)
  # The two kernels compute the same Wilcoxon p-values; one just pre-subsets.
  expect_equal(pv_sub, pv_full[union_idx, , drop = FALSE], tolerance = 1e-10)
})

# ======================================================================
# az_blas_gemm : the general DGEMM dispatch (allowlist leftover)
# ======================================================================

test_that("az_blas_gemm equals R's %*% (and respects transA/transB)", {
  info_fn <- .cpp_omp("az_blas_info")
  if (!isTRUE(info_fn()$available)) {
    testthat::skip("no dynamic BLAS backend loadable on this host")
  }
  fn <- .cpp_omp("az_blas_gemm")
  set.seed(55)
  A <- matrix(stats::rnorm(6 * 4), 6, 4)
  B <- matrix(stats::rnorm(4 * 5), 4, 5)
  expect_equal(fn(A, B, FALSE, FALSE, 1L), A %*% B, tolerance = 1e-10)
  # transA: A is supplied transposed, so the math is t(A) %*% C.
  At <- t(A)                      # 4 x 6
  C  <- matrix(stats::rnorm(6 * 3), 6, 3)
  expect_equal(fn(At, C, TRUE, FALSE, 1L), t(At) %*% C, tolerance = 1e-10)
  # transB: B supplied transposed.
  D  <- matrix(stats::rnorm(5 * 4), 5, 4)
  expect_equal(fn(A, D, FALSE, TRUE, 1L), A %*% t(D), tolerance = 1e-10)
})

# ======================================================================
# nb_dDeta_log_cpp : eta-space chain rule over the NB mu-space derivatives
# ======================================================================

test_that("nb_dDeta_log_cpp eta-space derivatives match the NB closed forms", {
  fn <- .cpp_omp("nb_dDeta_log_cpp")
  y <- c(0, 1, 3, 7); wt <- rep(1, 4)
  theta_log <- log(4.0); theta <- exp(theta_log)
  mu <- c(0.6, 1.2, 2.8, 6.5)
  out <- fn(y, mu, wt, theta_log, deriv = 1L)

  # mu-space derivatives (the same as nb_Dd_cpp's), then chain-ruled to eta
  # via the log link (dmu/deta = mu).
  yth <- y + theta; muth <- mu + theta
  Dmu   <- 2 * wt * (yth / muth - y / mu)
  Dmu2  <- -2 * wt * (yth / muth^2 - y / mu^2)
  EDmu2 <- 2 * wt * (1 / mu - 1 / muth)
  Dmu3  <- 4 * wt * (yth / muth^3 - y / mu^3)
  r_ym  <- yth / muth

  expect_equal(out$Deta,   Dmu * mu, tolerance = 1e-12)             # dD/deta
  expect_equal(out$Deta2,  Dmu2 * mu^2 + Dmu * mu, tolerance = 1e-12)
  expect_equal(out$EDeta2, EDmu2 * mu^2, tolerance = 1e-12)
  expect_equal(out$Deta3,  Dmu3 * mu^3 + 3 * Dmu2 * mu^2 + Dmu * mu,
               tolerance = 1e-12)
  # theta-derivative present at deriv > 0.
  expect_equal(out$Dth, -2 * wt * theta * (log(r_ym) + (1 - r_ym)),
               tolerance = 1e-12)
  expect_true(all(out$good))

  # Cross-check: Deta really is the eta-gradient of the NB deviance
  # (central finite difference of nb_dev_resids_cpp under the log link).
  dev_fn <- .cpp_omp("nb_dev_resids_cpp")
  h <- 1e-6; eta0 <- log(mu)
  fd <- vapply(seq_along(y), function(i)
    (dev_fn(y[i], exp(eta0[i] + h), wt[i], theta_log) -
     dev_fn(y[i], exp(eta0[i] - h), wt[i], theta_log)) / (2 * h), numeric(1))
  expect_equal(out$Deta, fd, tolerance = 1e-6)

  # deriv = 0 omits the higher-order theta terms (returns scalar 0 placeholders).
  out0 <- fn(y, mu, wt, theta_log, deriv = 0L)
  expect_equal(out0$Deta, Dmu * mu, tolerance = 1e-12)
  expect_equal(out0$Dth, 0)
})
