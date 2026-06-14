# Wave-3 direct unit tests for the rctd spline / IRWLS solver kernels.
#
# These eleven kernels (rctd_cpp_calc_log_l_{sum,vec}, get_d1_d2,
# get_der_fast_nonbulk, solve_wls_p1/p2, irwls_sparse_p12,
# irwls_full_nonbulk, score_sparse_candidates, fit_sparse_pair) all consume
# Q_mat / SQ_mat / X_vals / K_val as explicit arguments. In production,
# spacexr::set_likelihood_vars writes those tables into .GlobalEnv and the
# R-side fast_* wrappers (inst/patches/rctd/patch.R) grab them by bare name.
#
# Building a *real* Poisson-convolution likelihood basis is expensive, but
# the kernels' contract is purely "given these spline tables, evaluate the
# natural cubic spline (and its derivatives)". So we fabricate a small,
# self-consistent set of tables and check the kernels against a pure-R
# re-implementation of the very same spline math that the patch's file-scope
# helper fast_calc_Q_all uses. That reference is bit-exact (the kernels are
# a straight C++ port), so these are exact-equivalence tests, not just
# invariants. The iterative IRWLS solvers (which would diverge on a
# non-likelihood random basis) are checked for shape / finiteness / the
# single-step closed form instead.
#
# None of these kernels use `#pragma omp`, so they're safe under any
# OMP_NUM_THREADS.

.cpp_rctd <- function(name) {
  fn <- tryCatch(get(name, envir = asNamespace("autozyme")),
                 error = function(e) NULL)
  if (is.null(fn)) testthat::skip(paste0("kernel ", name, " not exported"))
  fn
}

# A small, strictly-increasing spline basis. K_val caps the row index; the
# m_index map lands in columns ~21..131 for lambda in [1e-2, 5], so 200
# columns leaves headroom for the col+1 lookahead the kernel does.
.rctd_tables <- function(seed = 11) {
  set.seed(seed)
  K_val <- 12L
  nrowq <- K_val + 1L          # rows are 0-based row_index in [0, K_val]
  ncolq <- 200L
  X_vals <- as.numeric(cumsum(c(1e-5, rep(0.05, ncolq - 1L))))
  Q_mat  <- matrix(stats::rnorm(nrowq * ncolq, mean = -2, sd = 0.3), nrowq, ncolq)
  SQ_mat <- matrix(stats::rnorm(nrowq * ncolq, mean = 0.1, sd = 0.05), nrowq, ncolq)
  list(K_val = K_val, X_vals = X_vals, Q_mat = Q_mat, SQ_mat = SQ_mat)
}

# Pure-R port of spline_value_cpp / spline_derivs_cpp (mirrors the patch's
# fast_calc_Q_all). Returns d0 (the spline value), d1, d2.
.rctd_spline <- function(y, lambda_raw, tb) {
  epsilon <- 1e-4
  X_max <- max(tb$X_vals)
  lambda <- min(max(lambda_raw, epsilon), X_max - epsilon)
  row <- min(as.integer(y), tb$K_val)                 # 0-based row
  l <- floor(sqrt(lambda / 1e-6))
  col <- as.integer(min(l - 9, 40) +
                    max(ceiling(sqrt(max(l - 48.7499, 0) * 4)) - 2, 0)) - 1L  # 0-based
  ti1 <- tb$X_vals[col + 1L]; ti <- tb$X_vals[col + 2L]; hi <- ti - ti1
  fti1 <- tb$Q_mat[row + 1L, col + 1L]; fti <- tb$Q_mat[row + 1L, col + 2L]
  zi1  <- tb$SQ_mat[row + 1L, col + 1L]; zi  <- tb$SQ_mat[row + 1L, col + 2L]
  diff1 <- lambda - ti1; diff2 <- ti - lambda
  diff3 <- fti / hi - zi * hi / 6; diff4 <- fti1 / hi - zi1 * hi / 6
  zdi <- zi / hi; zdi1 <- zi1 / hi
  d0 <- zdi * diff1^3 / 6 + zdi1 * diff2^3 / 6 + diff3 * diff1 + diff4 * diff2
  d1 <- zdi * diff1^2 / 2 - zdi1 * diff2^2 / 2 + diff3 - diff4
  d2 <- zdi * diff1 + zdi1 * diff2
  c(d0 = d0, d1 = d1, d2 = d2)
}

# ======================================================================
# calc_log_l_vec / calc_log_l_sum : negative spline value, per-element & summed
# ======================================================================

test_that("rctd_cpp_calc_log_l_vec returns -spline_value elementwise", {
  fn <- .cpp_rctd("rctd_cpp_calc_log_l_vec")
  tb <- .rctd_tables()
  y   <- c(0, 1, 3, 7, 12, 15)
  lam <- c(0.01, 0.05, 0.1, 0.5, 1, 2)
  out <- fn(lam, y, NULL, tb$Q_mat, tb$SQ_mat, tb$X_vals, tb$K_val)
  ref <- -vapply(seq_along(y),
                 function(i) .rctd_spline(y[i], lam[i], tb)[["d0"]],
                 numeric(1))
  expect_equal(out, ref, tolerance = 1e-10)
})

test_that("rctd_cpp_calc_log_l_sum is the sum of the per-element vec", {
  vfn <- .cpp_rctd("rctd_cpp_calc_log_l_vec")
  sfn <- .cpp_rctd("rctd_cpp_calc_log_l_sum")
  tb <- .rctd_tables()
  y   <- c(0, 1, 3, 7, 12, 15)
  lam <- c(0.01, 0.05, 0.1, 0.5, 1, 2)
  vec <- vfn(lam, y, NULL, tb$Q_mat, tb$SQ_mat, tb$X_vals, tb$K_val)
  sum_out <- sfn(lam, y, NULL, tb$Q_mat, tb$SQ_mat, tb$X_vals, tb$K_val)
  expect_equal(sum_out, sum(vec), tolerance = 1e-10)
})

test_that("rctd row_idx (non-null) path matches the null path on the same row", {
  fn <- .cpp_rctd("rctd_cpp_calc_log_l_sum")
  tb <- .rctd_tables()
  y <- c(0, 1, 3, 7, 12, 15)
  lam <- c(0.01, 0.05, 0.1, 0.5, 1, 2)
  # When row_idx[i] == pmin(floor(y_i), K_val) + 1 the explicit-index path
  # must reproduce the implicit row_index_cpp computation exactly.
  row_idx <- as.numeric(pmin(as.integer(y), tb$K_val) + 1L)
  with_idx    <- fn(lam, y, row_idx, tb$Q_mat, tb$SQ_mat, tb$X_vals, tb$K_val)
  without_idx <- fn(lam, y, NULL,    tb$Q_mat, tb$SQ_mat, tb$X_vals, tb$K_val)
  expect_equal(with_idx, without_idx, tolerance = 1e-12)
})

# ======================================================================
# get_d1_d2 : the (d1, d2) spline derivatives per element
# ======================================================================

test_that("rctd_cpp_get_d1_d2 matches the R spline-derivative reference", {
  fn <- .cpp_rctd("rctd_cpp_get_d1_d2")
  tb <- .rctd_tables()
  y   <- c(0, 1, 3, 7, 12, 15)
  lam <- c(0.01, 0.05, 0.1, 0.5, 1, 2)
  out <- fn(y, lam, NULL, tb$Q_mat, tb$SQ_mat, tb$X_vals, tb$K_val)
  ref <- vapply(seq_along(y),
                function(i) .rctd_spline(y[i], lam[i], tb)[c("d1", "d2")],
                numeric(2))
  expect_equal(out$d1_vec, unname(ref["d1", ]), tolerance = 1e-10)
  expect_equal(out$d2_vec, unname(ref["d2", ]), tolerance = 1e-10)
})

# ======================================================================
# get_der_fast_nonbulk : grad(1,p) = -sum d1*S ; hess(a,b) = -sum d2*S_a*S_b
# ======================================================================

test_that("rctd_cpp_get_der_fast_nonbulk builds grad/hess from the derivatives", {
  fn <- .cpp_rctd("rctd_cpp_get_der_fast_nonbulk")
  tb <- .rctd_tables()
  set.seed(101)
  n <- 6L; p <- 2L
  S <- matrix(abs(stats::rnorm(n * p, 1, 0.3)), n, p)
  y <- c(0, 1, 3, 7, 12, 15)
  lambda <- abs(as.numeric(S %*% c(0.4, 0.6))) + 0.05
  out <- fn(S, y, lambda, NULL, tb$Q_mat, tb$SQ_mat, tb$X_vals, tb$K_val)

  d12 <- vapply(seq_along(y),
                function(i) .rctd_spline(y[i], lambda[i], tb)[c("d1", "d2")],
                numeric(2))
  grad_ref <- -colSums(d12["d1", ] * S)
  hess_ref <- matrix(0, p, p)
  for (i in seq_len(n)) hess_ref <- hess_ref - d12["d2", i] * (S[i, ] %*% t(S[i, ]))

  expect_equal(as.numeric(out$grad), grad_ref, tolerance = 1e-10)
  expect_equal(out$hess, hess_ref, tolerance = 1e-10)
  # hess is symmetric (lower triangle mirrored from upper).
  expect_equal(out$hess, t(out$hess), tolerance = 1e-12)
})

# ======================================================================
# solve_wls_p1 : one damped Newton step for the 1-type WLS solve
# ======================================================================

test_that("rctd_cpp_solve_wls_p1 reproduces the closed-form damped step", {
  fn <- .cpp_rctd("rctd_cpp_solve_wls_p1")
  tb <- .rctd_tables(7)
  set.seed(7)
  S <- matrix(abs(stats::rnorm(6, 1, 0.3)), 6, 1)
  y <- c(0, 1, 3, 7, 12, 15)
  n_umi <- 100; initial <- 0.5
  out <- fn(S, y, NULL, initial, n_umi, tb$Q_mat, tb$SQ_mat, tb$X_vals, tb$K_val)

  threshold <- max(1e-4, n_umi * 1e-7)
  sol <- max(initial, 0); grad <- 0; hess <- 0
  for (i in 1:6) {
    si <- S[i, 1]; pred <- abs(si * sol); if (pred < threshold) pred <- threshold
    d <- .rctd_spline(y[i], pred, tb)
    grad <- grad - d[["d1"]] * si
    hess <- hess - d[["d2"]] * si * si
  }
  d_vec <- -grad; d_mat <- max(hess, 1e-3); nf <- abs(d_mat)
  step <- (d_vec / nf) / (d_mat / nf + 1e-7)
  if (step < -sol) step <- -sol
  expect_equal(out[1], sol + 0.3 * step, tolerance = 1e-10)
})

# ======================================================================
# solve_wls_p2 : one bound-constrained 2-type QP step (finite, structure)
# ======================================================================

test_that("rctd_cpp_solve_wls_p2 returns a finite length-2 step that respects the lower bound", {
  fn <- .cpp_rctd("rctd_cpp_solve_wls_p2")
  tb <- .rctd_tables(7)
  set.seed(8)
  S <- matrix(abs(stats::rnorm(6 * 2, 1, 0.3)), 6, 2)
  y <- c(0, 1, 3, 7, 12, 15)
  initial <- c(0.5, 0.5)
  out <- fn(S, y, NULL, initial, 100, tb$Q_mat, tb$SQ_mat, tb$X_vals, tb$K_val)
  expect_length(out, 2L)
  expect_true(all(is.finite(out)))
  # The QP enforces step >= -solution, so the updated weights never drop
  # below zero (sol + 0.3*step >= sol - 0.3*sol = 0.7*sol > 0).
  expect_true(all(out >= 0))
})

# ======================================================================
# irwls_sparse_p12 / irwls_full_nonbulk : iterative solvers (shape + finiteness)
# ======================================================================

test_that("rctd_cpp_irwls_sparse_p12 returns weights/converged/score of the right shape", {
  fn <- .cpp_rctd("rctd_cpp_irwls_sparse_p12")
  tb <- .rctd_tables()
  set.seed(102)
  S <- matrix(abs(stats::rnorm(6 * 2, 1, 0.3)), 6, 2)
  y <- c(0, 1, 3, 7, 12, 15)
  fit <- fn(S, y, NULL, 100, 50L, 1e-3, tb$Q_mat, tb$SQ_mat, tb$X_vals, tb$K_val)
  expect_named(fit, c("weights", "converged", "score"))
  expect_length(fit$weights, 2L)
  expect_true(all(is.finite(fit$weights)))
  expect_true(is.logical(fit$converged))
  expect_true(is.finite(fit$score))
  # p == 1 branch (single column) returns a length-1 weight.
  fit1 <- fn(S[, 1, drop = FALSE], y, NULL, 100, 50L, 1e-3,
             tb$Q_mat, tb$SQ_mat, tb$X_vals, tb$K_val)
  expect_length(fit1$weights, 1L)
})

test_that("rctd_cpp_irwls_full_nonbulk returns p>2 weights via the active-set QP", {
  fn <- .cpp_rctd("rctd_cpp_irwls_full_nonbulk")
  tb <- .rctd_tables()
  set.seed(103)
  S <- matrix(abs(stats::rnorm(6 * 3, 1, 0.3)), 6, 3)
  y <- c(0, 1, 3, 7, 12, 15)
  fit <- fn(S, y, NULL, 100, 50L, 1e-3, tb$Q_mat, tb$SQ_mat, tb$X_vals, tb$K_val)
  expect_named(fit, c("weights", "converged"))
  expect_length(fit$weights, 3L)
  expect_true(all(is.finite(fit$weights)))
  expect_true(is.logical(fit$converged))
})

# ======================================================================
# score_sparse_candidates / fit_sparse_pair : doublet candidate scoring
# ======================================================================

test_that("rctd_cpp_score_sparse_candidates returns the singlet/pair score structure", {
  fn <- .cpp_rctd("rctd_cpp_score_sparse_candidates")
  tb <- .rctd_tables()
  set.seed(104)
  profiles <- matrix(abs(stats::rnorm(6 * 4, 1, 0.3)), 6, 4)
  y <- c(0, 1, 3, 7, 12, 15)
  res <- fn(profiles, y, 1:4L, NULL, 100, 1e-3,
            tb$Q_mat, tb$SQ_mat, tb$X_vals, tb$K_val)
  expect_named(res, c("singlet_scores", "score_mat", "min_score", "min_i", "min_j"))
  expect_length(res$singlet_scores, 4L)
  expect_equal(dim(res$score_mat), c(4L, 4L))
  # score_mat is symmetric; the argmin pair lives off the diagonal.
  expect_equal(res$score_mat, t(res$score_mat), tolerance = 1e-12)
  expect_true(res$min_i >= 1 && res$min_i <= 4)
  expect_true(res$min_j >= 1 && res$min_j <= 4)
  expect_true(res$min_i != res$min_j)
  # The chosen pair score equals the matrix entry it was selected from.
  expect_equal(res$min_score, res$score_mat[res$min_i, res$min_j], tolerance = 1e-12)
})

test_that("rctd_cpp_fit_sparse_pair fits a 2-type pair to convergence shape", {
  fn <- .cpp_rctd("rctd_cpp_fit_sparse_pair")
  tb <- .rctd_tables()
  set.seed(105)
  profiles <- matrix(abs(stats::rnorm(6 * 4, 1, 0.3)), 6, 4)
  y <- c(0, 1, 3, 7, 12, 15)
  fit <- fn(profiles, y, c(1L, 2L), NULL, 100, 50L, 1e-3,
            tb$Q_mat, tb$SQ_mat, tb$X_vals, tb$K_val)
  expect_named(fit, c("weights", "converged", "score"))
  expect_length(fit$weights, 2L)
  expect_true(all(is.finite(fit$weights)))
  expect_true(is.finite(fit$score))
})
