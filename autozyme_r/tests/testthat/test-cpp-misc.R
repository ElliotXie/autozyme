# Direct unit tests for the remaining per-patch Rcpp kernels:
#   rctd.cpp            rctd_cpp_row_idx1 / rctd_cpp_row_idx_mat
#   cellchat.cpp        cpp_aggregate_triMean
#   tradeseq.cpp        nb_dev_resids_cpp / linkinv_log_cpp / nb_Dd_cpp / nb_ls_cpp
#   maftools.cpp        zyme_fill_dcast
#   scriabin.cpp        fast_lr_outer_triplets_cpp
#   soft_assignment.cpp zyme_soft_assignment
#   score_ligands.cpp   score_ligands_cpp
#   fgsea.cpp           calcEsLeBatchCpp
#   seurat_neighbors.cpp turbo_annoy_build_search
#   native_blas.cpp     az_blas_info / az_blas_crossprod
#   wgcna.cpp           accelerate_crossprod (macOS only)
#   decontx.cpp         fast_decontXEM_cpp
#
# All of these are RcppParallel / std::thread / Armadillo / serial, so they do
# NOT use `#pragma omp`, so they run safely under OMP_NUM_THREADS=2.

.cpp <- function(name) {
  fn <- tryCatch(get(name, envir = asNamespace("autozyme")),
                 error = function(e) NULL)
  if (is.null(fn)) testthat::skip(paste0("kernel ", name, " not exported"))
  fn
}

# cpp_outer_Pnull carries a `#pragma omp parallel for` with no n_threads arg,
# so it segfaults under OMP_NUM_THREADS>1 on this macOS host (libomp-duplicate
# hazard). Skip under the threads=2 recipe; covered when run single-threaded.
.skip_if_omp_unsafe <- function() {
  v <- Sys.getenv("OMP_NUM_THREADS", unset = "")
  if (!v %in% c("", "0", "1")) {
    testthat::skip("OMP-parallel kernel: native segfault under OMP_NUM_THREADS>1 on this host")
  }
}

# ======================================================================
# rctd.cpp : row index = min(floor(y), k_val) + 1
# ======================================================================

test_that("rctd_cpp_row_idx1 caps y at k_val and is 1-based", {
  fn <- .cpp("rctd_cpp_row_idx1")
  y <- c(0.0, 0.9, 3.7, 12.0, 15.4)
  k <- 12L
  out <- fn(y, k)
  ref <- pmin(as.integer(y), k) + 1L      # truncation toward zero, then cap
  expect_equal(out, as.numeric(ref))
})

test_that("rctd_cpp_row_idx_mat applies the same map elementwise", {
  fn <- .cpp("rctd_cpp_row_idx_mat")
  m <- matrix(c(0.2, 5.8, 20.0, 8.1, 12.0, 13.5), 2, 3)
  k <- 12L
  out <- fn(m, k)
  ref <- pmin(matrix(as.integer(m), 2, 3), k) + 1L
  expect_equal(out, matrix(as.numeric(ref), 2, 3))
})

# ======================================================================
# cellchat.cpp : cpp_aggregate_triMean (type-7 quantile tri-mean per group)
# ======================================================================

test_that("cpp_aggregate_triMean matches a base-R tri-mean per (gene, group)", {
  fn <- .cpp("cpp_aggregate_triMean")
  set.seed(301)
  ngenes <- 4L; ncells <- 24L
  data <- matrix(rnorm(ngenes * ncells), ngenes, ncells)
  group <- as.integer(c(rep(1L, 10), rep(2L, 8), rep(3L, 6)))
  ng <- 3L
  out <- fn(data, group, ng)

  trimean <- function(x) {
    q <- stats::quantile(x, probs = c(.25, .50, .75), names = FALSE, type = 7)
    (q[1] + 2 * q[2] + q[3]) / 4
  }
  ref <- matrix(0, ngenes, ng)
  for (g in 1:ng) for (i in seq_len(ngenes))
    ref[i, g] <- trimean(data[i, group == g])
  expect_equal(unname(out), ref, tolerance = 1e-12)
})

test_that("cpp_aggregate_triMean returns 0 for an empty group", {
  fn <- .cpp("cpp_aggregate_triMean")
  data <- matrix(rnorm(6), 2, 3)
  group <- as.integer(c(1, 1, 1))   # group 2 has no cells
  out <- fn(data, group, 2L)
  expect_equal(out[, 2], c(0, 0))
})

# ======================================================================
# cellchat.cpp : cpp_outer_Pnull (Hill-function LR outer product)
# ======================================================================

test_that("cpp_outer_Pnull computes the Hill-function LR probability outer product", {
  .skip_if_omp_unsafe()
  fn <- .cpp("cpp_outer_Pnull")
  nC <- 2L; nLR <- 2L; Kh <- 0.5; n <- 1
  L <- matrix(c(1, 2, 3, 4), nC, nLR)   # numCluster x nLR
  R <- matrix(c(0.5, 1, 2, 1), nC, nLR)
  ag <- matrix(1, nC, nLR)              # ones = no agonist
  ant <- matrix(1, nC, nLR)             # ones = no antagonist
  out <- fn(L, R, ag, ant, nLR, nC, Kh, n)

  # Reference: for each LR pair k and cluster pair (c1, c2):
  #   dataLR = L[c1,k] * R[c2,k]; P = dataLR / (Kh + dataLR)  (n == 1, no ag/ant)
  ref <- numeric(nC * nC * nLR)
  pos <- 1
  for (k in seq_len(nLR)) for (c2 in seq_len(nC)) for (c1 in seq_len(nC)) {
    dataLR <- L[c1, k] * R[c2, k]
    ref[pos] <- dataLR / (Kh + dataLR)
    pos <- pos + 1
  }
  expect_equal(out, ref, tolerance = 1e-12)
})

# ======================================================================
# tradeseq.cpp : NB family scalar kernels
# ======================================================================

test_that("linkinv_log_cpp is exp(eta) floored at machine eps", {
  fn <- .cpp("linkinv_log_cpp")
  eta <- c(-1000, -1, 0, 1, 2)
  out <- fn(eta)
  ref <- pmax(exp(eta), .Machine$double.eps)
  expect_equal(out, ref, tolerance = 1e-15)
  expect_true(all(out >= .Machine$double.eps))
})

test_that("nb_dev_resids_cpp matches the NB deviance-residual formula", {
  fn <- .cpp("nb_dev_resids_cpp")
  y <- c(0, 1, 3, 7)
  mu <- c(0.5, 1.2, 2.5, 6.0)
  wt <- c(1, 1, 1, 1)
  theta_log <- log(4.5)
  theta <- exp(theta_log)
  out <- fn(y, mu, wt, theta_log)
  y_or_1 <- pmax(y, 1)
  ref <- 2 * wt * (y * log(y_or_1 / mu) - (y + theta) * log((y + theta) / (mu + theta)))
  expect_equal(out, ref, tolerance = 1e-12)
})

test_that("nb_Dd_cpp level-0 derivatives match the closed form", {
  fn <- .cpp("nb_Dd_cpp")
  y <- c(1, 2, 5); mu <- c(1.1, 2.2, 4.5); wt <- c(1, 1, 1)
  theta_log <- log(3.0); theta <- exp(theta_log)
  out <- fn(y, mu, theta_log, wt, 0L)
  yth <- y + theta; muth <- mu + theta
  ref_Dmu  <- 2 * wt * (yth / muth - y / mu)
  ref_Dmu2 <- -2 * wt * (yth / muth^2 - y / mu^2)
  ref_EDmu2 <- 2 * wt * (1 / mu - 1 / muth)
  expect_equal(out$Dmu, ref_Dmu, tolerance = 1e-12)
  expect_equal(out$Dmu2, ref_Dmu2, tolerance = 1e-12)
  expect_equal(out$EDmu2, ref_EDmu2, tolerance = 1e-12)
  # level 0 returns only the three base derivatives
  expect_false("Dth" %in% names(out))
})

test_that("nb_ls_cpp saddle-point ls term matches an R reference", {
  fn <- .cpp("nb_ls_cpp")
  y <- c(0, 2, 4, 1); w <- c(1, 1, 1, 1)
  theta_log <- log(2.5); Theta <- exp(theta_log)
  out <- fn(y, w, theta_log, 1.0)
  # ls = -sum_i w_i * [ (y+T)log(y+T) - ylogy + lgamma(y+1)
  #                     - T*log(T) + lgamma(T) - lgamma(T+y) ]
  yth <- y + Theta
  ylogy <- ifelse(y > 0, y * log(y), y)
  term <- yth * log(yth) - ylogy + lgamma(y + 1) -
    Theta * log(Theta) + lgamma(Theta) - lgamma(Theta + y)
  ref_ls <- -sum(term * w)
  expect_equal(out$ls, ref_ls, tolerance = 1e-9)
})

# ======================================================================
# maftools.cpp : zyme_fill_dcast (triplet -> dense integer matrix)
# ======================================================================

test_that("zyme_fill_dcast scatters (row, col, count) triplets into a matrix", {
  fn <- .cpp("zyme_fill_dcast")
  rows <- as.integer(c(1, 2, 1, 3))   # 1-based
  cols <- as.integer(c(1, 1, 2, 2))
  cnts <- as.integer(c(5, 7, 9, 11))
  out <- fn(rows, cols, cnts, 3L, 2L)
  ref <- matrix(0L, 3, 2)
  for (k in seq_along(rows)) ref[rows[k], cols[k]] <- cnts[k]
  expect_equal(out, ref)
})

test_that("zyme_fill_dcast last write wins on duplicate cells", {
  fn <- .cpp("zyme_fill_dcast")
  out <- fn(as.integer(c(1, 1)), as.integer(c(1, 1)), as.integer(c(3, 8)), 1L, 1L)
  expect_equal(out[1, 1], 8L)        # overwrites, not accumulates
})

# ======================================================================
# scriabin.cpp : fast_lr_outer_triplets_cpp (sparse LR outer products)
# ======================================================================

test_that("fast_lr_outer_triplets_cpp reconstructs the column-wise tcrossprod", {
  fn <- .cpp("fast_lr_outer_triplets_cpp")
  testthat::skip_if_not_installed("Matrix")
  # a: n_pairs x n_senders ; b: n_pairs x n_receivers
  a <- matrix(c(1, 0,
                2, 3), nrow = 2, byrow = TRUE)   # 2 pairs, 2 senders
  b <- matrix(c(1, 1,
                0, 2), nrow = 2, byrow = TRUE)   # 2 pairs, 2 receivers
  res <- fn(a, b)
  S <- Matrix::sparseMatrix(i = res$i, j = res$j, x = res$x,
                            dims = res$dims)
  out <- as.matrix(S)

  # Reference: column `pair` is vec(outer(a[pair, ], b[pair, ])) with senders
  # varying fastest (sender + receiver*n_senders).
  ref <- matrix(0, ncol(a) * ncol(b), nrow(a))
  for (pair in seq_len(nrow(a)))
    ref[, pair] <- as.vector(outer(a[pair, ], b[pair, ]))
  expect_equal(out, ref, tolerance = 1e-12)
})

# ======================================================================
# soft_assignment.cpp : zyme_soft_assignment (squared-dist softmax + objective)
# ======================================================================

test_that("zyme_soft_assignment matches an R softmax over squared distances", {
  fn <- .cpp("zyme_soft_assignment")
  set.seed(401)
  d <- 3L; N <- 5L; K <- 4L
  X <- matrix(rnorm(d * N), d, N)     # d x N (cols = points)
  C <- matrix(rnorm(d * K), d, K)     # d x K (cols = centers)
  sigma <- 0.7
  res <- fn(X, C, sigma)

  # Reference: squared distances, subtract per-row min, softmax(-dist/sigma).
  D2 <- matrix(0, N, K)
  for (i in seq_len(N)) for (k in seq_len(K))
    D2[i, k] <- sum((X[, i] - C[, k])^2)
  mind <- apply(D2, 1L, min)
  W <- exp(-(D2 - mind) / sigma)
  P <- W / rowSums(W)
  expect_equal(res$P, P, tolerance = 1e-9)
  expect_equal(rowSums(res$P), rep(1, N), tolerance = 1e-9)

  obj_ref <- -sigma * sum(log(rowSums(W)) - mind / sigma)
  expect_equal(res$obj, obj_ref, tolerance = 1e-9)
})

# ======================================================================
# score_ligands.cpp : AUROC / AUPR / Pearson over pos/neg row sets
# ======================================================================

test_that("score_ligands_cpp Pearson column matches cor() with a binary label", {
  fn <- .cpp("score_ligands_cpp")
  set.seed(501)
  nrow <- 8L; ncol <- 3L
  mat <- matrix(rnorm(nrow * ncol), nrow, ncol)
  pos <- as.integer(c(1, 2, 3))       # 1-based positive rows
  neg <- as.integer(c(4, 5, 6, 7, 8)) # negative rows
  cols <- as.integer(seq_len(ncol))
  out <- fn(mat, pos, neg, cols, 1L)  # out: ncol x 4 (auroc, aupr, aupr_adj, pearson)

  label <- numeric(nrow)
  label[pos] <- 1
  for (j in seq_len(ncol)) {
    # Pearson over the pos+neg rows (the kernel uses n_rows = n_pos + n_neg)
    used <- c(pos, neg)
    r <- stats::cor(mat[used, j], label[used])
    expect_equal(out[j, 4], r, tolerance = 1e-9)
  }
  # AUROC and AUPR are valid probabilities in [0, 1]
  expect_true(all(out[, 1] >= -1e-9 & out[, 1] <= 1 + 1e-9))
  expect_true(all(out[, 2] >= -1e-9 & out[, 2] <= 1 + 1e-9))
})

# ======================================================================
# fgsea.cpp : calcEsLeBatchCpp (enrichment score + leading edge)
# ======================================================================

test_that("calcEsLeBatchCpp ES matches the cumulative GSEA statistic", {
  fn <- .cpp("calcEsLeBatchCpp")
  # Caller passes stats already sorted desc; a clean monotone ranking.
  stats <- c(5, 4, 3, 2, 1, -1, -2, -3)   # N = 8, descending
  selected <- list(as.integer(c(1, 2, 3)))  # top-3 -> strong positive ES
  res <- fn(stats, selected, "std")

  # Reference: calcGseaStat (running ES) for the std score type.
  gsea_es <- function(stats, S) {
    N <- length(stats); k <- length(S)
    S <- sort(S)
    NR <- sum(stats[S])
    cumSum <- 0; maxTop <- -Inf; minBot <- Inf
    Nm <- N - k
    for (j in seq_len(k)) {
      cur <- stats[S[j]]
      base <- (S[j] - j) / Nm
      cumSum <- cumSum + cur / NR
      top <- cumSum - base
      bottom <- (cumSum - cur / NR) - base
      maxTop <- max(maxTop, top)
      minBot <- min(minBot, bottom)
    }
    if (maxTop > -minBot) maxTop else if (maxTop < -minBot) minBot else 0
  }
  expect_equal(res$es[1], gsea_es(stats, c(1, 2, 3)), tolerance = 1e-10)
  # leading edge for a positive ES is a prefix of the sorted set
  expect_true(length(res$le[[1]]) >= 1)
  expect_true(all(res$le[[1]] %in% c(1, 2, 3)))
})

# ======================================================================
# seurat_neighbors.cpp : turbo_annoy_build_search (approx kNN indices)
# ======================================================================

test_that("turbo_annoy_build_search returns self as the nearest neighbor", {
  fn <- .cpp("turbo_annoy_build_search")
  testthat::skip_if_not_installed("RcppAnnoy")
  set.seed(601)
  n <- 12L; f <- 4L; k <- 3L
  # well-separated clusters so kNN is unambiguous
  data <- rbind(
    matrix(rnorm(6 * f, mean = 0), 6, f),
    matrix(rnorm(6 * f, mean = 20), 6, f)
  )
  idx <- fn(data, k, 50L, 1L)
  expect_equal(dim(idx), c(n, k))
  # nearest neighbor of each point is itself (1-based)
  expect_equal(idx[, 1], seq_len(n))
  # all returned indices are valid row ids
  expect_true(all(idx >= 1 & idx <= n))
})

# ======================================================================
# native_blas.cpp : az_blas_info / az_blas_crossprod
# ======================================================================

test_that("az_blas_info returns a well-formed availability list", {
  fn <- .cpp("az_blas_info")
  info <- fn()
  expect_true(is.list(info))
  expect_true(all(c("available", "backend", "abi", "thread_control") %in% names(info)))
  expect_true(is.logical(info$available))
})

test_that("az_blas_crossprod equals crossprod when a backend is available", {
  info_fn <- .cpp("az_blas_info")
  if (!isTRUE(info_fn()$available)) {
    testthat::skip("no dynamic BLAS backend loadable on this host")
  }
  fn <- .cpp("az_blas_crossprod")
  set.seed(701)
  X <- matrix(rnorm(20), 5, 4)
  expect_equal(fn(X, 1L), crossprod(X), tolerance = 1e-10)
})

# ======================================================================
# wgcna.cpp : accelerate_crossprod (macOS Apple Accelerate)
# ======================================================================

test_that("accelerate_crossprod equals t(X) %*% X on macOS", {
  if (Sys.info()[["sysname"]] != "Darwin") {
    testthat::skip("accelerate_crossprod is a macOS-only Accelerate kernel")
  }
  fn <- .cpp("accelerate_crossprod")
  set.seed(801)
  X <- matrix(rnorm(24), 6, 4)
  out <- fn(X)
  expect_equal(out, crossprod(X), tolerance = 1e-10)
})

# ======================================================================
# decontx.cpp : fast_decontXEM_cpp (single EM step, serial branch)
# ======================================================================

test_that("fast_decontXEM_cpp single EM step matches an R reference", {
  fn <- .cpp("fast_decontXEM_cpp")
  testthat::skip_if_not_installed("Matrix")
  set.seed(901)
  nG <- 6L; nC <- 4L; K <- 2L
  counts <- methods::as(
    Matrix::Matrix(matrix(rpois(nG * nC, 2) * 1.0, nG, nC), sparse = TRUE),
    "CsparseMatrix")
  colsums <- Matrix::colSums(counts)
  phi <- matrix(runif(nG * K), nG, K); phi <- sweep(phi, 2L, colSums(phi), "/")
  eta <- matrix(runif(nG * K), nG, K); eta <- sweep(eta, 2L, colSums(eta), "/")
  z <- as.integer(c(1, 2, 1, 2))
  theta <- runif(nC, 0.1, 0.5)
  delta <- c(1, 1)
  pc <- 1e-20
  res <- fn(counts, colsums, theta, FALSE, eta, phi, z, FALSE, delta, pc, 1L)

  # Reference E/M for new_phi (native counts) + theta + contamination.
  dense <- as.matrix(counts)
  new_phi <- matrix(0, nG, K)
  native_total <- numeric(nC)
  for (j in seq_len(nC)) {
    k <- z[j]
    tj <- theta[j] + pc; tjc <- 1 - theta[j] + pc
    for (i in seq_len(nG)) {
      x <- dense[i, j]
      if (x == 0) next
      pn <- (phi[i, k] + pc) * tj
      pcn <- (eta[i, k] + pc) * tjc
      px <- pn / (pn + pcn) * x
      new_phi[i, k] <- new_phi[i, k] + px
      native_total[j] <- native_total[j] + px
    }
  }
  new_phi <- sweep(new_phi, 2L, colSums(new_phi), "/")
  new_theta <- (native_total + delta[1]) / (colsums + delta[1] + delta[2])
  contamination <- 1 - native_total / colsums

  expect_equal(res$phi, new_phi, tolerance = 1e-9)
  expect_equal(res$theta, as.numeric(new_theta), tolerance = 1e-9)
  expect_equal(res$contamination, as.numeric(contamination), tolerance = 1e-9)
})
