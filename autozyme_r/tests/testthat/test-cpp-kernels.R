# Direct unit tests for the pure-numeric sparse/dense reduction and Seurat
# scale/normalize/VST Rcpp kernels.
#
# These kernels are exported from RcppExports.R but were only exercised
# *indirectly* (via patched Seurat / upstream calls) by the contract tests.
# Here we call autozyme:::<kernel>() directly on tiny synthetic inputs and
# assert against a hand-written base-R / Matrix reference.
#
# Conventions match helper-tiny-seurat.R: skip cleanly when an upstream is
# absent, otherwise assert real numeric behavior with a tight tolerance.

# ---- helpers -------------------------------------------------------------

.cpp <- function(name) {
  fn <- tryCatch(get(name, envir = asNamespace("autozyme")),
                 error = function(e) NULL)
  if (is.null(fn)) testthat::skip(paste0("kernel ", name, " not exported"))
  fn
}

# Several kernels carry a `#pragma omp parallel for`. Entering an OpenMP
# parallel region segfaults (invalid-permissions fault at ~0x540) on this
# macOS host when OMP_NUM_THREADS > 1 (the libomp-duplicate hazard noted in
# the project memory). The campaign env recipe pins OMP_NUM_THREADS=2, so we
# skip OMP-parallel kernel calls cleanly under that recipe; they are still
# covered when the file is run single-threaded (OMP_NUM_THREADS=1).
.skip_if_omp_unsafe <- function() {
  v <- Sys.getenv("OMP_NUM_THREADS", unset = "")
  if (!v %in% c("", "0", "1")) {
    testthat::skip("OMP-parallel kernel: native segfault under OMP_NUM_THREADS>1 on this host")
  }
}

.tiny_dgc <- function(seed = 1, nr = 6, nc = 5, dens = 0.5) {
  testthat::skip_if_not_installed("Matrix")
  set.seed(seed)
  m <- matrix(0, nr, nc)
  k <- max(1L, round(nr * nc * dens))
  idx <- sample(seq_len(nr * nc), k)
  m[idx] <- round(runif(k, 0.1, 5), 3)
  methods::as(Matrix::Matrix(m, sparse = TRUE), "CsparseMatrix")
}

# ======================================================================
# shared_sparse.cpp : az_dgc_row_stats_cpp
# ======================================================================

test_that("az_dgc_row_stats_cpp matches Matrix::rowSums / row var", {
  fn <- .cpp("az_dgc_row_stats_cpp")
  X <- .tiny_dgc(seed = 11, nr = 7, nc = 9)
  dense <- as.matrix(X)
  res <- fn(X, 0.0)

  expect_equal(res$sum, Matrix::rowSums(X), tolerance = 1e-10)
  expect_equal(res$mean, rowMeans(dense), tolerance = 1e-10)
  expect_equal(res$variance, apply(dense, 1L, stats::var), tolerance = 1e-10)
  expect_equal(res$nnz, rowSums(dense != 0))
  expect_equal(res$detected, rowSums(dense > 0))
})

test_that("az_dgc_row_stats_cpp detected_threshold counts strictly greater", {
  fn <- .cpp("az_dgc_row_stats_cpp")
  X <- .tiny_dgc(seed = 12, nr = 5, nc = 8)
  dense <- as.matrix(X)
  res <- fn(X, 1.0)
  expect_equal(res$detected, rowSums(dense > 1.0))
})

test_that("az_dgc_row_stats_cpp single-column gives NA variance", {
  fn <- .cpp("az_dgc_row_stats_cpp")
  X <- .tiny_dgc(seed = 13, nr = 4, nc = 1)
  res <- fn(X, 0.0)
  expect_true(all(is.na(res$variance)))
  expect_equal(res$mean, as.numeric(as.matrix(X)), tolerance = 1e-12)
})

# ======================================================================
# shared_sparse.cpp : az_dgc_group_summary_cpp
# ======================================================================

test_that("az_dgc_group_summary_cpp sum/mean by group match a base-R loop", {
  fn <- .cpp("az_dgc_group_summary_cpp")
  X <- .tiny_dgc(seed = 21, nr = 6, nc = 8)
  dense <- as.matrix(X)
  groups <- as.integer(c(1, 1, 2, 2, 2, 3, 3, 1))  # 1-based codes
  ng <- 3L
  res <- fn(X, groups, ng, 0.0)

  ref_sum <- sapply(seq_len(ng), function(g) rowSums(dense[, groups == g, drop = FALSE]))
  ref_size <- as.integer(table(factor(groups, levels = seq_len(ng))))
  ref_mean <- sweep(ref_sum, 2L, ref_size, "/")

  expect_equal(unname(res$sum_by_group), unname(ref_sum), tolerance = 1e-10)
  expect_equal(unname(res$mean_by_group), unname(ref_mean), tolerance = 1e-10)
  expect_equal(as.integer(res$group_size), ref_size)
})

test_that("az_dgc_group_summary_cpp ignores out-of-range group codes", {
  fn <- .cpp("az_dgc_group_summary_cpp")
  X <- .tiny_dgc(seed = 22, nr = 4, nc = 6)
  dense <- as.matrix(X)
  groups <- as.integer(c(1, 2, 2, 99, NA, 1))  # 99 and NA must be dropped
  ng <- 2L
  res <- fn(X, groups, ng, 0.0)
  valid <- groups %in% c(1L, 2L)
  ref_size <- c(sum(groups == 1, na.rm = TRUE), sum(groups == 2, na.rm = TRUE))
  expect_equal(as.integer(res$group_size), as.integer(ref_size))
  expect_equal(res$sum_by_group[, 1], rowSums(dense[, which(groups == 1), drop = FALSE]),
               tolerance = 1e-10)
})

# ======================================================================
# shared_sparse.cpp : az_dense_group_summary_cpp
# ======================================================================

test_that("az_dense_group_summary_cpp matches the R fallback (no NA)", {
  fn <- .cpp("az_dense_group_summary_cpp")
  ns <- asNamespace("autozyme")
  set.seed(31)
  x <- matrix(rnorm(5 * 9), 5, 9)
  groups <- as.integer(c(1, 1, 2, 3, 3, 3, 2, 1, 2))
  ng <- 3L
  res <- fn(x, groups, ng, 0.0, FALSE)
  ref <- ns$.az_dense_group_summary_fallback(x, groups, 0, FALSE)
  expect_equal(unname(res$sum_by_group), unname(ref$sum_by_group), tolerance = 1e-10)
  expect_equal(unname(res$mean_by_group), unname(ref$mean_by_group), tolerance = 1e-10)
  expect_equal(as.integer(res$group_size), as.integer(ref$group_size))
})

test_that("az_dense_group_summary_cpp na_rm vs propagate", {
  fn <- .cpp("az_dense_group_summary_cpp")
  x <- matrix(c(1, 2, NA, 4,
                5, 6, 7, 8), nrow = 2, byrow = TRUE)
  groups <- as.integer(c(1, 1, 2, 2))
  ng <- 2L
  # propagate: group 2 of row 1 has an NA -> sum NA, mean NA
  prop <- fn(x, groups, ng, 0.0, FALSE)
  expect_true(is.na(prop$mean_by_group[1, 2]))
  # na_rm: drop the NA, mean of remaining valid {4} = 4
  rm_ <- fn(x, groups, ng, 0.0, TRUE)
  expect_equal(rm_$mean_by_group[1, 2], 4, tolerance = 1e-12)
})

# ======================================================================
# seurat_variable_features.cpp : turbo_FastSparseRowMeanVar / RowVarStd
# ======================================================================

test_that("turbo_FastSparseRowMeanVar matches row mean/variance", {
  fn <- .cpp("turbo_FastSparseRowMeanVar")
  X <- .tiny_dgc(seed = 41, nr = 8, nc = 12)
  dense <- as.matrix(X)
  res <- fn(X@p, X@i, X@x, nrow(X), ncol(X))
  expect_equal(res$mean, rowMeans(dense), tolerance = 1e-10)
  # kernel clamps negative variance to 0; tiny data has positive var
  expect_equal(res$variance, apply(dense, 1L, stats::var), tolerance = 1e-10)
  expect_equal(res$nnz, rowSums(dense != 0))
})

test_that("turbo_FastSparseRowVarStd matches a clipped standardized row var", {
  fn <- .cpp("turbo_FastSparseRowVarStd")
  X <- .tiny_dgc(seed = 42, nr = 6, nc = 20)
  dense <- as.matrix(X)
  mu <- rowMeans(dense)
  sdv <- apply(dense, 1L, stats::sd)
  vmax <- sqrt(ncol(X))
  nnz <- rowSums(dense != 0)
  res <- fn(X@p, X@i, X@x, nrow(X), ncol(X), mu, sdv, vmax, as.integer(nnz))

  # Reference: kernel clips only the POSITIVE tail at vmax (z > vmax).
  ref <- vapply(seq_len(nrow(dense)), function(r) {
    if (sdv[r] == 0) return(0)
    z <- (dense[r, ] - mu[r]) / sdv[r]
    z[z > vmax] <- vmax
    sum(z * z) / (ncol(dense) - 1)
  }, numeric(1))
  expect_equal(res, ref, tolerance = 1e-9)
})

# ======================================================================
# seurat_scale.cpp : turbo_scale_sparse_full
# ======================================================================

test_that("turbo_scale_sparse_full matches a Seurat-style one-sided z-scale", {
  .skip_if_omp_unsafe()
  fn <- .cpp("turbo_scale_sparse_full")
  X <- .tiny_dgc(seed = 51, nr = 6, nc = 25)
  dense <- as.matrix(X)
  rows0 <- as.integer(c(0, 2, 4))  # 0-based selected genes
  scale_max <- 10
  res <- fn(X, rows0, scale_max)

  ref <- t(sapply(rows0 + 1L, function(g) {
    v <- dense[g, ]
    m <- mean(v)
    s <- stats::sd(v)
    if (!is.finite(s) || s == 0) s <- 1
    z <- (v - m) / s
    z[z > scale_max] <- scale_max  # Seurat caps only positive tail
    z
  }))
  expect_equal(unname(res), unname(ref), tolerance = 1e-9)
})

# ======================================================================
# seurat_normalize.cpp : seurat_log_normalize_dgc (in-place)
# ======================================================================

test_that("seurat_log_normalize_dgc approximates log1p(scaled counts)", {
  .skip_if_omp_unsafe()
  fn <- .cpp("seurat_log_normalize_dgc")
  X <- .tiny_dgc(seed = 61, nr = 8, nc = 10, dens = 0.6)
  # use integer-ish counts so columns sum > 0
  X@x <- round(X@x) + 1
  dense <- as.matrix(X)
  scale_factor <- 1e4

  Xc <- methods::as(X, "CsparseMatrix")  # mutate a copy
  Xc@x <- X@x + 0  # ensure own storage
  fn(Xc, scale_factor, 100L)

  ref <- apply(dense, 2L, function(col) {
    s <- sum(col)
    if (s > 0) log1p(col * (scale_factor / s)) else col
  })
  # kernel uses a fast polynomial log1p approximation; loose tolerance
  expect_equal(as.matrix(Xc), ref, tolerance = 1e-6)
})

# ======================================================================
# seurat_sctransform.cpp : turbo_csc_to_csr
# ======================================================================

test_that("turbo_csc_to_csr round-trips a sparse matrix to CSR", {
  fn <- .cpp("turbo_csc_to_csr")
  X <- .tiny_dgc(seed = 71, nr = 5, nc = 6)
  dense <- as.matrix(X)
  res <- fn(X@i, X@p, X@x, nrow(X), ncol(X))

  # Reconstruct dense from CSR (row_ptr / col_idx / vals).
  rebuilt <- matrix(0, nrow(X), ncol(X))
  rp <- res$row_ptr
  for (r in seq_len(nrow(X))) {
    for (k in seq.int(rp[r] + 1L, rp[r + 1L], length.out = max(0L, rp[r + 1L] - rp[r]))) {
      if (rp[r + 1L] <= rp[r]) break
      rebuilt[r, res$col_idx[k] + 1L] <- res$vals[k]
    }
  }
  expect_equal(rebuilt, dense, tolerance = 1e-12)
  expect_equal(length(res$col_idx), length(X@x))
})

# ======================================================================
# seurat_sctransform.cpp : turbo_fused_resid_center_sparse
# ======================================================================

test_that("turbo_fused_resid_center_sparse produces row-centered residuals", {
  .skip_if_omp_unsafe()
  fn <- .cpp("turbo_fused_resid_center_sparse")
  set.seed(81)
  ncells <- 12L
  ngenes <- 3L
  cell_mu_base <- runif(ncells, 0.5, 1.5)
  intercepts <- rnorm(ngenes)
  theta <- runif(ngenes, 5, 20)
  # Build a CSR layout: full-gene rows, all cells nonzero counts.
  counts <- matrix(rpois(ngenes * ncells, 3), ngenes, ncells)
  row_ptr <- integer(ngenes + 1L)
  col_idx <- integer(0)
  vals <- numeric(0)
  for (g in seq_len(ngenes)) {
    nz <- which(counts[g, ] != 0)
    col_idx <- c(col_idx, nz - 1L)
    vals <- c(vals, counts[g, nz])
    row_ptr[g + 1L] <- row_ptr[g] + length(nz)
  }
  gene_idx <- as.integer(seq_len(ngenes) - 1L)
  min_var <- 1e-8
  wlo <- -50; whi <- 50; nlo <- -sqrt(ncells); nhi <- sqrt(ncells)

  res <- fn(intercepts, cell_mu_base, as.integer(row_ptr), as.integer(col_idx),
            vals, gene_idx, theta, min_var, wlo, whi, nlo, nhi)

  # Reference per gene: pearson residual, clip wide then narrow, center.
  ref <- matrix(0, ngenes, ncells)
  for (g in seq_len(ngenes)) {
    mu <- exp(intercepts[g]) * cell_mu_base
    v <- mu + mu^2 / theta[g]
    v[v < min_var] <- min_var
    r <- (counts[g, ] - mu) / sqrt(v)
    r <- pmin(pmax(r, wlo), whi)
    r <- pmin(pmax(r, nlo), nhi)
    ref[g, ] <- r - mean(r)
  }
  expect_equal(res, ref, tolerance = 1e-9)
  # rows are centered: mean ~ 0
  expect_equal(rowMeans(res), rep(0, ngenes), tolerance = 1e-9)
})

# ======================================================================
# seurat_sctransform.cpp : turbo_stats_correct_sparse (stats-only branch)
# ======================================================================

test_that("turbo_stats_correct_sparse residual var/mean match a base-R pass", {
  .skip_if_omp_unsafe()
  fn <- .cpp("turbo_stats_correct_sparse")
  set.seed(91)
  ncells <- 10L
  ngenes <- 2L
  cell_mu_base <- runif(ncells, 0.5, 1.5)
  intercepts <- rnorm(ngenes)
  theta <- runif(ngenes, 5, 20)
  counts <- matrix(rpois(ngenes * ncells, 3), ngenes, ncells)
  row_ptr <- integer(ngenes + 1L); col_idx <- integer(0); vals <- numeric(0)
  for (g in seq_len(ngenes)) {
    nz <- which(counts[g, ] != 0)
    col_idx <- c(col_idx, nz - 1L); vals <- c(vals, counts[g, nz])
    row_ptr[g + 1L] <- row_ptr[g] + length(nz)
  }
  clip_lo <- -sqrt(ncells); clip_hi <- sqrt(ncells); min_var <- 1e-8
  res <- fn(intercepts, cell_mu_base, as.integer(row_ptr), as.integer(col_idx),
            vals, as.integer(seq_len(ngenes) - 1L), theta, numeric(ncells),
            min_var, clip_lo, clip_hi, FALSE)

  ref_var <- numeric(ngenes); ref_mean <- numeric(ngenes)
  for (g in seq_len(ngenes)) {
    mu <- exp(intercepts[g]) * cell_mu_base
    v <- mu + mu^2 / theta[g]; v[v < min_var] <- min_var
    r <- (counts[g, ] - mu) / sqrt(v)
    r <- pmin(pmax(r, clip_lo), clip_hi)
    ref_mean[g] <- mean(r)
    ref_var[g] <- stats::var(r)
  }
  expect_equal(res$res_mean, ref_mean, tolerance = 1e-9)
  expect_equal(res$res_var, ref_var, tolerance = 1e-9)
})
