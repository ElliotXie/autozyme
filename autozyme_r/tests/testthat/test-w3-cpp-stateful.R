# Wave-3 direct unit tests for the stateful MCMC / IRLS / EM / bootstrap
# kernels that wave-1 deferred because they need a fully-initialized state:
#
#   bayesspace.cpp      fast_iterate_t_impl          (Gibbs/MH spatial cluster loop)
#   cpp_updateState.cpp cpp_updateState              (MAST bayesglm IRLS step)
#   fgsea.cpp           fastFgseaMultilevelBatchCpp  (EsRuler multilevel MCMC)
#   cellchat.cpp        cpp_aggregate_triMean_boot   (bootstrap triMean tensor)
#                       cpp_unified_inner            (bootstrap reject counts)
#
# Strategy: build the smallest valid initialized state and either
#   (a) assert a deterministic step matches an exact R reference
#       (cpp_updateState gaussian = augmented WLS; cellchat boot/unified =
#        re-derived numeric reference), or
#   (b) assert invariants (shape, range, monotone-RNG determinism, finiteness)
#       for the genuinely stochastic loops (bayesspace Gibbs, fgsea MCMC).
#
# cpp_aggregate_triMean_boot and cpp_unified_inner carry a bare
# `#pragma omp parallel` (no num_threads arg), so they segfault under
# OMP_NUM_THREADS>1 on the libomp-duplicate macOS host. Skip those two under
# the threads>1 recipe; they are covered single-threaded (OMP_NUM_THREADS=1).
# The other three use std::thread / serial Armadillo and are OMP-safe.

.cpp_sf <- function(name) {
  fn <- tryCatch(get(name, envir = asNamespace("autozyme")),
                 error = function(e) NULL)
  if (is.null(fn)) testthat::skip(paste0("kernel ", name, " not exported"))
  fn
}

.skip_if_omp_unsafe <- function() {
  v <- Sys.getenv("OMP_NUM_THREADS", unset = "")
  if (!v %in% c("", "0", "1")) {
    testthat::skip("OMP-parallel kernel: native segfault under OMP_NUM_THREADS>1 on this host")
  }
}

# ======================================================================
# cpp_updateState (MAST IRLS) — gaussian + identity is fully deterministic;
# assert it equals the augmented ridge-WLS solve done in R.
# ======================================================================

test_that("cpp_updateState gaussian-identity step equals the augmented WLS solve", {
  fn <- .cpp_sf("cpp_updateState")
  set.seed(3)
  nobs <- 20L; nvars <- 3L
  X <- cbind(1, matrix(stats::rnorm(nobs * (nvars - 1)), nobs, nvars - 1))
  y <- as.numeric(X %*% c(1.5, -0.7, 0.3) + stats::rnorm(nobs, 0, 0.4))
  weights <- rep(1, nobs); offset <- rep(0, nobs)
  # gaussian + identity: mu = eta, mu_eta_val = 1, varmu = 1.
  eta <- as.numeric(X %*% c(1, 0, 0)); mu <- eta
  mu_eta_val <- rep(1, nobs); varmu <- rep(1, nobs)
  dispersion <- 1.0
  prior_mean  <- rep(0, nvars)
  prior_scale <- rep(1e6, nvars)   # negligible ridge
  prior_df    <- rep(Inf, nvars)   # Inf -> skip the prior.sd update
  prior_sd    <- rep(1, nvars)
  x_aug <- rbind(X, diag(nvars))   # nstar x nvars

  res <- fn(eta, mu, mu_eta_val, varmu, dispersion, prior_sd, x_aug, X, y,
            weights, offset, prior_mean, prior_scale, prior_df,
            family_idx = 2L, intercept = 1L, scaled = 1L)

  # Reference augmented WLS.
  z <- (eta - offset) + (y - mu) / mu_eta_val
  w <- sqrt(weights * mu_eta_val^2 / varmu)
  z_star <- c(z, prior_mean)
  w_star <- c(w, sqrt(dispersion) / prior_scale)
  Xw <- x_aug * w_star; yw <- z_star * w_star
  coefs_ref <- as.numeric(solve(t(Xw) %*% Xw, t(Xw) %*% yw))

  expect_equal(res$Start, coefs_ref, tolerance = 1e-10)
  expect_equal(res$Coefold, coefs_ref, tolerance = 1e-10)
  # gaussian: new_mu == new_eta == X %*% coefs + offset.
  new_eta_ref <- as.numeric(X %*% coefs_ref + offset)
  expect_equal(res$eta, new_eta_ref, tolerance = 1e-10)
  expect_equal(res$mu, res$eta, tolerance = 1e-12)
  expect_equal(res$mu.eta.val, rep(1, nobs))
  expect_equal(res$varmu, rep(1, nobs))
  # gaussian deviance = sum weights * (y - mu)^2.
  expect_equal(res$dev, sum(weights * (y - new_eta_ref)^2), tolerance = 1e-10)
})

test_that("cpp_updateState binomial-logit step yields valid mu/varmu/dev invariants", {
  fn <- .cpp_sf("cpp_updateState")
  set.seed(5)
  nobs <- 30L; nvars <- 2L
  X <- cbind(1, stats::rnorm(nobs))
  y <- rbinom(nobs, 1, 0.5) * 1.0
  eta <- as.numeric(X %*% c(0, 0)); mu <- 1 / (1 + exp(-eta))
  mu_eta <- mu * (1 - mu); varmu <- mu_eta
  x_aug <- rbind(X, diag(nvars))
  res <- fn(eta, mu, mu_eta, varmu, 1.0, rep(1, nvars), x_aug, X, y,
            rep(1, nobs), rep(0, nobs), rep(0, nvars), rep(1e6, nvars),
            rep(Inf, nvars), family_idx = 1L, intercept = 1L, scaled = 1L)
  # mu = logistic(eta_new), strictly inside (0, 1); varmu = mu(1-mu).
  new_eta <- as.numeric(X %*% res$Start)
  expect_equal(res$mu, 1 / (1 + exp(-new_eta)), tolerance = 1e-10)
  expect_true(all(res$mu > 0 & res$mu < 1))
  expect_equal(res$varmu, res$mu * (1 - res$mu), tolerance = 1e-10)
  expect_true(is.finite(res$dev) && res$dev >= 0)
  # binomial path leaves dispersion fixed.
  expect_equal(res$dispersion, 1.0)
})

# ======================================================================
# fast_iterate_t_impl (BayesSpace Gibbs) — stochastic loop; assert shape,
# trajectory invariants (z stays in 1..q, init preserved), finiteness.
# ======================================================================

test_that("fast_iterate_t_impl runs a tiny Gibbs chain with valid structure", {
  fn <- .cpp_sf("fast_iterate_t_impl")
  set.seed(9)
  n <- 12L; d <- 2L; q <- 3L
  Y <- matrix(stats::rnorm(n * d), n, d)
  # df_j: per-spot 0-based neighbor index sets (a simple chain graph).
  df_j <- lapply(seq_len(n), function(j) {
    nb <- c(j - 1L, j + 1L); nb <- nb[nb >= 1 & nb <= n]; as.integer(nb - 1L)
  })
  init <- as.integer(sample(1:q, n, replace = TRUE))
  mu0 <- rep(0, d); lambda0 <- diag(d)
  gamma <- 2; alpha <- 2; beta <- 1
  nrep <- 20L; thin <- 5L
  res <- fn(Y, df_j, nrep, thin, n, d, gamma, q, init, mu0, lambda0, alpha, beta)

  n_saved <- nrep / thin + 1
  expect_named(res, c("z", "mu", "lambda", "weights", "plogLik"))
  expect_equal(dim(res$z), c(n_saved, n))
  expect_equal(dim(res$mu), c(n_saved, q * d))
  expect_equal(dim(res$weights), c(n_saved, n))
  expect_length(res$lambda, n_saved)
  # cluster labels stay in the valid range every saved iteration.
  expect_true(all(res$z >= 1 & res$z <= q))
  # the first saved row is the initial labeling.
  expect_equal(as.integer(res$z[1, ]), init)
  # gamma-distributed weights are strictly positive and finite.
  expect_true(all(res$weights > 0 & is.finite(res$weights)))
  expect_true(all(is.finite(res$mu)))
  # plogLik is filled for i = 1..nrep-1 (index 0 left NA by upstream).
  expect_length(res$plogLik, nrep)
  expect_true(all(is.finite(res$plogLik[-1])))
})

# ======================================================================
# fastFgseaMultilevelBatchCpp (fgsea EsRuler MCMC) — RNG seeded, so
# deterministic for a fixed seed; assert p-values in [0,1] + reproducibility.
# nthreads pinned to 1 (std::thread parallel; 1 keeps it single-threaded).
# ======================================================================

test_that("fastFgseaMultilevelBatchCpp gives reproducible p-values in [0,1]", {
  fn <- .cpp_sf("fastFgseaMultilevelBatchCpp")
  ranks <- sort(stats::rnorm(200), decreasing = TRUE)
  groupES <- list(c(0.3, -0.2, 0.5), c(0.1, 0.4))
  pathwaySizes <- c(10L, 15L)
  eps_per_group <- c(1e-10, 1e-10)
  o1 <- fn(groupES, pathwaySizes, eps_per_group, ranks,
           sampleSize = 101L, seed = 42L, sign = FALSE, nthreads = 1L)
  o2 <- fn(groupES, pathwaySizes, eps_per_group, ranks,
           sampleSize = 101L, seed = 42L, sign = FALSE, nthreads = 1L)

  expect_length(o1, length(groupES))
  for (g in seq_along(o1)) {
    df <- o1[[g]]
    expect_true(all(c("cppMPval", "cppIsCpGeHalf") %in% colnames(df)))
    expect_equal(nrow(df), length(groupES[[g]]))
    expect_true(all(df$cppMPval >= 0 & df$cppMPval <= 1))
    expect_true(is.logical(df$cppIsCpGeHalf))
  }
  # Same seed -> identical output (the EsRuler RNG path is deterministic).
  expect_equal(o1, o2)
})

test_that("fastFgseaMultilevelBatchCpp sign=TRUE path also yields valid p-values", {
  fn <- .cpp_sf("fastFgseaMultilevelBatchCpp")
  ranks <- sort(stats::rnorm(150), decreasing = TRUE)
  groupES <- list(c(0.4, 0.6))
  out <- fn(groupES, c(12L), c(1e-10), ranks,
            sampleSize = 81L, seed = 7L, sign = TRUE, nthreads = 1L)
  expect_length(out, 1L)
  expect_true(all(out[[1]]$cppMPval >= 0 & out[[1]]$cppMPval <= 1))
})

# ======================================================================
# cpp_aggregate_triMean_boot (cellchat bootstrap triMean) — OMP-parallel;
# identity permutation must reproduce the non-bootstrap triMean exactly.
# ======================================================================

test_that("cpp_aggregate_triMean_boot identity permutation equals cpp_aggregate_triMean", {
  .skip_if_omp_unsafe()
  boot_fn  <- .cpp_sf("cpp_aggregate_triMean_boot")
  plain_fn <- .cpp_sf("cpp_aggregate_triMean")
  set.seed(13)
  ngenes <- 4L; nC <- 24L; ng <- 3L
  data <- matrix(stats::rnorm(ngenes * nC), ngenes, nC)
  group <- as.integer(c(rep(1L, 10), rep(2L, 8), rep(3L, 6)))
  perm_id <- matrix(seq_len(nC), nC, 1)             # 1 bootstrap, identity
  boot <- boot_fn(data, group, ng, perm_id)
  plain <- plain_fn(data, group, ng)
  # boot is a flat ngenes*ng*nboot tensor, gene varying fastest.
  expect_length(boot, ngenes * ng * 1L)
  boot_mat <- matrix(boot[seq_len(ngenes * ng)], ngenes, ng)
  expect_equal(boot_mat, plain, tolerance = 1e-12)
})

test_that("cpp_aggregate_triMean_boot honors a real permutation", {
  .skip_if_omp_unsafe()
  boot_fn <- .cpp_sf("cpp_aggregate_triMean_boot")
  set.seed(14)
  ngenes <- 4L; nC <- 24L; ng <- 3L
  data <- matrix(stats::rnorm(ngenes * nC), ngenes, nC)
  group <- as.integer(c(rep(1L, 10), rep(2L, 8), rep(3L, 6)))
  perm <- cbind(seq_len(nC), sample(nC))            # 2 bootstraps
  boot <- boot_fn(data, group, ng, perm)

  trimean <- function(x) {
    qs <- stats::quantile(x, c(.25, .5, .75), names = FALSE, type = 7)
    (qs[1] + 2 * qs[2] + qs[3]) / 4
  }
  g_perm <- group[perm[, 2]]                        # groups under the permutation
  ref <- matrix(0, ngenes, ng)
  for (g in seq_len(ng)) for (i in seq_len(ngenes))
    ref[i, g] <- trimean(data[i, g_perm == g])
  boot_p2 <- matrix(boot[(ngenes * ng + 1):(2 * ngenes * ng)], ngenes, ng)
  expect_equal(boot_p2, ref, tolerance = 1e-12)
})

# ======================================================================
# cpp_unified_inner (cellchat bootstrap reject counts) — OMP-parallel;
# re-derive the Hill-function reject count in R for a 1-LR-pair case.
# ======================================================================

test_that("cpp_unified_inner counts Prob' > Pnull rejections per cluster pair", {
  .skip_if_omp_unsafe()
  fn <- .cpp_sf("cpp_unified_inner")
  set.seed(21)
  ngenes <- 3L; ngroups <- 2L; nboot <- 4L; Kh <- 0.5; n_hill <- 1
  # boot_tensor: flat ngenes x ngroups x nboot, gene varying fastest.
  boot_tensor <- runif(ngenes * ngroups * nboot, 0.1, 2.0)
  # One active LR pair: ligand = gene 0, receptor = gene 1 (0-based); no
  # co-receptors / agonists / antagonists (empty flat arrays + zero offsets).
  Lflat <- 0L;  Loff <- c(0L, 1L)
  Rflat <- 1L;  Roff <- c(0L, 1L)
  empty <- integer(0); zoff <- c(0L, 0L)
  nLR_active <- 1L
  Pnull_arr <- runif(ngroups * ngroups * nLR_active, 0.1, 0.5)

  nR <- fn(boot_tensor, ngenes, ngroups, nboot,
           Lflat, Loff, Rflat, Roff, empty, zoff, empty, zoff,
           empty, zoff, empty, zoff, Pnull_arr, nLR_active, Kh, n_hill)

  plane <- ngenes * ngroups
  ref <- numeric(ngroups * ngroups)
  for (nE in 0:(nboot - 1)) {
    bm <- matrix(boot_tensor[(nE * plane + 1):((nE + 1) * plane)], ngenes, ngroups)
    Lvec <- bm[1, ]; Rvec <- bm[2, ]
    for (c2 in seq_len(ngroups)) for (c1 in seq_len(ngroups)) {
      dataLR <- Lvec[c1] * Rvec[c2]; P1 <- dataLR / (Kh + dataLR)
      idx <- c1 + (c2 - 1) * ngroups
      if (P1 > Pnull_arr[idx]) ref[idx] <- ref[idx] + 1
    }
  }
  expect_length(nR, ngroups * ngroups * nLR_active)
  expect_equal(as.numeric(nR), ref)
})
