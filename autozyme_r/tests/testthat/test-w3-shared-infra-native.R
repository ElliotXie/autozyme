# Wave-3 coverage for the NATIVE-DISPATCH branches of R/shared_infra.R.
#
# The wave-1 file (test-unit-shared-infra.R) deliberately avoided the C++
# kernels because the OMP `#pragma`-parallel regions segfault at threads>1.
# This file covers the *default* (feature-flag ON) native branch of every
# dispatch wrapper, and proves the native kernel agrees with the pure-R /
# base-R fallback. It is SAFE only at OMP_NUM_THREADS=1 (the kernels run
# single-threaded there). Every native call is guarded by
# .skip_if_omp_unsafe() so running this file at threads>1 SKIPS rather than
# crashes.
#
# Native kernels exercised (all at OMP=1):
#   az_dgc_row_stats_cpp, az_dgc_group_summary_cpp, az_dense_group_summary_cpp,
#   turbo_scale_sparse_full, turbo_FastSparseRowVarStd, cpp_aggregate_triMean,
#   cpp_aggregate_triMean_boot, parallel_all_in_one_dgc.
#
# DUAL-MODE NAME RESOLUTION. The .az_* helpers are internal (unexported). To
# get coverage credit, covr::file_coverage() instruments a fresh source copy
# whose closures are reachable ONLY by bare name. But testthat::test_file()
# runs against the installed namespace, where bare internal names do not
# resolve. The shim below bridges both: under file_coverage the instrumented
# bare names already exist (shim is a no-op, instrumented copies are used and
# credited); under test_file they do not, so we bind them from the namespace.
# Measure coverage with:
#   covr::file_coverage("R/shared_infra.R",
#     c("tests/testthat/test-unit-shared-infra.R",
#       "tests/testthat/test-w3-shared-infra-native.R"),
#     parent_env = asNamespace("autozyme"))

if (!exists(".az_dgc_row_stats", inherits = TRUE)) {
  .az_ns_w3 <- asNamespace("autozyme")
  for (.nm_w3 in c(
      ".az_truthy", ".az_as_group",
      ".az_dgc_row_stats", ".az_dgc_row_stats_fallback",
      ".az_dgc_group_summary", ".az_dense_group_summary",
      ".az_dense_group_summary_fallback",
      ".az_dgc_scale_center", ".az_group_trimean", ".az_group_trimean_boot",
      ".az_dgc_row_var_standardized", ".az_dgc_grouped_wilcox",
      "az_dgc_row_stats_cpp", "az_dgc_group_summary_cpp",
      "az_dense_group_summary_cpp", "turbo_scale_sparse_full",
      "turbo_FastSparseRowVarStd", "cpp_aggregate_triMean",
      "cpp_aggregate_triMean_boot", "parallel_all_in_one_dgc")) {
    if (exists(.nm_w3, envir = .az_ns_w3, inherits = FALSE)) {
      assign(.nm_w3, get(.nm_w3, envir = .az_ns_w3), envir = environment())
    }
  }
  rm(.az_ns_w3, .nm_w3)
}

# The native OMP kernels are only segfault-safe single-threaded. Guard each
# native-branch test so the file is harmless if run at threads>1.
.skip_if_omp_unsafe <- function() {
  testthat::skip_if_not_installed("Matrix")
  if (!identical(Sys.getenv("OMP_NUM_THREADS"), "1")) {
    testthat::skip("native kernels are only segfault-safe at OMP_NUM_THREADS=1")
  }
}

# Set a feature flag for the duration of the calling test, restoring on exit.
# Also clears AUTOZYME_DISABLE so the global kill-switch never masks the flag.
.with_flag <- function(name, value, env = parent.frame()) {
  old <- Sys.getenv(name, unset = NA_character_)
  old_dis <- Sys.getenv("AUTOZYME_DISABLE", unset = NA_character_)
  withr::defer({
    if (is.na(old)) Sys.unsetenv(name) else do.call(Sys.setenv, setNames(list(old), name))
    if (is.na(old_dis)) Sys.unsetenv("AUTOZYME_DISABLE")
    else Sys.setenv(AUTOZYME_DISABLE = old_dis)
  }, envir = env)
  Sys.unsetenv("AUTOZYME_DISABLE")
  do.call(Sys.setenv, setNames(list(value), name))
}

.tiny_dgc <- function(dense) {
  methods::as(Matrix::Matrix(dense, sparse = TRUE), "dgCMatrix")
}

# ---- .az_dgc_row_stats native branch + parity ------------------------------

test_that(".az_dgc_row_stats native kernel == base-R rowwise stats", {
  .skip_if_omp_unsafe()
  .with_flag("AUTOZYME_SPARSE_KERNELS", "1")
  dense <- matrix(c(1, 0, 2, 3,
                    0, 0, 5, 4,
                    2, 1, 0, 0), nrow = 3, byrow = TRUE)
  rownames(dense) <- paste0("r", 1:3)
  sm <- .tiny_dgc(dense)
  rs <- .az_dgc_row_stats(sm, detect_threshold = 0)
  expect_equal(unname(rs$sum), unname(rowSums(dense)))
  expect_equal(unname(rs$mean), unname(rowMeans(dense)))
  expect_equal(unname(rs$variance), unname(apply(dense, 1L, stats::var)))
  expect_equal(unname(rs$nnz), unname(as.integer(rowSums(dense != 0))))
  expect_equal(unname(rs$detected), unname(as.integer(rowSums(dense > 0))))
  # native attaches rownames
  expect_identical(names(rs$sum), rownames(dense))
})

test_that(".az_dgc_row_stats native == forced fallback (parity)", {
  .skip_if_omp_unsafe()
  dense <- matrix(c(0.5, 2, 1.5, 3,
                    4, 0, 0, 1,
                    0, 0, 2, 2), nrow = 3, byrow = TRUE)
  sm <- .tiny_dgc(dense)
  .with_flag("AUTOZYME_SPARSE_KERNELS", "1")
  nat <- .az_dgc_row_stats(sm, detect_threshold = 1)
  .with_flag("AUTOZYME_SPARSE_KERNELS", "0")
  fb <- .az_dgc_row_stats(sm, detect_threshold = 1)
  expect_equal(unname(nat$sum), unname(fb$sum))
  expect_equal(unname(nat$mean), unname(fb$mean))
  expect_equal(unname(nat$variance), unname(fb$variance))
  expect_equal(unname(nat$nnz), unname(fb$nnz))
  expect_equal(unname(nat$detected), unname(fb$detected))
})

test_that(".az_dgc_row_stats native single-row / single-col edge cases", {
  .skip_if_omp_unsafe()
  .with_flag("AUTOZYME_SPARSE_KERNELS", "1")
  # single row
  d1 <- matrix(c(0, 3, 0, 5), nrow = 1)
  rs1 <- .az_dgc_row_stats(.tiny_dgc(d1))
  expect_equal(unname(rs1$sum), 8)
  expect_equal(unname(rs1$variance), stats::var(as.numeric(d1)))
  # single column: var is NA (n<=1) in the kernel
  d2 <- matrix(c(2, 0, 5), ncol = 1)
  rs2 <- .az_dgc_row_stats(.tiny_dgc(d2))
  expect_true(all(is.na(rs2$variance)))
  expect_equal(unname(rs2$sum), c(2, 0, 5))
})

test_that(".az_dgc_row_stats native all-zero matrix", {
  .skip_if_omp_unsafe()
  .with_flag("AUTOZYME_SPARSE_KERNELS", "1")
  z <- matrix(0, 3, 4)
  rs <- .az_dgc_row_stats(.tiny_dgc(z))
  expect_equal(unname(rs$sum), rep(0, 3))
  expect_equal(unname(rs$variance), rep(0, 3))
  expect_equal(unname(rs$nnz), rep(0L, 3))
  expect_equal(unname(rs$detected), rep(0L, 3))
})

# ---- .az_dgc_group_summary native branch + parity --------------------------

test_that(".az_dgc_group_summary native == base-R per-group reductions", {
  .skip_if_omp_unsafe()
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "1")
  dense <- matrix(c(1, 0, 2, 0, 4, 5,
                    3, 0, 0, 2, 0, 1,
                    0, 7, 0, 0, 2, 0), nrow = 3, byrow = TRUE)
  rownames(dense) <- paste0("r", 1:3)
  group <- factor(c("a", "b", "a", "b", "c", "c"), levels = c("a", "b", "c"))
  res <- .az_dgc_group_summary(.tiny_dgc(dense), group)
  base_sum <- sapply(levels(group), function(g) rowSums(dense[, group == g, drop = FALSE]))
  base_mean <- sapply(levels(group), function(g) rowMeans(dense[, group == g, drop = FALSE]))
  base_det <- sapply(levels(group), function(g) rowSums(dense[, group == g, drop = FALSE] > 0))
  expect_equal(unname(res$sum_by_group), unname(base_sum))
  expect_equal(unname(res$mean_by_group), unname(base_mean))
  expect_equal(unname(res$detected_by_group), unname(base_det))
  expect_equal(unname(res$group_size), as.integer(table(group)))
  # native attaches dimnames
  expect_identical(rownames(res$sum_by_group), rownames(dense))
  expect_identical(colnames(res$sum_by_group), levels(group))
})

test_that(".az_dgc_group_summary native == forced fallback (parity)", {
  .skip_if_omp_unsafe()
  dense <- matrix(c(1, 0, 2, 0,
                    3, 2, 0, 1,
                    0, 5, 0, 0), nrow = 3, byrow = TRUE)
  sm <- .tiny_dgc(dense)
  group <- factor(c("a", "b", "a", "b"))
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "1")
  nat <- .az_dgc_group_summary(sm, group)
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "0")
  fb <- .az_dgc_group_summary(sm, group)
  expect_equal(unname(nat$sum_by_group), unname(fb$sum_by_group))
  expect_equal(unname(nat$mean_by_group), unname(fb$mean_by_group))
  expect_equal(unname(nat$detected_by_group), unname(fb$detected_by_group))
})

test_that(".az_dgc_group_summary native single-group edge case", {
  .skip_if_omp_unsafe()
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "1")
  dense <- matrix(c(1, 2, 3, 0, 0, 4), nrow = 2, byrow = TRUE)
  group <- factor(rep("only", 3))
  res <- .az_dgc_group_summary(.tiny_dgc(dense), group)
  expect_equal(unname(res$sum_by_group), cbind(rowSums(dense)))
  expect_equal(unname(res$mean_by_group), cbind(rowMeans(dense)))
  expect_equal(unname(res$group_size), 3L)
})

# ---- .az_dense_group_summary native branch + parity ------------------------

test_that(".az_dense_group_summary native == forced fallback (parity)", {
  .skip_if_omp_unsafe()
  dense <- matrix(c(1, 0, 2, 0, 4, 5,
                    3, 0, 0, 2, 0, 1,
                    0, 7, 0, 0, 2, 0), nrow = 3, byrow = TRUE)
  rownames(dense) <- paste0("r", 1:3)
  group <- factor(c("a", "b", "a", "b", "c", "c"), levels = c("a", "b", "c"))
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "1")
  nat <- .az_dense_group_summary(dense, group)
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "0")
  fb <- .az_dense_group_summary(dense, group)
  expect_equal(nat$sum_by_group, fb$sum_by_group)
  expect_equal(nat$mean_by_group, fb$mean_by_group)
  expect_equal(nat$detected_by_group, fb$detected_by_group)
  expect_equal(nat$group_size, fb$group_size)
  # native attaches dimnames
  expect_identical(rownames(nat$sum_by_group), rownames(dense))
  expect_identical(colnames(nat$sum_by_group), levels(group))
})

test_that("az_dense_group_summary_cpp na.rm path matches base R", {
  .skip_if_omp_unsafe()
  # Exercise the na_rm=TRUE branch of the dense kernel directly (the wrapper
  # does not forward na.rm to the native call, so cover it at the kernel level).
  dense <- matrix(c(1, NA, 3,
                    NA, 5, 6), nrow = 2, byrow = TRUE)
  group <- c(1L, 1L, 1L)
  res_narm <- az_dense_group_summary_cpp(dense, group, 1L, 0, TRUE)
  # mean ignoring NA, per row
  expect_equal(res_narm$mean_by_group[1, 1], mean(c(1, 3)))
  expect_equal(res_narm$mean_by_group[2, 1], mean(c(5, 6)))
  # na_rm=FALSE -> NA propagates into the group with an NA
  res_keep <- az_dense_group_summary_cpp(dense, group, 1L, 0, FALSE)
  expect_true(is.na(res_keep$mean_by_group[1, 1]))
  expect_true(is.na(res_keep$mean_by_group[2, 1]))
})

# ---- .az_dgc_scale_center native branch + parity ---------------------------

test_that(".az_dgc_scale_center native == Seurat-style center/scale", {
  .skip_if_omp_unsafe()
  .with_flag("AUTOZYME_SPARSE_KERNELS", "1")
  dense <- matrix(c(2, 5, 1, 3,
                    0, 4, 0, 1,
                    6, 2, 0, 0), nrow = 3, byrow = TRUE)
  sm <- .tiny_dgc(dense)
  rows <- c(0L, 2L)
  scaled <- .az_dgc_scale_center(sm, rows, scale_max = 10)
  # Reference matching the kernel: cap only the POSITIVE tail (Seurat semantics)
  base <- dense[rows + 1L, , drop = FALSE]
  mu <- rowMeans(base)
  sd <- apply(base, 1L, stats::sd)
  sd[!is.finite(sd) | sd == 0] <- 1
  ref <- sweep(sweep(base, 1L, mu, "-"), 1L, sd, "/")
  ref[ref > 10] <- 10  # kernel caps only the upper tail
  expect_equal(unname(scaled), unname(ref))
  expect_equal(dim(scaled), c(2L, 4L))
})

test_that(".az_dgc_scale_center native upper-tail capping at small scale_max", {
  .skip_if_omp_unsafe()
  .with_flag("AUTOZYME_SPARSE_KERNELS", "1")
  set.seed(5)
  dense <- matrix(rpois(40, 2), nrow = 4) * 1.0
  sm <- .tiny_dgc(dense)
  scaled <- .az_dgc_scale_center(sm, 0:3, scale_max = 1.0)
  expect_true(all(scaled <= 1.0 + 1e-9))      # upper tail is capped
  expect_equal(dim(scaled), c(4L, ncol(dense)))
})

test_that(".az_dgc_scale_center native single-row selection", {
  .skip_if_omp_unsafe()
  .with_flag("AUTOZYME_SPARSE_KERNELS", "1")
  dense <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2, byrow = TRUE)
  sm <- .tiny_dgc(dense)
  scaled <- .az_dgc_scale_center(sm, 1L, scale_max = 10)  # second row only
  v <- dense[2, ]
  ref <- (v - mean(v)) / stats::sd(v)
  ref[ref > 10] <- 10
  expect_equal(as.numeric(scaled), as.numeric(ref))
  expect_equal(nrow(scaled), 1L)
})

# ---- .az_dgc_row_var_standardized native branch + parity -------------------

test_that(".az_dgc_row_var_standardized native == forced fallback (parity)", {
  .skip_if_omp_unsafe()
  set.seed(7)
  dense <- matrix(rpois(36, 1), nrow = 4) * 1.0
  sm <- .tiny_dgc(dense)
  mu <- rowMeans(dense)
  sd <- apply(dense, 1L, stats::sd)
  vmax <- 3
  nnz <- as.integer(rowSums(dense != 0))
  .with_flag("AUTOZYME_SPARSE_KERNELS", "1")
  nat <- .az_dgc_row_var_standardized(sm, mu, sd, vmax, nnz)
  .with_flag("AUTOZYME_SPARSE_KERNELS", "0")
  fb <- .az_dgc_row_var_standardized(sm, mu, sd, vmax, nnz)
  expect_equal(nat, fb)
})

test_that(".az_dgc_row_var_standardized native zero-sd row -> 0", {
  .skip_if_omp_unsafe()
  .with_flag("AUTOZYME_SPARSE_KERNELS", "1")
  dense <- matrix(c(2, 2, 2, 2,      # constant row -> sd 0
                    0, 1, 0, 3),     # varying row
                  nrow = 2, byrow = TRUE)
  sm <- .tiny_dgc(dense)
  mu <- rowMeans(dense)
  sd <- apply(dense, 1L, stats::sd)
  res <- .az_dgc_row_var_standardized(sm, mu, sd, vmax = 10, nnz = rowSums(dense != 0))
  expect_equal(res[1], 0)               # zero-sd row clamps to 0
  expect_true(is.finite(res[2]))
})

# ---- .az_group_trimean native branch + parity ------------------------------

test_that(".az_group_trimean native == type-7 trimean recipe", {
  .skip_if_omp_unsafe()
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "1")
  set.seed(11)
  x <- matrix(rnorm(24), nrow = 4)
  rownames(x) <- paste0("g", 1:4)
  group <- factor(rep(c("a", "b", "c"), each = 2), levels = c("a", "b", "c"))
  tri <- function(v) mean(stats::quantile(v, c(0.25, 0.5, 0.5, 0.75), names = FALSE))
  expected <- sapply(levels(group),
                     function(g) apply(x[, group == g, drop = FALSE], 1, tri))
  got <- .az_group_trimean(x, group)
  expect_equal(unname(got), unname(expected), tolerance = 1e-9)
  expect_identical(rownames(got), rownames(x))   # native attaches dimnames
  expect_identical(colnames(got), levels(group))
})

test_that(".az_group_trimean native == forced fallback (parity)", {
  .skip_if_omp_unsafe()
  set.seed(13)
  x <- matrix(rnorm(30), nrow = 5)
  group <- factor(rep(c("p", "q", "r"), each = 2)[1:6], levels = c("p", "q", "r"))
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "1")
  nat <- .az_group_trimean(x, group)
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "0")
  fb <- .az_group_trimean(x, group)
  expect_equal(unname(nat), unname(fb), tolerance = 1e-9)
})

test_that(".az_group_trimean native single group", {
  .skip_if_omp_unsafe()
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "1")
  x <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8), nrow = 2, byrow = TRUE)
  group <- factor(rep("only", 4))
  got <- .az_group_trimean(x, group)
  tri <- function(v) mean(stats::quantile(v, c(0.25, 0.5, 0.5, 0.75), names = FALSE))
  expect_equal(unname(got[1, 1]), tri(x[1, ]), tolerance = 1e-9)
  expect_equal(unname(got[2, 1]), tri(x[2, ]), tolerance = 1e-9)
})

# ---- .az_group_trimean_boot native branch + parity -------------------------

test_that(".az_group_trimean_boot native == forced fallback (parity)", {
  .skip_if_omp_unsafe()
  set.seed(17)
  data <- matrix(rnorm(36), nrow = 4)
  rownames(data) <- paste0("g", 1:4)
  group <- factor(rep(c("a", "b", "c"), 3), levels = c("a", "b", "c"))
  perm <- replicate(4, sample(ncol(data)))
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "1")
  nat <- .az_group_trimean_boot(data, group, perm)
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "0")
  fb <- .az_group_trimean_boot(data, group, perm)
  # NOTE: the native branch returns the raw flat NumericVector from the kernel
  # (dim NULL); the R fallback wraps it into an (ngene, ngroup, nboot) array.
  # Values agree; we compare dim-agnostically and document the shape gap.
  expect_equal(as.numeric(nat), as.numeric(fb), tolerance = 1e-9)
  expect_equal(length(as.numeric(nat)), nrow(data) * nlevels(group) * ncol(perm))
})

# ---- .az_dgc_grouped_wilcox native branch + reference ----------------------

test_that(".az_dgc_grouped_wilcox native sum / detected == base R (exact)", {
  .skip_if_omp_unsafe()
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "1")
  set.seed(3)
  P <- 5; N <- 8
  dense <- matrix(rpois(P * N, 1.2), nrow = P) * 1.0
  dense[dense > 0] <- log1p(dense[dense > 0])      # log-normalized-ish values
  sm <- .tiny_dgc(dense)
  grp <- c(1L, 1L, 1L, 2L, 2L, 3L, 3L, 3L)
  gsz <- as.integer(table(grp))
  res <- .az_dgc_grouped_wilcox(sm, grp, gsz)
  expect_identical(names(res),
                   c("pval_by_group", "sum_by_group", "detected_by_group"))
  expect_equal(dim(res$pval_by_group), c(P, length(gsz)))
  # sum_by_group = sum of expm1(value) per group; detected = nonzero count
  ref_sum <- matrix(0, P, 3); ref_det <- matrix(0, P, 3)
  for (f in seq_len(P)) for (g in 1:3) {
    ing <- grp == g
    ref_sum[f, g] <- sum(expm1(dense[f, ing]))
    ref_det[f, g] <- sum(dense[f, ing] != 0)
  }
  expect_equal(res$sum_by_group, ref_sum)
  expect_equal(res$detected_by_group, ref_det)
})

test_that(".az_dgc_grouped_wilcox native pval is a valid tie-corrected approx", {
  .skip_if_omp_unsafe()
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "1")
  set.seed(3)
  P <- 5; N <- 8
  dense <- matrix(rpois(P * N, 1.2), nrow = P) * 1.0
  dense[dense > 0] <- log1p(dense[dense > 0])
  sm <- .tiny_dgc(dense)
  grp <- c(1L, 1L, 1L, 2L, 2L, 3L, 3L, 3L)
  gsz <- as.integer(table(grp))
  res <- .az_dgc_grouped_wilcox(sm, grp, gsz)
  # all p-values are proper probabilities
  expect_true(all(res$pval_by_group >= 0 & res$pval_by_group <= 1))
  # they track a base-R tie-corrected normal-approximation (loose tolerance:
  # the kernel's exact tie/continuity bookkeeping differs slightly, see report)
  x1 <- N^3 - N; x2 <- 1 / (12 * (N^2 - N))
  ref_p <- matrix(NA_real_, P, 3)
  for (f in seq_len(P)) {
    v <- dense[f, ]; r <- rank(v, ties.method = "average")
    tie_sum <- sum(vapply(as.integer(table(v)), function(t) t^3 - t, numeric(1)))
    rhs <- (x1 - tie_sum) * x2
    for (g in 1:3) {
      n1 <- gsz[g]; n2 <- N - n1
      u <- sum(r[grp == g]) - n1 * (n1 + 1) / 2
      z <- u - 0.5 * n1 * n2
      z <- if (z > 0) z - 0.5 else if (z < 0) z + 0.5 else z
      ref_p[f, g] <- 2 * stats::pnorm(-abs(z / sqrt(n1 * n2 * rhs)))
    }
  }
  expect_equal(res$pval_by_group, ref_p, tolerance = 0.05)
})

test_that(".az_dgc_grouped_wilcox native single feature / two groups", {
  .skip_if_omp_unsafe()
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "1")
  dense <- matrix(c(0, 1.5, 0, 2.0, 3.0, 0), nrow = 1)
  sm <- .tiny_dgc(dense)
  grp <- c(1L, 1L, 1L, 2L, 2L, 2L)
  gsz <- as.integer(table(grp))
  res <- .az_dgc_grouped_wilcox(sm, grp, gsz)
  expect_equal(dim(res$pval_by_group), c(1L, 2L))
  expect_equal(res$sum_by_group[1, 1], sum(expm1(dense[1, 1:3])))
  expect_equal(res$sum_by_group[1, 2], sum(expm1(dense[1, 4:6])))
  expect_true(all(res$pval_by_group >= 0 & res$pval_by_group <= 1))
})

# These bare-name fallback-body tests are pure-R (feature OFF) and run at any
# thread count; they fill the instrumented-coverage gap for the same wrapper
# bodies wave-1 exercises only via the (uninstrumented) installed namespace.

test_that(".az_dgc_scale_center fallback body (feature off) two-sided cap", {
  testthat::skip_if_not_installed("Matrix")
  .with_flag("AUTOZYME_SPARSE_KERNELS", "0")
  dense <- matrix(c(2, 5, 1, 3,
                    0, 4, 0, 1,
                    6, 2, 0, 0), nrow = 3, byrow = TRUE)
  sm <- .tiny_dgc(dense)
  scaled <- .az_dgc_scale_center(sm, c(0L, 2L), scale_max = 0.5)
  # fallback caps BOTH tails (Scanpy-style), unlike the native upper-only cap
  expect_true(all(scaled <= 0.5 + 1e-9))
  expect_true(all(scaled >= -0.5 - 1e-9))
})

test_that(".az_dgc_row_var_standardized fallback body (feature off)", {
  testthat::skip_if_not_installed("Matrix")
  .with_flag("AUTOZYME_SPARSE_KERNELS", "0")
  dense <- matrix(c(2, 2, 2, 2,
                    0, 1, 0, 3), nrow = 2, byrow = TRUE)
  sm <- .tiny_dgc(dense)
  mu <- rowMeans(dense)
  sd <- apply(dense, 1L, stats::sd)
  res <- .az_dgc_row_var_standardized(sm, mu, sd, vmax = 10,
                                      nnz_per_row = rowSums(dense != 0))
  expect_equal(res[1], 0)               # zero-sd row -> 0 in the fallback
  expect_true(is.finite(res[2]))
})

test_that(".az_group_trimean fallback body (feature off) uses aggregate path", {
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "0")
  set.seed(21)
  x <- matrix(rnorm(18), nrow = 3)
  rownames(x) <- paste0("g", 1:3)
  group <- factor(rep(c("a", "b", "c"), each = 2), levels = c("a", "b", "c"))
  got <- .az_group_trimean(x, group)
  tri <- function(v) mean(stats::quantile(v, c(0.25, 0.5, 0.5, 0.75), names = FALSE))
  expected <- sapply(levels(group),
                     function(g) apply(x[, group == g, drop = FALSE], 1, tri))
  expect_equal(unname(got), unname(expected), tolerance = 1e-9)
  expect_identical(colnames(got), levels(group))
})

test_that(".az_group_trimean_boot fallback body (feature off) loops fallback", {
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "0")
  set.seed(23)
  data <- matrix(rnorm(18), nrow = 3)
  group <- factor(rep(c("a", "b", "c"), 2), levels = c("a", "b", "c"))
  perm <- replicate(3, sample(ncol(data)))
  arr <- .az_group_trimean_boot(data, group, perm)
  expect_equal(dim(arr), c(nrow(data), nlevels(group), ncol(perm)))
  # first slab equals a single-permutation trimean
  slab1 <- .az_group_trimean(data, group[perm[, 1]])
  expect_equal(unname(arr[, , 1]), unname(slab1), tolerance = 1e-9)
})

test_that(".az_dense_group_summary_fallback na.rm branch ignores NA", {
  dense <- matrix(c(1, NA, 3, 0,
                    2,  4, NA, 1), nrow = 2, byrow = TRUE)
  group <- factor(c("a", "a", "b", "b"), levels = c("a", "b"))
  res <- .az_dense_group_summary_fallback(dense, group, detect_threshold = 0,
                                          na.rm = TRUE)
  # group "a" cols 1:2: row1 -> only col1 (col2 NA) summed = 1; row2 -> 2+4 = 6
  expect_equal(unname(res$sum_by_group[1, "a"]), 1)
  expect_equal(unname(res$sum_by_group[2, "a"]), 6)
  # group "b" cols 3:4: row1 -> 3+0 = 3; row2 -> only col4 (col3 NA) = 1
  expect_equal(unname(res$sum_by_group[1, "b"]), 3)
  expect_equal(unname(res$sum_by_group[2, "b"]), 1)
})

test_that(".az_dense_group_summary_fallback anyNA (na.rm=FALSE) keeps NA", {
  dense <- matrix(c(1, NA, 3, 0,
                    2,  4, 5,  1), nrow = 2, byrow = TRUE)
  group <- factor(c("a", "a", "b", "b"), levels = c("a", "b"))
  res <- .az_dense_group_summary_fallback(dense, group, na.rm = FALSE)
  # row1 group a has an NA in col2 -> propagates
  expect_true(is.na(res$sum_by_group[1, "a"]))
  # row2 group a has no NA -> normal sum 2+4 = 6
  expect_equal(unname(res$sum_by_group[2, "a"]), 6)
})

test_that(".az_dense_group_summary_fallback skips out-of-range group codes", {
  # Pass a pre-built group list with a code pointing past ngroups -> the
  # `gj > ngroups` guard (line 87) is exercised and that column is skipped.
  g <- list(codes = c(1L, 2L, 5L), levels = c("a", "b"), ngroups = 2L)
  dense <- matrix(c(1, 2, 9,
                    3, 4, 9), nrow = 2, byrow = TRUE)
  res <- .az_dense_group_summary_fallback(dense, g)
  # the third column (code 5, out of range) is dropped from both groups
  expect_equal(res$sum_by_group[, "a"], c(1, 3), ignore_attr = TRUE)
  expect_equal(res$sum_by_group[, "b"], c(2, 4), ignore_attr = TRUE)
})

# ---- .az_dgc_grouped_wilcox disabled / error branches (no native) ----------

test_that(".az_dgc_grouped_wilcox returns NULL when grouped reductions OFF", {
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "0")
  expect_null(.az_dgc_grouped_wilcox(NULL, integer(0), integer(0), fallback = TRUE))
})

test_that(".az_dgc_grouped_wilcox errors when OFF + fallback=FALSE", {
  .with_flag("AUTOZYME_GROUPED_REDUCTIONS", "0")
  expect_error(
    .az_dgc_grouped_wilcox(NULL, integer(0), integer(0), fallback = FALSE),
    "disabled"
  )
})
