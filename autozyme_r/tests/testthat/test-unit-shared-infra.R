# Unit tests for the PURE-R surface of R/shared_infra.R.
#
# SAFETY: the dispatch wrappers (.az_dgc_row_stats, .az_dgc_group_summary,
# .az_dgc_scale_center, .az_group_trimean, .az_dgc_row_var_standardized,
# .az_group_trimean_boot, .az_dgc_grouped_wilcox, .az_dense_group_summary)
# call native C++ kernels by default. On this Mac that native path SEGFAULTS
# (the existing tests/testthat/test-shared-infra.R hits it). So:
#   * We test the truthy / feature-gate / group-coercion helpers directly.
#   * We test the pure-R *_fallback functions directly (no dispatch).
#   * We exercise the dispatch wrappers ONLY with the relevant feature flag
#     set to "0", which routes every wrapper to its pure-R fallback and never
#     reaches the segfaulting native kernel.
#
# AVOIDED (native-segfault-unsafe; covered only via forced fallback, never the
# default native branch): az_dgc_row_stats_cpp, az_dgc_group_summary_cpp,
# az_dense_group_summary_cpp, turbo_scale_sparse_full, turbo_FastSparseRowVarStd,
# cpp_aggregate_triMean(_boot), parallel_all_in_one_dgc. The
# .az_dgc_grouped_wilcox helper has NO pure-R fallback (it always tries the
# native kernel), so we cover only its feature-disabled early return.

# ---- .az_truthy ------------------------------------------------------------

test_that(".az_truthy parses truthy / falsy strings case-insensitively", {
  ns <- asNamespace("autozyme")
  for (s in c("1", "true", "TRUE", "Yes", "on", " On ")) {
    expect_true(ns$.az_truthy(s))
  }
  for (s in c("0", "false", "FALSE", "No", "off", " off ")) {
    expect_false(ns$.az_truthy(s))
  }
})

test_that(".az_truthy returns NA for unrecognized / empty input", {
  ns <- asNamespace("autozyme")
  expect_true(is.na(ns$.az_truthy("maybe")))
  expect_true(is.na(ns$.az_truthy("")))
  expect_true(is.na(ns$.az_truthy(character(0))))
  expect_true(is.na(ns$.az_truthy(NA_character_)))
})

test_that(".az_truthy passes through scalar logicals", {
  ns <- asNamespace("autozyme")
  expect_true(ns$.az_truthy(TRUE))
  expect_false(ns$.az_truthy(FALSE))
  expect_true(is.na(ns$.az_truthy(NA)))
})

# ---- .az_feature_key / .az_option_values -----------------------------------

test_that(".az_feature_key uppercases and underscores non-alnum runs", {
  ns <- asNamespace("autozyme")
  expect_identical(ns$.az_feature_key("dynamic blas"), "DYNAMIC_BLAS")
  expect_identical(ns$.az_feature_key("sparse-kernels.v2"), "SPARSE_KERNELS_V2")
})

test_that(".az_option_values reads both underscore and dot option spellings", {
  ns <- asNamespace("autozyme")
  old <- options(autozyme.my_feature = TRUE, autozyme.my.feature = FALSE)
  on.exit(options(old), add = TRUE)
  vals <- ns$.az_option_values("my_feature")
  expect_true(TRUE %in% vals)
  expect_true(FALSE %in% vals)
})

# ---- .az_first_truthy ------------------------------------------------------

test_that(".az_first_truthy returns the first parseable value", {
  ns <- asNamespace("autozyme")
  expect_true(ns$.az_first_truthy(c(NA, "garbage", "1", "0")))
  expect_false(ns$.az_first_truthy(c("nope", "off", "on")))
  expect_true(is.na(ns$.az_first_truthy(c("a", "b", NA))))
})

# ---- .az_global_disabled / .az_feature_enabled -----------------------------

test_that(".az_global_disabled reflects AUTOZYME_DISABLE/DISABLED", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv(c("AUTOZYME_DISABLE", "AUTOZYME_DISABLED"), unset = NA_character_)
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
  }, add = TRUE)
  Sys.unsetenv(c("AUTOZYME_DISABLE", "AUTOZYME_DISABLED"))
  expect_false(ns$.az_global_disabled())
  Sys.setenv(AUTOZYME_DISABLE = "1")
  expect_true(ns$.az_global_disabled())
  Sys.setenv(AUTOZYME_DISABLE = "0")
  Sys.setenv(AUTOZYME_DISABLED = "yes")
  expect_true(ns$.az_global_disabled())
})

test_that(".az_feature_enabled honors default when nothing is set", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv(c("AUTOZYME_FOO", "AUTOZYME_DISABLE"), unset = NA_character_)
  old_opt <- options(autozyme.foo = NULL)
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
    options(old_opt)
  }, add = TRUE)
  Sys.unsetenv(c("AUTOZYME_FOO", "AUTOZYME_DISABLE"))
  expect_true(ns$.az_feature_enabled("foo", default = TRUE))
  expect_false(ns$.az_feature_enabled("foo", default = FALSE))
})

test_that(".az_feature_enabled: global env wins over default", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv(c("AUTOZYME_FOO", "AUTOZYME_DISABLE"), unset = NA_character_)
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
  }, add = TRUE)
  Sys.unsetenv("AUTOZYME_DISABLE")
  Sys.setenv(AUTOZYME_FOO = "0")
  expect_false(ns$.az_feature_enabled("foo", default = TRUE))
})

test_that(".az_feature_enabled: per-patch override beats global enable", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv(c("AUTOZYME_FOO", "AUTOZYME_MYPATCH_FOO", "AUTOZYME_DISABLE"),
                    unset = NA_character_)
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
  }, add = TRUE)
  Sys.unsetenv("AUTOZYME_DISABLE")
  Sys.setenv(AUTOZYME_FOO = "1", AUTOZYME_MYPATCH_FOO = "0")
  expect_false(ns$.az_feature_enabled("foo", patch = "mypatch", default = TRUE))
})

test_that(".az_feature_enabled is FALSE whenever globally disabled", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv(c("AUTOZYME_FOO", "AUTOZYME_DISABLE"), unset = NA_character_)
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
  }, add = TRUE)
  Sys.setenv(AUTOZYME_FOO = "1", AUTOZYME_DISABLE = "1")
  expect_false(ns$.az_feature_enabled("foo", default = TRUE))
})

# ---- .az_as_group ----------------------------------------------------------

test_that(".az_as_group handles factors, integer+ngroups, and bare vectors", {
  ns <- asNamespace("autozyme")
  g1 <- ns$.az_as_group(factor(c("b", "a", "b"), levels = c("a", "b")))
  expect_identical(g1$levels, c("a", "b"))
  expect_identical(g1$codes, c(2L, 1L, 2L))
  expect_identical(g1$ngroups, 2L)

  g2 <- ns$.az_as_group(c(1L, 3L, 2L), ngroups = 3L)
  expect_identical(g2$ngroups, 3L)
  expect_identical(g2$levels, as.character(1:3))

  g3 <- ns$.az_as_group(c("x", "y", "x"))
  expect_identical(g3$ngroups, 2L)
})

# ---- pure-R fallbacks (no dispatch, no native) -----------------------------

test_that(".az_dgc_row_stats_fallback matches base R rowwise stats", {
  ns <- asNamespace("autozyme")
  m <- matrix(c(1, 0, 2, 3,
                0, 0, 5, 4,
                2, 1, 0, 0), nrow = 3, byrow = TRUE)
  rs <- ns$.az_dgc_row_stats_fallback(m)
  expect_equal(rs$sum, rowSums(m))
  expect_equal(rs$mean, rowMeans(m))
  expect_equal(rs$variance, apply(m, 1L, stats::var))
  expect_equal(rs$nnz, rowSums(m != 0))
  expect_equal(rs$detected, rowSums(m > 0))
})

test_that(".az_dgc_row_stats_fallback honors a non-zero detect_threshold", {
  ns <- asNamespace("autozyme")
  m <- matrix(c(0.5, 2, 1.5, 3), nrow = 2)
  rs <- ns$.az_dgc_row_stats_fallback(m, detect_threshold = 1)
  expect_equal(rs$detected, rowSums(m > 1))
})

test_that(".az_dense_group_summary_fallback matches per-group base R reductions", {
  ns <- asNamespace("autozyme")
  dense <- matrix(c(1, 0, 2, 0, 4, 5,
                    3, 0, 0, 2, 0, 1,
                    0, 7, 0, 0, 2, 0), nrow = 3, byrow = TRUE)
  rownames(dense) <- paste0("g", 1:3)
  group <- factor(c("a", "b", "a", "b", "c", "c"), levels = c("a", "b", "c"))
  res <- ns$.az_dense_group_summary_fallback(dense, group)

  base_sum <- sapply(levels(group), function(g) rowSums(dense[, group == g, drop = FALSE]))
  base_mean <- sapply(levels(group), function(g) rowMeans(dense[, group == g, drop = FALSE]))
  base_detect <- sapply(levels(group), function(g) rowSums(dense[, group == g, drop = FALSE] > 0))
  expect_equal(res$sum_by_group, base_sum)
  expect_equal(res$mean_by_group, base_mean)
  expect_equal(res$detected_by_group, base_detect)
  expect_equal(unname(res$group_size), as.integer(table(group)))
})

# ---- dispatch wrappers routed through the SAFE fallback (feature disabled) --

test_that(".az_dgc_row_stats(feature off) routes to fallback, matches base R", {
  skip_if_not_installed("Matrix")
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("AUTOZYME_SPARSE_KERNELS", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_SPARSE_KERNELS")
    else Sys.setenv(AUTOZYME_SPARSE_KERNELS = old)
  }, add = TRUE)
  Sys.setenv(AUTOZYME_SPARSE_KERNELS = "0")  # force fallback; never call native

  dense <- matrix(c(1, 0, 2, 3, 0, 0, 5, 4, 2, 1, 0, 0), nrow = 3, byrow = TRUE)
  sm <- methods::as(Matrix::Matrix(dense, sparse = TRUE), "dgCMatrix")
  rs <- ns$.az_dgc_row_stats(sm)
  expect_equal(unname(rs$sum), rowSums(dense))
  expect_equal(unname(rs$mean), rowMeans(dense))
})

test_that(".az_dgc_group_summary(feature off) matches base R per group", {
  skip_if_not_installed("Matrix")
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("AUTOZYME_GROUPED_REDUCTIONS", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_GROUPED_REDUCTIONS")
    else Sys.setenv(AUTOZYME_GROUPED_REDUCTIONS = old)
  }, add = TRUE)
  Sys.setenv(AUTOZYME_GROUPED_REDUCTIONS = "0")

  dense <- matrix(c(1, 0, 2, 0, 3, 2, 0, 1, 0, 5, 0, 0), nrow = 3, byrow = TRUE)
  sm <- methods::as(Matrix::Matrix(dense, sparse = TRUE), "dgCMatrix")
  group <- factor(c("a", "b", "a", "b"))
  res <- ns$.az_dgc_group_summary(sm, group)
  base_sum <- sapply(levels(group),
                     function(g) rowSums(dense[, group == g, drop = FALSE]))
  expect_equal(unname(res$sum_by_group), unname(base_sum))
})

test_that(".az_dgc_scale_center(feature off) matches base R center/scale", {
  skip_if_not_installed("Matrix")
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("AUTOZYME_SPARSE_KERNELS", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_SPARSE_KERNELS")
    else Sys.setenv(AUTOZYME_SPARSE_KERNELS = old)
  }, add = TRUE)
  Sys.setenv(AUTOZYME_SPARSE_KERNELS = "0")

  dense <- matrix(c(2, 5, 1, 3, 0, 4, 0, 1, 6, 2, 0, 0), nrow = 3, byrow = TRUE)
  sm <- methods::as(Matrix::Matrix(dense, sparse = TRUE), "dgCMatrix")
  rows <- c(0L, 2L)
  scaled <- ns$.az_dgc_scale_center(sm, rows, scale_max = 10)

  base <- dense[rows + 1L, , drop = FALSE]
  mu <- rowMeans(base)
  sd <- apply(base, 1L, stats::sd)
  sd[!is.finite(sd) | sd == 0] <- 1
  base <- sweep(sweep(base, 1L, mu, "-"), 1L, sd, "/")
  base[base > 10] <- 10; base[base < -10] <- -10
  expect_equal(unname(scaled), unname(base))
})

test_that(".az_group_trimean(feature off) matches the type-7 quantile recipe", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("AUTOZYME_GROUPED_REDUCTIONS", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_GROUPED_REDUCTIONS")
    else Sys.setenv(AUTOZYME_GROUPED_REDUCTIONS = old)
  }, add = TRUE)
  Sys.setenv(AUTOZYME_GROUPED_REDUCTIONS = "0")

  set.seed(11)
  x <- matrix(rnorm(24), nrow = 4)
  rownames(x) <- paste0("g", 1:4)
  group <- factor(rep(c("a", "b", "c"), each = 2), levels = c("a", "b", "c"))
  tri <- function(v) mean(stats::quantile(v, c(0.25, 0.5, 0.5, 0.75), names = FALSE))
  expected <- sapply(levels(group),
                     function(g) apply(x[, group == g, drop = FALSE], 1, tri))
  got <- ns$.az_group_trimean(x, group)
  expect_equal(unname(got), unname(expected))
})

test_that(".az_dgc_grouped_wilcox returns NULL when grouped reductions disabled", {
  ns <- asNamespace("autozyme")
  # No pure-R fallback exists; with the feature off + fallback=TRUE it returns
  # NULL WITHOUT calling the native kernel (so this never segfaults).
  old <- Sys.getenv("AUTOZYME_GROUPED_REDUCTIONS", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_GROUPED_REDUCTIONS")
    else Sys.setenv(AUTOZYME_GROUPED_REDUCTIONS = old)
  }, add = TRUE)
  Sys.setenv(AUTOZYME_GROUPED_REDUCTIONS = "0")
  expect_null(ns$.az_dgc_grouped_wilcox(NULL, integer(0), integer(0),
                                        fallback = TRUE))
})

test_that(".az_dgc_grouped_wilcox errors (no native) when disabled + fallback=FALSE", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("AUTOZYME_GROUPED_REDUCTIONS", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_GROUPED_REDUCTIONS")
    else Sys.setenv(AUTOZYME_GROUPED_REDUCTIONS = old)
  }, add = TRUE)
  Sys.setenv(AUTOZYME_GROUPED_REDUCTIONS = "0")
  expect_error(
    ns$.az_dgc_grouped_wilcox(NULL, integer(0), integer(0), fallback = FALSE),
    "disabled"
  )
})
