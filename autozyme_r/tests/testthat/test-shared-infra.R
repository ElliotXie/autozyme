test_that("feature gates honor env, options, and global disable", {
  ns <- asNamespace("autozyme")
  old_env <- Sys.getenv(c("AUTOZYME_TEST_FEATURE", "AUTOZYME_TEST_FEATURE_DISABLE",
                          "AUTOZYME_DISABLE"), unset = NA_character_)
  old_opt <- getOption("autozyme.test_feature", NULL)
  on.exit({
    for (nm in names(old_env)) {
      if (is.na(old_env[[nm]])) {
        Sys.unsetenv(nm)
      } else {
        do.call(Sys.setenv, setNames(list(old_env[[nm]]), nm))
      }
    }
    options(autozyme.test_feature = old_opt)
  }, add = TRUE)

  Sys.unsetenv(c("AUTOZYME_TEST_FEATURE", "AUTOZYME_DISABLE"))
  options(autozyme.test_feature = NULL)
  expect_true(ns$.az_feature_enabled("test_feature", default = TRUE))
  expect_false(ns$.az_feature_enabled("test_feature", default = FALSE))

  Sys.setenv(AUTOZYME_TEST_FEATURE = "0")
  expect_false(ns$.az_feature_enabled("test_feature", default = TRUE))
  Sys.setenv(AUTOZYME_TEST_FEATURE = "1")
  expect_true(ns$.az_feature_enabled("test_feature", default = FALSE))

  Sys.setenv(AUTOZYME_DISABLE = "1")
  expect_false(ns$.az_feature_enabled("test_feature", default = TRUE))
})

test_that("thread scope restores thread environment", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("OMP_NUM_THREADS", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("OMP_NUM_THREADS")
    else Sys.setenv(OMP_NUM_THREADS = old)
  }, add = TRUE)

  Sys.setenv(OMP_NUM_THREADS = "7")
  ns$.az_thread_scope(1L, {
    expect_identical(Sys.getenv("OMP_NUM_THREADS"), "1")
  })
  expect_identical(Sys.getenv("OMP_NUM_THREADS"), "7")
})

test_that("sparse row stats and scaling match dense base R", {
  skip_if_not_installed("Matrix")
  ns <- asNamespace("autozyme")
  m <- Matrix::sparseMatrix(
    i = c(1, 3, 1, 2, 4),
    j = c(1, 1, 3, 4, 4),
    x = c(2, 5, 1, 3, 4),
    dims = c(4, 5)
  )
  dense <- as.matrix(m)

  stats <- ns$.az_dgc_row_stats(m)
  expect_equal(stats$sum, rowSums(dense), tolerance = 1e-12)
  expect_equal(stats$mean, rowMeans(dense), tolerance = 1e-12)
  expect_equal(stats$variance, apply(dense, 1, var), tolerance = 1e-12)
  expect_equal(as.integer(stats$nnz), as.integer(rowSums(dense != 0)))
  expect_equal(as.integer(stats$detected), as.integer(rowSums(dense > 0)))

  rows <- c(0L, 2L)
  scaled <- ns$.az_dgc_scale_center(m, rows, scale_max = 10)
  base <- dense[rows + 1L, , drop = FALSE]
  base_mu <- rowMeans(base)
  base_sd <- apply(base, 1L, stats::sd)
  base_sd[!is.finite(base_sd) | base_sd == 0] <- 1
  base <- sweep(sweep(base, 1L, base_mu, "-"), 1L, base_sd, "/")
  expect_equal(unname(scaled), unname(base), tolerance = 1e-12)
})

test_that("grouped reductions match base R for dense and sparse matrices", {
  skip_if_not_installed("Matrix")
  ns <- asNamespace("autozyme")
  dense <- matrix(c(
    1, 0, 2, 0, 4, 5,
    3, 0, 0, 2, 0, 1,
    0, 7, 0, 0, 2, 0
  ), nrow = 3, byrow = TRUE)
  colnames(dense) <- paste0("c", seq_len(ncol(dense)))
  rownames(dense) <- paste0("g", seq_len(nrow(dense)))
  group <- factor(c("a", "b", "a", "b", "c", "c"), levels = c("a", "b", "c"))
  sparse <- methods::as(Matrix::Matrix(dense, sparse = TRUE), "dgCMatrix")

  base_sum <- sapply(levels(group), function(g) rowSums(dense[, group == g, drop = FALSE]))
  base_mean <- sapply(levels(group), function(g) rowMeans(dense[, group == g, drop = FALSE]))
  base_detect <- sapply(levels(group), function(g) rowSums(dense[, group == g, drop = FALSE] > 0))

  dres <- ns$.az_dense_group_summary(dense, group)
  sres <- ns$.az_dgc_group_summary(sparse, group)
  expect_equal(dres$sum_by_group, base_sum, tolerance = 1e-12)
  expect_equal(dres$mean_by_group, base_mean, tolerance = 1e-12)
  expect_equal(dres$detected_by_group, base_detect, tolerance = 1e-12)
  expect_equal(sres$sum_by_group, base_sum, tolerance = 1e-12)
  expect_equal(sres$mean_by_group, base_mean, tolerance = 1e-12)
  expect_equal(sres$detected_by_group, base_detect, tolerance = 1e-12)
})

test_that("group triMean helper matches type-7 quantile fallback", {
  ns <- asNamespace("autozyme")
  set.seed(10)
  x <- matrix(rnorm(24), nrow = 4)
  rownames(x) <- paste0("g", seq_len(nrow(x)))
  group <- factor(rep(c("a", "b", "c"), each = 2), levels = c("a", "b", "c"))
  tri <- function(v) mean(stats::quantile(v, c(0.25, 0.5, 0.5, 0.75),
                                          names = FALSE))
  expected <- sapply(levels(group), function(g) {
    apply(x[, group == g, drop = FALSE], 1, tri)
  })
  expect_equal(ns$.az_group_trimean(x, group), expected, tolerance = 1e-12)
})

test_that("Windows parallel helper preserves lapply results", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv(c("AUTOZYME_WINDOWS_PARALLEL",
                      "AUTOZYME_WINDOWS_PARALLEL_MIN_TASKS"),
                    unset = NA_character_)
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) {
        Sys.unsetenv(nm)
      } else {
        do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
      }
    }
  }, add = TRUE)

  Sys.setenv(AUTOZYME_WINDOWS_PARALLEL = "0")
  expect_equal(ns$.zyme_mclapply(1:4, function(x) x + 1L, mc.cores = 2L),
               lapply(1:4, function(x) x + 1L))

  if (.Platform$OS.type == "windows") {
    Sys.setenv(AUTOZYME_WINDOWS_PARALLEL = "1",
               AUTOZYME_WINDOWS_PARALLEL_MIN_TASKS = "1")
    expect_equal(ns$.zyme_mclapply(1:4, function(x) x + 2L, mc.cores = 2L),
                 lapply(1:4, function(x) x + 2L))
  }
})
