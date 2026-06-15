# Extracted from test-w3-shared-infra-native.R:507

# prequel ----------------------------------------------------------------------
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
.skip_if_omp_unsafe <- function() {
  testthat::skip_if_not_installed("Matrix")
  if (!identical(Sys.getenv("OMP_NUM_THREADS"), "1")) {
    testthat::skip("native kernels are only segfault-safe at OMP_NUM_THREADS=1")
  }
}
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

# test -------------------------------------------------------------------------
dense <- matrix(c(1, NA, 3, 0,
                    2,  4, NA, 1), nrow = 2, byrow = TRUE)
group <- factor(c("a", "a", "b", "b"), levels = c("a", "b"))
res <- .az_dense_group_summary_fallback(dense, group, detect_threshold = 0,
                                          na.rm = TRUE)
expect_equal(res$sum_by_group[1, "a"], 1)
expect_equal(res$sum_by_group[2, "a"], 6)
