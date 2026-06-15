# Extracted from test-w3-shared-infra-native.R:345

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
.skip_if_omp_unsafe()
.with_flag("AUTOZYME_GROUPED_REDUCTIONS", "1")
x <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8), nrow = 2, byrow = TRUE)
group <- factor(rep("only", 4))
got <- .az_group_trimean(x, group)
tri <- function(v) mean(stats::quantile(v, c(0.25, 0.5, 0.5, 0.75), names = FALSE))
expect_equal(got[1, 1], tri(x[1, ]), tolerance = 1e-9)
expect_equal(got[2, 1], tri(x[2, ]), tolerance = 1e-9)
