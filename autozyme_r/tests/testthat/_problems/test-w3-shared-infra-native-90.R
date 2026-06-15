# Extracted from test-w3-shared-infra-native.R:90

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
.with_flag("AUTOZYME_SPARSE_KERNELS", "1")
dense <- matrix(c(1, 0, 2, 3,
                    0, 0, 5, 4,
                    2, 1, 0, 0), nrow = 3, byrow = TRUE)
rownames(dense) <- paste0("r", 1:3)
sm <- .tiny_dgc(dense)
rs <- .az_dgc_row_stats(sm, detect_threshold = 0)
expect_equal(unname(rs$sum), rowSums(dense))
expect_equal(unname(rs$mean), rowMeans(dense))
expect_equal(unname(rs$variance), apply(dense, 1L, stats::var))
