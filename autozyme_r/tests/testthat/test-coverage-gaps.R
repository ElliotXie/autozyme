# Coverage-gap meta-tests for the autozyme R package.
#
# Two jobs:
#  (1) Mirror the CI coverage-drift check (.github/scripts/ci_check_coverage.R):
#      assert every registered patch has a contract test, and flag any patch
#      that does not. Run inside testthat so a missing contract test fails the
#      local suite, not only CI.
#  (2) Enumerate every exported Rcpp/C++ entry point in R/RcppExports.R and
#      assert each is either exercised directly by a test-cpp-*.R file or is on
#      the documented "not-directly-exercised" allowlist (with the reason it is
#      hard to unit-test in isolation). This makes the C++ coverage frontier
#      explicit and fails loudly if a NEW kernel is added with no test.

# --------------------------------------------------------------------------
# (1) Every registered patch has a contract test
# --------------------------------------------------------------------------

# Mirror of KNOWN_SPLITS / META_TESTS in ci_check_coverage.R.
.KNOWN_SPLITS <- list(
  seurat = c(
    "FindAllMarkers", "FindIntegrationAnchors", "FindNeighbors",
    "FindVariableFeatures", "NormalizeData", "RunCCA", "RunPCA",
    "ScaleData", "SCTransform", "seurat-s3-stragglers"
  )
)
.META_TESTS <- c("all-patches-activate", "zzz-options")

test_that("every registered patch has a test-contract-<name>.R file", {
  skip_if_not_installed("autozyme")
  patches <- autozyme::list_patches()
  expect_gt(length(patches), 0L)

  test_dir <- "."  # test_file() runs with wd = tests/testthat
  contract_files <- sub("^test-contract-", "",
                        sub("\\.R$", "",
                            list.files(test_dir, pattern = "^test-contract-.*\\.R$")))

  expected <- character()
  for (p in patches) {
    if (p %in% names(.KNOWN_SPLITS)) {
      expected <- c(expected, .KNOWN_SPLITS[[p]])
    } else {
      expected <- c(expected, p)
    }
  }
  expected <- unique(expected)

  missing <- setdiff(expected, contract_files)
  expect_identical(
    missing, character(0),
    info = paste("patches without a contract test:", paste(missing, collapse = ", "))
  )
})

test_that("no stale contract test points at an unknown patch", {
  skip_if_not_installed("autozyme")
  patches <- autozyme::list_patches()
  contract_files <- sub("^test-contract-", "",
                        sub("\\.R$", "",
                            list.files(".", pattern = "^test-contract-.*\\.R$")))
  expected <- unique(unlist(lapply(patches, function(p) {
    if (p %in% names(.KNOWN_SPLITS)) .KNOWN_SPLITS[[p]] else p
  })))
  extra <- setdiff(contract_files, c(expected, .META_TESTS))
  expect_identical(
    extra, character(0),
    info = paste("contract tests for unknown patches:", paste(extra, collapse = ", "))
  )
})

# --------------------------------------------------------------------------
# (2) Exported Rcpp kernel coverage frontier
# --------------------------------------------------------------------------

# Kernels NOT exercised directly by a test-cpp-*.R file, each with the reason.
# Adding a NEW export without a test (and without listing it here) fails the
# enumeration test below.
.UNTESTED_KERNELS <- c(
  # Stateful MCMC / IRLS / EM inner loops that need a fully-initialized
  # upstream object or .GlobalEnv likelihood tables; not unit-testable on a
  # tiny synthetic input without reconstructing the whole solver state.
  "fast_iterate_t_impl",          # bayesspace Gibbs/MH loop (RcppDist, RNG state)
  "cpp_updateState",              # MAST IRLS state update (full design + priors)
  "cpp_aggregate_triMean_boot",   # cellchat bootstrap tensor (OMP + permutation set)
  "cpp_unified_inner",            # cellchat unified bootstrap reject counts (OMP + flat-offset args)
  "fastFgseaMultilevelBatchCpp",  # fgsea multilevel EsRuler MCMC (RNG-driven p-values)

  # rctd spline / IRWLS solvers: every one consumes Q_mat / SQ_mat / X_vals /
  # K_val likelihood tables that spacexr::set_likelihood_vars writes into
  # .GlobalEnv. Building a valid spline basis by hand is out of scope for a
  # tiny unit test; covered indirectly by test-contract-rctd.R.
  "rctd_cpp_calc_log_l_sum",
  "rctd_cpp_calc_log_l_vec",
  "rctd_cpp_get_d1_d2",
  "rctd_cpp_get_der_fast_nonbulk",
  "rctd_cpp_solve_wls_p1",
  "rctd_cpp_solve_wls_p2",
  "rctd_cpp_irwls_sparse_p12",
  "rctd_cpp_irwls_full_nonbulk",
  "rctd_cpp_score_sparse_candidates",
  "rctd_cpp_fit_sparse_pair",

  # for_paper V3 OMP kernels with multi-output filter/rank/pval pipelines that
  # only make sense as a chained sequence; the entry kernel
  # omp_sum_nnz_expm1_groups IS covered (test-cpp-markers.R, n_threads=1L).
  "omp_filter_pct_lfc",
  "omp_subset_transpose_rank_pval",
  "omp_rank_ustat_pval",

  # az_blas_gemm: the general DGEMM dispatch. Its specialization
  # az_blas_crossprod IS covered (test-cpp-misc.R) when a backend is loadable;
  # gemm has no extra pure-numeric surface beyond crossprod here.
  "az_blas_gemm",

  # nb_dDeta_log_cpp: eta-space chain-rule wrapper around nb_Dd_cpp's mu-space
  # derivatives (which ARE covered, test-cpp-misc.R). Its closed form is a long
  # chain of the already-tested mu derivatives.
  "nb_dDeta_log_cpp"
)

test_that("every RcppExports kernel is covered or explicitly allowlisted", {
  rcpp_exports <- file.path("..", "..", "R", "RcppExports.R")
  # When run from tests/testthat the package source tree is two levels up;
  # fall back to the installed-package location otherwise.
  if (!file.exists(rcpp_exports)) {
    rcpp_exports <- system.file("R", "RcppExports.R", package = "autozyme")
  }
  skip_if(!nzchar(rcpp_exports) || !file.exists(rcpp_exports),
          "RcppExports.R source not locatable from the test wd")

  rx <- readLines(rcpp_exports)
  exported <- sub(" <- function.*$", "",
                  grep("^[A-Za-z_0-9.]+ <- function", rx, value = TRUE))
  expect_gt(length(exported), 50L)

  # Gather the text of all direct-kernel test files.
  cpp_test_files <- list.files(".", pattern = "^test-cpp.*\\.R$", full.names = TRUE)
  expect_gt(length(cpp_test_files), 0L)
  test_text <- unlist(lapply(cpp_test_files, readLines))

  exercised <- vapply(exported,
                      function(fn) any(grepl(fn, test_text, fixed = TRUE)),
                      logical(1))

  # Each exported kernel must be either exercised directly or allowlisted.
  uncovered <- exported[!exercised]
  not_accounted <- setdiff(uncovered, .UNTESTED_KERNELS)
  expect_identical(
    not_accounted, character(0),
    info = paste0(
      "These exported Rcpp kernels are neither exercised by a test-cpp-*.R ",
      "file nor on the documented .UNTESTED_KERNELS allowlist: ",
      paste(not_accounted, collapse = ", ")
    )
  )

  # The allowlist must not go stale: nothing on it should actually be tested
  # (if it is, remove it from the allowlist).
  stale <- intersect(.UNTESTED_KERNELS, exported[exercised])
  expect_identical(
    stale, character(0),
    info = paste0("allowlist entries that ARE now tested (remove them): ",
                  paste(stale, collapse = ", "))
  )
})

test_that("the direct-kernel tests exercise a majority of the export surface", {
  rcpp_exports <- file.path("..", "..", "R", "RcppExports.R")
  if (!file.exists(rcpp_exports)) {
    rcpp_exports <- system.file("R", "RcppExports.R", package = "autozyme")
  }
  skip_if(!nzchar(rcpp_exports) || !file.exists(rcpp_exports),
          "RcppExports.R source not locatable")
  rx <- readLines(rcpp_exports)
  exported <- sub(" <- function.*$", "",
                  grep("^[A-Za-z_0-9.]+ <- function", rx, value = TRUE))
  test_text <- unlist(lapply(
    list.files(".", pattern = "^test-cpp.*\\.R$", full.names = TRUE), readLines))
  exercised <- vapply(exported,
                      function(fn) any(grepl(fn, test_text, fixed = TRUE)),
                      logical(1))
  # Before this campaign: 0 exports were directly exercised by name.
  # Assert we now cover a clear majority of the surface.
  expect_gt(mean(exercised), 0.6)
})
