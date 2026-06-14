# Unit tests for R/native_blas.R. Focus on the PURE-R surface: path
# splitting/normalization, disable gating, the dense-double coercion, the base
# R GEMM fallback (which .az_gemm uses when native is off/unavailable), thread
# resolution, and the dynamic-BLAS enable predicate. The native az_blas_gemm /
# az_blas_info entry points are exercised only through their safe fallbacks or
# the public blas_info(), never forced into a thread-spinning kernel here.
#
# Complements the existing test-native-blas.R (which checks blas_info/.az_gemm
# correctness against base R %*%) with finer-grained helper coverage.

# ---- gating ----------------------------------------------------------------

test_that(".az_blas_disabled tracks AUTOZYME_BLAS_DISABLE=1", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("AUTOZYME_BLAS_DISABLE", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_BLAS_DISABLE")
    else Sys.setenv(AUTOZYME_BLAS_DISABLE = old)
  }, add = TRUE)

  Sys.unsetenv("AUTOZYME_BLAS_DISABLE")
  expect_false(ns$.az_blas_disabled())
  Sys.setenv(AUTOZYME_BLAS_DISABLE = "1")
  expect_true(ns$.az_blas_disabled())
  Sys.setenv(AUTOZYME_BLAS_DISABLE = "0")  # only "1" disables
  expect_false(ns$.az_blas_disabled())
})

test_that("blas_info() reports disabled when AUTOZYME_BLAS_DISABLE=1", {
  old <- Sys.getenv("AUTOZYME_BLAS_DISABLE", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_BLAS_DISABLE")
    else Sys.setenv(AUTOZYME_BLAS_DISABLE = old)
  }, add = TRUE)
  Sys.setenv(AUTOZYME_BLAS_DISABLE = "1")
  info <- blas_info()
  expect_type(info, "list")
  expect_false(info$available)
  expect_true(info$disabled)
})

test_that("blas_info() returns the documented fields", {
  info <- blas_info()
  expect_type(info, "list")
  expect_true(all(c("available", "backend", "disabled") %in% names(info)))
  expect_type(info$available, "logical")
})

# ---- path helpers ----------------------------------------------------------

test_that(".az_split_paths splits on the platform path separator + dedups", {
  ns <- asNamespace("autozyme")
  sep <- .Platform$path.sep
  joined <- paste("/a", "/b", "/a", sep = sep)
  expect_setequal(ns$.az_split_paths(joined), c("/a", "/b"))
})

test_that(".az_split_paths drops NA/empty and returns character(0) for none", {
  ns <- asNamespace("autozyme")
  expect_identical(ns$.az_split_paths(c(NA_character_, "", NA)), character())
  expect_identical(ns$.az_split_paths(character()), character())
})

test_that(".az_norm_paths normalizes + dedups, drops empties", {
  ns <- asNamespace("autozyme")
  td <- normalizePath(tempdir(), winslash = "/")
  out <- ns$.az_norm_paths(c(td, td, "", NA))
  expect_length(out, 1L)
  expect_identical(out, normalizePath(td, winslash = "/", mustWork = FALSE))
})

test_that(".az_blas_patterns returns regexes for the current OS shared-lib ext", {
  ns <- asNamespace("autozyme")
  pats <- ns$.az_blas_patterns()
  expect_type(pats, "character")
  expect_true(length(pats) >= 1L)
  ext <- if (.Platform$OS.type == "windows") "dll"
         else if (identical(Sys.info()[["sysname"]], "Darwin")) "dylib"
         else "so"
  expect_true(any(grepl(ext, pats, fixed = TRUE)))
})

test_that(".az_list_blas_files returns character(0) for non-existent dirs", {
  ns <- asNamespace("autozyme")
  expect_identical(ns$.az_list_blas_files(c("/no/such/dir", "")), character())
})

test_that(".az_blas_paths returns nothing when BLAS is disabled", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("AUTOZYME_BLAS_DISABLE", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_BLAS_DISABLE")
    else Sys.setenv(AUTOZYME_BLAS_DISABLE = old)
  }, add = TRUE)
  Sys.setenv(AUTOZYME_BLAS_DISABLE = "1")
  expect_identical(ns$.az_blas_paths(), character())
})

# ---- dense coercion + base GEMM fallback -----------------------------------

test_that(".az_as_dense_double passes matrices through as double", {
  ns <- asNamespace("autozyme")
  m <- matrix(1:6, 2, 3)  # integer
  out <- ns$.az_as_dense_double(m)
  expect_true(is.matrix(out))
  expect_true(is.double(out))
  expect_equal(out, matrix(as.double(1:6), 2, 3))
})

test_that(".az_as_dense_double promotes a plain numeric vector to a column", {
  ns <- asNamespace("autozyme")
  out <- ns$.az_as_dense_double(c(1, 2, 3))
  expect_equal(dim(out), c(3L, 1L))
})

test_that(".az_as_dense_double returns NULL for non-numeric input", {
  ns <- asNamespace("autozyme")
  expect_null(ns$.az_as_dense_double("not numeric"))
  expect_null(ns$.az_as_dense_double(list(1, 2)))
})

test_that(".az_base_gemm matches base R %*% under each transpose combo", {
  ns <- asNamespace("autozyme")
  set.seed(1)
  A <- matrix(rnorm(6), 2, 3)
  B <- matrix(rnorm(12), 3, 4)
  expect_equal(ns$.az_base_gemm(A, B), A %*% B)
  expect_equal(ns$.az_base_gemm(A, A, transA = TRUE), crossprod(A))      # t(A) %*% A
  expect_equal(ns$.az_base_gemm(B, B, transB = TRUE), tcrossprod(B))     # B %*% t(B)
})

test_that(".az_gemm falls back to base %*% and equals it (native off)", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("AUTOZYME_BLAS_DISABLE", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_BLAS_DISABLE")
    else Sys.setenv(AUTOZYME_BLAS_DISABLE = old)
  }, add = TRUE)
  Sys.setenv(AUTOZYME_BLAS_DISABLE = "1")  # force the base R path
  set.seed(2)
  A <- matrix(rnorm(8), 4, 2)
  B <- matrix(rnorm(6), 2, 3)
  Y <- matrix(rnorm(12), 4, 3)             # nrow matches A for X'Y
  expect_equal(ns$.az_gemm(A, B), A %*% B)
  expect_equal(ns$.az_crossprod(A), crossprod(A))         # X'X
  expect_equal(ns$.az_xtx(A), crossprod(A))
  expect_equal(ns$.az_crossprod(A, Y), crossprod(A, Y))   # X'Y
})

test_that(".az_gemm with fallback=FALSE on non-matrix inputs errors", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("AUTOZYME_BLAS_DISABLE", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_BLAS_DISABLE")
    else Sys.setenv(AUTOZYME_BLAS_DISABLE = old)
  }, add = TRUE)
  Sys.setenv(AUTOZYME_BLAS_DISABLE = "1")
  expect_error(ns$.az_gemm("x", "y", fallback = FALSE),
               "dense numeric")
})

# ---- thread + enable predicates --------------------------------------------

test_that(".az_default_blas_threads honors ZYME_THREADS then falls through", {
  ns <- asNamespace("autozyme")
  old_z <- Sys.getenv("ZYME_THREADS", unset = NA_character_)
  old_opt <- getOption("autozyme.threads")
  on.exit({
    if (is.na(old_z)) Sys.unsetenv("ZYME_THREADS")
    else Sys.setenv(ZYME_THREADS = old_z)
    options(autozyme.threads = old_opt)
  }, add = TRUE)

  Sys.setenv(ZYME_THREADS = "5")
  expect_identical(ns$.az_default_blas_threads(), 5L)

  Sys.unsetenv("ZYME_THREADS")
  options(autozyme.threads = 3L)
  expect_identical(ns$.az_default_blas_threads(), 3L)

  options(autozyme.threads = NULL)
  n <- ns$.az_default_blas_threads()   # falls to auto_threads()
  expect_true(is.integer(n) || is.numeric(n))
  expect_gte(n, 1L)
})

test_that(".az_dynamic_blas_enabled is FALSE when BLAS is globally disabled", {
  ns <- asNamespace("autozyme")
  old <- Sys.getenv("AUTOZYME_BLAS_DISABLE", unset = NA_character_)
  on.exit({
    if (is.na(old)) Sys.unsetenv("AUTOZYME_BLAS_DISABLE")
    else Sys.setenv(AUTOZYME_BLAS_DISABLE = old)
  }, add = TRUE)
  Sys.setenv(AUTOZYME_BLAS_DISABLE = "1")
  expect_false(ns$.az_dynamic_blas_enabled())
})

test_that(".az_dynamic_blas_enabled honors a per-patch env override", {
  ns <- asNamespace("autozyme")
  old_g <- Sys.getenv("AUTOZYME_DYNAMIC_BLAS", unset = NA_character_)
  old_p <- Sys.getenv("AUTOZYME_SEURAT_BLAS", unset = NA_character_)
  old_bd <- Sys.getenv("AUTOZYME_BLAS_DISABLE", unset = NA_character_)
  old_dis <- Sys.getenv("AUTOZYME_DISABLE", unset = NA_character_)
  on.exit({
    restore <- function(nm, v) if (is.na(v)) Sys.unsetenv(nm) else do.call(Sys.setenv, setNames(list(v), nm))
    restore("AUTOZYME_DYNAMIC_BLAS", old_g)
    restore("AUTOZYME_SEURAT_BLAS", old_p)
    restore("AUTOZYME_BLAS_DISABLE", old_bd)
    restore("AUTOZYME_DISABLE", old_dis)
  }, add = TRUE)
  Sys.unsetenv(c("AUTOZYME_DYNAMIC_BLAS", "AUTOZYME_BLAS_DISABLE", "AUTOZYME_DISABLE"))

  Sys.setenv(AUTOZYME_SEURAT_BLAS = "0")
  expect_false(ns$.az_dynamic_blas_enabled(patch = "seurat"))
  Sys.setenv(AUTOZYME_SEURAT_BLAS = "1")
  expect_true(ns$.az_dynamic_blas_enabled(patch = "seurat"))
})

test_that(".az_windows_crossprod uses base crossprod off Windows", {
  ns <- asNamespace("autozyme")
  skip_on_os("windows")
  set.seed(3)
  X <- matrix(rnorm(12), 4, 3)
  expect_equal(ns$.az_windows_crossprod(X), crossprod(X))
})
