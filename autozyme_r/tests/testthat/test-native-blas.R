test_that("dynamic BLAS helpers fall back to base R when disabled", {
  old <- Sys.getenv(c("AUTOZYME_BLAS_DISABLE", "AUTOZYME_DYNAMIC_BLAS"),
                    unset = NA_character_)
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
  }, add = TRUE)
  Sys.setenv(AUTOZYME_BLAS_DISABLE = "1")

  ns <- asNamespace("autozyme")
  set.seed(1)
  X <- matrix(rnorm(30), nrow = 10)
  Y <- matrix(rnorm(20), nrow = 10)
  A <- matrix(rnorm(24), nrow = 6)
  B <- matrix(rnorm(30), nrow = 6)

  expect_equal(ns$.az_xtx(X), crossprod(X), tolerance = 1e-12)
  expect_equal(ns$.az_crossprod(X, Y), crossprod(X, Y), tolerance = 1e-12)
  expect_equal(ns$.az_crossprod(X, Y[, 1]), crossprod(X, Y[, 1]),
               tolerance = 1e-12)
  expect_equal(ns$.az_gemm(A, B, transA = TRUE), t(A) %*% B,
               tolerance = 1e-12)

  info <- ns$blas_info()
  expect_false(isTRUE(info$available))
  expect_true(isTRUE(info$disabled))

  Sys.setenv(AUTOZYME_BLAS_DISABLE = "0", AUTOZYME_DYNAMIC_BLAS = "0")
  expect_equal(ns$.az_xtx(X), crossprod(X), tolerance = 1e-12)
  expect_equal(ns$.az_gemm(A, B, transA = TRUE), t(A) %*% B,
               tolerance = 1e-12)
})
