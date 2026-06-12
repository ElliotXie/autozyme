# Contract: MAST::lrTest (hybrid likelihood-ratio test)
#
# Patched signature: fast_lrTest_hybrid(object, hypothesis, ...) where
# object is a ZlmFit and hypothesis is either a CoefficientHypothesis or
# a single character (coefficient name). Non-character / non-CH input
# delegates to vanilla.
#
# Building a real ZlmFit needs MAST::zlm on a SingleCellAssay. We build
# the smallest possible one (a few cells x a few genes) and only assert
# the return shape — numerical correctness is heavy to verify in a
# contract test and already covered by upstream's own tests.

.skip_if_no_mast <- function() {
  testthat::skip_if_not_installed("MAST")
  testthat::skip_if_not_installed("SummarizedExperiment")
  # MAST's S4 classes (e.g. CoefficientHypothesis) are only registered
  # after the namespace is fully loaded; bare MAST:: access can miss them.
  suppressMessages(requireNamespace("MAST", quietly = TRUE))
}

.make_tiny_zlm <- function(seed = 0) {
  set.seed(seed)
  n_cells <- 60
  n_genes <- 25
  cond <- rep(c("a", "b"), each = n_cells / 2)
  # Real differential signal in the first 8 genes: +3 in condition b.
  # Without this, fast_lrTest's `termIdx != 1` guard fires on every
  # gene that has no conditionb effect, delegating to vanilla which
  # then hits the "doesn't alter the model" check.
  expr <- matrix(stats::rnorm(n_genes * n_cells, mean = 3, sd = 1.5),
                 nrow = n_genes, ncol = n_cells)
  signal_genes <- seq_len(8)
  for (g in signal_genes) {
    expr[g, cond == "b"] <- expr[g, cond == "b"] + 3
  }
  # Zero-inflate ~20% to make MAST's hurdle model meaningful.
  expr[matrix(stats::runif(length(expr)) < 0.2, dim(expr))] <- 0
  rownames(expr) <- paste0("g", seq_len(n_genes))
  colnames(expr) <- paste0("c", seq_len(n_cells))

  cdat <- data.frame(
    wellKey = colnames(expr),
    cngeneson = colMeans(expr > 0),
    condition = factor(cond)
  )
  fdat <- data.frame(primerid = rownames(expr))

  sca <- suppressWarnings(
    MAST::FromMatrix(expr, cdat, fdat, check_sanity = FALSE)
  )
  suppressWarnings(suppressMessages(
    MAST::zlm(~ condition + cngeneson, sca, parallel = FALSE)
  ))
}

test_that("lrTest with CoefficientHypothesis returns a 3D array", {
  .skip_if_no_mast()
  fit <- .make_tiny_zlm()
  # Force the S4 class to be visible. `MAST::CoefficientHypothesis` can
  # fail S4 class lookup if MAST isn't attached; attaching once here
  # avoids that without scattering library(MAST) through the file.
  attachNamespace("MAST")
  on.exit(try(detach("package:MAST", unload = FALSE, character.only = TRUE),
              silent = TRUE), add = TRUE)
  hyp <- CoefficientHypothesis("conditionb")
  out <- suppressWarnings(MAST::lrTest(fit, hyp))
  expect_true(is.array(out))
  expect_equal(dim(out)[1], 25L)  # 25 genes -> 25 rows
})

test_that("lrTest with_disabled() matches patched output", {
  .skip_if_no_mast()
  fit <- .make_tiny_zlm()
  attachNamespace("MAST")
  on.exit(try(detach("package:MAST", unload = FALSE, character.only = TRUE),
              silent = TRUE), add = TRUE)
  hyp <- CoefficientHypothesis("conditionb")
  # The MAST patch's fast_lrTest_hybrid has no `zyme` arg in its
  # signature -- it relies on the framework's is_disabled() toggle for
  # bypass. Use with_disabled({...}) for the vanilla baseline.
  vanilla <- autozyme::with_disabled(
    suppressWarnings(MAST::lrTest(fit, hyp))
  )
  patched <- suppressWarnings(MAST::lrTest(fit, hyp))
  expect_equal(dim(vanilla), dim(patched))
  expect_equal(dimnames(vanilla), dimnames(patched))
  # Hurdle Pr(>Chisq) is the canonical MAST output; same math under
  # both paths -> match within tight float tolerance.
  expect_equal(patched[, "hurdle", "Pr(>Chisq)"],
               vanilla[, "hurdle", "Pr(>Chisq)"],
               tolerance = 1e-4)
})

test_that("lrTest non-Coefficient hypothesis delegates cleanly", {
  .skip_if_no_mast()
  fit <- .make_tiny_zlm()
  # Numeric-matrix hypothesis form: the fast path doesn't handle it
  # (per the patch's `else { return(.mast_orig_lrTest(...)) }` branch).
  # Both patched and disabled should error identically — same upstream
  # raise, not a new error class.
  out_p <- tryCatch(suppressWarnings(MAST::lrTest(fit, matrix(0, 0, 0))),
                    error = function(e) e)
  out_v <- tryCatch(
    autozyme::with_disabled(
      suppressWarnings(MAST::lrTest(fit, matrix(0, 0, 0)))
    ),
    error = function(e) e
  )
  expect_equal(class(out_p), class(out_v))
})
