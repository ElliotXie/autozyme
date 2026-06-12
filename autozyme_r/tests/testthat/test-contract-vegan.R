# Contract: vegan::adonis2 (PERMANOVA) + permutest.cca
#
# Patched surface: 3 targets (adonis0, adonis2, permutest.cca). Public
# entry is `vegan::adonis2(formula, data, ...)`. fast_adonis0 has an
# explicit `zyme = TRUE` arg; fast_adonis2 forwards via ...

.skip_if_no_vegan <- function() {
  testthat::skip_if_not_installed("vegan")
}

.make_vegan_inputs <- function(seed = 0) {
  set.seed(seed)
  k <- 4
  per_group <- 20
  n_samples <- k * per_group
  n_species <- 30
  group <- factor(rep(letters[seq_len(k)], each = per_group))
  abund <- matrix(stats::rpois(n_samples * n_species, lambda = 2),
                  nrow = n_samples, ncol = n_species)
  for (g in seq_len(k)) {
    rows_g <- which(as.integer(group) == g)
    species_g <- ((g - 1) * 5 + 1):(g * 5 + 5)
    abund[rows_g, species_g] <- abund[rows_g, species_g] +
      stats::rpois(length(rows_g) * length(species_g), lambda = 6)
  }
  rownames(abund) <- paste0("site", seq_len(n_samples))
  colnames(abund) <- paste0("sp", seq_len(n_species))
  list(abund = abund, group = group)
}

# adonis2 strictly evaluates LHS in `parent.frame()`, which bypasses both
# `enclos = environment(formula)` and any wrapping like suppressWarnings.
# The robust pattern is to bind the LHS into globalenv() briefly and
# restore on exit. The full abundance matrix is the LHS so the patch's
# Bray-Curtis fast path stays in play (vs handing in a precomputed dist).
.run_adonis2 <- function(abund, df, ..., zyme_disabled = FALSE) {
  assign("..az_abund..", abund, envir = globalenv())
  on.exit(rm("..az_abund..", envir = globalenv()), add = TRUE)
  fml <- as.formula("..az_abund.. ~ group", env = globalenv())
  if (zyme_disabled) {
    autozyme::with_disabled(
      vegan::adonis2(fml, data = df, ...)
    )
  } else {
    vegan::adonis2(fml, data = df, ...)
  }
}

test_that("adonis2 returns an anova.cca data.frame with expected columns", {
  .skip_if_no_vegan()
  inp <- .make_vegan_inputs()
  df <- data.frame(group = inp$group)
  out <- .run_adonis2(inp$abund, df, permutations = 99, method = "bray")
  expect_s3_class(out, "anova.cca")
  for (col in c("Df", "SumOfSqs", "F", "Pr(>F)")) {
    expect_true(col %in% colnames(out),
                info = paste("missing adonis2 column:", col))
  }
  expect_gte(nrow(out), 2L)
})

test_that("adonis2 with_disabled() matches patched F-statistic", {
  .skip_if_no_vegan()
  inp <- .make_vegan_inputs()
  df <- data.frame(group = inp$group)
  set.seed(42)
  vanilla <- .run_adonis2(inp$abund, df, permutations = 99,
                          method = "bray", zyme_disabled = TRUE)
  set.seed(42)
  patched <- .run_adonis2(inp$abund, df, permutations = 99,
                          method = "bray")
  expect_equal(class(patched), class(vanilla))
  expect_equal(nrow(patched), nrow(vanilla))
  # F-statistic depends only on data + design (no permutation RNG),
  # so should match within tight float tolerance.
  f_v <- vanilla[["F"]]
  f_p <- patched[["F"]]
  f_v <- f_v[!is.na(f_v)]
  f_p <- f_p[!is.na(f_p)]
  expect_equal(f_p, f_v, tolerance = 1e-6,
               info = "F-statistic drift between patched and vanilla")
})

test_that("adonis2 zyme=FALSE per-call delegates cleanly", {
  .skip_if_no_vegan()
  # Fixed 2026-05-28: fast_adonis0 is now permissive on call shape, so
  # vanilla adonis2's internal `adonis0(formula, data=, method=)` call
  # into our still-patched namespace lands in the dots-delegation branch
  # and forwards to .orig_adonis0 cleanly. See vegan/patch.R Bug 7 note.
  inp <- .make_vegan_inputs()
  df <- data.frame(group = inp$group)
  out <- .run_adonis2(inp$abund, df, permutations = 99,
                      method = "bray", zyme = FALSE)
  expect_s3_class(out, "anova.cca")
})
