# Contract: CellChat::triMean (and computeCommunProb)
#
# Patched surface: 2 targets (triMean, computeCommunProb).
# triMean is a pure numeric utility -- trivially testable.
# computeCommunProb needs a full CellChat object (signaling DB + grouped
# normalized expression). Skip when that fixture isn't feasible; the
# universal activation smoke already proves the patch loads cleanly.

.skip_if_no_cellchat <- function() {
  testthat::skip_if_not_installed("CellChat")
}

.make_toy_cellchat <- function() {
  data <- matrix(c(5, 4, 0, 0,
                   0, 0, 6, 5),
                 nrow = 2, byrow = TRUE)
  rownames(data) <- c("L1", "R1")
  colnames(data) <- paste0("c", seq_len(4))
  meta <- data.frame(
    group = factor(c("A", "A", "B", "B")),
    row.names = colnames(data)
  )
  obj <- CellChat::createCellChat(data, meta = meta, group.by = "group")
  pairLR <- data.frame(
    interaction_name = "L1_R1",
    pathway_name = "toy",
    ligand = "L1",
    receptor = "R1",
    agonist = NA_character_,
    antagonist = NA_character_,
    co_A_receptor = NA_character_,
    co_I_receptor = NA_character_,
    annotation = "Secreted Signaling",
    interaction_name_2 = "L1 - R1",
    stringsAsFactors = FALSE
  )
  rownames(pairLR) <- pairLR$interaction_name
  obj@DB <- list(
    interaction = pairLR,
    complex = data.frame(subunit_1 = character(), row.names = character()),
    cofactor = data.frame(cofactor1 = character(), row.names = character())
  )
  obj@LR <- list(LRsig = pairLR)
  obj@data.signaling <- obj@data[c("L1", "R1"), , drop = FALSE]
  obj
}

test_that("triMean returns the canonical (Q1 + 2*Q2 + Q3) / 4 average", {
  .skip_if_no_cellchat()
  x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  out <- CellChat::triMean(x)
  # Vanilla triMean = mean(quantile(x, c(0.25, 0.50, 0.50, 0.75))).
  # For x = 1..10: q1 = 3.25, q2 = 5.5 (twice), q3 = 7.75 -> mean = 5.5.
  expect_equal(out, 5.5, tolerance = 1e-9)
})

test_that("triMean honors na.rm", {
  .skip_if_no_cellchat()
  x <- c(1, 2, NA, 4, 5)
  # na.rm=TRUE (default) -> drops NA before quantile.
  out <- CellChat::triMean(x, na.rm = TRUE)
  expect_true(is.finite(out))
  # na.rm=FALSE -> quantile errors on NA; capture the error to confirm
  # the patch doesn't silently swallow the NA.
  out_strict <- tryCatch(
    CellChat::triMean(x, na.rm = FALSE),
    error = function(e) e,
    warning = function(w) w
  )
  # Either error/warning OR NA result -- vanilla varies by R version.
  # The contract is "doesn't silently return a finite value despite NA".
  if (is.numeric(out_strict)) {
    expect_true(is.na(out_strict),
                info = "na.rm=FALSE with NA should yield NA, not silent finite")
  }
})

test_that("triMean zyme=FALSE matches patched output", {
  .skip_if_no_cellchat()
  x <- c(1.5, 2.7, 3.1, 4.9, 5.0, 6.6, 7.2, 8.8, 9.1, 10.0)
  patched <- CellChat::triMean(x)
  vanilla <- CellChat::triMean(x, zyme = FALSE)
  expect_equal(patched, vanilla, tolerance = 1e-9)
})

test_that("triMean on empty input returns NaN (matches vanilla)", {
  .skip_if_no_cellchat()
  patched <- suppressWarnings(CellChat::triMean(numeric(0)))
  vanilla <- suppressWarnings(CellChat::triMean(numeric(0), zyme = FALSE))
  # Both should be NaN -- compare directly via is.na.
  expect_true(is.na(patched))
  expect_true(is.na(vanilla))
})

test_that("computeCommunProb contract reachable via full CellChat object", {
  .skip_if_no_cellchat()
  obj <- .make_toy_cellchat()
  patched <- suppressWarnings(suppressMessages(
    CellChat::computeCommunProb(
      obj, raw.use = TRUE, distance.use = FALSE, nboot = 2,
      seed.use = 1, k.min = 1)
  ))
  vanilla <- suppressWarnings(suppressMessages(autozyme::with_disabled(
    CellChat::computeCommunProb(
      obj, raw.use = TRUE, distance.use = FALSE, nboot = 2,
      seed.use = 1, k.min = 1)
  )))
  expect_equal(dim(patched@net$prob), c(2L, 2L, 1L))
  expect_equal(dim(patched@net$pval), c(2L, 2L, 1L))
  expect_equal(dim(patched@net$prob), dim(vanilla@net$prob))
  expect_equal(dimnames(patched@net$prob), dimnames(vanilla@net$prob))
})
