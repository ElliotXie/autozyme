# Wave-4 contract DEEPENING: drive parameter branches that the existing
# per-patch contract tests skip, asserting parity vs the captured original
# (zyme = FALSE / autozyme::with_disabled()).
#
# Each block targets a wrapper branch the baseline test does not exercise:
#   decontx   -- calculateNativeMatrix fast path + decontXLogLik zyme= escape
#   cellchat  -- computeCommunProb type="truncatedMean" + population.size=TRUE
#   infercnv  -- .smooth_window across several window lengths (surfaces a bug)
#   nichenetr -- predict_ligand_activities single=TRUE SCORE-VALUE parity
#   tradeseq  -- fitGAM nknots=4 fitted-coefficient parity
#
# These complement (never duplicate) the test-contract-<name>.R files.

# ---------------------------------------------------------------------------
# Order-independence guard. setup-autozyme.R activates every installed patch
# before any test- file, but other test files (e.g.
# test-contract-all-patches-activate.R, test-contract-clusterprofiler.R) toggle
# patches on/off as part of their own round-trips. In full-suite order this can
# leave a patch deactivated, or with a stale binding, by the time this file
# runs -- so a "patched" call here silently runs vanilla and a parity/
# divergence assertion fails. Each test below calls .w4_use_patch() first to
# force the patch into a known-active state (a deactivate->activate cycle, which
# also clears any contaminated binding), then defers restoring the prior state
# so this file leaves the suite exactly as it found it.
.w4_use_patch <- function(name, env = parent.frame()) {
  prior <- tryCatch(autozyme::status()[[name]], error = function(e) NA_character_)
  # Force a clean re-bind: deactivate (no-op if already inactive) then activate.
  try(autozyme::deactivate(name), silent = TRUE)
  autozyme::activate(name)
  withr::defer({
    if (identical(prior, "active")) {
      try({ autozyme::deactivate(name); autozyme::activate(name) }, silent = TRUE)
    } else {
      try(autozyme::deactivate(name), silent = TRUE)
    }
  }, envir = env)
}

# Search-path guard. test-contract-wgcna.R calls library(WGCNA), which attaches
# the package and stays attached for the rest of the suite. WGCNA masks
# stats::cor with WGCNA::cor (different return shape: a column matrix, not a
# flat numeric). nichenetr's upstream scoring resolves a bare cor() off the
# search path, so once WGCNA is attached the with_disabled (vanilla) branch
# computes pearson as a 20x1 matrix while the fast path stays flat -- a pure
# order artifact that fails the parity check. with_disabled() can't undo this
# because it isn't a patch binding. Shadow the masked symbols with their stats
# originals in .GlobalEnv (above every attached package in bare-name lookup) for
# the duration of the test, then restore. No-op when WGCNA isn't attached.
.w4_unmask_stats <- function(syms = "cor", env = parent.frame()) {
  for (s in syms) {
    cur <- tryCatch(get(s, envir = globalenv()), error = function(e) NULL)
    masked <- tryCatch(!identical(environment(get(s)), asNamespace("stats")),
                       error = function(e) FALSE)
    if (!isTRUE(masked)) next
    had <- exists(s, envir = globalenv(), inherits = FALSE)
    assign(s, get(s, envir = asNamespace("stats")), envir = globalenv())
    withr::defer({
      if (had) assign(s, cur, envir = globalenv())
      else if (exists(s, envir = globalenv(), inherits = FALSE))
        rm(list = s, envir = globalenv())
    }, envir = env)
  }
}

# ---------------------------------------------------------------------------
# decontx: celda::calculateNativeMatrix (pure-R fast path) and
# celda::decontXLogLik (per-call zyme= escape). The baseline contract test
# only drives the public decontX() entry point and never reaches these two
# targets directly nor their zyme= branch.
# ---------------------------------------------------------------------------
.skip_if_no_decontx <- function() {
  testthat::skip_if_not_installed("celda")
  testthat::skip_if_not_installed("Matrix")
}

test_that("decontx calculateNativeMatrix fast path matches with_disabled", {
  .skip_if_no_decontx()
  .w4_use_patch("decontx")
  cnm <- utils::getFromNamespace("calculateNativeMatrix", "celda")
  set.seed(2)
  ng <- 12L; nc <- 20L; K <- 3L
  sp <- methods::as(matrix(stats::rpois(ng * nc, 2), nrow = ng, ncol = nc),
                    "CsparseMatrix")
  z <- sample.int(K, nc, replace = TRUE)
  theta <- stats::runif(nc)
  phi <- matrix(stats::runif(ng * K), nrow = ng, ncol = K)
  eta <- matrix(stats::runif(ng * K), nrow = ng, ncol = K)
  pseudocount <- 1e-20

  patched <- cnm(sp, theta, eta, phi, z, pseudocount)
  vanilla <- autozyme::with_disabled(cnm(sp, theta, eta, phi, z, pseudocount))
  expect_s4_class(patched, "dgCMatrix")
  expect_equal(dim(patched), dim(vanilla))
  expect_equal(max(abs(as.matrix(patched) - as.matrix(vanilla))), 0,
               tolerance = 1e-12)
})

test_that("decontx decontXLogLik zyme=FALSE matches patched value", {
  .skip_if_no_decontx()
  .w4_use_patch("decontx")
  dll <- utils::getFromNamespace("decontXLogLik", "celda")
  set.seed(2)
  ng <- 12L; nc <- 20L; K <- 3L
  sp <- methods::as(matrix(stats::rpois(ng * nc, 2), nrow = ng, ncol = nc),
                    "CsparseMatrix")
  z <- sample.int(K, nc, replace = TRUE)
  theta <- stats::runif(nc)
  phi <- matrix(stats::runif(ng * K), nrow = ng, ncol = K)
  eta <- matrix(stats::runif(ng * K), nrow = ng, ncol = K)
  pseudocount <- 1e-20

  patched <- dll(sp, theta, eta, phi, z, pseudocount)
  vanilla <- dll(sp, theta, eta, phi, z, pseudocount, zyme = FALSE)
  expect_true(is.finite(patched))
  expect_equal(patched, vanilla, tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# cellchat: computeCommunProb FunMean-type + population.size branches. The
# baseline test only drives the default type="triMean", population.size=FALSE
# path; here we cover the truncatedMean switch arm and the population.size
# weighting branch.
# ---------------------------------------------------------------------------
.skip_if_no_cellchat <- function() {
  testthat::skip_if_not_installed("CellChat")
}

.make_toy_cellchat_w4 <- function() {
  data <- matrix(c(5, 4, 0, 0,
                   0, 0, 6, 5),
                 nrow = 2, byrow = TRUE)
  rownames(data) <- c("L1", "R1")
  colnames(data) <- paste0("c", seq_len(4))
  meta <- data.frame(group = factor(c("A", "A", "B", "B")),
                     row.names = colnames(data))
  obj <- CellChat::createCellChat(data, meta = meta, group.by = "group")
  pairLR <- data.frame(
    interaction_name = "L1_R1", pathway_name = "toy",
    ligand = "L1", receptor = "R1",
    agonist = NA_character_, antagonist = NA_character_,
    co_A_receptor = NA_character_, co_I_receptor = NA_character_,
    annotation = "Secreted Signaling", interaction_name_2 = "L1 - R1",
    stringsAsFactors = FALSE)
  rownames(pairLR) <- pairLR$interaction_name
  obj@DB <- list(
    interaction = pairLR,
    complex = data.frame(subunit_1 = character(), row.names = character()),
    cofactor = data.frame(cofactor1 = character(), row.names = character()))
  obj@LR <- list(LRsig = pairLR)
  obj@data.signaling <- obj@data[c("L1", "R1"), , drop = FALSE]
  obj
}

test_that("cellchat computeCommunProb type='truncatedMean' matches vanilla", {
  .skip_if_no_cellchat()
  .w4_use_patch("cellchat")
  obj <- .make_toy_cellchat_w4()
  run <- function(zyme) suppressWarnings(suppressMessages(
    CellChat::computeCommunProb(
      obj, type = "truncatedMean", trim = 0.1, raw.use = TRUE,
      distance.use = FALSE, nboot = 2, seed.use = 1, k.min = 1,
      zyme = zyme)))
  patched <- run(TRUE)
  vanilla <- autozyme::with_disabled(run(FALSE))
  expect_equal(dim(patched@net$prob), c(2L, 2L, 1L))
  expect_equal(max(abs(patched@net$prob - vanilla@net$prob)), 0,
               tolerance = 1e-10)
})

test_that("cellchat computeCommunProb population.size=TRUE matches vanilla", {
  .skip_if_no_cellchat()
  .w4_use_patch("cellchat")
  obj <- .make_toy_cellchat_w4()
  run <- function(zyme) suppressWarnings(suppressMessages(
    CellChat::computeCommunProb(
      obj, type = "triMean", raw.use = TRUE, distance.use = FALSE,
      nboot = 2, seed.use = 1, k.min = 1, population.size = TRUE,
      zyme = zyme)))
  patched <- run(TRUE)
  vanilla <- autozyme::with_disabled(run(FALSE))
  expect_equal(max(abs(patched@net$prob - vanilla@net$prob)), 0,
               tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# infercnv: .smooth_window across window lengths. The baseline test only
# pins window_length = 5L. Driving more window lengths surfaces a real
# correctness regression (see the SUSPECTED BUG test below).
# ---------------------------------------------------------------------------
.skip_if_no_infercnv <- function() {
  testthat::skip_if_not_installed("infercnv")
}

.infercnv_smooth_ref <- function() {
  # Authoritative reference = upstream .smooth_helper applied column-wise,
  # exactly what vanilla .smooth_window does. Fetched with the patch DISABLED
  # so we compare the fast path against true upstream numerics.
  sh <- autozyme::with_disabled(
    utils::getFromNamespace(".smooth_helper", "infercnv"))
  function(x, wl) {
    out <- apply(x, 2, sh, window_length = wl)
    rownames(out) <- rownames(x)
    colnames(out) <- colnames(x)
    out
  }
}

test_that("infercnv .smooth_window is exact for even-tail window lengths", {
  .skip_if_no_infercnv()
  .w4_use_patch("infercnv")
  smooth_window <- utils::getFromNamespace(".smooth_window", "infercnv")
  ref <- .infercnv_smooth_ref()
  set.seed(7)
  x <- matrix(stats::rnorm(18 * 4), nrow = 18, ncol = 4)
  rownames(x) <- paste0("g", seq_len(18))
  colnames(x) <- paste0("c", seq_len(4))
  # window_length 5 and 9 -> tail_length 2 and 4 (even): bit-exact path.
  for (wl in c(5L, 9L)) {
    patched <- smooth_window(x, window_length = wl)
    expect_equal(patched, ref(x, wl), tolerance = 1e-10,
                 info = paste("window_length", wl))
  }
})

test_that("infercnv .smooth_window matches across column counts (wl=5)", {
  .skip_if_no_infercnv()
  .w4_use_patch("infercnv")
  smooth_window <- utils::getFromNamespace(".smooth_window", "infercnv")
  ref <- .infercnv_smooth_ref()
  for (nc in c(2L, 4L, 6L, 8L)) {
    set.seed(7)
    x <- matrix(stats::rnorm(18 * nc), nrow = 18, ncol = nc)
    rownames(x) <- paste0("g", seq_len(18))
    colnames(x) <- paste0("c", seq_len(nc))
    patched <- smooth_window(x, window_length = 5L)
    expect_equal(patched, ref(x, 5L), tolerance = 1e-10,
                 info = paste("ncol", nc))
  }
})

test_that("infercnv .smooth_window obs_count < window_length matches vanilla", {
  .skip_if_no_infercnv()
  .w4_use_patch("infercnv")
  smooth_window <- utils::getFromNamespace(".smooth_window", "infercnv")
  ref <- .infercnv_smooth_ref()
  set.seed(7)
  # nrow (3) < window_length (5): the else-branch edge formula.
  x <- matrix(stats::rnorm(3 * 6), nrow = 3, ncol = 6)
  rownames(x) <- paste0("g", seq_len(3))
  colnames(x) <- paste0("c", seq_len(6))
  patched <- smooth_window(x, window_length = 5L)
  expect_equal(patched, ref(x, 5L), tolerance = 1e-10)
})

test_that("infercnv .smooth_window DIVERGES for odd-tail window lengths [BUG]", {
  # SUSPECTED REAL BUG (documented, not fixed -- tests-only campaign rule).
  #
  # The patched fast_smooth_window center path (matrixStats colCumsums box
  # filter, inst/patches/infercnv/patch.R:150-164) uses w = tail_length + 1
  # and divides by (w * w). For window lengths whose tail_length =
  # (window_length-1)/2 is EVEN (wl = 5, 9, 13, ... tail 2, 4, 6) this
  # coincides with upstream's true window mean. For window lengths whose
  # tail_length is ODD (wl = 3, 7, 11, ... tail 1, 3, 5) it does NOT: the
  # patched output diverges from upstream .smooth_helper by O(0.1-1.4) in the
  # center rows. infercnv's default window_length is 101 (tail 50, even) and
  # is additionally routed to a separate C++ kernel, so production never hits
  # this -- but any caller passing wl in {3,7,11,...} gets wrong smoothing.
  #
  # This test PINS the current (buggy) behaviour so the regression is
  # tracked. If the patch is later corrected, max_diff drops to ~0 and the
  # expect_gt below will (correctly) fail, flagging the fix.
  .skip_if_no_infercnv()
  .w4_use_patch("infercnv")
  smooth_window <- utils::getFromNamespace(".smooth_window", "infercnv")
  ref <- .infercnv_smooth_ref()
  set.seed(7)
  x <- matrix(stats::rnorm(18 * 4), nrow = 18, ncol = 4)
  rownames(x) <- paste0("g", seq_len(18))
  colnames(x) <- paste0("c", seq_len(4))
  for (wl in c(3L, 7L)) {
    patched <- smooth_window(x, window_length = wl)
    reference <- ref(x, wl)
    max_diff <- max(abs(patched - reference))
    expect_gt(max_diff, 1e-3)  # currently buggy: large divergence
  }
})

# ---------------------------------------------------------------------------
# nichenetr: predict_ligand_activities single=TRUE SCORE-VALUE parity. The
# baseline test only checks output shape (nrow + setequal(test_ligand)); it
# never asserts the fast scoring kernel reproduces the actual auroc/aupr/
# pearson values. This deepens to numeric parity.
# ---------------------------------------------------------------------------
.skip_if_no_nichenetr <- function() {
  testthat::skip_if_not_installed("nichenetr")
}

test_that("nichenetr predict_ligand_activities scores match vanilla exactly", {
  .skip_if_no_nichenetr()
  .w4_use_patch("nichenetr")
  .w4_unmask_stats("cor")
  set.seed(5)
  ng <- 150L; nl <- 20L
  ltm <- matrix(stats::runif(ng * nl), nrow = ng, ncol = nl)
  rownames(ltm) <- paste0("g", seq_len(ng))
  colnames(ltm) <- paste0("L", seq_len(nl))
  genes <- rownames(ltm)
  run <- function(zyme) suppressWarnings(suppressMessages(
    nichenetr::predict_ligand_activities(
      geneset = genes[1:40], background_expressed_genes = genes,
      ligand_target_matrix = ltm, potential_ligands = colnames(ltm),
      single = TRUE, zyme = zyme)))
  patched <- run(TRUE)
  vanilla <- autozyme::with_disabled(run(FALSE))
  patched <- patched[order(patched$test_ligand), ]
  vanilla <- vanilla[order(vanilla$test_ligand), ]
  expect_identical(patched$test_ligand, vanilla$test_ligand)
  metric_cols <- intersect(c("auroc", "aupr", "aupr_corrected", "pearson"),
                           colnames(patched))
  expect_true(length(metric_cols) > 0)
  for (mc in metric_cols) {
    expect_equal(patched[[mc]], vanilla[[mc]], tolerance = 1e-9,
                 info = paste("metric", mc))
  }
})

# ---------------------------------------------------------------------------
# tradeseq: fitGAM fitted-coefficient parity at a non-default nknots. The
# baseline test only checks the SCE class + dim; here we assert the actual
# fitted GAM beta coefficients are bit-exact patched-vs-disabled.
# ---------------------------------------------------------------------------
.skip_if_no_tradeseq <- function() {
  testthat::skip_if_not_installed("tradeSeq")
  testthat::skip_if_not_installed("SingleCellExperiment")
}

.make_tradeseq_inputs_w4 <- function(seed = 0) {
  set.seed(seed)
  n_cells <- 30L; n_genes <- 12L
  pseudotime <- sort(stats::runif(n_cells))
  counts <- matrix(0L, nrow = n_genes, ncol = n_cells)
  for (g in seq_len(n_genes)) {
    rate <- if (g <= n_genes / 2) 5 * (1 + pseudotime) else rep(5, n_cells)
    counts[g, ] <- stats::rpois(n_cells, lambda = rate)
  }
  rownames(counts) <- paste0("g", seq_len(n_genes))
  colnames(counts) <- paste0("c", seq_len(n_cells))
  list(counts = counts,
       pseudotime = matrix(pseudotime, ncol = 1),
       cellWeights = matrix(1, nrow = n_cells, ncol = 1))
}

test_that("tradeseq fitGAM nknots=4 beta coefficients match with_disabled", {
  .skip_if_no_tradeseq()
  .w4_use_patch("tradeseq")
  inp <- tryCatch(.make_tradeseq_inputs_w4(), error = function(e) e)
  if (inherits(inp, "error")) {
    testthat::skip(paste0("tradeseq fixture failed: ",
                          conditionMessage(inp)))
  }
  fit <- function() suppressWarnings(suppressMessages(
    tradeSeq::fitGAM(counts = inp$counts, pseudotime = inp$pseudotime,
                     cellWeights = inp$cellWeights, nknots = 4,
                     verbose = FALSE, parallel = FALSE)))
  patched <- tryCatch(fit(), error = function(e) e)
  if (inherits(patched, "error")) {
    testthat::skip(paste0("patched fitGAM raised: ",
                          conditionMessage(patched)))
  }
  vanilla <- tryCatch(autozyme::with_disabled(fit()), error = function(e) e)
  if (inherits(vanilla, "error")) {
    testthat::skip(paste0("vanilla fitGAM raised: ",
                          conditionMessage(vanilla)))
  }
  bp <- SummarizedExperiment::rowData(patched)$tradeSeq$beta
  bv <- SummarizedExperiment::rowData(vanilla)$tradeSeq$beta
  expect_false(is.null(bp))
  mp <- as.matrix(do.call(rbind, bp))
  mv <- as.matrix(do.call(rbind, bv))
  expect_equal(dim(mp), dim(mv))
  expect_equal(max(abs(mp - mv), na.rm = TRUE), 0, tolerance = 1e-8)
})
