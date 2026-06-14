# Wave-3 deep tests for R/verify.R ORCHESTRATION — the end-to-end paths the
# wave-1 unit file (test-unit-verify.R) deliberately left untouched:
# verify_patch() -> .verify_one_tier -> .run_worker (real Rscript subprocess)
# -> .run_evaluate -> package_verify.tsv writeback, plus the cached-baseline
# (no_baseline_confirm) hit path, multi-tier/multi-rep loops, gte/lte
# comparators, failing-metric + crash + patched-only branches, and the helper
# functions those paths reach (.cached_baseline_stats, .persist_reference_output,
# .sync_worker_thread_options, .ensure_task_thread_env, .append_package_verify_tsv).
#
# Strategy (mirrors the python wave-2 agent): drop a TRIVIAL self-contained patch
# file into the installed package's inst/patches/ dir (so the spawned worker
# subprocess can `library(autozyme); .ensure_registered(name)` and find it), on a
# tiny pure base-R upstream (`tools`), with a fast smoke recipe and a tiny task
# dir (task.yaml + evaluate.R). Each test cleans up its patch file + registry slot.
# Everything runs in <1s per worker spawn; fully deterministic.

ns_az_w3 <- function() asNamespace("autozyme")

# --- fixture builders --------------------------------------------------------

# Each patch must claim a DISTINCT (upstream, attr) — the registry rejects
# overlap — and we run many in one session, so hand out unique `tools` targets.
# These MUST be REAL functions in the tools namespace: activate() captures the
# original via get(fn_name, envir = ns), which errors on a non-existent symbol.
# They are obscure (Rd/vignette/assert tooling) so nothing else in the test run
# depends on them; the smoke `call` body does the real work, the target itself
# is never invoked, and cleanup restores it.
.w3_targets <- c("assertCondition", "assertError", "assertWarning", "bibstyle",
                 "Rd2txt", "Rd2HTML", "Rd2latex", "Rd2ex", "parse_Rd", "checkRd",
                 "buildVignette", "buildVignettes", "toTitleCase",
                 "encoded_text_to_latex", "showNonASCII", "showNonASCIIfile",
                 "texi2dvi", "texi2pdf", "compactPDF", "delimMatch",
                 "getVignetteInfo", "loadRdMacros", "makevars_site",
                 "makevars_user", "getDepList", "CRAN_package_db",
                 "BioC_package_db", "package_dependencies", "write_PACKAGES")
.w3_counter <- local({ i <- 0L; function() { i <<- i + 1L; i } })

.w3_patches_dir <- function() system.file("patches", package = "autozyme")

# Write a trivial registered patch into the installed package's patches/ dir.
# `call_body` is R source for the timed smoke region (gets `input`).
# `load_body` (optional) is R source for smoke$load (gets task_dir, tier).
.w3_mk_patch <- function(pname, call_body,
                         load_body = "paste0(\"f\", seq_len(40L))",
                         save_body = "saveRDS(result, file.path(dir, \"result.rds\"))") {
  idx <- .w3_counter()
  target <- .w3_targets[((idx - 1L) %% length(.w3_targets)) + 1L]
  pfile <- file.path(.w3_patches_dir(), paste0(pname, ".R"))
  src <- sprintf('
register_patch(
  name = "%s",
  upstream = "tools",
  targets = list(%s = function(x) x),
  smoke = list(
    load = function(task_dir, tier) { %s },
    call = function(input) { %s },
    save = function(result, dir, tier = NULL) { %s }
  ),
  tested_against = "tools 4.5.0"
)', pname, target, load_body, call_body, save_body)
  writeLines(src, pfile)
  pfile
}

# Tiny task dir with task.yaml (datasets + metrics) and an evaluate.R.
.w3_mk_task <- function(metric_lines, ev_body,
                        datasets_lines = c("datasets:",
                                           "  - tier: tiny",
                                           "    name: ds_tiny")) {
  td <- tempfile("w3task_"); dir.create(td)
  writeLines(c(datasets_lines, metric_lines), file.path(td, "task.yaml"))
  writeLines(ev_body, file.path(td, "evaluate.R"))
  td
}

# Default evaluate.R: compare patched vs reference result.rds, print n_match.
.w3_ev_match <- paste(
  'ref  <- readRDS(file.path(Sys.getenv("ZYME_REFERENCE_DIR"), "result.rds"))',
  'test <- readRDS(file.path(Sys.getenv("ZYME_TEST_DIR"), "result.rds"))',
  'cat(sprintf("n_match: %.6f\\n", mean(ref == test)))',
  sep = "\n")

.w3_cleanup <- function(pname, pfile, td = NULL) {
  ns <- ns_az_w3()
  try(autozyme::deactivate(pname), silent = TRUE)
  reg <- ns$.zyme_registry
  if (!is.null(reg[[pname]])) suppressWarnings(rm(list = pname, envir = reg))
  if (!is.null(pfile)) unlink(pfile)
  if (!is.null(td)) unlink(td, recursive = TRUE)
}

# A registered, fast (gte-passing) patch + matching task. Returns a closeable
# fixture. `sleep` controls the timed region length so we can steer speedup.
.w3_fixture <- function(pname, sleep_patched = 0.01, metric_lines = NULL,
                        ev_body = .w3_ev_match) {
  if (is.null(metric_lines)) {
    metric_lines <- c("metrics:",
                      "  - name: n_match",
                      "    comparator: gte",
                      "    threshold: 1.0")
  }
  pfile <- .w3_mk_patch(pname, sprintf("Sys.sleep(%g); toupper(input)", sleep_patched))
  td <- .w3_mk_task(metric_lines, ev_body)
  list(pname = pname, pfile = pfile, td = td,
       close = function() .w3_cleanup(pname, pfile, td))
}

# Read package_verify.tsv into a data frame (NA-aware, char columns).
.w3_read_pv <- function(td) {
  pv <- file.path(td, "package_verify.tsv")
  utils::read.table(pv, header = TRUE, sep = "\t", quote = "", comment.char = "",
                    stringsAsFactors = FALSE, na.strings = c("", "NA"), fill = TRUE,
                    colClasses = "character")
}

# ===========================================================================
# 1. verify_patch end-to-end: cache MISS -> real worker spawn -> TSV writeback
# ===========================================================================

test_that("verify_patch runs baseline+patched subprocesses and writes the TSV", {
  skip_if_not_installed("yaml")
  skip_on_os("windows")
  fx <- .w3_fixture("w3e2e_pass")
  on.exit(fx$close(), add = TRUE)

  out <- verify_patch(fx$pname, fx$td, tiers = "tiny", reps = 1, verbose = FALSE)

  # Summary data frame contract.
  expect_s3_class(out, "data.frame")
  expect_identical(out$tier, "tiny")
  expect_true(isTRUE(out$all_pass))
  expect_true(is.finite(out$baseline_sec) && out$baseline_sec > 0)
  expect_true(is.finite(out$patched_sec) && out$patched_sec > 0)
  expect_equal(out$speedup_x, out$baseline_sec / out$patched_sec)
  expect_equal(out$reps, 1L)

  # A reference_output_<tier> dir was persisted for future cache-hit reuse.
  expect_true(dir.exists(file.path(fx$td, "reference_output_tiny")))

  # package_verify.tsv: canonical 21-col header, baseline sorts above patched.
  df <- .w3_read_pv(fx$td)
  expect_equal(ncol(df), 21L)
  expect_true(all(c("baseline", "patched") %in% df$variant))
  expect_true(which(df$variant == "baseline")[1] < which(df$variant == "patched")[1])
  expect_true(all(df$dataset == "ds_tiny"))
  # patched row carries the pass flag + a metrics_json with n_match.
  patched <- df[df$variant == "patched", ]
  expect_identical(patched$pass[1], "true")
  expect_match(patched$metrics_json[1], "n_match")
  # tested_against propagated into package_version.
  expect_true(all(df$package_version == "tools 4.5.0"))
})

# ===========================================================================
# 2. Multi-rep loop: median over reps, both reps streamed to the TSV
# ===========================================================================

test_that("verify_patch reps>1 measures multiple reps and medians them", {
  skip_if_not_installed("yaml")
  skip_on_os("windows")
  # reps=3 disables auto-escalation -> exactly 3 reps measured.
  fx <- .w3_fixture("w3multirep")
  on.exit(fx$close(), add = TRUE)

  out <- verify_patch(fx$pname, fx$td, tiers = "tiny", reps = 3, verbose = FALSE)
  expect_equal(out$reps, 3L)

  df <- .w3_read_pv(fx$td)
  # 3 baseline + 3 patched rows, all for tier tiny.
  expect_equal(sum(df$variant == "baseline"), 3L)
  expect_equal(sum(df$variant == "patched"), 3L)
  expect_setequal(unique(df$rep_idx), c("1", "2", "3"))
})

# ===========================================================================
# 3. lte comparator + failing metric -> verdict FAIL
# ===========================================================================

test_that("verify_patch FAILs when an lte metric exceeds its threshold", {
  skip_if_not_installed("yaml")
  skip_on_os("windows")
  pfile <- .w3_mk_patch("w3lte_fail", "Sys.sleep(0.01); toupper(input)")
  td <- .w3_mk_task(
    c("metrics:", "  - name: err", "    comparator: lte", "    threshold: 0.01"),
    'cat("err: 0.5\n")')   # 0.5 > 0.01 -> lte fails
  on.exit(.w3_cleanup("w3lte_fail", pfile, td), add = TRUE)

  out <- verify_patch("w3lte_fail", td, tiers = "tiny", reps = 1, verbose = FALSE)
  expect_false(isTRUE(out$all_pass))

  df <- .w3_read_pv(td)
  expect_identical(df$pass[df$variant == "patched"][1], "false")
})

test_that("verify_patch PASSes an lte metric below its threshold", {
  skip_if_not_installed("yaml")
  skip_on_os("windows")
  pfile <- .w3_mk_patch("w3lte_pass", "Sys.sleep(0.01); toupper(input)")
  td <- .w3_mk_task(
    c("metrics:", "  - name: err", "    comparator: lte", "    threshold: 0.5"),
    'cat("err: 0.001\n")')
  on.exit(.w3_cleanup("w3lte_pass", pfile, td), add = TRUE)

  out <- verify_patch("w3lte_pass", td, tiers = "tiny", reps = 1, verbose = FALSE)
  expect_true(isTRUE(out$all_pass))
})

# ===========================================================================
# 4. crash path: evaluate.R exits non-zero -> tier skipped, sentinel TSV row
# ===========================================================================

test_that("verify_patch records a skip sentinel when evaluate.R crashes", {
  skip_if_not_installed("yaml")
  skip_on_os("windows")
  pfile <- .w3_mk_patch("w3crash", "Sys.sleep(0.01); toupper(input)")
  td <- .w3_mk_task(
    c("metrics:", "  - name: n_match", "    comparator: gte", "    threshold: 1.0"),
    'stop("boom in evaluate")')
  on.exit(.w3_cleanup("w3crash", pfile, td), add = TRUE)

  # The crashing evaluate.R makes system2() emit a benign "had status 1"
  # warning; the failure is handled (caught into a skip), so suppress it.
  out <- suppressWarnings(
    verify_patch("w3crash", td, tiers = "tiny", reps = 1, verbose = FALSE))
  # The tier is reported as a skip: NA baseline/patched, note carries the cause.
  expect_true(is.na(out$baseline_sec))
  expect_match(out$note, "evaluate.R exited")

  # A sentinel row is written (the .append_package_verify_tsv skip branch).
  df <- .w3_read_pv(td)
  expect_true(nrow(df) >= 1L)
  expect_true(any(grepl("evaluate.R exited", df$note)))
})

# ===========================================================================
# 5. patched_only mode: no baseline, speedup/pass NA, patched-only TSV note
# ===========================================================================

test_that("verify_patch patched_only measures only the patch (speedup NA)", {
  skip_if_not_installed("yaml")
  skip_on_os("windows")
  fx <- .w3_fixture("w3patchedonly")
  on.exit(fx$close(), add = TRUE)

  out <- verify_patch(fx$pname, fx$td, tiers = "tiny", reps = 1, verbose = FALSE,
                      patched_only = TRUE)
  expect_true(is.na(out$baseline_sec))
  expect_true(is.na(out$speedup_x))
  expect_true(is.na(out$all_pass))
  expect_true(is.finite(out$patched_sec) && out$patched_sec > 0)

  df <- .w3_read_pv(fx$td)
  expect_true(all(df$variant == "patched"))
  expect_true(any(df$note == "patched-only"))
})

# ===========================================================================
# 6. no_baseline_confirm: cache HIT — reuse baseline timing + reference output
# ===========================================================================

test_that("verify_patch no_baseline_confirm reuses cached baseline + reference", {
  skip_if_not_installed("yaml")
  skip_on_os("windows")
  fx <- .w3_fixture("w3cachehit")
  on.exit(fx$close(), add = TRUE)

  # First a full run to populate baseline rows + reference_output_tiny/.
  invisible(verify_patch(fx$pname, fx$td, tiers = "tiny", reps = 1, verbose = FALSE))
  expect_true(dir.exists(file.path(fx$td, "reference_output_tiny")))

  out <- verify_patch(fx$pname, fx$td, tiers = "tiny", reps = 1, verbose = FALSE,
                      no_baseline_confirm = TRUE)
  expect_true(isTRUE(out$all_pass))
  expect_true(is.finite(out$speedup_x))          # speedup vs cached baseline
  expect_match(out$note, "cached baseline")

  # The cache-hit run added a patched row but no NEW baseline rows beyond the
  # first run's (baseline reused, not re-measured).
  df <- .w3_read_pv(fx$td)
  expect_equal(sum(df$variant == "baseline"), 1L)
  expect_true(sum(df$variant == "patched") >= 2L)
})

test_that("verify_patch no_baseline_confirm errors cleanly with no cached baseline", {
  skip_if_not_installed("yaml")
  skip_on_os("windows")
  fx <- .w3_fixture("w3nocache")
  on.exit(fx$close(), add = TRUE)

  # No prior run -> no package_verify.tsv -> .cached_baseline_stats throws,
  # caught by verify_patch's tryCatch into a skip row.
  out <- verify_patch(fx$pname, fx$td, tiers = "tiny", reps = 1, verbose = FALSE,
                      no_baseline_confirm = TRUE)
  expect_true(is.na(out$baseline_sec))
  expect_match(out$note, "package_verify.tsv not found|no cached")
})

# ===========================================================================
# 7. Multi-tier: a good tier + a crashing tier, both reported in one summary
# ===========================================================================

test_that("verify_patch reports per-tier rows across multiple tiers", {
  skip_if_not_installed("yaml")
  skip_on_os("windows")
  # smoke$load stop()s for tier 'medium' -> worker fails -> tier skipped.
  load_body <- 'if (identical(tier, "medium")) stop("no medium dataset"); paste0("f", seq_len(40L))'
  pfile <- .w3_mk_patch("w3multitier", "Sys.sleep(0.01); toupper(input)",
                        load_body = load_body)
  td <- .w3_mk_task(
    c("metrics:", "  - name: n_match", "    comparator: gte", "    threshold: 1.0"),
    .w3_ev_match,
    datasets_lines = c("datasets:",
                       "  - tier: tiny",  "    name: ds_tiny",
                       "  - tier: medium", "    name: ds_med"))
  on.exit(.w3_cleanup("w3multitier", pfile, td), add = TRUE)

  out <- verify_patch("w3multitier", td, tiers = c("tiny", "medium"),
                      reps = 1, verbose = FALSE)
  expect_equal(nrow(out), 2L)
  expect_identical(out$tier, c("tiny", "medium"))
  # tiny passed, medium was skipped (NA baseline + a note).
  expect_true(isTRUE(out$all_pass[out$tier == "tiny"]))
  expect_true(is.na(out$baseline_sec[out$tier == "medium"]))
  expect_true(nzchar(out$note[out$tier == "medium"]))
})

# ===========================================================================
# 8. verbose=TRUE drives the summary-table + per-rep print branches
# ===========================================================================

test_that("verify_patch verbose=TRUE prints the summary table without erroring", {
  skip_if_not_installed("yaml")
  skip_on_os("windows")
  fx <- .w3_fixture("w3verbose")
  on.exit(fx$close(), add = TRUE)

  expect_output(
    out <- verify_patch(fx$pname, fx$td, tiers = "tiny", reps = 2, verbose = TRUE),
    "verify_patch|verdict|tier")
  expect_true(isTRUE(out$all_pass))
})

# ===========================================================================
# 9. verify_patch input validation / guard rails
# ===========================================================================

test_that("verify_patch errors on an unregisterable patch name", {
  skip_if_not_installed("yaml")
  expect_error(
    verify_patch("definitely_not_a_real_patch_xyz", tempdir(),
                 tiers = "tiny", reps = 1, verbose = FALSE),
    "no patch named|could not be registered")
})

test_that("verify_patch errors when the task has no smoke recipe", {
  skip_if_not_installed("yaml")
  # A patch registered WITHOUT smoke=, and a task dir without attest/smoke.R.
  pname <- "w3nosmoke"
  pfile <- file.path(.w3_patches_dir(), paste0(pname, ".R"))
  writeLines(sprintf(
    'register_patch(name="%s", upstream="tools", targets=list(md5sum=function(x) x))',
    pname), pfile)
  td <- tempfile("w3ns_"); dir.create(td)
  writeLines("metrics:\n  - name: x\n    comparator: gte\n    threshold: 1", file.path(td, "task.yaml"))
  writeLines("cat('x: 1\n')", file.path(td, "evaluate.R"))
  on.exit(.w3_cleanup(pname, pfile, td), add = TRUE)

  expect_error(verify_patch(pname, td, tiers = "tiny", reps = 1, verbose = FALSE),
               "no smoke recipe")
})

test_that("verify_patch errors on missing task.yaml metrics section", {
  skip_if_not_installed("yaml")
  skip_on_os("windows")
  pfile <- .w3_mk_patch("w3nometrics", "toupper(input)")
  td <- tempfile("w3nm_"); dir.create(td)
  # task.yaml with datasets but NO metrics: section.
  writeLines(c("datasets:", "  - tier: tiny", "    name: ds"), file.path(td, "task.yaml"))
  writeLines("cat('x: 1\n')", file.path(td, "evaluate.R"))
  on.exit(.w3_cleanup("w3nometrics", pfile, td), add = TRUE)

  expect_error(verify_patch("w3nometrics", td, tiers = "tiny", reps = 1, verbose = FALSE),
               "no .metrics")
})

# ===========================================================================
# 10. Helper-level coverage for branches the e2e path can't easily reach
# ===========================================================================

# ---- .append_package_verify_tsv: full baseline+patched row generation -------

test_that(".append_package_verify_tsv generates baseline+patched rows with speedups", {
  skip_if_not_installed("yaml")
  ns <- ns_az_w3()
  td <- tempfile("w3app_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  writeLines(c("datasets:", "  - tier: tiny", "    name: ds_tiny"),
             file.path(td, "task.yaml"))

  rows <- list(list(
    tier = "tiny",
    baseline_secs = c(2.0, 2.2),
    patched_secs  = c(1.0, 1.1),
    baseline_peaks_mb = c(200, 210),
    patched_peaks_mb  = c(100, 110),
    per_rep_pass = c(TRUE, TRUE),
    metrics_json = '{"acc": 0.99}',
    note = ""
  ))
  ns$.append_package_verify_tsv(td, "demo", rows)

  df <- .w3_read_pv(td)
  # 2 baseline + 2 patched.
  expect_equal(sum(df$variant == "baseline"), 2L)
  expect_equal(sum(df$variant == "patched"), 2L)
  patched <- df[df$variant == "patched", ]
  # speedup_x present and ~2x (baseline median 2.1 / patched ~1.0-1.1).
  sx <- as.numeric(patched$speedup_x)
  expect_true(all(sx > 1.5 & sx < 2.5))
  # peak_mb_fold present (~2x) and metrics_json carried.
  expect_true(all(as.numeric(patched$peak_mb_fold) > 1.5))
  expect_true(all(grepl("acc", patched$metrics_json)))
})

test_that(".append_package_verify_tsv uses baseline_sec_for_speedup fallback", {
  skip_if_not_installed("yaml")
  ns <- ns_az_w3()
  td <- tempfile("w3app2_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  writeLines(c("datasets:", "  - tier: tiny", "    name: ds_tiny"),
             file.path(td, "task.yaml"))
  # No baseline_secs (cached-baseline shape) but a baseline_sec_for_speedup.
  rows <- list(list(
    tier = "tiny",
    baseline_secs = numeric(0),
    patched_secs  = c(1.0),
    baseline_peaks_mb = numeric(0),
    patched_peaks_mb  = c(100),
    per_rep_pass = c(TRUE),
    metrics_json = "{}",
    note = "cached baseline X",
    baseline_sec_for_speedup = 3.0,
    baseline_peak_mb_for_speedup = 300
  ))
  ns$.append_package_verify_tsv(td, "demo", rows)
  df <- .w3_read_pv(td)
  patched <- df[df$variant == "patched", ]
  expect_equal(as.numeric(patched$speedup_x[1]), 3.0, tolerance = 1e-6)
  expect_equal(as.numeric(patched$peak_mb_fold[1]), 3.0, tolerance = 1e-6)
})

# ---- .cached_baseline_stats: host/thread-exact reuse + error branches -------

test_that(".cached_baseline_stats returns the latest comparable baseline rows", {
  skip_if_not_installed("yaml")
  ns <- ns_az_w3()
  td <- tempfile("w3cbs_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  writeLines(c("datasets:", "  - tier: tiny", "    name: ds_tiny"),
             file.path(td, "task.yaml"))
  # Write baseline rows matching THIS host's fingerprint so the exact-match
  # filter keeps them.
  si <- ns$.collect_system_info()
  hdr <- ns$.package_verify_header
  mkrow <- function(ts, sec, peak) paste(
    ts, "demo", "tiny", "ds_tiny", "1", "baseline",
    sprintf("%.6f", sec), "", "", sprintf("%.6f", peak), "", "", "", "",
    "0.3.0", "tools 4.5.0", "",
    si$system_os, si$system_cpu, si$system_ram_gb, si$system_threads,
    sep = "\t")
  writeLines(c(hdr,
               mkrow("2026-01-01T00:00:00", 5.0, 500),
               mkrow("2026-02-01T00:00:00", 3.0, 300)),
             file.path(td, "package_verify.tsv"))

  st <- ns$.cached_baseline_stats(td, "demo", "tiny")
  # Latest timestamp wins.
  expect_identical(st$timestamp, "2026-02-01T00:00:00")
  expect_equal(st$secs, 3.0)
  expect_equal(st$peaks_mb, 300)
})

test_that(".cached_baseline_stats errors when no baseline row matches the patch/tier", {
  skip_if_not_installed("yaml")
  ns <- ns_az_w3()
  td <- tempfile("w3cbs2_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  writeLines(c("datasets:", "  - tier: tiny", "    name: ds_tiny"),
             file.path(td, "task.yaml"))
  # Only a PATCHED row exists -> no baseline -> error.
  si <- ns$.collect_system_info()
  hdr <- ns$.package_verify_header
  row <- paste("2026-01-01T00:00:00", "demo", "tiny", "ds_tiny", "1", "patched",
               "1.0", "", "", "", "", "", "true", "{}", "0.3.0", "tools 4.5.0", "",
               si$system_os, si$system_cpu, si$system_ram_gb, si$system_threads,
               sep = "\t")
  writeLines(c(hdr, row), file.path(td, "package_verify.tsv"))
  expect_error(ns$.cached_baseline_stats(td, "demo", "tiny"),
               "no cached baseline timing")
})

test_that(".cached_baseline_stats rejects a host that differs in fingerprint", {
  skip_if_not_installed("yaml")
  ns <- ns_az_w3()
  td <- tempfile("w3cbs3_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  writeLines(c("datasets:", "  - tier: tiny", "    name: ds_tiny"),
             file.path(td, "task.yaml"))
  hdr <- ns$.package_verify_header
  # Deliberately-wrong system_os/cpu so the exact-match filter drops the row.
  row <- paste("2026-01-01T00:00:00", "demo", "tiny", "ds_tiny", "1", "baseline",
               "3.0", "", "", "300", "", "", "", "", "0.3.0", "tools 4.5.0", "",
               "SomeOtherOS 99", "Fictional CPU 9000", "999.9", "77",
               sep = "\t")
  writeLines(c(hdr, row), file.path(td, "package_verify.tsv"))
  expect_error(ns$.cached_baseline_stats(td, "demo", "tiny"),
               "no exact cached baseline")
})

test_that(".cached_baseline_stats rejects only-non-comparable (absorbed:) baselines", {
  skip_if_not_installed("yaml")
  ns <- ns_az_w3()
  td <- tempfile("w3cbs4_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  writeLines(c("datasets:", "  - tier: tiny", "    name: ds_tiny"),
             file.path(td, "task.yaml"))
  si <- ns$.collect_system_info()
  hdr <- ns$.package_verify_header
  row <- paste("2026-01-01T00:00:00", "demo", "tiny", "ds_tiny", "1", "baseline",
               "3.0", "", "", "300", "", "", "", "", "0.3.0", "tools 4.5.0",
               "absorbed: legacy bench",
               si$system_os, si$system_cpu, si$system_ram_gb, si$system_threads,
               sep = "\t")
  writeLines(c(hdr, row), file.path(td, "package_verify.tsv"))
  expect_error(ns$.cached_baseline_stats(td, "demo", "tiny"),
               "non-comparable")
})

# ---- .persist_reference_output / .copy_cached_reference_output --------------

test_that(".persist_reference_output copies into reference_output_<tier> and clears stale", {
  ns <- ns_az_w3()
  src <- tempfile("w3psrc_"); dir.create(src)
  td  <- tempfile("w3ptask_"); dir.create(td)
  on.exit({ unlink(src, recursive = TRUE); unlink(td, recursive = TRUE) }, add = TRUE)
  writeLines("new", file.path(src, "result.rds"))
  # Pre-seed a stale file in the destination to exercise the unlink branch.
  dst <- file.path(td, "reference_output_tiny"); dir.create(dst)
  writeLines("stale", file.path(dst, "old.txt"))

  ns$.persist_reference_output(src, td, "tiny")
  expect_true(file.exists(file.path(dst, "result.rds")))
  expect_false(file.exists(file.path(dst, "old.txt")))   # stale removed
})

test_that(".persist_reference_output errors on an empty source", {
  ns <- ns_az_w3()
  src <- tempfile("w3psrc2_"); dir.create(src)
  td  <- tempfile("w3ptask2_"); dir.create(td)
  on.exit({ unlink(src, recursive = TRUE); unlink(td, recursive = TRUE) }, add = TRUE)
  expect_error(ns$.persist_reference_output(src, td, "tiny"), "cannot persist empty")
})

# ---- thread-env propagation helpers ----------------------------------------

test_that(".ensure_task_thread_env sets the thread env vars from task.yaml", {
  skip_if_not_installed("yaml")
  ns <- ns_az_w3()
  td <- tempfile("w3tenv_"); dir.create(td)
  keys <- c("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
  old <- Sys.getenv(keys, unset = NA_character_)
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
    unlink(td, recursive = TRUE)
  }, add = TRUE)
  Sys.unsetenv(c("ZYME_THREADS", "AUTOZYME_THREADS"))
  writeLines(c("baseline_threads:", "  - 2"), file.path(td, "task.yaml"))

  th <- ns$.ensure_task_thread_env(td)
  expect_identical(th, 2L)
  expect_identical(Sys.getenv("OMP_NUM_THREADS"), "2")
  expect_identical(Sys.getenv("ZYME_THREADS"), "2")
})

test_that(".sync_worker_thread_options resolves threads from ZYME_THREADS", {
  ns <- ns_az_w3()
  old <- Sys.getenv(c("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS"),
                    unset = NA_character_)
  old_opt <- getOption("autozyme.threads")
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
    options(autozyme.threads = old_opt)
  }, add = TRUE)
  Sys.setenv(ZYME_THREADS = "2")
  th <- ns$.sync_worker_thread_options()
  expect_identical(th, 2L)
  expect_identical(getOption("autozyme.threads"), 2L)
})

test_that(".sync_worker_thread_options falls back to OMP_NUM_THREADS / default 1", {
  ns <- ns_az_w3()
  old <- Sys.getenv(c("ZYME_THREADS", "AUTOZYME_THREADS", "OMP_NUM_THREADS"),
                    unset = NA_character_)
  old_opt <- getOption("autozyme.threads")
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
    options(autozyme.threads = old_opt)
  }, add = TRUE)
  Sys.unsetenv(c("ZYME_THREADS", "AUTOZYME_THREADS"))
  Sys.setenv(OMP_NUM_THREADS = "1")
  th <- ns$.sync_worker_thread_options()
  expect_true(is.integer(th) && th >= 1L)
})

# ---- .run_worker error surfaces (no real subprocess success) ----------------

test_that(".run_worker raises a descriptive error when the worker crashes", {
  skip_on_os("windows")
  ns <- ns_az_w3()
  # Register a patch whose smoke$load stop()s -> the worker subprocess exits
  # non-zero with no JSON -> .run_worker should stop() with the stderr blob.
  pname <- "w3workercrash"
  pfile <- .w3_mk_patch(pname, "toupper(input)",
                        load_body = 'stop("intentional worker load failure")')
  td <- .w3_mk_task(
    c("metrics:", "  - name: n_match", "    comparator: gte", "    threshold: 1.0"),
    .w3_ev_match)
  on.exit(.w3_cleanup(pname, pfile, td), add = TRUE)

  out_dir <- file.path(td, "out"); dir.create(out_dir)
  expect_error(
    ns$.run_worker(pname, normalizePath(td), "tiny", out_dir,
                   activate = FALSE, verbose = FALSE),
    "verify-worker|intentional worker load failure")
})

test_that(".run_worker returns elapsed_sec + peak_mb on a successful spawn", {
  skip_on_os("windows")
  ns <- ns_az_w3()
  fx <- .w3_fixture("w3workerok")
  on.exit(fx$close(), add = TRUE)
  out_dir <- file.path(fx$td, "out"); dir.create(out_dir)
  res <- ns$.run_worker(fx$pname, normalizePath(fx$td), "tiny", out_dir,
                        activate = FALSE, verbose = FALSE)
  expect_true(is.list(res))
  expect_true(is.finite(res$elapsed_sec) && res$elapsed_sec >= 0)
  # peak_mb is NA on platforms without /usr/bin/time, else a positive number.
  expect_true(is.na(res$peak_mb) || res$peak_mb > 0)
  expect_true(file.exists(file.path(out_dir, "result.rds")))
})

# ===========================================================================
# 11. More helper branch coverage (cheap, no subprocess)
# ===========================================================================

# ---- .parse_peak_mb success path -------------------------------------------

test_that(".parse_peak_mb parses a value matching the detector pattern", {
  ns <- ns_az_w3()
  ti <- ns$.detect_time_cmd()
  skip_if(is.null(ti))
  f <- tempfile("w3peak_"); on.exit(unlink(f), add = TRUE)
  # Synthesize a stats line that the active detector's pattern will match.
  if (identical(Sys.info()[["sysname"]], "Darwin")) {
    writeLines("  1048576  maximum resident set size", f)
    mb <- ns$.parse_peak_mb(f, ti)
    expect_equal(mb, 1.0, tolerance = 1e-9)       # 1048576 bytes -> 1 MB
  } else {
    writeLines("\tMaximum resident set size (kbytes): 1024", f)
    mb <- ns$.parse_peak_mb(f, ti)
    expect_equal(mb, 1.0, tolerance = 1e-9)       # 1024 kB -> 1 MB
  }
  # Non-matching content -> NA.
  writeLines("nothing useful here", f)
  expect_true(is.na(ns$.parse_peak_mb(f, ti)))
})

# ---- .resolve_smoke: attest/smoke.R that omits `smoke` -> hard error --------

test_that(".resolve_smoke errors when attest/smoke.R never defines `smoke`", {
  ns <- ns_az_w3()
  td <- tempfile("w3rs_"); dir.create(file.path(td, "attest"), recursive = TRUE)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  writeLines("x <- 1  # no `smoke` object defined", file.path(td, "attest", "smoke.R"))
  expect_error(ns$.resolve_smoke(td, NULL), "must define")
})

# ---- .append_one_pv_row with no task.yaml (dataset backfills to '') --------

test_that(".append_one_pv_row works without task.yaml (empty dataset column)", {
  ns <- ns_az_w3()
  td <- tempfile("w3ap1_"); dir.create(td)   # no task.yaml
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  ns$.append_one_pv_row(td, "demo", "tiny", "patched", 1, sec = 1.0, peak_mb = 50,
                        rep_pass = TRUE, metrics_json = '{"x": 1}', speedup_x = 2.0)
  df <- .w3_read_pv(td)
  expect_equal(nrow(df), 1L)
  expect_identical(df$variant, "patched")
  expect_identical(df$pass, "true")
  expect_true(is.na(df$dataset) || df$dataset == "")  # no tier map -> blank
})

# ---- no_baseline_confirm + verbose: cached-baseline print + reuse path ------

test_that("verify_patch no_baseline_confirm + verbose prints the cached-baseline block", {
  skip_if_not_installed("yaml")
  skip_on_os("windows")
  fx <- .w3_fixture("w3cacheverbose")
  on.exit(fx$close(), add = TRUE)
  invisible(verify_patch(fx$pname, fx$td, tiers = "tiny", reps = 1, verbose = FALSE))
  expect_output(
    out <- verify_patch(fx$pname, fx$td, tiers = "tiny", reps = 2, verbose = TRUE,
                        no_baseline_confirm = TRUE),
    "cached|baseline")
  expect_true(isTRUE(out$all_pass))
  expect_match(out$note, "cached baseline")
})
