# Wave-4 framework-polish tests: push REACHABLE remaining coverage in the
# R framework modules (verify.R / core.R / threads.R / speedups.R /
# subsets.R / zzz.R) that wave-1 (test-unit-*.R) and wave-3
# (test-w3-verify-deep.R) left uncovered. This file deliberately AVOIDS the
# Windows/subprocess paths (unreachable on this host) and instead exercises
# the pure comparison / tolerance / parse / TSV-schema / thread-resolution /
# registry-dispatch branches.
#
# DUAL-RUNNER COVERAGE NOTE
# -------------------------
# covr::file_coverage(parent_env = asNamespace("autozyme")) only credits an
# INTERNAL (`.dotted`) helper when the test calls it by BARE name (so lexical
# lookup resolves to covr's instrumented copy). But a bare internal name does
# not resolve under plain `testthat::test_file()` ("could not find function").
# `.w4_use()` reconciles both: under file_coverage the bare name already
# exists (instrumented) so it is left alone; under test_file it imports the
# real namespace binding so the bare call resolves. EXPORTED functions
# (set_threads, auto_threads, speedups, status, ...) resolve as bare names in
# both runners, so they are called directly.

.w4_az_ns <- asNamespace("autozyme")

.w4_use <- function(...) {
  nms <- c(...)
  for (nm in nms) {
    if (!exists(nm, inherits = TRUE)) {
      assign(nm, get(nm, envir = .w4_az_ns), envir = parent.frame())
    }
  }
}

# ---------------------------------------------------------------------------
# zzz.R :: .onLoad option branches (only run at package load otherwise, so
# file_coverage never triggers them; drive each branch explicitly here).
# ---------------------------------------------------------------------------

test_that(".onLoad short-circuits when AUTOZYME_DISABLED is truthy", {
  .w4_use(".onLoad")
  withr::with_envvar(c(AUTOZYME_DISABLED = "1", AUTOZYME_DISABLE = ""), {
    expect_null(.onLoad("lib", "autozyme"))
  })
})

test_that(".onLoad short-circuits when AUTOZYME_DISABLE (no D) is truthy", {
  .w4_use(".onLoad")
  withr::with_envvar(c(AUTOZYME_DISABLE = "true", AUTOZYME_DISABLED = ""), {
    expect_null(.onLoad("lib", "autozyme"))
  })
})

test_that(".onLoad RAISES a too-small future.globals.maxSize to 16 GiB", {
  .w4_use(".onLoad")
  old <- getOption("future.globals.maxSize")
  on.exit(options(future.globals.maxSize = old), add = TRUE)
  withr::with_envvar(c(AUTOZYME_DISABLED = "", AUTOZYME_DISABLE = ""), {
    options(future.globals.maxSize = 1024L)            # tiny -> must be raised
    suppressMessages(.onLoad("lib", "autozyme"))
    expect_gte(getOption("future.globals.maxSize"), 16 * 1024^3)
  })
})

test_that(".onLoad does NOT lower an already-large future.globals.maxSize", {
  .w4_use(".onLoad")
  old <- getOption("future.globals.maxSize")
  on.exit(options(future.globals.maxSize = old), add = TRUE)
  big <- 64 * 1024^3
  withr::with_envvar(c(AUTOZYME_DISABLED = "", AUTOZYME_DISABLE = ""), {
    options(future.globals.maxSize = big)
    suppressMessages(.onLoad("lib", "autozyme"))
    expect_identical(getOption("future.globals.maxSize"), big)
  })
})

test_that(".onLoad raises a non-finite future.globals.maxSize", {
  .w4_use(".onLoad")
  old <- getOption("future.globals.maxSize")
  on.exit(options(future.globals.maxSize = old), add = TRUE)
  withr::with_envvar(c(AUTOZYME_DISABLED = "", AUTOZYME_DISABLE = ""), {
    options(future.globals.maxSize = Inf)              # !is.finite branch
    suppressMessages(.onLoad("lib", "autozyme"))
    expect_gte(getOption("future.globals.maxSize"), 16 * 1024^3)
  })
})

# ---------------------------------------------------------------------------
# threads.R :: the capture/apply/restore state machine + auto_threads edges
# (bare-name calls so file_coverage credits the .az_* internals).
# ---------------------------------------------------------------------------

test_that(".az_capture/apply/restore round-trips env vars and the R option", {
  .w4_use(".az_capture_thread_state", ".az_apply_thread_count",
          ".az_restore_thread_state", ".thread_env_vars")
  withr::with_envvar(stats::setNames(rep("7", length(.thread_env_vars)),
                                     .thread_env_vars), {
    old_opt <- getOption("autozyme.threads")
    on.exit(options(autozyme.threads = old_opt), add = TRUE)
    options(autozyme.threads = 9L)
    state <- .az_capture_thread_state()
    expect_type(state, "list")
    expect_true(all(c("env", "option", "blas") %in% names(state)))
    .az_apply_thread_count(3L)
    expect_identical(Sys.getenv("OMP_NUM_THREADS"), "3")
    expect_identical(getOption("autozyme.threads"), 9L)  # apply does NOT touch the option
    .az_restore_thread_state(state)
    expect_identical(Sys.getenv("OMP_NUM_THREADS"), "7")
    expect_identical(getOption("autozyme.threads"), 9L)
  })
})

test_that(".az_restore_thread_state unsets env vars that were NA at capture", {
  .w4_use(".az_capture_thread_state", ".az_apply_thread_count",
          ".az_restore_thread_state")
  # Capture with VECLIB unset -> restore must Sys.unsetenv it again.
  Sys.unsetenv("VECLIB_MAXIMUM_THREADS")
  state <- .az_capture_thread_state()
  .az_apply_thread_count(2L)
  expect_identical(Sys.getenv("VECLIB_MAXIMUM_THREADS"), "2")
  .az_restore_thread_state(state)
  expect_identical(Sys.getenv("VECLIB_MAXIMUM_THREADS"), "")  # back to unset
})

test_that(".az_restore_thread_state clears the option when it was NULL at capture", {
  .w4_use(".az_capture_thread_state", ".az_apply_thread_count",
          ".az_restore_thread_state")
  old_opt <- getOption("autozyme.threads")
  on.exit(options(autozyme.threads = old_opt), add = TRUE)
  options(autozyme.threads = NULL)            # captured as NULL
  state <- .az_capture_thread_state()
  options(autozyme.threads = 5L)              # mutate
  .az_restore_thread_state(state)             # NULL branch
  expect_null(getOption("autozyme.threads"))
})

test_that(".az_thread_scope restores state and returns the expr value", {
  .w4_use(".az_thread_scope")
  Sys.setenv(OMP_NUM_THREADS = "6")
  on.exit(Sys.unsetenv("OMP_NUM_THREADS"), add = TRUE)
  val <- .az_thread_scope(2L, 40L + 2L)
  expect_identical(val, 42L)
  expect_identical(Sys.getenv("OMP_NUM_THREADS"), "6")   # restored after scope
})

test_that(".az_thread_enter then .az_thread_exit round-trips", {
  .w4_use(".az_thread_enter", ".az_thread_exit")
  Sys.setenv(MKL_NUM_THREADS = "8")
  on.exit(Sys.unsetenv("MKL_NUM_THREADS"), add = TRUE)
  st <- .az_thread_enter(1L)
  expect_identical(Sys.getenv("MKL_NUM_THREADS"), "1")
  .az_thread_exit(st)
  expect_identical(Sys.getenv("MKL_NUM_THREADS"), "8")
})

test_that("auto_threads default path never returns < 1 (cores-NA-safe)", {
  # auto_threads is exported -> resolves bare in both runners.
  withr::with_envvar(c(AUTOZYME_THREADS = ""), {
    old <- getOption("autozyme.threads"); on.exit(options(autozyme.threads = old), add = TRUE)
    options(autozyme.threads = NULL)
    n <- auto_threads()
    expect_true(is.integer(n) && n >= 1L && n <= 16L)
    # cap below hardware default clamps; cap above default does not raise it
    expect_lte(auto_threads(cap = 1L), 1L)
    expect_gte(auto_threads(cap = 1L), 1L)
    # an invalid (non-coercible) cap is ignored, not an error
    expect_identical(auto_threads(cap = "not-a-number"), auto_threads())
  })
})

# ---------------------------------------------------------------------------
# speedups.R :: umbrella aggregation + .load_speedups_tsv parse edge cases
# (lines 62/72/74/78/93/115/148/153/173 in the source).
# ---------------------------------------------------------------------------

test_that("speedups() returns the empty frame for an umbrella with no sub-steps", {
  # A name with no direct TSV and no `<name>_*` sub-dirs -> empty frame
  # (covers the `!length(subdirs)` return at speedups.R:67 and the empty-df
  # builder). Use a name that is neither a patch nor a sub-step prefix.
  df <- speedups("zzz_no_such_umbrella_xyz")
  expect_equal(nrow(df), 0L)
  expect_true("sub_step" %in% names(df))
})

test_that(".finalize_speedups_df is a no-op on an empty frame", {
  .w4_use(".finalize_speedups_df", ".empty_speedups_df")
  e <- .empty_speedups_df()
  out <- .finalize_speedups_df(e)        # nrow == 0 early return (line 153)
  expect_equal(nrow(out), 0L)
})

test_that(".first_nonempty returns '' when every candidate is NA/empty", {
  .w4_use(".first_nonempty")
  expect_identical(.first_nonempty(NA_character_, "", NA_character_), "")  # fallthrough
  expect_identical(.first_nonempty(NA_character_, "hit", "x"), "hit")
})

test_that(".parse_reps drops NA / blank / non-finite tokens", {
  .w4_use(".parse_reps")
  expect_identical(.parse_reps(NA_character_), numeric(0))
  expect_identical(.parse_reps(""), numeric(0))
  expect_equal(.parse_reps("1.0, 2.0, nope, Inf, 3.0"), c(1, 2, 3))
})

test_that(".load_speedups_tsv returns empty for a header-only TSV", {
  .w4_use(".load_speedups_tsv", ".empty_speedups_df_no_substep")
  hdr <- paste(c("patch", "package_version", "tier", "threads", "platform",
                 "dataset", "variant", "sec_reps", "sec_mean", "mem_mean",
                 "speedup_x_mean", "pass_rate", "status", "n_reps",
                 "ts_first", "ts_last", "metrics_json_median", "fw_versions"),
               collapse = "\t")
  tf <- tempfile(fileext = ".tsv"); on.exit(unlink(tf), add = TRUE)
  writeLines(hdr, tf)
  df <- .load_speedups_tsv(tf)           # nrow(long)==0 -> empty (line 93)
  expect_equal(nrow(df), 0L)
})

test_that(".load_speedups_tsv skips a batch that lacks a baseline OR patched variant", {
  .w4_use(".load_speedups_tsv")
  hdr <- c("patch", "package_version", "tier", "threads", "platform",
           "dataset", "variant", "sec_reps", "sec_mean", "mem_mean",
           "speedup_x_mean", "pass_rate", "status", "n_reps",
           "ts_first", "ts_last", "metrics_json_median", "fw_versions")
  # patched-only batch -> no baseline -> `next` at line 107.
  row <- c("p", "1.0", "tiny", "1", "macOS", "ds", "patched",
           "0.5,0.5", "0.5", "100", "2.0", "1.0", "ok", "2",
           "t0", "t1", "{}", "fw")
  tf <- tempfile(fileext = ".tsv"); on.exit(unlink(tf), add = TRUE)
  writeLines(c(paste(hdr, collapse = "\t"), paste(row, collapse = "\t")), tf)
  df <- .load_speedups_tsv(tf)           # only one variant -> i stays 0 -> empty
  expect_equal(nrow(df), 0L)
})

test_that(".load_speedups_tsv aggregates a baseline+patched pair and flags status!=ok", {
  .w4_use(".load_speedups_tsv")
  hdr <- c("patch", "package_version", "tier", "threads", "platform",
           "dataset", "variant", "sec_reps", "sec_mean", "mem_mean",
           "speedup_x_mean", "pass_rate", "status", "n_reps",
           "ts_first", "ts_last", "metrics_json_median", "fw_versions")
  base <- c("p", "1.0", "tiny", "1", "macOS", "ds", "baseline",
            "1.0,1.0", "1.0", "200", "", "", "ok", "2",
            "t0", "t1", "", "fw")
  patched <- c("p", "1.0", "tiny", "1", "macOS", "ds", "patched",
               "0.5,0.5", "0.5", "100", "2.0", "0.5", "fail", "2",
               "t0", "t1", "{}", "fw")  # status=fail, pass_rate 0.5
  tf <- tempfile(fileext = ".tsv"); on.exit(unlink(tf), add = TRUE)
  writeLines(c(paste(hdr, collapse = "\t"),
               paste(base, collapse = "\t"),
               paste(patched, collapse = "\t")), tf)
  df <- .load_speedups_tsv(tf)
  expect_equal(nrow(df), 1L)
  expect_false(isTRUE(df$all_pass[1]))   # status != ok -> all_pass FALSE (line 115)
  expect_equal(df$speedup_x[1], 2.0)
})

# ---------------------------------------------------------------------------
# verify.R :: pure formatting / comparison / TSV-schema / threshold helpers
# (all reachable without spawning a worker subprocess).
# ---------------------------------------------------------------------------

test_that(".metrics_to_json emits null for NULL and non-finite values", {
  .w4_use(".metrics_to_json")
  expect_identical(.metrics_to_json(list()), "{}")
  m <- list(a = list(value = 1.5), b = list(value = NaN),
            c = list(value = NULL), d = list(value = Inf))
  js <- .metrics_to_json(m)
  expect_match(js, '"a": 1.5')
  expect_match(js, '"b": null')
  expect_match(js, '"c": null')
  expect_match(js, '"d": null')
})

test_that(".fmt_num / .fmt_array blank out NA / non-finite / empty", {
  .w4_use(".fmt_num", ".fmt_array")
  expect_identical(.fmt_num(NULL), "")
  expect_identical(.fmt_num(numeric(0)), "")
  expect_identical(.fmt_num(NA_real_), "")
  expect_identical(.fmt_num(Inf), "")
  expect_identical(.fmt_num(2.5), "2.500000")
  expect_identical(.fmt_array(NULL), "")
  expect_identical(.fmt_array(numeric(0)), "")
  expect_identical(.fmt_array(c(1, 2.5)), "1.000000;2.500000")
})

test_that(".tsv_sanitize collapses tab/CR/LF and maps NULL/NA to ''", {
  .w4_use(".tsv_sanitize")
  expect_identical(.tsv_sanitize(NULL), "")
  expect_identical(.tsv_sanitize(NA), "")
  expect_identical(.tsv_sanitize("a\tb\r\nc"), "a b  c")
  expect_identical(.tsv_sanitize(42), "42")
})

test_that(".effective_threshold covers absolute, floor, lte, gte, and the error", {
  .w4_use(".effective_threshold")
  # deterministic absolute
  abs_th <- .effective_threshold(list(name = "m", threshold = 0.95), "tiny", list())
  expect_identical(abs_th$value, 0.95)
  expect_identical(abs_th$label, "absolute")
  # absolute_floor, no calibration
  fl <- .effective_threshold(list(name = "m", absolute_floor = 0.9,
                                  comparator = "lte"), "tiny", list())
  expect_identical(fl$value, 0.9)
  expect_match(fl$label, "no intrinsic_noise")
  # lte with calibration that relaxes above the floor
  noise <- list(tiny = list(m = 0.1))
  lte <- .effective_threshold(list(name = "m", absolute_floor = 0.05,
                                   comparator = "lte"), "tiny", noise)
  expect_equal(lte$value, max(0.05, 2.0 * 0.1))   # default multiplier 2.0
  # lte where the floor still wins (tiny noise)
  noise_tiny <- list(tiny = list(m = 0.001))
  lte2 <- .effective_threshold(list(name = "m", absolute_floor = 0.5,
                                    comparator = "lte"), "tiny", noise_tiny)
  expect_equal(lte2$value, 0.5)
  # gte branch
  noise_g <- list(tiny = list(m = 0.95))
  gte <- .effective_threshold(list(name = "m", absolute_floor = 0.5,
                                   comparator = "gte"), "tiny", noise_g)
  expect_equal(gte$value, max(0.5, 1 - 2.0 * (1 - 0.95)))
  # custom noise_multiplier respected
  cm <- .effective_threshold(list(name = "m", absolute_floor = 0.0,
                                  comparator = "lte", noise_multiplier = 3.0),
                             "tiny", list(tiny = list(m = 0.1)))
  expect_equal(cm$value, 0.3)
  # neither threshold nor floor -> error
  expect_error(.effective_threshold(list(name = "broken"), "tiny", list()),
               "neither")
})

test_that(".normalize_tsv_header_line strips BOM / CR / surrounding whitespace", {
  .w4_use(".normalize_tsv_header_line", ".header_cols_of_line")
  expect_identical(.normalize_tsv_header_line(""), "")
  expect_identical(.normalize_tsv_header_line(character(0)), "")
  expect_identical(.normalize_tsv_header_line("﻿  a\tb \r"), "a\tb")
  expect_identical(.header_cols_of_line("a\tb\tc"), c("a", "b", "c"))
})

test_that(".is_known_package_verify_header_cols accepts known + forward-compat superset", {
  .w4_use(".is_known_package_verify_header_cols",
          ".package_verify_header_cols",
          ".package_verify_header_cols_legacy",
          ".package_verify_header_cols_pre_pkgver")
  expect_false(.is_known_package_verify_header_cols(character(0)))
  expect_false(.is_known_package_verify_header_cols(c("totally", "wrong")))
  expect_true(.is_known_package_verify_header_cols(.package_verify_header_cols()))
  expect_true(.is_known_package_verify_header_cols(.package_verify_header_cols_legacy()))
  expect_true(.is_known_package_verify_header_cols(.package_verify_header_cols_pre_pkgver()))
  # superset of current schema = forward-compatible -> accepted
  superset <- c(.package_verify_header_cols(), "future_extra_col")
  expect_true(.is_known_package_verify_header_cols(superset))
})

test_that(".assert_long_format_or_empty: ok for known headers, errors on garbage", {
  .w4_use(".assert_long_format_or_empty", ".package_verify_header",
          ".package_verify_header_legacy")
  # missing file
  expect_silent(.assert_long_format_or_empty(tempfile()))
  # empty / header-only-with-blank-first-line file
  blank <- tempfile(); writeLines("", blank); on.exit(unlink(blank), add = TRUE)
  expect_silent(.assert_long_format_or_empty(blank))
  # canonical header
  ok <- tempfile(); writeLines(.package_verify_header, ok); on.exit(unlink(ok), add = TRUE)
  expect_silent(.assert_long_format_or_empty(ok))
  # legacy header
  leg <- tempfile(); writeLines(.package_verify_header_legacy, leg); on.exit(unlink(leg), add = TRUE)
  expect_silent(.assert_long_format_or_empty(leg))
  # superset header (forward compat) -> accepted via cols check
  sup <- tempfile()
  writeLines(paste(.package_verify_header, "extra_col", sep = "\t"), sup)
  on.exit(unlink(sup), add = TRUE)
  expect_silent(.assert_long_format_or_empty(sup))
  # unrecognized header -> stop
  bad <- tempfile(); writeLines("col1\tcol2\tcol3", bad); on.exit(unlink(bad), add = TRUE)
  expect_error(.assert_long_format_or_empty(bad), "unrecognized header")
})

test_that(".row_sort_key orders variant -> tier -> platform and tolerates NA cells", {
  .w4_use(".row_sort_key")
  base_key    <- .row_sort_key(list(variant = "baseline", tier = "tiny",
                                    system_os = "macOS 24", system_threads = "1",
                                    timestamp = "t", rep_idx = "1"))
  patched_key <- .row_sort_key(list(variant = "patched", tier = "tiny",
                                    system_os = "macOS 24", system_threads = "1",
                                    timestamp = "t", rep_idx = "1"))
  expect_lt(base_key, patched_key)            # baseline sorts before patched
  # unknown variant falls to index 2 (after patched=1); platform mapping
  win_key <- .row_sort_key(list(variant = "baseline", tier = "tiny",
                                system_os = "Windows 11", system_threads = "4",
                                timestamp = "t", rep_idx = "1"))
  unk_key <- .row_sort_key(list(variant = "baseline", tier = "tiny",
                                system_os = "Plan9", system_threads = "x",
                                timestamp = "t", rep_idx = "y"))
  expect_match(win_key, "win")
  expect_match(unk_key, "unknown")
  # fully-NA row must not error (the .nz coercion guard)
  na_key <- .row_sort_key(list(variant = NA, tier = NA, system_os = NA,
                               system_threads = NA, timestamp = NA, rep_idx = NA))
  expect_type(na_key, "character")
  # empty system_os -> "mac" default
  empty_os <- .row_sort_key(list(variant = "patched", tier = "medium",
                                 system_os = "", system_threads = "2",
                                 timestamp = "t", rep_idx = "2"))
  expect_match(empty_os, "mac")
})

test_that(".is_subprocess_peak_baseline_note: empty/plain comparable, absorbed/migrated not", {
  .w4_use(".is_subprocess_peak_baseline_note")
  expect_true(.is_subprocess_peak_baseline_note(""))
  expect_true(.is_subprocess_peak_baseline_note(NA_character_))
  expect_true(.is_subprocess_peak_baseline_note("cached baseline 2026-01-01"))
  expect_false(.is_subprocess_peak_baseline_note("absorbed: legacy bench"))
  expect_false(.is_subprocess_peak_baseline_note("MIGRATED: old"))   # case-insensitive
})

test_that(".parse_peak_mb returns NA when no detector or no match, parses on match", {
  .w4_use(".parse_peak_mb")
  expect_identical(.parse_peak_mb("nope", NULL), NA_real_)            # NULL time_info
  expect_identical(.parse_peak_mb(tempfile(), list(pattern = "x")), NA_real_)  # file missing
  ti <- list(pattern = "([0-9]+)\\s+maximum resident set size",
             to_mb = function(x) x / (1024 * 1024))
  f <- tempfile(); writeLines("  2097152  maximum resident set size", f)
  on.exit(unlink(f), add = TRUE)
  expect_equal(.parse_peak_mb(f, ti), 2.0)                            # 2 MiB
  # file present but no pattern match -> NA
  g <- tempfile(); writeLines("no rusage here", g); on.exit(unlink(g), add = TRUE)
  expect_identical(.parse_peak_mb(g, ti), NA_real_)
})

test_that(".collect_system_info returns the four fingerprint fields", {
  .w4_use(".collect_system_info")
  info <- .collect_system_info()
  expect_setequal(names(info),
                  c("system_os", "system_cpu", "system_ram_gb", "system_threads"))
  expect_type(info$system_os, "character")
  expect_true(nzchar(info$system_os))
})

test_that(".collect_system_info threads-field falls through the env-var ladder", {
  .w4_use(".collect_system_info")
  withr::with_envvar(c(ZYME_THREADS = "", AUTOZYME_THREADS = "",
                       OMP_NUM_THREADS = "", MKL_NUM_THREADS = "5",
                       OPENBLAS_NUM_THREADS = ""), {
    expect_identical(.collect_system_info()$system_threads, "5")  # 4th ladder rung
  })
})

test_that(".detect_time_cmd returns NULL or a complete parser spec", {
  .w4_use(".detect_time_cmd")
  ti <- .detect_time_cmd()
  if (!is.null(ti)) {
    expect_true(all(c("kind", "pattern", "to_mb") %in% names(ti)))
    expect_true(is.function(ti$to_mb))
  } else {
    succeed()
  }
})

test_that(".tier_dataset_map / .dataset_name_for_tier read task.yaml pairs", {
  .w4_use(".tier_dataset_map", ".dataset_name_for_tier")
  td <- file.path(tempdir(), paste0("w4_tdmap_", as.integer(runif(1, 1, 1e7))))
  dir.create(td); on.exit(unlink(td, recursive = TRUE), add = TRUE)
  writeLines(c("datasets:",
               "  - tier: tiny",
               "    name: tiny_ds",
               "  - tier: medium",
               "    name: med_ds"),
             file.path(td, "task.yaml"))
  m <- .tier_dataset_map(td)
  expect_identical(m[["tiny"]], "tiny_ds")
  expect_identical(.dataset_name_for_tier(td, "medium"), "med_ds")
  expect_identical(.dataset_name_for_tier(td, "absent_tier"), "")   # not-found branch
})

test_that(".tier_dataset_map is empty for a missing task.yaml", {
  .w4_use(".tier_dataset_map")
  m <- .tier_dataset_map(file.path(tempdir(), "no_such_task_dir_w4"))
  expect_equal(length(m), 0L)
})

test_that(".resolve_smoke prefers attest/smoke.R then falls back to patch$smoke then NULL", {
  .w4_use(".resolve_smoke")
  td <- file.path(tempdir(), paste0("w4_smoke_", as.integer(runif(1, 1, 1e7))))
  dir.create(file.path(td, "attest"), recursive = TRUE)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  # 1. valid smoke.R wins over patch$smoke
  writeLines(c("smoke <- list(",
               "  load = function(task_dir, tier) 1,",
               "  call = function(input) input,",
               "  save = function(result, dir, tier = NULL) invisible())"),
             file.path(td, "attest", "smoke.R"))
  patch_smoke <- list(smoke = list(load = function(...) 99,
                                   call = function(...) 99,
                                   save = function(...) 99))
  out <- .resolve_smoke(td, patch_smoke)
  expect_identical(out$load(NULL, NULL), 1)        # from smoke.R, not patch
  # 2. no smoke.R -> fall back to patch$smoke
  td2 <- file.path(tempdir(), paste0("w4_smoke2_", as.integer(runif(1, 1, 1e7))))
  dir.create(td2); on.exit(unlink(td2, recursive = TRUE), add = TRUE)
  out2 <- .resolve_smoke(td2, patch_smoke)
  expect_identical(out2$load(), 99)
  # 3. neither -> NULL
  expect_null(.resolve_smoke(td2, list(smoke = NULL)))
})

test_that(".resolve_smoke errors on a malformed smoke list in smoke.R", {
  .w4_use(".resolve_smoke")
  td <- file.path(tempdir(), paste0("w4_smokebad_", as.integer(runif(1, 1, 1e7))))
  dir.create(file.path(td, "attest"), recursive = TRUE)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  # defines `smoke` but it is not a list(load, call, save) of functions
  writeLines("smoke <- list(load = 1, call = 2, save = 3)",
             file.path(td, "attest", "smoke.R"))
  expect_error(.resolve_smoke(td, NULL), "list\\(load")
  # defines nothing named smoke
  writeLines("not_smoke <- 1", file.path(td, "attest", "smoke.R"))
  expect_error(.resolve_smoke(td, NULL), "must define")
})

test_that(".append_package_verify_tsv computes speedup_x and peak fold relationally", {
  .w4_use(".append_package_verify_tsv")
  td <- file.path(tempdir(), paste0("w4_pv_", as.integer(runif(1, 1, 1e7))))
  dir.create(td); on.exit(unlink(td, recursive = TRUE), add = TRUE)
  rows <- list(list(
    tier = "tiny",
    baseline_secs = c(2.0, 2.0),
    patched_secs = c(1.0, 1.0),
    baseline_peaks_mb = c(200, 200),
    patched_peaks_mb = c(100, 100),
    per_rep_pass = c(TRUE, TRUE),
    metrics_json = '{"x": 1}'
  ))
  .append_package_verify_tsv(td, "demo_patch", rows)
  df <- utils::read.table(file.path(td, "package_verify.tsv"), header = TRUE,
                          sep = "\t", quote = "", comment.char = "",
                          stringsAsFactors = FALSE, na.strings = c("", "NA"),
                          colClasses = "character")
  patched <- df[df$variant == "patched", , drop = FALSE]
  # baseline median 2.0 / patched 1.0 = 2x
  expect_equal(as.numeric(patched$speedup_x[1]), 2.0, tolerance = 1e-6)
  expect_equal(as.numeric(patched$speedup_pct[1]), 50.0, tolerance = 1e-6)
  # baseline peak 200 / patched 100 = 2x fold
  expect_equal(as.numeric(patched$peak_mb_fold[1]), 2.0, tolerance = 1e-6)
  expect_true(all(df$variant[df$variant != ""] %in% c("baseline", "patched")))
})

test_that(".append_package_verify_tsv uses baseline_sec/peak_for_speedup fallbacks", {
  .w4_use(".append_package_verify_tsv")
  td <- file.path(tempdir(), paste0("w4_pvfb_", as.integer(runif(1, 1, 1e7))))
  dir.create(td); on.exit(unlink(td, recursive = TRUE), add = TRUE)
  # No baseline reps measured; rely on the explicit *_for_speedup fallback.
  rows <- list(list(
    tier = "tiny",
    baseline_secs = numeric(0),
    patched_secs = c(2.0),
    baseline_peaks_mb = numeric(0),
    patched_peaks_mb = c(50),
    per_rep_pass = c(TRUE),
    baseline_sec_for_speedup = 8.0,
    baseline_peak_mb_for_speedup = 200,
    metrics_json = "{}"
  ))
  .append_package_verify_tsv(td, "fb_patch", rows)
  df <- utils::read.table(file.path(td, "package_verify.tsv"), header = TRUE,
                          sep = "\t", quote = "", comment.char = "",
                          stringsAsFactors = FALSE, na.strings = c("", "NA"),
                          colClasses = "character")
  patched <- df[df$variant == "patched", , drop = FALSE]
  expect_equal(as.numeric(patched$speedup_x[1]), 4.0, tolerance = 1e-6)   # 8/2
  expect_equal(as.numeric(patched$peak_mb_fold[1]), 4.0, tolerance = 1e-6) # 200/50
})

test_that(".append_package_verify_tsv writes a sentinel row for a crashed/empty tier", {
  .w4_use(".append_package_verify_tsv")
  td <- file.path(tempdir(), paste0("w4_pvsent_", as.integer(runif(1, 1, 1e7))))
  dir.create(td); on.exit(unlink(td, recursive = TRUE), add = TRUE)
  rows <- list(list(tier = "large", baseline_secs = numeric(0),
                    patched_secs = numeric(0), note = "OOM"))
  .append_package_verify_tsv(td, "sent_patch", rows)
  df <- utils::read.table(file.path(td, "package_verify.tsv"), header = TRUE,
                          sep = "\t", quote = "", comment.char = "",
                          stringsAsFactors = FALSE, na.strings = c("", "NA"),
                          colClasses = "character", fill = TRUE)
  expect_equal(nrow(df), 1L)
  expect_identical(df$tier[1], "large")
  expect_identical(df$note[1], "OOM")
})

test_that(".append_package_verify_tsv preserves+re-sorts pre-existing rows", {
  .w4_use(".append_package_verify_tsv", ".package_verify_header")
  td <- file.path(tempdir(), paste0("w4_pvexist_", as.integer(runif(1, 1, 1e7))))
  dir.create(td); on.exit(unlink(td, recursive = TRUE), add = TRUE)
  # seed an existing canonical-header file with a patched row
  tsv <- file.path(td, "package_verify.tsv")
  seed_cols <- strsplit(.package_verify_header, "\t", fixed = TRUE)[[1]]
  seed <- setNames(rep("", length(seed_cols)), seed_cols)
  seed[["patch_name"]] <- "demo"; seed[["tier"]] <- "tiny"
  seed[["variant"]] <- "patched"; seed[["rep_idx"]] <- "1"; seed[["sec"]] <- "5.0"
  seed[["timestamp"]] <- "2020-01-01T00:00:00"
  writeLines(c(.package_verify_header, paste(seed, collapse = "\t")), tsv)
  # append a new baseline row; baseline must sort ABOVE the existing patched
  .append_package_verify_tsv(td, "demo", list(list(
    tier = "tiny", baseline_secs = c(9.0), patched_secs = numeric(0),
    baseline_peaks_mb = c(123))))
  df <- utils::read.table(tsv, header = TRUE, sep = "\t", quote = "",
                          comment.char = "", stringsAsFactors = FALSE,
                          na.strings = c("", "NA"), colClasses = "character",
                          fill = TRUE)
  expect_true(nrow(df) >= 2L)
  expect_identical(df$variant[1], "baseline")   # baseline rows sort first
})

test_that(".cached_baseline_stats errors clearly without a package_verify.tsv", {
  .w4_use(".cached_baseline_stats")
  td <- file.path(tempdir(), paste0("w4_cbs_", as.integer(runif(1, 1, 1e7))))
  dir.create(td); on.exit(unlink(td, recursive = TRUE), add = TRUE)
  expect_error(.cached_baseline_stats(td, "p", "tiny"), "not found")
})

test_that(".copy_cached_reference_output copies entries and errors on empty source", {
  .w4_use(".copy_cached_reference_output")
  src <- file.path(tempdir(), paste0("w4_src_", as.integer(runif(1, 1, 1e7))))
  dst <- file.path(tempdir(), paste0("w4_dst_", as.integer(runif(1, 1, 1e7))))
  dir.create(src); on.exit(unlink(c(src, dst), recursive = TRUE), add = TRUE)
  # empty source -> error
  expect_error(.copy_cached_reference_output(src, dst), "empty")
  # populated source -> copies
  writeLines("hello", file.path(src, "a.txt"))
  .copy_cached_reference_output(src, dst)
  expect_true(file.exists(file.path(dst, "a.txt")))
})

test_that(".cached_reference_dir resolves both supported layouts, else errors", {
  .w4_use(".cached_reference_dir")
  td <- file.path(tempdir(), paste0("w4_ref_", as.integer(runif(1, 1, 1e7))))
  dir.create(td); on.exit(unlink(td, recursive = TRUE), add = TRUE)
  # neither layout present -> error
  expect_error(.cached_reference_dir(td, "tiny"), "no cached reference")
  # reference_outputs/<tier> layout
  dir.create(file.path(td, "reference_outputs", "tiny"), recursive = TRUE)
  expect_true(dir.exists(.cached_reference_dir(td, "tiny")))
  # reference_output_<tier> layout
  dir.create(file.path(td, "reference_output_medium"))
  expect_true(dir.exists(.cached_reference_dir(td, "medium")))
})

# ---------------------------------------------------------------------------
# core.R :: dispatch / registry / status / conflict / mirror branches that
# wave-1 left untouched. All use a synthetic `tools`-backed patch and clean up
# their own registry slot + binding.
# ---------------------------------------------------------------------------

.w4_clean_patch <- function(pname) {
  ns <- asNamespace("autozyme")
  try(autozyme::deactivate(pname), silent = TRUE)
  if (exists(pname, envir = ns$.zyme_registry)) rm(list = pname, envir = ns$.zyme_registry)
}

test_that("inject_all returns activated/skipped vectors and is callable", {
  # inject_all() probes every shipped upstream; a half-installed upstream whose
  # transitive deps error during namespace LOAD (vs. returning FALSE from the
  # requireNamespace gate) can throw here. That is host state, not a framework
  # bug, so tolerate it and assert the contract only when the call completes.
  out <- tryCatch(inject_all(), error = function(e) e)
  # Use a plain conditional (not skip()) so the test resolves identically under
  # the testthat reporter and under bare covr::file_coverage sourcing.
  if (inherits(out, "condition")) {
    succeed("inject_all() hit a broken upstream namespace load on this host")
  } else {
    expect_setequal(names(out), c("activated", "skipped"))
    expect_type(out$activated, "character")
    expect_type(out$skipped, "character")
    # union covers every shipped patch name
    expect_setequal(c(out$activated, out$skipped), list_patches())
  }
})

test_that("status() reports 'active' for an activated patch and 'inactive' after deactivate", {
  on.exit(.w4_clean_patch("w4_status_demo"), add = TRUE)
  orig <- tools::toTitleCase
  register_patch("w4_status_demo", "tools",
                 list(toTitleCase = function(x) paste0(orig(x), "!")))
  expect_true(activate("w4_status_demo"))
  st <- status()
  expect_identical(st[["w4_status_demo"]], "active")
  deactivate("w4_status_demo")
  expect_identical(status()[["w4_status_demo"]], "inactive")
})

test_that("deactivate_all rebinds every active patch back to its original", {
  on.exit(.w4_clean_patch("w4_deall_demo"), add = TRUE)
  orig <- tools::showNonASCII
  register_patch("w4_deall_demo", "tools",
                 list(showNonASCII = function(x) "patched"))
  activate("w4_deall_demo")
  expect_identical(status()[["w4_deall_demo"]], "active")
  deactivate_all()
  expect_identical(status()[["w4_deall_demo"]], "inactive")
})

test_that(".check_conflicts warns when a known-bad pair is co-active", {
  .w4_use(".check_conflicts", ".zyme_conflicts")
  ns <- asNamespace("autozyme")
  old_conf <- ns$.zyme_conflicts
  # temporarily inject a synthetic conflict entry; restore after.
  unlockBinding(".zyme_conflicts", ns)
  assign(".zyme_conflicts",
         list(list(pair = c("w4_cA", "w4_cB"), reason = "synthetic test conflict")),
         envir = ns)
  on.exit({ assign(".zyme_conflicts", old_conf, envir = ns)
            lockBinding(".zyme_conflicts", ns) }, add = TRUE)
  # both members present in `newly_activating` -> warning fires
  expect_warning(.check_conflicts(c("w4_cA", "w4_cB")), "interact badly")
  # only one member -> no warning
  expect_silent(.check_conflicts(c("w4_cA")))
})

test_that(".did_you_mean suggests one and lists two near-matches", {
  .w4_use(".did_you_mean")
  expect_identical(.did_you_mean("x", character(0)), "")
  one <- .did_you_mean("seruat", c("seurat", "vegan"))
  expect_match(one, "Did you mean 'seurat'")
  # two close matches -> "one of"
  two <- .did_you_mean("mas", c("mast", "mass", "vegan"))
  expect_match(two, "Did you mean one of")
})

test_that(".resolve_activation_target rejects a non-character input", {
  .w4_use(".resolve_activation_target")
  expect_error(.resolve_activation_target(42L), "expects character")
})

test_that("activate(vector) returns a named logical map across mixed patches", {
  on.exit({ .w4_clean_patch("w4_vec_a"); .w4_clean_patch("w4_vec_b") }, add = TRUE)
  o1 <- tools::Rd2txt; o2 <- tools::Rd2HTML
  register_patch("w4_vec_a", "tools", list(Rd2txt = function(...) o1(...)))
  register_patch("w4_vec_b", "tools", list(Rd2HTML = function(...) o2(...)))
  out <- activate(c("w4_vec_a", "w4_vec_b"))
  expect_type(out, "logical")
  expect_setequal(names(out), c("w4_vec_a", "w4_vec_b"))
  expect_true(all(out))
})

test_that("inspect() exposes an s4-kind target view", {
  on.exit(.w4_clean_patch("tools"), add = TRUE)
  # Register under the installed `tools` package so the upstream probe passes,
  # but declare an s4-kind target so the s4 branch of the target view renders.
  register_patch("tools", "tools",
                 list(SomeGeneric = list(kind = "s4",
                                         signature = "ANY",
                                         fn = function(...) NULL)))
  info <- inspect("tools")
  expect_identical(info$targets[[1]]$kind, "s4")
  expect_identical(info$targets[[1]]$signature, "ANY")
})

test_that("inspect() short-circuits 'uninstalled' before sourcing a missing upstream", {
  # A name with no .zyme_upstreams entry and no real package -> uninstalled.
  info <- inspect_or_skip <- tryCatch(
    autozyme::inspect("zzz_definitely_absent_pkg_w4"),
    error = function(e) e)
  if (inherits(info, "condition")) {
    # name isn't a registered patch/subset -> resolve error path
    expect_match(conditionMessage(info), "neither a registered patch")
  } else {
    expect_identical(info$status, "uninstalled")
  }
})

test_that("subset_patches errors on an unknown subset name", {
  expect_error(subset_patches("no_such_subset_w4"), "no subset named")
  expect_identical(subset_patches("scrna_spatial"),
                   c("seurat", "bayesspace", "infercnv", "rctd"))
})

test_that(".rebind mirrors into package:tools when tools is attached", {
  .w4_use(".rebind")
  ns <- asNamespace("tools")
  was_attached <- "package:tools" %in% search()
  if (!was_attached) {
    attachNamespace("tools")
    on.exit(detach("package:tools"), add = TRUE)
  }
  orig <- get("toTitleCase", envir = ns)
  on.exit(.rebind(ns, "toTitleCase", orig), add = TRUE)
  marker <- function(x) "REBIND_MARKER"
  .rebind(ns, "toTitleCase", marker)
  # both the namespace and the attached package: env must see the new binding
  expect_identical(get("toTitleCase", envir = ns)("x"), "REBIND_MARKER")
  if ("package:tools" %in% search()) {
    expect_identical(get("toTitleCase", envir = as.environment("package:tools"))("x"),
                     "REBIND_MARKER")
  }
})

test_that(".emit_activation_marker warns on a version-drift mismatch", {
  .w4_use(".emit_activation_marker")
  # tested_against names tools at an impossible version -> drift WARN string.
  p <- list(name = "drift_demo", upstream = "tools",
            targets = list(file_ext = function(x) x),
            tested_against = "tools 0.0.0-never",
            tested_upstream_versions = NULL)
  withr::with_envvar(c(AUTOZYME_QUIET = ""), {
    msg <- tryCatch(suppressWarnings(
      utils::capture.output(.emit_activation_marker(p), type = "message")),
      error = function(e) "")
    expect_true(any(grepl("activated drift_demo", msg)))
    expect_true(any(grepl("may be unstable", msg)))
  })
})

test_that(".emit_activation_marker is silent under AUTOZYME_QUIET", {
  .w4_use(".emit_activation_marker")
  p <- list(name = "quiet_demo", upstream = "tools",
            targets = list(file_ext = function(x) x), tested_against = NULL)
  withr::with_envvar(c(AUTOZYME_QUIET = "1"), {
    expect_null(.emit_activation_marker(p))
  })
})
