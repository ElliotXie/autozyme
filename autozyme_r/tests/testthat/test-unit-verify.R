# Unit tests for the PURE-R helpers of R/verify.R: JSON/number formatting,
# threshold/equivalence logic, header schema validation, the long-format TSV
# sort key, tier->dataset mapping, the package_verify.tsv append round-trip,
# smoke-recipe resolution, baseline-thread resolution, and the cached-baseline
# / reference-dir lookups.
#
# We deliberately do NOT touch the heavy end-to-end paths (verify_patch,
# .verify_one_tier, .run_worker, .run_evaluate) — those spawn subprocesses and
# need a full task + upstream. The pure helpers below carry the comparison /
# tolerance / equivalence and parsing logic that's worth pinning down.

ns_az <- function() asNamespace("autozyme")

# ---- .metrics_to_json (NA / Inf / NaN -> null) -----------------------------

test_that(".metrics_to_json emits a JSON object and nulls non-finite values", {
  ns <- ns_az()
  j <- ns$.metrics_to_json(list(acc = list(value = 0.95),
                                inf = list(value = Inf),
                                nan = list(value = NaN)))
  expect_match(j, '"acc": 0.95')
  expect_match(j, '"inf": null')
  expect_match(j, '"nan": null')
  expect_identical(ns$.metrics_to_json(list()), "{}")
})

test_that(".metrics_to_json nulls a NULL value entry", {
  ns <- ns_az()
  j <- ns$.metrics_to_json(list(x = list(value = NULL)))
  expect_match(j, '"x": null')
})

# ---- .fmt_num / .fmt_array / .tsv_sanitize ---------------------------------

test_that(".fmt_num formats finite numbers and blanks NA/Inf/empty", {
  ns <- ns_az()
  expect_identical(ns$.fmt_num(1.5), "1.500000")
  expect_identical(ns$.fmt_num(NA_real_), "")
  expect_identical(ns$.fmt_num(Inf), "")
  expect_identical(ns$.fmt_num(-Inf), "")
  expect_identical(ns$.fmt_num(numeric(0)), "")
  expect_identical(ns$.fmt_num(NULL), "")
})

test_that(".fmt_array joins with ';' and blanks empties", {
  ns <- ns_az()
  expect_identical(ns$.fmt_array(c(1.5, 2.5)), "1.500000;2.500000")
  expect_identical(ns$.fmt_array(numeric(0)), "")
  expect_identical(ns$.fmt_array(NULL), "")
})

test_that(".tsv_sanitize replaces tabs/newlines with spaces and NA->''", {
  ns <- ns_az()
  expect_identical(ns$.tsv_sanitize("a\tb\nc\rd"), "a b c d")
  expect_identical(ns$.tsv_sanitize(NA), "")
  expect_identical(ns$.tsv_sanitize(NULL), "")
  expect_identical(ns$.tsv_sanitize(42), "42")
})

# ---- .framework_version ----------------------------------------------------

test_that(".framework_version returns a non-empty version string", {
  ns <- ns_az()
  v <- ns$.framework_version()
  expect_type(v, "character")
  expect_true(nzchar(v))
})

# ---- .effective_threshold (deterministic + stochastic) ---------------------

test_that(".effective_threshold returns a deterministic absolute threshold as-is", {
  ns <- ns_az()
  out <- ns$.effective_threshold(
    list(name = "acc", threshold = 0.9, comparator = "gte"), "tiny", list())
  expect_equal(out$value, 0.9)
  expect_identical(out$label, "absolute")
})

test_that(".effective_threshold falls back to absolute_floor without calibration", {
  ns <- ns_az()
  out <- ns$.effective_threshold(
    list(name = "ari", absolute_floor = 0.8, comparator = "gte"), "tiny", list())
  expect_equal(out$value, 0.8)
  expect_match(out$label, "no intrinsic_noise")
})

test_that(".effective_threshold (lte) relaxes by multiplier*noise above the floor", {
  ns <- ns_az()
  out <- ns$.effective_threshold(
    list(name = "err", absolute_floor = 0.01, comparator = "lte",
         noise_multiplier = 2),
    "tiny", list(tiny = list(err = 0.02)))
  expect_equal(out$value, 0.04)             # max(0.01, 2*0.02)
  expect_match(out$label, "max\\(floor")
})

test_that(".effective_threshold (gte) relaxes toward 1 - mult*(1-noise)", {
  ns <- ns_az()
  out <- ns$.effective_threshold(
    list(name = "ari", absolute_floor = 0.5, comparator = "gte",
         noise_multiplier = 2),
    "tiny", list(tiny = list(ari = 0.9)))
  # relaxed = 1 - 2*(1-0.9) = 0.8; max(0.5, 0.8) = 0.8
  expect_equal(out$value, 0.8)
})

test_that(".effective_threshold (lte) keeps the floor when noise is tiny", {
  ns <- ns_az()
  out <- ns$.effective_threshold(
    list(name = "err", absolute_floor = 0.10, comparator = "lte",
         noise_multiplier = 2),
    "tiny", list(tiny = list(err = 0.001)))
  expect_equal(out$value, 0.10)             # floor wins over 2*0.001
})

test_that(".effective_threshold defaults noise_multiplier to 2.0", {
  ns <- ns_az()
  out <- ns$.effective_threshold(
    list(name = "err", absolute_floor = 0, comparator = "lte"),
    "tiny", list(tiny = list(err = 0.03)))
  expect_equal(out$value, 0.06)             # 2.0 * 0.03
})

test_that(".effective_threshold errors when neither threshold nor floor given", {
  ns <- ns_az()
  expect_error(
    ns$.effective_threshold(list(name = "x", comparator = "gte"), "tiny", list()),
    "neither")
})

# ---- header schema constants + validation ----------------------------------

test_that("package_verify header has 21 columns and the key fields", {
  ns <- ns_az()
  cols <- ns$.package_verify_header_cols()
  expect_length(cols, 21L)
  expect_true(all(c("timestamp", "patch_name", "tier", "dataset", "variant",
                    "sec", "speedup_x", "pass", "package_version") %in% cols))
})

test_that("legacy / pre-pkgver headers are recognized as known", {
  ns <- ns_az()
  expect_true(ns$.is_known_package_verify_header_cols(ns$.package_verify_header_cols()))
  expect_true(ns$.is_known_package_verify_header_cols(ns$.package_verify_header_cols_legacy()))
  expect_true(ns$.is_known_package_verify_header_cols(ns$.package_verify_header_cols_pre_pkgver()))
})

test_that(".is_known_package_verify_header_cols accepts a forward-compat superset", {
  ns <- ns_az()
  superset <- c(ns$.package_verify_header_cols(), "future_col")
  expect_true(ns$.is_known_package_verify_header_cols(superset))
})

test_that(".is_known_package_verify_header_cols rejects garbage / empty", {
  ns <- ns_az()
  expect_false(ns$.is_known_package_verify_header_cols(c("a", "b", "c")))
  expect_false(ns$.is_known_package_verify_header_cols(character(0)))
})

test_that(".normalize_tsv_header_line strips BOM, CR, and surrounding space", {
  ns <- ns_az()
  expect_identical(ns$.normalize_tsv_header_line("﻿a\tb\r"), "a\tb")
  expect_identical(ns$.normalize_tsv_header_line("  x\ty  "), "x\ty")
  expect_identical(ns$.normalize_tsv_header_line(""), "")
})

test_that(".header_cols_of_line splits a normalized header into columns", {
  ns <- ns_az()
  expect_identical(ns$.header_cols_of_line("﻿a\tb\tc\r"), c("a", "b", "c"))
})

test_that(".assert_long_format_or_empty: ok for canonical, missing, header-only", {
  ns <- ns_az()
  td <- tempfile("vassert_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)

  expect_null(ns$.assert_long_format_or_empty(file.path(td, "absent.tsv")))

  canon <- file.path(td, "good.tsv")
  writeLines(ns$.package_verify_header, canon)
  expect_null(ns$.assert_long_format_or_empty(canon))

  empty <- file.path(td, "empty.tsv")
  file.create(empty)
  expect_null(ns$.assert_long_format_or_empty(empty))
})

test_that(".assert_long_format_or_empty errors on an unrecognized header", {
  ns <- ns_az()
  td <- tempfile("vassert2_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  bad <- file.path(td, "bad.tsv")
  writeLines("col1\tcol2\tcol3", bad)
  expect_error(ns$.assert_long_format_or_empty(bad), "unrecognized header")
})

# ---- .row_sort_key ---------------------------------------------------------

test_that(".row_sort_key orders baseline before patched within a tier", {
  ns <- ns_az()
  base <- ns$.row_sort_key(list(variant = "baseline", tier = "tiny",
                                system_os = "macOS 14", system_threads = "1",
                                timestamp = "2026-01-01", rep_idx = "1"))
  patched <- ns$.row_sort_key(list(variant = "patched", tier = "tiny",
                                   system_os = "macOS 14", system_threads = "1",
                                   timestamp = "2026-01-01", rep_idx = "1"))
  expect_lt(base, patched)
})

test_that(".row_sort_key orders tiers tiny < medium < large", {
  ns <- ns_az()
  mk <- function(tier) ns$.row_sort_key(list(variant = "baseline", tier = tier,
                                             system_os = "macOS", system_threads = "1",
                                             timestamp = "t", rep_idx = "1"))
  expect_lt(mk("tiny"), mk("medium"))
  expect_lt(mk("medium"), mk("large"))
})

test_that(".row_sort_key tolerates NA / missing cells", {
  ns <- ns_az()
  k <- ns$.row_sort_key(list(variant = NA, tier = NA, system_os = NA,
                             system_threads = NA, timestamp = NA, rep_idx = NA))
  expect_type(k, "character")
  expect_true(nzchar(k))
})

test_that(".row_sort_key maps Windows/macOS/unknown platform prefixes", {
  ns <- ns_az()
  win <- ns$.row_sort_key(list(variant = "baseline", tier = "tiny",
                               system_os = "Windows 11", system_threads = "1",
                               timestamp = "t", rep_idx = "1"))
  mac <- ns$.row_sort_key(list(variant = "baseline", tier = "tiny",
                               system_os = "macOS 14", system_threads = "1",
                               timestamp = "t", rep_idx = "1"))
  expect_match(win, "win")
  expect_match(mac, "mac")
})

# ---- .is_subprocess_peak_baseline_note -------------------------------------

test_that(".is_subprocess_peak_baseline_note: empty/plain TRUE, absorbed/migrated FALSE", {
  ns <- ns_az()
  expect_true(ns$.is_subprocess_peak_baseline_note(""))
  expect_true(ns$.is_subprocess_peak_baseline_note("cached baseline 2026"))
  expect_false(ns$.is_subprocess_peak_baseline_note("absorbed: legacy bench"))
  expect_false(ns$.is_subprocess_peak_baseline_note("migrated: old"))
})

# ---- .tier_dataset_map / .dataset_name_for_tier ----------------------------

test_that(".tier_dataset_map reads tier->name pairs from task.yaml", {
  skip_if_not_installed("yaml")
  ns <- ns_az()
  td <- tempfile("vmap_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  writeLines(c("datasets:",
               "  - tier: tiny",
               "    name: ds_tiny",
               "  - tier: medium",
               "    name: ds_med"),
             file.path(td, "task.yaml"))
  m <- ns$.tier_dataset_map(td)
  expect_identical(m[["tiny"]], "ds_tiny")
  expect_identical(m[["medium"]], "ds_med")
  expect_identical(ns$.dataset_name_for_tier(td, "tiny"), "ds_tiny")
  expect_identical(ns$.dataset_name_for_tier(td, "absent"), "")
})

test_that(".tier_dataset_map is empty when task.yaml is missing", {
  ns <- ns_az()
  td <- tempfile("vmap2_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  expect_length(ns$.tier_dataset_map(td), 0L)
})

# ---- .resolve_baseline_threads ---------------------------------------------

test_that(".resolve_baseline_threads honors ZYME_THREADS / AUTOZYME_THREADS env", {
  ns <- ns_az()
  td <- tempfile("vthreads_"); dir.create(td)
  old <- Sys.getenv(c("ZYME_THREADS", "AUTOZYME_THREADS"), unset = NA_character_)
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
    unlink(td, recursive = TRUE)
  }, add = TRUE)

  Sys.setenv(ZYME_THREADS = "4")
  expect_identical(ns$.resolve_baseline_threads(td), 4L)
  Sys.unsetenv("ZYME_THREADS")
  Sys.setenv(AUTOZYME_THREADS = "3")
  expect_identical(ns$.resolve_baseline_threads(td), 3L)
})

test_that(".resolve_baseline_threads defaults to 1 with no env and no task.yaml", {
  ns <- ns_az()
  td <- tempfile("vthreads2_"); dir.create(td)
  old <- Sys.getenv(c("ZYME_THREADS", "AUTOZYME_THREADS"), unset = NA_character_)
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
    unlink(td, recursive = TRUE)
  }, add = TRUE)
  Sys.unsetenv(c("ZYME_THREADS", "AUTOZYME_THREADS"))
  expect_identical(ns$.resolve_baseline_threads(td), 1L)
})

test_that(".resolve_baseline_threads reads baseline_threads[1] from task.yaml", {
  skip_if_not_installed("yaml")
  ns <- ns_az()
  td <- tempfile("vthreads3_"); dir.create(td)
  old <- Sys.getenv(c("ZYME_THREADS", "AUTOZYME_THREADS"), unset = NA_character_)
  on.exit({
    for (nm in names(old)) {
      if (is.na(old[[nm]])) Sys.unsetenv(nm)
      else do.call(Sys.setenv, setNames(list(old[[nm]]), nm))
    }
    unlink(td, recursive = TRUE)
  }, add = TRUE)
  Sys.unsetenv(c("ZYME_THREADS", "AUTOZYME_THREADS"))
  writeLines(c("baseline_threads:", "  - 2", "  - 8"), file.path(td, "task.yaml"))
  expect_identical(ns$.resolve_baseline_threads(td), 2L)
})

# ---- .resolve_smoke --------------------------------------------------------

test_that(".resolve_smoke loads attest/smoke.R when present", {
  ns <- ns_az()
  td <- tempfile("vsmoke_"); dir.create(file.path(td, "attest"), recursive = TRUE)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  writeLines(
    "smoke <- list(load = function(...) 1, call = function(...) 2, save = function(...) 3)",
    file.path(td, "attest", "smoke.R"))
  sm <- ns$.resolve_smoke(td, NULL)
  expect_true(is.list(sm))
  expect_true(all(vapply(sm[c("load", "call", "save")], is.function, logical(1))))
})

test_that(".resolve_smoke falls back to the patch's registered smoke", {
  ns <- ns_az()
  td <- tempfile("vsmoke2_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  patch <- list(smoke = list(load = function(...) 1, call = function(...) 2,
                             save = function(...) 3))
  sm <- ns$.resolve_smoke(td, patch)
  expect_identical(sm, patch$smoke)
})

test_that(".resolve_smoke returns NULL when neither file nor patch smoke exists", {
  ns <- ns_az()
  td <- tempfile("vsmoke3_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  expect_null(ns$.resolve_smoke(td, list(smoke = NULL)))
})

test_that(".resolve_smoke errors when smoke.R defines a malformed `smoke`", {
  ns <- ns_az()
  td <- tempfile("vsmoke4_"); dir.create(file.path(td, "attest"), recursive = TRUE)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  writeLines("smoke <- list(load = function(...) 1)",  # missing call/save
             file.path(td, "attest", "smoke.R"))
  expect_error(ns$.resolve_smoke(td, NULL), "list\\(load")
})

# ---- system-info collectors (best-effort, must not error) ------------------

test_that(".collect_system_info returns the four documented fields", {
  ns <- ns_az()
  info <- ns$.collect_system_info()
  expect_setequal(names(info),
                  c("system_os", "system_cpu", "system_ram_gb", "system_threads"))
  expect_type(info$system_os, "character")
  expect_true(nzchar(info$system_os))       # OS string always present
})

test_that(".detect_cpu_model and .detect_ram_gb degrade gracefully", {
  ns <- ns_az()
  cpu <- ns$.detect_cpu_model()
  expect_type(cpu, "character")
  expect_length(cpu, 1L)
  ram <- ns$.detect_ram_gb()
  expect_true(is.na(ram) || (is.numeric(ram) && ram > 0))
})

test_that(".detect_time_cmd returns NULL or a parser spec; .parse_peak_mb is NA-safe", {
  ns <- ns_az()
  ti <- ns$.detect_time_cmd()
  expect_true(is.null(ti) || is.list(ti))
  # NA-safe: nonexistent stats file -> NA_real_
  expect_true(is.na(ns$.parse_peak_mb(tempfile(), ti)))
})

# ---- package_verify.tsv append round-trip ----------------------------------

test_that(".append_one_pv_row writes a readable canonical-header TSV row", {
  skip_if_not_installed("yaml")
  ns <- ns_az()
  td <- tempfile("vpv_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  writeLines(c("datasets:", "  - tier: tiny", "    name: ds_tiny"),
             file.path(td, "task.yaml"))

  ns$.append_one_pv_row(td, "demo", "tiny", "baseline", 1, sec = 2.0, peak_mb = 100)
  ns$.append_one_pv_row(td, "demo", "tiny", "patched", 1, sec = 1.0,
                        peak_mb = 50, speedup_x = 2.0)

  pv <- file.path(td, "package_verify.tsv")
  expect_true(file.exists(pv))
  df <- utils::read.table(pv, header = TRUE, sep = "\t", quote = "",
                          comment.char = "", stringsAsFactors = FALSE,
                          na.strings = c("", "NA"))
  expect_equal(nrow(df), 2L)
  expect_equal(ncol(df), 21L)
  # baseline sorts above patched.
  expect_identical(df$variant, c("baseline", "patched"))
  # dataset backfilled from task.yaml tier map.
  expect_true(all(df$dataset == "ds_tiny"))
  expect_identical(df$patch_name, c("demo", "demo"))
})

test_that(".append_package_verify_tsv writes a sentinel row for a skipped tier", {
  skip_if_not_installed("yaml")
  ns <- ns_az()
  td <- tempfile("vpv2_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  writeLines(c("datasets:", "  - tier: tiny", "    name: ds_tiny"),
             file.path(td, "task.yaml"))

  rows <- list(list(tier = "tiny", baseline_secs = numeric(0),
                    patched_secs = numeric(0), per_rep_pass = logical(0),
                    metrics_json = "", note = "missing dataset"))
  ns$.append_package_verify_tsv(td, "demo", rows)

  pv <- file.path(td, "package_verify.tsv")
  expect_true(file.exists(pv))
  df <- utils::read.table(pv, header = TRUE, sep = "\t", quote = "",
                          comment.char = "", stringsAsFactors = FALSE,
                          na.strings = c("", "NA"), fill = TRUE)
  expect_equal(nrow(df), 1L)
  expect_identical(df$note, "missing dataset")
})

# ---- cached reference / baseline lookups -----------------------------------

test_that(".cached_reference_dir resolves either supported layout, else errors", {
  ns <- ns_az()
  td <- tempfile("vref_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)

  expect_error(ns$.cached_reference_dir(td, "tiny"),
               "no cached reference output")

  dir.create(file.path(td, "reference_outputs", "tiny"), recursive = TRUE)
  got <- ns$.cached_reference_dir(td, "tiny")
  expect_true(dir.exists(got))
})

test_that(".cached_baseline_stats errors when package_verify.tsv is absent", {
  ns <- ns_az()
  td <- tempfile("vbase_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  expect_error(ns$.cached_baseline_stats(td, "demo", "tiny"),
               "package_verify.tsv not found")
})

test_that(".copy_cached_reference_output errors on an empty source dir", {
  ns <- ns_az()
  src <- tempfile("vsrc_"); dir.create(src)
  dst <- tempfile("vdst_")
  on.exit({ unlink(src, recursive = TRUE); unlink(dst, recursive = TRUE) }, add = TRUE)
  expect_error(ns$.copy_cached_reference_output(src, dst), "empty")
})

test_that(".copy_cached_reference_output copies files into the destination", {
  ns <- ns_az()
  src <- tempfile("vsrc2_"); dir.create(src)
  dst <- tempfile("vdst2_")
  on.exit({ unlink(src, recursive = TRUE); unlink(dst, recursive = TRUE) }, add = TRUE)
  writeLines("hello", file.path(src, "out.txt"))
  ns$.copy_cached_reference_output(src, dst)
  expect_true(file.exists(file.path(dst, "out.txt")))
})
