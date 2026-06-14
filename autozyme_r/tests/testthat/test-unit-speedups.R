# Unit tests for R/speedups.R: the public speedups() reader against bundled
# finalized TSVs (data-frame shape + direct/umbrella lookup + empty fallback),
# plus the internal long->summary aggregation parser fed synthetic TSV files.

EXPECTED_COLS <- c(
  "timestamp", "patch_name", "tier", "dataset", "reps", "baseline_sec",
  "patched_sec", "speedup_x", "all_pass", "baseline_secs", "patched_secs",
  "baseline_peak_mb", "patched_peak_mb", "metrics_json", "framework_version",
  "note", "system_os", "system_cpu", "system_ram_gb", "system_threads",
  "sub_step")

# ---- input validation ------------------------------------------------------

test_that("speedups() validates its name argument", {
  expect_error(speedups(123))
  expect_error(speedups(c("a", "b")))
  expect_error(speedups(""))
})

# ---- empty / missing fallback ----------------------------------------------

test_that("speedups() returns the canonical empty frame for an unknown patch", {
  df <- speedups("definitely_no_such_patch_xyz")
  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 0L)
  expect_identical(names(df), EXPECTED_COLS)
})

test_that(".empty_speedups_df has the documented columns and types", {
  ns <- asNamespace("autozyme")
  df <- ns$.empty_speedups_df()
  expect_identical(names(df), EXPECTED_COLS)
  expect_type(df$reps, "integer")
  expect_type(df$baseline_sec, "double")
  expect_type(df$all_pass, "logical")
  expect_identical(names(ns$.empty_speedups_df_no_substep()),
                   setdiff(EXPECTED_COLS, "sub_step"))
})

# ---- bundled direct lookup -------------------------------------------------

test_that("speedups() on a bundled patch returns rows with the full schema", {
  # vegan ships a direct speedups_finalized.tsv in the installed package.
  skip_if(!nzchar(system.file("patches", "vegan", "speedups_finalized.tsv",
                              package = "autozyme")),
          "vegan finalized TSV not bundled")
  df <- speedups("vegan")
  expect_s3_class(df, "data.frame")
  expect_gt(nrow(df), 0L)
  expect_identical(names(df), EXPECTED_COLS)
  # Direct hit => sub_step is empty for every row.
  expect_true(all(df$sub_step == ""))
  expect_true(all(df$patch_name == "vegan" | is.na(df$patch_name)))
})

# ---- bundled umbrella lookup -----------------------------------------------

test_that("speedups('seurat') aggregates sub-steps and tags sub_step", {
  root <- system.file("patches", package = "autozyme")
  has_seurat_substeps <- length(grep("^seurat_",
    list.dirs(root, full.names = FALSE, recursive = FALSE))) > 0L
  skip_if(!has_seurat_substeps, "no bundled seurat_* sub-step dirs")

  df <- speedups("seurat")
  expect_s3_class(df, "data.frame")
  expect_gt(nrow(df), 0L)
  expect_identical(names(df), EXPECTED_COLS)
  # Umbrella rows carry a non-empty sub_step derived from the dir name.
  expect_true(any(nzchar(df$sub_step)))
})

# ---- .parse_reps -----------------------------------------------------------

test_that(".parse_reps splits comma lists and drops non-finite", {
  ns <- asNamespace("autozyme")
  expect_equal(ns$.parse_reps("1.5, 2.0 ,3"), c(1.5, 2.0, 3))
  expect_equal(ns$.parse_reps("4"), 4)
  expect_equal(ns$.parse_reps(NA_character_), numeric(0))
  expect_equal(ns$.parse_reps(""), numeric(0))
  # garbage tokens become NA -> dropped
  expect_equal(ns$.parse_reps("1, x, 3"), c(1, 3))
})

# ---- .first_nonempty -------------------------------------------------------

test_that(".first_nonempty returns the first non-NA non-empty value", {
  ns <- asNamespace("autozyme")
  expect_identical(ns$.first_nonempty(NA, "", "hit", "later"), "hit")
  expect_identical(ns$.first_nonempty("a", "b"), "a")
  expect_identical(ns$.first_nonempty(NA_character_, ""), "")
})

# ---- .finalize_speedups_df -------------------------------------------------

test_that(".finalize_speedups_df sorts by tier/os/threads/dataset and resets rownames", {
  ns <- asNamespace("autozyme")
  df <- data.frame(
    tier = c("medium", "tiny"), system_os = c("macOS", "macOS"),
    system_threads = c(1L, 1L), dataset = c("d2", "d1"),
    sub_step = c("", ""), timestamp = c("t2", "t1"),
    stringsAsFactors = FALSE)
  out <- ns$.finalize_speedups_df(df)
  # "medium" sorts after... actually alphabetical: medium < tiny, so order by
  # tier string puts "medium" first; assert it's deterministic + rownames reset.
  expect_equal(rownames(out), as.character(seq_len(nrow(out))))
  expect_equal(out$tier, sort(df$tier))
})

# ---- .load_speedups_tsv on a synthetic long-format TSV ---------------------

.write_long_tsv <- function(path) {
  hdr <- c("patch", "package_version", "tier", "threads", "platform", "dataset",
           "variant", "sec_reps", "sec_mean", "speedup_x_mean", "pass_rate",
           "status", "n_reps", "mem_mean", "metrics_json_median", "fw_versions",
           "ts_first", "ts_last")
  rows <- list(
    # baseline + patched for one (patch, tier) batch
    c("demo", "demo 1.0", "tiny", "1", "macOS", "ds1",
      "baseline", "2.0,2.2", "2.1", "", "", "ok", "2", "100", "{}", "fw1",
      "2026-01-01T00:00:00", "2026-01-01T00:00:01"),
    c("demo", "demo 1.0", "tiny", "1", "macOS", "ds1",
      "patched", "1.0,1.1", "1.05", "2.0", "1", "ok", "2", "50", "{\"x\":1}", "fw1",
      "2026-01-01T00:00:02", "2026-01-01T00:00:03")
  )
  con <- file(path, open = "w")
  writeLines(paste(hdr, collapse = "\t"), con)
  for (r in rows) writeLines(paste(r, collapse = "\t"), con)
  close(con)
}

test_that(".load_speedups_tsv aggregates baseline+patched into one summary row", {
  ns <- asNamespace("autozyme")
  tsv <- tempfile(fileext = ".tsv")
  on.exit(unlink(tsv), add = TRUE)
  .write_long_tsv(tsv)

  df <- ns$.load_speedups_tsv(tsv)
  expect_equal(nrow(df), 1L)
  expect_identical(df$patch_name, "demo")
  expect_identical(df$tier, "tiny")
  expect_identical(df$dataset, "ds1")
  expect_equal(df$reps, 2L)
  expect_equal(df$baseline_sec, 2.1)
  expect_equal(df$patched_sec, 1.05)
  expect_equal(df$speedup_x, 2.0)
  expect_true(df$all_pass)                       # pass_rate >= 1 + status ok
  # Raw rep arrays preserved as ";"-joined strings.
  expect_match(df$baseline_secs, "2.0")
  expect_match(df$patched_secs, "1.1")
})

test_that(".load_speedups_tsv marks all_pass FALSE when status != ok", {
  ns <- asNamespace("autozyme")
  tsv <- tempfile(fileext = ".tsv")
  on.exit(unlink(tsv), add = TRUE)
  hdr <- c("patch", "package_version", "tier", "threads", "platform", "dataset",
           "variant", "sec_reps", "sec_mean", "speedup_x_mean", "pass_rate",
           "status", "n_reps", "mem_mean", "metrics_json_median", "fw_versions",
           "ts_first", "ts_last")
  con <- file(tsv, open = "w")
  writeLines(paste(hdr, collapse = "\t"), con)
  writeLines(paste(c("demo", "v", "tiny", "1", "macOS", "ds", "baseline",
                     "2.0", "2.0", "", "", "ok", "1", "10", "{}", "fw",
                     "t", "t"), collapse = "\t"), con)
  writeLines(paste(c("demo", "v", "tiny", "1", "macOS", "ds", "patched",
                     "1.0", "1.0", "2.0", "1", "fail", "1", "5", "{}", "fw",
                     "t", "t"), collapse = "\t"), con)
  close(con)
  df <- ns$.load_speedups_tsv(tsv)
  expect_equal(nrow(df), 1L)
  expect_false(df$all_pass)
})

test_that(".load_speedups_tsv returns empty (no substep) for a header-only TSV", {
  ns <- asNamespace("autozyme")
  tsv <- tempfile(fileext = ".tsv")
  on.exit(unlink(tsv), add = TRUE)
  hdr <- c("patch", "package_version", "tier", "threads", "platform", "dataset",
           "variant", "sec_reps", "sec_mean", "speedup_x_mean", "pass_rate",
           "status", "n_reps", "mem_mean", "metrics_json_median", "fw_versions",
           "ts_first", "ts_last")
  writeLines(paste(hdr, collapse = "\t"), tsv)
  df <- ns$.load_speedups_tsv(tsv)
  expect_equal(nrow(df), 0L)
  expect_identical(names(df), setdiff(EXPECTED_COLS, "sub_step"))
})

test_that(".load_speedups_tsv skips a batch with only one variant", {
  ns <- asNamespace("autozyme")
  tsv <- tempfile(fileext = ".tsv")
  on.exit(unlink(tsv), add = TRUE)
  hdr <- c("patch", "package_version", "tier", "threads", "platform", "dataset",
           "variant", "sec_reps", "sec_mean", "speedup_x_mean", "pass_rate",
           "status", "n_reps", "mem_mean", "metrics_json_median", "fw_versions",
           "ts_first", "ts_last")
  con <- file(tsv, open = "w")
  writeLines(paste(hdr, collapse = "\t"), con)
  # baseline only -- no patched partner; batch is dropped.
  writeLines(paste(c("demo", "v", "tiny", "1", "macOS", "ds", "baseline",
                     "2.0", "2.0", "", "", "ok", "1", "10", "{}", "fw",
                     "t", "t"), collapse = "\t"), con)
  close(con)
  df <- ns$.load_speedups_tsv(tsv)
  expect_equal(nrow(df), 0L)
})
