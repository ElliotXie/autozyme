#!/usr/bin/env Rscript
# Aggregate raw speedups.tsv -> speedups_finalized.tsv per patch.
# Rows: tier x threads x platform x variant
# Cells: comma-separated per-rep values + mean/median
#
# Usage:
#   Rscript scripts/finalize_speedups.R            # all patches
#   Rscript scripts/finalize_speedups.R bayesspace # one patch

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(jsonlite)
})

PATCHES_DIR <- "inst/patches"

fmt_num <- function(x, digits = 3) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return("")
  paste(formatC(x, format = "f", digits = digits, drop0trailing = FALSE), collapse = ", ")
}

aggregate_metrics_json <- function(vals) {
  vals <- vals[!is.na(vals)]
  vals <- vals[nzchar(vals) & toupper(vals) != "NA"]
  if (length(vals) == 0) return("")
  by_key <- list()
  for (s in vals) {
    parsed <- tryCatch(fromJSON(s, simplifyVector = FALSE),
                       error = function(e) NULL)
    if (is.null(parsed) || !is.list(parsed)) next
    for (k in names(parsed)) {
      v <- parsed[[k]]
      if (is.null(v) || length(v) != 1) next
      vn <- suppressWarnings(as.numeric(v))
      if (is.na(vn) || !is.finite(vn)) next
      by_key[[k]] <- c(by_key[[k]], vn)
    }
  }
  if (length(by_key) == 0) return("")
  medians <- lapply(by_key[sort(names(by_key))], function(x) round(median(x), 6))
  as.character(toJSON(medians, auto_unbox = TRUE, digits = NA))
}

classify_platform <- function(os) {
  ifelse(grepl("macOS|Darwin|Apple", os, ignore.case = TRUE), "macOS",
  ifelse(grepl("Windows|Win", os, ignore.case = TRUE), "Windows",
  ifelse(grepl("Linux", os, ignore.case = TRUE), "Linux",
         "unknown")))
}

# Resolve the upstream package version that was tested. Priority:
# 1. manifest.yml compatibility_target in patch dir
# 2. manifest.yml compatibility_target in parent (e.g. seurat_normalize -> seurat/manifest.yml)
# 3. patch.R `tested_against = "<Pkg> <ver>"`
get_package_version <- function(patch_dir) {
  read_target <- function(mf) {
    if (!file.exists(mf)) return(NA_character_)
    lines <- readLines(mf, warn = FALSE)
    m <- regmatches(lines, regexpr("compatibility_target:\\s*\\K.+", lines, perl = TRUE))
    if (length(m) == 0) return(NA_character_)
    trimws(m[1])
  }
  v <- read_target(file.path(patch_dir, "manifest.yml"))
  if (!is.na(v)) return(v)
  pkg <- basename(patch_dir)
  if (grepl("^seurat_", pkg)) {
    v <- read_target(file.path(dirname(patch_dir), "seurat", "manifest.yml"))
    if (!is.na(v)) return(v)
  }
  pf <- file.path(patch_dir, "patch.R")
  if (file.exists(pf)) {
    lines <- readLines(pf, warn = FALSE)
    m <- regmatches(lines, regexpr('tested_against\\s*=\\s*"([^"]+)"', lines, perl = TRUE))
    if (length(m) > 0) {
      v <- sub('.*tested_against\\s*=\\s*"([^"]+)".*', "\\1", m[1])
      return(v)
    }
  }
  NA_character_
}

# Packages where pre-2026-05-17 blank-platform rows were confirmed to be Mac
# (medium/patched timings match existing Mac data within ~10%, far from Win times)
INFERRED_MAC_PACKAGES <- c("fgsea", "maftools", "rctd", "slingshot", "tradeseq", "vegan")

# The per-task threading rubric (paper/rubric/task_threading.yaml) is the SHARED
# single source of truth for the scan and both finalizers. .REDUCED maps each
# task to the (platform, variant) cells to reduce to t1: schedule==single reduces
# BOTH variants on BOTH platforms; reduce_t1 lists per-platform variants. The
# finalizer then pools per-thread reps within +/-.TOL of t1 onto t1 (more reps;
# the +/- noise is double-sided and cancels) and DROPS threads farther off (fork
# / thread overhead). Reports the operating point (t1) robustly.
.load_reduced <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grepl("^--file=", a)])
  sdir <- if (length(f)) dirname(normalizePath(f, mustWork = FALSE)) else getwd()
  cands <- c(file.path(sdir, "..", "..", "paper", "rubric", "task_threading.yaml"),
             file.path("..", "paper", "rubric", "task_threading.yaml"),
             file.path("paper", "rubric", "task_threading.yaml"))
  yp <- c(cands[file.exists(cands)], cands[1])[1]
  out <- list()
  tryCatch({
    tasks <- yaml::read_yaml(yp)[["tasks"]]
    for (name in names(tasks)) {
      e <- tasks[[name]]
      m <- list(mac = character(0), win = character(0))
      if (identical(e[["schedule"]], "single"))
        m <- list(mac = c("baseline", "patched"), win = c("baseline", "patched"))
      rt <- e[["reduce_t1"]]
      if (!is.null(rt)) for (plat in names(rt))
        m[[plat]] <- union(m[[plat]], as.character(rt[[plat]]))
      out[[name]] <- m
    }
  }, error = function(e) message(sprintf(
    "[finalize] WARN: task_threading.yaml unreadable (%s); no reduction",
    conditionMessage(e))))
  out
}
.REDUCED <- .load_reduced()

# thread_scaling_flat (paper/rubric/task_threading.yaml): (task -> {mac,win} of
# variants) whose thread_scaling LABEL is forced to "flat" despite being kept
# per-thread (empirically thread-flat). Label-only: does NOT reduce or drop rows.
.load_flat_label <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grepl("^--file=", a)])
  sdir <- if (length(f)) dirname(normalizePath(f, mustWork = FALSE)) else getwd()
  cands <- c(file.path(sdir, "..", "..", "paper", "rubric", "task_threading.yaml"),
             file.path("..", "paper", "rubric", "task_threading.yaml"),
             file.path("paper", "rubric", "task_threading.yaml"))
  yp <- c(cands[file.exists(cands)], cands[1])[1]
  out <- list()
  tryCatch({
    fl <- yaml::read_yaml(yp)[["thread_scaling_flat"]]
    for (name in names(fl)) {
      e <- fl[[name]]; m <- list(mac = character(0), win = character(0))
      for (plat in names(e)) m[[plat]] <- as.character(e[[plat]])
      out[[name]] <- m
    }
  }, error = function(e) message(sprintf(
    "[finalize] WARN: thread_scaling_flat unreadable (%s)", conditionMessage(e))))
  out
}
.FLAT_LABEL <- .load_flat_label()
.TOL <- 0.10  # +/-10%: pool a t>1 thread onto t1 if within this band, else drop

finalize_one <- function(patch_dir) {
  # Per-platform split: each machine writes its own speedups.<plat>.tsv
  # (mac/win/other) so cross-machine git merges stay disjoint. Read and bind
  # all shards. Backward-compatible: fall back to legacy combined speedups.tsv.
  shards <- file.path(patch_dir, paste0("speedups.", c("mac", "win", "other", "linux"), ".tsv"))
  shards <- shards[file.exists(shards)]
  if (length(shards) == 0) {
    legacy <- file.path(patch_dir, "speedups.tsv")
    if (file.exists(legacy)) shards <- legacy
  }
  if (length(shards) == 0) return(invisible(NULL))

  # Read every shard as all-character first so per-shard type inference can't
  # diverge (e.g. one shard guesses `timestamp` datetime, another character ->
  # bind_rows refuses). type_convert then re-infers types ONCE on the combined
  # frame, so the column type is uniform and sec/peak_mb come back numeric.
  raw <- suppressMessages(dplyr::bind_rows(lapply(
    shards, function(s) read_tsv(s, na = c("", "NA"), show_col_types = FALSE,
                                 col_types = readr::cols(.default = readr::col_character())))))
  if (nrow(raw) == 0) return(invisible(NULL))
  raw <- suppressMessages(readr::type_convert(raw))

  # drop error-log rows: keep explicit markers (note ~ "OOM" or "NA_applicable") even with NA sec
  raw <- raw %>% filter(!is.na(variant))
  raw <- raw %>% filter(!is.na(sec) | grepl("OOM|NA_applicable", note, ignore.case = TRUE))
  if (nrow(raw) == 0) {
    message(sprintf("[%s] no usable rows (all error logs)", basename(patch_dir)))
    return(invisible(NULL))
  }

  # dedupe: framework append-bug wrote identical rows twice (differ only in note "" vs "NA")
  n_before <- nrow(raw)
  raw <- raw %>%
    distinct(timestamp, patch_name, tier, rep_idx, variant, sec, peak_mb,
             system_os, system_cpu, system_threads, .keep_all = TRUE)
  n_dupe <- n_before - nrow(raw)

  pkg_name <- raw$patch_name[1]
  patch_dir_name <- basename(patch_dir)
  has_dataset <- "dataset" %in% names(raw)
  if (!has_dataset) raw$dataset <- NA_character_
  # Mac/inferred-Mac rows in seurat/scanpy patches (except integrate_cca) with blank dataset → pbmc68k inferred
  apply_mac_pbmc68k_inference <- grepl("^seurat_|^scanpy_", patch_dir_name) &&
                                  !grepl("integrate_cca", patch_dir_name)
  d <- raw %>%
    mutate(
      threads_lbl  = ifelse(is.na(system_threads), "unknown", as.character(system_threads)),
      # (Per-rubric thread reduction happens after platform_lbl / dataset_lbl are
      # finalized -- see the conditional-pool block before the baseline reference.)
      is_oom       = grepl("OOM", note, ignore.case = TRUE) & is.na(sec),
      is_na        = grepl("NA_applicable", note, ignore.case = TRUE) & is.na(sec),
      platform_raw = classify_platform(ifelse(is.na(system_os), "", system_os)),
      # Blank-OS rows of INFERRED_MAC_PACKAGES were timing-confirmed as macOS,
      # so they are formalized to the plain "macOS" label (no "(inferred)").
      platform_lbl = ifelse(platform_raw == "unknown" & pkg_name %in% INFERRED_MAC_PACKAGES,
                            "macOS", platform_raw),
      dataset_lbl  = ifelse(is.na(dataset) | dataset == "", "—", as.character(dataset)),
      dataset_lbl  = ifelse(apply_mac_pbmc68k_inference &
                              platform_lbl %in% c("macOS", "macOS (inferred)") &
                              dataset_lbl == "—",
                            "pbmc68k (inferred)", dataset_lbl),
      pass_bool    = tolower(as.character(pass)) %in% c("true", "1")
    )

  # ----------------------------------------------------------------------
  # Backfill dataset_lbl on legacy "—" rows (rows recorded before the
  # `dataset` column was populated into speedups.tsv). For each
  # (tier, threads, platform) cell where there's a real dataset name AND
  # there are "—" rows, copy the real name into the "—" rows so they merge
  # into the same group_by below — recovering the legacy reps.
  #
  # Consistency guard, all-or-nothing per (tier, threads, platform): rescue
  # only when BOTH baseline AND patched legacy reps are within
  # ±RESCUE_TOLERANCE_PCT of the established medians. A per-variant decision
  # would let baseline get pooled when patched diverges (different code path
  # between batches), producing a speedup_x ratio that mixes new patched
  # against old+new baseline — a misleading headline number. Requiring both
  # variants to pass ensures speedup_x stays internally consistent.
  #
  # The post-grouping filter at line 205+ then drops the diverged "—" rows
  # exactly as before (existing solo-"—" patches unaffected).
  RESCUE_TOLERANCE_PCT <- 0.25
  rescue_log <- list(rescued = list(), skipped = list())
  cell_check <- d %>%
    filter(!(is_oom | is_na)) %>%
    group_by(tier, threads_lbl, platform_lbl, variant) %>%
    summarise(
      has_em = any(dataset_lbl == "—"),
      has_real = any(dataset_lbl != "—"),
      em_median = if (any(dataset_lbl == "—"))
                    median(sec[dataset_lbl == "—"], na.rm = TRUE) else NA_real_,
      real_median = if (any(dataset_lbl != "—"))
                    median(sec[dataset_lbl != "—"], na.rm = TRUE) else NA_real_,
      real_name = if (any(dataset_lbl != "—"))
                    unique(dataset_lbl[dataset_lbl != "—"])[1] else NA_character_,
      n_em = sum(dataset_lbl == "—"),
      .groups = "drop"
    ) %>%
    mutate(rel_diff = ifelse(has_em & has_real & is.finite(em_median) & real_median > 0,
                              abs(em_median - real_median) / real_median, NA_real_),
           within = !is.na(rel_diff) & rel_diff <= RESCUE_TOLERANCE_PCT)

  # Per-(tier, threads, platform) verdict: all variants present must (a) have
  # an em-vs-real comparison and (b) be within tolerance. If any single
  # variant fails or is missing on either side, the whole cell is skipped.
  cell_verdict <- cell_check %>%
    filter(has_em & has_real) %>%
    group_by(tier, threads_lbl, platform_lbl) %>%
    summarise(
      all_within = all(within),
      n_variants_with_em = n(),
      n_em_total = sum(n_em),
      worst_diff = max(rel_diff, na.rm = TRUE),
      real_name = first(real_name),
      .groups = "drop"
    )

  for (i in seq_len(nrow(cell_verdict))) {
    row <- cell_verdict[i, ]
    cell_label <- sprintf("%s/%s/%s",
                          row$tier, row$threads_lbl, row$platform_lbl)
    if (isTRUE(row$all_within)) {
      d$dataset_lbl[d$tier == row$tier & d$threads_lbl == row$threads_lbl &
                    d$platform_lbl == row$platform_lbl &
                    d$dataset_lbl == "—"] <- row$real_name
      rescue_log$rescued[[length(rescue_log$rescued) + 1L]] <- sprintf(
        "%s: %d legacy reps (worst variant %+.0f%% within %.0f%% tol)",
        cell_label, row$n_em_total, row$worst_diff * 100,
        RESCUE_TOLERANCE_PCT * 100)
    } else {
      rescue_log$skipped[[length(rescue_log$skipped) + 1L]] <- sprintf(
        "%s: %d legacy reps skipped (worst variant %+.0f%% > %.0f%% tol)",
        cell_label, row$n_em_total, row$worst_diff * 100,
        RESCUE_TOLERANCE_PCT * 100)
    }
  }
  if (length(rescue_log$rescued) > 0L) {
    message(sprintf("[%s] rescued %d legacy-dataset cells:",
                    basename(patch_dir), length(rescue_log$rescued)))
    for (entry in rescue_log$rescued) message("  rescued: ", entry)
  }
  if (length(rescue_log$skipped) > 0L) {
    message(sprintf("[%s] skipped %d legacy-dataset cells (out of tolerance):",
                    basename(patch_dir), length(rescue_log$skipped)))
    for (entry in rescue_log$skipped) message("  skipped: ", entry)
  }

  # === Per-rubric thread reduction (conditional pool) ===
  # For each (platform, variant) the rubric marks thread-inapplicable, collapse
  # its per-thread reps onto t1: pool reps of threads within +/-.TOL of t1
  # (statistically flat -> more reps; double-sided noise cancels) and DROP threads
  # farther off (fork / thread overhead). paper/rubric/task_threading.yaml.
  red <- .REDUCED[[patch_dir_name]]
  red_mac <- if (!is.null(red) && !is.null(red$mac)) red$mac else character(0)
  red_win <- if (!is.null(red) && !is.null(red$win)) red$win else character(0)
  # Per-row threading verdict for Supplementary Data 2: "flat" if this
  # (platform, variant) is either (a) rubric-reduced (its t>1 reps pool onto t1)
  # OR (b) in thread_scaling_flat (kept per-thread but empirically thread-flat);
  # else "scale". The flat_* set is LABEL-ONLY -- the rows below are not reduced.
  flat <- .FLAT_LABEL[[patch_dir_name]]
  flat_mac <- if (!is.null(flat) && !is.null(flat$mac)) flat$mac else character(0)
  flat_win <- if (!is.null(flat) && !is.null(flat$win)) flat$win else character(0)
  d <- d %>% mutate(thread_scaling = ifelse(
    (platform_lbl == "macOS"   & variant %in% c(red_mac, flat_mac)) |
    (platform_lbl == "Windows" & variant %in% c(red_win, flat_win)),
    "flat", "scale"))
  if (length(red_mac) > 0 || length(red_win) > 0) {
    thr_med <- d %>%
      group_by(tier, platform_lbl, dataset_lbl, variant, threads_lbl) %>%
      summarise(thrmed_ = median(sec, na.rm = TRUE), .groups = "drop")
    t1_med <- thr_med %>%
      filter(threads_lbl == "1") %>%
      transmute(tier, platform_lbl, dataset_lbl, variant, t1med_ = thrmed_)
    d <- d %>%
      left_join(thr_med, by = c("tier", "platform_lbl", "dataset_lbl", "variant", "threads_lbl")) %>%
      left_join(t1_med, by = c("tier", "platform_lbl", "dataset_lbl", "variant")) %>%
      mutate(
        is_reduced_ = (platform_lbl == "macOS" & variant %in% red_mac) |
                      (platform_lbl == "Windows" & variant %in% red_win),
        within_ = !is.na(t1med_) & t1med_ > 0 & !is.na(thrmed_) &
                  abs(thrmed_ / t1med_ - 1) <= .TOL,
        drop_ = is_reduced_ & threads_lbl != "1" & !is.na(t1med_) & !within_,
        threads_lbl = ifelse(is_reduced_ & threads_lbl != "1" & within_, "1", threads_lbl)
      ) %>%
      filter(!drop_) %>%
      select(-thrmed_, -t1med_, -is_reduced_, -within_, -drop_)
  }

  baselines <- d %>%
    filter(variant == "baseline") %>%
    group_by(tier, threads_lbl, platform_lbl, dataset_lbl) %>%
    summarise(baseline_sec_mean = mean(sec, na.rm = TRUE),
              .groups = "drop")
  # FLAT baselines (thread_scaling=="flat") measure the same serial quantity at
  # every thread, so their reps are pooled ACROSS threads and the speedup uses the
  # robust pooled MEDIAN -- this is why a flat baseline may show one rep per thread
  # (they aggregate), and it de-noises single-thread outliers (e.g. a stray slow
  # baseline rep no longer inflates that thread's speedup). SCALE baselines vary by
  # thread, so they keep same-thread mean pairing (below).
  baselines_pooled <- d %>%
    filter(variant == "baseline") %>%
    group_by(tier, platform_lbl, dataset_lbl) %>%
    summarise(baseline_pooled_med = median(sec, na.rm = TRUE),
              baseline_is_flat = any(thread_scaling == "flat"),
              .groups = "drop")
  # Wildcard pairing: a reduced (serial) baseline lives only at threads_lbl="1",
  # so a per-thread patched kept because it scales (e.g. vegan on Windows) has no
  # same-thread baseline -> fall back to the t1 baseline (a serial baseline is the
  # same at every thread). Per-thread baselines keep DIRECT pairing.
  baselines_t1 <- d %>%
    filter(variant == "baseline", threads_lbl == "1") %>%
    group_by(tier, platform_lbl, dataset_lbl) %>%
    summarise(baseline_t1_mean = mean(sec, na.rm = TRUE), .groups = "drop")

  d <- d %>%
    left_join(baselines, by = c("tier", "threads_lbl", "platform_lbl", "dataset_lbl")) %>%
    left_join(baselines_t1, by = c("tier", "platform_lbl", "dataset_lbl")) %>%
    left_join(baselines_pooled, by = c("tier", "platform_lbl", "dataset_lbl")) %>%
    mutate(eff_baseline = ifelse(
             !is.na(baseline_is_flat) & baseline_is_flat & !is.na(baseline_pooled_med),
             baseline_pooled_med,                                                   # flat: pooled median across threads
             ifelse(!is.na(baseline_sec_mean), baseline_sec_mean, baseline_t1_mean)), # scale: same-thread mean, fallback t1
           speedup_x_calc = ifelse(variant == "patched" & !is.na(eff_baseline) & eff_baseline > 0,
                                   eff_baseline / sec, NA_real_))

  fallback_pkg_version <- get_package_version(patch_dir)
  placeholder_re <- "<\\w+\\.__version__>"
  if ("package_version" %in% names(d)) {
    pv <- ifelse(is.na(d$package_version), "", as.character(d$package_version))
    real_versions <- unique(pv[pv != "" & !grepl(placeholder_re, pv)])
    canonical <- if (length(real_versions) > 0) real_versions[1] else fallback_pkg_version
    if (!is.null(canonical) && !is.na(canonical) && nzchar(canonical) &&
        !grepl(placeholder_re, canonical)) {
      is_bad <- pv == "" | grepl(placeholder_re, pv)
      pv[is_bad] <- canonical
    } else {
      pv[pv == ""] <- if (is.null(canonical) || is.na(canonical)) "" else canonical
    }
    d$package_version <- pv
  } else {
    d$package_version <- fallback_pkg_version
  }
  # Group without package_version: reps across nearby upstream patch
  # releases (e.g. seurat 5.4.0 vs 5.4.1) merge into one cell rather
  # than producing two n=1 rows. Reported package_version becomes the
  # latest (max-timestamp) row's value.
  pkg_per_cell <- d %>%
    arrange(timestamp) %>%
    group_by(patch = patch_name, tier, threads = threads_lbl,
             platform = platform_lbl, dataset = dataset_lbl, variant) %>%
    summarise(package_version = dplyr::last(package_version), .groups = "drop")
  out <- d %>%
    group_by(patch = patch_name, tier, threads = threads_lbl, platform = platform_lbl, dataset = dataset_lbl, variant) %>%
    summarise(
      status           = ifelse(all(is_na), "N/A",
                           ifelse(all(is_oom | is_na), "OOM",
                             ifelse(any(is_oom | is_na), "partial", "ok"))),
      n_reps           = sum(!(is_oom | is_na)),
      sec_reps         = fmt_num(sec[!(is_oom | is_na)], 3),
      sec_mean         = mean(sec[!(is_oom | is_na)], na.rm = TRUE),
      sec_median       = median(sec[!(is_oom | is_na)], na.rm = TRUE),
      mem_reps         = fmt_num(peak_mb[!(is_oom | is_na)], 1),
      mem_mean         = mean(peak_mb[!(is_oom | is_na)], na.rm = TRUE),
      mem_median       = median(peak_mb[!(is_oom | is_na)], na.rm = TRUE),
      speedup_x_reps   = fmt_num(speedup_x_calc, 3),
      speedup_x_mean   = mean(speedup_x_calc, na.rm = TRUE),
      speedup_x_median = median(speedup_x_calc, na.rm = TRUE),
      # Only count reps that were actually concordance-assessed. An unassessed
      # rep (pass NA/blank, e.g. a perf-only run with no runnable baseline to
      # diff against) must NOT be scored as a failure — leave pass_rate blank.
      pass_rate        = { .pa <- !is.na(pass) & toupper(trimws(as.character(pass))) %in% c("TRUE","FALSE","1","0")
                           if (any(.pa)) mean(pass_bool[.pa]) else NA_real_ },
      metrics_json_median = if ("metrics_json" %in% names(d))
                              aggregate_metrics_json(metrics_json[!(is_oom | is_na)])
                            else "",
      fw_versions      = paste(sort(unique(na.omit(framework_version))), collapse = ", "),
      ts_first         = min(timestamp, na.rm = TRUE),
      ts_last          = max(timestamp, na.rm = TRUE),
      thread_scaling   = dplyr::first(thread_scaling),
      .groups = "drop"
    ) %>%
    left_join(pkg_per_cell,
              by = c("patch", "tier", "threads", "platform", "dataset", "variant")) %>%
    arrange(patch, platform, threads, dataset, tier, variant) %>%
    mutate(across(c(sec_mean, sec_median, mem_mean, mem_median,
                    speedup_x_mean, speedup_x_median, pass_rate),
                  ~ ifelse(is.nan(.x), NA_real_, round(.x, 3)))) %>%
    # Preserve historical column order: patch, package_version, tier, ...
    select(patch, package_version, tier, threads, platform, dataset, variant,
           everything())

  # blank speedup + pass cols for baseline rows
  out <- out %>%
    mutate(
      speedup_x_reps   = ifelse(variant == "baseline", "", speedup_x_reps),
      speedup_x_mean   = ifelse(variant == "baseline", NA_real_, speedup_x_mean),
      speedup_x_median = ifelse(variant == "baseline", NA_real_, speedup_x_median),
      pass_rate        = ifelse(variant == "baseline", NA_real_, pass_rate),
      metrics_json_median = ifelse(variant == "baseline", "", metrics_json_median)
    )

  # Drop legacy dataset="—" rows when a real-dataset row exists for the same
  # (patch, tier, threads, platform). Keeps solo-"—" tools (e.g. clusterprofiler).
  n_before_drop <- nrow(out)
  out <- out %>%
    group_by(patch, tier, threads, platform) %>%
    filter(!(dataset == "—" & any(dataset != "—"))) %>%
    ungroup()
  n_dropped_em <- n_before_drop - nrow(out)

  out_path <- file.path(patch_dir, "speedups_finalized.tsv")
  # base write.table with quote=FALSE: JSON strings would be CSV-escaped (""..."")
  # by readr::write_tsv without outer-quote wrapping, which breaks downstream parse.
  write.table(out, file = out_path, sep = "\t", quote = FALSE,
              row.names = FALSE, na = "", eol = "\n")
  message(sprintf("[%s] %d raw rows (%d dupes dropped) -> %d summary rows (%d redundant '—' dropped)",
                  basename(patch_dir), n_before, n_dupe, nrow(out), n_dropped_em))
  invisible(out)
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  patches <- if (length(args) > 0) args else list.dirs(PATCHES_DIR, recursive = FALSE, full.names = FALSE)
  for (p in patches) {
    d <- file.path(PATCHES_DIR, p)
    if (!dir.exists(d)) {
      message(sprintf("skip: %s (no dir)", p)); next
    }
    finalize_one(d)
  }
}

if (sys.nframe() == 0) main()
