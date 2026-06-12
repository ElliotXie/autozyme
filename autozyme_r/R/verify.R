# Internal: hand-roll JSON for {metric_name: value} dicts. We avoid a hard
# jsonlite dep -- inst/verify_worker.R uses the same trick. Numeric NaN / Inf
# become null so the resulting string parses as valid JSON.
.metrics_to_json <- function(metrics) {
  if (!length(metrics)) return("{}")
  pairs <- character(length(metrics))
  for (i in seq_along(metrics)) {
    nm <- names(metrics)[i]
    val <- metrics[[i]]$value
    if (is.null(val) || !is.finite(val)) {
      pairs[i] <- sprintf('"%s": null', nm)
    } else {
      pairs[i] <- sprintf('"%s": %.9g', nm, val)
    }
  }
  paste0("{", paste(pairs, collapse = ", "), "}")
}

.framework_version <- function() {
  tryCatch(
    as.character(utils::packageVersion("autozyme")),
    error = function(e) "unknown"
  )
}

.tsv_sanitize <- function(s) {
  if (is.null(s) || (length(s) == 1L && is.na(s))) return("")
  s <- as.character(s)
  gsub("[\t\r\n]", " ", s)
}

.fmt_num <- function(x) {
  if (is.null(x) || length(x) == 0L) return("")
  if (is.na(x) || !is.finite(x)) return("")
  sprintf("%.6f", x)
}

.fmt_array <- function(xs) {
  if (is.null(xs) || !length(xs)) return("")
  paste(sprintf("%.6f", xs), collapse = ";")
}

.package_verify_header <- paste(
  "timestamp", "patch_name", "tier", "dataset",
  "rep_idx", "variant",
  "sec", "speedup_pct", "speedup_x",
  "peak_mb", "peak_mb_change_pct", "peak_mb_fold",
  "pass", "metrics_json",
  "framework_version", "package_version", "note",
  "system_os", "system_cpu", "system_ram_gb", "system_threads",
  sep = "\t"
)

# Pre-2026-05-28 long format (had dataset, no package_version). Still accepted; upgraded on append.
.package_verify_header_pre_pkgver <- paste(
  "timestamp", "patch_name", "tier", "dataset",
  "rep_idx", "variant",
  "sec", "speedup_pct", "speedup_x",
  "peak_mb", "peak_mb_change_pct", "peak_mb_fold",
  "pass", "metrics_json",
  "framework_version", "note",
  "system_os", "system_cpu", "system_ram_gb", "system_threads",
  sep = "\t"
)

# Pre-2026-05-26 long format (no dataset column). Still accepted; upgraded on append.
.package_verify_header_legacy <- paste(
  "timestamp", "patch_name", "tier",
  "rep_idx", "variant",
  "sec", "speedup_pct", "speedup_x",
  "peak_mb", "peak_mb_change_pct", "peak_mb_fold",
  "pass", "metrics_json",
  "framework_version", "note",
  "system_os", "system_cpu", "system_ram_gb", "system_threads",
  sep = "\t"
)



# Best-effort CPU model name string. Empty on failure.
.detect_cpu_model <- function() {
  sysname <- Sys.info()[["sysname"]]
  out <- tryCatch({
    if (identical(sysname, "Windows")) {
      txt <- suppressWarnings(system2("wmic", c("cpu", "get", "name", "/value"),
                                       stdout = TRUE, stderr = FALSE))
      for (ln in txt) {
        ln <- trimws(ln)
        m <- regmatches(ln, regexec("^Name=(.+)$", ln))[[1]]
        if (length(m) >= 2L && nzchar(trimws(m[2]))) return(trimws(m[2]))
      }
      ""
    } else if (identical(sysname, "Linux")) {
      txt <- readLines("/proc/cpuinfo", warn = FALSE)
      hits <- grep("^model name", txt, value = TRUE)
      if (length(hits)) sub("^[^:]*:\\s*", "", hits[1]) else ""
    } else if (identical(sysname, "Darwin")) {
      txt <- suppressWarnings(system2("sysctl", c("-n", "machdep.cpu.brand_string"),
                                       stdout = TRUE, stderr = FALSE))
      if (length(txt)) trimws(txt[1]) else ""
    } else ""
  }, error = function(e) "")
  if (is.null(out) || !length(out)) "" else as.character(out)[1]
}

# Total physical RAM in GB (1 decimal). NA_real_ on failure.
.detect_ram_gb <- function() {
  sysname <- Sys.info()[["sysname"]]
  bytes <- tryCatch({
    if (identical(sysname, "Windows")) {
      txt <- suppressWarnings(system2(
        "wmic",
        c("computersystem", "get", "TotalPhysicalMemory", "/value"),
        stdout = TRUE, stderr = FALSE
      ))
      hit <- NA_real_
      for (ln in txt) {
        m <- regmatches(ln, regexec("TotalPhysicalMemory=([0-9]+)", ln))[[1]]
        if (length(m) >= 2L) { hit <- as.numeric(m[2]); break }
      }
      hit
    } else if (identical(sysname, "Linux")) {
      txt <- readLines("/proc/meminfo", warn = FALSE)
      hits <- grep("^MemTotal:", txt, value = TRUE)
      if (length(hits)) {
        kb <- as.numeric(sub(".*?([0-9]+).*", "\\1", hits[1]))
        kb * 1024
      } else NA_real_
    } else if (identical(sysname, "Darwin")) {
      txt <- suppressWarnings(system2("sysctl", c("-n", "hw.memsize"),
                                       stdout = TRUE, stderr = FALSE))
      if (length(txt)) as.numeric(txt[1]) else NA_real_
    } else NA_real_
  }, error = function(e) NA_real_)
  if (is.null(bytes) || !length(bytes) || !is.finite(bytes)) return(NA_real_)
  round(bytes / (1024^3), 1)
}

# Compact system fingerprint for the attest row.
# Returns list(system_os, system_cpu, system_ram_gb, system_threads).
.collect_system_info <- function() {
  info <- Sys.info()
  sysname <- info[["sysname"]]
  release <- info[["release"]]
  if (identical(sysname, "Darwin")) {
    os_str <- paste0("macOS ", release)
  } else if (identical(sysname, "Windows")) {
    # Sys.info()[["release"]] reports "10 x64" on both Win10 and Win11
    # because Microsoft kept the kernel release "10" for app compat. Use the
    # build number from `ver` to disambiguate -- Win11 starts at build 22000.
    release_simple <- sub("\\s.*", "", release)  # strip " x64" suffix
    build_release <- tryCatch({
      ver_out <- suppressWarnings(shell("ver", intern = TRUE))
      m <- regmatches(ver_out, regexec("\\[Version\\s+\\d+\\.\\d+\\.(\\d+)", ver_out))
      build <- NA_integer_
      for (mm in m) if (length(mm) >= 2L) { build <- suppressWarnings(as.integer(mm[2])); break }
      if (is.finite(build) && build >= 22000L && identical(release_simple, "10")) "11"
      else release_simple
    }, error = function(e) release_simple)
    os_str <- paste0("Windows ", build_release)
  } else {
    os_str <- paste(sysname, release)
  }
  cpu <- .detect_cpu_model()
  ram <- .detect_ram_gb()
  threads <- Sys.getenv("ZYME_THREADS", "")
  if (!nzchar(threads)) threads <- Sys.getenv("AUTOZYME_THREADS", "")
  if (!nzchar(threads)) threads <- Sys.getenv("OMP_NUM_THREADS", "")
  if (!nzchar(threads)) threads <- Sys.getenv("MKL_NUM_THREADS", "")
  if (!nzchar(threads)) threads <- Sys.getenv("OPENBLAS_NUM_THREADS", "")
  list(
    system_os      = os_str,
    system_cpu     = cpu,
    system_ram_gb  = if (is.finite(ram)) sprintf("%.1f", ram) else "",
    system_threads = threads
  )
}

# OS-level peak RSS detection. Returns a list describing how to wrap the
# verify-worker subprocess so its peak working set can be recovered after
# exit, or NULL when no detection mechanism is available.
#
#   kind == "unix"    -- wrap with `/usr/bin/time -l|-v -o <file>` (macOS,
#                       Linux). The Unix tool writes rusage stats when the
#                       child exits; we parse maximum RSS.
#   kind == "windows" -- wrap with a PowerShell launcher
#                       (inst/win_peak_wrapper.ps1) that polls the kernel's
#                       per-process PeakWorkingSet64 counter every 100ms
#                       while the child is alive and writes the max to
#                       <stats_file>.
.detect_time_cmd <- function() {
  if (.Platform$OS.type == "windows") {
    ps1 <- system.file("win_peak_wrapper.ps1", package = "autozyme")
    if (!nzchar(ps1)) return(NULL)
    # `powershell.exe` is Windows-bundled (PowerShell 5.1). `pwsh.exe`
    # (PowerShell 7) isn't guaranteed; sticking with `powershell.exe`.
    return(list(
      kind    = "windows",
      ps1     = ps1,
      pattern = "PEAK_WS_BYTES=([0-9]+)",
      to_mb   = function(x) x / (1024 * 1024)
    ))
  }
  if (!file.exists("/usr/bin/time")) return(NULL)
  sysname <- Sys.info()[["sysname"]]
  if (identical(sysname, "Darwin")) {
    # BSD time: "<bytes>  maximum resident set size"
    return(list(
      kind    = "unix",
      cmd     = "/usr/bin/time",
      flag    = "-l",
      pattern = "([0-9]+)\\s+maximum resident set size",
      to_mb   = function(x) x / (1024 * 1024)
    ))
  }
  # Linux GNU time -v: "Maximum resident set size (kbytes): NNN"
  list(
    kind    = "unix",
    cmd     = "/usr/bin/time",
    flag    = "-v",
    pattern = "Maximum resident set size \\(kbytes\\):\\s*([0-9]+)",
    to_mb   = function(x) x / 1024
  )
}

.parse_peak_mb <- function(stats_file, time_info) {
  if (is.null(time_info) || !file.exists(stats_file)) return(NA_real_)
  txt <- tryCatch(readLines(stats_file, warn = FALSE),
                  error = function(e) character(0))
  if (!length(txt)) return(NA_real_)
  blob <- paste(txt, collapse = "\n")
  m <- regmatches(blob, regexec(time_info$pattern, blob))[[1]]
  if (length(m) < 2L) return(NA_real_)
  v <- suppressWarnings(as.numeric(m[2]))
  if (!is.finite(v)) return(NA_real_)
  time_info$to_mb(v)
}

# Write per-rep x variant rows (long format) to <task_dir>/package_verify.tsv,
# read-modify-write so the file stays globally sorted (variant -> tier ->
# platform -> threads -> timestamp -> rep_idx). All baseline rows above all
# patched rows even across many attest runs.
# Refuses to write into a legacy wide-format file -- user must run the
# one-shot migration script first.
.tier_dataset_map <- function(task_dir) {
  yaml_path <- file.path(task_dir, "task.yaml")
  if (!file.exists(yaml_path) || !requireNamespace("yaml", quietly = TRUE)) {
    return(setNames(character(0), character(0)))
  }
  task <- tryCatch(yaml::read_yaml(yaml_path), error = function(e) NULL)
  if (is.null(task) || is.null(task$datasets)) {
    return(setNames(character(0), character(0)))
  }
  out <- character(0)
  for (d in task$datasets) {
    tier <- if (is.null(d$tier)) "" else as.character(d$tier)
    name <- if (is.null(d$name)) "" else as.character(d$name)
    if (nzchar(tier) && nzchar(name)) out[[tier]] <- name
  }
  out
}

.dataset_name_for_tier <- function(task_dir, tier) {
  m <- .tier_dataset_map(task_dir)
  if (length(m) && tier %in% names(m)) m[[tier]] else ""
}

.append_package_verify_tsv <- function(task_dir, name, tsv_rows) {
  tsv_path <- file.path(task_dir, "package_verify.tsv")
  .assert_long_format_or_empty(tsv_path)
  timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
  fw_version <- .framework_version()
  sysinfo <- .collect_system_info()
  tier_map <- .tier_dataset_map(task_dir)
  # Read the active patch's declared tested_against (e.g. "Seurat 5.4.0")
  pkg_version <- ""
  if (exists(".zyme_registry") && !is.null(.zyme_registry[[name]])) {
    ta <- .zyme_registry[[name]]$tested_against
    if (!is.null(ta) && nzchar(ta)) pkg_version <- as.character(ta)
  }

  header_cols <- .package_verify_header_cols()
  legacy_cols <- .package_verify_header_cols_legacy()

  # Read existing rows so we can sort the union with new ones.
  existing_rows <- list()
  if (file.exists(tsv_path)) {
    df <- tryCatch(
      utils::read.table(tsv_path, header = TRUE, sep = "\t", quote = "",
                        comment.char = "", stringsAsFactors = FALSE,
                        na.strings = c("", "NA"), fill = TRUE,
                        colClasses = "character", encoding = "UTF-8"),
      error = function(e) NULL
    )
    if (!is.null(df) && nrow(df) > 0L) {
      # Ensure required columns exist; backfill dataset from tier_map if missing.
      if (!"dataset" %in% names(df)) {
        df$dataset <- vapply(df$tier, function(t) {
          if (length(tier_map) && t %in% names(tier_map)) tier_map[[t]] else ""
        }, character(1))
      }
      # Select only the columns this build knows about; ignore extra columns
      # from newer builds (forward compat).
      keep <- intersect(header_cols, names(df))
      df <- df[, keep, drop = FALSE]
      # Pad any columns this build expects but the file lacks.
      for (col in setdiff(header_cols, names(df))) {
        df[[col]] <- ""
      }
      df <- df[, header_cols, drop = FALSE]
      existing_rows <- lapply(seq_len(nrow(df)), function(i) {
        as.list(df[i, , drop = FALSE])
      })
    }
  }

  pass_cell <- function(value) {
    if (is.null(value) || (length(value) == 1L && is.na(value))) ""
    else if (isTRUE(value)) "true" else "false"
  }

  make_row <- function(rep_idx_cell, variant,
                       sec_cell, speedup_pct_cell, speedup_x_cell,
                       peak_cell, peak_pct_cell, peak_fold_cell,
                       pass_val, metrics_json, tier, note) {
    ds <- if (length(tier_map) && tier %in% names(tier_map)) tier_map[[tier]] else ""
    list(
      timestamp         = timestamp,
      patch_name        = .tsv_sanitize(name),
      tier              = .tsv_sanitize(tier),
      dataset           = .tsv_sanitize(ds),
      rep_idx           = rep_idx_cell,
      variant           = variant,
      sec               = sec_cell,
      speedup_pct       = speedup_pct_cell,
      speedup_x         = speedup_x_cell,
      peak_mb           = peak_cell,
      peak_mb_change_pct = peak_pct_cell,
      peak_mb_fold      = peak_fold_cell,
      pass              = pass_val,
      metrics_json      = .tsv_sanitize(metrics_json),
      framework_version = .tsv_sanitize(fw_version),
      package_version   = .tsv_sanitize(pkg_version),
      note              = .tsv_sanitize(note),
      system_os         = .tsv_sanitize(sysinfo$system_os),
      system_cpu        = .tsv_sanitize(sysinfo$system_cpu),
      system_ram_gb     = .tsv_sanitize(sysinfo$system_ram_gb),
      system_threads    = .tsv_sanitize(sysinfo$system_threads)
    )
  }

  new_rows <- list()
  for (r in tsv_rows) {
    tier <- r$tier
    note <- if (is.null(r$note)) "" else r$note
    baseline_secs <- if (is.null(r$baseline_secs)) numeric(0) else r$baseline_secs
    patched_secs <- if (is.null(r$patched_secs)) numeric(0) else r$patched_secs
    baseline_peaks <- if (is.null(r$baseline_peaks_mb)) numeric(0) else r$baseline_peaks_mb
    patched_peaks <- if (is.null(r$patched_peaks_mb)) numeric(0) else r$patched_peaks_mb
    per_rep_pass <- if (is.null(r$per_rep_pass)) logical(0) else r$per_rep_pass
    metrics_json <- if (is.null(r$metrics_json)) "" else r$metrics_json
    baseline_sec_for_speedup <- if (is.null(r$baseline_sec_for_speedup)) NA_real_ else r$baseline_sec_for_speedup
    baseline_peak_for_speedup <- if (is.null(r$baseline_peak_mb_for_speedup)) NA_real_ else r$baseline_peak_mb_for_speedup

    if (!length(baseline_secs) && !length(patched_secs)) {
      # Crash / skipped tier -- single sentinel row.
      new_rows[[length(new_rows) + 1L]] <-
        make_row("", "", "", "", "", "", "", "",
                 "", "", tier, note)
      next
    }

    finite_baseline <- baseline_secs[is.finite(baseline_secs)]
    baseline_median_sec <- if (length(finite_baseline)) stats::median(finite_baseline) else NA_real_
    if (!is.finite(baseline_median_sec) &&
        length(baseline_sec_for_speedup) == 1L &&
        is.finite(baseline_sec_for_speedup) &&
        baseline_sec_for_speedup > 0) {
      baseline_median_sec <- baseline_sec_for_speedup
    }
    finite_baseline_mb <- baseline_peaks[is.finite(baseline_peaks) & baseline_peaks > 0]
    baseline_median_mb <- if (length(finite_baseline_mb)) stats::median(finite_baseline_mb) else NA_real_
    if (!is.finite(baseline_median_mb) &&
        length(baseline_peak_for_speedup) == 1L &&
        is.finite(baseline_peak_for_speedup) &&
        baseline_peak_for_speedup > 0) {
      baseline_median_mb <- baseline_peak_for_speedup
    }

    for (i in seq_along(baseline_secs)) {
      peak <- if (i <= length(baseline_peaks)) baseline_peaks[i] else NA_real_
      new_rows[[length(new_rows) + 1L]] <- make_row(
        as.character(i), "baseline",
        .fmt_num(baseline_secs[i]), "", "",
        .fmt_num(peak), "", "",
        "", "", tier, note
      )
    }
    for (i in seq_along(patched_secs)) {
      peak <- if (i <= length(patched_peaks)) patched_peaks[i] else NA_real_
      rep_pass <- if (i <= length(per_rep_pass)) per_rep_pass[i] else NA
      speedup_x_cell <- ""
      speedup_pct_cell <- ""
      peak_fold_cell <- ""
      peak_pct_cell <- ""
      sec_i <- patched_secs[i]
      if (!is.na(baseline_median_sec) && baseline_median_sec > 0 &&
          is.finite(sec_i) && sec_i > 0) {
        speedup_x_cell <- .fmt_num(baseline_median_sec / sec_i)
        speedup_pct_cell <- .fmt_num((baseline_median_sec - sec_i) / baseline_median_sec * 100)
      }
      if (!is.na(baseline_median_mb) && baseline_median_mb > 0 &&
          is.finite(peak) && peak > 0) {
        peak_fold_cell <- .fmt_num(baseline_median_mb / peak)
        peak_pct_cell <- .fmt_num((baseline_median_mb - peak) / baseline_median_mb * 100)
      }
      new_rows[[length(new_rows) + 1L]] <- make_row(
        as.character(i), "patched",
        .fmt_num(sec_i), speedup_pct_cell, speedup_x_cell,
        .fmt_num(peak), peak_pct_cell, peak_fold_cell,
        pass_cell(rep_pass), metrics_json, tier, note
      )
    }
  }

  combined <- c(existing_rows, new_rows)
  combined <- combined[order(sapply(combined, .row_sort_key))]

  # Atomic rewrite: temp + rename.
  tmp_path <- paste0(tsv_path, ".tmp")
  con <- file(tmp_path, open = "w")
  on.exit({ if (isOpen(con)) close(con); if (file.exists(tmp_path)) file.remove(tmp_path) },
          add = TRUE)
  writeLines(.package_verify_header, con)
  for (row in combined) {
    cells <- vapply(header_cols, function(col) {
      v <- row[[col]]
      if (is.null(v)) "" else as.character(v)
    }, character(1))
    writeLines(paste(cells, collapse = "\t"), con)
  }
  close(con)
  file.rename(tmp_path, tsv_path)
  on.exit()  # clear handler since rename succeeded
}


# Stream one row to package_verify.tsv after each rep's measurement so a
# mid-sweep crash never silently drops finished work.
.append_one_pv_row <- function(task_dir, name, tier, variant, rep_idx,
                                sec = NA_real_, peak_mb = NA_real_,
                                speedup_pct = NA_real_, speedup_x = NA_real_,
                                peak_pct = NA_real_, peak_fold = NA_real_,
                                rep_pass = NA, metrics_json = "",
                                note = "") {
  tsv_path <- file.path(task_dir, "package_verify.tsv")
  .assert_long_format_or_empty(tsv_path)
  timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
  fw_version <- .framework_version()
  sysinfo <- .collect_system_info()
  tier_map <- .tier_dataset_map(task_dir)
  pkg_version <- ""
  if (exists(".zyme_registry") && !is.null(.zyme_registry[[name]])) {
    ta <- .zyme_registry[[name]]$tested_against
    if (!is.null(ta) && nzchar(ta)) pkg_version <- as.character(ta)
  }
  header_cols <- .package_verify_header_cols()
  ds <- if (length(tier_map) && tier %in% names(tier_map)) tier_map[[tier]] else ""
  pass_cell <- if (is.null(rep_pass) || (length(rep_pass) == 1L && is.na(rep_pass))) ""
               else if (isTRUE(rep_pass)) "true" else "false"
  row <- list(
    timestamp = timestamp, patch_name = .tsv_sanitize(name),
    tier = .tsv_sanitize(tier), dataset = .tsv_sanitize(ds),
    rep_idx = as.character(rep_idx), variant = as.character(variant),
    sec = .fmt_num(sec), speedup_pct = .fmt_num(speedup_pct),
    speedup_x = .fmt_num(speedup_x), peak_mb = .fmt_num(peak_mb),
    peak_mb_change_pct = .fmt_num(peak_pct), peak_mb_fold = .fmt_num(peak_fold),
    pass = pass_cell, metrics_json = .tsv_sanitize(metrics_json),
    framework_version = .tsv_sanitize(fw_version),
    package_version = .tsv_sanitize(pkg_version),
    note = .tsv_sanitize(note),
    system_os = .tsv_sanitize(sysinfo$system_os),
    system_cpu = .tsv_sanitize(sysinfo$system_cpu),
    system_ram_gb = .tsv_sanitize(sysinfo$system_ram_gb),
    system_threads = .tsv_sanitize(sysinfo$system_threads)
  )
  existing_rows <- list()
  if (file.exists(tsv_path)) {
    df <- tryCatch(
      utils::read.table(tsv_path, header = TRUE, sep = "\t", quote = "",
                        comment.char = "", stringsAsFactors = FALSE,
                        na.strings = c("", "NA"), fill = TRUE,
                        colClasses = "character", encoding = "UTF-8"),
      error = function(e) NULL
    )
    if (!is.null(df) && nrow(df) > 0L) {
      if (!"dataset" %in% names(df)) {
        df$dataset <- vapply(df$tier, function(t) {
          if (length(tier_map) && t %in% names(tier_map)) tier_map[[t]] else ""
        }, character(1))
      }
      keep <- intersect(header_cols, names(df))
      df <- df[, keep, drop = FALSE]
      for (col in setdiff(header_cols, names(df))) df[[col]] <- ""
      df <- df[, header_cols, drop = FALSE]
      existing_rows <- lapply(seq_len(nrow(df)), function(i) as.list(df[i, , drop = FALSE]))
    }
  }
  combined <- c(existing_rows, list(row))
  combined <- combined[order(sapply(combined, .row_sort_key))]
  tmp_path <- paste0(tsv_path, ".tmp")
  con <- file(tmp_path, open = "w")
  on.exit({ if (isOpen(con)) close(con); if (file.exists(tmp_path)) file.remove(tmp_path) },
          add = TRUE)
  writeLines(.package_verify_header, con)
  for (r in combined) {
    cells <- vapply(header_cols, function(col) {
      v <- r[[col]]
      if (is.null(v)) "" else as.character(v)
    }, character(1))
    writeLines(paste(cells, collapse = "\t"), con)
  }
  close(con)
  file.rename(tmp_path, tsv_path)
  on.exit()
  invisible(NULL)
}


# Compute a single sortable scalar by packing (variant, tier, plat, threads,
# ts, rep_idx) into a delimited string. Matches the Python
# parsers.package_verify_tsv.sort_rows_for_output ordering.
.tiers_order_long <- c("tiny", "small", "medium", "large", "ood_large",
                       "ood_xlarge")
.variant_order_long <- c(baseline = 0L, patched = 1L)

.row_sort_key <- function(row) {
  # Coerce NULL/NA to "" up front. read.table reads empty cells as NA when
  # na.strings includes ""; without this, nzchar(NA) returns NA and the
  # subsequent `if (!nzchar(...))` blows up with
  # "missing value where TRUE/FALSE needed".
  .nz <- function(v, default = "") {
    if (is.null(v)) return(default)
    s <- as.character(v)
    if (length(s) == 0L || is.na(s[1])) return(default)
    trimws(s[1])
  }
  variant <- .nz(row$variant)
  variant_idx <- if (variant %in% names(.variant_order_long))
    .variant_order_long[[variant]] else 2L
  tier <- .nz(row$tier)
  tier_idx <- match(tier, .tiers_order_long, nomatch = length(.tiers_order_long) + 1L)
  if (is.na(tier_idx)) tier_idx <- length(.tiers_order_long) + 1L
  os_raw <- tolower(.nz(row$system_os))
  plat <- if (!nzchar(os_raw)) "mac"
          else if (startsWith(os_raw, "windows")) "win"
          else if (startsWith(os_raw, "macos") || startsWith(os_raw, "darwin")) "mac"
          else "unknown"
  threads <- suppressWarnings(as.integer(.nz(row$system_threads, "0")))
  if (is.na(threads)) threads <- 0L
  ts <- .nz(row$timestamp)
  rep_idx <- suppressWarnings(as.integer(.nz(row$rep_idx, "0")))
  if (is.na(rep_idx)) rep_idx <- 0L
  # Pack: integers zero-padded to keep lexicographic order = numeric.
  sprintf("%d\037%02d\037%s\037%04d\037%s\037%05d",
          variant_idx, tier_idx, plat, threads, ts, rep_idx)
}

# Normalize a TSV header line for comparison (BOM / CRLF / stray whitespace).
.normalize_tsv_header_line <- function(line) {
  if (!length(line) || !nzchar(line)) return("")
  line <- sub("^\ufeff", "", line, perl = TRUE)
  line <- gsub("\r$", "", line)
  trimws(line)
}

.package_verify_header_cols <- function() {
  strsplit(.package_verify_header, "\t", fixed = TRUE)[[1]]
}

.package_verify_header_cols_legacy <- function() {
  strsplit(.package_verify_header_legacy, "\t", fixed = TRUE)[[1]]
}

.package_verify_header_cols_pre_pkgver <- function() {
  strsplit(.package_verify_header_pre_pkgver, "\t", fixed = TRUE)[[1]]
}

.header_cols_of_line <- function(line) {
  strsplit(.normalize_tsv_header_line(line), "\t", fixed = TRUE)[[1]]
}

# Accept canonical long headers even when the on-disk line differs only by
# encoding (UTF-8 BOM, CRLF) from the in-memory constant string. Forward-
# compatible: if the on-disk header is a superset of the columns this build
# knows about, accept it (a newer build may have added columns).
.is_known_package_verify_header_cols <- function(cols) {
  if (!length(cols)) return(FALSE)
  current <- .package_verify_header_cols()
  pre_pkgver <- .package_verify_header_cols_pre_pkgver()
  legacy <- .package_verify_header_cols_legacy()
  known <- list(current, pre_pkgver, legacy)
  if (any(vapply(known, function(k) identical(cols, k), logical(1)))) {
    return(TRUE)
  }
  # Forward compat: on-disk header is a strict superset of a known schema.
  if (any(vapply(known, function(k) all(k %in% cols), logical(1)))) {
    return(TRUE)
  }
  FALSE
}

# Refuse to append to a legacy wide-format TSV. Mixing wide rows on top of a
# migrated long-format file would corrupt the schema; bail loudly so the user
# runs the one-shot migration script instead.
.assert_long_format_or_empty <- function(tsv_path) {
  if (!file.exists(tsv_path)) return(invisible(NULL))
  lines <- readLines(tsv_path, warn = FALSE, n = 1L, encoding = "UTF-8")
  if (!length(lines) || !nzchar(.normalize_tsv_header_line(lines[1]))) {
    return(invisible(NULL))
  }
  norm <- .normalize_tsv_header_line(lines[1])
  if (identical(norm, .package_verify_header) ||
      identical(norm, .package_verify_header_pre_pkgver) ||
      identical(norm, .package_verify_header_legacy)) {
    return(invisible(NULL))
  }
  cols <- .header_cols_of_line(lines[1])
  if (.is_known_package_verify_header_cols(cols)) return(invisible(NULL))
  stop(sprintf("%s has an unrecognized header: %s", tsv_path, norm),
       call. = FALSE)
}

# Resolve the smoke recipe for verify_patch / verify_worker.
#
# Priority:
#   1. <task_dir>/attest/smoke.R defines `smoke <- list(load=, call=, save=)`
#   2. patch$smoke from register_patch() (legacy default for single-function patches)
#
# Multi-step packages (e.g. seurat) ship one patch but many attest tasks; each
# task owns its timed region + I/O schema in attest/smoke.R.
.resolve_smoke <- function(task_dir, patch) {
  task_dir <- normalizePath(task_dir, mustWork = FALSE)
  smoke_file <- file.path(task_dir, "attest", "smoke.R")
  if (file.exists(smoke_file)) {
    env <- new.env(parent = globalenv())
    sys.source(smoke_file, envir = env)
    if (!exists("smoke", envir = env, inherits = FALSE)) {
      stop("attest/smoke.R must define `smoke <- list(load=, call=, save=)`: ",
           smoke_file, call. = FALSE)
    }
    out <- env$smoke
    if (!is.list(out) ||
        !all(c("load", "call", "save") %in% names(out)) ||
        !all(vapply(out[c("load", "call", "save")], is.function, logical(1)))) {
      stop("attest/smoke.R `smoke` must be a list(load, call, save) of functions: ",
           smoke_file, call. = FALSE)
    }
    return(out)
  }
  if (!is.null(patch) && !is.null(patch$smoke)) return(patch$smoke)
  NULL
}

# Internal: run verify on a single tier with reps measurement loop.
# Returns a list with timing aggregates + per-rep details, OR throws.
.effective_threshold <- function(metric, tier, intrinsic_noise) {
  # Mirror of autozyme_cli/zyme/parsers/task_yaml.py::effective_threshold --
  # handles both schemas the iteration phase already supports:
  #   Deterministic: metric$threshold present -> return as-is.
  #   Stochastic: metric$absolute_floor + optional metric$noise_multiplier
  #     (default 2.0). With calibrated intrinsic_noise[[tier]][[name]]:
  #       lte: effective = max(floor, multiplier * noise)
  #       gte: effective = max(floor, 1 - multiplier * (1 - noise))
  #     Without calibration, falls back to absolute_floor.
  if (!is.null(metric$threshold)) {
    return(list(value = as.numeric(metric$threshold), label = "absolute"))
  }
  if (is.null(metric$absolute_floor)) {
    stop(sprintf("metric '%s' has neither `threshold` nor `absolute_floor`",
                 metric$name), call. = FALSE)
  }
  floor_v <- as.numeric(metric$absolute_floor)
  mult    <- if (is.null(metric$noise_multiplier)) 2.0 else as.numeric(metric$noise_multiplier)
  raw     <- intrinsic_noise[[tier]][[metric$name]]
  if (is.null(raw)) {
    return(list(value = floor_v, label = "absolute_floor (no intrinsic_noise calibrated)"))
  }
  raw <- as.numeric(raw)
  if (identical(metric$comparator, "lte")) {
    relaxed <- mult * raw
    list(value = max(floor_v, relaxed),
         label = sprintf("max(floor=%g, %gxnoise=%.4g)", floor_v, mult, relaxed))
  } else {
    relaxed <- 1.0 - mult * (1.0 - raw)
    list(value = max(floor_v, relaxed),
         label = sprintf("max(floor=%g, 1-%gx(1-noise)=%.4g)", floor_v, mult, relaxed))
  }
}

# Pin BLAS / RcppParallel threads to task.yaml::baseline_threads[0] (default 1),
# matching legacy testing/drivers/test_seurat_turbo.R and zyme baseline reference.
.resolve_baseline_threads <- function(task_dir) {
  raw <- Sys.getenv("ZYME_THREADS", unset = "")
  if (!nzchar(raw)) raw <- Sys.getenv("AUTOZYME_THREADS", unset = "")
  if (nzchar(raw)) {
    threads <- suppressWarnings(as.integer(raw))
  } else {
    yaml_path <- file.path(task_dir, "task.yaml")
    if (!file.exists(yaml_path)) return(1L)
    task <- yaml::read_yaml(yaml_path)
    bt <- task$baseline_threads
    threads <- if (length(bt)) as.integer(bt[[1L]]) else 1L
  }
  if (is.na(threads) || threads < 1L) 1L else threads
}

.ensure_task_thread_env <- function(task_dir) {
  threads <- .resolve_baseline_threads(task_dir)
  t <- as.character(threads)
  Sys.setenv(
    ZYME_THREADS = t,
    AUTOZYME_THREADS = t,
    OMP_NUM_THREADS = t,
    OPENBLAS_NUM_THREADS = t,
    MKL_NUM_THREADS = t,
    VECLIB_MAXIMUM_THREADS = t,
    NUMEXPR_NUM_THREADS = t
  )
  if (requireNamespace("RcppParallel", quietly = TRUE)) {
    RcppParallel::setThreadOptions(numThreads = threads)
  }
  invisible(threads)
}

.sync_worker_thread_options <- function() {
  threads <- suppressWarnings(as.integer(Sys.getenv("ZYME_THREADS", unset = "")))
  if (length(threads) == 0L || is.na(threads) || threads < 1L) {
    raw <- Sys.getenv("AUTOZYME_THREADS", unset = "")
    if (!nzchar(raw)) raw <- Sys.getenv("OMP_NUM_THREADS", unset = "1")
    threads <- suppressWarnings(as.integer(raw))
    if (is.na(threads) || threads < 1L) threads <- 1L
  }
  # set_threads() does the full propagation: env vars (OMP/MKL/OPENBLAS/...)
  # + RcppParallel cap + RhpcBLASctl + the R-level `options(autozyme.threads
  # = N)` that patches like nichenetr/vegan/wgcna/clusterprofiler check
  # via getOption("autozyme.threads", default). Without setting the R
  # option, those patches default to N=14 regardless of attest's --threads,
  # producing unfair speedup numbers (T=1 baseline vs 14-thread patched).
  if (exists("set_threads", envir = asNamespace("autozyme"))) {
    autozyme::set_threads(threads)
  } else {
    if (requireNamespace("RcppParallel", quietly = TRUE)) {
      RcppParallel::setThreadOptions(numThreads = threads)
    }
    options(autozyme.threads = threads)
  }
  invisible(threads)
}

# Spawn a fresh-Rscript subprocess for one (baseline OR patched) measurement.
# Returns list(elapsed_sec, peak_mb):
#   - elapsed_sec: timed `patch$smoke$call` region from the worker's JSON.
#   - peak_mb: OS peak RSS over the subprocess's whole lifetime, captured
#     by wrapping Rscript with /usr/bin/time -l/-v -o <file>. Returns NA on
#     platforms where /usr/bin/time isn't available (Windows, minimal
#     containers). Includes shared/mmap + native (Rcpp/BLAS) allocations --
#     numbers diverge from R's gc()$"max used" by design.
.run_worker <- function(name, task_dir, tier, output_dir, activate, verbose) {
  worker_path <- system.file("verify_worker.R", package = "autozyme")
  if (!nzchar(worker_path)) {
    stop("autozyme: inst/verify_worker.R missing from installed package -- ",
         "reinstall the package with R CMD INSTALL autozyme_r/")
  }
  rscript <- file.path(R.home("bin"), "Rscript")
  worker_args <- c(
    shQuote(worker_path),
    "--patch", shQuote(name),
    "--task-dir", shQuote(task_dir),
    "--tier", shQuote(tier),
    "--output-dir", shQuote(output_dir)
  )
  if (isTRUE(activate)) worker_args <- c(worker_args, "--activate")

  # Capture stderr into the eventual package_verify.tsv note on failures.
  # Streaming it directly in verbose mode made OOM rows unhelpful because the
  # live stderr reached the terminal but not the persisted sentinel row.
  stderr_file <- tempfile("autozyme_worker_stderr_", fileext = ".txt")
  stderr_dest <- stderr_file

  time_info <- .detect_time_cmd()
  peak_stats_file <- NULL
  if (!is.null(time_info)) {
    peak_stats_file <- tempfile("autozyme_peak_", fileext = ".txt")
    if (identical(time_info$kind, "windows")) {
      # PowerShell wrapper polls PeakWorkingSet64 every 100ms while the
      # child is alive, then writes "PEAK_WS_BYTES=<n>" to the stats file.
      # Child stderr inherits this script's stderr -> verbose streaming
      # still works.
      cmd_bin  <- "powershell.exe"
      cmd_args <- c("-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", shQuote(time_info$ps1),
                    shQuote(peak_stats_file),
                    shQuote(rscript), worker_args)
    } else {
      # Unix /usr/bin/time -l/-v -o <file> writes rusage to <file>;
      # stderr from Rscript flows through normally so verbose live
      # streaming works.
      cmd_bin  <- time_info$cmd
      cmd_args <- c(time_info$flag, "-o", shQuote(peak_stats_file),
                    shQuote(rscript), worker_args)
    }
  } else {
    cmd_bin  <- rscript
    cmd_args <- worker_args
  }

  out <- suppressWarnings(system2(
    cmd_bin,
    args = cmd_args,
    stdout = TRUE,
    stderr = stderr_dest
  ))
  status <- attr(out, "status")
  stderr_lines <- if (file.exists(stderr_file)) {
    tryCatch(readLines(stderr_file, warn = FALSE),
             error = function(e) character(0))
  } else character(0)
  if (isTRUE(verbose) && length(stderr_lines)) {
    cat(paste(stderr_lines, collapse = "\n"), "\n", file = stderr(), sep = "")
  }
  if (file.exists(stderr_file)) unlink(stderr_file)
  peak_mb <- if (is.null(peak_stats_file)) NA_real_
             else .parse_peak_mb(peak_stats_file, time_info)
  if (!is.null(peak_stats_file)) unlink(peak_stats_file)

  # Parse the last non-empty stdout line that looks like JSON with
  # "elapsed_sec". Tolerates stray prints from upstream packages.
  lines <- out[nzchar(trimws(out))]
  failure_blob <- paste(c(out, stderr_lines), collapse = "\n")
  if (!length(lines) && !is.null(status) && status != 0L) {
    stop(sprintf(
      "verify-worker subprocess exited %d (activate=%s):\n%s",
      status, isTRUE(activate), failure_blob
    ), call. = FALSE)
  }
  if (!length(lines)) {
    stop("verify-worker printed no stdout (expected JSON)", call. = FALSE)
  }
  pat <- '\\{\\s*"elapsed_sec"\\s*:\\s*([-+0-9.eE]+)\\s*\\}'
  for (ln in rev(lines)) {
    m <- regmatches(ln, regexec(pat, ln))[[1]]
    if (length(m) == 2L) {
      return(list(elapsed_sec = as.numeric(m[2]), peak_mb = peak_mb))
    }
  }
  if (!is.null(status) && status != 0L) {
    stop(sprintf(
      "verify-worker subprocess exited %d (activate=%s):\n%s",
      status, isTRUE(activate), failure_blob
    ), call. = FALSE)
  }
  stop("verify-worker stdout had no JSON line with 'elapsed_sec':\n",
       paste(lines, collapse = "\n"), call. = FALSE)
}

.cached_reference_dir <- function(task_dir, tier) {
  candidates <- c(
    file.path(task_dir, "reference_outputs", tier),
    file.path(task_dir, sprintf("reference_output_%s", tier))
  )
  for (p in candidates) {
    if (dir.exists(p)) return(normalizePath(p, mustWork = TRUE))
  }
  stop(
    "no cached reference output found for tier '", tier, "'. Expected one of:\n  ",
    paste(candidates, collapse = "\n  "),
    "\nRun verify_patch without no_baseline_confirm once to create it.",
    call. = FALSE
  )
}

.copy_cached_reference_output <- function(src_dir, dst_dir) {
  dir.create(dst_dir, recursive = TRUE, showWarnings = FALSE)
  entries <- list.files(src_dir, all.files = TRUE, no.. = TRUE,
                        full.names = TRUE)
  if (!length(entries)) {
    stop("cached reference output is empty: ", src_dir, call. = FALSE)
  }
  ok <- file.copy(entries, dst_dir, recursive = TRUE, overwrite = TRUE,
                  copy.mode = FALSE, copy.date = FALSE)
  if (!all(ok)) {
    stop("failed to copy cached reference output from ", src_dir,
         " to ", dst_dir, call. = FALSE)
  }
  invisible(dst_dir)
}

.persist_reference_output <- function(src_dir, task_dir, tier) {
  dst_dir <- file.path(task_dir, sprintf("reference_output_%s", tier))
  dir.create(dst_dir, recursive = TRUE, showWarnings = FALSE)
  entries <- list.files(src_dir, all.files = TRUE, no.. = TRUE,
                        full.names = TRUE)
  if (!length(entries)) {
    stop("cannot persist empty reference output: ", src_dir, call. = FALSE)
  }
  stale <- list.files(dst_dir, all.files = TRUE, no.. = TRUE,
                      full.names = TRUE)
  if (length(stale)) unlink(stale, recursive = TRUE, force = TRUE)
  ok <- file.copy(entries, dst_dir, recursive = TRUE, overwrite = TRUE,
                  copy.mode = FALSE, copy.date = FALSE)
  if (!all(ok)) {
    stop("failed to persist reference output from ", src_dir,
         " to ", dst_dir, call. = FALSE)
  }
  invisible(dst_dir)
}

# Peak_mb from package_verify baseline rows is only comparable to patched
# subprocess RSS when the baseline was measured by verify_patch / zyme attest
# (empty note, or "cached baseline <ts>" from a prior attest). Rows imported
# from legacy benchmarks use gc() max-used (note prefix absorbed:/migrated:)
# and must not drive peak_mb_change_pct on patched rows.
.is_subprocess_peak_baseline_note <- function(note) {
  note <- trimws(as.character(note))
  if (!nzchar(note) || is.na(note)) {
    return(TRUE)
  }
  if (grepl("^absorbed:", note, ignore.case = TRUE)) {
    return(FALSE)
  }
  if (grepl("^migrated:", note, ignore.case = TRUE)) {
    return(FALSE)
  }
  TRUE
}

.cached_baseline_stats <- function(task_dir, name, tier) {
  tsv_path <- file.path(task_dir, "package_verify.tsv")
  if (!file.exists(tsv_path)) {
    stop("package_verify.tsv not found; cannot reuse baseline timing: ",
         tsv_path, call. = FALSE)
  }
  df <- tryCatch(
    utils::read.table(tsv_path, header = TRUE, sep = "\t", quote = "",
                      comment.char = "", stringsAsFactors = FALSE,
                      na.strings = c("", "NA"), fill = TRUE,
                      colClasses = "character"),
    error = function(e) {
      stop("could not read package_verify.tsv: ", conditionMessage(e),
           call. = FALSE)
    }
  )
  if (!nrow(df) || !all(c("timestamp", "patch_name", "tier", "variant", "sec") %in% names(df))) {
    stop("package_verify.tsv has no long-format baseline rows", call. = FALSE)
  }
  keep <- df$patch_name == name & df$tier == tier & df$variant == "baseline"
  df <- df[keep, , drop = FALSE]
  if (!nrow(df)) {
    stop("no cached baseline timing for patch '", name, "', tier '", tier,
         "' in ", tsv_path, call. = FALSE)
  }

  # no_baseline_confirm deliberately skips a fresh upstream confirmation run,
  # so never let it borrow baselines from another host or thread setting.
  # This matters on Windows where Rblas/OpenBLAS selection changes the thread
  # row semantics and old package_verify files may mix macOS + Windows rows.
  sysinfo <- .collect_system_info()
  required_cols <- c("system_os", "system_cpu", "system_ram_gb", "system_threads")
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols)) {
    stop(
      "cached baseline rows for patch '", name, "', tier '", tier,
      "' are missing system columns required for exact reuse: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }
  .same_cell <- function(col, value) {
    x <- as.character(df[[col]])
    x[is.na(x)] <- ""
    identical(value, "") | x == value
  }
  exact <- .same_cell("system_os", sysinfo$system_os) &
           .same_cell("system_cpu", sysinfo$system_cpu) &
           .same_cell("system_ram_gb", sysinfo$system_ram_gb) &
           .same_cell("system_threads", sysinfo$system_threads)
  df <- df[exact, , drop = FALSE]
  if (!nrow(df)) {
    stop(
      "no exact cached baseline timing for patch '", name, "', tier '", tier,
      "' on current host/thread (",
      "system_os=", sysinfo$system_os,
      ", system_cpu=", sysinfo$system_cpu,
      ", system_ram_gb=", sysinfo$system_ram_gb,
      ", system_threads=", sysinfo$system_threads,
      ") in ", tsv_path,
      call. = FALSE
    )
  }

  secs <- suppressWarnings(as.numeric(df$sec))
  ok <- is.finite(secs) & secs > 0 & nzchar(df$timestamp)
  df <- df[ok, , drop = FALSE]
  if (!nrow(df)) {
    stop("cached baseline timing rows for tier '", tier,
         "' have no finite positive sec values", call. = FALSE)
  }
  note_col <- if ("note" %in% names(df)) as.character(df$note) else rep("", nrow(df))
  comparable <- vapply(note_col, .is_subprocess_peak_baseline_note, logical(1L))
  df_comp <- df[comparable, , drop = FALSE]
  if (!nrow(df_comp)) {
    stop(
      "cached baseline for patch '", name, "', tier '", tier,
      "' has only non-comparable peak_mb rows (note absorbed:/migrated:). ",
      "Re-run verify_patch without no_baseline_confirm to measure baseline ",
      "in the same subprocess RSS protocol as patched.",
      call. = FALSE
    )
  }
  latest <- max(df_comp$timestamp)
  df_latest <- df_comp[df_comp$timestamp == latest, , drop = FALSE]
  secs <- suppressWarnings(as.numeric(df_latest$sec))
  secs <- secs[is.finite(secs) & secs > 0]
  peaks <- if ("peak_mb" %in% names(df_latest)) {
    suppressWarnings(as.numeric(df_latest$peak_mb))
  } else {
    numeric(0)
  }
  peaks <- peaks[is.finite(peaks) & peaks > 0]
  list(secs = secs, peaks_mb = peaks, timestamp = latest)
}

.verify_one_tier <- function(patch, name, task_dir, tier, thresholds,
                             evaluate_path, reps, verbose,
                             intrinsic_noise = list(),
                             no_baseline_confirm = FALSE,
                             patched_only = FALSE) {
  # Auto-escalation: when reps == 2, the loop may grow `target` by 1 if the
  # two reps' speedup_x disagree by >20% (max/min > 1.20). Explicit reps>=3
  # disables this -- caller asked for a fixed sample size and gets exactly it.
  baseline_secs <- numeric(0)
  patched_secs <- numeric(0)
  baseline_peaks_mb <- numeric(0)
  patched_peaks_mb <- numeric(0)
  per_rep_results <- list()
  per_rep_all_pass <- logical(0)
  cached_baseline <- NULL
  cached_ref_source <- NULL

  if (isTRUE(no_baseline_confirm)) {
    cached_baseline <- .cached_baseline_stats(task_dir, name, tier)
    baseline_secs <- cached_baseline$secs
    baseline_peaks_mb <- cached_baseline$peaks_mb
    cached_ref_source <- .cached_reference_dir(task_dir, tier)
  }

  # patched-only mode: measure ONLY the patch (no baseline, no concordance).
  # For cells whose baseline OOMs on this machine but whose patch fits -- proves
  # the optimized version runs here and records its wall-time + peak RSS.
  # speedup/pass are left NA (no local baseline); correctness is verified
  # separately on a box that can run the baseline. evaluate.R (concordance) is
  # skipped, so no reference output is needed.
  if (isTRUE(patched_only)) {
    po_secs <- numeric(0)
    po_peaks <- numeric(0)
    for (po_rep in seq_len(reps)) {
      if (verbose) cat(sprintf("=== rep %d/%d (patched-only) ===\n", po_rep, reps))
      temp_dir <- tempfile(".autozyme_verify_", tmpdir = task_dir)
      pipeline_dir <- file.path(temp_dir, "pipeline")
      dir.create(pipeline_dir, recursive = TRUE)
      if (verbose) cat(sprintf(
        "--- patched (autozyme::%s) [patched-only, no baseline] ---\n", name))
      p_res <- .run_worker(name, task_dir, tier, pipeline_dir,
                           activate = TRUE, verbose = verbose)
      po_secs[po_rep] <- p_res$elapsed_sec
      po_peaks[po_rep] <- p_res$peak_mb
      if (verbose) {
        peak_s <- if (is.finite(p_res$peak_mb))
          sprintf("  peak_rss: %.1f MB", p_res$peak_mb) else ""
        cat(sprintf("  time: %.3f sec%s\n", p_res$elapsed_sec, peak_s))
      }
      tryCatch(
        .append_one_pv_row(task_dir, name, tier, "patched", po_rep,
                           sec = p_res$elapsed_sec, peak_mb = p_res$peak_mb,
                           note = "patched-only"),
        error = function(e) message("checkpoint patched-only write failed: ",
                                    conditionMessage(e))
      )
      unlink(temp_dir, recursive = TRUE)
    }
    pt_peaks <- po_peaks[is.finite(po_peaks)]
    if (verbose) {
      cat(sprintf("\n--- timing (patched-only, median of %d reps) ---\n",
                  length(po_secs)))
      cat(sprintf("  patched:  %.3f sec  (reps: %s)\n",
                  stats::median(po_secs),
                  paste(sprintf("%.2f", po_secs), collapse = ", ")))
      cat("  baseline: SKIPPED (patched-only) -- speedup NA, concordance NA\n")
    }
    return(list(
      tier              = tier,
      baseline_sec      = NA_real_,
      patched_sec       = stats::median(po_secs),
      baseline_peak_mb  = NA_real_,
      patched_peak_mb   = if (length(pt_peaks)) stats::median(pt_peaks) else NA_real_,
      metrics           = list(),
      all_pass          = NA,
      reps              = length(po_secs),
      baseline_secs     = numeric(0),
      patched_secs      = po_secs,
      baseline_peaks_mb = numeric(0),
      patched_peaks_mb  = po_peaks,
      per_rep_pass      = rep(NA, length(po_secs)),
      baseline_cached   = FALSE,
      baseline_cache_timestamp = ""
    ))
  }

  # Per-rep evaluate.R runner. Each call gets its own on.exit frame so cwd
  # and ZYME_* env vars are cleanly restored between reps — registering the
  # on.exit handlers inside the while loop instead would accumulate one
  # handler per iteration, all referencing the same outer-scope `.old_env`
  # variable (which by the final iteration captures a post-Sys.setenv polluted
  # state), and would leak ZYME_TIER / ZYME_REFERENCE_DIR / ZYME_TEST_DIR
  # back to the caller after verify_patch returns.
  #
  # NOTE: system2(env = ...) on Windows prepends env strings to the command
  # as positional args (Rscript then can't find the file). Use Sys.setenv +
  # on.exit restore for cross-platform behavior.
  #
  # ZYME_TEST_DIR = pipeline_dir matches the autozyme convention (smoke save
  # writes to `dir/result.rds`). evaluate.R variants that default to
  # `pipeline/output_<tier>` (e.g. find_all_markers/v2) would otherwise miss
  # the patched output the smoke recipe placed at `dir/result.rds`.
  .run_evaluate <- function(temp_dir, ref_dir, pipeline_dir) {
    old_wd <- getwd()
    on.exit(setwd(old_wd), add = TRUE)
    setwd(temp_dir)
    keys <- c("ZYME_TIER", "ZYME_REFERENCE_DIR", "ZYME_TEST_DIR")
    old_env <- Sys.getenv(keys, unset = NA)
    on.exit({
      for (k in names(old_env)) {
        if (is.na(old_env[[k]])) Sys.unsetenv(k)
        else do.call(Sys.setenv, setNames(list(old_env[[k]]), k))
      }
    }, add = TRUE)
    Sys.setenv(ZYME_TIER = tier, ZYME_REFERENCE_DIR = ref_dir,
               ZYME_TEST_DIR = pipeline_dir)
    system2(
      "Rscript",
      args = shQuote(file.path(temp_dir, "evaluate.R")),
      stdout = TRUE, stderr = TRUE
    )
  }

  target <- reps
  rep <- 0L
  while (rep < target) {
    rep <- rep + 1L
    if (verbose && target > 1L) {
      cat(sprintf("=== rep %d/%d ===\n", rep, target))
    }

    # Anchor temp_dir under task_dir, mirroring Python's verify_patch. Two
    # reasons: (1) the evaluate.R upward walk for autozyme-framework works
    # without symlinks/junctions when temp_dir is a descendant of the same
    # drive as the framework; (2) Windows junctions can't cross volumes --
    # default R temp on C:\ vs framework on D:\ would silently fail to
    # link, breaking lookups.
    temp_dir <- tempfile(".autozyme_verify_", tmpdir = task_dir)
    pipeline_dir <- file.path(temp_dir, "pipeline")
    ref_dir <- file.path(temp_dir, sprintf("reference_output_%s", tier))
    dir.create(pipeline_dir, recursive = TRUE)
    dir.create(ref_dir,      recursive = TRUE)

    # Some tasks' evaluate.R resolves autozyme-framework by walking up from
    # SCRIPT_DIR (= temp_dir under our staging). /tmp has no such ancestor,
    # so source(...helpers.R) would fail. Stage a symlink from temp_dir to
    # the category-level autozyme-framework alongside the task; the upward
    # walk then resolves on the first iteration. No-op for tasks that use
    # the more common `file.path(TASK_DIR, "..", "autozyme-framework")` form.
    fw_link <- file.path(dirname(task_dir), "autozyme-framework")
    if (file.exists(fw_link)) {
      .fw_target <- normalizePath(fw_link, mustWork = TRUE)
      .fw_link_dst <- file.path(temp_dir, "autozyme-framework")
      if (.Platform$OS.type == "windows") {
        # On Windows, file.symlink requires SeCreateSymbolicLinkPrivilege
        # (admin or Developer Mode). Use a junction instead -- junctions
        # cover the upward-walk lookup, work for any user, and only span
        # local volumes (autozyme-framework is always on the same disk
        # as the task dir in practice).
        system2("cmd.exe",
                c("/c", "mklink", "/J",
                  shQuote(gsub("/", "\\\\", .fw_link_dst)),
                  shQuote(gsub("/", "\\\\", .fw_target))),
                stdout = FALSE, stderr = FALSE)
      } else {
        tryCatch(
          file.symlink(.fw_target, .fw_link_dst),
          error = function(e) invisible(NULL)
        )
      }
      rm(.fw_target, .fw_link_dst)
    }

    if (isTRUE(no_baseline_confirm)) {
      .copy_cached_reference_output(cached_ref_source, ref_dir)
      if (verbose) {
        if (rep == 1L) {
          peak_s <- if (length(baseline_peaks_mb)) {
            sprintf("  peak_rss reps: %s",
                    paste(sprintf("%.1f", baseline_peaks_mb), collapse = ", "))
          } else {
            ""
          }
          cat(sprintf("--- baseline (cached upstream %s; no rerun) ---\n", patch$upstream))
          cat(sprintf("  timestamp: %s\n", cached_baseline$timestamp))
          cat(sprintf("  median time: %.3f sec  (cached reps: %s)%s\n",
                      stats::median(baseline_secs),
                      paste(sprintf("%.2f", baseline_secs), collapse = ", "),
                      peak_s))
        }
      }
    } else {
      if (verbose) cat(sprintf("--- baseline (upstream %s) ---\n", patch$upstream))
      b_res <- .run_worker(
        name, task_dir, tier, ref_dir, activate = FALSE, verbose = verbose
      )
      baseline_secs[rep] <- b_res$elapsed_sec
      baseline_peaks_mb[rep] <- b_res$peak_mb
      if (verbose) {
        peak_s <- if (is.finite(b_res$peak_mb))
          sprintf("  peak_rss: %.1f MB", b_res$peak_mb) else ""
        cat(sprintf("  time: %.3f sec%s\n", b_res$elapsed_sec, peak_s))
      }
      .persist_reference_output(ref_dir, task_dir, tier)
      tryCatch(
        .append_one_pv_row(task_dir, name, tier, "baseline", rep,
                           sec = b_res$elapsed_sec, peak_mb = b_res$peak_mb),
        error = function(e) message("checkpoint baseline write failed: ",
                                    conditionMessage(e))
      )
    }

    if (verbose) cat(sprintf("--- patched (autozyme::%s) ---\n", name))
    p_res <- .run_worker(
      name, task_dir, tier, pipeline_dir, activate = TRUE, verbose = verbose
    )
    patched_secs[rep] <- p_res$elapsed_sec
    patched_peaks_mb[rep] <- p_res$peak_mb
    if (verbose) {
      peak_s <- if (is.finite(p_res$peak_mb))
        sprintf("  peak_rss: %.1f MB", p_res$peak_mb) else ""
      cat(sprintf("  time: %.3f sec%s\n", p_res$elapsed_sec, peak_s))
    }

    file.copy(evaluate_path, file.path(temp_dir, "evaluate.R"), overwrite = TRUE)

    # Run evaluate.R with cwd = temp_dir so task scripts that resolve
    # PIPE_DIR via `basename(getwd()) == "pipeline"` (or fall back to
    # file.path(getwd(), "pipeline")) see the staged temp_dir layout.
    # Scripts that use --file= parsing for SCRIPT_DIR are unaffected.
    # `.run_evaluate` owns its own on.exit frame -- see definition above.
    out <- .run_evaluate(temp_dir, ref_dir, pipeline_dir)
    status <- attr(out, "status")
    if (!is.null(status) && status != 0L) {
      unlink(temp_dir, recursive = TRUE)
      stop("evaluate.R exited non-zero:\n", paste(out, collapse = "\n"))
    }

    metric_pat <- "^([a-zA-Z_][a-zA-Z0-9_]*):\\s*([-+0-9.eE]+)\\s*$"
    metrics <- list()
    for (line in out) {
      m <- regmatches(line, regexec(metric_pat, line))[[1]]
      if (length(m) == 3L) metrics[[m[2]]] <- as.numeric(m[3])
    }

    rep_results <- list()
    rep_all_pass <- TRUE
    for (t in thresholds) {
      nm <- t$name
      if (!nm %in% names(metrics)) {
        unlink(temp_dir, recursive = TRUE)
        stop(sprintf("metric '%s' declared in task.yaml not printed by evaluate.R", nm))
      }
      val <- metrics[[nm]]
      th_info <- .effective_threshold(t, tier, intrinsic_noise)
      th  <- th_info$value
      op  <- t$comparator
      pass <- switch(op,
        gte = val >= th,
        lte = val <= th,
        stop(sprintf("unknown comparator '%s' for metric '%s'", op, nm))
      )
      if (!pass) rep_all_pass <- FALSE
      rep_results[[nm]] <- list(value = val, threshold = th,
                                comparator = op, pass = pass,
                                threshold_label = th_info$label)
    }
    per_rep_results[[rep]] <- rep_results
    per_rep_all_pass[rep] <- rep_all_pass

    {
      finite_bs <- baseline_secs[is.finite(baseline_secs)]
      bl_med_sec <- if (length(finite_bs)) stats::median(finite_bs) else NA_real_
      finite_bp <- baseline_peaks_mb[is.finite(baseline_peaks_mb) & baseline_peaks_mb > 0]
      bl_med_mb <- if (length(finite_bp)) stats::median(finite_bp) else NA_real_
      sp_x_val <- NA_real_; sp_pct_val <- NA_real_
      if (is.finite(bl_med_sec) && bl_med_sec > 0 &&
          is.finite(p_res$elapsed_sec) && p_res$elapsed_sec > 0) {
        sp_x_val   <- bl_med_sec / p_res$elapsed_sec
        sp_pct_val <- (bl_med_sec - p_res$elapsed_sec) / bl_med_sec * 100
      }
      pk_fold_val <- NA_real_; pk_pct_val <- NA_real_
      if (is.finite(bl_med_mb) && bl_med_mb > 0 &&
          is.finite(p_res$peak_mb) && p_res$peak_mb > 0) {
        pk_fold_val <- bl_med_mb / p_res$peak_mb
        pk_pct_val  <- (bl_med_mb - p_res$peak_mb) / bl_med_mb * 100
      }
      ckpt_note <- if (isTRUE(no_baseline_confirm))
        sprintf("cached baseline %s",
                if (is.null(cached_baseline)) "" else cached_baseline$timestamp)
      else ""
      tryCatch(
        .append_one_pv_row(task_dir, name, tier, "patched", rep,
                           sec = p_res$elapsed_sec, peak_mb = p_res$peak_mb,
                           speedup_pct = sp_pct_val, speedup_x = sp_x_val,
                           peak_pct = pk_pct_val, peak_fold = pk_fold_val,
                           rep_pass = rep_all_pass,
                           metrics_json = .metrics_to_json(rep_results),
                           note = ckpt_note),
        error = function(e) message("checkpoint patched write failed: ",
                                    conditionMessage(e))
      )
    }

    unlink(temp_dir, recursive = TRUE)

    # Auto-escalation check: after the 2nd rep on a target=2 run, look at the
    # spread of per-rep speedup_x. If >20% disagreement, run one more rep.
    if (identical(rep, 2L) && identical(target, 2L)) {
      ratios <- if (isTRUE(no_baseline_confirm)) {
        stats::median(baseline_secs) / patched_secs
      } else {
        baseline_secs / patched_secs
      }
      ratios <- ratios[is.finite(ratios) & ratios > 0]
      if (length(ratios) >= 2L) {
        spread <- max(ratios) / min(ratios)
        if (spread > 1.20) {
          target <- 3L
          if (verbose) cat(sprintf(
            "  -> auto-escalate: speedup spread %.2fx > 1.20; running rep 3\n",
            spread
          ))
        }
      }
    }
  }

  reps_actual  <- length(patched_secs)
  baseline_sec <- stats::median(baseline_secs)
  patched_sec  <- stats::median(patched_secs)
  bl_peaks <- baseline_peaks_mb[is.finite(baseline_peaks_mb)]
  pt_peaks <- patched_peaks_mb[is.finite(patched_peaks_mb)]
  baseline_peak_mb <- if (length(bl_peaks)) stats::median(bl_peaks) else NA_real_
  patched_peak_mb  <- if (length(pt_peaks)) stats::median(pt_peaks) else NA_real_
  results      <- per_rep_results[[1]]
  all_pass     <- all(per_rep_all_pass)

  if (verbose) {
    cat("\n--- metrics (rep 1) ---\n")
    op_str <- function(op) if (op == "gte") ">=" else "<="
    for (nm in names(results)) {
      r <- results[[nm]]
      label <- if (is.null(r$threshold_label)) "absolute" else r$threshold_label
      suffix <- if (identical(label, "absolute")) "" else sprintf("  [%s]", label)
      cat(sprintf("  %-32s %.6f   %s %.3f   %s%s\n",
                  nm, r$value, op_str(r$comparator), r$threshold,
                  if (r$pass) "PASS" else "FAIL", suffix))
    }
    cat(sprintf("\n--- timing (median of %d rep%s) ---\n",
                reps_actual, if (reps_actual > 1L) "s" else ""))
    baseline_label <- if (isTRUE(no_baseline_confirm)) "baseline (cached)" else "baseline"
    cat(sprintf("  %s: %.3f sec  (reps: %s)\n", baseline_label, baseline_sec,
                paste(sprintf("%.2f", baseline_secs), collapse = ", ")))
    cat(sprintf("  patched:  %.3f sec  (reps: %s)\n", patched_sec,
                paste(sprintf("%.2f", patched_secs), collapse = ", ")))
    speedup_pct <- (baseline_sec - patched_sec) / baseline_sec * 100
    cat(sprintf("  speedup:  %.1f%% (%.1fx)\n",
                speedup_pct, baseline_sec / patched_sec))
    if (is.finite(baseline_peak_mb) || is.finite(patched_peak_mb)) {
      bp_s <- if (length(bl_peaks)) paste(sprintf("%.1f", bl_peaks), collapse = ", ") else "--"
      pp_s <- if (length(pt_peaks)) paste(sprintf("%.1f", pt_peaks), collapse = ", ") else "--"
      cat(sprintf("  peak_rss baseline: %.1f MB  (reps: %s)\n",
                  baseline_peak_mb, bp_s))
      cat(sprintf("  peak_rss patched:  %.1f MB  (reps: %s)\n",
                  patched_peak_mb, pp_s))
      if (is.finite(baseline_peak_mb) && is.finite(patched_peak_mb) &&
          baseline_peak_mb > 0) {
        cat(sprintf("  peak_rss saving:   %+.1f%%\n",
                    (baseline_peak_mb - patched_peak_mb) / baseline_peak_mb * 100))
      }
    }
    if (reps_actual > 1L) {
      cat(sprintf("  all reps pass: %s (%d/%d)\n",
                  if (all_pass) "yes" else "NO",
                  sum(per_rep_all_pass), reps_actual))
    }
    cat(sprintf("\n--- verdict (%s): %s ---\n", tier,
                if (all_pass) "ALL PASS" else "SOME FAIL"))
  }

  list(
    baseline_sec      = baseline_sec,
    patched_sec       = patched_sec,
    baseline_peak_mb  = baseline_peak_mb,
    patched_peak_mb   = patched_peak_mb,
    metrics           = results,
    all_pass          = all_pass,
    reps              = reps_actual,
    baseline_secs     = baseline_secs,
    patched_secs      = patched_secs,
    baseline_peaks_mb = baseline_peaks_mb,
    patched_peaks_mb  = patched_peaks_mb,
    per_rep_pass      = per_rep_all_pass,
    baseline_cached   = isTRUE(no_baseline_confirm),
    baseline_cache_timestamp = if (is.null(cached_baseline)) "" else cached_baseline$timestamp
  )
}


#' Verify a registered patch end-to-end against an autozyme task
#'
#' Runs the patch's smoke recipe across one or more dataset tiers and
#' produces a per-tier summary table. Each tier's run does its own baseline
#' + patched measurement (each spawned in a fresh-Rscript subprocess), saves
#' outputs, then runs the task's evaluate.R once on the saved pair, with
#' \code{reps} repeats to mitigate timing noise. \code{all_pass} requires
#' every rep at every tier to pass thresholds.
#'
#' Default \code{tiers} are designed for both CI and external reporting:
#' \itemize{
#'   \item \code{tiny}: best-case algorithmic speedup (smallest dev tier)
#'   \item \code{medium}: primary iterate-loop tier (in-distribution)
#'   \item \code{large}: dev-set max scale (in-distribution scaling check)
#'   \item \code{ood_large}: held-out OOD at dev-tier scale (generalization)
#'   \item \code{ood_xlarge}: production-scale OOD (full generalization + scale)
#' }
#' For CI smoke tests, pass \code{tiers = "tiny"} explicitly. A tier whose
#' dataset is missing is reported with NA values and a skip note (common
#' for naturally-bounded functions without an \code{ood_xlarge} candidate).
#'
#' Side effect: appends per-tier rows to
#' \code{<task_dir>/package_verify.tsv} (header written on first call).
#' This is the persisted form of the paper-headline "package" number --
#' fresh-subprocess timings matched in thread config, distinct from
#' \code{results.tsv} (iter, in-process) and \code{verify.tsv} (Phase 3
#' validate, threading x OOD sweep). Each invocation appends; the file
#' accumulates history like \code{results.tsv}.
#'
#' @param name Registered patch name.
#' @param task_dir Path to the autozyme task directory (must contain
#'   \code{task.yaml}, \code{evaluate.R}, and the patch's smoke-load data).
#' @param tiers Character vector of tier names. Default
#'   \code{c("tiny", "medium", "large", "ood_large", "ood_xlarge")}.
#' @param reps Number of measurement reps per tier. Each rep spawns two
#'   fresh Rscript subprocesses (baseline + patched). Timing is the median
#'   across reps; \code{all_pass} requires every rep to pass thresholds.
#'   Default 2 -- fast-path for stable patches. When \code{reps == 2}, an
#'   auto-escalation kicks in: if the two reps' speedup_x disagree by >20%
#'   (max/min ratio > 1.20), one extra rep is added to stabilize the median.
#'   Explicit \code{reps >= 3} disables this -- the caller asked for a fixed
#'   sample size and gets exactly that.
#' @param no_baseline_confirm If TRUE, reuse the latest baseline timing from
#'   \code{package_verify.tsv} and the saved \code{reference_outputs/<tier>/}
#'   files, running only patched subprocesses. This skips environmental drift
#'   detection and requires prior baseline artifacts.
#' @param verbose If TRUE (default), prints per-tier verbose progress and
#'   a final summary table.
#' @return Invisible data frame with one row per tier and columns
#'   \code{tier}, \code{baseline_sec}, \code{patched_sec},
#'   \code{speedup_x}, \code{all_pass}, \code{reps}, \code{note}.
#' @export
verify_patch <- function(name, task_dir,
                         tiers = c("tiny", "medium", "large",
                                   "ood_large", "ood_xlarge"),
                         reps = 2, verbose = TRUE,
                         no_baseline_confirm = FALSE,
                         patched_only = FALSE) {
  if (!.ensure_registered(name)) {
    stop(sprintf(
      "patch '%s' could not be registered (upstream namespace likely missing)",
      name))
  }
  patch <- .zyme_registry[[name]]
  task_dir_norm <- normalizePath(task_dir, mustWork = TRUE)
  if (is.null(.resolve_smoke(task_dir_norm, patch))) {
    stop(sprintf(
      "no smoke recipe for patch '%s' -- add <task_dir>/attest/smoke.R or register_patch(smoke=...)",
      name), call. = FALSE)
  }
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop("verify_patch requires the 'yaml' package; install.packages('yaml')")
  }
  reps <- as.integer(reps)
  stopifnot(length(reps) == 1L, !is.na(reps), reps >= 1L)
  stopifnot(is.character(tiers), length(tiers) >= 1L)

  task_dir <- normalizePath(task_dir, mustWork = TRUE)
  yaml_path <- file.path(task_dir, "task.yaml")
  evaluate_path <- file.path(task_dir, "evaluate.R")
  if (!file.exists(yaml_path))     stop("task.yaml not found at ", yaml_path)
  if (!file.exists(evaluate_path)) stop("evaluate.R not found at ", evaluate_path)

  .ensure_task_thread_env(task_dir)

  task <- yaml::read_yaml(yaml_path)
  thresholds <- task$metrics
  if (is.null(thresholds) || !length(thresholds)) {
    stop("task.yaml has no `metrics:` section")
  }
  intrinsic_noise <- if (is.null(task$intrinsic_noise)) list() else task$intrinsic_noise

  rows <- vector("list", length(tiers))
  tsv_rows <- vector("list", length(tiers))
  for (i in seq_along(tiers)) {
    tier <- tiers[i]
    ds_label <- .dataset_name_for_tier(task_dir, tier)
    tier_label <- if (nzchar(ds_label)) sprintf("%s: %s", tier, ds_label) else tier
    if (verbose) cat(sprintf("\n############ tier = %s ############\n", tier_label))
    res <- tryCatch(
      .verify_one_tier(patch, name, task_dir, tier, thresholds,
                       evaluate_path, reps, verbose,
                       intrinsic_noise = intrinsic_noise,
                       no_baseline_confirm = no_baseline_confirm,
                       patched_only = patched_only),
      error = function(e) list(.skip_reason = conditionMessage(e))
    )
    if (!is.null(res$.skip_reason)) {
      if (verbose) cat(sprintf("  [skip] %s: %s\n", tier, res$.skip_reason))
      rows[[i]] <- data.frame(
        tier             = tier,
        baseline_sec     = NA_real_,
        patched_sec      = NA_real_,
        speedup_x        = NA_real_,
        all_pass         = NA,
        reps             = NA_integer_,
        baseline_peak_mb = NA_real_,
        patched_peak_mb  = NA_real_,
        note             = res$.skip_reason,
        stringsAsFactors = FALSE
      )
      tsv_rows[[i]] <- list(
        tier              = tier,
        baseline_secs     = numeric(0),
        patched_secs      = numeric(0),
        baseline_peaks_mb = numeric(0),
        patched_peaks_mb  = numeric(0),
        per_rep_pass      = logical(0),
        metrics_json      = "",
        note              = res$.skip_reason
      )
    } else {
      note <- if (isTRUE(res$baseline_cached)) {
        sprintf("cached baseline %s", res$baseline_cache_timestamp)
      } else {
        ""
      }
      rows[[i]] <- data.frame(
        tier             = tier,
        baseline_sec     = res$baseline_sec,
        patched_sec      = res$patched_sec,
        speedup_x        = res$baseline_sec / res$patched_sec,
        all_pass         = res$all_pass,
        reps             = res$reps,
        baseline_peak_mb = res$baseline_peak_mb,
        patched_peak_mb  = res$patched_peak_mb,
        note             = note,
        stringsAsFactors = FALSE
      )
      tsv_rows[[i]] <- list(
        tier              = tier,
        baseline_secs     = if (isTRUE(res$baseline_cached)) numeric(0) else res$baseline_secs,
        patched_secs      = res$patched_secs,
        baseline_peaks_mb = if (isTRUE(res$baseline_cached)) numeric(0) else res$baseline_peaks_mb,
        patched_peaks_mb  = res$patched_peaks_mb,
        per_rep_pass      = if (is.null(res$per_rep_pass)) logical(0) else res$per_rep_pass,
        metrics_json      = .metrics_to_json(res$metrics),
        note              = note,
        baseline_sec_for_speedup = if (isTRUE(res$baseline_cached)) res$baseline_sec else NULL,
        baseline_peak_mb_for_speedup = if (isTRUE(res$baseline_cached)) res$baseline_peak_mb else NULL
      )
    }
  }
  out <- do.call(rbind, rows)

  # Per-rep streaming via .append_one_pv_row already persisted success rows
  # to pv.tsv. Only skip-sentinel rows (tiers that crashed before producing
  # any patched timing) still need writing here.
  skip_tsv_rows <- Filter(
    function(r) length(r$patched_secs) == 0L && length(r$baseline_secs) == 0L,
    tsv_rows
  )
  tryCatch(
    if (length(skip_tsv_rows) > 0L)
      .append_package_verify_tsv(task_dir, name, skip_tsv_rows),
    error = function(e) message(
      "warning: could not write package_verify.tsv: ",
      conditionMessage(e)
    )
  )

  if (verbose) {
    cat(sprintf("\n############ verify_patch: %s ############\n", name))
    cat(sprintf("%-28s  %12s  %12s  %10s  %8s  %8s  %-8s  %s\n",
                "tier:dataset", "baseline_sec", "patched_sec", "speedup",
                "bl_mb", "pt_mb", "all_pass", "note"))
    for (i in seq_len(nrow(out))) {
      r <- out[i, ]
      ds <- .dataset_name_for_tier(task_dir, r$tier)
      tier_label <- if (nzchar(ds)) sprintf("%s: %s", r$tier, ds) else r$tier
      if (is.na(r$baseline_sec)) {
        cat(sprintf("%-28s  %12s  %12s  %10s  %8s  %8s  %-8s  %s\n",
                    tier_label, "--", "--", "--", "--", "--", "--", r$note))
      } else {
        bl_s <- if (is.finite(r$baseline_peak_mb))
          sprintf("%8.1f", r$baseline_peak_mb) else sprintf("%8s", "--")
        pt_s <- if (is.finite(r$patched_peak_mb))
          sprintf("%8.1f", r$patched_peak_mb) else sprintf("%8s", "--")
        cat(sprintf("%-28s  %12.3f  %12.3f  %9.1fx  %s  %s  %-8s  %s\n",
                    tier_label, r$baseline_sec, r$patched_sec, r$speedup_x,
                    bl_s, pt_s,
                    if (isTRUE(r$all_pass)) "yes" else "NO", r$note))
      }
    }
  }

  invisible(out)
}
