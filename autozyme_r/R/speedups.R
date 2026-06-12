#' Bundled finalized "package speedup" snapshots for each patch
#'
#' Reads the bundled finalized speedup TSV that ships with the package and
#' returns one summary row per finalized platform/thread/tier cell.
#'
#' Storage layout: \code{inst/patches/<name>/speedups_finalized.tsv}. Raw
#' \code{speedups.tsv} histories stay in the framework repository and are not
#' shipped in the release package.
#'
#' "Package speedup only": these numbers are \emph{exclusively} the
#' fresh-subprocess \code{verify_patch} protocol output — the headline
#' number that ships to the paper. They are NOT the in-process iter
#' speedups from \code{results.tsv} and NOT the threading-sweep numbers
#' from \code{verify.tsv}. See \code{4_package.md} ("three speedup
#' contexts") for the methodology distinction.
#'
#' Snapshots are finalized from the framework-side attest history before each
#' release.
#'
#' Lookup order:
#' \enumerate{
#'   \item \code{inst/patches/<name>/speedups_finalized.tsv} (direct hit).
#'   \item Otherwise scan \code{inst/patches/} for sibling directories whose
#'     name starts with \code{<name>_} and aggregate their TSVs. The umbrella
#'     \code{speedups("seurat")} returns the union of \code{seurat_normalize/},
#'     \code{seurat_scale/}, \code{seurat_markers/}, ... with one extra column
#'     \code{sub_step} tagging the source sub-step name.
#' }
#' The \code{sub_step} column is empty (\code{""}) for direct hits.
#'
#' @param name Registered patch name (same key as \code{activate(name)}), or
#'   an umbrella name whose sub-steps share a \code{<name>_} prefix.
#' @param history Accepted for backward compatibility. Finalized snapshots
#'   contain only release-curated rows, so both values return the same data.
#' @return Data frame with one row per batch: timestamp, patch_name, tier,
#'   dataset (from task.yaml when present in the bundled TSV),
#'   reps, baseline_sec, patched_sec, speedup_x, all_pass,
#'   baseline_secs, patched_secs (";"-joined raw rep arrays — for analysis,
#'   read the raw long-format TSV directly), baseline_peak_mb,
#'   patched_peak_mb, metrics_json, framework_version, note, system_os,
#'   system_cpu, system_ram_gb, system_threads, sub_step. Empty data frame
#'   when no TSV is found by either lookup.
#' @export
speedups <- function(name, history = FALSE) {
  stopifnot(is.character(name), length(name) == 1L, nzchar(name))

  # 1. Direct lookup: inst/patches/<name>/speedups_finalized.tsv.
  direct <- system.file("patches", name, "speedups_finalized.tsv",
                        package = "autozyme")
  if (nzchar(direct) && file.exists(direct)) {
    df <- .load_speedups_tsv(direct)
    if (nrow(df) > 0L) df$sub_step <- ""
    return(.finalize_speedups_df(df))
  }

  # 2. Umbrella fallback: aggregate `<name>_*/speedups_finalized.tsv`. Lets
  #    `speedups("seurat")` rbind across `seurat_normalize/`, `seurat_scale/`,
  #    ... when the umbrella dir itself has no TSV. Each row is tagged with
  #    a `sub_step` column derived from the sub-dir name (prefix stripped).
  patches_root <- system.file("patches", package = "autozyme")
  if (!nzchar(patches_root) || !dir.exists(patches_root)) {
    return(.empty_speedups_df())
  }
  prefix <- paste0("^", name, "_")
  subdirs <- list.dirs(patches_root, full.names = FALSE, recursive = FALSE)
  subdirs <- subdirs[grepl(prefix, subdirs)]
  if (!length(subdirs)) return(.empty_speedups_df())

  parts <- list()
  for (sd in subdirs) {
    tsv <- file.path(patches_root, sd, "speedups_finalized.tsv")
    if (!file.exists(tsv)) next
    df_i <- .load_speedups_tsv(tsv)
    if (nrow(df_i) == 0L) next
    df_i$sub_step <- sub(prefix, "", sd)
    parts[[length(parts) + 1L]] <- df_i
  }
  if (!length(parts)) return(.empty_speedups_df())
  .finalize_speedups_df(do.call(rbind, parts))
}

# Read one finalized TSV and aggregate per (patch, package_version, tier,
# threads, platform, dataset) into one row. Returns a data frame *without*
# the `sub_step` column — caller decides what to put in it.
.load_speedups_tsv <- function(path) {
  long <- utils::read.table(
    path,
    header = TRUE, sep = "\t", quote = "",
    comment.char = "", stringsAsFactors = FALSE,
    na.strings = c("", "NA"), fill = TRUE,
    colClasses = "character"
  )
  if (nrow(long) == 0L) return(.empty_speedups_df_no_substep())

  key_cols <- c("patch", "package_version", "tier",
                "threads", "platform", "dataset")
  batch_id <- do.call(paste, c(long[, key_cols, drop = FALSE], sep = "\x1f"))
  batches <- split(seq_len(nrow(long)), batch_id)

  out_list <- vector("list", length(batches))
  i <- 0L
  for (bid in names(batches)) {
    idx <- batches[[bid]]
    sub <- long[idx, , drop = FALSE]
    base <- sub[sub$variant == "baseline", , drop = FALSE]
    patched <- sub[sub$variant == "patched", , drop = FALSE]
    if (!nrow(base) || !nrow(patched)) next

    b_secs <- .parse_reps(base$sec_reps[1])
    p_secs <- .parse_reps(patched$sec_reps[1])

    pass_rate <- suppressWarnings(as.numeric(patched$pass_rate[1]))
    patched_status <- tolower(patched$status[1])
    if (!is.na(patched_status) && nzchar(patched_status) && patched_status != "ok") {
      all_pass <- FALSE
    } else if (is.finite(pass_rate)) {
      all_pass <- pass_rate >= 1
    } else {
      all_pass <- NA
    }

    i <- i + 1L
    out_list[[i]] <- data.frame(
      timestamp         = .first_nonempty(patched$ts_last[1], patched$ts_first[1]),
      patch_name        = patched$patch[1],
      tier              = base$tier[1],
      dataset           = base$dataset[1],
      reps              = suppressWarnings(as.integer(patched$n_reps[1])),
      baseline_sec      = suppressWarnings(as.numeric(base$sec_mean[1])),
      patched_sec       = suppressWarnings(as.numeric(patched$sec_mean[1])),
      speedup_x         = suppressWarnings(as.numeric(patched$speedup_x_mean[1])),
      all_pass          = all_pass,
      baseline_secs     = paste(sprintf("%.6f", b_secs), collapse = ";"),
      patched_secs      = paste(sprintf("%.6f", p_secs), collapse = ";"),
      baseline_peak_mb  = suppressWarnings(as.numeric(base$mem_mean[1])),
      patched_peak_mb   = suppressWarnings(as.numeric(patched$mem_mean[1])),
      metrics_json      = patched$metrics_json_median[1],
      framework_version = patched$fw_versions[1],
      note              = patched$status[1],
      system_os         = patched$platform[1],
      system_cpu        = "",
      system_ram_gb     = NA_real_,
      system_threads    = suppressWarnings(as.integer(patched$threads[1])),
      stringsAsFactors  = FALSE
    )
  }
  out_list <- out_list[seq_len(i)]
  if (!length(out_list)) return(.empty_speedups_df_no_substep())
  do.call(rbind, out_list)
}

.finalize_speedups_df <- function(df) {
  if (!nrow(df)) return(df)
  ord <- order(df$tier, df$system_os, df$system_threads, df$dataset,
               df$sub_step, df$timestamp)
  df <- df[ord, , drop = FALSE]
  rownames(df) <- NULL
  df
}

.parse_reps <- function(x) {
  if (is.na(x) || !nzchar(x)) return(numeric(0))
  vals <- trimws(strsplit(x, ",", fixed = TRUE)[[1]])
  nums <- suppressWarnings(as.numeric(vals))
  nums[is.finite(nums)]
}

.first_nonempty <- function(...) {
  vals <- list(...)
  for (v in vals) {
    if (!is.na(v) && nzchar(v)) return(v)
  }
  ""
}

.empty_speedups_df_no_substep <- function() {
  data.frame(
    timestamp         = character(0),
    patch_name        = character(0),
    tier              = character(0),
    dataset           = character(0),
    reps              = integer(0),
    baseline_sec      = numeric(0),
    patched_sec       = numeric(0),
    speedup_x         = numeric(0),
    all_pass          = logical(0),
    baseline_secs     = character(0),
    patched_secs      = character(0),
    baseline_peak_mb  = numeric(0),
    patched_peak_mb   = numeric(0),
    metrics_json      = character(0),
    framework_version = character(0),
    note              = character(0),
    system_os         = character(0),
    system_cpu        = character(0),
    system_ram_gb     = numeric(0),
    system_threads    = integer(0),
    stringsAsFactors  = FALSE
  )
}

.empty_speedups_df <- function() {
  df <- .empty_speedups_df_no_substep()
  df$sub_step <- character(0)
  df
}
