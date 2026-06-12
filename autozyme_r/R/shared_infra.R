# Shared runtime controls and matrix helpers used by multiple patches.

.az_truthy <- function(x) {
  if (is.logical(x) && length(x) == 1L && !is.na(x)) return(isTRUE(x))
  if (!is.character(x) || length(x) == 0L || is.na(x[[1L]])) return(NA)
  value <- tolower(trimws(x[[1L]]))
  if (value %in% c("1", "true", "yes", "on")) return(TRUE)
  if (value %in% c("0", "false", "no", "off")) return(FALSE)
  NA
}

.az_feature_key <- function(x) {
  gsub("[^A-Za-z0-9]+", "_", toupper(x))
}

.az_option_values <- function(name) {
  c(getOption(paste0("autozyme.", name), NULL),
    getOption(paste0("autozyme.", gsub("_", ".", name)), NULL))
}

.az_first_truthy <- function(values) {
  for (v in values) {
    parsed <- .az_truthy(v)
    if (!is.na(parsed)) return(parsed)
  }
  NA
}

.az_global_disabled <- function() {
  isTRUE(.az_truthy(Sys.getenv("AUTOZYME_DISABLE", unset = ""))) ||
    isTRUE(.az_truthy(Sys.getenv("AUTOZYME_DISABLED", unset = ""))) ||
    isTRUE(.zyme_state$disabled)
}

.az_feature_enabled <- function(feature, patch = NULL, default = TRUE) {
  if (.az_global_disabled()) return(FALSE)

  feature <- tolower(gsub("[^A-Za-z0-9]+", "_", feature))
  feature_key <- .az_feature_key(feature)
  global <- .az_first_truthy(c(
    Sys.getenv(paste0("AUTOZYME_", feature_key), unset = NA_character_),
    .az_option_values(feature)
  ))
  if (!is.na(global) && !isTRUE(global)) return(FALSE)

  if (!is.null(patch) && nzchar(patch)) {
    patch <- tolower(gsub("[^A-Za-z0-9]+", "_", patch))
    patch_key <- .az_feature_key(patch)
    patch_value <- .az_first_truthy(c(
      Sys.getenv(paste0("AUTOZYME_", patch_key, "_", feature_key),
                 unset = NA_character_),
      .az_option_values(paste0(patch, ".", feature))
    ))
    if (!is.na(patch_value)) return(patch_value)
  }

  if (!is.na(global)) return(global)
  isTRUE(default)
}

.az_as_group <- function(group, ngroups = NULL) {
  if (is.factor(group)) {
    f <- group
  } else if (is.integer(group) && !is.null(ngroups)) {
    f <- factor(group, levels = seq_len(ngroups))
  } else {
    f <- factor(group)
  }
  list(codes = as.integer(f), levels = levels(f), ngroups = nlevels(f))
}

.az_dense_group_summary_fallback <- function(x, group, detect_threshold = 0,
                                             na.rm = FALSE) {
  g <- if (is.list(group) &&
           all(c("codes", "levels", "ngroups") %in% names(group))) {
    group
  } else {
    .az_as_group(group)
  }
  x <- as.matrix(x)
  out_sum <- matrix(0, nrow(x), g$ngroups)
  out_detect <- matrix(0, nrow(x), g$ngroups)
  out_mean <- matrix(NA_real_, nrow(x), g$ngroups)
  group_size <- tabulate(g$codes, nbins = g$ngroups)
  for (j in seq_len(ncol(x))) {
    gj <- g$codes[j]
    if (is.na(gj) || gj < 1L || gj > g$ngroups) next
    col <- x[, j]
    if (isTRUE(na.rm)) {
      keep <- !is.na(col)
      out_sum[keep, gj] <- out_sum[keep, gj] + col[keep]
      out_detect[keep, gj] <- out_detect[keep, gj] +
        as.numeric(col[keep] > detect_threshold)
      out_sum[!keep, gj] <- out_sum[!keep, gj] + 0
    } else if (anyNA(col)) {
      out_sum[, gj] <- out_sum[, gj] + col
      out_detect[, gj] <- out_detect[, gj] +
        as.numeric(col > detect_threshold)
    } else {
      out_sum[, gj] <- out_sum[, gj] + col
      out_detect[, gj] <- out_detect[, gj] +
        as.numeric(col > detect_threshold)
    }
  }
  for (k in seq_len(g$ngroups)) {
    if (group_size[k] > 0L) out_mean[, k] <- out_sum[, k] / group_size[k]
  }
  dimnames(out_sum) <- dimnames(out_detect) <- dimnames(out_mean) <-
    list(rownames(x), g$levels)
  list(sum_by_group = out_sum, mean_by_group = out_mean,
       detected_by_group = out_detect, group_size = group_size)
}

.az_dense_group_summary <- function(x, group, detect_threshold = 0,
                                    na.rm = FALSE, fallback = TRUE,
                                    patch = NULL) {
  g <- .az_as_group(group)
  if (.az_feature_enabled("grouped_reductions", patch = patch, default = TRUE) &&
      is.matrix(x) && is.double(x)) {
    res <- tryCatch(
      az_dense_group_summary_cpp(x, g$codes, g$ngroups,
                                 detect_threshold, isTRUE(na.rm)),
      error = function(e) {
        if (isTRUE(fallback)) return(NULL)
        stop(e)
      }
    )
    if (!is.null(res)) {
      dimnames(res$sum_by_group) <- dimnames(res$mean_by_group) <-
        dimnames(res$detected_by_group) <- list(rownames(x), g$levels)
      return(res)
    }
  }
  .az_dense_group_summary_fallback(x, g, detect_threshold, na.rm)
}

.az_dgc_row_stats_fallback <- function(x, detect_threshold = 0) {
  m <- as.matrix(x)
  list(
    sum = rowSums(m),
    mean = rowMeans(m),
    variance = apply(m, 1L, stats::var),
    nnz = rowSums(m != 0),
    detected = rowSums(m > detect_threshold)
  )
}

.az_dgc_row_stats <- function(x, detect_threshold = 0, fallback = TRUE,
                              patch = NULL) {
  if (.az_feature_enabled("sparse_kernels", patch = patch, default = TRUE) &&
      inherits(x, "dgCMatrix")) {
    res <- tryCatch(
      az_dgc_row_stats_cpp(x, detect_threshold),
      error = function(e) {
        if (isTRUE(fallback)) return(NULL)
        stop(e)
      }
    )
    if (!is.null(res)) {
      rn <- rownames(x)
      if (!is.null(rn)) {
        names(res$sum) <- names(res$mean) <- names(res$variance) <-
          names(res$nnz) <- names(res$detected) <- rn
      }
      return(res)
    }
  }
  .az_dgc_row_stats_fallback(x, detect_threshold)
}

.az_dgc_group_summary <- function(x, group, detect_threshold = 0,
                                  fallback = TRUE, patch = NULL) {
  g <- .az_as_group(group)
  if (.az_feature_enabled("grouped_reductions", patch = patch, default = TRUE) &&
      inherits(x, "dgCMatrix")) {
    res <- tryCatch(
      az_dgc_group_summary_cpp(x, g$codes, g$ngroups, detect_threshold),
      error = function(e) {
        if (isTRUE(fallback)) return(NULL)
        stop(e)
      }
    )
    if (!is.null(res)) {
      dimnames(res$sum_by_group) <- dimnames(res$mean_by_group) <-
        dimnames(res$detected_by_group) <- list(rownames(x), g$levels)
      return(res)
    }
  }
  .az_dense_group_summary_fallback(as.matrix(x), g, detect_threshold)
}

.az_dgc_row_var_standardized <- function(x, mu, sd, vmax, nnz_per_row,
                                         fallback = TRUE, patch = NULL) {
  if (.az_feature_enabled("sparse_kernels", patch = patch, default = TRUE) &&
      inherits(x, "dgCMatrix")) {
    res <- tryCatch(
      turbo_FastSparseRowVarStd(
        p = x@p, i = x@i, x = x@x, nrow = nrow(x), ncol = ncol(x),
        mu = mu, sd = sd, vmax = vmax, nnzPerRow = as.integer(nnz_per_row)),
      error = function(e) {
        if (isTRUE(fallback)) return(NULL)
        stop(e)
      }
    )
    if (!is.null(res)) return(res)
  }
  dense <- as.matrix(x)
  z <- sweep(dense, 1L, mu, "-")
  ok <- is.finite(sd) & sd != 0
  z[ok, ] <- sweep(z[ok, , drop = FALSE], 1L, sd[ok], "/")
  z[!ok, ] <- 0
  z[z > vmax] <- vmax
  z[z < -vmax] <- -vmax
  rowSums(z * z) / (ncol(dense) - 1)
}

.az_dgc_scale_center <- function(x, rows, scale_max = 10, fallback = TRUE,
                                 patch = NULL) {
  if (.az_feature_enabled("sparse_kernels", patch = patch, default = TRUE) &&
      inherits(x, "dgCMatrix")) {
    res <- tryCatch(
      turbo_scale_sparse_full(x, as.integer(rows), scale_max),
      error = function(e) {
        if (isTRUE(fallback)) return(NULL)
        stop(e)
      }
    )
    if (!is.null(res)) return(res)
  }
  dense <- as.matrix(x)[as.integer(rows) + 1L, , drop = FALSE]
  mu <- rowMeans(dense)
  sd <- apply(dense, 1L, stats::sd)
  sd[!is.finite(sd) | sd == 0] <- 1
  out <- sweep(sweep(dense, 1L, mu, "-"), 1L, sd, "/")
  out[out > scale_max] <- scale_max
  out[out < -scale_max] <- -scale_max
  out
}

.az_group_trimean <- function(data, group, fallback = TRUE, patch = NULL) {
  g <- .az_as_group(group)
  if (.az_feature_enabled("grouped_reductions", patch = patch, default = TRUE) &&
      is.matrix(data) && is.double(data)) {
    res <- tryCatch(
      cpp_aggregate_triMean(data, g$codes, g$ngroups),
      error = function(e) {
        if (isTRUE(fallback)) return(NULL)
        stop(e)
      }
    )
    if (!is.null(res)) {
      dimnames(res) <- list(rownames(data), g$levels)
      return(res)
    }
  }
  tri <- function(x) mean(stats::quantile(x, probs = c(0.25, 0.50, 0.50, 0.75),
                                          na.rm = TRUE, names = FALSE))
  out <- stats::aggregate(t(data), list(factor(group, levels = g$levels)), FUN = tri)
  out <- t(out[, -1, drop = FALSE])
  colnames(out) <- g$levels
  rownames(out) <- rownames(data)
  out
}

.az_group_trimean_boot <- function(data, group, permutation,
                                   fallback = TRUE, patch = NULL) {
  g <- .az_as_group(group)
  if (.az_feature_enabled("grouped_reductions", patch = patch, default = TRUE) &&
      is.matrix(data) && is.double(data)) {
    res <- tryCatch(
      cpp_aggregate_triMean_boot(data, g$codes, g$ngroups, permutation),
      error = function(e) {
        if (isTRUE(fallback)) return(NULL)
        stop(e)
      }
    )
    if (!is.null(res)) return(res)
  }
  nboot <- ncol(permutation)
  out <- vector("list", nboot)
  for (b in seq_len(nboot)) {
    out[[b]] <- .az_group_trimean(data, group[permutation[, b]],
                                  fallback = TRUE, patch = patch)
  }
  array(unlist(out, use.names = FALSE),
        dim = c(nrow(data), g$ngroups, nboot))
}

.az_dgc_grouped_wilcox <- function(x, groups, group_sizes,
                                   fallback = TRUE, patch = NULL) {
  if (!.az_feature_enabled("grouped_reductions", patch = patch,
                           default = TRUE)) {
    if (isTRUE(fallback)) return(NULL)
    stop("az_dgc_grouped_wilcox: grouped reductions are disabled",
         call. = FALSE)
  }
  res <- tryCatch(
    parallel_all_in_one_dgc(x_sexp = x, groups = as.integer(groups),
                            group_sizes = as.integer(group_sizes)),
    error = function(e) {
      if (isTRUE(fallback)) return(NULL)
      stop(e)
    }
  )
  if (!is.null(res)) return(res)
  if (isTRUE(fallback)) return(NULL)
  stop("az_dgc_grouped_wilcox: fallback is not implemented for this path",
       call. = FALSE)
}
