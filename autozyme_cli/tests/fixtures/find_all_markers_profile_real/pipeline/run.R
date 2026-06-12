#!/usr/bin/env Rscript
# pipeline/run.R — first-round pipeline mirror for Seurat::FindAllMarkers.
#
# Future optimization rounds edit this file only. Round 0 intentionally calls
# the upstream function with the same input, same arguments, and same output
# format as reference.R.

suppressPackageStartupMessages({
  library(Seurat)
})

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("--file=", args, fixed = TRUE)
  if (length(m) > 0) {
    return(dirname(normalizePath(sub("--file=", "", args[m[1]], fixed = TRUE))))
  }
  getwd()
}

source_helpers <- function(start) {
  cur <- normalizePath(start, mustWork = FALSE)
  repeat {
    candidate <- file.path(cur, "autozyme-framework", "autozyme_cli", "zyme", "helpers.R")
    if (file.exists(candidate)) {
      source(candidate)
      return(invisible(TRUE))
    }
    parent <- dirname(cur)
    if (identical(parent, cur)) {
      stop("autozyme-framework helpers.R not found above ", start)
    }
    cur <- parent
  }
}

SCRIPT_DIR <- get_script_dir()
TASK_DIR <- dirname(SCRIPT_DIR)
source_helpers(TASK_DIR)

orig_FindAllMarkers <- getFromNamespace("FindAllMarkers", "Seurat")

Rcpp::sourceCpp(code = '
// [[Rcpp::depends(RcppParallel)]]
#include <Rcpp.h>
#include <RcppParallel.h>
#include <algorithm>
#include <vector>
using namespace Rcpp;
using namespace RcppParallel;

struct AllInOneWorker : public Worker {
  const std::vector<int>& row_ptr;
  const std::vector<int>& row_col;
  const std::vector<double>& row_val;
  const RVector<int> groups;
  const RVector<int> group_sizes;
  RMatrix<double> pval_out;
  RMatrix<double> sum_out;
  RMatrix<double> count_out;
  int N;
  int G;
  int max_nnz;
  double x1;
  double x2;

  AllInOneWorker(
    const std::vector<int>& row_ptr,
    const std::vector<int>& row_col,
    const std::vector<double>& row_val,
    IntegerVector groups,
    IntegerVector group_sizes,
    NumericMatrix pval_out,
    NumericMatrix sum_out,
    NumericMatrix count_out,
    int N,
    int max_nnz
  ) : row_ptr(row_ptr), row_col(row_col), row_val(row_val),
      groups(groups), group_sizes(group_sizes), pval_out(pval_out),
      sum_out(sum_out), count_out(count_out), N(N), G(group_sizes.size()),
      max_nnz(max_nnz) {
    double n_const = static_cast<double>(N);
    x1 = n_const * n_const * n_const - n_const;
    x2 = 1.0 / (12.0 * (n_const * n_const - n_const));
  }

  void operator()(std::size_t begin, std::size_t end) {
    std::vector<int> ord;
    ord.reserve(max_nnz);
    std::vector<int> nz_count(G);
    std::vector<double> expm1_sum(G);
    std::vector<double> rank_sum(G);

    for (std::size_t feat = begin; feat < end; ++feat) {
      int start = row_ptr[feat];
      int stop = row_ptr[feat + 1];
      int m = stop - start;
      int zero_count = N - m;
      std::fill(nz_count.begin(), nz_count.end(), 0);
      std::fill(expm1_sum.begin(), expm1_sum.end(), 0.0);
      std::fill(rank_sum.begin(), rank_sum.end(), 0.0);
      ord.resize(m);

      for (int k = 0; k < m; ++k) {
        ord[k] = k;
        int g = groups[row_col[start + k]] - 1;
        if (g >= 0 && g < G) {
          nz_count[g]++;
          expm1_sum[g] += std::expm1(row_val[start + k]);
        }
      }

      std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        double xa = row_val[start + a];
        double xb = row_val[start + b];
        if (xa < xb) {
          return true;
        }
        if (xa > xb) {
          return false;
        }
        return row_col[start + a] < row_col[start + b];
      });

      double zero_rank = (zero_count + 1) / 2.0;
      for (int g = 0; g < G; ++g) {
        rank_sum[g] = (static_cast<double>(group_sizes[g]) - nz_count[g]) * zero_rank;
        count_out(feat, g) = static_cast<double>(nz_count[g]);
        sum_out(feat, g) = expm1_sum[g];
      }

      double tie_sum = 0.0;
      if (m > 0 && zero_count > 0) {
        double zc = static_cast<double>(zero_count);
        tie_sum += zc * zc * zc - zc;
      }

      int pos = 0;
      int rank_start = zero_count + 1;
      while (pos < m) {
        int next = pos + 1;
        double val = row_val[start + ord[pos]];
        while (next < m && row_val[start + ord[next]] == val) {
          next++;
        }
        int len = next - pos;
        double avg_rank = rank_start + (len - 1) / 2.0;
        for (int r = pos; r < next; ++r) {
          int g = groups[row_col[start + ord[r]]] - 1;
          if (g >= 0 && g < G) {
            rank_sum[g] += avg_rank;
          }
        }
        if (len > 1 && next < m) {
          double tl = static_cast<double>(len);
          tie_sum += tl * tl * tl - tl;
        }
        rank_start += len;
        pos = next;
      }

      double rhs = (x1 - tie_sum) * x2;
      for (int g = 0; g < G; ++g) {
        double n1 = static_cast<double>(group_sizes[g]);
        double n2 = static_cast<double>(N - group_sizes[g]);
        double n1n2 = n1 * n2;
        double u = rank_sum[g] - n1 * (n1 + 1.0) / 2.0;
        double z = u - 0.5 * n1n2;
        if (z > 0) {
          z -= 0.5;
        } else if (z < 0) {
          z += 0.5;
        }
        double sigma = std::sqrt(n1n2 * rhs);
        pval_out(feat, g) = 2.0 * R::pnorm5(-std::abs(z / sigma), 0.0, 1.0, 1, 0);
      }
    }
  }
};

// [[Rcpp::export]]
List parallel_all_in_one_dgc(
  SEXP x_sexp,
  IntegerVector groups,
  IntegerVector group_sizes
) {
  S4 X(x_sexp);
  NumericVector x = X.slot("x");
  IntegerVector p = X.slot("p");
  IntegerVector row_i = X.slot("i");
  IntegerVector dims = X.slot("Dim");
  int P = dims[0];
  int N = dims[1];
  int nnz_total = x.size();

  std::vector<int> row_count(P, 0);
  for (int j = 0; j < nnz_total; ++j) {
    row_count[row_i[j]]++;
  }
  int max_nnz = *std::max_element(row_count.begin(), row_count.end());
  std::vector<int> row_ptr(P + 1, 0);
  for (int feat = 0; feat < P; ++feat) {
    row_ptr[feat + 1] = row_ptr[feat] + row_count[feat];
  }

  std::vector<int> row_col(nnz_total);
  std::vector<double> row_val(nnz_total);
  std::vector<int> next_pos(row_ptr);
  for (int col = 0; col < N; ++col) {
    for (int j = p[col]; j < p[col + 1]; ++j) {
      int feat = row_i[j];
      int pos = next_pos[feat]++;
      row_col[pos] = col;
      row_val[pos] = x[j];
    }
  }

  NumericMatrix pval_out(P, group_sizes.size());
  NumericMatrix sum_out(P, group_sizes.size());
  NumericMatrix count_out(P, group_sizes.size());
  AllInOneWorker worker(
    row_ptr, row_col, row_val, groups, group_sizes,
    pval_out, sum_out, count_out, N, max_nnz
  );
  parallelFor(0, P, worker);
  return List::create(
    Named("pval_by_group") = pval_out,
    Named("sum_by_group") = sum_out,
    Named("detected_by_group") = count_out
  );
}
')

fast_FindAllMarkers <- function(
  object,
  assay = NULL,
  features = NULL,
  group.by = NULL,
  logfc.threshold = 0.1,
  test.use = 'wilcox',
  slot = 'data',
  min.pct = 0.01,
  min.diff.pct = -Inf,
  node = NULL,
  verbose = TRUE,
  only.pos = FALSE,
  max.cells.per.ident = Inf,
  random.seed = 1,
  latent.vars = NULL,
  min.cells.feature = 3,
  min.cells.group = 3,
  mean.fxn = NULL,
  fc.name = NULL,
  base = 2,
  return.thresh = 1e-2,
  densify = FALSE,
  ...
) {
  fallback <- function() {
    orig_FindAllMarkers(
      object = object,
      assay = assay,
      features = features,
      group.by = group.by,
      logfc.threshold = logfc.threshold,
      test.use = test.use,
      slot = slot,
      min.pct = min.pct,
      min.diff.pct = min.diff.pct,
      node = node,
      verbose = verbose,
      only.pos = only.pos,
      max.cells.per.ident = max.cells.per.ident,
      random.seed = random.seed,
      latent.vars = latent.vars,
      min.cells.feature = min.cells.feature,
      min.cells.group = min.cells.group,
      mean.fxn = mean.fxn,
      fc.name = fc.name,
      base = base,
      return.thresh = return.thresh,
      densify = densify,
      ...
    )
  }

  dots <- list(...)
  if (
    length(dots) > 0L ||
      !is.null(features) ||
      !is.null(node) ||
      (!is.null(group.by) && !identical(group.by, "ident")) ||
      !identical(test.use, "wilcox") ||
      !identical(slot, "data") ||
      isTRUE(densify) ||
      !is.null(latent.vars) ||
      !is.null(mean.fxn) ||
      !is.null(fc.name) ||
      isTRUE(only.pos) ||
      max.cells.per.ident < Inf ||
      min.diff.pct > -Inf ||
      base != 2
  ) {
    return(fallback())
  }

  assay <- assay %||% DefaultAssay(object = object)
  assay_obj <- object[[assay]]
  if (length(x = Layers(object = assay_obj, search = slot)) > 1) {
    stop(slot, " layers are not joined. Please run JoinLayers")
  }

  data.use <- assay_obj@layers[[slot]]
  if (is.null(x = data.use)) {
    return(fallback())
  }
  dimnames(x = data.use) <- list(rownames(x = assay_obj), colnames(x = assay_obj))
  if (!inherits(x = data.use, what = "dgCMatrix")) {
    return(fallback())
  }

  cellnames.use <- colnames(x = data.use)
  idents <- Idents(object = object)
  if (is.null(names(x = idents)) || !all(cellnames.use %in% names(x = idents))) {
    return(fallback())
  }
  idents <- idents[cellnames.use]
  idents.all <- sort(x = unique(x = idents))
  group.labels <- as.character(idents.all)
  group.factor <- factor(x = as.character(idents), levels = group.labels)
  group.sizes <- tabulate(bin = as.integer(group.factor), nbins = length(group.labels))
  if (
    any(group.sizes < min.cells.group) ||
      any((length(group.factor) - group.sizes) < min.cells.group)
  ) {
    return(fallback())
  }

  feature.names <- rownames(x = data.use)
  n.features.total <- nrow(x = data.use)

  native.results <- parallel_all_in_one_dgc(
    x_sexp = data.use,
    groups = as.integer(x = group.factor),
    group_sizes = group.sizes
  )
  sum.by.group <- native.results$sum_by_group
  dimnames(x = sum.by.group) <- list(feature.names, group.labels)
  total.sum <- Matrix::rowSums(x = sum.by.group)

  detected.by.group <- native.results$detected_by_group
  dimnames(x = detected.by.group) <- list(feature.names, group.labels)
  total.detected <- Matrix::rowSums(x = detected.by.group)

  pval.by.group <- native.results$pval_by_group
  dimnames(x = pval.by.group) <- list(feature.names, group.labels)
  rm(native.results)

  genes.de <- vector(mode = "list", length = length(group.labels))
  names(x = genes.de) <- group.labels
  n.cells <- length(group.factor)

  for (i in seq_along(group.labels)) {
    n.1 <- group.sizes[i]
    n.2 <- n.cells - n.1
    sums.1 <- sum.by.group[, i]
    counts.1 <- detected.by.group[, i]
    counts.2 <- total.detected - counts.1

    fc <- log(x = (sums.1 + 1) / n.1, base = base) -
      log(x = (total.sum - sums.1 + 1) / n.2, base = base)

    features.use <- which(x = counts.1 >= (min.pct * n.1) | counts.2 >= (min.pct * n.2))
    if (length(x = features.use) == 0L) {
      warning("No features pass min.pct threshold; returning empty data.frame")
      next
    }

    keep.diff <- abs(x = fc) >= logfc.threshold
    features.use <- features.use[keep.diff[features.use]]
    if (length(x = features.use) == 0L) {
      warning("No features pass logfc.threshold threshold; returning empty data.frame")
      next
    }

    pct.1 <- round(x = counts.1[features.use] / n.1, digits = 3)
    pct.2 <- round(x = counts.2[features.use] / n.2, digits = 3)
    de.results <- data.frame(
      p_val = pval.by.group[features.use, i],
      avg_log2FC = fc[features.use],
      pct.1 = pct.1,
      pct.2 = pct.2,
      row.names = feature.names[features.use],
      check.names = FALSE
    )
    de.results <- de.results[
      order(de.results$p_val, -abs(de.results$pct.1 - de.results$pct.2)),
      ,
      drop = FALSE
    ]
    de.results$p_val_adj <- p.adjust(
      p = de.results$p_val,
      method = "bonferroni",
      n = n.features.total
    )

    de.results <- de.results[
      order(de.results$p_val, -abs(de.results$pct.1 - de.results$pct.2)),
      ,
      drop = FALSE
    ]
    de.results <- subset(x = de.results, subset = p_val < return.thresh)
    if (nrow(x = de.results) == 0L) {
      next
    }
    de.results$cluster <- factor(
      x = rep(group.labels[i], nrow(x = de.results)),
      levels = group.labels
    )
    de.results$gene <- rownames(x = de.results)
    genes.de[[i]] <- de.results
  }

  genes.de <- genes.de[!vapply(X = genes.de, FUN = is.null, FUN.VALUE = logical(1))]
  gde.all <- if (length(x = genes.de) == 0L) {
    data.frame()
  } else {
    do.call(what = rbind, args = genes.de)
  }
  if (nrow(x = gde.all) == 0) {
    warning("No DE genes identified", call. = FALSE, immediate. = TRUE)
    return(gde.all)
  }
  gde.all
}

install_override("FindAllMarkers", "Seurat", fast_FindAllMarkers)

TIER <- Sys.getenv("ZYME_TIER", unset = "tiny")
DATA_PATH <- Sys.getenv(
  "ZYME_DATA_PATH",
  unset = file.path(TASK_DIR, "data", "pbmc68k_30k.rds")
)
DATA_PATH <- gsub('^"|"$', "", DATA_PATH)
OUTPUT_DIR <- file.path(SCRIPT_DIR, sprintf("output_%s", TIER))
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

if (!file.exists(DATA_PATH)) {
  stop("Input not found at ", DATA_PATH, ". Run setup/prepare_inputs.R first.")
}

cat(sprintf("[pipeline] tier=%s data=%s\n", TIER, DATA_PATH))
cat("[pipeline] Loading Seurat object...\n")
obj <- readRDS(DATA_PATH)
cat(sprintf(
  "[pipeline] Dataset: %d genes x %d cells, %d identities\n",
  nrow(obj), ncol(obj), length(levels(Idents(obj)))
))

options(Seurat.presto.wilcox.msg = FALSE)

invisible(gc(verbose = FALSE))
cat("[pipeline] Running Seurat::FindAllMarkers...\n")
t0 <- Sys.time()
result <- with_profile({
  Seurat::FindAllMarkers(object = obj, verbose = FALSE)
})
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
peak_mb <- peak_memory_mb()

out <- list(
  markers = result,
  n_cells = ncol(obj),
  n_genes = nrow(obj),
  identities = levels(Idents(obj))
)
saveRDS(out, file = file.path(OUTPUT_DIR, "result.rds"))

emit_summary(speed_sec = elapsed, peak_mb = peak_mb)
cat(sprintf("[pipeline] Done in %.1fs\n", elapsed))
