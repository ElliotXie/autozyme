// turbo_markers.cpp — Wilcoxon kernels for FindAllMarkers fast path
//
// Two kernels live here:
//
//   turbo_all_in_one_wilcox     — SERIAL, returns (rank_sums + rhs)
//                                  for R-side z/pnorm computation.
//                                  Retained for back-compat; not used by the
//                                  current fast_FindAllMarkers path.
//
//   parallel_all_in_one_dgc     — RcppParallel Worker version that returns
//                                  pre-computed p-values directly (R-side
//                                  z/pnorm work eliminated). Ported verbatim
//                                  from optimized_task/test_find_all_markers/
//                                  pipeline/run.R (claude_r1_fusion, fig4
//                                  round 37 + 38 fused).

// [[Rcpp::depends(RcppParallel)]]
#include <Rcpp.h>
#include <RcppParallel.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <utility>
using namespace Rcpp;
using namespace RcppParallel;

// ---------------------------------------------------------------------------
// SERIAL kernel — kept for back-compat
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
List turbo_all_in_one_wilcox(NumericVector x, IntegerVector p, IntegerVector i,
                              int ncol_orig, int nrow_orig,
                              IntegerVector groups, int ngroups) {
  int n_features = nrow_orig;
  int n_cells = ncol_orig;
  int nnz_total = x.size();

  std::vector<int> row_count(n_features, 0);
  for (int j = 0; j < nnz_total; j++) row_count[i[j]]++;
  std::vector<int> row_ptr(n_features + 1, 0);
  for (int f = 0; f < n_features; f++) row_ptr[f + 1] = row_ptr[f] + row_count[f];
  std::vector<int> row_col(nnz_total);
  std::vector<double> row_val(nnz_total);
  std::vector<int> row_pos(n_features, 0);
  for (int col = 0; col < n_cells; col++) {
    for (int j = p[col]; j < p[col + 1]; j++) {
      int feat = i[j];
      int pos = row_ptr[feat] + row_pos[feat];
      row_col[pos] = col;
      row_val[pos] = x[j];
      row_pos[feat]++;
    }
  }

  NumericMatrix nnz_group_out(ngroups, n_features);
  NumericMatrix expm1_sums_out(ngroups, n_features);
  NumericMatrix rank_sums_out(ngroups, n_features);
  NumericVector rhs_out(n_features);

  double N = (double)n_cells;
  double x1 = N * N * N - N;
  double x2 = 1.0 / (12.0 * (N * N - N));

  for (int feat = 0; feat < n_features; feat++) {
    int start = row_ptr[feat];
    int end = row_ptr[feat + 1];
    int nnz = end - start;
    int nzeros = n_cells - nnz;

    for (int j = start; j < end; j++) {
      int grp = groups[row_col[j]];
      nnz_group_out(grp, feat) += 1.0;
      expm1_sums_out(grp, feat) += std::expm1(row_val[j]);
    }

    if (nnz == 0) { rhs_out[feat] = 0.0; continue; }

    std::vector<std::pair<double, int>> vals(nnz);
    for (int j = 0; j < nnz; j++)
      vals[j] = std::make_pair(row_val[start + j], row_col[start + j]);
    std::sort(vals.begin(), vals.end());

    double tie_sum = 0.0;
    if (nzeros > 1) { double tz = (double)nzeros; tie_sum += tz * tz * tz - tz; }

    int idx = 0;
    while (idx < nnz) {
      int tie_start = idx;
      double tie_val = vals[idx].first;
      while (idx < nnz && vals[idx].first == tie_val) idx++;
      int tie_count = idx - tie_start;
      double mean_rank = (double)nzeros + ((double)tie_start + (double)idx + 1.0) / 2.0;
      if (tie_count > 1) { double tc = (double)tie_count; tie_sum += tc * tc * tc - tc; }
      for (int j = tie_start; j < idx; j++) {
        int grp = groups[vals[j].second];
        rank_sums_out(grp, feat) += mean_rank;
      }
    }
    rhs_out[feat] = (x1 - tie_sum) * x2;
  }

  return List::create(Named("nnz_group") = nnz_group_out,
                      Named("expm1_sums") = expm1_sums_out,
                      Named("rank_sums") = rank_sums_out,
                      Named("rhs") = rhs_out);
}

// ---------------------------------------------------------------------------
// PARALLEL kernel — fast_FindAllMarkers uses this one (fig4 round 37 + 38)
// ---------------------------------------------------------------------------
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
    // Pair-based per-feature scratch (value, col). Cache-friendly sort: pairs
    // are 16 bytes contiguous, std::sort compares directly without indirect
    // row_val[start+idx] loads. Critical for high-NNZ features in large
    // datasets (e.g. heart_adult 486k cells: housekeeping genes touch
    // nearly every cell, blowing L3 if sort uses index-with-indirection).
    std::vector<std::pair<double, int>> vals;
    vals.reserve(max_nnz);
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
      vals.resize(m);

      for (int k = 0; k < m; ++k) {
        int col = row_col[start + k];
        double v = row_val[start + k];
        vals[k] = std::make_pair(v, col);
        int g = groups[col] - 1;
        if (g >= 0 && g < G) {
          nz_count[g]++;
          expm1_sum[g] += std::expm1(v);
        }
      }

      // default pair operator< compares first (value) then second (col) —
      // same ordering as the previous index+lambda sort.
      std::sort(vals.begin(), vals.end());

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
        double val = vals[pos].first;
        while (next < m && vals[next].first == val) next++;
        int len = next - pos;
        double avg_rank = rank_start + (len - 1) / 2.0;
        for (int r = pos; r < next; ++r) {
          int g = groups[vals[r].second] - 1;
          if (g >= 0 && g < G) rank_sum[g] += avg_rank;
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
        if (z > 0) z -= 0.5;
        else if (z < 0) z += 0.5;
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
  for (int j = 0; j < nnz_total; ++j) row_count[row_i[j]]++;
  int max_nnz = *std::max_element(row_count.begin(), row_count.end());
  std::vector<int> row_ptr(P + 1, 0);
  for (int feat = 0; feat < P; ++feat) row_ptr[feat + 1] = row_ptr[feat] + row_count[feat];

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

