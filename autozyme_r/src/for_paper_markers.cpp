// for_paper_markers.cpp — kernels backing fast_FindAllMarkers_for_paper.
//
// AUTOZYME_MODE=for_paper switches the FindAllMarkers patch from the
// fusion all-in-one kernel (parallel_all_in_one_dgc, in seurat_markers.cpp)
// to the V3 filter-then-rank pipeline whose speed numbers are the ones
// reported in the paper. The three kernels below replace the presto:::*
// private API that the standalone V3 prototype depended on, so the package
// stays presto-free.
//
//   count_sum_by_group_dgc  — per (feature, group) nnz + expm1 sum on a
//                              (features x cells) dgCMatrix
//   portable_rank_dgc       — per-column sparse ranking with tie sums on a
//                              (cells x features-subset) dgCMatrix
//   rank_sum_by_group_dgc   — per (feature, group) Wilcoxon rank sum on a
//                              ranked (cells x features-subset) dgCMatrix

// [[Rcpp::depends(RcppParallel)]]
#include <Rcpp.h>
#include <RcppParallel.h>

#include <algorithm>
#include <numeric>
#include <vector>

using namespace Rcpp;
using namespace RcppParallel;

// ---------------------------------------------------------------------------
// count_sum_by_group_dgc
// Input X: dgCMatrix sized (n_features x n_cells); each cell column carries
// nnz row entries. groups[cell] is 1-based cluster id (length n_cells).
// Output: nnz_by_group, expm1_sum_by_group — each (n_features x ngroups).
// Serial: this step is O(nnz) and trivial vs. the ranking step.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
List count_sum_by_group_dgc(SEXP x_sexp,
                            IntegerVector groups,
                            int ngroups) {
  S4 X(x_sexp);
  NumericVector x = X.slot("x");
  IntegerVector p = X.slot("p");
  IntegerVector row_i = X.slot("i");
  IntegerVector dims = X.slot("Dim");
  int n_features = dims[0];
  int n_cells = dims[1];

  NumericMatrix nnz_out(n_features, ngroups);
  NumericMatrix sum_out(n_features, ngroups);

  for (int col = 0; col < n_cells; ++col) {
    int g = groups[col] - 1;
    if (g < 0 || g >= ngroups) continue;
    int s = p[col];
    int e = p[col + 1];
    for (int j = s; j < e; ++j) {
      int feat = row_i[j];
      nnz_out(feat, g) += 1.0;
      sum_out(feat, g) += std::expm1(x[j]);
    }
  }

  return List::create(
      Named("nnz_by_group") = nnz_out,
      Named("sum_by_group") = sum_out);
}

// ---------------------------------------------------------------------------
// portable_rank_dgc
// Ranks the non-zero entries of each column of a dgCMatrix (so the dense
// zero block shares the average zero-rank). Returns rank-substituted x and
// the per-column tie-correction Σ(t^3 - t) including the zero block.
// Used after subsetting the data matrix down to features that passed the
// pct / lfc gate, so n_cols is the # of pass-through features.
// Ported verbatim from optimized_task/test_seurat_scanpy/find_all_markers/
// v3/pipeline/portable_rank.cpp (kept as a separate kernel so V3's behavior
// is byte-for-byte identical to the prototype it was benchmarked from).
// ---------------------------------------------------------------------------
struct SparseRankWorker : public Worker {
  const RVector<double> x;
  const RVector<int> p;
  RVector<double> ranked_x;
  RVector<double> tie_sum;
  const int n_rows;

  SparseRankWorker(const NumericVector x,
                   const IntegerVector p,
                   NumericVector ranked_x,
                   NumericVector tie_sum,
                   int n_rows)
      : x(x), p(p), ranked_x(ranked_x), tie_sum(tie_sum), n_rows(n_rows) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t col = begin; col < end; ++col) {
      const int start = p[col];
      const int stop = p[col + 1];
      const int nnz = stop - start;
      const int n_zero = n_rows - nnz;

      double ties = 0.0;
      if (n_zero > 1) {
        ties += static_cast<double>(n_zero) * n_zero * n_zero - n_zero;
      }

      if (nnz == 0) {
        tie_sum[col] = ties;
        continue;
      }

      std::vector<int> order(nnz);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](int a, int b) {
        return x[start + a] < x[start + b];
      });

      int run_start = 0;
      while (run_start < nnz) {
        int run_end = run_start + 1;
        const double value = x[start + order[run_start]];
        while (run_end < nnz && x[start + order[run_end]] == value) {
          ++run_end;
        }

        const int run_len = run_end - run_start;
        if (run_len > 1) {
          ties += static_cast<double>(run_len) * run_len * run_len - run_len;
        }

        const double rank = n_zero + ((run_start + 1.0) + run_end) / 2.0;
        for (int k = run_start; k < run_end; ++k) {
          ranked_x[start + order[k]] = rank;
        }
        run_start = run_end;
      }

      tie_sum[col] = ties;
    }
  }
};

// [[Rcpp::export]]
List portable_rank_dgc(NumericVector x, IntegerVector p, int n_rows) {
  const int n_cols = p.size() - 1;
  NumericVector ranked_x(x.size());
  NumericVector tie_sum(n_cols);

  SparseRankWorker worker(x, p, ranked_x, tie_sum, n_rows);
  parallelFor(0, n_cols, worker);

  return List::create(
      Named("x") = ranked_x,
      Named("tie_sum") = tie_sum);
}

// ---------------------------------------------------------------------------
// rank_sum_by_group_dgc
// Input X: dgCMatrix sized (n_cells x n_features_subset) with x = ranked
// values (typically from portable_rank_dgc). groups[cell] is 1-based cluster
// id (length n_cells). group_sizes[g] is the # cells in cluster g.
// Output: (n_features_subset x ngroups) rank-sum matrix where each entry
// counts the sum of ranks of cells in cluster g for that feature, with the
// zero block contributing (group_size[g] - nnz_in_group) * zero_rank where
// zero_rank = (n_zero + 1) / 2.
// Parallel over features (independent columns, no write conflict).
// ---------------------------------------------------------------------------
struct RankSumByGroupWorker : public Worker {
  const std::vector<int>& col_p;
  const std::vector<int>& row_i;
  const std::vector<double>& val_x;
  const RVector<int> groups;
  const RVector<int> group_sizes;
  RMatrix<double> rank_sum_out;
  int n_cells;
  int ngroups;

  RankSumByGroupWorker(const std::vector<int>& col_p,
                       const std::vector<int>& row_i,
                       const std::vector<double>& val_x,
                       IntegerVector groups,
                       IntegerVector group_sizes,
                       NumericMatrix rank_sum_out,
                       int n_cells,
                       int ngroups)
      : col_p(col_p), row_i(row_i), val_x(val_x),
        groups(groups), group_sizes(group_sizes),
        rank_sum_out(rank_sum_out),
        n_cells(n_cells), ngroups(ngroups) {}

  void operator()(std::size_t begin, std::size_t end) {
    std::vector<int> nnz_g(ngroups);
    std::vector<double> rs_g(ngroups);
    for (std::size_t feat = begin; feat < end; ++feat) {
      std::fill(nnz_g.begin(), nnz_g.end(), 0);
      std::fill(rs_g.begin(), rs_g.end(), 0.0);
      int s = col_p[feat];
      int e = col_p[feat + 1];
      int nnz = e - s;
      int n_zero = n_cells - nnz;
      double zero_rank = (n_zero + 1) / 2.0;
      for (int j = s; j < e; ++j) {
        int cell = row_i[j];
        int g = groups[cell] - 1;
        if (g < 0 || g >= ngroups) continue;
        rs_g[g] += val_x[j];
        nnz_g[g]++;
      }
      for (int g = 0; g < ngroups; ++g) {
        rank_sum_out(feat, g) =
            rs_g[g] + (static_cast<double>(group_sizes[g]) - nnz_g[g]) * zero_rank;
      }
    }
  }
};

// [[Rcpp::export]]
NumericMatrix rank_sum_by_group_dgc(SEXP x_sexp,
                                    IntegerVector groups,
                                    IntegerVector group_sizes) {
  S4 X(x_sexp);
  NumericVector x = X.slot("x");
  IntegerVector p = X.slot("p");
  IntegerVector row_i = X.slot("i");
  IntegerVector dims = X.slot("Dim");
  int n_cells = dims[0];
  int n_features = dims[1];
  int ngroups = group_sizes.size();

  std::vector<int> col_p(p.begin(), p.end());
  std::vector<int> row_i_vec(row_i.begin(), row_i.end());
  std::vector<double> val_x(x.begin(), x.end());

  NumericMatrix rank_sum_out(n_features, ngroups);
  RankSumByGroupWorker worker(col_p, row_i_vec, val_x, groups, group_sizes,
                              rank_sum_out, n_cells, ngroups);
  parallelFor(0, n_features, worker);
  return rank_sum_out;
}
