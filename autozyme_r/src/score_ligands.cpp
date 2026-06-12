// [[Rcpp::depends(RcppParallel)]]
#include <Rcpp.h>
#include <RcppParallel.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

using namespace Rcpp;
using namespace RcppParallel;

struct LigandScoreWorker : public Worker {
  RMatrix<double> mat;
  RVector<int> pos_row_idx;
  RVector<int> neg_row_idx;
  RVector<int> col_idx;
  RMatrix<double> out;
  int n_pos;
  int n_neg;
  int n_rows;

  LigandScoreWorker(NumericMatrix mat, IntegerVector pos_row_idx,
                    IntegerVector neg_row_idx, IntegerVector col_idx,
                    NumericMatrix out, int n_pos, int n_neg)
      : mat(mat), pos_row_idx(pos_row_idx), neg_row_idx(neg_row_idx),
        col_idx(col_idx), out(out), n_pos(n_pos), n_neg(n_neg),
        n_rows(n_pos + n_neg) {}

  void operator()(std::size_t begin, std::size_t end) {
    std::vector<double> pos_scores;
    std::vector<double> neg_scores;
    pos_scores.reserve(n_pos);
    neg_scores.reserve(n_neg);

    for (std::size_t col = begin; col < end; ++col) {
      pos_scores.clear();
      neg_scores.clear();
      double sum_x = 0.0;
      double sum_x2 = 0.0;
      double sum_xy = 0.0;

      const double* col_data = mat.begin() +
        static_cast<std::size_t>(col_idx[col] - 1) * mat.nrow();
      for (int row = 0; row < n_pos; ++row) {
        const double x = col_data[pos_row_idx[row] - 1];
        pos_scores.push_back(x);
        sum_x += x;
        sum_x2 += x * x;
        sum_xy += x;
      }
      for (int row = 0; row < n_neg; ++row) {
        const double x = col_data[neg_row_idx[row] - 1];
        neg_scores.push_back(x);
        sum_x += x;
        sum_x2 += x * x;
      }

      std::sort(pos_scores.begin(), pos_scores.end(), std::greater<double>());
      std::sort(neg_scores.begin(), neg_scores.end(), std::greater<double>());

      double aupr = 0.0;
      double auroc_numerator = 0.0;
      int tp_after = 0;
      std::size_t neg_gt = 0;
      std::size_t neg_ge = 0;

      std::size_t i = 0;
      while (i < pos_scores.size()) {
        const double threshold = pos_scores[i];
        int group_size = 0;
        while (i < pos_scores.size() && pos_scores[i] == threshold) {
          ++group_size;
          ++i;
        }

        while (neg_gt < neg_scores.size() && neg_scores[neg_gt] > threshold) {
          ++neg_gt;
        }
        while (neg_ge < neg_scores.size() && neg_scores[neg_ge] >= threshold) {
          ++neg_ge;
        }

        const int fp_before = static_cast<int>(neg_gt);
        const int fp_after = static_cast<int>(neg_ge);
        const int tp_before = tp_after;
        tp_after += group_size;

        const double recall_before = static_cast<double>(tp_before) / n_pos;
        const double recall_after = static_cast<double>(tp_after) / n_pos;
        const double precision_before = (tp_before + fp_before) == 0
          ? 1.0
          : static_cast<double>(tp_before) / (tp_before + fp_before);
        const double precision_after =
          static_cast<double>(tp_after) / (tp_after + fp_after);

        aupr += (recall_after - recall_before) *
          (precision_before + precision_after) / 2.0;
        auroc_numerator += group_size *
          ((n_neg - fp_after) + 0.5 * (fp_after - fp_before));
      }

      const double n = static_cast<double>(n_rows);
      const double numerator = n * sum_xy - sum_x * n_pos;
      const double denom_x = n * sum_x2 - sum_x * sum_x;
      const double denom_y = n * n_pos - static_cast<double>(n_pos) * n_pos;
      const double pearson = numerator / std::sqrt(denom_x * denom_y);

      out(col, 0) = auroc_numerator / (static_cast<double>(n_pos) * n_neg);
      out(col, 1) = aupr;
      out(col, 2) = aupr - (static_cast<double>(n_pos) / n_rows);
      out(col, 3) = pearson;
    }
  }
};

struct LigandScoreBinaryWorker : public Worker {
  RMatrix<double> mat;
  RVector<int> pos_row_idx;
  RVector<int> neg_row_idx;
  RVector<int> col_idx;
  RMatrix<double> out;
  int n_pos;
  int n_neg;
  int n_rows;

  LigandScoreBinaryWorker(NumericMatrix mat, IntegerVector pos_row_idx,
                          IntegerVector neg_row_idx, IntegerVector col_idx,
                          NumericMatrix out, int n_pos, int n_neg)
      : mat(mat), pos_row_idx(pos_row_idx), neg_row_idx(neg_row_idx),
        col_idx(col_idx), out(out), n_pos(n_pos), n_neg(n_neg),
        n_rows(n_pos + n_neg) {}

  void operator()(std::size_t begin, std::size_t end) {
    std::vector<double> pos_scores;
    std::array<double, 2048> group_thresholds;
    std::array<int, 2048> group_sizes;
    std::array<int, 2049> neg_gt_starts;
    std::array<int, 2049> neg_ge_starts;
    pos_scores.reserve(n_pos);

    for (std::size_t col = begin; col < end; ++col) {
      pos_scores.clear();
      double sum_x = 0.0;
      double sum_x2 = 0.0;
      double sum_xy = 0.0;

      const double* col_data = mat.begin() +
        static_cast<std::size_t>(col_idx[col] - 1) * mat.nrow();
      for (int row = 0; row < n_pos; ++row) {
        const double x = col_data[pos_row_idx[row] - 1];
        pos_scores.push_back(x);
        sum_x += x;
        sum_x2 += x * x;
        sum_xy += x;
      }

      std::sort(pos_scores.begin(), pos_scores.end(), std::greater<double>());

      int n_groups = 0;
      std::size_t i = 0;
      while (i < pos_scores.size()) {
        const double threshold = pos_scores[i];
        int group_size = 0;
        while (i < pos_scores.size() && pos_scores[i] == threshold) {
          ++group_size;
          ++i;
        }
        group_thresholds[n_groups] = threshold;
        group_sizes[n_groups] = group_size;
        ++n_groups;
      }

      std::fill(neg_gt_starts.begin(), neg_gt_starts.begin() + n_groups + 1, 0);
      std::fill(neg_ge_starts.begin(), neg_ge_starts.begin() + n_groups + 1, 0);

      const bool use_range_shortcut = n_pos > 512;
      const double max_threshold = group_thresholds[0];
      const double min_threshold = group_thresholds[n_groups - 1];

      int zero_ge_start = n_groups;
      int zero_gt_start = n_groups;
      {
        int lo = 0;
        int hi = n_groups;
        while (lo < hi) {
          const int mid = lo + (hi - lo) / 2;
          if (group_thresholds[mid] <= 0.0) {
            hi = mid;
          } else {
            lo = mid + 1;
          }
        }
        zero_ge_start = lo;
        if (lo < n_groups) {
          zero_gt_start = (group_thresholds[lo] == 0.0) ? lo + 1 : lo;
        }
      }

      for (int row = 0; row < n_neg; ++row) {
        const double x = col_data[neg_row_idx[row] - 1];
        sum_x += x;
        sum_x2 += x * x;

        if (use_range_shortcut) {
          if (x > max_threshold) {
            ++neg_ge_starts[0];
            ++neg_gt_starts[0];
            continue;
          }
          if (x < min_threshold) {
            continue;
          }
        }

        if (x == 0.0) {
          if (zero_ge_start < n_groups) {
            ++neg_ge_starts[zero_ge_start];
          }
          if (zero_gt_start < n_groups) {
            ++neg_gt_starts[zero_gt_start];
          }
          continue;
        }

        int lo = 0;
        int hi = n_groups;
        while (lo < hi) {
          const int mid = lo + (hi - lo) / 2;
          if (group_thresholds[mid] <= x) {
            hi = mid;
          } else {
            lo = mid + 1;
          }
        }
        if (lo < n_groups) {
          ++neg_ge_starts[lo];
          const int first_less = (group_thresholds[lo] == x) ? lo + 1 : lo;
          if (first_less < n_groups) {
            ++neg_gt_starts[first_less];
          }
        }
      }

      double aupr = 0.0;
      double auroc_numerator = 0.0;
      int tp_after = 0;
      int neg_gt = 0;
      int neg_ge = 0;

      for (int group = 0; group < n_groups; ++group) {
        neg_gt += neg_gt_starts[group];
        neg_ge += neg_ge_starts[group];

        const int group_size = group_sizes[group];
        const int fp_before = neg_gt;
        const int fp_after = neg_ge;
        const int tp_before = tp_after;
        tp_after += group_size;

        const double recall_before = static_cast<double>(tp_before) / n_pos;
        const double recall_after = static_cast<double>(tp_after) / n_pos;
        const double precision_before = (tp_before + fp_before) == 0
          ? 1.0
          : static_cast<double>(tp_before) / (tp_before + fp_before);
        const double precision_after =
          static_cast<double>(tp_after) / (tp_after + fp_after);

        aupr += (recall_after - recall_before) *
          (precision_before + precision_after) / 2.0;
        auroc_numerator += group_size *
          ((n_neg - fp_after) + 0.5 * (fp_after - fp_before));
      }

      const double n = static_cast<double>(n_rows);
      const double numerator = n * sum_xy - sum_x * n_pos;
      const double denom_x = n * sum_x2 - sum_x * sum_x;
      const double denom_y = n * n_pos - static_cast<double>(n_pos) * n_pos;
      const double pearson = numerator / std::sqrt(denom_x * denom_y);

      out(col, 0) = auroc_numerator / (static_cast<double>(n_pos) * n_neg);
      out(col, 1) = aupr;
      out(col, 2) = aupr - (static_cast<double>(n_pos) / n_rows);
      out(col, 3) = pearson;
    }
  }
};

// [[Rcpp::export]]
NumericMatrix score_ligands_cpp(NumericMatrix mat, IntegerVector pos_row_idx,
                                IntegerVector neg_row_idx,
                                IntegerVector col_idx, int threads) {
  const int n_cols = col_idx.size();
  const int n_pos = pos_row_idx.size();
  const int n_neg = neg_row_idx.size();
  NumericMatrix out(n_cols, 4);
  if (n_pos <= 2048) {
    LigandScoreBinaryWorker worker(mat, pos_row_idx, neg_row_idx, col_idx, out,
                                   n_pos, n_neg);
    parallelFor(0, n_cols, worker, 1, threads);
  } else {
    LigandScoreWorker worker(mat, pos_row_idx, neg_row_idx, col_idx, out,
                             n_pos, n_neg);
    parallelFor(0, n_cols, worker, 1, threads);
  }
  return out;
}
