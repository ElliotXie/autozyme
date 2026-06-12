// Shared sparse and grouped-reduction kernels for autozyme patches.
//
// These kernels are intentionally small, deterministic building blocks. Patch
// code should call the R wrappers in shared_infra.R so feature gates and base-R
// fallbacks stay centralized.

// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <vector>
#include <cmath>

using namespace Rcpp;

namespace {

struct DgCView {
  IntegerVector p;
  IntegerVector i;
  NumericVector x;
  IntegerVector dim;
  int nrow;
  int ncol;
};

DgCView as_dgc(SEXP x_sexp) {
  S4 X(x_sexp);
  DgCView out;
  out.p = X.slot("p");
  out.i = X.slot("i");
  out.x = X.slot("x");
  out.dim = X.slot("Dim");
  if (out.dim.size() != 2) stop("expected a two-dimensional dgCMatrix");
  out.nrow = out.dim[0];
  out.ncol = out.dim[1];
  return out;
}

void validate_groups(IntegerVector groups, int ncol, int ngroups) {
  if (groups.size() != ncol) {
    stop("group length must match ncol(x)");
  }
  if (ngroups < 0) stop("ngroups must be non-negative");
}

} // namespace

// [[Rcpp::export]]
List az_dgc_row_stats_cpp(SEXP x_sexp, double detected_threshold = 0.0) {
  DgCView X = as_dgc(x_sexp);
  std::vector<double> sum(X.nrow, 0.0);
  std::vector<double> sumsq(X.nrow, 0.0);
  std::vector<int> nnz(X.nrow, 0);
  std::vector<int> detected(X.nrow, 0);

  const int* pp = INTEGER(X.p);
  const int* ip = INTEGER(X.i);
  const double* xp = REAL(X.x);

  for (int col = 0; col < X.ncol; ++col) {
    for (int k = pp[col]; k < pp[col + 1]; ++k) {
      int row = ip[k];
      double v = xp[k];
      sum[row] += v;
      sumsq[row] += v * v;
      nnz[row]++;
      if (v > detected_threshold) detected[row]++;
    }
  }

  NumericVector row_sum(X.nrow);
  NumericVector row_mean(X.nrow);
  NumericVector row_var(X.nrow);
  IntegerVector row_nnz(X.nrow);
  IntegerVector row_detected(X.nrow);
  double n = static_cast<double>(X.ncol);
  double denom = n - 1.0;
  for (int row = 0; row < X.nrow; ++row) {
    int implicit_zeros = X.ncol - nnz[row];
    if (detected_threshold < 0.0) detected[row] += implicit_zeros;
    double mu = (X.ncol > 0) ? sum[row] / n : R_NaReal;
    row_sum[row] = sum[row];
    row_mean[row] = mu;
    if (X.ncol <= 1) {
      row_var[row] = R_NaReal;
    } else {
      double var = (sumsq[row] - n * mu * mu) / denom;
      row_var[row] = (var < 0.0 && var > -1e-12) ? 0.0 : var;
    }
    row_nnz[row] = nnz[row];
    row_detected[row] = detected[row];
  }

  return List::create(
    _["sum"] = row_sum,
    _["mean"] = row_mean,
    _["variance"] = row_var,
    _["nnz"] = row_nnz,
    _["detected"] = row_detected
  );
}

// [[Rcpp::export]]
List az_dgc_group_summary_cpp(SEXP x_sexp, IntegerVector groups, int ngroups,
                              double detected_threshold = 0.0) {
  DgCView X = as_dgc(x_sexp);
  validate_groups(groups, X.ncol, ngroups);

  NumericMatrix sum_out(X.nrow, ngroups);
  NumericMatrix mean_out(X.nrow, ngroups);
  NumericMatrix detected_out(X.nrow, ngroups);
  IntegerMatrix explicit_nnz(X.nrow, ngroups);
  IntegerVector group_size(ngroups);

  const int* gp = INTEGER(groups);
  for (int col = 0; col < X.ncol; ++col) {
    int g = gp[col];
    if (g != NA_INTEGER && g >= 1 && g <= ngroups) group_size[g - 1]++;
  }

  const int* pp = INTEGER(X.p);
  const int* ip = INTEGER(X.i);
  const double* xp = REAL(X.x);
  for (int col = 0; col < X.ncol; ++col) {
    int g = gp[col];
    if (g == NA_INTEGER || g < 1 || g > ngroups) continue;
    int gg = g - 1;
    for (int k = pp[col]; k < pp[col + 1]; ++k) {
      int row = ip[k];
      double v = xp[k];
      sum_out(row, gg) += v;
      explicit_nnz(row, gg)++;
      if (v > detected_threshold) detected_out(row, gg) += 1.0;
    }
  }

  for (int g = 0; g < ngroups; ++g) {
    for (int row = 0; row < X.nrow; ++row) {
      if (detected_threshold < 0.0) {
        detected_out(row, g) += group_size[g] - explicit_nnz(row, g);
      }
      mean_out(row, g) = group_size[g] > 0
        ? sum_out(row, g) / static_cast<double>(group_size[g])
        : R_NaReal;
    }
  }

  return List::create(
    _["sum_by_group"] = sum_out,
    _["mean_by_group"] = mean_out,
    _["detected_by_group"] = detected_out,
    _["group_size"] = group_size
  );
}

// [[Rcpp::export]]
List az_dense_group_summary_cpp(NumericMatrix x, IntegerVector groups,
                                int ngroups,
                                double detected_threshold = 0.0,
                                bool na_rm = false) {
  int nr = x.nrow();
  int nc = x.ncol();
  validate_groups(groups, nc, ngroups);

  NumericMatrix sum_out(nr, ngroups);
  NumericMatrix mean_out(nr, ngroups);
  NumericMatrix detected_out(nr, ngroups);
  IntegerVector group_size(ngroups);
  IntegerMatrix valid_count(nr, ngroups);
  IntegerMatrix na_count(nr, ngroups);

  const int* gp = INTEGER(groups);
  for (int col = 0; col < nc; ++col) {
    int g = gp[col];
    if (g != NA_INTEGER && g >= 1 && g <= ngroups) group_size[g - 1]++;
  }

  for (int col = 0; col < nc; ++col) {
    int g = gp[col];
    if (g == NA_INTEGER || g < 1 || g > ngroups) continue;
    int gg = g - 1;
    for (int row = 0; row < nr; ++row) {
      double v = x(row, col);
      if (NumericVector::is_na(v)) {
        na_count(row, gg)++;
        if (!na_rm) {
          sum_out(row, gg) = R_NaReal;
          detected_out(row, gg) = R_NaReal;
        }
        continue;
      }
      valid_count(row, gg)++;
      if (!NumericVector::is_na(sum_out(row, gg))) sum_out(row, gg) += v;
      if (!NumericVector::is_na(detected_out(row, gg)) && v > detected_threshold) {
        detected_out(row, gg) += 1.0;
      }
    }
  }

  for (int g = 0; g < ngroups; ++g) {
    for (int row = 0; row < nr; ++row) {
      int denom = na_rm ? valid_count(row, g) : group_size[g];
      if (!na_rm && na_count(row, g) > 0) {
        mean_out(row, g) = R_NaReal;
      } else {
        mean_out(row, g) = denom > 0
          ? sum_out(row, g) / static_cast<double>(denom)
          : R_NaReal;
      }
    }
  }

  return List::create(
    _["sum_by_group"] = sum_out,
    _["mean_by_group"] = mean_out,
    _["detected_by_group"] = detected_out,
    _["group_size"] = group_size
  );
}
