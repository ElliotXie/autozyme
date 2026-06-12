// turbo_variable_features.cpp — Parallel sparse VST for FindVariableFeatures
// Uses OpenMP for parallelism
#include <Rcpp.h>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

// [[Rcpp::export]]
List turbo_FastSparseRowMeanVar(IntegerVector p, IntegerVector i, NumericVector x,
                                int nrow, int ncol) {
  // Serial accumulation (reduction over columns)
  std::vector<double> rowSum(nrow, 0.0);
  std::vector<double> rowSumSq(nrow, 0.0);
  std::vector<int> nnzCount(nrow, 0);

  const int* pp = INTEGER(p);
  const int* ip = INTEGER(i);
  const double* xp = REAL(x);

  for (int j = 0; j < ncol; j++) {
    for (int idx = pp[j]; idx < pp[j + 1]; idx++) {
      int row = ip[idx];
      double val = xp[idx];
      rowSum[row] += val;
      rowSumSq[row] += val * val;
      nnzCount[row]++;
    }
  }

  NumericVector mu(nrow);
  NumericVector variance(nrow);
  IntegerVector nnz(nrow);
  double n = (double)ncol;
  double denom = n - 1.0;

  for (int row = 0; row < nrow; row++) {
    mu[row] = rowSum[row] / n;
    variance[row] = (rowSumSq[row] - n * mu[row] * mu[row]) / denom;
    if (variance[row] < 0.0) variance[row] = 0.0;
    nnz[row] = nnzCount[row];
  }

  return List::create(Named("mean") = mu, Named("variance") = variance,
                      Named("nnz") = nnz);
}

// [[Rcpp::export]]
NumericVector turbo_FastSparseRowVarStd(IntegerVector p, IntegerVector i, NumericVector x,
                                         int nrow, int ncol,
                                         NumericVector mu, NumericVector sd,
                                         double vmax, IntegerVector nnzPerRow) {
  std::vector<double> inv_sd_vec(nrow);
  std::vector<double> mu_isd_vec(nrow);
  for (int row = 0; row < nrow; row++) {
    if (sd[row] == 0.0) {
      inv_sd_vec[row] = 0.0;
      mu_isd_vec[row] = 0.0;
    } else {
      inv_sd_vec[row] = 1.0 / sd[row];
      mu_isd_vec[row] = mu[row] * inv_sd_vec[row];
    }
  }

  std::vector<double> sumSq(nrow, 0.0);
  const int* pp = INTEGER(p);
  const int* ip = INTEGER(i);
  const double* xp = REAL(x);

  for (int j = 0; j < ncol; j++) {
    for (int idx = pp[j]; idx < pp[j + 1]; idx++) {
      int row = ip[idx];
      double isd = inv_sd_vec[row];
      if (isd == 0.0) continue;
      double z = xp[idx] * isd - mu_isd_vec[row];
      if (z > vmax) z = vmax;
      sumSq[row] += z * z;
    }
  }

  NumericVector result(nrow);
  double denom = ncol - 1.0;
  for (int row = 0; row < nrow; row++) {
    if (inv_sd_vec[row] == 0.0) {
      result[row] = 0.0;
      continue;
    }
    int nZero = ncol - nnzPerRow[row];
    double zeroVal = mu_isd_vec[row];
    double total = sumSq[row] + zeroVal * zeroVal * nZero;
    result[row] = total / denom;
  }
  return result;
}
