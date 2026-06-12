// seurat_normalize.cpp — Parallel in-place LogNormalize for dgCMatrix.
// Ported from seurat-zyme (turbo_normalize.cpp). Used by the Seurat patch in
// inst/patches/seurat.R to replace Seurat::NormalizeData.Seurat.
#include <Rcpp.h>
#include <cmath>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

inline double seurat_fast_log(double x) {
    static const double LN2 = 0.6931471805599453;
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    int exp_raw = (int)((bits >> 52) & 0x7FF) - 1023;
    bits = (bits & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    double m;
    std::memcpy(&m, &bits, sizeof(m));
    if (m > 1.4142135623730951) { m *= 0.5; exp_raw++; }
    double f = (m - 1.0) / (m + 1.0);
    double f2 = f * f;
    double poly = 1.0 + f2 * (1.0/3.0 + f2 * (1.0/5.0 + f2 * (1.0/7.0 + f2 * (1.0/9.0 + f2 * (1.0/11.0)))));
    return (double)exp_raw * LN2 + 2.0 * f * poly;
}

inline double seurat_fast_log1p(double x) {
    if (x < 1e-4) return x * (1.0 - x * 0.5);
    return seurat_fast_log(1.0 + x);
}

// [[Rcpp::export]]
void seurat_log_normalize_dgc(S4 mat, double scale_factor, int grain_size = 100) {
  NumericVector x = mat.slot("x");
  IntegerVector p = mat.slot("p");
  int ncol = p.size() - 1;
  double* xptr = REAL(x);
  int* pptr = INTEGER(p);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, grain_size)
  #endif
  for (int j = 0; j < ncol; j++) {
    int s = pptr[j];
    int e = pptr[j + 1];
    double col_sum = 0.0;
    for (int idx = s; idx < e; idx++) {
      col_sum += xptr[idx];
    }
    if (col_sum > 0.0) {
      double factor = scale_factor / col_sum;
      for (int idx = s; idx < e; idx++) {
        xptr[idx] = seurat_fast_log1p(xptr[idx] * factor);
      }
    }
  }
}
