// turbo_scale.cpp — Single-pass sparse stats + parallel dense fill for ScaleData
// Uses OpenMP for parallelism
#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix turbo_scale_sparse_full(S4 sparse_mat,
                                       IntegerVector gene_indices, double scale_max) {
  IntegerVector i_vec = sparse_mat.slot("i");
  IntegerVector p_vec = sparse_mat.slot("p");
  NumericVector x_vec = sparse_mat.slot("x");
  IntegerVector dim_vec = sparse_mat.slot("Dim");
  const int* ip = i_vec.begin();
  const int* pp = p_vec.begin();
  const double* xp = x_vec.begin();
  int n_total_genes = dim_vec[0];
  int n_cells = dim_vec[1];
  int n_sel = gene_indices.size();

  std::vector<int> gmap(n_total_genes, -1);
  for (int g = 0; g < n_sel; g++) gmap[gene_indices[g]] = g;

  std::vector<double> gsum(n_sel, 0.0), gsq(n_sel, 0.0);
  for (int j = 0; j < n_cells; j++) {
    for (int k = pp[j]; k < pp[j+1]; k++) {
      int r = gmap[ip[k]];
      if (r >= 0) {
        double v = xp[k];
        gsum[r] += v;
        gsq[r] += v * v;
      }
    }
  }

  std::vector<double> mn(n_sel), isd(n_sel), zs(n_sel);
  for (int g = 0; g < n_sel; g++) {
    double m = gsum[g] / n_cells;
    mn[g] = m;
    double var = (gsq[g] / n_cells - m * m) * n_cells / (n_cells - 1);
    double sd = std::sqrt(var);
    if (sd == 0.0) sd = 1.0;
    isd[g] = 1.0 / sd;
    zs[g] = -m / sd;
    // Match Seurat's FastSparseRowScale: ScaleData caps only the positive
    // tail. Scanpy uses two-sided clipping, but Seurat leaves negative
    // background z-scores below -scale_max unchanged.
    if (zs[g] > scale_max) zs[g] = scale_max;
  }

  NumericMatrix result = Rcpp::no_init_matrix(n_sel, n_cells);
  double* out = REAL(result);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 100)
  #endif
  for (int j = 0; j < n_cells; j++) {
    double* col = out + (size_t)j * n_sel;
    std::memcpy(col, zs.data(), n_sel * sizeof(double));
    for (int k = pp[j]; k < pp[j+1]; k++) {
      int r = gmap[ip[k]];
      if (r >= 0) {
        double v = (xp[k] - mn[r]) * isd[r];
        if (v > scale_max) v = scale_max;
        col[r] = v;
      }
    }
  }

  return result;
}
