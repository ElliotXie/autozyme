// turbo_sctransform.cpp — SCTransform C++ kernels
// CSC-to-CSR conversion, stats+correct pass, fused residual+center pass
#include <Rcpp.h>
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

// [[Rcpp::export]]
List turbo_csc_to_csr(IntegerVector csc_i, IntegerVector csc_p,
                       NumericVector csc_x, int nrow, int ncol) {
  int nnz = csc_x.size();
  IntegerVector row_ptr(nrow + 1, 0);
  for (int k = 0; k < nnz; k++) row_ptr[csc_i[k] + 1]++;
  for (int i = 1; i <= nrow; i++) row_ptr[i] += row_ptr[i-1];
  IntegerVector col_idx(nnz);
  NumericVector vals(nnz);
  std::vector<int> pos(nrow, 0);
  for (int col = 0; col < ncol; col++) {
    for (int k = csc_p[col]; k < csc_p[col+1]; k++) {
      int row = csc_i[k];
      int dest = row_ptr[row] + pos[row];
      col_idx[dest] = col;
      vals[dest] = csc_x[k];
      pos[row]++;
    }
  }
  return List::create(
    Named("row_ptr") = row_ptr,
    Named("col_idx") = col_idx,
    Named("vals") = vals);
}

// [[Rcpp::export]]
List turbo_stats_correct_sparse(
    NumericVector intercepts, NumericVector cell_mu_base,
    IntegerVector csr_row_ptr, IntegerVector csr_col_idx,
    NumericVector csr_vals, IntegerVector gene_idx,
    NumericVector theta, NumericVector corr_factor,
    double min_var, double clip_lo, double clip_hi, bool do_correct) {
  int ngenes = gene_idx.size(), ncells = cell_mu_base.size();
  NumericVector res_var(ngenes), res_mean(ngenes);

  std::vector<std::vector<int>> out_cols(do_correct ? ngenes : 0);
  std::vector<std::vector<double>> out_vals(do_correct ? ngenes : 0);

  const int* rp = INTEGER(csr_row_ptr);
  const int* ci = INTEGER(csr_col_idx);
  const double* rv = REAL(csr_vals);
  const int* gidx = INTEGER(gene_idx);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 50)
  #endif
  for (int i = 0; i < ngenes; i++) {
    int grow = gidx[i];
    double th = theta[i];
    double exp_int = std::exp(intercepts[i]);
    double sum_r = 0.0, sum_r2 = 0.0;
    int nz_pos = rp[grow];
    int nz_end = rp[grow + 1];
    for (int j = 0; j < ncells; j++) {
      double mu = exp_int * cell_mu_base[j];
      double var_val = mu + mu * mu / th;
      if (var_val < min_var) var_val = min_var;
      double y_val = 0.0;
      if (nz_pos < nz_end && ci[nz_pos] == j) {
        y_val = rv[nz_pos]; nz_pos++;
      }
      double r = (y_val - mu) / std::sqrt(var_val);
      if (r < clip_lo) r = clip_lo;
      if (r > clip_hi) r = clip_hi;
      sum_r += r; sum_r2 += r * r;
      if (do_correct) {
        double mu_c = mu * corr_factor[j];
        double var_c = mu_c + mu_c * mu_c / th;
        double c_val = mu_c + r * std::sqrt(var_c);
        c_val = std::round(c_val);
        if (c_val > 0.0) {
          out_cols[i].push_back(j);
          out_vals[i].push_back(c_val);
        }
      }
    }
    res_mean[i] = sum_r / ncells;
    res_var[i] = (sum_r2 / ncells) - (sum_r / ncells) * (sum_r / ncells);
    res_var[i] = res_var[i] * ncells / (ncells - 1.0);
  }

  if (!do_correct) {
    return List::create(Named("res_var") = res_var, Named("res_mean") = res_mean);
  }

  int total_nnz = 0;
  for (int i = 0; i < ngenes; i++) total_nnz += (int)out_cols[i].size();

  std::vector<int> col_counts(ncells, 0);
  for (int i = 0; i < ngenes; i++) {
    for (size_t k = 0; k < out_cols[i].size(); k++) {
      col_counts[out_cols[i][k]]++;
    }
  }
  IntegerVector p(ncells + 1, 0);
  for (int j = 0; j < ncells; j++) p[j+1] = p[j] + col_counts[j];
  IntegerVector csc_i_out(total_nnz);
  NumericVector csc_x_out(total_nnz);
  std::vector<int> col_pos(ncells, 0);
  for (int i = 0; i < ngenes; i++) {
    for (size_t k = 0; k < out_cols[i].size(); k++) {
      int j = out_cols[i][k];
      int dest = p[j] + col_pos[j];
      csc_i_out[dest] = i;
      csc_x_out[dest] = out_vals[i][k];
      col_pos[j]++;
    }
  }

  return List::create(
    Named("csc_i") = csc_i_out, Named("csc_p") = p,
    Named("csc_x") = csc_x_out,
    Named("res_var") = res_var, Named("res_mean") = res_mean);
}

// [[Rcpp::export]]
NumericMatrix turbo_fused_resid_center_sparse(
    NumericVector intercepts, NumericVector cell_mu_base,
    IntegerVector csr_row_ptr, IntegerVector csr_col_idx,
    NumericVector csr_vals, IntegerVector gene_idx,
    NumericVector theta,
    double min_var, double wide_clip_lo, double wide_clip_hi,
    double narrow_clip_lo, double narrow_clip_hi) {
  int ngenes = gene_idx.size(), ncells = cell_mu_base.size();
  NumericMatrix result(ngenes, ncells);
  double* out = REAL(result);

  const int* rp = INTEGER(csr_row_ptr);
  const int* ci = INTEGER(csr_col_idx);
  const double* rv = REAL(csr_vals);
  const int* gidx = INTEGER(gene_idx);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 50)
  #endif
  for (int i = 0; i < ngenes; i++) {
    int grow = gidx[i];
    double th = theta[i];
    double exp_int = std::exp(intercepts[i]);
    double sum_narrow = 0.0;
    int nz_pos = rp[grow];
    int nz_end = rp[grow + 1];
    double* row_out = out + (size_t)i;  // column-major
    // Use temporary buffer for row
    std::vector<double> buf(ncells);
    for (int j = 0; j < ncells; j++) {
      double mu = exp_int * cell_mu_base[j];
      double var_val = mu + mu * mu / th;
      if (var_val < min_var) var_val = min_var;
      double y_val = 0.0;
      if (nz_pos < nz_end && ci[nz_pos] == j) {
        y_val = rv[nz_pos]; nz_pos++;
      }
      double r = (y_val - mu) / std::sqrt(var_val);
      if (r < wide_clip_lo) r = wide_clip_lo;
      if (r > wide_clip_hi) r = wide_clip_hi;
      if (r < narrow_clip_lo) r = narrow_clip_lo;
      if (r > narrow_clip_hi) r = narrow_clip_hi;
      buf[j] = r;
      sum_narrow += r;
    }
    double mean = sum_narrow / ncells;
    for (int j = 0; j < ncells; j++) {
      result(i, j) = buf[j] - mean;
    }
  }
  return result;
}
