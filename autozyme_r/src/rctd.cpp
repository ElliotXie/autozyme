// C++ kernels for autozyme's rctd patch.
//
// Lifted from test_RCTD/pipeline/run.R (one inline Rcpp::sourceCpp block
// around lines 220-982). The inline cppFunction approach was hot-rebuilt
// per-process during iteration; here it's compiled once into autozyme.so
// and called from R-side fast_* wrappers in inst/patches/rctd.R.
//
// All exported functions consume Q_mat / SQ_mat / X_vals / K_val as
// explicit arguments — the R-side wrapper grabs them from the .GlobalEnv
// bindings that spacexr::set_likelihood_vars writes into.

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <limits>
using namespace Rcpp;

inline double clamp_lambda_cpp(double lambda, double x_max) {
  const double epsilon = 1e-4;
  if (lambda < epsilon) return epsilon;
  const double upper = x_max - epsilon;
  if (lambda > upper) return upper;
  return lambda;
}

inline int row_index_cpp(double y, int k_val) {
  return static_cast<int>(y > k_val ? k_val : y);
}

inline int m_index_cpp(double lambda) {
  const double delta = 1e-6;
  const double l = std::floor(std::sqrt(lambda / delta));
  const double left = std::min(l - 9.0, 40.0);
  const double right_base = std::max(l - 48.7499, 0.0) * 4.0;
  const double right = std::max(std::ceil(std::sqrt(right_base)) - 2.0, 0.0);
  return static_cast<int>(left + right) - 1;
}

inline void spline_terms_cpp(double y, double lambda_raw, int i,
                             bool has_row_idx, const NumericVector& row_idx,
                             const NumericMatrix& q_mat, const NumericMatrix& sq_mat,
                             const NumericVector& x_vals, int k_val,
                             double x_max, double& d0, double& d1, double& d2) {
  const double lambda = clamp_lambda_cpp(lambda_raw, x_max);
  const int row = has_row_idx ? static_cast<int>(row_idx[i]) - 1 : row_index_cpp(y, k_val);
  const int col = m_index_cpp(lambda);

  const double ti1 = x_vals[col];
  const double ti = x_vals[col + 1];
  const double hi = ti - ti1;
  const double fti1 = q_mat(row, col);
  const double fti = q_mat(row, col + 1);
  const double zi1 = sq_mat(row, col);
  const double zi = sq_mat(row, col + 1);

  const double diff1 = lambda - ti1;
  const double diff2 = ti - lambda;
  const double diff1_sq = diff1 * diff1;
  const double diff2_sq = diff2 * diff2;
  const double diff3 = fti / hi - zi * hi / 6.0;
  const double diff4 = fti1 / hi - zi1 * hi / 6.0;
  const double zdi = zi / hi;
  const double zdi1 = zi1 / hi;

  d0 = zdi * diff1_sq * diff1 / 6.0 + zdi1 * diff2_sq * diff2 / 6.0 +
       diff3 * diff1 + diff4 * diff2;
  d1 = zdi * diff1_sq / 2.0 - zdi1 * diff2_sq / 2.0 +
       diff3 - diff4;
  d2 = zdi * diff1 + zdi1 * diff2;
}

inline double spline_value_cpp(double y, double lambda_raw, int i,
                               bool has_row_idx, const NumericVector& row_idx,
                               const NumericMatrix& q_mat, const NumericMatrix& sq_mat,
                               const NumericVector& x_vals, int k_val, double x_max) {
  const double lambda = clamp_lambda_cpp(lambda_raw, x_max);
  const int row = has_row_idx ? static_cast<int>(row_idx[i]) - 1 : row_index_cpp(y, k_val);
  const int col = m_index_cpp(lambda);

  const double ti1 = x_vals[col];
  const double ti = x_vals[col + 1];
  const double hi = ti - ti1;
  const double fti1 = q_mat(row, col);
  const double fti = q_mat(row, col + 1);
  const double zi1 = sq_mat(row, col);
  const double zi = sq_mat(row, col + 1);

  const double diff1 = lambda - ti1;
  const double diff2 = ti - lambda;
  const double diff1_sq = diff1 * diff1;
  const double diff2_sq = diff2 * diff2;
  const double diff3 = fti / hi - zi * hi / 6.0;
  const double diff4 = fti1 / hi - zi1 * hi / 6.0;
  const double zdi = zi / hi;
  const double zdi1 = zi1 / hi;
  return zdi * diff1_sq * diff1 / 6.0 + zdi1 * diff2_sq * diff2 / 6.0 +
         diff3 * diff1 + diff4 * diff2;
}

inline void spline_derivs_cpp(double y, double lambda_raw, int i,
                              bool has_row_idx, const NumericVector& row_idx,
                              const NumericMatrix& q_mat, const NumericMatrix& sq_mat,
                              const NumericVector& x_vals, int k_val, double x_max,
                              double& d1, double& d2) {
  const double lambda = clamp_lambda_cpp(lambda_raw, x_max);
  const int row = has_row_idx ? static_cast<int>(row_idx[i]) - 1 : row_index_cpp(y, k_val);
  const int col = m_index_cpp(lambda);

  const double ti1 = x_vals[col];
  const double ti = x_vals[col + 1];
  const double hi = ti - ti1;
  const double fti1 = q_mat(row, col);
  const double fti = q_mat(row, col + 1);
  const double zi1 = sq_mat(row, col);
  const double zi = sq_mat(row, col + 1);

  const double diff1 = lambda - ti1;
  const double diff2 = ti - lambda;
  const double diff3 = fti / hi - zi * hi / 6.0;
  const double diff4 = fti1 / hi - zi1 * hi / 6.0;
  const double zdi = zi / hi;
  const double zdi1 = zi1 / hi;
  d1 = zdi * diff1 * diff1 / 2.0 - zdi1 * diff2 * diff2 / 2.0 + diff3 - diff4;
  d2 = zdi * diff1 + zdi1 * diff2;
}

// [[Rcpp::export]]
NumericVector rctd_cpp_row_idx1(NumericVector y, int k_val) {
  NumericVector out(y.size());
  for (int i = 0; i < y.size(); ++i) {
    out[i] = static_cast<double>(row_index_cpp(y[i], k_val) + 1);
  }
  return out;
}

// [[Rcpp::export]]
NumericMatrix rctd_cpp_row_idx_mat(NumericMatrix y, int k_val) {
  NumericMatrix out(y.nrow(), y.ncol());
  for (int j = 0; j < y.ncol(); ++j) {
    for (int i = 0; i < y.nrow(); ++i) {
      out(i, j) = static_cast<double>(row_index_cpp(y(i, j), k_val) + 1);
    }
  }
  return out;
}

// [[Rcpp::export]]
double rctd_cpp_calc_log_l_sum(NumericVector lambda, NumericVector y, Nullable<NumericVector> row_idx_in,
                               NumericMatrix q_mat, NumericMatrix sq_mat, NumericVector x_vals, int k_val) {
  const bool has_row_idx = row_idx_in.isNotNull();
  NumericVector row_idx;
  if (has_row_idx) row_idx = NumericVector(row_idx_in);
  const double x_max = max(x_vals);
  double total = 0.0;
  for (int i = 0; i < lambda.size(); ++i) {
    total -= spline_value_cpp(y[i], lambda[i], i, has_row_idx, row_idx, q_mat, sq_mat, x_vals, k_val, x_max);
  }
  return total;
}

// [[Rcpp::export]]
NumericVector rctd_cpp_calc_log_l_vec(NumericVector lambda, NumericVector y, Nullable<NumericVector> row_idx_in,
                                      NumericMatrix q_mat, NumericMatrix sq_mat, NumericVector x_vals, int k_val) {
  const bool has_row_idx = row_idx_in.isNotNull();
  NumericVector row_idx;
  if (has_row_idx) row_idx = NumericVector(row_idx_in);
  const double x_max = max(x_vals);
  NumericVector out(lambda.size());
  for (int i = 0; i < lambda.size(); ++i) {
    out[i] = -spline_value_cpp(y[i], lambda[i], i, has_row_idx, row_idx, q_mat, sq_mat, x_vals, k_val, x_max);
  }
  return out;
}

// [[Rcpp::export]]
List rctd_cpp_get_d1_d2(NumericVector y, NumericVector lambda, Nullable<NumericVector> row_idx_in,
                        NumericMatrix q_mat, NumericMatrix sq_mat, NumericVector x_vals, int k_val) {
  const bool has_row_idx = row_idx_in.isNotNull();
  NumericVector row_idx;
  if (has_row_idx) row_idx = NumericVector(row_idx_in);
  const double x_max = max(x_vals);
  NumericVector d1_vec(lambda.size());
  NumericVector d2_vec(lambda.size());
  for (int i = 0; i < lambda.size(); ++i) {
    double d1, d2;
    spline_derivs_cpp(y[i], lambda[i], i, has_row_idx, row_idx, q_mat, sq_mat, x_vals, k_val, x_max, d1, d2);
    d1_vec[i] = d1;
    d2_vec[i] = d2;
  }
  return List::create(_["d1_vec"] = d1_vec, _["d2_vec"] = d2_vec);
}

// [[Rcpp::export]]
List rctd_cpp_get_der_fast_nonbulk(NumericMatrix s, NumericVector y, NumericVector lambda,
                                   Nullable<NumericVector> row_idx_in, NumericMatrix q_mat,
                                   NumericMatrix sq_mat, NumericVector x_vals, int k_val) {
  const bool has_row_idx = row_idx_in.isNotNull();
  NumericVector row_idx;
  if (has_row_idx) row_idx = NumericVector(row_idx_in);
  const double x_max = max(x_vals);
  const int n = s.nrow();
  const int p = s.ncol();
  NumericMatrix grad(1, p);
  NumericMatrix hess(p, p);

  for (int i = 0; i < n; ++i) {
    double d1, d2;
    spline_derivs_cpp(y[i], lambda[i], i, has_row_idx, row_idx, q_mat, sq_mat, x_vals, k_val, x_max, d1, d2);
    for (int a = 0; a < p; ++a) {
      const double sa = s(i, a);
      grad(0, a) -= d1 * sa;
      for (int b = a; b < p; ++b) {
        hess(a, b) -= d2 * sa * s(i, b);
      }
    }
  }

  for (int a = 1; a < p; ++a) {
    for (int b = 0; b < a; ++b) {
      hess(a, b) = hess(b, a);
    }
  }
  return List::create(_["grad"] = grad, _["hess"] = hess);
}

// [[Rcpp::export]]
NumericVector rctd_cpp_solve_wls_p1(NumericMatrix s, NumericVector y, Nullable<NumericVector> row_idx_in,
                                    double initial, double n_umi, NumericMatrix q_mat,
                                    NumericMatrix sq_mat, NumericVector x_vals, int k_val) {
  const bool has_row_idx = row_idx_in.isNotNull();
  NumericVector row_idx;
  if (has_row_idx) row_idx = NumericVector(row_idx_in);
  const double x_max = max(x_vals);
  const double threshold = std::max(1e-4, n_umi * 1e-7);
  double solution = std::max(initial, 0.0);
  double grad = 0.0;
  double hess = 0.0;

  for (int i = 0; i < s.nrow(); ++i) {
    const double si = s(i, 0);
    double prediction = std::abs(si * solution);
    if (prediction < threshold) prediction = threshold;
    double d1, d2;
    spline_derivs_cpp(y[i], prediction, i, has_row_idx, row_idx, q_mat, sq_mat, x_vals, k_val, x_max, d1, d2);
    grad -= d1 * si;
    hess -= d2 * si * si;
  }

  const double d_vec = -grad;
  const double d_mat = std::max(hess, 1e-3);
  const double norm_factor = std::abs(d_mat);
  const double d_norm = d_vec / norm_factor;
  const double D_norm = d_mat / norm_factor + 1e-7;
  double step = d_norm / D_norm;
  if (step < -solution) step = -solution;
  NumericVector out(1);
  out[0] = solution + 0.3 * step;
  return out;
}

inline double qp2_objective(double x1, double x2, double D11, double D12, double D22,
                            double d1, double d2) {
  return 0.5 * (D11 * x1 * x1 + 2.0 * D12 * x1 * x2 + D22 * x2 * x2) -
         d1 * x1 - d2 * x2;
}

inline void consider_qp2_candidate(double x1, double x2, double lb1, double lb2,
                                   double D11, double D12, double D22,
                                   double d1, double d2,
                                   double& best_x1, double& best_x2, double& best_obj) {
  const double tol = 1e-10;
  if (x1 < lb1 - tol || x2 < lb2 - tol) return;
  if (x1 < lb1) x1 = lb1;
  if (x2 < lb2) x2 = lb2;
  const double obj = qp2_objective(x1, x2, D11, D12, D22, d1, d2);
  if (obj < best_obj) {
    best_obj = obj;
    best_x1 = x1;
    best_x2 = x2;
  }
}

// [[Rcpp::export]]
NumericVector rctd_cpp_solve_wls_p2(NumericMatrix s, NumericVector y, Nullable<NumericVector> row_idx_in,
                                    NumericVector initial, double n_umi, NumericMatrix q_mat,
                                    NumericMatrix sq_mat, NumericVector x_vals, int k_val) {
  const bool has_row_idx = row_idx_in.isNotNull();
  NumericVector row_idx;
  if (has_row_idx) row_idx = NumericVector(row_idx_in);
  const double x_max = max(x_vals);
  const double threshold = std::max(1e-4, n_umi * 1e-7);
  const double sol1 = std::max(static_cast<double>(initial[0]), 0.0);
  const double sol2 = std::max(static_cast<double>(initial[1]), 0.0);
  double grad1 = 0.0, grad2 = 0.0;
  double h11 = 0.0, h12 = 0.0, h22 = 0.0;

  for (int i = 0; i < s.nrow(); ++i) {
    const double s1 = s(i, 0);
    const double s2 = s(i, 1);
    double prediction = std::abs(s1 * sol1 + s2 * sol2);
    if (prediction < threshold) prediction = threshold;
    double d1, d2;
    spline_derivs_cpp(y[i], prediction, i, has_row_idx, row_idx, q_mat, sq_mat, x_vals, k_val, x_max, d1, d2);
    grad1 -= d1 * s1;
    grad2 -= d1 * s2;
    h11 -= d2 * s1 * s1;
    h12 -= d2 * s1 * s2;
    h22 -= d2 * s2 * s2;
  }

  const double disc = std::sqrt((h11 - h22) * (h11 - h22) + 4.0 * h12 * h12);
  const double lambda1 = (h11 + h22 + disc) / 2.0;
  const double lambda2 = (h11 + h22 - disc) / 2.0;
  const double f1 = std::max(lambda1, 1e-3);
  const double f2 = std::max(lambda2, 1e-3);
  double D11, D12, D22;
  if (disc < 1e-12) {
    D11 = f1;
    D12 = 0.0;
    D22 = f1;
  } else {
    const double scale = (f1 - f2) / (lambda1 - lambda2);
    D11 = f2 + scale * (h11 - lambda2);
    D12 = scale * h12;
    D22 = f2 + scale * (h22 - lambda2);
  }

  const double norm_factor = std::max(f1, f2);
  D11 = D11 / norm_factor + 1e-7;
  D12 = D12 / norm_factor;
  D22 = D22 / norm_factor + 1e-7;
  const double d1n = -grad1 / norm_factor;
  const double d2n = -grad2 / norm_factor;
  const double lb1 = -sol1;
  const double lb2 = -sol2;

  double best_x1 = lb1;
  double best_x2 = lb2;
  double best_obj = qp2_objective(lb1, lb2, D11, D12, D22, d1n, d2n);
  const double det = D11 * D22 - D12 * D12;
  if (std::abs(det) > 1e-14) {
    consider_qp2_candidate((D22 * d1n - D12 * d2n) / det,
                           (D11 * d2n - D12 * d1n) / det,
                           lb1, lb2, D11, D12, D22, d1n, d2n,
                           best_x1, best_x2, best_obj);
  }
  consider_qp2_candidate(lb1, (d2n - D12 * lb1) / D22,
                         lb1, lb2, D11, D12, D22, d1n, d2n,
                         best_x1, best_x2, best_obj);
  consider_qp2_candidate((d1n - D12 * lb2) / D11, lb2,
                         lb1, lb2, D11, D12, D22, d1n, d2n,
                         best_x1, best_x2, best_obj);

  NumericVector out(2);
  out[0] = sol1 + 0.3 * best_x1;
  out[1] = sol2 + 0.3 * best_x2;
  return out;
}

// [[Rcpp::export]]
List rctd_cpp_irwls_sparse_p12(NumericMatrix s, NumericVector y, Nullable<NumericVector> row_idx_in,
                               double n_umi, int n_iter, double min_change,
                               NumericMatrix q_mat, NumericMatrix sq_mat,
                               NumericVector x_vals, int k_val) {
  const int p = s.ncol();
  NumericVector solution(p);
  for (int j = 0; j < p; ++j) solution[j] = 1.0 / p;
  double change = 1.0;
  int iterations = 0;
  while (change > min_change && iterations < n_iter) {
    NumericVector new_solution;
    if (p == 1) {
      new_solution = rctd_cpp_solve_wls_p1(s, y, row_idx_in, solution[0], n_umi, q_mat, sq_mat, x_vals, k_val);
    } else {
      new_solution = rctd_cpp_solve_wls_p2(s, y, row_idx_in, solution, n_umi, q_mat, sq_mat, x_vals, k_val);
    }
    change = 0.0;
    for (int j = 0; j < p; ++j) {
      change += std::abs(new_solution[j] - solution[j]);
      solution[j] = new_solution[j];
    }
    ++iterations;
  }

  const bool has_row_idx = row_idx_in.isNotNull();
  NumericVector row_idx;
  if (has_row_idx) row_idx = NumericVector(row_idx_in);
  const double x_max = max(x_vals);
  double score = 0.0;
  for (int i = 0; i < s.nrow(); ++i) {
    double prediction = 0.0;
    for (int j = 0; j < p; ++j) {
      prediction += s(i, j) * solution[j];
    }
    score -= spline_value_cpp(y[i], prediction, i, has_row_idx, row_idx, q_mat, sq_mat, x_vals, k_val, x_max);
  }
  return List::create(_["weights"] = solution, _["converged"] = (change <= min_change), _["score"] = score);
}

inline arma::vec solve_bound_qp_active(const arma::mat& D, const arma::vec& d,
                                       const arma::vec& lb) {
  const int p = d.n_elem;
  arma::vec x(p);
  arma::uvec active(p, arma::fill::zeros);
  const double tol = 1e-10;

  for (int iter = 0; iter < 64; ++iter) {
    arma::uvec free_idx = arma::find(active == 0);
    arma::uvec active_idx = arma::find(active == 1);
    x = lb;

    if (free_idx.n_elem > 0) {
      arma::vec rhs = d.elem(free_idx);
      if (active_idx.n_elem > 0) {
        rhs -= D.submat(free_idx, active_idx) * lb.elem(active_idx);
      }
      arma::mat Dff = D.submat(free_idx, free_idx);
      arma::vec xf = arma::solve(Dff, rhs, arma::solve_opts::likely_sympd);
      x.elem(free_idx) = xf;

      arma::vec violation = lb.elem(free_idx) - xf;
      arma::uword worst_pos = violation.index_max();
      if (violation[worst_pos] > tol) {
        active[free_idx[worst_pos]] = 1;
        continue;
      }
    }

    arma::vec grad = D * x - d;
    active_idx = arma::find(active == 1);
    if (active_idx.n_elem > 0) {
      arma::vec active_grad = grad.elem(active_idx);
      arma::uword worst_pos = active_grad.index_min();
      if (active_grad[worst_pos] < -tol) {
        active[active_idx[worst_pos]] = 0;
        continue;
      }
    }
    return x;
  }
  return x;
}

// [[Rcpp::export]]
List rctd_cpp_irwls_full_nonbulk(NumericMatrix s, NumericVector y, Nullable<NumericVector> row_idx_in,
                                 double n_umi, int n_iter, double min_change,
                                 NumericMatrix q_mat, NumericMatrix sq_mat,
                                 NumericVector x_vals, int k_val) {
  const bool has_row_idx = row_idx_in.isNotNull();
  NumericVector row_idx;
  if (has_row_idx) row_idx = NumericVector(row_idx_in);
  const double x_max = max(x_vals);
  const double threshold = std::max(1e-4, n_umi * 1e-7);
  const int n = s.nrow();
  const int p = s.ncol();
  arma::vec solution(p);
  solution.fill(1.0 / p);

  double change = 1.0;
  int iterations = 0;
  while (change > min_change && iterations < n_iter) {
    arma::vec grad(p, arma::fill::zeros);
    arma::mat hess(p, p, arma::fill::zeros);

    for (int i = 0; i < n; ++i) {
      double prediction = 0.0;
      for (int j = 0; j < p; ++j) prediction += s(i, j) * solution[j];
      prediction = std::abs(prediction);
      if (prediction < threshold) prediction = threshold;

      double d1, d2;
      spline_derivs_cpp(y[i], prediction, i, has_row_idx, row_idx,
                        q_mat, sq_mat, x_vals, k_val, x_max, d1, d2);
      for (int a = 0; a < p; ++a) {
        const double sa = s(i, a);
        grad[a] -= d1 * sa;
        for (int b = a; b < p; ++b) {
          hess(a, b) -= d2 * sa * s(i, b);
        }
      }
    }
    for (int a = 1; a < p; ++a) {
      for (int b = 0; b < a; ++b) hess(a, b) = hess(b, a);
    }

    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, hess);
    for (int j = 0; j < p; ++j) {
      if (eigval[j] < 1e-3) eigval[j] = 1e-3;
    }
    arma::mat D = eigvec * arma::diagmat(eigval) * eigvec.t();
    const double norm_factor = eigval.max();
    D /= norm_factor;
    D.diag() += 1e-7;
    arma::vec d = -grad / norm_factor;
    arma::vec lb = -solution;
    arma::vec step = solve_bound_qp_active(D, d, lb);
    arma::vec new_solution = solution + 0.3 * step;

    change = arma::sum(arma::abs(new_solution - solution));
    solution = new_solution;
    ++iterations;
  }

  NumericVector weights(p);
  for (int j = 0; j < p; ++j) weights[j] = solution[j];
  return List::create(_["weights"] = weights, _["converged"] = (change <= min_change));
}

struct RctdSparseFit2 {
  double w1;
  double w2;
  bool converged;
  double score;
};

inline double subset_score_p1(const NumericMatrix& profiles, const NumericVector& y,
                              int c1, double w1, bool has_row_idx,
                              const NumericVector& row_idx, const NumericMatrix& q_mat,
                              const NumericMatrix& sq_mat, const NumericVector& x_vals,
                              int k_val, double x_max) {
  double score = 0.0;
  for (int i = 0; i < profiles.nrow(); ++i) {
    score -= spline_value_cpp(y[i], profiles(i, c1) * w1, i, has_row_idx, row_idx,
                              q_mat, sq_mat, x_vals, k_val, x_max);
  }
  return score;
}

inline double subset_score_p2(const NumericMatrix& profiles, const NumericVector& y,
                              int c1, int c2, double w1, double w2,
                              bool has_row_idx, const NumericVector& row_idx,
                              const NumericMatrix& q_mat, const NumericMatrix& sq_mat,
                              const NumericVector& x_vals, int k_val, double x_max) {
  double score = 0.0;
  for (int i = 0; i < profiles.nrow(); ++i) {
    const double prediction = profiles(i, c1) * w1 + profiles(i, c2) * w2;
    score -= spline_value_cpp(y[i], prediction, i, has_row_idx, row_idx,
                              q_mat, sq_mat, x_vals, k_val, x_max);
  }
  return score;
}

inline double subset_wls_p1_update(const NumericMatrix& profiles, const NumericVector& y,
                                   int c1, double initial, double n_umi,
                                   bool has_row_idx, const NumericVector& row_idx,
                                   const NumericMatrix& q_mat, const NumericMatrix& sq_mat,
                                   const NumericVector& x_vals, int k_val, double x_max) {
  const double threshold = std::max(1e-4, n_umi * 1e-7);
  double solution = std::max(initial, 0.0);
  double grad = 0.0;
  double hess = 0.0;

  for (int i = 0; i < profiles.nrow(); ++i) {
    const double si = profiles(i, c1);
    double prediction = std::abs(si * solution);
    if (prediction < threshold) prediction = threshold;
    double d1, d2;
    spline_derivs_cpp(y[i], prediction, i, has_row_idx, row_idx,
                      q_mat, sq_mat, x_vals, k_val, x_max, d1, d2);
    grad -= d1 * si;
    hess -= d2 * si * si;
  }

  const double d_vec = -grad;
  const double d_mat = std::max(hess, 1e-3);
  const double norm_factor = std::abs(d_mat);
  const double d_norm = d_vec / norm_factor;
  const double D_norm = d_mat / norm_factor + 1e-7;
  double step = d_norm / D_norm;
  if (step < -solution) step = -solution;
  return solution + 0.3 * step;
}

inline void subset_wls_p2_update(const NumericMatrix& profiles, const NumericVector& y,
                                 int c1, int c2, double initial1, double initial2,
                                 double n_umi, bool has_row_idx,
                                 const NumericVector& row_idx, const NumericMatrix& q_mat,
                                 const NumericMatrix& sq_mat, const NumericVector& x_vals,
                                 int k_val, double x_max, double& out1, double& out2) {
  const double threshold = std::max(1e-4, n_umi * 1e-7);
  const double sol1 = std::max(initial1, 0.0);
  const double sol2 = std::max(initial2, 0.0);
  double grad1 = 0.0, grad2 = 0.0;
  double h11 = 0.0, h12 = 0.0, h22 = 0.0;

  for (int i = 0; i < profiles.nrow(); ++i) {
    const double s1 = profiles(i, c1);
    const double s2 = profiles(i, c2);
    double prediction = std::abs(s1 * sol1 + s2 * sol2);
    if (prediction < threshold) prediction = threshold;
    double d1, d2;
    spline_derivs_cpp(y[i], prediction, i, has_row_idx, row_idx,
                      q_mat, sq_mat, x_vals, k_val, x_max, d1, d2);
    grad1 -= d1 * s1;
    grad2 -= d1 * s2;
    h11 -= d2 * s1 * s1;
    h12 -= d2 * s1 * s2;
    h22 -= d2 * s2 * s2;
  }

  const double disc = std::sqrt((h11 - h22) * (h11 - h22) + 4.0 * h12 * h12);
  const double lambda1 = (h11 + h22 + disc) / 2.0;
  const double lambda2 = (h11 + h22 - disc) / 2.0;
  const double f1 = std::max(lambda1, 1e-3);
  const double f2 = std::max(lambda2, 1e-3);
  double D11, D12, D22;
  if (disc < 1e-12) {
    D11 = f1;
    D12 = 0.0;
    D22 = f1;
  } else {
    const double scale = (f1 - f2) / (lambda1 - lambda2);
    D11 = f2 + scale * (h11 - lambda2);
    D12 = scale * h12;
    D22 = f2 + scale * (h22 - lambda2);
  }

  const double norm_factor = std::max(f1, f2);
  D11 = D11 / norm_factor + 1e-7;
  D12 = D12 / norm_factor;
  D22 = D22 / norm_factor + 1e-7;
  const double d1n = -grad1 / norm_factor;
  const double d2n = -grad2 / norm_factor;
  const double lb1 = -sol1;
  const double lb2 = -sol2;

  double best_x1 = lb1;
  double best_x2 = lb2;
  double best_obj = qp2_objective(lb1, lb2, D11, D12, D22, d1n, d2n);
  const double det = D11 * D22 - D12 * D12;
  if (std::abs(det) > 1e-14) {
    consider_qp2_candidate((D22 * d1n - D12 * d2n) / det,
                           (D11 * d2n - D12 * d1n) / det,
                           lb1, lb2, D11, D12, D22, d1n, d2n,
                           best_x1, best_x2, best_obj);
  }
  consider_qp2_candidate(lb1, (d2n - D12 * lb1) / D22,
                         lb1, lb2, D11, D12, D22, d1n, d2n,
                         best_x1, best_x2, best_obj);
  consider_qp2_candidate((d1n - D12 * lb2) / D11, lb2,
                         lb1, lb2, D11, D12, D22, d1n, d2n,
                         best_x1, best_x2, best_obj);

  out1 = sol1 + 0.3 * best_x1;
  out2 = sol2 + 0.3 * best_x2;
}

inline RctdSparseFit2 irwls_subset_p1(const NumericMatrix& profiles, const NumericVector& y,
                                     int c1, double n_umi, int n_iter, double min_change,
                                     bool has_row_idx, const NumericVector& row_idx,
                                     const NumericMatrix& q_mat, const NumericMatrix& sq_mat,
                                     const NumericVector& x_vals, int k_val, double x_max) {
  RctdSparseFit2 out;
  out.w1 = 1.0;
  out.w2 = 0.0;
  double change = 1.0;
  int iterations = 0;
  while (change > min_change && iterations < n_iter) {
    const double new_w1 = subset_wls_p1_update(profiles, y, c1, out.w1, n_umi,
                                               has_row_idx, row_idx, q_mat, sq_mat,
                                               x_vals, k_val, x_max);
    change = std::abs(new_w1 - out.w1);
    out.w1 = new_w1;
    ++iterations;
  }
  out.converged = (change <= min_change);
  out.score = subset_score_p1(profiles, y, c1, out.w1, has_row_idx, row_idx,
                              q_mat, sq_mat, x_vals, k_val, x_max);
  return out;
}

inline RctdSparseFit2 irwls_subset_p2(const NumericMatrix& profiles, const NumericVector& y,
                                     int c1, int c2, double n_umi, int n_iter,
                                     double min_change, bool has_row_idx,
                                     const NumericVector& row_idx, const NumericMatrix& q_mat,
                                     const NumericMatrix& sq_mat, const NumericVector& x_vals,
                                     int k_val, double x_max) {
  RctdSparseFit2 out;
  out.w1 = 0.5;
  out.w2 = 0.5;
  double change = 1.0;
  int iterations = 0;
  while (change > min_change && iterations < n_iter) {
    double new_w1, new_w2;
    subset_wls_p2_update(profiles, y, c1, c2, out.w1, out.w2, n_umi,
                         has_row_idx, row_idx, q_mat, sq_mat, x_vals, k_val,
                         x_max, new_w1, new_w2);
    change = std::abs(new_w1 - out.w1) + std::abs(new_w2 - out.w2);
    out.w1 = new_w1;
    out.w2 = new_w2;
    ++iterations;
  }
  out.converged = (change <= min_change);
  out.score = subset_score_p2(profiles, y, c1, c2, out.w1, out.w2,
                              has_row_idx, row_idx, q_mat, sq_mat, x_vals,
                              k_val, x_max);
  return out;
}

// [[Rcpp::export]]
List rctd_cpp_score_sparse_candidates(NumericMatrix profiles, NumericVector y,
                                      IntegerVector candidate_cols,
                                      Nullable<NumericVector> row_idx_in, double n_umi,
                                      double min_change, NumericMatrix q_mat,
                                      NumericMatrix sq_mat, NumericVector x_vals,
                                      int k_val) {
  const bool has_row_idx = row_idx_in.isNotNull();
  NumericVector row_idx;
  if (has_row_idx) row_idx = NumericVector(row_idx_in);
  const double x_max = max(x_vals);
  const int m = candidate_cols.size();
  NumericVector singlet_scores(m);
  NumericMatrix score_mat(m, m);
  int min_i = 0;
  int min_j = 1;
  double min_score = 0.0;
  bool have_pair = false;

  for (int i = 0; i < m; ++i) {
    const int c1 = candidate_cols[i] - 1;
    RctdSparseFit2 fit = irwls_subset_p1(profiles, y, c1, n_umi, 8,
                                         min_change, has_row_idx, row_idx,
                                         q_mat, sq_mat, x_vals, k_val, x_max);
    singlet_scores[i] = fit.score;
  }

  for (int i = 0; i < m - 1; ++i) {
    const int c1 = candidate_cols[i] - 1;
    for (int j = i + 1; j < m; ++j) {
      const int c2 = candidate_cols[j] - 1;
      RctdSparseFit2 fit = irwls_subset_p2(profiles, y, c1, c2, n_umi, 8,
                                           min_change, has_row_idx, row_idx,
                                           q_mat, sq_mat, x_vals, k_val, x_max);
      score_mat(i, j) = fit.score;
      score_mat(j, i) = fit.score;
      if (!have_pair || fit.score < min_score) {
        min_score = fit.score;
        min_i = i + 1;
        min_j = j + 1;
        have_pair = true;
      }
    }
  }

  return List::create(_["singlet_scores"] = singlet_scores,
                      _["score_mat"] = score_mat,
                      _["min_score"] = min_score,
                      _["min_i"] = min_i,
                      _["min_j"] = min_j);
}

// [[Rcpp::export]]
List rctd_cpp_fit_sparse_pair(NumericMatrix profiles, NumericVector y, IntegerVector pair_cols,
                              Nullable<NumericVector> row_idx_in, double n_umi,
                              int n_iter, double min_change, NumericMatrix q_mat,
                              NumericMatrix sq_mat, NumericVector x_vals, int k_val) {
  const bool has_row_idx = row_idx_in.isNotNull();
  NumericVector row_idx;
  if (has_row_idx) row_idx = NumericVector(row_idx_in);
  const double x_max = max(x_vals);
  RctdSparseFit2 fit = irwls_subset_p2(profiles, y, pair_cols[0] - 1, pair_cols[1] - 1,
                                       n_umi, n_iter, min_change, has_row_idx,
                                       row_idx, q_mat, sq_mat, x_vals, k_val, x_max);
  NumericVector weights(2);
  weights[0] = fit.w1;
  weights[1] = fit.w2;
  return List::create(_["weights"] = weights, _["converged"] = fit.converged,
                      _["score"] = fit.score);
}
