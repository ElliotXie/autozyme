// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11)]]
//
// EM kernel for the decontx patch (replaces celda::decontXEM).
// Lifted verbatim from test_decontx/pipeline/run.R (round 45+60).
//
// Parallelism: std::thread with per-thread phi/native accumulators (no shared-
// memory race — reductions happen after join). Bit-exact vs serial on macOS
// at all tested tiers (per the iteration log). Threshold: parallel when
// nC >= 500 AND n_threads_request > 1 (round 61).

#include <RcppEigen.h>
#include <thread>
#include <vector>

using namespace Rcpp;

// [[Rcpp::export]]
List fast_decontXEM_cpp(const Eigen::MappedSparseMatrix<double>& counts,
                        const NumericVector& counts_colsums,
                        const NumericVector& theta,
                        const bool estimate_eta,
                        const NumericMatrix& eta,
                        const NumericMatrix& phi,
                        const IntegerVector& z,
                        const bool estimate_delta,
                        const NumericVector& delta,
                        const double pseudocount,
                        const int n_threads_request) {
  const int nC = counts.cols();
  const int nG = phi.nrow();
  const int K  = phi.ncol();
  NumericVector new_theta(nC);
  NumericVector native_total(nC);
  NumericMatrix new_phi(nG, K);
  NumericMatrix new_eta(nG, K);

  const int n_threads = (nC >= 500 && n_threads_request > 1) ? n_threads_request : 1;
  if (n_threads > 1) {
    std::vector<std::vector<double>> tphi(n_threads, std::vector<double>((size_t)nG * K, 0.0));
    std::vector<std::vector<double>> tnat(n_threads, std::vector<double>((size_t)nC, 0.0));
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
      const int j_start = (int)((long long)nC * t / n_threads);
      const int j_end   = (int)((long long)nC * (t + 1) / n_threads);
      threads.emplace_back([&, t, j_start, j_end]() {
        std::vector<double>& phi_t = tphi[t];
        std::vector<double>& nat_t = tnat[t];
        for (int j = j_start; j < j_end; ++j) {
          const int k = z[j] - 1;
          const double t_j  = theta[j] + pseudocount;
          const double t_jc = 1 - theta[j] + pseudocount;
          for (Eigen::MappedSparseMatrix<double>::InnerIterator it(counts, j); it; ++it) {
            const int i = it.index();
            const double x = it.value();
            const double pn = (phi[nG * k + i] + pseudocount) * t_j;
            const double pc = (eta[nG * k + i] + pseudocount) * t_jc;
            const double normp = pn / (pn + pc);
            const double px = normp * x;
            phi_t[(size_t)nG * k + i] += px;
            nat_t[j] += px;
          }
        }
      });
    }
    for (auto& th : threads) th.join();
    for (int t = 0; t < n_threads; ++t) {
      const std::vector<double>& phi_t = tphi[t];
      const std::vector<double>& nat_t = tnat[t];
      for (size_t idx = 0, end = (size_t)nG * K; idx < end; ++idx)
        new_phi[idx] += phi_t[idx];
      for (int j = 0; j < nC; ++j)
        native_total[j] += nat_t[j];
    }
  } else {
    for (int j = 0; j < nC; ++j) {
      const int k = z[j] - 1;
      const double t_j  = theta[j] + pseudocount;
      const double t_jc = 1 - theta[j] + pseudocount;
      for (Eigen::MappedSparseMatrix<double>::InnerIterator it(counts, j); it; ++it) {
        const int i = it.index();
        const double x = it.value();
        const double pn = (phi[nG * k + i] + pseudocount) * t_j;
        const double pc = (eta[nG * k + i] + pseudocount) * t_jc;
        const double normp = pn / (pn + pc);
        const double px = normp * x;
        new_phi[nG * k + i] += px;
        native_total[j] += px;
      }
    }
  }

  if (estimate_eta) {
    NumericVector phi_rowsum(nG);
    for (int k = 0; k < K; ++k)
      for (int i = 0; i < nG; ++i)
        phi_rowsum[i] += new_phi[nG * k + i];
    for (int k = 0; k < K; ++k)
      for (int i = 0; i < nG; ++i)
        new_eta[nG * k + i] = phi_rowsum[i] - new_phi[nG * k + i];
  }
  NumericVector phi_colsum(K);
  for (int k = 0; k < K; ++k)
    for (int i = 0; i < nG; ++i)
      phi_colsum[k] += new_phi[nG * k + i];
  for (int k = 0; k < K; ++k)
    for (int i = 0; i < nG; ++i)
      new_phi[nG * k + i] /= phi_colsum[k];
  if (estimate_eta) {
    NumericVector eta_colsum(K);
    for (int k = 0; k < K; ++k)
      for (int i = 0; i < nG; ++i)
        eta_colsum[k] += new_eta[nG * k + i];
    for (int k = 0; k < K; ++k)
      for (int i = 0; i < nG; ++i)
        new_eta[nG * k + i] /= eta_colsum[k];
  } else {
    new_eta = eta;
  }
  for (int j = 0; j < nC; ++j)
    new_theta[j] = (native_total[j] + delta[0]) /
                   (counts_colsums[j] + delta[0] + delta[1]);
  NumericVector contamination(nC);
  for (int j = 0; j < nC; ++j) {
    if (counts_colsums[j] > 0)
      contamination[j] = 1.0 - native_total[j] / counts_colsums[j];
  }
  NumericVector new_delta = delta;
  if (estimate_delta) {
    Environment pkg = Environment::namespace_env("MCMCprecision");
    Function fd = pkg["fit_dirichlet"];
    NumericMatrix tm(nC, 2);
    for (int j = 0; j < nC; ++j) { tm(j, 0) = new_theta[j]; tm(j, 1) = 1 - new_theta[j]; }
    List fr = fd(tm);
    new_delta = fr["alpha"];
  }
  return List::create(Named("phi") = new_phi, Named("eta") = new_eta,
                      Named("theta") = new_theta, Named("delta") = new_delta,
                      Named("contamination") = contamination);
}
