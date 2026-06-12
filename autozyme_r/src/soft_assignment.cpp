// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

// Fused soft-assignment kernel for monocle3's principal-graph fitting.
// Replaces R-side dist + min + exp + normalize + objective passes (each
// allocating ~770MB at the medium tier) with one C++ pass that backs the
// output P onto a pre-allocated NumericMatrix buffer (no copy on return).

// [[Rcpp::export]]
List zyme_soft_assignment(const arma::mat& X, const arma::mat& C, double sigma) {
  const arma::uword N = X.n_cols;
  const arma::uword K = C.n_cols;

  NumericMatrix P_out(N, K);
  arma::mat d(P_out.begin(), N, K, false, true);

  d = X.t() * C;                          // BLAS dgemm into d
  d *= -2.0;

  arma::vec X_sq = arma::sum(arma::square(X), 0).t();
  arma::vec C_sq = arma::sum(arma::square(C), 0).t();
  d.each_col() += X_sq;
  d.each_row() += C_sq.t();

  arma::vec min_dist = arma::min(d, 1);
  d.each_col() -= min_dist;

  d /= -sigma;
  d.transform([](double v) { return std::exp(v); });

  arma::vec rs = arma::sum(d, 1);
  d.each_col() /= rs;

  double obj = -sigma * arma::accu(arma::log(rs) - min_dist / sigma);
  return List::create(_["P"] = P_out, _["obj"] = obj);
}
