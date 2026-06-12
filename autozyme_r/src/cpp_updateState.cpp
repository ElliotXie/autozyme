// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

// Fused IRLS-iter body for MAST's bayesglm.fit.loop.updateState path.
// Replaces ~10 vector ops × thousands of iters per gene with a single
// C++ call. Hard-codes binomial+logit and gaussian+identity families
// (the only ones MAST::zlm uses).

// [[Rcpp::export]]
List cpp_updateState(NumericVector eta_, NumericVector mu_,
                     NumericVector mu_eta_val_, NumericVector varmu_,
                     double dispersion, NumericVector prior_sd_,
                     NumericMatrix x_aug_, NumericMatrix x_nobs_,
                     NumericVector y_, NumericVector weights_, NumericVector offset_,
                     NumericVector prior_mean_, NumericVector prior_scale_,
                     NumericVector prior_df_,
                     int family_idx, int intercept, int scaled) {
  const int nobs = y_.size();
  const int nvars = x_nobs_.cols();
  const int nstar = nobs + nvars;
  bool is_binomial = (family_idx == 1);

  arma::vec eta(eta_.begin(), nobs, false);
  arma::vec mu(mu_.begin(), nobs, false);
  arma::vec mu_eta_val(mu_eta_val_.begin(), nobs, false);
  arma::vec varmu(varmu_.begin(), nobs, false);
  arma::mat x_aug(x_aug_.begin(), nstar, nvars, false);
  arma::mat x_nobs(x_nobs_.begin(), nobs, nvars, false);
  arma::vec y(y_.begin(), nobs, false);
  arma::vec weights(weights_.begin(), nobs, false);
  arma::vec offset(offset_.begin(), nobs, false);
  arma::vec prior_mean(prior_mean_.begin(), nvars, false);
  arma::vec prior_scale(prior_scale_.begin(), nvars, false);
  arma::vec prior_df(prior_df_.begin(), nvars, false);
  arma::vec prior_sd(prior_sd_.begin(), nvars);

  arma::vec z = (eta - offset) + (y - mu) / mu_eta_val;
  arma::vec w = arma::sqrt((weights % arma::square(mu_eta_val)) / varmu);

  arma::vec z_star(nstar);
  z_star.head(nobs) = z;
  z_star.tail(nvars) = prior_mean;
  arma::vec w_star(nstar);
  w_star.head(nobs) = w;
  double sqrt_disp = std::sqrt(dispersion);
  w_star.tail(nvars) = sqrt_disp / prior_scale;

  arma::mat Xw = x_aug.each_col() % w_star;
  arma::vec yw = z_star % w_star;

  arma::mat A = Xw.t() * Xw;
  arma::vec b = Xw.t() * yw;
  arma::mat R = arma::chol(A);
  arma::vec coefs = arma::solve(arma::trimatu(R),
                                arma::solve(arma::trimatl(R.t()), b));
  arma::mat V_coefs = arma::inv_sympd(A);

  bool has_finite_df = false;
  for (int j = 0; j < nvars; ++j) {
    if (!std::isinf(prior_df(j))) { has_finite_df = true; break; }
  }
  if (has_finite_df) {
    arma::vec colMeansX = arma::mean(x_nobs).t();
    arma::vec centered_coefs = coefs;
    arma::vec sampling_var = V_coefs.diag();
    if (intercept != 0 && scaled != 0) {
      centered_coefs(0) = arma::dot(coefs, colMeansX);
      sampling_var(0) = arma::dot(V_coefs * colMeansX, colMeansX);
    }
    arma::vec sd_tmp = arma::sqrt(
      (arma::square(centered_coefs - prior_mean)
       + sampling_var * dispersion
       + prior_df % arma::square(prior_sd))
      / (1.0 + prior_df)
    );
    for (int j = 0; j < nvars; ++j) {
      if (!std::isinf(prior_df(j))) prior_sd(j) = sd_tmp(j);
    }
  }

  arma::vec predictions = x_nobs * coefs;

  double new_dispersion = dispersion;
  if (!is_binomial) {
    arma::vec resid_part = z % w - w % predictions;
    double mse_resid = arma::mean(arma::square(resid_part));
    arma::mat xV = x_nobs * V_coefs;
    double mse_uncertainty = std::max(0.0,
      arma::mean(arma::sum(xV % x_nobs, 1)) * dispersion);
    new_dispersion = mse_resid + mse_uncertainty;
  }

  arma::vec new_eta = predictions + offset;
  arma::vec new_mu(nobs), new_mu_eta_val(nobs), new_varmu(nobs);
  double new_dev = 0.0;

  if (is_binomial) {
    for (int i = 0; i < nobs; ++i) {
      double e = new_eta(i);
      new_mu(i) = (e >= 0) ? 1.0 / (1.0 + std::exp(-e))
                            : std::exp(e) / (1.0 + std::exp(e));
    }
    new_mu_eta_val = new_mu % (1.0 - new_mu);
    new_varmu = new_mu_eta_val;
    for (int i = 0; i < nobs; ++i) {
      double yi = y(i), mui = new_mu(i), wi = weights(i);
      double t1 = (yi == 0.0) ? 0.0 : yi * std::log(mui / yi);
      double t2 = (yi == 1.0) ? 0.0 : (1.0 - yi) * std::log((1.0 - mui) / (1.0 - yi));
      new_dev += -2.0 * wi * (t1 + t2);
    }
  } else {
    new_mu = new_eta;
    new_mu_eta_val.ones();
    new_varmu.ones();
    arma::vec resid = y - new_mu;
    new_dev = arma::accu(weights % arma::square(resid));
  }

  NumericVector coefs_R(coefs.begin(), coefs.end());
  NumericMatrix R_out(nvars, nvars);
  std::copy(R.begin(), R.end(), R_out.begin());
  IntegerVector pivot(nvars);
  for (int j = 0; j < nvars; ++j) pivot[j] = j + 1;
  List fit_qr = List::create(
    _["qr"] = R_out, _["qraux"] = NumericVector(nvars),
    _["pivot"] = pivot, _["tol"] = 1e-7, _["rank"] = nvars
  );
  fit_qr.attr("class") = "qr";
  List fit = List::create(
    _["coefficients"] = coefs_R, _["rank"] = nvars, _["qr"] = fit_qr
  );

  return List::create(
    _["Start"]      = coefs_R,
    _["Coefold"]    = coefs_R,
    _["eta"]        = NumericVector(new_eta.begin(), new_eta.end()),
    _["mu"]         = NumericVector(new_mu.begin(), new_mu.end()),
    _["mu.eta.val"] = NumericVector(new_mu_eta_val.begin(), new_mu_eta_val.end()),
    _["varmu"]      = NumericVector(new_varmu.begin(), new_varmu.end()),
    _["good"]       = LogicalVector(nobs, true),
    _["dispersion"] = new_dispersion,
    _["dev"]        = new_dev,
    _["fit"]        = fit,
    _["conv"]       = false,
    _["boundary"]   = false,
    _["prior.sd"]   = NumericVector(prior_sd.begin(), prior_sd.end()),
    _["z"]          = NumericVector(z.begin(), z.end()),
    _["w"]          = NumericVector(w.begin(), w.end())
  );
}
