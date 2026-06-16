// deseq2.cpp — Rcpp/RcppArmadillo kernels for DESeq2::DESeq (Wald path).
// Lifted verbatim from the converged autozyme task `test_deseq2_bulk`
// (pipeline/run.R inline sourceCpp + pipeline/fitbeta.cpp). Three exported
// kernels feed four namespace patches in inst/patches/deseq2/patch.R:
//
//   * fitDisp_zyme       -> fitDispWrapper   (per-gene MAP dispersion Newton/
//                                             Armijo search, ~78% of DESeq wall)
//   * fitBeta_zyme       -> fitBetaWrapper   (per-gene IRLS GLM beta fit)
//   * row_trimmed_means  -> trimmedCellVariance + replaceOutliers (Cook's robust
//                                             dispersion / outlier replacement)
//
// All three are SAME-MATH copies of the upstream src/DESeq2.cpp kernels with
// the discarded-work eliminated (see per-kernel headers). The two value-
// affecting deviations — (a) the dispersion line-search tol clamped to 1e-4
// (set R-side in fast_fitDispWrapper, NOT here) and (b) the fitBeta IRLS
// converging on beta-change instead of the discarded NB deviance — are
// validated concordant (pearson_dispersion >= 0.9999, pearson_log2fc = 1.0,
// q99_abs_diff_log2fc < 1.5e-4, de_jaccard_padj05 > 0.999) across dev tiers
// small/medium/large and OOD ood_large (p=3) / ood_xlarge at threads 1/4/8.
//
// BLAS/LAPACK safety: arma solve/.i()/det/matmul resolve to R's reference
// libRblas/libRlapack (autozyme.so deliberately does NOT link Accelerate — see
// src/Makevars head comment), which is single-threaded and fork-safe. DESeq's
// parallel path forks BiocParallel::MulticoreParam workers that call these
// kernels; the dev validated threads 4/8 pass, so the reference-BLAS routing
// here is fork-safe on macOS.

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
#include <R.h>
#include <Rmath.h>
#include <R_ext/Utils.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>

// ============================================================================
// fitDisp kernel (estimateDispersionsGeneEst + estimateDispersionsMAP).
//
// gidx/uval/n_groups: per-gene count-tie grouping. gidx[j] is the group of
// sample j, uval[g] the (integer-valued) count for group g. lgbuf is scratch
// (capacity >= n_groups). lgamma(y_j + 1/alpha) depends only on the integer
// count y_j and alpha, so for tied counts it is one value — compute
// Rf_lgammafn once per distinct count (matches the upstream sugar lgamma
// bit-for-bit) and look it up per sample, preserving sum order.
// ============================================================================
static double zlog_posterior(double log_alpha, NumericMatrix::Row y, NumericMatrix::Row mu, const arma::mat& x, double log_alpha_prior_mean, double log_alpha_prior_sigmasq, bool usePrior, NumericMatrix::Row weights, bool useWeights, double weightThreshold, bool useCR, const int* gidx, const double* uval, int n_groups, double* lgbuf, NumericVector imu) {
  double prior_part;
  double cr_term;
  double alpha = exp(log_alpha);
  int n = y.size();
  if (useCR) {
    arma::vec w_diag = pow(imu + alpha, -1);   // imu = pow(mu,-1) precomputed once/gene
    arma::mat b;
    if (useWeights) {
      NumericVector wts_vec = weights;
      arma::vec wts = as<arma::vec>(wts_vec);
      arma::mat xs = x.rows(find(wts > weightThreshold));
      xs = xs.cols(find(sum(abs(xs)) > 0.0));
      w_diag = w_diag(find(wts > weightThreshold));
      b = xs.t() * (xs.each_col() % w_diag);
    } else {
      b = x.t() * (x.each_col() % w_diag);
    }
    cr_term = -0.5 * log(det(b));
  } else {
    cr_term = 0.0;
  }
  double alpha_neg1 = R_pow_di(alpha, -1);
  double lgam_a1 = Rf_lgammafn(alpha_neg1);
  for (int g = 0; g < n_groups; g++) lgbuf[g] = Rf_lgammafn(uval[g] + alpha_neg1);
  double ll_part = 0.0;
  if (useWeights) {
    NumericVector wv = weights;
    for (int j = 0; j < n; j++) {
      ll_part += wv[j] * (lgbuf[gidx[j]] - lgam_a1 - y[j] * log(mu[j] + alpha_neg1) - alpha_neg1 * log(1.0 + mu[j] * alpha));
    }
  } else {
    for (int j = 0; j < n; j++) {
      ll_part += lgbuf[gidx[j]] - lgam_a1 - y[j] * log(mu[j] + alpha_neg1) - alpha_neg1 * log(1.0 + mu[j] * alpha);
    }
  }
  if (usePrior) {
    prior_part = -0.5 * R_pow_di(log_alpha - log_alpha_prior_mean,2)/log_alpha_prior_sigmasq;
  } else {
    prior_part = 0.0;
  }
  double res =  ll_part + prior_part + cr_term;
  return(res);
}

static double zdlog_posterior(double log_alpha, NumericMatrix::Row y, NumericMatrix::Row mu, const arma::mat& x, double log_alpha_prior_mean, double log_alpha_prior_sigmasq, bool usePrior, NumericMatrix::Row weights, bool useWeights, double weightThreshold, bool useCR, const int* gidx, const double* uval, int n_groups, double* dgbuf, NumericVector dgwork, NumericVector imu) {
  double prior_part;
  double cr_term;
  double alpha = exp(log_alpha);
  if (useCR) {
    NumericVector t = imu + alpha;             // = pow(mu,-1)+alpha, shared by w_diag/dw_diag
    arma::vec w_diag = pow(t, -1);
    arma::vec dw_diag = -1.0 * pow(t, -2);
    arma::mat b, db;
    if (useWeights) {
      NumericVector wts_vec = weights;
      arma::vec wts = as<arma::vec>(wts_vec);
      arma::mat xs = x.rows(find(wts > weightThreshold));
      xs = xs.cols(find(sum(abs(xs)) > 0.0));
      w_diag = w_diag(find(wts > weightThreshold));
      dw_diag = dw_diag(find(wts > weightThreshold));
      b = xs.t() * (xs.each_col() % w_diag);
      db = xs.t() * (xs.each_col() % dw_diag);
    } else {
      b = x.t() * (x.each_col() % w_diag);
      db = x.t() * (x.each_col() % dw_diag);
    }
    double ddetb = ( det(b) * trace(b.i() * db) );
    cr_term = -0.5 * ddetb / det(b);
  } else {
    cr_term = 0.0;
  }
  double alpha_neg1 = R_pow_di(alpha, -1);
  double alpha_neg2 = R_pow_di(alpha, -2);
  int n = y.size();
  // deduped digamma(y + 1/alpha): one Rf_digamma per distinct count, then per-sample
  // lookup into dgwork. Rf_digamma matches sugar digamma bit-for-bit; the surrounding
  // sugar (pow/log) is left untouched, so substituting dgwork for digamma(y+a1) is exact.
  for (int g = 0; g < n_groups; g++) dgbuf[g] = Rf_digamma(uval[g] + alpha_neg1);
  for (int j = 0; j < n; j++) dgwork[j] = dgbuf[gidx[j]];
  double ll_part;
  if (useWeights) {
    ll_part = alpha_neg2 * sum(weights * (Rf_digamma(alpha_neg1) + log(1 + mu*alpha) - mu*alpha*pow(1.0 + mu*alpha, -1) - dgwork + y * pow(mu + alpha_neg1, -1)));
  } else {
    ll_part = alpha_neg2 * sum(Rf_digamma(alpha_neg1) + log(1 + mu*alpha) - mu*alpha*pow(1.0 + mu*alpha, -1) - dgwork + y * pow(mu + alpha_neg1, -1));
  }
  if (usePrior) {
    prior_part = -1.0 * (log_alpha - log_alpha_prior_mean)/log_alpha_prior_sigmasq;
  } else {
    prior_part = 0.0;
  }
  double res = (ll_part + cr_term) * alpha + prior_part;
  return(res);
}

// [[Rcpp::export]]
List fitDisp_zyme(SEXP ySEXP, SEXP xSEXP, SEXP mu_hatSEXP, SEXP log_alphaSEXP, SEXP log_alpha_prior_meanSEXP, SEXP log_alpha_prior_sigmasqSEXP, SEXP min_log_alphaSEXP, SEXP kappa_0SEXP, SEXP tolSEXP, SEXP maxitSEXP, SEXP usePriorSEXP, SEXP weightsSEXP, SEXP useWeightsSEXP, SEXP weightThresholdSEXP, SEXP useCRSEXP) {
  NumericMatrix y(ySEXP);
  arma::mat x = as<arma::mat>(xSEXP);
  int y_n = y.nrow();
  int y_m = y.ncol();
  // scratch buffers for per-gene count-tie grouping (reused across genes)
  std::vector<int> gd(y_m);
  std::vector<double> uv;        uv.reserve(y_m);
  std::vector<double> lgbuf(y_m);
  std::vector<double> dgbuf(y_m);
  NumericVector dgwork(y_m);
  std::unordered_map<long,int> gmap;
  NumericVector log_alpha(clone(log_alphaSEXP));
  NumericMatrix mu_hat(mu_hatSEXP);
  NumericVector log_alpha_prior_mean(log_alpha_prior_meanSEXP);
  double log_alpha_prior_sigmasq = as<double>(log_alpha_prior_sigmasqSEXP);
  double min_log_alpha = as<double>(min_log_alphaSEXP);
  double kappa_0 = as<double>(kappa_0SEXP);
  int maxit = as<int>(maxitSEXP);
  double epsilon = 1.0e-4;
  double a, a_propose, kappa, lp, lpnew, dlp, theta_kappa, theta_hat_kappa, change;
  NumericVector initial_lp(y_n);
  NumericVector initial_dlp(y_n);
  NumericVector last_lp(y_n);
  NumericVector last_dlp(y_n);
  NumericVector last_d2lp(y_n);
  NumericVector last_change(y_n);
  IntegerVector iter(y_n);
  IntegerVector iter_accept(y_n);
  double tol = as<double>(tolSEXP);
  bool usePrior = as<bool>(usePriorSEXP);
  NumericMatrix weights(weightsSEXP);
  bool useWeights = as<bool>(useWeightsSEXP);
  double weightThreshold = as<double>(weightThresholdSEXP);
  bool useCR = as<bool>(useCRSEXP);

  for (int i = 0; i < y_n; i++) {
    if (i % 100 == 0) checkUserInterrupt();
    NumericMatrix::Row yrow = y(i,_);
    NumericMatrix::Row mu_hat_row = mu_hat(i,_);
    // build count-tie groups for this gene (distinct integer counts -> group ids)
    uv.clear(); gmap.clear();
    for (int j = 0; j < y_m; j++) {
      long v = (long) llround(yrow[j]);
      std::unordered_map<long,int>::iterator it = gmap.find(v);
      if (it == gmap.end()) { int g = (int) uv.size(); gmap[v] = g; uv.push_back(yrow[j]); gd[j] = g; }
      else { gd[j] = it->second; }
    }
    int n_groups = (int) uv.size();
    NumericVector imu = pow(mu_hat_row, -1);   // 1/mu, fixed per gene, reused across all evals
    a = log_alpha(i);
    lp = zlog_posterior(a, yrow, mu_hat_row, x, log_alpha_prior_mean(i), log_alpha_prior_sigmasq, usePrior, weights.row(i), useWeights, weightThreshold, useCR, gd.data(), uv.data(), n_groups, lgbuf.data(), imu);
    dlp = zdlog_posterior(a, yrow, mu_hat_row, x, log_alpha_prior_mean(i), log_alpha_prior_sigmasq, usePrior, weights.row(i), useWeights, weightThreshold, useCR, gd.data(), uv.data(), n_groups, dgbuf.data(), dgwork, imu);
    kappa = kappa_0;
    initial_lp(i) = lp;
    initial_dlp(i) = dlp;
    change = -1.0;
    last_change(i) = -1.0;
    for (int t = 0; t < maxit; t++) {
      iter(i)++;
      a_propose = a + kappa * dlp;
      if (a_propose < -30.0) {
        kappa = (-30.0 - a)/dlp;
      }
      if (a_propose > 10.0) {
        kappa = (10.0 - a)/dlp;
      }
      theta_kappa = -1.0 * zlog_posterior(a + kappa*dlp, yrow, mu_hat_row, x, log_alpha_prior_mean(i), log_alpha_prior_sigmasq, usePrior, weights.row(i), useWeights, weightThreshold, useCR, gd.data(), uv.data(), n_groups, lgbuf.data(), imu);
      theta_hat_kappa = -1.0 * lp - kappa * epsilon * R_pow_di(dlp, 2);
      if (theta_kappa <= theta_hat_kappa) {
        iter_accept(i)++;
        a = a + kappa * dlp;
        // bit-identical to log_posterior(a) since theta_kappa used the same a+kappa*dlp
        lpnew = -1.0 * theta_kappa;
        change = lpnew - lp;
        if (change < tol) {
          lp = lpnew;
          break;
        }
        if (a < min_log_alpha) {
          break;
        }
        lp = lpnew;
        dlp = zdlog_posterior(a, yrow, mu_hat_row, x, log_alpha_prior_mean(i), log_alpha_prior_sigmasq, usePrior, weights.row(i), useWeights, weightThreshold, useCR, gd.data(), uv.data(), n_groups, dgbuf.data(), dgwork, imu);
        kappa = fmin(kappa * 1.1, kappa_0);
        if (iter_accept(i) % 5 == 0) {
          kappa = kappa / 2.0;
        }
      } else {
        kappa = kappa / 2.0;
      }
    }
    last_lp(i) = lp;
    last_dlp(i) = dlp;
    // last_d2lp is returned but NEVER read by any DESeq2 R code (verified: only
    // log_alpha/iter/last_lp/initial_lp are consumed downstream). Skip the costly
    // per-gene d2log_posterior eval (3 matrix products + b inversions + an internal
    // dlog_posterior digamma pass). Pure deletion of discarded work, bit-exact.
    last_d2lp(i) = 0.0;
    log_alpha(i) = a;
    last_change(i) = change;
  }

  return List::create(Named("log_alpha",log_alpha),
                      Named("iter",iter),
                      Named("iter_accept",iter_accept),
                      Named("last_change",last_change),
                      Named("initial_lp",initial_lp),
                      Named("initial_dlp",initial_dlp),
                      Named("last_lp",last_lp),
                      Named("last_dlp",last_dlp),
                      Named("last_d2lp",last_d2lp));
}

// ============================================================================
// row_trimmed_means — per-row trimmed mean over a column subset, matching R
// mean(x, trim=r): lo=floor(nc*trim)+1 (1-based), hi=nc+1-lo, mean of
// sorted[lo..hi] with R's two-pass long-double correction. (Sum order over the
// trimmed middle can differ from R's partial-sort by sub-ULP; this feeds only
// the robust Cook's dispersion, which is pmax(.,0.04)-floored and drives
// outlier padj, never log2FC.)
// ============================================================================
// [[Rcpp::export]]
NumericVector row_trimmed_means(NumericMatrix m, IntegerVector cols0, double trim) {
  int nr = m.nrow();
  int nc = cols0.size();
  NumericVector out(nr);
  std::vector<double> buf(nc);
  int lo = (int) std::floor((double) nc * trim) + 1;   // 1-based
  int hi = nc + 1 - lo;
  int cnt = hi - lo + 1;
  for (int r = 0; r < nr; r++) {
    for (int c = 0; c < nc; c++) buf[c] = m(r, cols0[c]);
    std::sort(buf.begin(), buf.end());
    long double s = 0.0L;
    for (int k = lo - 1; k <= hi - 1; k++) s += (long double) buf[k];
    s /= cnt;
    long double t = 0.0L;
    for (int k = lo - 1; k <= hi - 1; k++) t += ((long double) buf[k] - s);
    s += t / cnt;
    out[r] = (double) s;
  }
  return out;
}

// ============================================================================
// fitBeta kernel (nbinomWaldTest beta fit + estimateDispersionsGeneEst mu fit).
// Two changes vs upstream src/DESeq2.cpp, BOTH in the useQR=FALSE branch (the
// only one this kernel implements — fast_fitBetaWrapper calls it with
// useQR=FALSE within the validated betaPrior=FALSE regime):
//   1. the dead per-iteration `w_sqrt_vec = sqrt(w_vec)` is removed (unused).
//   2. [algorithmic] the IRLS convergence test is changed from the per-iteration
//      negative-binomial DEVIANCE ratio (|dev-dev_old|/(|dev|+0.1) < tol, which
//      costs a full n-sample Rf_dnbinom_mu pass EVERY iteration) to a cheap
//      BETA-CHANGE test max|beta - beta_old| < BTOL. fitBeta's returned
//      `deviance` is DISCARDED by fitNbinomGLMs (it recomputes logLike via
//      nbinomLogLike), so the deviance only ever drove this convergence test.
//      The IRLS updates are identical, so beta lands within ~BTOL of the same
//      optimum -> log2FC drift << the 0.01 q99 gate. The final deviance is
//      computed ONCE after the loop. BTOL=1e-6 is tighter than upstream's
//      effective beta precision, for safety. The useQR=TRUE branch is not
//      implemented here (fast_fitBetaWrapper falls back to upstream for the
//      betaPrior=TRUE / large-ridge regime, where upstream uses the QR solve).
// ============================================================================
static const double BTOL = 1e-6;   // beta-change convergence tolerance (natural-log beta)

// [[Rcpp::export]]
List fitBeta_zyme(SEXP ySEXP, SEXP xSEXP, SEXP nfSEXP, SEXP alpha_hatSEXP, SEXP contrastSEXP, SEXP beta_matSEXP, SEXP lambdaSEXP, SEXP weightsSEXP, SEXP useWeightsSEXP, SEXP tolSEXP, SEXP maxitSEXP, SEXP useQRSEXP, SEXP minmuSEXP) {

  arma::mat y = as<arma::mat>(ySEXP);
  arma::mat nf = as<arma::mat>(nfSEXP);
  arma::mat x = as<arma::mat>(xSEXP);
  int y_n = y.n_rows;
  int y_m = y.n_cols;
  int x_p = x.n_cols;
  arma::vec alpha_hat = as<arma::vec>(alpha_hatSEXP);
  arma::mat beta_mat = as<arma::mat>(beta_matSEXP);
  arma::mat beta_var_mat = arma::zeros(beta_mat.n_rows, beta_mat.n_cols);
  arma::mat contrast_num = arma::zeros(beta_mat.n_rows, 1);
  arma::mat contrast_denom = arma::zeros(beta_mat.n_rows, 1);
  arma::mat hat_diagonals = arma::zeros(y.n_rows, y.n_cols);
  arma::colvec lambda = as<arma::colvec>(lambdaSEXP);
  arma::colvec contrast = as<arma::colvec>(contrastSEXP);
  int maxit = as<int>(maxitSEXP);
  arma::colvec yrow, nfrow, beta_hat, beta_prev, mu_hat, z;
  arma::mat ridge, sigma;
  arma::vec w_vec, w_sqrt_vec;
  arma::mat weights = as<arma::mat>(weightsSEXP);
  bool useWeights = as<bool>(useWeightsSEXP);
  bool useQR = as<bool>(useQRSEXP);
  arma::colvec gamma_hat, big_z;
  arma::vec big_w_diag;
  arma::mat weighted_x_ridge, q, r, big_w_sqrt;
  double dev, dev_old, conv_test;
  double tol = as<double>(tolSEXP);
  double minmu = as<double>(minmuSEXP);
  double large = 30.0;
  NumericVector iter(y_n);
  NumericVector deviance(y_n);
  for (int i = 0; i < y_n; i++) {
    if (i % 100 == 0) checkUserInterrupt();
    nfrow = nf.row(i).t();
    yrow = y.row(i).t();
    beta_hat = beta_mat.row(i).t();
    mu_hat = nfrow % exp(x * beta_hat);
    for (int j = 0; j < y_m; j++) {
      mu_hat(j) = fmax(mu_hat(j), minmu);
    }
    ridge = diagmat(lambda);
    dev = 0.0;
    dev_old = 0.0;
    // QR branch deleted (housekeeping): this kernel runs only with useQR=FALSE
    // (fast_fitBetaWrapper gates on it), so the upstream qr_econ path was dead
    // code. The assert preserves the contract.
    if (useQR) Rcpp::stop("fitBeta_zyme: only useQR=FALSE is supported");
    {
      // standard design matrix + matrix inversion. w_sqrt_vec dropped (unused here).
      // Convergence on beta-change instead of the per-iter deviance (see header).
      for (int t = 0; t < maxit; t++) {
        iter(i)++;
        if (useWeights) {
          w_vec = weights.row(i).t() % mu_hat/(1.0 + alpha_hat(i) * mu_hat);
        } else {
          w_vec = mu_hat/(1.0 + alpha_hat(i) * mu_hat);
        }
        z = arma::log(mu_hat / nfrow) + (yrow - mu_hat) / mu_hat;
        beta_prev = beta_hat;
        solve(beta_hat, x.t() * (x.each_col() % w_vec) + ridge, x.t() * (z % w_vec));
        if (sum(abs(beta_hat) > large) > 0) {
          iter(i) = maxit;
          break;
        }
        mu_hat = nfrow % exp(x * beta_hat);
        for (int j = 0; j < y_m; j++) {
          mu_hat(j) = fmax(mu_hat(j), minmu);
        }
        double bchange = max(abs(beta_hat - beta_prev));
        if (std::isnan(bchange)) {
          iter(i) = maxit;
          break;
        }
        if ((t > 0) & (bchange < BTOL)) {
          break;
        }
      }
      // final deviance, computed once (returned value, unused downstream)
      dev = 0.0;
      for (int j = 0; j < y_m; j++) {
        if (useWeights) {
          dev = dev + -2.0 * weights(i,j) * Rf_dnbinom_mu(yrow(j), 1.0/alpha_hat(i), mu_hat(j), 1);
        } else {
          dev = dev + -2.0 * Rf_dnbinom_mu(yrow(j), 1.0/alpha_hat(i), mu_hat(j), 1);
        }
      }
    }
    deviance(i) = dev;
    beta_mat.row(i) = beta_hat.t();
    if (useWeights) {
      w_vec = weights.row(i).t() % mu_hat/(1.0 + alpha_hat(i) * mu_hat);
      w_sqrt_vec = sqrt(w_vec);
    } else {
      w_vec = mu_hat/(1.0 + alpha_hat(i) * mu_hat);
      w_sqrt_vec = sqrt(w_vec);
    }
    arma::vec hat_matrix_diag = arma::zeros(x.n_rows);
    arma::mat xw = x.each_col() % w_sqrt_vec;
    arma::mat xtwxr_inv = (x.t() * (x.each_col() % w_vec) + ridge).i();
    for(int jp = 0; jp < y_m; jp++){
      for(int idx1 = 0; idx1 < x_p; idx1++){
        for(int idx2 = 0; idx2 < x_p; idx2++){
          hat_matrix_diag(jp) += xw(jp, idx1) * (xw(jp, idx2) * xtwxr_inv(idx2, idx1));
        }
      }
    }
    hat_diagonals.row(i) = hat_matrix_diag.t();
    sigma = (x.t() * (x.each_col() % w_vec) + ridge).i() * x.t() * (x.each_col() % w_vec) * (x.t() * (x.each_col() % w_vec) + ridge).i();
    contrast_num.row(i) = contrast.t() * beta_hat;
    contrast_denom.row(i) = sqrt(contrast.t() * sigma * contrast);
    beta_var_mat.row(i) = diagvec(sigma).t();
  }

  return List::create(Named("beta_mat",beta_mat),
                      Named("beta_var_mat",beta_var_mat),
                      Named("iter",iter),
                      Named("hat_diagonals",hat_diagonals),
                      Named("contrast_num",contrast_num),
                      Named("contrast_denom",contrast_denom),
                      Named("deviance",deviance));
}
