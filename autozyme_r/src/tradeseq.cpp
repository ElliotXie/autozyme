// tradeseq.cpp — Rcpp kernels for the NB family used by mgcv::gam inside
// tradeSeq::fitGAM. Lifted verbatim from the optimized task pipeline
// (optimized_task/.../test_tradeseq_fitgam/pipeline/run.R lines 24-229),
// where these were `Rcpp::sourceCpp(code = '...')`-compiled at run time.
// Promoting them to compiled package code skips the JIT step and makes them
// available on platforms (Windows) where the upstream task gated the fast
// path behind `.Platform$OS.type != "windows"`.
//
// All five kernels are scalar element-wise math + R math functions
// (lgammafn / digamma / trigamma). NO BLAS / LAPACK calls — that is
// intentional and load-bearing: per src/Makevars head comment, any BLAS
// symbol in autozyme.so triggers Accelerate routing on macOS, which is
// not fork-safe and segfaults mclapply workers at ~0x110. tradeSeq fits
// run under mclapply on Mac/Linux and PSOCK on Windows; keeping these
// kernels BLAS-free preserves fork-safety.
//
// No `#pragma omp parallel for` is used inside the per-element loops.
// Inner loop n is typically 200-30000 with 2-5 transcendentals; OMP
// region overhead competes badly with the outer gene-level parallelism
// (mclapply/PSOCK), and stacking would oversubscribe.

#include <Rcpp.h>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// [[Rcpp::export]]
List nb_Dd_cpp(NumericVector y, NumericVector mu, double theta_log,
               NumericVector wt, int level) {
  double theta = std::exp(theta_log);
  int n = y.size();
  NumericVector Dmu(n), Dmu2(n), EDmu2(n);
  for (int i = 0; i < n; ++i) {
    double yi = y[i], mui = mu[i], wti = wt[i];
    double yth = yi + theta;
    double muth = mui + theta;
    Dmu[i]   = 2.0 * wti * (yth/muth - yi/mui);
    Dmu2[i]  = -2.0 * wti * (yth/(muth*muth) - yi/(mui*mui));
    EDmu2[i] = 2.0 * wti * (1.0/mui - 1.0/muth);
  }
  List r = List::create(
    Named("Dmu") = Dmu,
    Named("Dmu2") = Dmu2,
    Named("EDmu2") = EDmu2
  );
  if (level > 0) {
    NumericVector Dth(n), Dmuth(n), Dmu3(n), Dmu2th(n), EDmu2th(n);
    for (int i = 0; i < n; ++i) {
      double yi = y[i], mui = mu[i], wti = wt[i];
      double yth = yi + theta;
      double muth = mui + theta;
      double r_ym = yth/muth;
      Dth[i]     = -2.0 * wti * theta * (std::log(r_ym) + (1.0 - r_ym));
      Dmuth[i]   = 2.0 * wti * theta * (1.0 - r_ym)/muth;
      // R `x^3` does not collapse to `x*x*x` in R_pow (only x^2 does);
      // it calls libm pow(x, 3.0) which on Win MinGW uses exp(3*log(x))
      // and gives last-bit-different results vs explicit multiplication.
      // Use R_pow to mirror mgcv's `muth^3` / `mu^3` arithmetic exactly,
      // so the Rcpp NB family stays bit-exact against mgcv on every libm.
      Dmu3[i]    = 4.0 * wti * (yth/::R_pow(muth, 3.0) - yi/::R_pow(mui, 3.0));
      Dmu2th[i]  = 2.0 * wti * theta * (2.0*r_ym - 1.0)/(muth*muth);
      EDmu2th[i] = 2.0 * wti / (muth*muth);
    }
    r["Dth"] = Dth; r["Dmuth"] = Dmuth; r["Dmu3"] = Dmu3;
    r["Dmu2th"] = Dmu2th; r["EDmu2th"] = EDmu2th;
  }
  if (level > 1) {
    NumericVector Dmu4(n), Dth2(n), Dmuth2(n), Dmu2th2(n), Dmu3th(n);
    for (int i = 0; i < n; ++i) {
      double yi = y[i], mui = mu[i], wti = wt[i];
      double yth = yi + theta;
      double muth = mui + theta;
      double muth2 = muth*muth;
      double r_ym = yth/muth;
      // R `x^4` and `x^3` both go through libm pow() — mirror with R_pow.
      Dmu4[i]    = 2.0 * wti * (6.0*yi/::R_pow(mui, 4.0) - 6.0*yth/::R_pow(muth, 4.0));
      Dth2[i]    = -2.0 * wti * theta * (std::log(r_ym) + theta*yth/muth2 - r_ym - 2.0*theta/muth + 1.0 + theta/yth);
      Dmuth2[i]  = 2.0 * wti * theta * (2.0*theta*yth/muth2 - r_ym - 2.0*theta/muth + 1.0)/muth;
      Dmu2th2[i] = 2.0 * wti * theta * (-6.0*yth*theta/muth2 + 2.0*r_ym + 4.0*theta/muth - 1.0)/muth2;
      // mgcv `(1 - 3 * yth/muth)` evaluates left-assoc as `(3*yth)/muth`
      // (multiply first, then divide). Using pre-computed r_ym=yth/muth
      // and then `3.0*r_ym` reorders to `3*(yth/muth)`, which differs at
      // last bit. Rewrite without r_ym to keep mgcv's eval order.
      Dmu3th[i]  = 4.0 * wti * theta * (1.0 - 3.0*yth/muth)/::R_pow(muth, 3.0);
    }
    r["Dmu4"] = Dmu4; r["Dth2"] = Dth2; r["Dmuth2"] = Dmuth2;
    r["Dmu2th2"] = Dmu2th2; r["Dmu3th"] = Dmu3th;
  }
  return r;
}

// [[Rcpp::export]]
List nb_dDeta_log_cpp(NumericVector y, NumericVector mu, NumericVector wt,
                     double theta_log, int deriv) {
  double theta = std::exp(theta_log);
  double eps = std::numeric_limits<double>::epsilon();
  int n = y.size();
  NumericVector Deta(n), Deta2(n), EDeta2(n), Deta_Deta2(n), Deta_EDeta2(n);
  NumericVector Dth, Detath, Deta3, Deta2th, EDeta2th;
  NumericVector Deta4, Dth2, Detath2, Deta2th2, Deta3th;
  if (deriv > 0) { Dth=NumericVector(n); Detath=NumericVector(n); Deta3=NumericVector(n); Deta2th=NumericVector(n); EDeta2th=NumericVector(n); }
  if (deriv > 1) { Deta4=NumericVector(n); Dth2=NumericVector(n); Detath2=NumericVector(n); Deta2th2=NumericVector(n); Deta3th=NumericVector(n); }
  LogicalVector good(n, true);

  for (int i = 0; i < n; ++i) {
    double yi=y[i], mui=mu[i], wti=wt[i];
    double yth = yi + theta;
    double muth = mui + theta;
    double Dmu_i   = 2.0 * wti * (yth/muth - yi/mui);
    double Dmu2_i  = -2.0 * wti * (yth/(muth*muth) - yi/(mui*mui));
    double EDmu2_i = 2.0 * wti * (1.0/mui - 1.0/muth);
    double ig1 = (mui > eps) ? mui : eps;
    double ig12 = ig1 * ig1;
    Deta[i]  = Dmu_i * ig1;
    Deta2[i] = Dmu2_i * ig12 + Dmu_i * ig1;
    EDeta2[i] = EDmu2_i * ig12;
    Deta_Deta2[i]  = Dmu_i / (Dmu2_i * ig1 + Dmu_i);
    Deta_EDeta2[i] = Dmu_i / (EDmu2_i * ig1);
    bool fin_basic = std::isfinite(Deta[i]) && std::isfinite(Deta2[i]);
    good[i] = fin_basic;

    double r_ym = 0.0;
    double Dmu3_i = 0.0, Dmuth_i = 0.0, Dmu2th_i = 0.0;
    double ig13 = 0.0;
    if (deriv > 0) {
      r_ym = yth/muth;
      double Dth_i     = -2.0 * wti * theta * (std::log(r_ym) + (1.0 - r_ym));
      Dmuth_i          = 2.0 * wti * theta * (1.0 - r_ym)/muth;
      Dmu3_i           = 4.0 * wti * (yth/::R_pow(muth, 3.0) - yi/::R_pow(mui, 3.0));
      Dmu2th_i         = 2.0 * wti * theta * (2.0*r_ym - 1.0)/(muth*muth);
      double EDmu2th_i = 2.0 * wti / (muth*muth);
      ig13 = ig12 * ig1;
      Dth[i]     = Dth_i;
      Detath[i]  = Dmuth_i * ig1;
      Deta3[i]   = Dmu3_i*ig13 + 3.0*Dmu2_i*ig12 + Dmu_i*ig1;
      Deta2th[i] = Dmu2th_i*ig12 + Dmuth_i*ig1;
      EDeta2th[i]= EDmu2th_i*ig12;
      good[i] = good[i] && std::isfinite(Deta3[i]) && std::isfinite(Dth_i) && std::isfinite(Detath[i]) && std::isfinite(Deta2th[i]);
    }
    if (deriv > 1) {
      double muth2 = muth*muth;
      double Dmu4_i    = 2.0 * wti * (6.0*yi/::R_pow(mui, 4.0) - 6.0*yth/::R_pow(muth, 4.0));
      double Dth2_i    = -2.0 * wti * theta * (std::log(r_ym) + theta*yth/muth2 - r_ym - 2.0*theta/muth + 1.0 + theta/yth);
      double Dmuth2_i  = 2.0 * wti * theta * (2.0*theta*yth/muth2 - r_ym - 2.0*theta/muth + 1.0)/muth;
      double Dmu2th2_i = 2.0 * wti * theta * (-6.0*yth*theta/muth2 + 2.0*r_ym + 4.0*theta/muth - 1.0)/muth2;
      double Dmu3th_i  = 4.0 * wti * theta * (1.0 - 3.0*r_ym)/::R_pow(muth, 3.0);
      double ig12sq = ig12*ig12;
      Deta4[i]    = ig12sq*Dmu4_i + 6.0*Dmu3_i*ig13 + 7.0*Dmu2_i*ig12 + Dmu_i*ig1;
      Dth2[i]     = Dth2_i;
      Detath2[i]  = Dmuth2_i * ig1;
      Deta2th2[i] = ig12*Dmu2th2_i + Dmuth2_i*ig1;
      Deta3th[i]  = ig13*Dmu3th_i + 3.0*Dmu2th_i*ig12 + Dmuth_i*ig1;
      good[i] = good[i] && std::isfinite(Deta4[i]) && std::isfinite(Dth2_i) && std::isfinite(Detath2[i]) && std::isfinite(Deta2th2[i]) && std::isfinite(Deta3th[i]);
    }
  }

  List d = List::create(
    Named("Deta") = Deta,
    Named("Dth") = (deriv > 0 ? (SEXP)Dth : Rcpp::wrap(0.0)),
    Named("Dth2") = (deriv > 1 ? (SEXP)Dth2 : Rcpp::wrap(0.0)),
    Named("Deta2") = Deta2,
    Named("EDeta2") = EDeta2,
    Named("Detath") = (deriv > 0 ? (SEXP)Detath : Rcpp::wrap(0.0)),
    Named("Deta3") = (deriv > 0 ? (SEXP)Deta3 : Rcpp::wrap(0.0)),
    Named("Deta2th") = (deriv > 0 ? (SEXP)Deta2th : Rcpp::wrap(0.0)),
    Named("Detath2") = (deriv > 1 ? (SEXP)Detath2 : Rcpp::wrap(0.0)),
    Named("Deta4") = (deriv > 1 ? (SEXP)Deta4 : Rcpp::wrap(0.0)),
    Named("Deta3th") = (deriv > 1 ? (SEXP)Deta3th : Rcpp::wrap(0.0)),
    Named("Deta2th2") = (deriv > 1 ? (SEXP)Deta2th2 : Rcpp::wrap(0.0))
  );
  d["EDeta2th"] = (deriv > 0 ? (SEXP)EDeta2th : Rcpp::wrap(0.0));
  d["Deta.Deta2"]  = Deta_Deta2;
  d["Deta.EDeta2"] = Deta_EDeta2;
  d["good"] = good;
  return d;
}

// [[Rcpp::export]]
NumericVector linkinv_log_cpp(NumericVector eta) {
  double eps = std::numeric_limits<double>::epsilon();
  int n = eta.size();
  NumericVector mu(n);
  for (int i = 0; i < n; ++i) {
    double v = std::exp(eta[i]);
    mu[i] = (v > eps) ? v : eps;
  }
  return mu;
}

// [[Rcpp::export]]
NumericVector nb_dev_resids_cpp(NumericVector y, NumericVector mu,
                                NumericVector wt, double theta_log) {
  double theta = std::exp(theta_log);
  int n = y.size();
  NumericVector out(n);
  for (int i = 0; i < n; ++i) {
    double yi = y[i], mui = mu[i], wti = wt[i];
    if (mui <= 0.0) { out[i] = R_NaReal; continue; }
    double y_or_1 = (yi > 1.0) ? yi : 1.0;
    double yth = yi + theta;
    double muth = mui + theta;
    out[i] = 2.0 * wti * (yi * std::log(y_or_1 / mui) - yth * std::log(yth / muth));
  }
  return out;
}

// [[Rcpp::export]]
List nb_ls_cpp(NumericVector y, NumericVector w, double theta_log, double scale) {
  double Theta = std::exp(theta_log);
  int n = y.size();
  double LTheta = std::log(Theta);
  double lgamma_Theta = R::lgammafn(Theta);
  double psi0_th = R::digamma(Theta);
  double psi1_th = R::trigamma(Theta);
  // Use long double accumulators to match R's sum() exactly. R's do_sum
  // (src/main/summary.c) accumulates into LDOUBLE for precision; a double
  // accumulator drifts by ~1e-10 over n~3500 terms, enough to push
  // shared-id-penalty IRLS (OOD tier) to a different fixed point.
  long double ls_sum   = 0.0L;
  long double lsth_sum = 0.0L;
  long double lsth2_sum = 0.0L;
  NumericVector LSTH(n);
  for (int i = 0; i < n; ++i) {
    double yi = y[i], wi = w[i];
    double yth = yi + Theta;
    double lyth = std::log(yth);
    double ylogy_i = (yi > 0.0) ? yi * std::log(yi) : yi;
    double term_ls = yth*lyth - ylogy_i + R::lgammafn(yi+1.0)
                     - Theta*LTheta + lgamma_Theta - R::lgammafn(Theta + yi);
    ls_sum += (long double)(term_ls * wi);
    double psi0_yth = R::digamma(yth);
    double term_lsth = Theta * (lyth - psi0_yth + psi0_th - theta_log);
    LSTH[i] = -term_lsth * wi;
    lsth_sum += (long double)(LSTH[i]);
    double psi1_yth = R::trigamma(yth);
    double term_lsth2 = Theta * (lyth - Theta*psi1_yth - psi0_yth + Theta/yth
                                  + Theta*psi1_th + psi0_th - theta_log - 1.0);
    lsth2_sum += (long double)(-term_lsth2 * wi);
  }
  return List::create(
    Named("ls") = (double)(-ls_sum),
    Named("lsth1") = (double)(lsth_sum),
    Named("LSTH1") = NumericMatrix(n, 1, LSTH.begin()),
    Named("lsth2") = (double)(lsth2_sum)
  );
}
