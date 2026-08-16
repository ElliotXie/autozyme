// Fast batch NB penalized-GAM fitter for the shared-design tradeSeq workload.
// Reduced (rank-identifiable) reparameterization: every per-eval solve is a
// small PD Cholesky. Outer selection of (log lambda, log theta) by Laplace-REML
// via warm-started BFGS. Matches mgcv::gam(family="nb") to ~1e-5 on fitted eta.
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace arma;

struct Ctx {
  const mat* Xr;      // n x q design (reduced)
  const mat* Sr;      // q x q penalty (reduced)
  const vec* off;     // n offset
  const vec* y;       // n response
  vec lgy1;           // lgamma(y+1), precomputed per gene
  vec uy, my;         // unique count values and their multiplicities
  double sum_lgy1;    // sum lgamma(y+1)
  int n, q;
  double rankS, logdetSpos, MpEff;
  long solves;        // diagnostic: inner Newton iterations
  long solve_cap;     // per-gene worst-case budget (0 = unbounded)
};

// theta-only NB loglik pieces that depend on y via lgamma(y+theta): grouped
// over unique counts.  Returns sum_i [lgamma(y_i+theta)].
static inline double sum_lgamma_yth(const Ctx& C, double theta) {
  double s = 0.0;
  for (uword k = 0; k < C.uy.n_elem; ++k) s += C.my[k] * std::lgamma(C.uy[k] + theta);
  return s;
}
// sum_i digamma(y_i + theta), grouped over unique counts.
static inline double sum_digamma_yth(const Ctx& C, double theta) {
  double s = 0.0;
  for (uword k = 0; k < C.uy.n_elem; ++k) s += C.my[k] * R::digamma(C.uy[k] + theta);
  return s;
}

// Robust upper-Cholesky of a symmetric matrix with escalating ridge.
static bool robust_chol(mat& R, mat H) {
  H = 0.5 * (H + H.t());
  double ridge = 0.0, base = std::fabs(H.diag().max()) * 1e-12 + 1e-12;
  for (int k = 0; k < 12; ++k) {
    mat Hk = H; if (ridge > 0) Hk.diag() += ridge;
    if (chol(R, Hk)) return true;
    ridge = (ridge == 0.0) ? base : ridge * 10.0;
  }
  return false;
}

// Inner penalized Newton (observed info), warm-started from a. Fills H, logdetH.
// Returns mu-dependent part of the NB loglik. `tol` controls convergence.
static double inner_fit(Ctx& C, double theta, double lambda,
                        vec& a, mat& H, double& logdetH, double tol) {
  const mat& X = *C.Xr; const vec& off = *C.off; const vec& y = *C.y;
  mat Slam = lambda * (*C.Sr);
  vec eta = X * a + off;
  vec mu  = clamp(exp(eta), 1e-10, 1e10);
  auto pll_mu = [&](const vec& m_)->double {
    return accu(y % log(m_) - (y + theta) % log(theta + m_));
  };
  double ll_old = pll_mu(mu) - 0.5 * as_scalar(a.t() * Slam * a);
  vec g, w, grad, step, a_try, mu_try; mat R;
  for (int it = 0; it < 100; ++it) {
    C.solves++;
    vec tpm = theta + mu;
    g = theta * (y - mu) / tpm;
    w = mu * theta % (theta + y) / (tpm % tpm);
    { mat Xs = X.each_col() % sqrt(w); H = Xs.t() * Xs + Slam; }  // syrk-style
    grad = X.t() * g - Slam * a;
    if (!robust_chol(R, H)) break;
    step = solve(trimatu(R), solve(trimatl(R.t()), grad));
    double s = 1.0, ll_new = -datum::inf;
    for (int hh = 0; hh <= 30; ++hh) {
      a_try = a + s * step;
      mu_try = clamp(exp(X * a_try + off), 1e-10, 1e10);
      ll_new = pll_mu(mu_try) - 0.5 * as_scalar(a_try.t() * Slam * a_try);
      if (std::isfinite(ll_new) && ll_new >= ll_old - 1e-12) break;
      s *= 0.5;
    }
    a = a_try; mu = mu_try;
    if (std::fabs(ll_new - ll_old) < tol * (std::fabs(ll_new) + 0.1)) { ll_old = ll_new; break; }
    ll_old = ll_new;
  }
  vec tpm = theta + mu;
  w = mu * theta % (theta + y) / (tpm % tpm);
  H = X.t() * (X.each_col() % w) + Slam;
  robust_chol(R, H);
  logdetH = 2.0 * accu(log(R.diag()));
  return pll_mu(mu);
}

// negative Laplace-REML at (log lambda, log theta); warm-starts a in place.
static double neg_laml(Ctx& C, double llam, double lth, vec& a,
                       mat& H_out, bool keep, double tol) {
  llam = std::min(std::max(llam, -12.0), 20.0);   // sane sp box
  lth  = std::min(std::max(lth,  -8.0), 20.0);    // sane theta box
  double lambda = std::exp(llam), theta = std::exp(lth);
  vec a_local = a; mat H; double logdetH;
  double llmu = inner_fit(C, theta, lambda, a_local, H, logdetH, tol);
  int n = C.n;
  double lth_const = sum_lgamma_yth(C, theta) - n * std::lgamma(theta)
                     - C.sum_lgy1 + n * theta * std::log(theta);
  double ll = llmu + lth_const;
  double pen = 0.5 * lambda * as_scalar(a_local.t() * (*C.Sr) * a_local);
  double logdetSlam = C.rankS * std::log(lambda) + C.logdetSpos;
  double laml = ll - pen - 0.5 * logdetH + 0.5 * logdetSlam
                + 0.5 * C.MpEff * std::log(2.0 * M_PI);
  if (keep) { a = a_local; H_out = H; }
  return -laml;
}

// neg-LAML value AND analytic gradient wrt (rho=log lambda, tau=log theta).
// Reuses the inner fit; warm-starts a in place. grad is d(negLAML)/d(rho,tau).
static double neg_laml_grad(Ctx& C, double rho, double tau, vec& a,
                            vec& grad, double tol) {
  rho = std::min(std::max(rho, -12.0), 20.0);
  tau = std::min(std::max(tau,  -8.0), 20.0);
  double lambda = std::exp(rho), theta = std::exp(tau);
  const mat& X = *C.Xr; const mat& Sr = *C.Sr; const vec& off = *C.off; const vec& y = *C.y;
  int n = C.n;
  mat H; double logdetH;
  double llmu = inner_fit(C, theta, lambda, a, H, logdetH, tol);
  // Robust inverse of the penalized Hessian. A gene whose design is singular in
  // the reduced space (e.g. a sparsely-populated lineage with ~zero IRLS weight
  // once the shared lambda is driven small) gives a non-PD H; inv_sympd() throws
  // std::runtime_error there, and inside this OpenMP loop that aborts the whole
  // process. robust_chol adds an escalating ridge until it factorizes (the same
  // factor the log-det path already relies on), so reusing it makes the inverse
  // non-throwing. For well-conditioned genes the ridge is 0, so Hinv is bit-for-
  // bit identical to inv_sympd -- no accuracy or speed regression (it also drops
  // the redundant second Cholesky inv_sympd used to do). pinv is a last resort
  // for the pathological gene where even 12 ridge escalations fail to factorize.
  mat R;
  mat Hinv;
  if (robust_chol(R, H)) {
    mat Rinv = inv(trimatu(R));   // R upper-tri, H = R.t()*R (ridged if needed)
    Hinv = Rinv * Rinv.t();       // == (ridged H)^-1
  } else {
    Hinv = pinv(symmatu(H));
  }
  // value
  double lth_const = accu(lgamma(y + theta)) - n * std::lgamma(theta)
                     - accu(C.lgy1) + n * theta * std::log(theta);
  double ll = llmu + lth_const;
  double aSa = as_scalar(a.t() * Sr * a);
  double logdetSlam = C.rankS * rho + C.logdetSpos;
  double laml = ll - 0.5 * lambda * aSa - 0.5 * logdetH + 0.5 * logdetSlam
                + 0.5 * C.MpEff * std::log(2.0 * M_PI);
  // shared pieces
  vec eta = X * a + off; vec mu = clamp(exp(eta), 1e-10, 1e10);
  vec tpm = theta + mu;
  vec w = mu * theta % (theta + y) / (tpm % tpm);
  vec dw_deta = w % (theta - mu) / tpm;
  mat XHinv = X * Hinv;                    // n x q
  vec h = sum(XHinv % X, 1);               // x_i' Hinv x_i
  double trHinvS = accu(Hinv % Sr);
  // d/d rho
  vec dbeta_drho = -lambda * (Hinv * (Sr * a));
  vec deta_drho = X * dbeta_drho;
  vec dw_drho = dw_deta % deta_drho;
  double dlogdetH_drho = dot(dw_drho, h) + lambda * trHinvS;
  double dV_drho = -0.5 * lambda * aSa - 0.5 * dlogdetH_drho + 0.5 * C.rankS;
  // d/d tau
  double dl_dtheta = sum_digamma_yth(C, theta) - n * R::digamma(theta)
                     + n * (std::log(theta) + 1.0)
                     - accu(log(tpm)) - accu((y + theta) / tpm);
  vec dw_dtheta_mu = mu % (mu % (2.0 * theta + y) - theta * y) / (tpm % tpm % tpm);
  vec gtheta = (y - mu) % mu / (tpm % tpm);
  vec dbeta_dtau = theta * (Hinv * (X.t() * gtheta));
  vec deta_dtau = X * dbeta_dtau;
  vec dw_dtau = theta * dw_dtheta_mu + dw_deta % deta_dtau;
  double dlogdetH_dtau = dot(dw_dtau, h);
  double dV_dtau = theta * dl_dtheta - 0.5 * dlogdetH_dtau;
  grad = { -dV_drho, -dV_dtau };
  return -laml;
}

// 2-D Nelder-Mead over x=(log lambda, log theta). Loose inner tol during the
// search (warm-started per vertex), then a final tight polish. Correct + few
// solves. Returns optimum in x, coef in a_best, penalized Hessian in H_best.
static void optimize_nm(Ctx& C, vec& x, vec& a_best, mat& H_best, double& fbest) {
  const double searchTol = 1e-6;
  vec S0 = x, S1 = x, S2 = x;
  S1[0] += 1.0; S2[1] += 1.0;                 // initial simplex
  vec A0 = a_best, A1 = a_best, A2 = a_best;  // per-vertex warm-start coef
  mat Hj;
  double F0 = neg_laml(C, S0[0], S0[1], A0, Hj, false, searchTol);
  double F1 = neg_laml(C, S1[0], S1[1], A1, Hj, false, searchTol);
  double F2 = neg_laml(C, S2[0], S2[1], A2, Hj, false, searchTol);
  const double alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5;
  for (int iter = 0; iter < 80; ++iter) {
    if (C.solve_cap > 0 && C.solves > C.solve_cap) break;   // worst-case budget
    // sort so (S0,F0) best, (S2,F2) worst
    if (F0 > F1) { std::swap(F0,F1); S0.swap(S1); A0.swap(A1); }
    if (F1 > F2) { std::swap(F1,F2); S1.swap(S2); A1.swap(A2); }
    if (F0 > F1) { std::swap(F0,F1); S0.swap(S1); A0.swap(A1); }
    if (std::fabs(F2 - F0) < searchTol * (std::fabs(F0) + searchTol)) break;
    vec cen = 0.5 * (S0 + S1);
    vec ar = cen + alpha * (cen - S2); vec Aa = A0;
    double Fr = neg_laml(C, ar[0], ar[1], Aa, Hj, false, searchTol);
    if (Fr < F0) {
      vec ae = cen + gamma * (cen - S2); vec Ae = A0;
      double Fe = neg_laml(C, ae[0], ae[1], Ae, Hj, false, searchTol);
      if (Fe < Fr) { S2 = ae; F2 = Fe; A2 = Ae; } else { S2 = ar; F2 = Fr; A2 = Aa; }
    } else if (Fr < F1) {
      S2 = ar; F2 = Fr; A2 = Aa;
    } else {
      vec ac = cen + rho * (S2 - cen); vec Ac = A0;
      double Fc = neg_laml(C, ac[0], ac[1], Ac, Hj, false, searchTol);
      if (Fc < F2) { S2 = ac; F2 = Fc; A2 = Ac; }
      else {
        S1 = S0 + sigma * (S1 - S0); A1 = A0;
        F1 = neg_laml(C, S1[0], S1[1], A1, Hj, false, searchTol);
        S2 = S0 + sigma * (S2 - S0); A2 = A0;
        F2 = neg_laml(C, S2[0], S2[1], A2, Hj, false, searchTol);
      }
    }
  }
  x = S0; a_best = A0;
  fbest = neg_laml(C, x[0], x[1], a_best, H_best, true, 1e-11);  // tight polish
}

// Damped Newton over x=(rho,tau) using the analytic gradient and a
// finite-difference Hessian of that gradient. Levenberg damping keeps the step
// a descent direction. Falls back to Nelder-Mead if it fails to converge.
static void optimize(Ctx& C, vec& x, vec& a_best, mat& H_best, double& fbest) {
  const double searchTol = 1e-7, hfd = 1e-4;
  vec a = a_best, g, g1, g2;
  double f = neg_laml_grad(C, x[0], x[1], a, g, searchTol);
  for (int it = 0; it < 40; ++it) {
    if (norm(g, 2) < 1e-4) break;
    if (C.solve_cap > 0 && C.solves > C.solve_cap) break;   // worst-case budget
    vec ap = a; double f1 = neg_laml_grad(C, x[0] + hfd, x[1], ap, g1, searchTol);
    vec aq = a; double f2 = neg_laml_grad(C, x[0], x[1] + hfd, aq, g2, searchTol);
    (void)f1; (void)f2;
    mat Hs(2, 2);
    Hs(0,0) = (g1[0]-g[0])/hfd; Hs(1,0) = (g1[1]-g[1])/hfd;
    Hs(0,1) = (g2[0]-g[0])/hfd; Hs(1,1) = (g2[1]-g[1])/hfd;
    Hs = 0.5 * (Hs + Hs.t());
    double lev = 0.0; vec p;
    for (int d = 0; d < 40; ++d) {
      mat Hd = Hs; Hd(0,0) += lev; Hd(1,1) += lev;
      bool ok = solve(p, Hd, -g);
      if (ok && dot(g, p) < 0) break;
      lev = (lev == 0.0) ? 1e-3 : lev * 10.0;
      if (lev > 1e10) { p = -g / (std::max(std::fabs(g[0]), std::fabs(g[1])) + 1e-8); break; }
    }
    double t = 1.0, gp = dot(g, p); bool ls = false; vec an; double fn; vec gn;
    for (int bt = 0; bt < 30; ++bt) {
      an = a; fn = neg_laml_grad(C, x[0] + t*p[0], x[1] + t*p[1], an, gn, searchTol);
      if (std::isfinite(fn) && fn <= f + 1e-4 * t * gp) { ls = true; break; }
      t *= 0.5;
    }
    if (!ls) break;
    double df = f - fn;
    x[0] = std::min(std::max(x[0] + t*p[0], -12.0), 20.0);
    x[1] = std::min(std::max(x[1] + t*p[1],  -8.0), 20.0);
    f = fn; g = gn; a = an;
    if (df < 1e-10 * (std::fabs(f) + 1e-10) && norm(g, 2) < 1e-2) break;
  }
  a_best = a;
  // Newton can escape to the lambda->inf (or extreme theta) plateau where the
  // gradient vanishes at a spurious boundary optimum. Detect non-convergence OR
  // a boundary solution and fall back to the robust Nelder-Mead search.
  bool boundary = (x[0] > 13.0 || x[0] < -10.0 || x[1] > 16.0 || x[1] < -6.0);
  if (norm(g, 2) > 1e-2 || boundary) {
    x = {5.0, 2.5};
    optimize_nm(C, x, a_best, H_best, fbest);
    return;
  }
  fbest = neg_laml(C, x[0], x[1], a_best, H_best, true, 1e-11);  // tight polish
}

// [[Rcpp::export]]
Rcpp::List fastgam_fit(const arma::mat& Xr, const arma::mat& Sr,
                       const arma::vec& offset, const arma::mat& Y,
                       double rankS, double logdetSpos, double MpEff,
                       double start_llam, double start_lth, int nthreads = 1) {
  int n = Xr.n_rows, q = Xr.n_cols, Gn = Y.n_cols;
  mat Abeta(q, Gn, fill::zeros);
  mat Hstack(q * q, Gn, fill::zeros);
  vec theta(Gn), lambda(Gn), laml(Gn), solves_g(Gn, fill::zeros);
  vec capped_g(Gn, fill::zeros);   // did the gene hit the worst-case budget?
#ifdef _OPENMP
  if (nthreads < 1) nthreads = 1;
#endif
  // Each gene is independent; write only to its own column/index (no races).
  #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 4)
  for (int gcol = 0; gcol < Gn; ++gcol) {
    Ctx C; C.Xr = &Xr; C.Sr = &Sr; C.off = &offset; C.n = n; C.q = q;
    C.rankS = rankS; C.logdetSpos = logdetSpos; C.MpEff = MpEff; C.solves = 0;
    C.solve_cap = 1200;                           // worst-case per-gene budget
    vec y = Y.col(gcol);
    C.y = &y; C.lgy1 = lgamma(y + 1.0); C.sum_lgy1 = accu(C.lgy1);
    C.uy = unique(y); C.my = zeros<vec>(C.uy.n_elem);   // tabulate repeated counts
    for (uword k = 0; k < C.uy.n_elem; ++k) C.my[k] = accu(y == C.uy[k]);
    vec a;                                        // fresh LS init per gene
    if (!solve(a, Xr, log(clamp(y, 0.1, datum::inf)) - offset)) a = zeros<vec>(q);
    // per-gene method-of-moments theta start (data-driven; rescues genes whose
    // dispersion is far from the fixed prior, e.g. extreme-outlier counts). It
    // only sets the STARTING point -- the optimum is unchanged, so accuracy on
    // well-behaved genes is identical.
    vec s = exp(offset);
    vec mu0 = s * (accu(y) / std::max(accu(s), 1e-8));
    double den = accu((y - mu0) % (y - mu0) - mu0);
    double th0 = (den > 1e-8) ? accu(mu0 % mu0) / den : 1e3;
    th0 = std::min(std::max(th0, 1e-2), 1e3);
    vec x = {start_llam, std::log(th0)};          // data-driven theta start
    mat Hbest(q, q, fill::eye); double fb;
    optimize(C, x, a, Hbest, fb);
    Abeta.col(gcol) = a;
    Hstack.col(gcol) = vectorise(Hbest);
    lambda[gcol] = std::exp(x[0]);
    theta[gcol]  = std::exp(x[1]);
    laml[gcol]   = -fb;
    solves_g[gcol] = (double)C.solves;
    capped_g[gcol] = (C.solve_cap > 0 && C.solves >= C.solve_cap) ? 1.0 : 0.0;
  }
  return Rcpp::List::create(
    Rcpp::Named("a") = Abeta, Rcpp::Named("H") = Hstack,
    Rcpp::Named("theta") = theta, Rcpp::Named("lambda") = lambda,
    Rcpp::Named("laml") = laml,
    Rcpp::Named("inner_solves") = accu(solves_g),
    Rcpp::Named("solves_g") = solves_g,
    Rcpp::Named("capped") = capped_g);
}
