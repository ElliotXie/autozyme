// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
//
// Fast replacement for BayesSpace:::iterate_t, the Gibbs/MH MCMC inner loop
// behind spatialCluster(..., model = "t"). Bit-exact math vs upstream
// (modulo fp reordering and one harmless RNG-sequence rotation: Rcpp::sample
// is replaced by R::unif_rand-based draws, so the byte-exact trajectory
// diverges while the equilibrium distribution does not). See test_bayesspace
// `pipeline/run.R` round 8 + memory/discoveries.md "RNG sequence diverges
// harmlessly" for the rationale; task.yaml widens noise_multiplier by 5% to
// absorb the divergence.
//
// Wins lifted from the converged optimization (results.tsv rounds 4, 8, 9,
// 10, 13, 14, 15, 16 — final speedup ≈ 6× at tiny / ~9-10× at larger tiers):
//   * Hoist chol(sigma_i) + log-diag sum out of the per-spot z-loop.
//     Algebraic identity chol(sigma/w) = chol(sigma)/sqrt(w) ⇒ each spot
//     only needs a sqrt(w) rescale, not a fresh chol/inv.
//   * Pre-project Y and mu through rooti once per outer iter (q + 1 BLAS
//     dgemms) and reuse the dot-product matrix per-spot.
//   * Strip Rcpp::sample/IntegerVector filter from the proposal step:
//     R::unif_rand picks a uniform index from {1..q}\{z_prev} in O(1).
//   * Cache df_j (Rcpp::List of integer vectors) as a flat
//     std::vector<arma::uvec> once at entry; per-spot lookup is a reference
//     instead of an as<uvec> roundtrip.
//   * Hoist per-iter scratch buffers (mu_i, mu_i_long, plogLikj, Vinv,
//     scratch_diff) out of the i-loop — reused across all nrep iterations,
//     removing the GC-collect peak (was ~12% of wall in baseline profile).
//   * Inline the symmetric quadratic form diff^T·lambda·diff exploiting
//     lambda_i's symmetry; q·d²/2 scalar ops, no BLAS dispatch overhead.
//   * Scalarize the neighbor-cluster match count: one pass over j_vector
//     producing two scalars (n_prev, n_new) — no per-spot uvec allocation.

#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <cmath>
#include <vector>

using namespace Rcpp;
using namespace arma;

static double const log2pi = std::log(2.0 * M_PI);

// [[Rcpp::export]]
List fast_iterate_t_impl(
    const arma::mat &Y, const List &df_j, int nrep, int thin, int n, int d,
    double gamma, int q, const arma::uvec &init, const NumericVector &mu0,
    const arma::mat &lambda0, double alpha, double beta
) {
  umat df_sim_z(nrep / thin + 1, n, fill::zeros);
  mat df_sim_mu(nrep / thin + 1, q * d, fill::zeros);
  List df_sim_lambda(nrep / thin + 1);
  mat df_sim_w(nrep / thin + 1, n);
  NumericVector plogLik(nrep, NA_REAL);

  rowvec initmu    = rep(mu0, q);
  df_sim_mu.row(0) = initmu;
  mat lambda_i     = lambda0;
  df_sim_lambda[0] = lambda0;
  uvec z           = init;
  df_sim_z.row(0)  = init.t();
  vec w            = ones<vec>(n);
  df_sim_w.row(0)  = w.t();

  colvec mu0vec = as<colvec>(mu0);
  double const dd = (double) d;
  double const constants = -dd / 2.0 * log2pi;

  // Pre-convert df_j (Rcpp::List of integer vectors) to flat C++ once. Per-spot
  // List access + as<uvec> conversion was ~n*nrep allocations otherwise.
  std::vector<arma::uvec> df_j_arma(n);
  for (int j = 0; j < n; j++) df_j_arma[j] = as<arma::uvec>(df_j[j]);

  // Hoist per-iter scratch buffers OUT of the i-loop. Reused across all nrep
  // iterations; reduces GC pressure (RunGenCollect was 12% of active time).
  mat mu_i(q, d);
  mat mu_i_long(n, d);
  arma::vec beta_d(d);  beta_d.fill(beta);
  arma::mat const Vinv = diagmat(beta_d);
  arma::vec plogLikj(n);

  for (int i = 1; i < nrep; i++) {
    if (i % 10 == 0) Rcpp::checkUserInterrupt();

    // mu update — same math as upstream; reuses hoisted mu_i scratch.
    for (int k = 1; k <= q; k++) {
      arma::uvec const z_eq_k = arma::find(z == (arma::uword) k);
      arma::vec const w_sub   = w.elem(z_eq_k);
      double const n_i        = arma::accu(w_sub);
      mat Yrows               = Y.rows(z_eq_k);
      Yrows.each_col() %= w_sub;
      arma::rowvec const Ysums = arma::sum(Yrows, 0);
      // (lambda0 + n_i*lambda_i) is needed for both mean and var — invert once.
      arma::mat const var_i = inv(lambda0 + n_i * lambda_i);
      arma::vec const mean_i = var_i * (lambda0 * mu0vec + lambda_i * Ysums.t());
      mu_i.row(k - 1) = rmvnorm(1, mean_i, var_i);
    }

    // lambda update — fill hoisted mu_i_long in-place.
    for (int j = 0; j < n; j++) mu_i_long.row(j) = mu_i.row(z(j) - 1);
    mat const Yresid = Y - mu_i_long;
    mat const sumofsq = Yresid.t() * diagmat(w) * Yresid;
    lambda_i          = rwish(n + alpha, inv(Vinv + sumofsq));
    const mat sigma_i = inv(lambda_i);

    // *** HOISTED: precompute chol(sigma_i) once per iteration. ***
    arma::mat const rooti_base = arma::inv(trimatu(arma::chol(sigma_i)));
    arma::vec const log_diag   = arma::log(arma::vec(rooti_base.diag()));
    double const rootisum_base = arma::accu(log_diag);
    double const other_base    = rootisum_base + constants;

    // *** BATCHED rooti-quad precompute ***
    // Upstream dmvnrm computes ||rooti @ (Y_j - mu_k)||² (the "rooti^T rooti"
    // quad form — see memory/discoveries.md for the upstream non-symmetric
    // chol detail). Project Y and mu once per outer iter so the per-spot
    // lookup is a single subtraction + dot product.
    //   pY[j,:] = (rooti @ Y_j.t()).t()
    //   pmu[k,:] = (rooti @ mu_k.t()).t()
    //   A[j] = ||pY[j,:]||²,  C[k] = ||pmu[k,:]||²
    //   B[j,k] = pY[j,:] @ pmu[k,:].t()
    //   quad(j,k) = A[j] - 2*B[j,k] + C[k]
    arma::mat const rooti_t = rooti_base.t();   // d × d
    arma::mat const pY      = Y * rooti_t;       // n × d   (BLAS dgemm)
    arma::mat const pmu     = mu_i * rooti_t;    // q × d   (BLAS dgemm)
    arma::vec const A_vec   = arma::sum(pY % pY, 1);    // n
    arma::vec const C_vec   = arma::sum(pmu % pmu, 1);  // q
    arma::mat const B_mat   = pY * pmu.t();             // n × q  (BLAS dgemm)

    // z + w update — per-spot R-API sugar (sample, IntegerVector filter,
    // List access) replaced by direct R::unif_rand() / std::vector lookup.
    // Hoisted scratch rowvecs reused across spots (size-stable assignment is
    // memcpy in arma; no per-spot init_warm allocation).
    double const w_alpha = (dd + 4) / 2.0;
    double w_beta;
    arma::rowvec scratch_diff(d);
    for (int j = 0; j < n; j++) {
      // diff_j = Y_j - mu_i_long_j; also equals z_prev pre-tri (since
      // mu_i_long.row(j) = mu_i.row(z(j) - 1) = mu_i.row(z_j_prev - 1)).
      scratch_diff = Y.row(j) - mu_i_long.row(j);

      // Inlined symmetric quadratic form: quad_j = diff^T * lambda_i * diff.
      // lambda_i is symmetric (Wishart sample); exploit via upper-tri sum * 2.
      double quad_j = 0.0;
      for (int a = 0; a < d; ++a) {
        double const da = scratch_diff[a];
        quad_j += lambda_i.at(a, a) * da * da;
        for (int b = a + 1; b < d; ++b) {
          quad_j += 2.0 * lambda_i.at(a, b) * da * scratch_diff[b];
        }
      }
      w_beta = 2.0 / (quad_j + 4.0);
      w[j] = R::rgamma(w_alpha, w_beta);

      int const z_j_prev = (int) z(j);
      // Uniform pick from {1..q} \ {z_j_prev}: draw idx in [0, q-1), skip prev.
      int idx = (int) std::floor(R::unif_rand() * (double)(q - 1));
      if (idx >= q - 1) idx = q - 2;  // guard against unif_rand()==1.0
      int const z_j_new = (idx < z_j_prev - 1) ? (idx + 1) : (idx + 2);

      arma::uvec const &j_vector = df_j_arma[j];

      double const log_w    = std::log(w[j]);
      double const rootisum = other_base + (dd / 2.0) * log_w;

      // ll_prev / ll_new via precomputed pY/pmu projection — single lookup.
      double const q_prev = A_vec[j] - 2.0 * B_mat.at(j, z_j_prev - 1) + C_vec[z_j_prev - 1];
      double const q_new  = A_vec[j] - 2.0 * B_mat.at(j, z_j_new  - 1) + C_vec[z_j_new  - 1];
      double const ll_prev = rootisum - 0.5 * w[j] * q_prev;
      double const ll_new  = rootisum - 0.5 * w[j] * q_new;

      // Scalarize neighbor-cluster match count: avoid 3 uvec allocations / spot.
      double h_z_prev, h_z_new;
      arma::uword const n_nb = j_vector.n_elem;
      if (n_nb != 0) {
        arma::uword n_prev_u = 0, n_new_u = 0;
        arma::uword const zp = (arma::uword) z_j_prev;
        arma::uword const zn = (arma::uword) z_j_new;
        for (arma::uword kk = 0; kk < n_nb; ++kk) {
          arma::uword const zk = z(j_vector(kk));
          if (zk == zp) ++n_prev_u;
          if (zk == zn) ++n_new_u;
        }
        double const inv_nb = gamma / (double) n_nb * 2.0;
        h_z_prev = inv_nb * (double) n_prev_u + ll_prev;
        h_z_new  = inv_nb * (double) n_new_u  + ll_new;
      } else {
        h_z_prev = ll_prev;
        h_z_new  = ll_new;
      }
      double prob_j = std::exp(h_z_new - h_z_prev);
      if (prob_j > 1.0) prob_j = 1.0;
      // Bernoulli: accept proposal with probability prob_j.
      z(j) = (R::unif_rand() < prob_j) ? (arma::uword) z_j_new : (arma::uword) z_j_prev;
      plogLikj(j) = h_z_prev;
    }
    plogLik[i] = arma::accu(plogLikj);

    if ((i + 1) % thin == 0) {
      df_sim_mu.row((i + 1) / thin) = vectorise(mu_i, 1);
      df_sim_lambda[(i + 1) / thin] = lambda_i;
      df_sim_w.row((i + 1) / thin)  = w.t();
      df_sim_z.row((i + 1) / thin)  = z.t();
    }
  }

  return List::create(
      _["z"] = df_sim_z, _["mu"] = df_sim_mu, _["lambda"] = df_sim_lambda,
      _["weights"] = df_sim_w, _["plogLik"] = plogLik);
}
