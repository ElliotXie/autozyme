// C++ kernels for autozyme's infercnv patch.
//
// Lifted from test_infercnv_hmm/pipeline/run.R (11 inline Rcpp::cppFunction
// blocks). Each kernel matches the R-side fast_* wrapper that calls it.
//
// Header-only Rcpp; no RcppParallel, RcppArmadillo, or RcppEigen — all loops
// are sequential per the task's threading: not_applicable constraint (HMM
// Gibbs sampling sets the pacing floor, and the converged kernels are tuned
// for column-major cache locality at single-thread).

// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <vector>

// ----- 1. fast_smooth_window_cpp ----------------------------------------
// Triangle-filter smoother on a (nr x nc) matrix at the window length 101
// fast path. Centered output via cumsum-of-cumsum; edge rows via precomputed
// left/right weight tables. Math identical to upstream stats::filter +
// per-row apply for NA-free input.

// [[Rcpp::export]]
Rcpp::NumericMatrix fast_smooth_window_cpp(Rcpp::NumericMatrix data,
                                           int window_length) {
  const int nr = data.nrow();
  const int nc = data.ncol();
  const int tail = (window_length - 1) / 2;
  const int w = tail + 1;
  const int w2 = (w - 1) / 2;
  const int pad = 2 * w2 + 2;
  const int max_len = 2 * tail;
  const double center_den = static_cast<double>(w) * w;
  const double edge_denom_c =
    static_cast<double>(tail) * tail + window_length;

  Rcpp::NumericMatrix out(nr, nc);
  std::vector<double> cc(nr + pad);
  std::vector<int> len_vec(tail);
  std::vector<double> left_w(tail * max_len, 0.0);
  std::vector<double> right_w(tail * max_len, 0.0);

  for (int k = 1; k <= tail; ++k) {
    const int row = k - 1;
    const int d_left = k - 1;
    const int r_left = tail - d_left;
    const int len = k + tail;
    const double den = edge_denom_c -
      (static_cast<double>(r_left) * (r_left + 1)) / 2.0;
    len_vec[row] = len;

    for (int jj = 0; jj < len; ++jj) {
      int p = tail + 2 - k + jj;
      double numer;
      if (p <= tail) {
        numer = p;
      } else if (p == tail + 1) {
        numer = tail + 1;
      } else {
        numer = 2 * tail + 2 - p;
      }
      left_w[row * max_len + jj] = numer / den;

      p = tail + 2 - k + (len - 1 - jj);
      if (p <= tail) {
        numer = p;
      } else if (p == tail + 1) {
        numer = tail + 1;
      } else {
        numer = 2 * tail + 2 - p;
      }
      right_w[row * max_len + jj] = numer / den;
    }
  }

  for (int col = 0; col < nc; ++col) {
    const double* in_col = data.begin() + static_cast<R_xlen_t>(col) * nr;
    double* out_col = out.begin() + static_cast<R_xlen_t>(col) * nr;
    std::fill(cc.begin(), cc.end(), 0.0);

    double c1 = 0.0;
    double c2 = 0.0;
    for (int i = 0; i < nr; ++i) {
      c1 += in_col[i];
      c2 += c1;
      cc[pad + i] = c2;
    }

    const int center_start = 2 * w2;
    const int center_end = nr - 2 * w2 - 1;
    for (int i = center_start; i <= center_end; ++i) {
      out_col[i] = (
          cc[i + 2 * w2 + pad]
        - 2.0 * cc[i + pad - 1]
        + cc[i - 2 * w2 + pad - 2]
      ) / center_den;
    }

    for (int k = 1; k <= tail; ++k) {
      const int row = k - 1;
      const int len = len_vec[row];
      const double* lw = &left_w[row * max_len];
      const double* rw = &right_w[row * max_len];

      double left_sum = 0.0;
      for (int jj = 0; jj < len; ++jj) {
        left_sum += lw[jj] * in_col[jj];
      }
      out_col[row] = left_sum;

      const int right_start = nr - k - tail;
      double right_sum = 0.0;
      for (int jj = 0; jj < len; ++jj) {
        right_sum += rw[jj] * in_col[right_start + jj];
      }
      out_col[nr - k] = right_sum;
    }
  }

  return out;
}


// ----- 2. fast_smooth_long_chromosomes_cpp -----------------------------
// Per-chromosome smoother across many contiguous gene blocks in one pass —
// avoids the R-level expr[idx, ] <- fast_smooth_window(...) round-trip.

// [[Rcpp::export]]
Rcpp::NumericMatrix fast_smooth_long_chromosomes_cpp(Rcpp::NumericMatrix data,
                                                     Rcpp::IntegerVector starts,
                                                     Rcpp::IntegerVector lens,
                                                     int window_length) {
  const int total_nr = data.nrow();
  const int nc = data.ncol();
  const int tail = (window_length - 1) / 2;
  const int w = tail + 1;
  const int w2 = (w - 1) / 2;
  const int pad = 2 * w2 + 2;
  const int max_len = 2 * tail;
  const double center_den = static_cast<double>(w) * w;
  const double edge_denom_c =
    static_cast<double>(tail) * tail + window_length;

  Rcpp::NumericMatrix out = Rcpp::clone(data);
  std::vector<int> len_vec(tail);
  std::vector<double> left_w(tail * max_len, 0.0);
  std::vector<double> right_w(tail * max_len, 0.0);

  for (int k = 1; k <= tail; ++k) {
    const int row = k - 1;
    const int d_left = k - 1;
    const int r_left = tail - d_left;
    const int len = k + tail;
    const double den = edge_denom_c -
      (static_cast<double>(r_left) * (r_left + 1)) / 2.0;
    len_vec[row] = len;

    for (int jj = 0; jj < len; ++jj) {
      int p = tail + 2 - k + jj;
      double numer;
      if (p <= tail) {
        numer = p;
      } else if (p == tail + 1) {
        numer = tail + 1;
      } else {
        numer = 2 * tail + 2 - p;
      }
      left_w[row * max_len + jj] = numer / den;

      p = tail + 2 - k + (len - 1 - jj);
      if (p <= tail) {
        numer = p;
      } else if (p == tail + 1) {
        numer = tail + 1;
      } else {
        numer = 2 * tail + 2 - p;
      }
      right_w[row * max_len + jj] = numer / den;
    }
  }

  for (int seg = 0; seg < starts.size(); ++seg) {
    const int nr = lens[seg];
    const int start = starts[seg] - 1;

    if (nr <= window_length) {
      if (nr <= 1) continue;
      const int edge_iter = (nr + 1) / 2;
      const int short_max_len = window_length;
      std::vector<int> short_len(edge_iter);
      std::vector<int> short_right_start(edge_iter);
      std::vector<double> short_left(edge_iter * short_max_len, 0.0);
      std::vector<double> short_right(edge_iter * short_max_len, 0.0);

      for (int k = 1; k <= edge_iter; ++k) {
        const int row = k - 1;
        const int d_left = k - 1;
        int d_right = nr - k;
        if (d_right > tail) d_right = tail;
        const int r_left = tail - d_left;
        const int r_right = tail - d_right;
        const int len = k + d_right;
        const double den = edge_denom_c -
          (static_cast<double>(r_left) * (r_left + 1)) / 2.0 -
          (static_cast<double>(r_right) * (r_right + 1)) / 2.0;
        short_len[row] = len;
        short_right_start[row] = nr - k - d_right;

        for (int jj = 0; jj < len; ++jj) {
          int p = tail + 2 - k + jj;
          double numer;
          if (p <= tail) {
            numer = p;
          } else if (p == tail + 1) {
            numer = tail + 1;
          } else {
            numer = 2 * tail + 2 - p;
          }
          short_left[row * short_max_len + jj] = numer / den;

          p = tail + 2 - k + (len - 1 - jj);
          if (p <= tail) {
            numer = p;
          } else if (p == tail + 1) {
            numer = tail + 1;
          } else {
            numer = 2 * tail + 2 - p;
          }
          short_right[row * short_max_len + jj] = numer / den;
        }
      }

      for (int col = 0; col < nc; ++col) {
        const double* in_col =
          data.begin() + static_cast<R_xlen_t>(col) * total_nr + start;
        double* out_col =
          out.begin() + static_cast<R_xlen_t>(col) * total_nr + start;
        for (int k = 1; k <= edge_iter; ++k) {
          const int row = k - 1;
          const int len = short_len[row];
          const double* lw = &short_left[row * short_max_len];
          const double* rw = &short_right[row * short_max_len];

          double left_sum = 0.0;
          for (int jj = 0; jj < len; ++jj) {
            left_sum += lw[jj] * in_col[jj];
          }
          out_col[row] = left_sum;

          const int right_start = short_right_start[row];
          double right_sum = 0.0;
          for (int jj = 0; jj < len; ++jj) {
            right_sum += rw[jj] * in_col[right_start + jj];
          }
          out_col[nr - k] = right_sum;
        }
      }
      continue;
    }

    std::vector<double> cc(nr + pad);

    for (int col = 0; col < nc; ++col) {
      const double* in_col =
        data.begin() + static_cast<R_xlen_t>(col) * total_nr + start;
      double* out_col =
        out.begin() + static_cast<R_xlen_t>(col) * total_nr + start;
      std::fill(cc.begin(), cc.end(), 0.0);

      double c1 = 0.0;
      double c2 = 0.0;
      for (int i = 0; i < nr; ++i) {
        c1 += in_col[i];
        c2 += c1;
        cc[pad + i] = c2;
      }

      const int center_start = 2 * w2;
      const int center_end = nr - 2 * w2 - 1;
      for (int i = center_start; i <= center_end; ++i) {
        out_col[i] = (
            cc[i + 2 * w2 + pad]
          - 2.0 * cc[i + pad - 1]
          + cc[i - 2 * w2 + pad - 2]
        ) / center_den;
      }

      for (int k = 1; k <= tail; ++k) {
        const int row = k - 1;
        const int len = len_vec[row];
        const double* lw = &left_w[row * max_len];
        const double* rw = &right_w[row * max_len];

        double left_sum = 0.0;
        for (int jj = 0; jj < len; ++jj) {
          left_sum += lw[jj] * in_col[jj];
        }
        out_col[row] = left_sum;

        const int right_start = nr - k - tail;
        double right_sum = 0.0;
        for (int jj = 0; jj < len; ++jj) {
          right_sum += rw[jj] * in_col[right_start + jj];
        }
        out_col[nr - k] = right_sum;
      }
    }
  }

  return out;
}


// ----- 3. fast_scale_columns_cpp ----------------------------------------
// Per-column multiply (used to replace sweep(M, 2, cs, "/") in normalize).

// [[Rcpp::export]]
Rcpp::NumericMatrix fast_scale_columns_cpp(Rcpp::NumericMatrix data,
                                           Rcpp::NumericVector scales) {
  const int nr = data.nrow();
  const int nc = data.ncol();
  Rcpp::NumericMatrix out(nr, nc);
  for (int col = 0; col < nc; ++col) {
    const double scale = scales[col];
    const double* in_col = data.begin() + static_cast<R_xlen_t>(col) * nr;
    double* out_col = out.begin() + static_cast<R_xlen_t>(col) * nr;
    for (int i = 0; i < nr; ++i) {
      out_col[i] = in_col[i] * scale;
    }
  }
  return out;
}


// ----- 4. fast_log1p_scale_cpp ------------------------------------------
// log2(x + 1) via log1p(x) * (1/ln 2). Bit-identical to log2(x + 1).

// [[Rcpp::export]]
Rcpp::NumericMatrix fast_log1p_scale_cpp(Rcpp::NumericMatrix expr,
                                         double inv_ln2) {
  const int nr = expr.nrow();
  const int nc = expr.ncol();
  Rcpp::NumericMatrix out(nr, nc);
  for (int col = 0; col < nc; ++col) {
    const double* in_col = expr.begin() + static_cast<R_xlen_t>(col) * nr;
    double* out_col = out.begin() + static_cast<R_xlen_t>(col) * nr;
    for (int i = 0; i < nr; ++i) {
      out_col[i] = std::log1p(in_col[i]) * inv_ln2;
    }
  }
  return out;
}


// ----- 5. fast_invert_log2_cpp ------------------------------------------
// 2^x matrix-wide via exp(x * ln 2) — single-pass, one allocation.

// [[Rcpp::export]]
Rcpp::NumericMatrix fast_invert_log2_cpp(Rcpp::NumericMatrix expr,
                                         double ln2) {
  const int nr = expr.nrow();
  const int nc = expr.ncol();
  Rcpp::NumericMatrix out(nr, nc);
  const R_xlen_t ntot = static_cast<R_xlen_t>(nr) * nc;
  const double* in_p = expr.begin();
  double* out_p = out.begin();
  for (R_xlen_t i = 0; i < ntot; ++i) {
    out_p[i] = std::exp(in_p[i] * ln2);
  }
  return out;
}


// ----- 6. fast_center_columns_cpp ---------------------------------------
// Per-column subtract — used after matrixStats::colMedians.

// [[Rcpp::export]]
Rcpp::NumericMatrix fast_center_columns_cpp(Rcpp::NumericMatrix expr,
                                            Rcpp::NumericVector centers) {
  const int nr = expr.nrow();
  const int nc = expr.ncol();
  Rcpp::NumericMatrix out(nr, nc);
  for (int col = 0; col < nc; ++col) {
    const double center = centers[col];
    const double* in_col = expr.begin() + static_cast<R_xlen_t>(col) * nr;
    double* out_col = out.begin() + static_cast<R_xlen_t>(col) * nr;
    for (int i = 0; i < nr; ++i) {
      out_col[i] = in_col[i] - center;
    }
  }
  return out;
}


// ----- 7. fast_subtract_ref_bounds_cpp ----------------------------------
// Fused subtract-reference + optional ±threshold clamp. Eliminates the two
// R-level loops over genes in upstream's .get_normal_gene_mean_bounds +
// .subtract_expr pipeline.

// [[Rcpp::export]]
Rcpp::NumericMatrix fast_subtract_ref_bounds_cpp(Rcpp::NumericMatrix expr,
                                                 Rcpp::List ref_groups,
                                                 bool use_bounds,
                                                 double threshold,
                                                 bool do_threshold) {
  const int nr = expr.nrow();
  const int nc = expr.ncol();
  const int ng = ref_groups.size();
  if (ng == 0) Rcpp::stop("no reference groups");

  Rcpp::NumericVector grp_min(nr), grp_max(nr), mean_sum(nr), means(nr);

  for (int k = 0; k < ng; ++k) {
    Rcpp::IntegerVector idx = ref_groups[k];
    const int nidx = idx.size();
    if (nidx == 0) Rcpp::stop("empty reference group");
    std::fill(means.begin(), means.end(), 0.0);

    for (int jj = 0; jj < nidx; ++jj) {
      const int col = idx[jj] - 1;
      if (col < 0 || col >= nc) Rcpp::stop("reference index out of bounds");
      const double* colp = expr.begin() + static_cast<R_xlen_t>(col) * nr;
      for (int i = 0; i < nr; ++i) {
        means[i] += colp[i];
      }
    }

    const double inv_n = 1.0 / static_cast<double>(nidx);
    if (use_bounds) {
      if (k == 0) {
        for (int i = 0; i < nr; ++i) {
          const double mu = means[i] * inv_n;
          grp_min[i] = mu;
          grp_max[i] = mu;
        }
      } else {
        for (int i = 0; i < nr; ++i) {
          const double mu = means[i] * inv_n;
          if (mu < grp_min[i]) grp_min[i] = mu;
          if (mu > grp_max[i]) grp_max[i] = mu;
        }
      }
    } else {
      for (int i = 0; i < nr; ++i) {
        mean_sum[i] += means[i] * inv_n;
      }
    }
  }

  Rcpp::NumericMatrix out(nr, nc);
  if (use_bounds) {
    for (int col = 0; col < nc; ++col) {
      const double* in_col = expr.begin() + static_cast<R_xlen_t>(col) * nr;
      double* out_col = out.begin() + static_cast<R_xlen_t>(col) * nr;
      for (int i = 0; i < nr; ++i) {
        const double x = in_col[i];
        double bounded = x;
        if (bounded < grp_min[i]) {
          bounded = grp_min[i];
        } else if (bounded > grp_max[i]) {
          bounded = grp_max[i];
        }
        double y = x - bounded;
        if (do_threshold) {
          if (y > threshold) {
            y = threshold;
          } else if (y < -threshold) {
            y = -threshold;
          }
        }
        out_col[i] = y;
      }
    }
  } else {
    const double inv_ng = 1.0 / static_cast<double>(ng);
    for (int col = 0; col < nc; ++col) {
      const double* in_col = expr.begin() + static_cast<R_xlen_t>(col) * nr;
      double* out_col = out.begin() + static_cast<R_xlen_t>(col) * nr;
      for (int i = 0; i < nr; ++i) {
        out_col[i] = in_col[i] - mean_sum[i] * inv_ng;
      }
    }
  }
  return out;
}


// ----- 8. fast_apply_dropout_cpp ----------------------------------------
// Per-gene dropout zeroing. Same RNG sequence as the R for-loop over rows.

// [[Rcpp::export]]
Rcpp::NumericMatrix fast_apply_dropout_cpp(Rcpp::NumericMatrix counts,
                                           Rcpp::NumericVector padj) {
  const int nr = counts.nrow();
  const int nc = counts.ncol();
  for (int g = 0; g < nr; ++g) {
    const double p = padj[g];
    for (int c = 0; c < nc; ++c) {
      const double u = R::unif_rand();
      if (u <= p) counts(g, c) = 0.0;
    }
  }
  return counts;
}


// ----- 9. fast_state_consensus_cpp --------------------------------------
// Per-row mode of integer HMM states, with smallest-state tie-break to match
// upstream's first-class-wins behavior.

// [[Rcpp::export]]
Rcpp::NumericVector fast_state_consensus_cpp(Rcpp::NumericMatrix mat) {
  const int nr = mat.nrow();
  const int nc = mat.ncol();
  Rcpp::NumericVector out(nr);
  std::vector<int> states;
  std::vector<int> counts;
  states.reserve(8);
  counts.reserve(8);

  for (int i = 0; i < nr; ++i) {
    states.clear();
    counts.clear();
    for (int j = 0; j < nc; ++j) {
      const double x = mat(i, j);
      if (Rcpp::NumericVector::is_na(x)) continue;
      const int state = static_cast<int>(x);
      bool found = false;
      for (std::size_t k = 0; k < states.size(); ++k) {
        if (states[k] == state) {
          ++counts[k];
          found = true;
          break;
        }
      }
      if (!found) {
        states.push_back(state);
        counts.push_back(1);
      }
    }

    int best_state = NA_INTEGER;
    int best_count = -1;
    for (std::size_t k = 0; k < states.size(); ++k) {
      if (counts[k] > best_count ||
          (counts[k] == best_count && states[k] < best_state)) {
        best_count = counts[k];
        best_state = states[k];
      }
    }
    out[i] = best_state;
  }
  return out;
}


// ----- 10. fast_cell_prob_cpp -------------------------------------------
// Per-cell HMM state probability table — column-wise tally of integer
// states then normalize to probabilities.

// [[Rcpp::export]]
Rcpp::NumericMatrix fast_cell_prob_cpp(Rcpp::NumericMatrix epsilons,
                                       int nstates) {
  const int nsamp = epsilons.nrow();
  const int nc = epsilons.ncol();
  Rcpp::NumericMatrix out(nstates, nc);

  for (int j = 0; j < nc; ++j) {
    int observed = 0;
    for (int i = 0; i < nsamp; ++i) {
      const double x = epsilons(i, j);
      if (Rcpp::NumericVector::is_na(x)) continue;
      const int state = static_cast<int>(x);
      if (state >= 1 && state <= nstates) {
        out(state - 1, j) += 1.0;
        ++observed;
      }
    }
    if (observed > 0) {
      for (int s = 0; s < nstates; ++s) {
        out(s, j) /= static_cast<double>(observed);
      }
    } else {
      for (int s = 0; s < nstates; ++s) {
        out(s, j) = NA_REAL;
      }
    }
  }
  return out;
}


// ----- 11. fast_viterbi_adj_cpp -----------------------------------------
// Viterbi decode for the dthmm with adjusted Gaussian-tail emission used by
// infercnv::Viterbi.dthmm.adj. Single allocation of nu + y; emission table
// reused across timesteps.

// [[Rcpp::export]]
Rcpp::NumericVector fast_viterbi_adj_cpp(Rcpp::NumericVector x,
                                         Rcpp::NumericMatrix Pi,
                                         Rcpp::NumericVector delta,
                                         Rcpp::NumericVector means,
                                         Rcpp::NumericVector sds) {
  const int n = x.size();
  if (n < 2) {
    Rcpp::NumericVector neutral(1);
    neutral[0] = 3.0;
    return neutral;
  }

  const int m = Pi.nrow();
  std::vector<double> sd_vals(sds.begin(), sds.end());
  std::sort(sd_vals.begin(), sd_vals.end());
  double sd = 0.0;
  if (m % 2 == 0) {
    sd = (sd_vals[m / 2 - 1] + sd_vals[m / 2]) / 2.0;
  } else {
    sd = sd_vals[m / 2];
  }

  Rcpp::NumericMatrix logPi(m, m);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      logPi(i, j) = std::log(Pi(i, j));
    }
  }

  Rcpp::NumericMatrix nu(n, m);
  Rcpp::IntegerVector y(n);
  std::vector<double> emission(m);

  auto fill_log_emission = [&](double xi) {
    double total = 0.0;
    for (int j = 0; j < m; ++j) {
      const double q = std::fabs(xi - means[j]) / sd;
      const double log_tail = R::pnorm(q, 0.0, 1.0, 0, 1);
      emission[j] = 1.0 / (-1.0 * log_tail);
      total += emission[j];
    }
    for (int j = 0; j < m; ++j) {
      emission[j] = std::log(emission[j] / total);
    }
  };

  fill_log_emission(x[0]);
  for (int j = 0; j < m; ++j) {
    nu(0, j) = std::log(delta[j]) + emission[j];
  }

  for (int i = 1; i < n; ++i) {
    fill_log_emission(x[i]);
    for (int j = 0; j < m; ++j) {
      double best = nu(i - 1, 0) + logPi(0, j);
      for (int k = 1; k < m; ++k) {
        const double val = nu(i - 1, k) + logPi(k, j);
        if (val > best) best = val;
      }
      nu(i, j) = best + emission[j];
    }
  }

  for (int j = 0; j < m; ++j) {
    if (nu(n - 1, j) == R_NegInf) {
      Rcpp::stop("Problems With Underflow");
    }
  }

  int best_state = 0;
  double best_val = nu(n - 1, 0);
  for (int j = 1; j < m; ++j) {
    if (nu(n - 1, j) > best_val) {
      best_val = nu(n - 1, j);
      best_state = j;
    }
  }
  y[n - 1] = best_state + 1;

  for (int i = n - 2; i >= 0; --i) {
    const int next_state = y[i + 1] - 1;
    int best_k = 0;
    double best = logPi(0, next_state) + nu(i, 0);
    for (int k = 1; k < m; ++k) {
      const double val = logPi(k, next_state) + nu(i, k);
      if (val > best) {
        best = val;
        best_k = k;
      }
    }
    y[i] = best_k + 1;
  }

  return Rcpp::as<Rcpp::NumericVector>(y);
}
