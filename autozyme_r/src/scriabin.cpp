// [[Rcpp::depends(Rcpp)]]
//
// Triplet builder for the sparse outer-product accumulator that scriabin's
// GenerateCCIM hot loop computes via `pbsapply(seq_len(nrow(a)),
// function(i) tcrossprod(a[i, ], b[i, ]))`. The upstream form materializes
// a dense (n_senders * n_receivers) x n_pairs matrix and squeezes to sparse
// at the end. Two orders of magnitude of allocation churn vanish if we
// emit (i, j, x) triplets in one pass, skipping every column whose ligand
// or receptor expression vector is fully zero.

#include <Rcpp.h>
#include <vector>

// [[Rcpp::export]]
Rcpp::List fast_lr_outer_triplets_cpp(Rcpp::NumericMatrix a, Rcpp::NumericMatrix b) {
  const int n_pairs = a.nrow();
  const int n_senders = a.ncol();
  const int n_receivers = b.ncol();

  std::vector< std::vector<int> > a_idx(n_pairs), b_idx(n_pairs);
  std::vector< std::vector<double> > a_vals(n_pairs), b_vals(n_pairs);
  std::vector<int> pair_nnz(n_pairs);
  R_xlen_t total_nnz = 0;

  for (int pair = 0; pair < n_pairs; ++pair) {
    for (int sender = 0; sender < n_senders; ++sender) {
      const double val = a(pair, sender);
      if (val != 0.0) {
        a_idx[pair].push_back(sender);
        a_vals[pair].push_back(val);
      }
    }
    for (int receiver = 0; receiver < n_receivers; ++receiver) {
      const double val = b(pair, receiver);
      if (val != 0.0) {
        b_idx[pair].push_back(receiver);
        b_vals[pair].push_back(val);
      }
    }
    pair_nnz[pair] = static_cast<int>(a_idx[pair].size() * b_idx[pair].size());
    total_nnz += pair_nnz[pair];
  }

  Rcpp::IntegerVector i(total_nnz);
  Rcpp::IntegerVector j(total_nnz);
  Rcpp::NumericVector x(total_nnz);
  R_xlen_t offset = 0;
  for (int pair = 0; pair < n_pairs; ++pair) {
    for (std::size_t receiver_pos = 0; receiver_pos < b_idx[pair].size(); ++receiver_pos) {
      const int receiver = b_idx[pair][receiver_pos];
      const double b_val = b_vals[pair][receiver_pos];
      const int receiver_offset = receiver * n_senders;
      for (std::size_t sender_pos = 0; sender_pos < a_idx[pair].size(); ++sender_pos) {
        i[offset] = a_idx[pair][sender_pos] + receiver_offset + 1;
        j[offset] = pair + 1;
        x[offset] = a_vals[pair][sender_pos] * b_val;
        ++offset;
      }
    }
  }

  return Rcpp::List::create(
    Rcpp::_["i"] = i,
    Rcpp::_["j"] = j,
    Rcpp::_["x"] = x,
    Rcpp::_["dims"] = Rcpp::IntegerVector::create(n_senders * n_receivers, n_pairs)
  );
}
