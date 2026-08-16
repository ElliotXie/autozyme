#include <Rcpp.h>

#include <algorithm>
#include <vector>

using namespace Rcpp;

static double capped_rank(double value,
                          const std::vector<double>& uval,
                          const std::vector<int>& ucnt,
                          const std::vector<int>& usuf,
                          int total_nnz, int nrow, int maxRank) {
  std::vector<double>::const_iterator it =
    std::lower_bound(uval.begin(), uval.end(), value);
  int j = static_cast<int>(it - uval.begin());
  double n_equal = 0.0;
  double n_greater = 0.0;

  if (it != uval.end() && *it == value) {
    n_equal = static_cast<double>(ucnt[j]);
    n_greater = static_cast<double>(usuf[j + 1]);
  } else {
    n_greater = static_cast<double>(usuf[j]);
  }

  double n_absent = static_cast<double>(nrow - total_nnz);
  if (0.0 > value) {
    n_greater += n_absent;
  } else if (0.0 == value) {
    n_equal += n_absent;
  }

  double rank = n_greater + (n_equal + 1.0) / 2.0;
  return rank >= maxRank ? static_cast<double>(maxRank) : rank;
}

static double score_one_signature(const std::vector<int>& idx,
                                  const std::vector<double>& gene_val,
                                  std::vector<double>& uval,
                                  std::vector<int>& ucnt,
                                  std::vector<int>& usuf,
                                  bool& dist_ready,
                                  const double* values_begin,
                                  int nvals,
                                  int nrow, int maxRank) {
  int len_sig = static_cast<int>(idx.size());
  if (len_sig <= 0) return 0.0;

  double rank_sum = 0.0;
  for (int k = 0; k < len_sig; ++k) {
    int gene = idx[k];
    if (gene <= 0) {
      rank_sum += maxRank;
      continue;
    }

    int row = gene - 1;
    double value = gene_val[row];
    if (value == 0.0 && nrow >= 2 * maxRank) {
      rank_sum += maxRank;
      continue;
    }
    if (!dist_ready) {
      uval.clear();
      ucnt.clear();
      for (int z = 0; z < nvals; ++z) {
        double v = values_begin[z];
        int d = static_cast<int>(uval.size());
        int pos = 0;
        while (pos < d && uval[pos] < v) ++pos;
        if (pos < d && uval[pos] == v) {
          ucnt[pos] += 1;
        } else {
          uval.insert(uval.begin() + pos, v);
          ucnt.insert(ucnt.begin() + pos, 1);
        }
      }
      int d = static_cast<int>(uval.size());
      usuf.assign(d + 1, 0);
      for (int z = d - 1; z >= 0; --z) usuf[z] = usuf[z + 1] + ucnt[z];
      dist_ready = true;
    }
    rank_sum += capped_rank(value, uval, ucnt, usuf, nvals, nrow, maxRank);
  }

  double rank_sum_min = static_cast<double>(len_sig) * (len_sig + 1.0) / 2.0;
  double denom = static_cast<double>(len_sig) * maxRank - rank_sum_min;
  return 1.0 - (rank_sum - rank_sum_min) / denom;
}

static const int UCELL_BIN_K = 256;

static double score_one_signature_int(const std::vector<int>& idx,
                                      const std::vector<double>& gene_val,
                                      std::vector<double>& uval,
                                      std::vector<int>& ucnt,
                                      std::vector<int>& usuf,
                                      std::vector<int>& small_cnt,
                                      std::vector<int>& used_bins,
                                      bool& dist_ready,
                                      const double* values_begin,
                                      int nvals,
                                      int nrow, int maxRank) {
  int len_sig = static_cast<int>(idx.size());
  if (len_sig <= 0) return 0.0;

  double rank_sum = 0.0;
  for (int k = 0; k < len_sig; ++k) {
    int gene = idx[k];
    if (gene <= 0) {
      rank_sum += maxRank;
      continue;
    }
    int row = gene - 1;
    double value = gene_val[row];
    if (value == 0.0 && nrow >= 2 * maxRank) {
      rank_sum += maxRank;
      continue;
    }
    if (!dist_ready) {
      uval.clear();
      ucnt.clear();
      used_bins.clear();
      for (int z = 0; z < nvals; ++z) {
        double v = values_begin[z];
        int iv = static_cast<int>(v);
        if (iv >= 1 && iv <= UCELL_BIN_K && static_cast<double>(iv) == v) {
          if (small_cnt[iv] == 0) used_bins.push_back(iv);
          small_cnt[iv] += 1;
        } else {
          int d = static_cast<int>(uval.size());
          int pos = 0;
          while (pos < d && uval[pos] < v) ++pos;
          if (pos < d && uval[pos] == v) {
            ucnt[pos] += 1;
          } else {
            uval.insert(uval.begin() + pos, v);
            ucnt.insert(ucnt.begin() + pos, 1);
          }
        }
      }
      std::sort(used_bins.begin(), used_bins.end());
      int nb = static_cast<int>(used_bins.size());
      uval.insert(uval.begin(), nb, 0.0);
      ucnt.insert(ucnt.begin(), nb, 0);
      for (int b = 0; b < nb; ++b) {
        uval[b] = static_cast<double>(used_bins[b]);
        ucnt[b] = small_cnt[used_bins[b]];
        small_cnt[used_bins[b]] = 0;
      }
      int d = static_cast<int>(uval.size());
      usuf.assign(d + 1, 0);
      for (int z = d - 1; z >= 0; --z) usuf[z] = usuf[z + 1] + ucnt[z];
      dist_ready = true;
    }
    rank_sum += capped_rank(value, uval, ucnt, usuf, nvals, nrow, maxRank);
  }

  double rank_sum_min = static_cast<double>(len_sig) * (len_sig + 1.0) / 2.0;
  double denom = static_cast<double>(len_sig) * maxRank - rank_sum_min;
  return 1.0 - (rank_sum - rank_sum_min) / denom;
}

// [[Rcpp::export]]
NumericMatrix ucell_fast_scores_dgC(IntegerVector p, IntegerVector i,
                                    NumericVector x, int nrow,
                                    List pos_list, List neg_list,
                                    int maxRank, double w_neg, int int_mat) {
  int ncells = p.size() - 1;
  int nsigs = pos_list.size();
  NumericMatrix out(ncells, nsigs);

  std::vector< std::vector<int> > pos_vec;
  std::vector< std::vector<int> > neg_vec;
  pos_vec.reserve(nsigs);
  neg_vec.reserve(nsigs);
  for (int sig = 0; sig < nsigs; ++sig) {
    IntegerVector pos_idx = as<IntegerVector>(pos_list[sig]);
    IntegerVector neg_idx = as<IntegerVector>(neg_list[sig]);
    std::vector<int> pos_current;
    std::vector<int> neg_current;
    pos_current.reserve(pos_idx.size());
    neg_current.reserve(neg_idx.size());
    for (int k = 0; k < pos_idx.size(); ++k) {
      int gene = pos_idx[k];
      pos_current.push_back(IntegerVector::is_na(gene) ? -1 : gene);
    }
    for (int k = 0; k < neg_idx.size(); ++k) {
      int gene = neg_idx[k];
      neg_current.push_back(IntegerVector::is_na(gene) ? -1 : gene);
    }
    pos_vec.push_back(pos_current);
    neg_vec.push_back(neg_current);
  }

  std::vector<char> is_sig(nrow, 0);
  for (int sig = 0; sig < nsigs; ++sig) {
    const std::vector<int>& pv = pos_vec[sig];
    const std::vector<int>& nv = neg_vec[sig];
    for (size_t k = 0; k < pv.size(); ++k) if (pv[k] > 0) is_sig[pv[k] - 1] = 1;
    for (size_t k = 0; k < nv.size(); ++k) if (nv[k] > 0) is_sig[nv[k] - 1] = 1;
  }

  const int* rows = INTEGER(i);
  const double* values = REAL(x);

  std::vector<double> uval;
  std::vector<int> ucnt;
  std::vector<int> usuf;
  std::vector<double> gene_val(nrow, 0.0);
  std::vector<int> touched;
  std::vector<int> small_cnt(UCELL_BIN_K + 1, 0);
  std::vector<int> used_bins;

  for (int cell = 0; cell < ncells; ++cell) {
    int start = p[cell];
    int end = p[cell + 1];
    int nvals = end - start;
    bool dist_ready = false;
    const double* x_begin = values + start;
    touched.clear();

    if (int_mat) {
      uval.clear();
      ucnt.clear();
      used_bins.clear();
      for (int z = start; z < end; ++z) {
        double v = values[z];
        int r = rows[z];
        if (is_sig[r]) {
          gene_val[r] = v;
          touched.push_back(r);
        }
        int iv = static_cast<int>(v);
        if (iv >= 1 && iv <= UCELL_BIN_K && static_cast<double>(iv) == v) {
          if (small_cnt[iv] == 0) used_bins.push_back(iv);
          small_cnt[iv] += 1;
        } else {
          int d = static_cast<int>(uval.size());
          int pos = 0;
          while (pos < d && uval[pos] < v) ++pos;
          if (pos < d && uval[pos] == v) {
            ucnt[pos] += 1;
          } else {
            uval.insert(uval.begin() + pos, v);
            ucnt.insert(ucnt.begin() + pos, 1);
          }
        }
      }
      std::sort(used_bins.begin(), used_bins.end());
      int nb = static_cast<int>(used_bins.size());
      uval.insert(uval.begin(), nb, 0.0);
      ucnt.insert(ucnt.begin(), nb, 0);
      for (int b = 0; b < nb; ++b) {
        uval[b] = static_cast<double>(used_bins[b]);
        ucnt[b] = small_cnt[used_bins[b]];
        small_cnt[used_bins[b]] = 0;
      }
      int d = static_cast<int>(uval.size());
      usuf.assign(d + 1, 0);
      for (int z = d - 1; z >= 0; --z) usuf[z] = usuf[z + 1] + ucnt[z];
      dist_ready = true;
      for (int sig = 0; sig < nsigs; ++sig) {
        double u_pos = score_one_signature_int(
          pos_vec[sig], gene_val, uval, ucnt, usuf,
          small_cnt, used_bins, dist_ready, x_begin, nvals, nrow, maxRank);
        double u_neg = score_one_signature_int(
          neg_vec[sig], gene_val, uval, ucnt, usuf,
          small_cnt, used_bins, dist_ready, x_begin, nvals, nrow, maxRank);
        double score = u_pos - w_neg * u_neg;
        out(cell, sig) = score < 0.0 ? 0.0 : score;
      }
    } else {
      for (int z = start; z < end; ++z) {
        int r = rows[z];
        if (is_sig[r]) {
          gene_val[r] = values[z];
          touched.push_back(r);
        }
      }
      for (int sig = 0; sig < nsigs; ++sig) {
        double u_pos = score_one_signature(
          pos_vec[sig], gene_val, uval, ucnt, usuf,
          dist_ready, x_begin, nvals, nrow, maxRank);
        double u_neg = score_one_signature(
          neg_vec[sig], gene_val, uval, ucnt, usuf,
          dist_ready, x_begin, nvals, nrow, maxRank);
        double score = u_pos - w_neg * u_neg;
        out(cell, sig) = score < 0.0 ? 0.0 : score;
      }
    }

    for (size_t t = 0; t < touched.size(); ++t) gene_val[touched[t]] = 0.0;
  }

  return out;
}
