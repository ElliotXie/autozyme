// [[Rcpp::depends(Rcpp)]]
// [[Rcpp::plugins(openmp)]]
//
// Kernels for the cellchat patch. Lifted verbatim from
// test_cellchat/pipeline/run.R (rounds 4, 12, 14, 17).
//
// OMP usage is wrapped in `#ifdef _OPENMP` so the file also compiles
// (serially) on toolchains without libomp. macOS users get parallelism
// when libomp is detected by src/Makevars (Apple Clang ships libomp
// separately; the Makevars hunts homebrew/anaconda/macports paths).

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace {

// type-7 linear-interp quantile at h = (n-1)*p; matches stats::quantile default
// and matrixStats::rowQuantiles.
inline double interp_quantile(std::vector<double>& buf, int n, double h) {
  int lo = static_cast<int>(h);
  double f = h - lo;
  std::nth_element(buf.begin(), buf.begin() + lo, buf.end());
  double v_lo = buf[lo];
  double v_hi = (f > 0)
    ? *std::min_element(buf.begin() + lo + 1, buf.end())
    : v_lo;
  return v_lo + f * (v_hi - v_lo);
}

inline double tri_mean_from_n_elements(std::vector<double>& buf, int n) {
  double h25 = 0.25 * (n - 1);
  double h50 = 0.50 * (n - 1);
  double h75 = 0.75 * (n - 1);
  double q25 = interp_quantile(buf, n, h25);
  double q50 = interp_quantile(buf, n, h50);
  double q75 = interp_quantile(buf, n, h75);
  return (q25 + 2.0 * q50 + q75) * 0.25;
}

} // namespace

// Per-(gene, group) triMean using std::nth_element. Same type-7 quantile as
// matrixStats::rowQuantiles within FP noise (max diff ~6e-17). Single-threaded
// (cheap; bootstrap-batched variant handles the parallelizable workload).
//
// [[Rcpp::export]]
NumericMatrix cpp_aggregate_triMean(NumericMatrix data, IntegerVector group_int, int ngroups) {
  int ngenes = data.nrow();
  int nC = data.ncol();
  NumericMatrix out(ngenes, ngroups);
  std::vector<std::vector<int>> cells_per_group(ngroups);
  for (int c = 0; c < nC; ++c) {
    int g = group_int[c] - 1;
    cells_per_group[g].push_back(c);
  }
  std::vector<double> buf;
  for (int g = 0; g < ngroups; ++g) {
    const std::vector<int>& cells = cells_per_group[g];
    int n = cells.size();
    if (n == 0) {
      for (int i = 0; i < ngenes; ++i) out(i, g) = 0.0;
      continue;
    }
    buf.resize(n);
    double h25 = 0.25 * (n - 1);
    double h50 = 0.50 * (n - 1);
    double h75 = 0.75 * (n - 1);
    int lo25 = (int)h25; double f25 = h25 - lo25;
    int lo50 = (int)h50; double f50 = h50 - lo50;
    int lo75 = (int)h75; double f75 = h75 - lo75;
    for (int i = 0; i < ngenes; ++i) {
      for (int j = 0; j < n; ++j) buf[j] = data(i, cells[j]);
      std::nth_element(buf.begin(), buf.begin() + lo25, buf.end());
      double v_lo25 = buf[lo25];
      double v_hi25 = (f25 > 0) ? *std::min_element(buf.begin() + lo25 + 1, buf.end()) : v_lo25;
      double q25 = v_lo25 + f25 * (v_hi25 - v_lo25);
      std::nth_element(buf.begin(), buf.begin() + lo50, buf.end());
      double v_lo50 = buf[lo50];
      double v_hi50 = (f50 > 0) ? *std::min_element(buf.begin() + lo50 + 1, buf.end()) : v_lo50;
      double q50 = v_lo50 + f50 * (v_hi50 - v_lo50);
      std::nth_element(buf.begin(), buf.begin() + lo75, buf.end());
      double v_lo75 = buf[lo75];
      double v_hi75 = (f75 > 0) ? *std::min_element(buf.begin() + lo75 + 1, buf.end()) : v_lo75;
      double q75 = v_lo75 + f75 * (v_hi75 - v_lo75);
      out(i, g) = (q25 + 2.0*q50 + q75) * 0.25;
    }
  }
  return out;
}

// Batched bootstrap aggregator — produces (ngenes × ngroups × nboot) tensor
// directly. Each thread holds its own grp_perm/cells_per_group/buf.
//
// [[Rcpp::export]]
NumericVector cpp_aggregate_triMean_boot(
    NumericMatrix data, IntegerVector group_int, int ngroups,
    IntegerMatrix permutation) {
  int ngenes = data.nrow();
  int nC = data.ncol();
  int nboot = permutation.ncol();
  size_t plane = (size_t)ngenes * ngroups;
  NumericVector out(plane * nboot);
  double* op = REAL(out);
  int* gp_data = INTEGER(group_int);
  int* perm_data = INTEGER(permutation);
  double* dp = REAL(data);

  #ifdef _OPENMP
  #pragma omp parallel
  #endif
  {
    std::vector<int> grp_perm(nC);
    std::vector<std::vector<int>> cells_per_group(ngroups);
    std::vector<double> buf;
    #ifdef _OPENMP
    #pragma omp for schedule(static)
    #endif
    for (int nE = 0; nE < nboot; ++nE) {
      int* perm_col = perm_data + (size_t)nE * nC;
      for (int c = 0; c < nC; ++c) grp_perm[c] = gp_data[perm_col[c] - 1];
      for (int g = 0; g < ngroups; ++g) cells_per_group[g].clear();
      for (int c = 0; c < nC; ++c) cells_per_group[grp_perm[c] - 1].push_back(c);

      double* out_nE = op + (size_t)nE * plane;
      for (int g = 0; g < ngroups; ++g) {
        const std::vector<int>& cells = cells_per_group[g];
        int n = cells.size();
        if (n == 0) {
          for (int i = 0; i < ngenes; ++i) out_nE[i + (size_t)g * ngenes] = 0.0;
          continue;
        }
        buf.resize(n);
        double h25 = 0.25 * (n - 1), h50 = 0.50 * (n - 1), h75 = 0.75 * (n - 1);
        int lo25 = (int)h25, lo50 = (int)h50, lo75 = (int)h75;
        double f25 = h25 - lo25, f50 = h50 - lo50, f75 = h75 - lo75;
        for (int i = 0; i < ngenes; ++i) {
          for (int j = 0; j < n; ++j) buf[j] = dp[i + (size_t)cells[j] * ngenes];
          std::nth_element(buf.begin(), buf.begin() + lo25, buf.end());
          double v_lo25 = buf[lo25];
          double v_hi25 = (f25 > 0) ? *std::min_element(buf.begin() + lo25 + 1, buf.end()) : v_lo25;
          double q25 = v_lo25 + f25 * (v_hi25 - v_lo25);
          std::nth_element(buf.begin(), buf.begin() + lo50, buf.end());
          double v_lo50 = buf[lo50];
          double v_hi50 = (f50 > 0) ? *std::min_element(buf.begin() + lo50 + 1, buf.end()) : v_lo50;
          double q50 = v_lo50 + f50 * (v_hi50 - v_lo50);
          std::nth_element(buf.begin(), buf.begin() + lo75, buf.end());
          double v_lo75 = buf[lo75];
          double v_hi75 = (f75 > 0) ? *std::min_element(buf.begin() + lo75 + 1, buf.end()) : v_lo75;
          double q75 = v_lo75 + f75 * (v_hi75 - v_lo75);
          out_nE[i + (size_t)g * ngenes] = (q25 + 2.0 * q50 + q75) * 0.25;
        }
      }
    }
  }
  return out;
}

// Pnull (= Prob_unperturbed) outer product over (c1, c2) for every LR pair.
// P.spatial = 1 for RNA mode (only mode this fast path supports).
//
// [[Rcpp::export]]
NumericVector cpp_outer_Pnull(
    NumericMatrix dataLavg_T,    /* numCluster × nLR */
    NumericMatrix dataRavg_T,    /* numCluster × nLR */
    NumericMatrix agonist_T,     /* numCluster × nLR (ones if not agonist) */
    NumericMatrix antagonist_T,  /* numCluster × nLR (ones if not antagonist) */
    int nLR, int numCluster,
    double Kh, double n) {
  int gg = numCluster * numCluster;
  NumericVector Prob(gg * nLR);
  double* dL = REAL(dataLavg_T);
  double* dR = REAL(dataRavg_T);
  double* dA = REAL(agonist_T);
  double* dAnt = REAL(antagonist_T);
  double* Pp = REAL(Prob);
  bool n_is_one = (n == 1.0);
  double Kh_n = n_is_one ? Kh : std::pow(Kh, n);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int k = 0; k < nLR; ++k) {
    double* Lcol  = dL  + (size_t)k * numCluster;
    double* Rcol  = dR  + (size_t)k * numCluster;
    double* Acol  = dA  + (size_t)k * numCluster;
    double* Antc  = dAnt + (size_t)k * numCluster;
    double* Pout  = Pp  + (size_t)k * gg;
    for (int c2 = 0; c2 < numCluster; ++c2) {
      double Rval = Rcol[c2];
      double Av2  = Acol[c2];
      double Antv2 = Antc[c2];
      for (int c1 = 0; c1 < numCluster; ++c1) {
        double dataLR = Lcol[c1] * Rval;
        double dataLR_n = n_is_one ? dataLR : std::pow(dataLR, n);
        double P1 = dataLR_n / (Kh_n + dataLR_n);
        Pout[c1 + c2 * numCluster] = P1 * Acol[c1] * Av2 * Antc[c1] * Antv2;
      }
    }
  }
  return Prob;
}

// Unified inner kernel — for every (LR, bootstrap), compute Prob' and
// accumulate nReject = #{Prob' > Pnull}. Subs are passed as flat int arrays
// + offsets so no R API calls happen inside the parallel region (thread-safe).
//
// [[Rcpp::export]]
NumericVector cpp_unified_inner(
    NumericVector boot_tensor, int ngenes, int ngroups, int nboot,
    IntegerVector Lflat,   IntegerVector Loff,
    IntegerVector Rflat,   IntegerVector Roff,
    IntegerVector coAflat, IntegerVector coAoff,
    IntegerVector coIflat, IntegerVector coIoff,
    IntegerVector agflat,  IntegerVector agoff,
    IntegerVector antflat, IntegerVector antoff,
    NumericVector Pnull_arr, int nLR_active,
    double Kh, double n) {
  int gg = ngroups * ngroups;
  NumericVector nReject(gg * nLR_active);
  double* bt = REAL(boot_tensor);
  double* Pn = REAL(Pnull_arr);
  double* nR = REAL(nReject);
  size_t plane = (size_t)ngenes * ngroups;
  bool n_is_one = (n == 1.0);
  double Kh_n = n_is_one ? Kh : std::pow(Kh, n);
  int* Lp = INTEGER(Lflat);   int* Lo = INTEGER(Loff);
  int* Rp = INTEGER(Rflat);   int* Ro = INTEGER(Roff);
  int* coAp = INTEGER(coAflat); int* coAo = INTEGER(coAoff);
  int* coIp = INTEGER(coIflat); int* coIo = INTEGER(coIoff);
  int* agp  = INTEGER(agflat);  int* ago  = INTEGER(agoff);
  int* antp = INTEGER(antflat); int* anto = INTEGER(antoff);

  #ifdef _OPENMP
  #pragma omp parallel
  #endif
  {
    std::vector<double> Lvec(ngroups), Rvec(ngroups), agvec(ngroups), antvec(ngroups);
    #ifdef _OPENMP
    #pragma omp for schedule(static)
    #endif
    for (int k = 0; k < nLR_active; ++k) {
      int* Ls = Lp + Lo[k]; int nL  = Lo[k+1] - Lo[k];
      int* Rs = Rp + Ro[k]; int nRs = Ro[k+1] - Ro[k];
      int* coAs = coAp + coAo[k]; int ncoA = coAo[k+1] - coAo[k];
      int* coIs = coIp + coIo[k]; int ncoI = coIo[k+1] - coIo[k];
      int* ags  = agp  + ago[k];  int nag  = ago[k+1]  - ago[k];
      int* ants = antp + anto[k]; int nant = anto[k+1] - anto[k];
      int Pn_off = k * gg;
      int nR_off = k * gg;

      for (int nE = 0; nE < nboot; ++nE) {
        double* boot_nE = bt + (size_t)nE * plane;
        if (nL == 1) {
          for (int c = 0; c < ngroups; ++c) Lvec[c] = boot_nE[Ls[0] + (size_t)c * ngenes];
        } else if (nL > 1) {
          for (int c = 0; c < ngroups; ++c) {
            double s = 0;
            for (int j = 0; j < nL; ++j) s += std::log(boot_nE[Ls[j] + (size_t)c * ngenes]);
            Lvec[c] = std::exp(s / nL);
          }
        } else {
          for (int c = 0; c < ngroups; ++c) Lvec[c] = R_NaReal;
        }
        if (nRs == 1) {
          for (int c = 0; c < ngroups; ++c) Rvec[c] = boot_nE[Rs[0] + (size_t)c * ngenes];
        } else if (nRs > 1) {
          for (int c = 0; c < ngroups; ++c) {
            double s = 0;
            for (int j = 0; j < nRs; ++j) s += std::log(boot_nE[Rs[j] + (size_t)c * ngenes]);
            Rvec[c] = std::exp(s / nRs);
          }
        } else {
          for (int c = 0; c < ngroups; ++c) Rvec[c] = R_NaReal;
        }
        if (ncoA > 0) {
          for (int c = 0; c < ngroups; ++c) {
            double pr = 1; for (int j = 0; j < ncoA; ++j) pr *= (1.0 + boot_nE[coAs[j] + (size_t)c * ngenes]);
            Rvec[c] *= pr;
          }
        }
        if (ncoI > 0) {
          for (int c = 0; c < ngroups; ++c) {
            double pr = 1; for (int j = 0; j < ncoI; ++j) pr *= (1.0 + boot_nE[coIs[j] + (size_t)c * ngenes]);
            Rvec[c] /= pr;
          }
        }
        if (nag > 0) {
          for (int c = 0; c < ngroups; ++c) {
            double pr = 1;
            for (int j = 0; j < nag; ++j) {
              double v = boot_nE[ags[j] + (size_t)c * ngenes];
              double v_n = n_is_one ? v : std::pow(v, n);
              pr *= (1.0 + v_n / (Kh_n + v_n));
            }
            agvec[c] = pr;
          }
        } else {
          for (int c = 0; c < ngroups; ++c) agvec[c] = 1.0;
        }
        if (nant > 0) {
          for (int c = 0; c < ngroups; ++c) {
            double pr = 1;
            for (int j = 0; j < nant; ++j) {
              double v = boot_nE[ants[j] + (size_t)c * ngenes];
              double v_n = n_is_one ? v : std::pow(v, n);
              pr *= (Kh_n / (Kh_n + v_n));
            }
            antvec[c] = pr;
          }
        } else {
          for (int c = 0; c < ngroups; ++c) antvec[c] = 1.0;
        }
        for (int c2 = 0; c2 < ngroups; ++c2) {
          double Rval = Rvec[c2]; double ag2 = agvec[c2]; double ant2 = antvec[c2];
          int col_off = c2 * ngroups;
          for (int c1 = 0; c1 < ngroups; ++c1) {
            double dataLR = Lvec[c1] * Rval;
            double dataLR_n = n_is_one ? dataLR : std::pow(dataLR, n);
            double P1 = dataLR_n / (Kh_n + dataLR_n);
            double Pb = P1 * agvec[c1] * ag2 * antvec[c1] * ant2;
            if (Pb > Pn[Pn_off + c1 + col_off]) nR[nR_off + c1 + col_off] += 1.0;
          }
        }
      }
    }
  }
  return nReject;
}
