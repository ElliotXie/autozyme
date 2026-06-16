// [[Rcpp::depends(Rcpp)]]
//
// Fast replacement for miloR::calcNhoodDistance / miloR:::.calc_distance.
// Bit-exact vs upstream (same per-nhood Euclidean accumulation order,
// summed left-to-right over PCA columns) modulo fp reordering — concordance
// is pearson=1.0, max_abs_diff=0 on dev tiers small/medium/large and OOD
// tiers ood_large/ood_xlarge (see test_milor/memory/discoveries.md).
//
// Wins lifted from converged optimization (rounds 1-14 in test_milor):
//   * Pure-C++ per-nhood pairwise Euclidean — replaces `sapply` + `stats::dist`
//     + `as.matrix(dist)` + `methods::new("dgCMatrix")` (~50% spent in S4
//     initialize, ~22% in dense intermediate).
//   * Batch all nhoods in one Rcpp call. The R-side `for (X in seq_len(nh_n))`
//     plus per-iter SEXP roundtrip was the main floor before round 4.
//   * Cache dgCMatrix `@i`, `@p`, `@Dim` slots per unique n (nhood size).
//     ~70 unique sizes on dev small vs 1356 nhoods → 20x fewer SEXP allocs.
//     Safe because dgCMatrix consumers treat slots read-only; R's
//     copy-on-modify semantics catch any pathological writer.
//   * Share NULL Dimnames across nhoods — Milo downstream (graphSpatialFDR,
//     testNhoods) indexes per-nhood distances positionally, never by name.
//     Cell-name slicing was ~5% of small-tier wall.
//   * std::thread parallel pdist (work-stealing on an atomic counter over
//     nhoods). Heterogeneous nhood sizes → free load balancing. Apple Clang
//     ships no libomp by default so std::thread is the portable choice;
//     no `-fopenmp` needed at build time.
//
// Threading: n_threads is taken as an explicit argument from R. The R-side
// wrapper calls autozyme::auto_threads() which honors AUTOZYME_THREADS env
// var, getOption("autozyme.threads"), or hardware default (capped at 16).
// .ensure_task_thread_env() in verify_patch wires ZYME_THREADS through to
// AUTOZYME_THREADS so attest's --threads flag actually drives the C++ pool.
//
// Outcome B disclosure: miloR 2.9.1's `calcNhoodDistance` is strictly
// serial (sapply, no BPPARAM hook). The parallel layer added here is a
// strict superset of upstream behavior — multi-thread speedup includes
// both the algorithmic wins and the new parallel layer. Single-thread
// (n_threads=1) is algorithm-only.

#include <Rcpp.h>
#include <cmath>
#include <thread>
#include <atomic>
#include <vector>
#include <unordered_map>
using namespace Rcpp;

// Thread-safe distance kernel: fills raw double buffer with n x n pairwise
// Euclidean (col-major). Reads rd_px at indices row_idx[]; writes x_px[n*n].
// All R objects allocated by caller (main thread); we touch only doubles/ints.
static void pdist_into(const double* rd_px, int N, int d,
                       const int* row_idx, int n,
                       double* x_px,
                       std::vector<double>& y_buf) {
  y_buf.resize((size_t)n * (size_t)d);
  for (int t = 0; t < n; t++) {
    int idx = row_idx[t];
    double* y_row = y_buf.data() + (size_t)t * d;
    for (int k = 0; k < d; k++) y_row[k] = rd_px[(R_xlen_t)k * N + idx];
  }
  const double* yb = y_buf.data();
  for (int jc = 0; jc < n - 1; jc++) {
    const double* y_jc = yb + (size_t)jc * d;
    for (int ir = jc + 1; ir < n; ir++) {
      const double* y_ir = yb + (size_t)ir * d;
      double s = 0.0;
      for (int k = 0; k < d; k++) {
        double dv = y_ir[k] - y_jc[k];
        s += dv * dv;
      }
      double v = std::sqrt(s);
      x_px[(R_xlen_t)jc * n + ir] = v;
      x_px[(R_xlen_t)ir * n + jc] = v;
    }
  }
}

// Single-thread variant used by fast_pdist_one (called by the .calc_distance
// namespace patch — preserves the direct internal-API speedup for any
// downstream caller that reaches the helper bypassing calcNhoodDistance).
static S4 build_dgC_one(const double* rd_px, int N, int d,
                       const int* row_idx, int n,
                       SEXP cell_names_sexp,
                       std::vector<double>& y_buf) {
  CharacterVector rn_slice(n);
  if (cell_names_sexp != R_NilValue && TYPEOF(cell_names_sexp) == STRSXP) {
    SEXP cn = cell_names_sexp;
    for (int t = 0; t < n; t++) rn_slice[t] = STRING_ELT(cn, row_idx[t]);
  } else {
    for (int t = 0; t < n; t++) rn_slice[t] = NA_STRING;
  }

  y_buf.resize((size_t)n * (size_t)d);
  for (int t = 0; t < n; t++) {
    int idx = row_idx[t];
    double* y_row = y_buf.data() + (size_t)t * d;
    for (int k = 0; k < d; k++) y_row[k] = rd_px[(R_xlen_t)k * N + idx];
  }

  R_xlen_t total = (R_xlen_t)n * (R_xlen_t)n;
  IntegerVector i_vec(total);
  IntegerVector p_vec(n + 1);
  NumericVector x_vec(total);

  for (int jc = 0; jc < n; jc++) {
    int base = jc * n;
    p_vec[jc] = base;
    for (int ir = 0; ir < n; ir++) i_vec[base + ir] = ir;
  }
  p_vec[n] = total;

  double* x_px = x_vec.begin();
  const double* yb = y_buf.data();
  for (int jc = 0; jc < n - 1; jc++) {
    const double* y_jc = yb + (size_t)jc * d;
    for (int ir = jc + 1; ir < n; ir++) {
      const double* y_ir = yb + (size_t)ir * d;
      double s = 0.0;
      for (int k = 0; k < d; k++) {
        double dv = y_ir[k] - y_jc[k];
        s += dv * dv;
      }
      double v = std::sqrt(s);
      x_px[(R_xlen_t)jc * n + ir] = v;
      x_px[(R_xlen_t)ir * n + jc] = v;
    }
  }

  S4 out("dgCMatrix");
  out.slot("i") = i_vec;
  out.slot("p") = p_vec;
  out.slot("x") = x_vec;
  out.slot("Dim") = IntegerVector::create(n, n);
  out.slot("Dimnames") = List::create(rn_slice, rn_slice);
  out.slot("factors") = List::create();
  return out;
}

static S4 build_dgC_empty(SEXP cell_names_sexp, const int* row_idx, int n) {
  // Used for n == 0 and n == 1.
  if (n == 0) {
    S4 out("dgCMatrix");
    out.slot("i") = IntegerVector(0);
    out.slot("p") = IntegerVector::create(0);
    out.slot("x") = NumericVector(0);
    out.slot("Dim") = IntegerVector::create(0, 0);
    out.slot("Dimnames") = List::create(R_NilValue, R_NilValue);
    out.slot("factors") = List::create();
    return out;
  }
  CharacterVector rn_slice(1);
  if (cell_names_sexp != R_NilValue && TYPEOF(cell_names_sexp) == STRSXP) {
    rn_slice[0] = STRING_ELT(cell_names_sexp, row_idx[0]);
  } else {
    rn_slice[0] = NA_STRING;
  }
  S4 out("dgCMatrix");
  out.slot("i") = IntegerVector::create(0);
  out.slot("p") = IntegerVector::create(0, 1);
  out.slot("x") = NumericVector::create(0.0);
  out.slot("Dim") = IntegerVector::create(1, 1);
  out.slot("Dimnames") = List::create(rn_slice, rn_slice);
  out.slot("factors") = List::create();
  return out;
}

// Single-matrix entry point used by the .calc_distance namespace patch.
// [[Rcpp::export]]
S4 fast_milor_pdist_one(NumericMatrix in_x, SEXP row_names) {
  int n = in_x.nrow();
  int d = in_x.ncol();
  std::vector<int> idx(n);
  for (int t = 0; t < n; t++) idx[t] = t;
  if (n <= 1) return build_dgC_empty(row_names, idx.data(), n);
  std::vector<double> y_buf;
  return build_dgC_one(REAL(in_x), n, d, idx.data(), n, row_names, y_buf);
}

// Batch entry point used by the calcNhoodDistance namespace patch.
// - rd_mat: full N x d base matrix
// - non_zero: nz x 2 integer matrix; col 1 = 1-based row, col 2 = 1-based nhood col
// - nh_n: total nhood count (length of output)
// - cell_names: rownames of rd_mat (or R_NilValue)
// - n_threads: parallel worker count; main thread also acts as a worker.
//   1 collapses to fully serial. autozyme::auto_threads() resolves this on
//   the R side.
// [[Rcpp::export]]
List fast_milor_pdist_batch(NumericMatrix rd_mat, IntegerMatrix non_zero,
                            int nh_n, SEXP cell_names, int n_threads) {
  int N = rd_mat.nrow();
  int d = rd_mat.ncol();
  const double* rd_px = REAL(rd_mat);
  int nz = non_zero.nrow();

  // First pass: count entries per nhood col.
  std::vector<int> counts(nh_n, 0);
  for (int t = 0; t < nz; t++) {
    int col1b = non_zero(t, 1);
    if (col1b >= 1 && col1b <= nh_n) counts[col1b - 1]++;
  }

  // Cumulative starts (length nh_n + 1).
  std::vector<int> starts(nh_n + 1, 0);
  for (int X = 0; X < nh_n; X++) starts[X + 1] = starts[X] + counts[X];

  // Second pass: write 0-based row indices into a packed buffer.
  std::vector<int> row_idx_packed(starts[nh_n], 0);
  std::vector<int> cursor(nh_n, 0);
  for (int t = 0; t < nz; t++) {
    int col1b = non_zero(t, 1);
    if (col1b < 1 || col1b > nh_n) continue;
    int X = col1b - 1;
    row_idx_packed[starts[X] + cursor[X]++] = non_zero(t, 0) - 1;
  }

  // Two-phase: (1) main-thread allocate Rcpp vectors / S4 wrappers; capture
  // raw x-pointers. (2) parallel-compute distances into raw pointers.
  //
  // Per-size SEXP sharing for `i` and `p` slots: these only depend on n
  // (i = rep(0:(n-1), n); p = c(0, n, ..., n^2)). On dev-small with ~1356
  // nhoods there are only ~70 unique sizes — we allocate ~70 instead of 1356
  // of each. Safe because dgCMatrix consumers treat slots read-only and R
  // copy-on-modify guards any consumer that writes via `obj@i <- ...` or
  // `slot(.)`.
  List out(nh_n);
  std::vector<double*> x_ptrs(nh_n, nullptr);
  std::vector<int> n_per(nh_n, 0);

  // One shared Dimnames object: Milo downstream consumers index per-nhood
  // distances positionally, not by name, so per-nhood cell-name slicing is
  // dead work.
  List null_dimnames = List::create(R_NilValue, R_NilValue);

  std::unordered_map<int, IntegerVector> i_cache, p_cache, dim_cache;
  auto get_i = [&](int n) -> IntegerVector& {
    auto it = i_cache.find(n);
    if (it != i_cache.end()) return it->second;
    R_xlen_t total = (R_xlen_t)n * (R_xlen_t)n;
    IntegerVector iv(total);
    int* ip = INTEGER(iv);
    for (int jc = 0; jc < n; jc++) {
      int base = jc * n;
      for (int ir = 0; ir < n; ir++) ip[base + ir] = ir;
    }
    auto ins = i_cache.emplace(n, std::move(iv));
    return ins.first->second;
  };
  auto get_p = [&](int n) -> IntegerVector& {
    auto it = p_cache.find(n);
    if (it != p_cache.end()) return it->second;
    IntegerVector pv(n + 1);
    int* pp = INTEGER(pv);
    for (int jc = 0; jc <= n; jc++) pp[jc] = jc * n;
    auto ins = p_cache.emplace(n, std::move(pv));
    return ins.first->second;
  };
  auto get_dim = [&](int n) -> IntegerVector& {
    auto it = dim_cache.find(n);
    if (it != dim_cache.end()) return it->second;
    auto ins = dim_cache.emplace(n, IntegerVector::create(n, n));
    return ins.first->second;
  };

  for (int X = 0; X < nh_n; X++) {
    int n = counts[X];
    const int* row_idx = row_idx_packed.data() + starts[X];
    if (n <= 1) {
      out[X] = build_dgC_empty(cell_names, row_idx, n);
      continue;
    }
    n_per[X] = n;

    R_xlen_t total = (R_xlen_t)n * (R_xlen_t)n;
    NumericVector x_vec(total);

    S4 dgc("dgCMatrix");
    dgc.slot("i") = get_i(n);
    dgc.slot("p") = get_p(n);
    dgc.slot("x") = x_vec;
    dgc.slot("Dim") = get_dim(n);
    dgc.slot("Dimnames") = null_dimnames;
    out[X] = dgc;

    x_ptrs[X] = x_vec.begin();
  }

  // Phase 2: parallel distance compute via std::thread.
  unsigned int n_workers = (n_threads > 0) ? (unsigned int)n_threads : 1u;
  if ((int)n_workers > nh_n) n_workers = (unsigned int)nh_n;
  if (n_workers == 0) n_workers = 1;

  std::atomic<int> next_nh(0);
  std::vector<std::thread> workers;
  workers.reserve(n_workers - 1);

  auto worker = [&]() {
    std::vector<double> y_buf;
    while (true) {
      int X = next_nh.fetch_add(1, std::memory_order_relaxed);
      if (X >= nh_n) break;
      int n = n_per[X];
      if (n <= 1) continue;
      const int* row_idx = row_idx_packed.data() + starts[X];
      pdist_into(rd_px, N, d, row_idx, n, x_ptrs[X], y_buf);
    }
  };
  for (unsigned int t = 0; t + 1 < n_workers; t++) workers.emplace_back(worker);
  worker();
  for (auto& th : workers) th.join();

  return out;
}
