#include <Rcpp.h>

#include <cstdint>
#include <cstdlib>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

extern "C" {
#include "scblas/knn.h"
#include "scblas/umap.h"
}

static int autozyme_scblas_loader_pid = -1;

static int autozyme_scblas_getpid(void) {
#ifdef _WIN32
  return static_cast<int>(_getpid());
#else
  return static_cast<int>(getpid());
#endif
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
#endif
static void autozyme_scblas_loader_init(void) {
  autozyme_scblas_loader_pid = autozyme_scblas_getpid();
}

extern "C" int scblas_in_loader_process(void) {
  if (autozyme_scblas_loader_pid < 0) {
    autozyme_scblas_loader_init();
  }
  return (autozyme_scblas_getpid() == autozyme_scblas_loader_pid) ? 1 : 0;
}

// [[Rcpp::export]]
Rcpp::List az_runumap_knn_cpp(
    Rcpp::NumericMatrix X,
    int k,
    int n_trees,
    int n_iters,
    int leaf_size,
    double seed,
    int n_threads) {
  const int n = X.nrow();
  const int d = X.ncol();
  if (n <= 1 || d <= 0 || k <= 0 || k >= n) {
    Rcpp::stop("invalid kNN dimensions");
  }

  std::vector<float> xf(static_cast<size_t>(n) * static_cast<size_t>(d));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      xf[static_cast<size_t>(i) * static_cast<size_t>(d) + static_cast<size_t>(j)] =
          static_cast<float>(X(i, j));
    }
  }

  std::vector<int32_t> idx(static_cast<size_t>(n) * static_cast<size_t>(k));
  std::vector<float> dist(static_cast<size_t>(n) * static_cast<size_t>(k));
  int rc = scblas_knn_descent_f32(
      n, d, k, xf.data(), n_trees, n_iters, leaf_size,
      static_cast<uint64_t>(seed), idx.data(), dist.data(), n_threads);
  if (rc != 0) {
    Rcpp::stop("scBLAS kNN failed with code %d", rc);
  }

  Rcpp::IntegerMatrix out_idx(n, k);
  Rcpp::NumericMatrix out_dist(n, k);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      const size_t off = static_cast<size_t>(i) * static_cast<size_t>(k) +
                         static_cast<size_t>(j);
      out_idx(i, j) = idx[off] + 1;
      out_dist(i, j) = static_cast<double>(dist[off]);
    }
  }

  return Rcpp::List::create(
      Rcpp::_["idx"] = out_idx,
      Rcpp::_["dist"] = out_dist,
      Rcpp::_["rc"] = rc);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix az_runumap_layout_cpp(
    Rcpp::NumericMatrix init,
    Rcpp::IntegerVector head,
    Rcpp::IntegerVector tail,
    Rcpp::NumericVector epochs_per_sample,
    int n_epochs,
    double a,
    double b,
    double gamma,
    double initial_alpha,
    double negative_sample_rate,
    double seed,
    int n_threads) {
  const int n = init.nrow();
  const int dim = init.ncol();
  const int64_t n_edges = static_cast<int64_t>(head.size());
  if (tail.size() != n_edges || epochs_per_sample.size() != n_edges) {
    Rcpp::stop("head, tail, and epochs_per_sample must have equal length");
  }
  if (n <= 0 || dim <= 0 || n_edges <= 0 || n_epochs <= 0) {
    Rcpp::stop("invalid UMAP layout dimensions");
  }

  std::vector<float> emb(static_cast<size_t>(n) * static_cast<size_t>(dim));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < dim; ++j) {
      emb[static_cast<size_t>(i) * static_cast<size_t>(dim) + static_cast<size_t>(j)] =
          static_cast<float>(init(i, j));
    }
  }

  std::vector<int32_t> h(static_cast<size_t>(n_edges));
  std::vector<int32_t> t(static_cast<size_t>(n_edges));
  std::vector<float> eps(static_cast<size_t>(n_edges));
  for (int64_t i = 0; i < n_edges; ++i) {
    h[static_cast<size_t>(i)] = static_cast<int32_t>(head[static_cast<R_xlen_t>(i)]);
    t[static_cast<size_t>(i)] = static_cast<int32_t>(tail[static_cast<R_xlen_t>(i)]);
    eps[static_cast<size_t>(i)] =
        static_cast<float>(epochs_per_sample[static_cast<R_xlen_t>(i)]);
  }

  int rc = scblas_umap_optimize_layout_euclidean_f32_parallel(
      emb.data(), n, dim, h.data(), t.data(), n_edges, eps.data(), n_epochs,
      static_cast<float>(a), static_cast<float>(b), static_cast<float>(gamma),
      static_cast<float>(initial_alpha), static_cast<float>(negative_sample_rate),
      static_cast<uint64_t>(seed), n_threads);

  Rcpp::NumericMatrix out(n, dim);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < dim; ++j) {
      out(i, j) = static_cast<double>(
          emb[static_cast<size_t>(i) * static_cast<size_t>(dim) + static_cast<size_t>(j)]);
    }
  }
  out.attr("scblas_return_code") = rc;
  return out;
}
