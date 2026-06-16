// turbo_neighbors.cpp — Parallel Annoy build+search for FindNeighbors
// [[Rcpp::depends(RcppAnnoy)]]
#include <Rcpp.h>
#include "annoylib.h"
#include "kissrandom.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#ifdef __APPLE__
#include <dlfcn.h>
#endif

using namespace Annoy;

typedef AnnoyIndex<int, float, Euclidean, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy> AnnoyIdx;

static inline void az_sift_down_f32(float* hd, int32_t* hi, int n) {
  int p = 0;
  for (;;) {
    int l = 2 * p + 1, r = 2 * p + 2, m = p;
    if (l < n && hd[l] > hd[m]) m = l;
    if (r < n && hd[r] > hd[m]) m = r;
    if (m == p) break;
    std::swap(hd[p], hd[m]);
    std::swap(hi[p], hi[m]);
    p = m;
  }
}

static inline void az_build_heap_f32(float* hd, int32_t* hi, int k) {
  for (int r = k / 2 - 1; r >= 0; --r) {
    int p = r;
    for (;;) {
      int l = 2 * p + 1, rr = 2 * p + 2, m = p;
      if (l < k && hd[l] > hd[m]) m = l;
      if (rr < k && hd[rr] > hd[m]) m = rr;
      if (m == p) break;
      std::swap(hd[p], hd[m]);
      std::swap(hi[p], hi[m]);
      p = m;
    }
  }
}

static inline void az_heap_to_sorted_f32(
    float* hd, int32_t* hi, int k, int32_t* out_idx, float* out_dist) {
  for (int m = k - 1; m >= 0; --m) {
    out_idx[m] = hi[0];
    out_dist[m] = hd[0];
    hd[0] = hd[m];
    hi[0] = hi[m];
    az_sift_down_f32(hd, hi, m);
  }
}

static inline float az_dist2_f32(const float* a, const float* b, int d) {
  float s = 0.0f;
  for (int j = 0; j < d; ++j) {
    float e = a[j] - b[j];
    s += e * e;
  }
  return s;
}

static int az_exact_knn_direct_f32(
    int n, int d, int k, const float* x,
    int32_t* out_idx, float* out_dist, int n_threads) {
  int eff = n_threads > 0 ? n_threads : static_cast<int>(std::thread::hardware_concurrency());
  if (eff < 1) eff = 1;
  if (eff > n) eff = n;
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(eff));
  for (int t = 0; t < eff; ++t) {
    int start = static_cast<int>((static_cast<int64_t>(t) * n) / eff);
    int end = static_cast<int>((static_cast<int64_t>(t + 1) * n) / eff);
    workers.emplace_back([=]() {
      std::vector<float> hd(static_cast<size_t>(k));
      std::vector<int32_t> hi(static_cast<size_t>(k));
      for (int i = start; i < end; ++i) {
        const float* xi = x + static_cast<size_t>(i) * d;
        for (int r = 0; r < k; ++r) {
          hd[static_cast<size_t>(r)] = az_dist2_f32(xi, x + static_cast<size_t>(r) * d, d);
          hi[static_cast<size_t>(r)] = r;
        }
        az_build_heap_f32(hd.data(), hi.data(), k);
        for (int r = k; r < n; ++r) {
          float dd = az_dist2_f32(xi, x + static_cast<size_t>(r) * d, d);
          if (dd < hd[0]) {
            hd[0] = dd;
            hi[0] = r;
            az_sift_down_f32(hd.data(), hi.data(), k);
          }
        }
        az_heap_to_sorted_f32(
          hd.data(), hi.data(), k,
          out_idx + static_cast<size_t>(i) * k,
          out_dist + static_cast<size_t>(i) * k);
      }
    });
  }
  for (auto& th : workers) th.join();
  return 0;
}

#ifdef __APPLE__
typedef void (*az_sgemm_fn)(int, int, int, int, int, int, float,
                            const float*, int, const float*, int,
                            float, float*, int);

static az_sgemm_fn az_load_sgemm(void) {
  static az_sgemm_fn fn = nullptr;
  static bool tried = false;
  if (!tried) {
    tried = true;
    void* h = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate",
                     RTLD_LAZY | RTLD_LOCAL);
    if (h != nullptr) {
      fn = reinterpret_cast<az_sgemm_fn>(dlsym(h, "cblas_sgemm"));
    }
  }
  return fn;
}

static int az_exact_knn_amx_f32(
    az_sgemm_fn sgemm, int n, int d, int k, const float* x,
    int32_t* out_idx, float* out_dist, int n_threads) {
  enum { CblasRowMajor = 101, CblasNoTrans = 111, CblasTrans = 112 };
  int eff = n_threads > 0 ? n_threads : static_cast<int>(std::thread::hardware_concurrency());
  if (eff < 1) eff = 1;
  if (eff > n) eff = n;

  long cap_mb = 32;
  if (const char* raw = std::getenv("AUTOZYME_SEURAT_KNN_TILE_MB")) {
    long parsed = std::atol(raw);
    if (parsed > 0) cap_mb = parsed;
  }
  long cap = (cap_mb << 20) / (static_cast<long>(n) * static_cast<long>(sizeof(float)));
  int tile_rows = cap < 64 ? 64 : static_cast<int>(cap);
  if (tile_rows > n) tile_rows = n;

  std::vector<float> qnorm(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const float* xi = x + static_cast<size_t>(i) * d;
    float s = 0.0f;
    for (int j = 0; j < d; ++j) s += xi[j] * xi[j];
    qnorm[static_cast<size_t>(i)] = s;
  }
  const std::vector<float>& rnorm = qnorm;
  std::vector<float> scores(static_cast<size_t>(tile_rows) * static_cast<size_t>(n));

  for (int row0 = 0; row0 < n; row0 += tile_rows) {
    int rows = std::min(tile_rows, n - row0);
    sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
          rows, n, d, 1.0f,
          x + static_cast<size_t>(row0) * d, d,
          x, d, 0.0f,
          scores.data(), n);

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(eff));
    for (int t = 0; t < eff; ++t) {
      workers.emplace_back([&, t]() {
        std::vector<float> hd(static_cast<size_t>(k));
        std::vector<int32_t> hi(static_cast<size_t>(k));
        for (int local = t; local < rows; local += eff) {
          const float* row_scores = scores.data() + static_cast<size_t>(local) * n;
          for (int j = 0; j < k; ++j) {
            hd[static_cast<size_t>(j)] = rnorm[static_cast<size_t>(j)] - 2.0f * row_scores[j];
            hi[static_cast<size_t>(j)] = j;
          }
          az_build_heap_f32(hd.data(), hi.data(), k);
          for (int j = k; j < n; ++j) {
            float dd = rnorm[static_cast<size_t>(j)] - 2.0f * row_scores[j];
            if (dd < hd[0]) {
              hd[0] = dd;
              hi[0] = j;
              az_sift_down_f32(hd.data(), hi.data(), k);
            }
          }
          const int q = row0 + local;
          int32_t* oi = out_idx + static_cast<size_t>(q) * k;
          float* od = out_dist + static_cast<size_t>(q) * k;
          for (int m = k - 1; m >= 0; --m) {
            oi[m] = hi[0];
            float v = qnorm[static_cast<size_t>(q)] + hd[0];
            od[m] = v > 0.0f ? v : 0.0f;
            hd[0] = hd[static_cast<size_t>(m)];
            hi[0] = hi[static_cast<size_t>(m)];
            az_sift_down_f32(hd.data(), hi.data(), m);
          }
        }
      });
    }
    for (auto& th : workers) th.join();
  }
  return 0;
}
#endif

static int az_exact_knn_f32(
    int n, int d, int k, const float* x,
    int32_t* out_idx, float* out_dist, int n_threads) {
  if (n <= 0 || d <= 0 || k <= 0 || k > n || x == nullptr) return -1;
#ifdef __APPLE__
  if (az_sgemm_fn sgemm = az_load_sgemm()) {
    return az_exact_knn_amx_f32(sgemm, n, d, k, x, out_idx, out_dist, n_threads);
  }
#endif
  return az_exact_knn_direct_f32(n, d, k, x, out_idx, out_dist, n_threads);
}

// [[Rcpp::export]]
Rcpp::IntegerMatrix turbo_annoy_build_search(Rcpp::NumericMatrix data, int k, int n_trees, int n_threads) {
  int n = data.nrow();
  int f = data.ncol();

  AnnoyIdx index(f);

  std::vector<float> row(f);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < f; j++) {
      row[j] = static_cast<float>(data(i, j));
    }
    index.add_item(i, row.data());
  }

  index.build(n_trees, -1);

  Rcpp::IntegerMatrix result(n, k);
  int* result_ptr = INTEGER(result);

  std::vector<std::thread> threads;
  for (int t = 0; t < n_threads; t++) {
    int start = (int64_t)t * n / n_threads;
    int end = (int64_t)(t + 1) * n / n_threads;
    threads.emplace_back([&index, result_ptr, n, f, k, start, end, &data]() {
      std::vector<int> neighbors;
      std::vector<float> distances;
      std::vector<float> query(f);
      for (int i = start; i < end; i++) {
        for (int j = 0; j < f; j++) {
          query[j] = static_cast<float>(data(i, j));
        }
        neighbors.clear();
        distances.clear();
        index.get_nns_by_vector(query.data(), k, -1, &neighbors, &distances);
        for (int c = 0; c < k; c++) {
          result_ptr[i + c * n] = neighbors[c] + 1;
        }
      }
    });
  }
  for (auto& th : threads) th.join();

  return result;
}

// [[Rcpp::export]]
Rcpp::IntegerMatrix seurat_exact_knn_f32(Rcpp::NumericMatrix data, int k, int n_threads) {
  const int n = data.nrow();
  const int d = data.ncol();
  if (n <= 0 || d <= 0 || k <= 0 || k > n) {
    Rcpp::stop("invalid exact kNN dimensions");
  }
  std::vector<float> x(static_cast<size_t>(n) * static_cast<size_t>(d));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      x[static_cast<size_t>(i) * static_cast<size_t>(d) + static_cast<size_t>(j)] =
          static_cast<float>(data(i, j));
    }
  }
  std::vector<int32_t> idx(static_cast<size_t>(n) * static_cast<size_t>(k));
  std::vector<float> dist(static_cast<size_t>(n) * static_cast<size_t>(k));
  int rc = az_exact_knn_f32(n, d, k, x.data(), idx.data(), dist.data(), n_threads);
  if (rc != 0) {
    Rcpp::stop("exact kNN failed with code %d", rc);
  }
  Rcpp::IntegerMatrix result(n, k);
  int* result_ptr = INTEGER(result);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      result_ptr[i + j * n] = idx[static_cast<size_t>(i) * static_cast<size_t>(k) +
                                  static_cast<size_t>(j)] + 1;
    }
  }
  return result;
}
