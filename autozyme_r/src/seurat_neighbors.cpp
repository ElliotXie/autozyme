// turbo_neighbors.cpp — Parallel Annoy build+search for FindNeighbors
// [[Rcpp::depends(RcppAnnoy)]]
#include <Rcpp.h>
#include "annoylib.h"
#include "kissrandom.h"
#include <thread>
#include <vector>

using namespace Annoy;

typedef AnnoyIndex<int, float, Euclidean, Kiss64Random, AnnoyIndexSingleThreadedBuildPolicy> AnnoyIdx;

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
