#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace Rcpp;

// [[Rcpp::export]]
NumericVector soupx_expand_corrected_x_cpp(IntegerVector p, IntegerVector row_i,
                                           NumericVector x, NumericMatrix n_soup,
                                           IntegerVector cluster_id,
                                           NumericVector cell_weights) {
  const int n_cells = cluster_id.size();
  const int n_genes = n_soup.nrow();
  const int n_clusters = n_soup.ncol();
  NumericVector out = clone(x);

  std::vector<std::vector<int> > cluster_cells(n_clusters);
  for (int cell = 0; cell < n_cells; ++cell) {
    const int cid = cluster_id[cell] - 1;
    if (cid >= 0 && cid < n_clusters) {
      cluster_cells[cid].push_back(cell);
    }
  }

  std::vector<double> totals(n_genes, 0.0);
  std::vector<int> seen(n_genes, 0);
  std::vector<int> active_genes;
  std::vector<std::vector<int> > gene_entries(n_genes);
  std::vector<std::vector<int> > gene_cells(n_genes);

  for (int cl = 0; cl < n_clusters; ++cl) {
    const int mark = cl + 1;
    active_genes.clear();

    for (int cell : cluster_cells[cl]) {
      for (int ptr = p[cell]; ptr < p[cell + 1]; ++ptr) {
        const int gene = row_i[ptr];
        if (seen[gene] != mark) {
          seen[gene] = mark;
          totals[gene] = 0.0;
          gene_entries[gene].clear();
          gene_cells[gene].clear();
          active_genes.push_back(gene);
        }
        totals[gene] += x[ptr];
        gene_entries[gene].push_back(ptr);
        gene_cells[gene].push_back(cell);
      }
    }

    for (int gene : active_genes) {
      const double target = n_soup(gene, cl);
      const double total = totals[gene];
      const std::vector<int>& entries = gene_entries[gene];
      const std::vector<int>& cells = gene_cells[gene];
      const int m = entries.size();

      if (target <= 0.0) {
        continue;
      }
      if (target >= total) {
        for (int idx = 0; idx < m; ++idx) {
          out[entries[idx]] = 0.0;
        }
        continue;
      }

      double sum_w = 0.0;
      for (int idx = 0; idx < m; ++idx) {
        sum_w += cell_weights[cells[idx]];
      }

      std::vector<double> w(m);
      bool feasible = true;
      for (int idx = 0; idx < m; ++idx) {
        w[idx] = cell_weights[cells[idx]] / sum_w;
        if (target * w[idx] > x[entries[idx]]) {
          feasible = false;
        }
      }

      if (feasible) {
        for (int idx = 0; idx < m; ++idx) {
          out[entries[idx]] = x[entries[idx]] - target * w[idx];
        }
        continue;
      }

      std::vector<int> ord(m);
      std::iota(ord.begin(), ord.end(), 0);
      std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        return (x[entries[a]] / w[a]) < (x[entries[b]] / w[b]);
      });

      std::vector<char> saturated(m, 0);
      double cw = 0.0;
      double cy = 0.0;
      for (int sorted_pos = 0; sorted_pos < m; ++sorted_pos) {
        const int idx = ord[sorted_pos];
        double k = x[entries[idx]] / w[idx] * (1.0 - cw) + cy;
        if (w[idx] == 0.0) {
          k = R_PosInf;
        }
        if (k <= target) {
          saturated[idx] = 1;
        }
        cw += w[idx];
        cy += x[entries[idx]];
      }

      double sum_y_sat = 0.0;
      double sum_w_sat = 0.0;
      for (int idx = 0; idx < m; ++idx) {
        if (saturated[idx]) {
          sum_y_sat += x[entries[idx]];
          sum_w_sat += w[idx];
        }
      }

      const double resid = target - sum_y_sat;
      const double denom = 1.0 - sum_w_sat;
      for (int idx = 0; idx < m; ++idx) {
        double allocated = x[entries[idx]];
        if (!saturated[idx]) {
          allocated = resid * (w[idx] / denom);
        }
        out[entries[idx]] = x[entries[idx]] - allocated;
      }
    }
  }

  return out;
}

// [[Rcpp::export]]
NumericMatrix soupx_cluster_soup_from_cells_cpp(IntegerVector p, IntegerVector row_i,
                                                NumericVector x,
                                                IntegerVector cluster_id,
                                                NumericVector cell_targets,
                                                NumericVector soup_frac,
                                                int n_genes, int n_clusters) {
  NumericMatrix soup(n_genes, n_clusters);
  std::vector<std::vector<int> > cluster_cells(n_clusters);
  std::vector<double> targets(n_clusters, 0.0);
  for (int cell = 0; cell < cluster_id.size(); ++cell) {
    const int cid = cluster_id[cell] - 1;
    if (cid >= 0 && cid < n_clusters) {
      cluster_cells[cid].push_back(cell);
      targets[cid] += cell_targets[cell];
    }
  }

  std::vector<double> weights(n_genes);
  double weight_sum = 0.0;
  for (int gene = 0; gene < n_genes; ++gene) {
    weight_sum += soup_frac[gene];
  }
  std::vector<int> positive_genes;
  positive_genes.reserve(n_genes);
  for (int gene = 0; gene < n_genes; ++gene) {
    weights[gene] = soup_frac[gene] / weight_sum;
    if (weights[gene] > 0.0) {
      positive_genes.push_back(gene);
    }
  }

  std::vector<double> bucket(n_genes, 0.0);
  std::vector<int> touched;

  for (int cl = 0; cl < n_clusters; ++cl) {
    touched.clear();
    for (int cell : cluster_cells[cl]) {
      for (int ptr = p[cell]; ptr < p[cell + 1]; ++ptr) {
        const int gene = row_i[ptr];
        if (bucket[gene] == 0.0) {
          touched.push_back(gene);
        }
        bucket[gene] += x[ptr];
      }
    }

    const double target = targets[cl];
    bool feasible = true;
    for (int gene : positive_genes) {
      if (target * weights[gene] > bucket[gene]) {
        feasible = false;
        break;
      }
    }

    if (feasible) {
      for (int gene : positive_genes) {
        soup(gene, cl) = target * weights[gene];
      }
    } else {
      std::vector<int> ord = positive_genes;
      std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        return (bucket[a] / weights[a]) < (bucket[b] / weights[b]);
      });

      std::vector<char> saturated(n_genes, 0);
      double cw = 0.0;
      double cy = 0.0;
      for (int sorted_pos = 0; sorted_pos < static_cast<int>(ord.size()); ++sorted_pos) {
        const int gene = ord[sorted_pos];
        const double k = bucket[gene] / weights[gene] * (1.0 - cw) + cy;
        if (k <= target) {
          saturated[gene] = 1;
        }
        cw += weights[gene];
        cy += bucket[gene];
      }

      double sum_y_sat = 0.0;
      double sum_w_sat = 0.0;
      for (int gene : positive_genes) {
        if (saturated[gene]) {
          sum_y_sat += bucket[gene];
          sum_w_sat += weights[gene];
        }
      }

      const double resid = target - sum_y_sat;
      const double denom = 1.0 - sum_w_sat;
      for (int gene : positive_genes) {
        if (saturated[gene]) {
          soup(gene, cl) = bucket[gene];
        } else {
          soup(gene, cl) = resid * (weights[gene] / denom);
        }
      }
    }

    for (int gene : touched) {
      bucket[gene] = 0.0;
    }
  }

  return soup;
}

// [[Rcpp::export]]
NumericVector soupx_adjust_counts_no_cluster_x_cpp(IntegerVector p, IntegerVector row_i,
                                                   NumericVector x,
                                                   NumericVector cell_targets,
                                                   NumericVector soup_frac,
                                                   int n_genes) {
  const int n_cells = cell_targets.size();
  NumericVector out = clone(x);

  std::vector<double> weights(n_genes);
  double weight_sum = 0.0;
  for (int gene = 0; gene < n_genes; ++gene) {
    weight_sum += soup_frac[gene];
  }
  std::vector<int> positive_genes;
  positive_genes.reserve(n_genes);
  for (int gene = 0; gene < n_genes; ++gene) {
    weights[gene] = soup_frac[gene] / weight_sum;
    if (weights[gene] > 0.0) {
      positive_genes.push_back(gene);
    }
  }

  std::vector<double> bucket(n_genes, 0.0);
  std::vector<int> touched;

  for (int cell = 0; cell < n_cells; ++cell) {
    touched.clear();
    for (int ptr = p[cell]; ptr < p[cell + 1]; ++ptr) {
      const int gene = row_i[ptr];
      bucket[gene] = x[ptr];
      touched.push_back(gene);
    }

    const double target = cell_targets[cell];
    bool feasible = true;
    for (int gene : positive_genes) {
      if (target * weights[gene] > bucket[gene]) {
        feasible = false;
        break;
      }
    }

    if (feasible) {
      for (int ptr = p[cell]; ptr < p[cell + 1]; ++ptr) {
        const int gene = row_i[ptr];
        out[ptr] = x[ptr] - target * weights[gene];
      }
    } else {
      std::vector<int> ord = positive_genes;
      std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        return (bucket[a] / weights[a]) < (bucket[b] / weights[b]);
      });

      std::vector<char> saturated(n_genes, 0);
      double cw = 0.0;
      double cy = 0.0;
      for (int sorted_pos = 0; sorted_pos < static_cast<int>(ord.size()); ++sorted_pos) {
        const int gene = ord[sorted_pos];
        const double k = bucket[gene] / weights[gene] * (1.0 - cw) + cy;
        if (k <= target) {
          saturated[gene] = 1;
        }
        cw += weights[gene];
        cy += bucket[gene];
      }

      double sum_y_sat = 0.0;
      double sum_w_sat = 0.0;
      for (int gene : positive_genes) {
        if (saturated[gene]) {
          sum_y_sat += bucket[gene];
          sum_w_sat += weights[gene];
        }
      }

      const double resid = target - sum_y_sat;
      const double denom = 1.0 - sum_w_sat;
      for (int ptr = p[cell]; ptr < p[cell + 1]; ++ptr) {
        const int gene = row_i[ptr];
        double allocated = 0.0;
        if (weights[gene] > 0.0) {
          allocated = saturated[gene] ? bucket[gene] : resid * (weights[gene] / denom);
        }
        out[ptr] = x[ptr] - allocated;
      }
    }

    for (int gene : touched) {
      bucket[gene] = 0.0;
    }
  }

  return out;
}
