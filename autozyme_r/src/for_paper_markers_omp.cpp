// kernel_win.cpp — OpenMP-threaded in-process Wilcoxon kernels for the
// Windows path of fast_FindAllMarkers. The dev-platform pipeline used
// parallel::mclapply (fork) for parallelism; Windows has no fork and
// PSOCK has too much startup overhead (~2.6s) to net win on these tiers.
// These kernels do the equivalent work via OpenMP threads within the
// single R process — no subprocess, no IPC, no closure serialization.
//
// Replaces three of the five heaviest steps in the serial all-union path:
//   - presto::nnzeroGroups + presto::sumGroups (combined here)
//   - presto::rank_matrix + presto::compute_ustat + presto::compute_pval
//
// Per-column work is independent; the OMP parallel scheduling is dynamic
// (column workload varies with nonzero density).

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

// [[Rcpp::plugins(openmp)]]

// Compute per-(feature, group) nonzero count AND per-(feature, group) sum
// of expm1(x), in a single OMP-parallel pass.
//
// sparse_mat: dgCMatrix features × cells (column-major; y indexes columns).
// y: 1-indexed cluster id per cell.
// Returns list with two features × n_groups matrices.
//
// [[Rcpp::export]]
List omp_sum_nnz_expm1_groups(S4 sparse_mat, IntegerVector y, int n_groups,
                              int n_threads = 0) {
    IntegerVector p     = sparse_mat.slot("p");
    IntegerVector i_idx = sparse_mat.slot("i");
    NumericVector x     = sparse_mat.slot("x");
    IntegerVector dim   = sparse_mat.slot("Dim");
    int n_rows = dim[0];  // features
    int n_cols = dim[1];  // cells

#ifdef _OPENMP
    if (n_threads <= 0) n_threads = omp_get_max_threads();
#endif

    std::vector<std::vector<int>> cells_per_group(n_groups);
    for (int g = 0; g < n_groups; ++g) cells_per_group[g].reserve(n_cols / n_groups + 1);
    for (int c = 0; c < n_cols; ++c) {
        int g = y[c] - 1;
        if (g >= 0 && g < n_groups) cells_per_group[g].push_back(c);
    }

    NumericMatrix nnz_mat(n_rows, n_groups);
    NumericMatrix sum_mat(n_rows, n_groups);

    #pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
    for (int g = 0; g < n_groups; ++g) {
        double* nnz_col = &nnz_mat(0, g);
        double* sum_col = &sum_mat(0, g);
        const std::vector<int>& cells = cells_per_group[g];
        for (size_t cc = 0; cc < cells.size(); ++cc) {
            int c = cells[cc];
            int s = p[c];
            int e = p[c + 1];
            for (int idx = s; idx < e; ++idx) {
                int feat = i_idx[idx];
                nnz_col[feat] += 1.0;
                sum_col[feat] += std::expm1(x[idx]);
            }
        }
    }

    return List::create(Named("nnz") = nnz_mat, Named("sum_expm1") = sum_mat);
}


// Vectorize the per-(feature, group) pct/lfc/pass computation that the R-level
// `sweep` chain performs after the omp_sum_nnz_expm1_groups call. R takes
// 0.3-0.4s here on tiny because of object dispatch per sweep; this is plain
// loops at memory bandwidth speed in C++.
//
// [[Rcpp::export]]
List omp_filter_pct_lfc(
    NumericMatrix nnz_mat,       // n_features × n_groups
    NumericMatrix expm1_sums,    // n_features × n_groups
    NumericVector total_nnz,     // n_features
    NumericVector total_expm1,   // n_features
    NumericVector cluster_sizes, // n_groups
    NumericVector sizes_rest,    // n_groups
    LogicalVector valid_mask,    // n_groups
    double min_pct, double min_diff_pct,
    double logfc_threshold, bool only_pos, double log_base,
    int n_threads = 0) {
    int n_features = nnz_mat.nrow();
    int n_groups   = nnz_mat.ncol();

#ifdef _OPENMP
    if (n_threads <= 0) n_threads = omp_get_max_threads();
#endif

    NumericMatrix pct1_mat(n_features, n_groups);
    NumericMatrix pct2_mat(n_features, n_groups);
    NumericMatrix lfc_mat(n_features, n_groups);
    LogicalMatrix pass_mat(n_features, n_groups);
    LogicalVector row_any(n_features);
    double inv_log_base = 1.0 / std::log(log_base);

    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int g = 0; g < n_groups; ++g) {
        double cs = cluster_sizes[g];
        double sr = sizes_rest[g];
        bool   valid = valid_mask[g];
        for (int f = 0; f < n_features; ++f) {
            double nnz_g  = nnz_mat(f, g);
            double exp_g  = expm1_sums(f, g);
            double pct1   = std::round((nnz_g / cs) * 1000.0) / 1000.0;
            double pct2   = std::round(((total_nnz[f] - nnz_g) / sr) * 1000.0) / 1000.0;
            double m1     = std::log((exp_g + 1.0) / cs) * inv_log_base;
            double m2     = std::log(((total_expm1[f] - exp_g) + 1.0) / sr) * inv_log_base;
            double lfc    = m1 - m2;
            double a_max  = std::max(pct1, pct2);
            double a_diff = a_max - std::min(pct1, pct2);
            bool pass = valid && (a_max >= min_pct) && (a_diff >= min_diff_pct);
            if (pass) {
                if (only_pos) pass = (lfc >= logfc_threshold);
                else          pass = (std::fabs(lfc) >= logfc_threshold);
            }
            pct1_mat(f, g) = pct1;
            pct2_mat(f, g) = pct2;
            lfc_mat(f, g)  = lfc;
            pass_mat(f, g) = pass;
        }
    }

    // row_any: did any group pass for this feature? Reduction over groups.
    #pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int f = 0; f < n_features; ++f) {
        bool any_pass = false;
        for (int g = 0; g < n_groups; ++g) {
            if (pass_mat(f, g)) { any_pass = true; break; }
        }
        row_any[f] = any_pass;
    }

    return List::create(Named("pct1") = pct1_mat,
                        Named("pct2") = pct2_mat,
                        Named("lfc")  = lfc_mat,
                        Named("pass") = pass_mat,
                        Named("row_any") = row_any);
}


// Combined: select a row-subset of `data_layer` (features × cells, dgCMatrix
// column-major) by feature index, transpose into row-major per-feature
// (cell, value) buffers via a CSR transpose pass, then per-feature compute
// rank / Wilcoxon U / continuity-corrected p-value in OMP-parallel. Skips
// constructing an intermediate dgCMatrix entirely — Matrix::t() on the
// 30k × 32k input was 0.4-0.6s and was the next bottleneck after the
// presto kernels were OMP-replaced.
//
// data_layer: dgCMatrix features × cells.
// union_idx: 1-indexed feature indices to keep.
// y: 1-indexed cluster id per cell.
// Returns pval matrix |union_idx| × n_groups.
//
// [[Rcpp::export]]
NumericMatrix omp_subset_transpose_rank_pval(
    S4 data_layer, IntegerVector union_idx, IntegerVector y,
    int n_groups, NumericVector cluster_sizes, int n_threads = 0) {
    IntegerVector p     = data_layer.slot("p");
    IntegerVector i_idx = data_layer.slot("i");
    NumericVector x     = data_layer.slot("x");
    IntegerVector dim   = data_layer.slot("Dim");
    int n_features = dim[0];
    int n_cells    = dim[1];
    int n_union    = union_idx.size();
    double n_total = (double)n_cells;

#ifdef _OPENMP
    if (n_threads <= 0) n_threads = omp_get_max_threads();
#endif

    // Build feature → output-column lookup; -1 means "skip".
    std::vector<int> feat_to_col(n_features, -1);
    for (int j = 0; j < n_union; ++j) {
        feat_to_col[union_idx[j] - 1] = j;
    }

    // Pass 1: count nonzeros per output column. Serial — pure O(nnz) scan.
    std::vector<int> col_nnz(n_union, 0);
    int total_x = x.size();
    for (int idx = 0; idx < total_x; ++idx) {
        int col_pos = feat_to_col[i_idx[idx]];
        if (col_pos >= 0) col_nnz[col_pos]++;
    }

    // Cumulative offsets.
    std::vector<int> col_off(n_union + 1, 0);
    for (int j = 0; j < n_union; ++j) col_off[j + 1] = col_off[j] + col_nnz[j];
    int out_nnz = col_off[n_union];

    // Pass 2: scatter (cell_idx, value) into per-feature contiguous buffers.
    // Reuse col_nnz as the per-feature write cursor (reset to 0).
    std::vector<int>    out_cell(out_nnz);
    std::vector<double> out_val(out_nnz);
    std::fill(col_nnz.begin(), col_nnz.end(), 0);
    for (int c = 0; c < n_cells; ++c) {
        int s = p[c], e = p[c + 1];
        for (int idx = s; idx < e; ++idx) {
            int col_pos = feat_to_col[i_idx[idx]];
            if (col_pos >= 0) {
                int w = col_off[col_pos] + col_nnz[col_pos]++;
                out_cell[w] = c;
                out_val[w]  = x[idx];
            }
        }
    }

    // Pass 3: per-feature rank + Wilcoxon U + p-value, OMP-parallel.
    NumericMatrix pval(n_union, n_groups);
    double denom_var = 12.0 * n_total * (n_total - 1.0);

    #pragma omp parallel num_threads(n_threads)
    {
        std::vector<std::pair<double, int>> col_vals;
        std::vector<double> rank_sum(n_groups);

        #pragma omp for schedule(dynamic, 16)
        for (int j = 0; j < n_union; ++j) {
            int s = col_off[j], e = col_off[j + 1];
            int n_nz = e - s;
            int n_zeros = n_cells - n_nz;
            double zero_avg_rank = (n_zeros + 1) / 2.0;

            for (int g = 0; g < n_groups; ++g) {
                rank_sum[g] = cluster_sizes[g] * zero_avg_rank;
            }
            double tie_sum = 0.0;
            if (n_zeros >= 2)
                tie_sum += (double)n_zeros * n_zeros * n_zeros - (double)n_zeros;

            col_vals.clear();
            col_vals.reserve(n_nz);
            for (int idx = s; idx < e; ++idx) {
                col_vals.push_back({out_val[idx], out_cell[idx]});
            }
            std::sort(col_vals.begin(), col_vals.end());

            int jj = 0;
            while (jj < n_nz) {
                int k = jj;
                while (k < n_nz && col_vals[k].first == col_vals[jj].first) k++;
                int run_len = k - jj;
                double avg_rank = (double)n_zeros + (double)jj + (double)(run_len + 1) / 2.0;
                if (run_len >= 2)
                    tie_sum += (double)run_len * run_len * run_len - (double)run_len;
                for (int m = jj; m < k; ++m) {
                    int cell_idx = col_vals[m].second;
                    int g = y[cell_idx] - 1;
                    rank_sum[g] += (avg_rank - zero_avg_rank);
                }
                jj = k;
            }

            for (int g = 0; g < n_groups; ++g) {
                double n1 = cluster_sizes[g];
                double n2 = n_total - n1;
                if (n1 < 1.0 || n2 < 1.0) { pval(j, g) = 1.0; continue; }
                double R_g = rank_sum[g];
                double U = R_g - n1 * (n1 + 1.0) / 2.0;
                double mean_U = n1 * n2 / 2.0;
                double tie_corr = (n1 * n2 * tie_sum) / denom_var;
                double var_U = n1 * n2 * (n_total + 1.0) / 12.0 - tie_corr;
                if (var_U <= 0.0) { pval(j, g) = 1.0; continue; }
                double delta = std::fabs(U - mean_U) - 0.5;
                if (delta < 0.0) delta = 0.0;
                double z = delta / std::sqrt(var_U);
                pval(j, g) = 2.0 * R::pnorm(-z, 0.0, 1.0, 1, 0);
            }
        }
    }

    return pval;
}


// Fused: rank values per column (handling ties via average rank), compute
// Wilcoxon U statistic per cluster, compute two-sided p-value with tie
// correction and continuity correction. Matches presto's rank_matrix +
// compute_ustat + compute_pval composition.
//
// sparse_mat: dgCMatrix cells × features (i.e. t(data_layer)[, union_idx]).
// y: 1-indexed cluster id per cell.
// cluster_sizes: length n_groups; n_cells in each cluster.
// Returns pval matrix n_features × n_groups.
//
// [[Rcpp::export]]
NumericMatrix omp_rank_ustat_pval(S4 sparse_mat, IntegerVector y, int n_groups,
                                  NumericVector cluster_sizes,
                                  int n_threads = 0) {
    IntegerVector p     = sparse_mat.slot("p");
    IntegerVector i_idx = sparse_mat.slot("i");
    NumericVector x     = sparse_mat.slot("x");
    IntegerVector dim   = sparse_mat.slot("Dim");
    int n_rows = dim[0];  // cells
    int n_cols = dim[1];  // features
    double n_total = (double)n_rows;

#ifdef _OPENMP
    if (n_threads <= 0) n_threads = omp_get_max_threads();
#endif

    NumericMatrix pval(n_cols, n_groups);

    #pragma omp parallel num_threads(n_threads)
    {
        std::vector<std::pair<double, int>> col_vals;
        std::vector<double> rank_sum(n_groups);

        #pragma omp for schedule(dynamic, 16)
        for (int c = 0; c < n_cols; ++c) {
            int s = p[c];
            int e = p[c + 1];
            int n_nz = e - s;
            int n_zeros = n_rows - n_nz;
            double zero_avg_rank = (n_zeros + 1) / 2.0;

            // Initialize rank_sum with zero-cell contribution per group.
            for (int g = 0; g < n_groups; ++g) {
                rank_sum[g] = cluster_sizes[g] * zero_avg_rank;
            }

            double tie_sum = 0.0;
            if (n_zeros >= 2)
                tie_sum += (double)n_zeros * n_zeros * n_zeros - (double)n_zeros;

            // Collect & sort non-zeros.
            col_vals.clear();
            col_vals.reserve(n_nz);
            for (int idx = s; idx < e; ++idx) {
                col_vals.push_back({x[idx], i_idx[idx]});
            }
            std::sort(col_vals.begin(), col_vals.end());

            // Walk runs of equal values, assign average rank, replace each
            // non-zero cell's zero-rank contribution with its actual rank.
            int j = 0;
            while (j < n_nz) {
                int k = j;
                while (k < n_nz && col_vals[k].first == col_vals[j].first) k++;
                int run_len = k - j;
                double avg_rank = (double)n_zeros + (double)j + (double)(run_len + 1) / 2.0;
                if (run_len >= 2)
                    tie_sum += (double)run_len * run_len * run_len - (double)run_len;
                for (int m = j; m < k; ++m) {
                    int cell_idx = col_vals[m].second;
                    int g = y[cell_idx] - 1;
                    rank_sum[g] += (avg_rank - zero_avg_rank);
                }
                j = k;
            }

            // Per-group: U statistic, mean/var with tie correction,
            // continuity-corrected z, two-sided p from R's pnorm.
            double denom_var = 12.0 * n_total * (n_total - 1.0);
            for (int g = 0; g < n_groups; ++g) {
                double n1 = cluster_sizes[g];
                double n2 = n_total - n1;
                if (n1 < 1.0 || n2 < 1.0) { pval(c, g) = 1.0; continue; }
                double R_g = rank_sum[g];
                double U = R_g - n1 * (n1 + 1.0) / 2.0;
                double mean_U = n1 * n2 / 2.0;
                double tie_corr = (n1 * n2 * tie_sum) / denom_var;
                double var_U = n1 * n2 * (n_total + 1.0) / 12.0 - tie_corr;
                if (var_U <= 0.0) { pval(c, g) = 1.0; continue; }
                double delta = std::fabs(U - mean_U) - 0.5;
                if (delta < 0.0) delta = 0.0;
                double z = delta / std::sqrt(var_U);
                double p_val = 2.0 * R::pnorm(-z, 0.0, 1.0, 1, 0);
                pval(c, g) = p_val;
            }
        }
    }

    return pval;
}
