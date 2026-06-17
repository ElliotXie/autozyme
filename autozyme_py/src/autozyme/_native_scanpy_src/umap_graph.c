/* umap_graph.c — scalar UMAP graph-prep primitives. */

#include "scanpy_kernels.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static float scblas_umap_mean_all(int n_samples, int n_neighbors,
                                  const float *distances) {
    if (n_samples <= 0 || n_neighbors <= 0 || distances == NULL) return 0.0f;
    double sum = 0.0;
    const long long n = (long long)n_samples * (long long)n_neighbors;
    for (long long i = 0; i < n; ++i) sum += (double)distances[i];
    return (float)(sum / (double)n);
}

static void scblas_umap_smooth_range(
    int start,
    int end,
    int n_neighbors,
    const float *distances,
    float k,
    int n_iter,
    float local_connectivity,
    float bandwidth,
    float mean_distances,
    float *sigmas,
    float *rhos
) {
    const double target = log2((double)k) * (double)bandwidth;
    const int lc_index = (int)floorf(local_connectivity);
    const float lc_interp = local_connectivity - (float)lc_index;

    for (int i = start; i < end; ++i) {
        const float *row = distances + (long long)i * n_neighbors;
        float rho = 0.0f;
        int non_zero_count = 0;
        float max_non_zero = 0.0f;
        float first_non_zero = 0.0f;
        float lc_value = 0.0f;
        float lc_next = 0.0f;
        double row_sum = 0.0;

        for (int j = 0; j < n_neighbors; ++j) {
            const float d = row[j];
            row_sum += (double)d;
            if (d > 0.0f) {
                if (non_zero_count == 0) first_non_zero = d;
                non_zero_count++;
                if (non_zero_count == 0 || d > max_non_zero) max_non_zero = d;
                if (non_zero_count == lc_index) lc_value = d;
                else if (non_zero_count == lc_index + 1) lc_next = d;
            }
        }

        if ((float)non_zero_count >= local_connectivity) {
            if (lc_index > 0) {
                rho = lc_value;
                if (lc_interp > SCBLAS_UMAP_SMOOTH_K_TOLERANCE &&
                    lc_index < non_zero_count) {
                    rho += lc_interp * (lc_next - lc_value);
                }
            } else if (non_zero_count > 0) {
                rho = lc_interp * first_non_zero;
            }
        } else if (non_zero_count > 0) {
            rho = max_non_zero;
        }
        rhos[i] = rho;

        float lo = 0.0f;
        float hi = INFINITY;
        float mid = 1.0f;
        for (int it = 0; it < n_iter; ++it) {
            float psum = 0.0f;
            for (int j = 1; j < n_neighbors; ++j) {
                const float d = row[j] - rho;
                if (d > 0.0f) psum += expf(-(d / mid));
                else psum += 1.0f;
            }

            if (fabs((double)psum - target) < (double)SCBLAS_UMAP_SMOOTH_K_TOLERANCE) break;
            if (psum > target) {
                hi = mid;
                mid = (lo + hi) * 0.5f;
            } else {
                lo = mid;
                if (isinf(hi)) mid *= 2.0f;
                else mid = (lo + hi) * 0.5f;
            }
        }

        sigmas[i] = mid;
        if (rho > 0.0f) {
            const float row_mean = (float)(row_sum / (double)n_neighbors);
            const float min_sigma = SCBLAS_UMAP_MIN_K_DIST_SCALE * row_mean;
            if (sigmas[i] < min_sigma) sigmas[i] = min_sigma;
        } else {
            const float min_sigma = SCBLAS_UMAP_MIN_K_DIST_SCALE * mean_distances;
            if (sigmas[i] < min_sigma) sigmas[i] = min_sigma;
        }
    }
}

void scblas_umap_smooth_knn_dist_f32(
    int          n_samples,
    int          n_neighbors,
    const float *distances,
    float        k,
    int          n_iter,
    float        local_connectivity,
    float        bandwidth,
    float       *sigmas,
    float       *rhos
) {
    if (n_samples <= 0 || n_neighbors <= 0 || distances == NULL ||
        sigmas == NULL || rhos == NULL || k <= 0.0f || n_iter <= 0) {
        return;
    }
    const float mean_distances = scblas_umap_mean_all(n_samples, n_neighbors, distances);
    scblas_umap_smooth_range(0, n_samples, n_neighbors, distances, k, n_iter,
                             local_connectivity, bandwidth, mean_distances,
                             sigmas, rhos);
}

static void scblas_umap_membership_range(
    int start,
    int end,
    int n_neighbors,
    const int32_t *knn_indices,
    const float *knn_dists,
    const float *sigmas,
    const float *rhos,
    int return_dists,
    int bipartite,
    int32_t *rows,
    int32_t *cols,
    float *vals,
    float *dists
) {
    for (int i = start; i < end; ++i) {
        const long long base = (long long)i * n_neighbors;
        const float sigma = sigmas[i];
        const float rho = rhos[i];
        for (int j = 0; j < n_neighbors; ++j) {
            const long long off = base + j;
            const int32_t col = knn_indices[off];
            if (col == -1) {
                rows[off] = 0;
                cols[off] = 0;
                vals[off] = 0.0f;
                if (return_dists && dists != NULL) dists[off] = 0.0f;
                continue;
            }

            float val;
            if (!bipartite && col == i) {
                val = 0.0f;
            } else if ((knn_dists[off] - rho) <= 0.0f || sigma == 0.0f) {
                val = 1.0f;
            } else {
                val = (float)exp((double)(-((knn_dists[off] - rho) / sigma)));
            }
            rows[off] = i;
            cols[off] = col;
            vals[off] = val;
            if (return_dists && dists != NULL) dists[off] = knn_dists[off];
        }
    }
}

void scblas_umap_membership_strengths_f32(
    int            n_samples,
    int            n_neighbors,
    const int32_t *knn_indices,
    const float   *knn_dists,
    const float   *sigmas,
    const float   *rhos,
    int            return_dists,
    int            bipartite,
    int32_t       *rows,
    int32_t       *cols,
    float         *vals,
    float         *dists
) {
    if (n_samples <= 0 || n_neighbors <= 0 || knn_indices == NULL ||
        knn_dists == NULL || sigmas == NULL || rhos == NULL ||
        rows == NULL || cols == NULL || vals == NULL) {
        return;
    }
    if (return_dists && dists == NULL) return;
    scblas_umap_membership_range(0, n_samples, n_neighbors, knn_indices,
                                 knn_dists, sigmas, rhos, return_dists,
                                 bipartite, rows, cols, vals, dists);
}

static float scblas_umap_reverse_value(
    int n_neighbors,
    const int32_t *rows,
    const int32_t *cols,
    const float *vals,
    int src,
    int dst
) {
    const long long base = (long long)dst * n_neighbors;
    for (int j = 0; j < n_neighbors; ++j) {
        const long long off = base + j;
        if (rows[off] == dst && cols[off] == src && vals[off] > 0.0f) {
            return vals[off];
        }
    }
    return 0.0f;
}

static void scblas_umap_symmetrize_range(
    int start,
    int end,
    int n_samples,
    int n_neighbors,
    const int32_t *rows,
    const int32_t *cols,
    const float *vals,
    float set_op_mix_ratio,
    int32_t *out_rows,
    int32_t *out_cols,
    float *out_vals
) {
    for (int i = start; i < end; ++i) {
        const long long base = (long long)i * n_neighbors;
        for (int j = 0; j < n_neighbors; ++j) {
            const long long off = base + j;
            const long long out0 = 2LL * off;
            const long long out1 = out0 + 1;
            out_rows[out0] = 0;
            out_cols[out0] = 0;
            out_vals[out0] = 0.0f;
            out_rows[out1] = 0;
            out_cols[out1] = 0;
            out_vals[out1] = 0.0f;

            const int32_t col = cols[off];
            const float a = vals[off];
            if (a <= 0.0f || col < 0 || col >= n_samples || col == i ||
                rows[off] != i) {
                continue;
            }

            const float b = scblas_umap_reverse_value(n_neighbors, rows, cols,
                                                      vals, i, col);
            const float prod = a * b;
            const float sym = set_op_mix_ratio * (a + b - prod) +
                              (1.0f - set_op_mix_ratio) * prod;
            if (sym <= 0.0f) continue;

            out_rows[out0] = i;
            out_cols[out0] = col;
            out_vals[out0] = sym;

            if (b <= 0.0f) {
                out_rows[out1] = col;
                out_cols[out1] = i;
                out_vals[out1] = sym;
            }
        }
    }
}

void scblas_umap_symmetrize_fuzzy_graph_f32(
    int            n_samples,
    int            n_neighbors,
    const int32_t *rows,
    const int32_t *cols,
    const float   *vals,
    float          set_op_mix_ratio,
    int32_t       *out_rows,
    int32_t       *out_cols,
    float         *out_vals
) {
    if (n_samples <= 0 || n_neighbors <= 0 || rows == NULL || cols == NULL ||
        vals == NULL || out_rows == NULL || out_cols == NULL ||
        out_vals == NULL) {
        return;
    }
    scblas_umap_symmetrize_range(0, n_samples, n_samples, n_neighbors,
                                 rows, cols, vals, set_op_mix_ratio,
                                 out_rows, out_cols, out_vals);
}
