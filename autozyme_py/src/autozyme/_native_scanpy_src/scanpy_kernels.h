#ifndef AUTOZYME_SCANPY_KERNELS_H
#define AUTOZYME_SCANPY_KERNELS_H

#include <stdint.h>

#define SCBLAS_API
#define SCBLAS_UMAP_SMOOTH_K_TOLERANCE 1.0e-5f
#define SCBLAS_UMAP_MIN_K_DIST_SCALE   1.0e-3f

int scblas_knn_descent_f32(int n_cells, int n_dims, int k, const float *X,
                           int n_trees, int n_iters, int leaf_size,
                           uint64_t seed, int32_t *knn_idx,
                           float *knn_dist, int n_threads);

void scblas_umap_smooth_knn_dist_f32(int n_samples, int n_neighbors,
                                     const float *distances, float k,
                                     int n_iter, float local_connectivity,
                                     float bandwidth, float *sigmas,
                                     float *rhos);

void scblas_umap_membership_strengths_f32(int n_samples, int n_neighbors,
                                          const int32_t *knn_indices,
                                          const float *knn_dists,
                                          const float *sigmas,
                                          const float *rhos,
                                          int return_dists, int bipartite,
                                          int32_t *rows, int32_t *cols,
                                          float *vals, float *dists);

void scblas_umap_symmetrize_fuzzy_graph_f32(int n_samples, int n_neighbors,
                                            const int32_t *rows,
                                            const int32_t *cols,
                                            const float *vals,
                                            float set_op_mix_ratio,
                                            int32_t *out_rows,
                                            int32_t *out_cols,
                                            float *out_vals);

void scblas_umap_optimize_layout_euclidean_f32(
    float *embedding, int n_samples, int dim, const int32_t *head,
    const int32_t *tail, int64_t n_edges, const float *epochs_per_sample,
    int n_epochs, float a, float b, float gamma, float initial_alpha,
    float negative_sample_rate, uint64_t seed);

int scblas_umap_optimize_layout_euclidean_f32_parallel(
    float *embedding, int n_samples, int dim, const int32_t *head,
    const int32_t *tail, int64_t n_edges, const float *epochs_per_sample,
    int n_epochs, float a, float b, float gamma, float initial_alpha,
    float negative_sample_rate, uint64_t seed, int n_threads);

#endif
