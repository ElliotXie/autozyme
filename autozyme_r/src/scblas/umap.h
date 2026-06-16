/* umap.h — public API for UMAP graph-prep primitives.
 *
 * Scope: deterministic graph construction helpers from umap-learn, not the
 * stochastic embedding/layout optimizer.
 *
 * Arrays are row-major with shape n_samples × n_neighbors. Distances and
 * membership weights use float32 because umap-learn's numba kernels operate
 * on float32 nearest-neighbor arrays in the Scanpy path.
 */

#ifndef SCBLAS_UMAP_H
#define SCBLAS_UMAP_H

#include <scblas/version.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SCBLAS_UMAP_SMOOTH_K_TOLERANCE 1.0e-5f
#define SCBLAS_UMAP_MIN_K_DIST_SCALE   1.0e-3f

/* Compute UMAP's smooth kNN distance parameters.
 *
 * Matches umap.umap_.smooth_knn_dist semantics:
 *   - distances are sorted per row, with self-distance usually at column 0
 *   - binary search uses columns j=1..n_neighbors-1
 *   - rhos are derived from non-zero distances and local_connectivity
 *   - sigmas are clamped by MIN_K_DIST_SCALE times row/global mean distance
 *
 * No allocation. Caller owns sigmas/rhos, each length n_samples.
 */
SCBLAS_API void scblas_umap_smooth_knn_dist_f32(
    int          n_samples,
    int          n_neighbors,
    const float *distances,
    float        k,
    int          n_iter,
    float        local_connectivity,
    float        bandwidth,
    float       *sigmas,
    float       *rhos
);

/* Opt-in OpenMP variant. Return 0 when threaded, -1 when degraded to the
 * single-threaded path (OpenMP unavailable or fork detected). Output is valid
 * in both cases.
 */
SCBLAS_API int scblas_umap_smooth_knn_dist_f32_parallel(
    int          n_samples,
    int          n_neighbors,
    const float *distances,
    float        k,
    int          n_iter,
    float        local_connectivity,
    float        bandwidth,
    float       *sigmas,
    float       *rhos,
    int          n_threads
);

/* Build UMAP membership-strength COO arrays.
 *
 * Matches umap.umap_.compute_membership_strengths semantics:
 *   rows/cols/vals offset = i * n_neighbors + j
 *   invalid neighbor -1 writes zero row/col/val/dist at that offset
 *   self edge has weight 0 when bipartite == 0
 *   if return_dists != 0, dists must be non-NULL and length n_samples*n_neighbors
 *
 * No allocation. Caller owns all output buffers.
 */
SCBLAS_API void scblas_umap_membership_strengths_f32(
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
);

SCBLAS_API int scblas_umap_membership_strengths_f32_parallel(
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
    float         *dists,
    int            n_threads
);

/* Symmetrize UMAP fuzzy membership strengths.
 *
 * Input rows/cols/vals are the row-major COO arrays produced by
 * scblas_umap_membership_strengths_f32, length n_samples*n_neighbors.
 *
 * For directed weights a = w(i,j), b = w(j,i), output weight is:
 *   set_op_mix_ratio * (a + b - a*b) + (1 - set_op_mix_ratio) * (a*b)
 *
 * Output buffers must have length 2*n_samples*n_neighbors. Slot 2*off stores
 * the symmetrized form of input edge off. Slot 2*off+1 is used only when the
 * reverse edge is absent but the set operation creates a non-zero reverse
 * entry. Unused slots are zero-initialized by this function; callers should
 * eliminate zero entries when building a sparse matrix.
 *
 * No allocation. Assumes each input row has no duplicate non-zero neighbor
 * columns, matching normal kNN output.
 */
SCBLAS_API void scblas_umap_symmetrize_fuzzy_graph_f32(
    int            n_samples,
    int            n_neighbors,
    const int32_t *rows,
    const int32_t *cols,
    const float   *vals,
    float          set_op_mix_ratio,
    int32_t       *out_rows,
    int32_t       *out_cols,
    float         *out_vals
);

SCBLAS_API int scblas_umap_symmetrize_fuzzy_graph_f32_parallel(
    int            n_samples,
    int            n_neighbors,
    const int32_t *rows,
    const int32_t *cols,
    const float   *vals,
    float          set_op_mix_ratio,
    int32_t       *out_rows,
    int32_t       *out_cols,
    float         *out_vals,
    int            n_threads
);

/* ---- UMAP embedding optimizer (layout SGD) ------------------------------ */
/* Unlike the graph-prep helpers above, this is the stochastic layout step.
 *
 * Optimize a UMAP embedding by SGD over the fuzzy 1-skeleton, mirroring
 * umap.layouts.optimize_layout_euclidean for the standard symmetric case:
 * `head` and `tail` index the same `embedding` (length n_samples*dim,
 * row-major) and both endpoints of each attractive edge move. `embedding` is
 * updated in place over n_epochs.
 *
 * Inputs: `epochs_per_sample` (length n_edges, umap make_epochs_per_sample
 * output; edges with value <= 0 are skipped); `a`, `b` the UMAP min-dist fit
 * parameters; `gamma` repulsion weight; `initial_alpha` learning rate;
 * `negative_sample_rate` negatives per positive; `seed` the scalar RNG seed.
 *
 * Determinism: the same inputs and `seed` always produce the same embedding;
 * the negative-sampling RNG is seeded statelessly per (edge, epoch). This is
 * NOT a bit-exact reproduction of umap-learn (float32 + a distinct RNG
 * stream); it is validated by embedding quality (trustworthiness / structure
 * preservation), not coordinate identity.
 *
 * Allocation: unlike the vector kernels, this driver allocates O(n_edges)
 * epoch-bookkeeping scratch (and, in the parallel variant, a frozen-position
 * copy plus per-thread gradient buffers). It is called once per layout, not in
 * a tight loop. See docs/umap_layout.md.
 */
SCBLAS_API void scblas_umap_optimize_layout_euclidean_f32(
    float         *embedding,
    int            n_samples,
    int            dim,
    const int32_t *head,
    const int32_t *tail,
    int64_t        n_edges,
    const float   *epochs_per_sample,
    int            n_epochs,
    float          a,
    float          b,
    float          gamma,
    float          initial_alpha,
    float          negative_sample_rate,
    uint64_t       seed
);

/* Deterministic parallel variant. Uses POSIX threads (NOT OpenMP) so it can run
 * in-process in a Python session that has already initialized numba/umap:
 * scBLAS's libomp deadlocks alongside numba's OpenMP, so OpenMP _parallel
 * kernels must run single-threaded there, whereas this pthreads kernel runs at
 * full thread count in-process (same approach as the find_neighbors SNN kernel).
 *
 * Uses synchronous per-epoch gradient accumulation with sub-epoch mini-batching
 * instead of in-place async updates, so the result is independent of
 * edge-to-thread scheduling and reproducible for a fixed n_threads. This
 * changes SGD dynamics relative to the serial path (Gauss-Seidel -> Jacobi);
 * quality is validated empirically. n_threads <= 0 auto-detects CPUs.
 *
 * Return code: 0 means the multi-threaded layout ran; -1 means it did not and
 * the deterministic serial path was used instead (n_threads resolved to 1,
 * post-fork, or thread/scratch allocation failure) or the call was a no-op on
 * invalid inputs. The embedding is valid and deterministic in all cases.
 * Reproducibility across different n_threads is not guaranteed (float reduction
 * grouping differs); pass a fixed n_threads for a stable embedding.
 */
SCBLAS_API int scblas_umap_optimize_layout_euclidean_f32_parallel(
    float         *embedding,
    int            n_samples,
    int            dim,
    const int32_t *head,
    const int32_t *tail,
    int64_t        n_edges,
    const float   *epochs_per_sample,
    int            n_epochs,
    float          a,
    float          b,
    float          gamma,
    float          initial_alpha,
    float          negative_sample_rate,
    uint64_t       seed,
    int            n_threads
);

#ifdef __cplusplus
}
#endif

#endif /* SCBLAS_UMAP_H */
