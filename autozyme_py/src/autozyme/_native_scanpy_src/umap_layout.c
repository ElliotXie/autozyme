/* umap_layout.c — serial UMAP embedding optimizer (SGD over the fuzzy graph).
 *
 * Faithful port of umap.layouts.optimize_layout_euclidean for the standard
 * symmetric case (head and tail index the same embedding, both endpoints of
 * each attractive edge move). In-place asynchronous updates, exactly like
 * umap-learn's serial path, so this is deterministic given (seed) and matches
 * umap serial in embedding quality.
 *
 * This kernel allocates O(n_edges) scratch internally (the per-edge epoch
 * bookkeeping umap also keeps); it is a once-per-run driver, not a tight
 * inner primitive, so the no-allocation rule that applies to the vector
 * kernels is relaxed here. See docs/umap_layout.md.
 */

#include "scanpy_kernels.h"
#include "umap_layout_internal.h"
#include <stdlib.h>

void scblas_umap_optimize_layout_euclidean_f32(
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
) {
    if (embedding == 0 || head == 0 || tail == 0 || epochs_per_sample == 0)
        return;
    if (n_samples <= 0 || dim <= 0 || n_edges <= 0 || n_epochs <= 0)
        return;
    if (negative_sample_rate <= 0.0f) return;

    const int n_vertices = n_samples;

    /* Per-edge epoch bookkeeping, mirroring umap. */
    double *eons = (double *)malloc((size_t)n_edges * sizeof(double));
    double *eonn = (double *)malloc((size_t)n_edges * sizeof(double));
    double *epn  = (double *)malloc((size_t)n_edges * sizeof(double));
    if (eons == 0 || eonn == 0 || epn == 0) {
        free(eons); free(eonn); free(epn);
        return;
    }
    for (int64_t i = 0; i < n_edges; ++i) {
        double eps = (double)epochs_per_sample[i];
        epn[i]  = eps / (double)negative_sample_rate;
        eons[i] = eps;
        eonn[i] = epn[i];
    }

    float alpha = initial_alpha;
    for (int n = 0; n < n_epochs; ++n) {
        for (int64_t i = 0; i < n_edges; ++i) {
            if (epochs_per_sample[i] <= 0.0f) continue; /* inactive edge */
            if (eons[i] > (double)n) continue;

            const int32_t j = head[i];
            const int32_t k = tail[i];
            float *cur = embedding + (int64_t)j * dim;
            float *oth = embedding + (int64_t)k * dim;

            float d2 = scblas_umap_rdist(cur, oth, dim);
            float gc = scblas_umap_attractive_coeff(d2, a, b);
            for (int d = 0; d < dim; ++d) {
                float grad_d = scblas_umap_clip(gc * (cur[d] - oth[d]));
                cur[d] += grad_d * alpha;
                oth[d] -= grad_d * alpha; /* move_other */
            }

            eons[i] += (double)epochs_per_sample[i];

            int n_neg = (int)(((double)n - eonn[i]) / epn[i]);
            if (n_neg > 0) {
                uint32_t rng[3];
                scblas_umap_seed_rng(seed, i, n, rng);
                for (int p = 0; p < n_neg; ++p) {
                    int32_t k2 = (int32_t)(scblas_umap_tau_rand(rng) % (uint32_t)n_vertices);
                    float *oth2 = embedding + (int64_t)k2 * dim;
                    float dd2 = scblas_umap_rdist(cur, oth2, dim);
                    if (dd2 <= 0.0f && j == k2) continue;
                    float gcn = scblas_umap_repulsive_coeff(dd2, a, b, gamma);
                    for (int d = 0; d < dim; ++d) {
                        float grad_d = (gcn > 0.0f)
                            ? scblas_umap_clip(gcn * (cur[d] - oth2[d]))
                            : 0.0f;
                        cur[d] += grad_d * alpha;
                    }
                }
                eonn[i] += (double)n_neg * epn[i];
            }
        }
        alpha = initial_alpha * (1.0f - (float)n / (float)n_epochs);
    }

    free(eons);
    free(eonn);
    free(epn);
}
