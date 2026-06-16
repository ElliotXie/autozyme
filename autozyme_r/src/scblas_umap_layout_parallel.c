/* umap_layout_parallel.c — deterministic pthreads UMAP embedding optimizer.
 *
 * Threading model: POSIX threads, NOT OpenMP. This is deliberate. scBLAS links
 * its own libomp; numba/umap link another OpenMP runtime, and running an scBLAS
 * OpenMP region in a process that has already initialized numba hangs on the
 * macOS/conda stack (the umap_graph/leiden_graph OMP kernels must run at 1
 * thread for that reason). The find_neighbors SNN kernel sidesteps this with
 * pthreads and runs in-process at many threads alongside numba; this kernel
 * follows the same pattern so the UMAP layout can be called in-process from a
 * Scanpy/numba session without isolation. See docs/umap_layout.md.
 *
 * Algorithm (unchanged from the OpenMP version it replaces): synchronous
 * per-epoch gradient accumulation (Jacobi) with sub-epoch mini-batching to
 * avoid overshoot, per-thread gradient buffers reduced in fixed (vertex,
 * thread) order, and a stateless per-(edge, epoch) RNG. Deterministic for a
 * fixed n_threads, independent of timing. Each edge is owned by exactly one
 * thread for the whole run (stable strided partition), so the per-edge epoch
 * bookkeeping needs no synchronization.
 */

#include <scblas/umap.h>
#include "umap_layout_internal.h"
int scblas_in_loader_process(void);
/* POSIX (not pure C99), by design: pthreads for in-process-with-numba
 * threading and sysconf for CPU detection, matching the find_neighbors SNN
 * kernel. The build targets POSIX platforms (macOS/Linux). */
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define SCBLAS_UMAP_SUBSTEP_CAP 64

static long scblas_umap_online_cpus(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (si.dwNumberOfProcessors > 0) ? (long)si.dwNumberOfProcessors : 1L;
#else
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    return (ncpu > 0) ? ncpu : 1L;
#endif
}

/* ---- portable counting barrier (macOS lacks pthread_barrier_t) ---------- */
typedef struct {
    pthread_mutex_t mtx;
    pthread_cond_t  cond;
    int             count;
    int             generation;
    int             n_threads;
    int             aborted;       /* set on partial-spawn failure */
} scblas_barrier_t;

static void scblas_barrier_init(scblas_barrier_t *b, int n) {
    pthread_mutex_init(&b->mtx, 0);
    pthread_cond_init(&b->cond, 0);
    b->count = 0;
    b->generation = 0;
    b->n_threads = n;
    b->aborted = 0;
}
static void scblas_barrier_destroy(scblas_barrier_t *b) {
    pthread_mutex_destroy(&b->mtx);
    pthread_cond_destroy(&b->cond);
}
/* Returns 1 if the barrier was aborted (caller should exit), else 0. Abort is
 * sticky, so a thread that arrives at the barrier after the abort still sees
 * it and never blocks. */
static int scblas_barrier_wait(scblas_barrier_t *b) {
    pthread_mutex_lock(&b->mtx);
    if (b->aborted) { pthread_mutex_unlock(&b->mtx); return 1; }
    const int gen = b->generation;
    if (++b->count == b->n_threads) {
        b->generation++;
        b->count = 0;
        pthread_cond_broadcast(&b->cond);
    } else {
        while (gen == b->generation && !b->aborted) {
            pthread_cond_wait(&b->cond, &b->mtx);
        }
    }
    const int aborted = b->aborted;
    pthread_mutex_unlock(&b->mtx);
    return aborted;
}
/* Used only on a partial-spawn failure: wake every waiter so the already-
 * started threads exit instead of deadlocking on a thread count that will
 * never arrive. */
static void scblas_barrier_abort(scblas_barrier_t *b) {
    pthread_mutex_lock(&b->mtx);
    b->aborted = 1;
    pthread_cond_broadcast(&b->cond);
    pthread_mutex_unlock(&b->mtx);
}

static int resolve_substeps(int64_t n_edges, int n_vertices) {
    const char *env = getenv("SCBLAS_UMAP_SUBSTEPS");
    if (env != 0) {
        long v = strtol(env, 0, 10);
        if (v >= 1) return (v > (long)n_edges) ? (int)n_edges : (int)v;
    }
    if (n_vertices <= 0) return 1;
    long s = (long)((n_edges + n_vertices - 1) / n_vertices); /* ceil avg degree */
    if (s < 1) s = 1;
    if (s > SCBLAS_UMAP_SUBSTEP_CAP) s = SCBLAS_UMAP_SUBSTEP_CAP;
    if (s > n_edges) s = (int)n_edges;
    return (int)s;
}

typedef struct {
    int            tid;
    int            n_threads;
    float         *embedding;
    int            dim;
    int            n_vertices;
    const int32_t *head;
    const int32_t *tail;
    int64_t        n_edges;
    const float   *epochs_per_sample;
    double        *eons;
    double        *eonn;
    double        *epn;
    float         *gbuf;            /* n_threads * emb_len */
    size_t         emb_len;
    int            n_epochs;
    int            n_substeps;
    float          a, b, gamma, initial_alpha, negative_sample_rate;
    uint64_t       seed;
    scblas_barrier_t *barrier;
} layout_worker_args;

static void *layout_worker(void *raw) {
    layout_worker_args *w = (layout_worker_args *)raw;
    const int T = w->n_threads;
    const int t = w->tid;
    const int dim = w->dim;
    const int n_vertices = w->n_vertices;
    const int64_t S = w->n_substeps;
    const size_t emb_len = w->emb_len;
    float *g = w->gbuf + (size_t)t * emb_len;

    /* contiguous vertex slice for the reduce/apply phase */
    const size_t v_lo = (emb_len * (size_t)t) / (size_t)T;
    const size_t v_hi = (emb_len * (size_t)(t + 1)) / (size_t)T;

    for (int n = 0; n < w->n_epochs; ++n) {
        /* match the serial/umap alpha schedule: epoch 0 uses initial_alpha,
         * epoch n>=1 uses initial_alpha * (1 - (n-1)/n_epochs). */
        const float alpha = (n == 0)
            ? w->initial_alpha
            : w->initial_alpha * (1.0f - (float)(n - 1) / (float)w->n_epochs);

        for (int64_t s = 0; s < S; ++s) {
            memset(g, 0, emb_len * sizeof(float));

            /* edges in substep s are i = s + k*S, k = 0..K-1; this thread owns
             * a contiguous slice of k -> edge i is always handled by the same
             * thread, so eons/eonn writes are race-free. */
            const int64_t K = (w->n_edges > s) ? ((w->n_edges - s + S - 1) / S) : 0;
            const int64_t k_lo = (K * (int64_t)t) / T;
            const int64_t k_hi = (K * (int64_t)(t + 1)) / T;

            for (int64_t k = k_lo; k < k_hi; ++k) {
                const int64_t i = s + k * S;
                if (w->epochs_per_sample[i] <= 0.0f) continue;
                if (w->eons[i] > (double)n) continue;

                const int32_t j = w->head[i];
                const int32_t kk = w->tail[i];
                const float *cur = w->embedding + (int64_t)j * dim;
                const float *oth = w->embedding + (int64_t)kk * dim;

                float d2 = scblas_umap_rdist(cur, oth, dim);
                float gc = scblas_umap_attractive_coeff(d2, w->a, w->b);
                float *gj = g + (int64_t)j * dim;
                float *gk = g + (int64_t)kk * dim;
                for (int d = 0; d < dim; ++d) {
                    float grad_d = scblas_umap_clip(gc * (cur[d] - oth[d]));
                    gj[d] += grad_d;
                    gk[d] -= grad_d; /* move_other */
                }

                w->eons[i] += (double)w->epochs_per_sample[i];

                int n_neg = (int)(((double)n - w->eonn[i]) / w->epn[i]);
                if (n_neg > 0) {
                    uint32_t rng[3];
                    scblas_umap_seed_rng(w->seed, i, n, rng);
                    for (int p = 0; p < n_neg; ++p) {
                        int32_t k2 = (int32_t)(scblas_umap_tau_rand(rng) % (uint32_t)n_vertices);
                        const float *oth2 = w->embedding + (int64_t)k2 * dim;
                        float dd2 = scblas_umap_rdist(cur, oth2, dim);
                        if (dd2 <= 0.0f && j == k2) continue;
                        float gcn = scblas_umap_repulsive_coeff(dd2, w->a, w->b, w->gamma);
                        if (gcn > 0.0f) {
                            for (int d = 0; d < dim; ++d) {
                                gj[d] += scblas_umap_clip(gcn * (cur[d] - oth2[d]));
                            }
                        }
                    }
                    w->eonn[i] += (double)n_neg * w->epn[i];
                }
            }

            /* all gradients accumulated; exit before touching embedding on abort */
            if (scblas_barrier_wait(w->barrier)) return 0;

            /* reduce per-thread buffers in fixed thread order and apply, over
             * this thread's disjoint vertex slice -> deterministic, race-free */
            for (size_t v = v_lo; v < v_hi; ++v) {
                float acc = 0.0f;
                for (int tt = 0; tt < T; ++tt) {
                    acc += w->gbuf[(size_t)tt * emb_len + v];
                }
                w->embedding[v] += alpha * acc;
            }

            if (scblas_barrier_wait(w->barrier)) return 0; /* updated before next substep */
        }
    }
    return 0;
}

int scblas_umap_optimize_layout_euclidean_f32_parallel(
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
) {
    /* Return-code contract (see include/scblas/umap.h): 0 means the
     * multi-threaded layout ran; -1 means it did not (invalid/no-op inputs,
     * n_threads resolved to 1, post-fork, or allocation/spawn failure), in
     * which case the deterministic serial path is used or the call no-ops. */
    if (embedding == 0 || head == 0 || tail == 0 || epochs_per_sample == 0)
        return -1;
    if (n_samples <= 0 || dim <= 0 || n_edges <= 0 || n_epochs <= 0)
        return -1;
    if (negative_sample_rate <= 0.0f) return -1;

    /* Post-fork safety: a forked worker must not spin up threads against state
     * inherited from the parent; degrade to the deterministic serial path. */
    if (!scblas_in_loader_process()) {
        scblas_umap_optimize_layout_euclidean_f32(
            embedding, n_samples, dim, head, tail, n_edges, epochs_per_sample,
            n_epochs, a, b, gamma, initial_alpha, negative_sample_rate, seed);
        return -1;
    }

    int eff_threads = n_threads;
    if (eff_threads <= 0) {
        /* _SC_NPROCESSORS_ONLN is POSIX; sysconf returns -1 when unsupported or
         * on error, which the (ncpu > 0) test maps to a safe 1-thread default. */
        long ncpu = scblas_umap_online_cpus();
        eff_threads = (ncpu > 0) ? (int)ncpu : 1;
    }
    if (eff_threads < 1) eff_threads = 1;
    if ((int64_t)eff_threads > n_edges) eff_threads = (int)n_edges;

    if (eff_threads == 1) {
        /* No threading needed; delegate to the serial path (the same algorithm
         * without the Jacobi sub-epoch split, and the deterministic reference).
         * Return -1: the multi-threaded path did not run. */
        scblas_umap_optimize_layout_euclidean_f32(
            embedding, n_samples, dim, head, tail, n_edges, epochs_per_sample,
            n_epochs, a, b, gamma, initial_alpha, negative_sample_rate, seed);
        return -1;
    }

    const int n_vertices = n_samples;
    const int n_substeps = resolve_substeps(n_edges, n_vertices);
    const size_t emb_len = (size_t)n_vertices * (size_t)dim;

    float  *gbuf = (float *)malloc((size_t)eff_threads * emb_len * sizeof(float));
    double *eons = (double *)malloc((size_t)n_edges * sizeof(double));
    double *eonn = (double *)malloc((size_t)n_edges * sizeof(double));
    double *epn  = (double *)malloc((size_t)n_edges * sizeof(double));
    pthread_t *threads = (pthread_t *)malloc((size_t)eff_threads * sizeof(pthread_t));
    layout_worker_args *args =
        (layout_worker_args *)malloc((size_t)eff_threads * sizeof(layout_worker_args));
    if (gbuf == 0 || eons == 0 || eonn == 0 || epn == 0 || threads == 0 || args == 0) {
        free(gbuf); free(eons); free(eonn); free(epn); free(threads); free(args);
        scblas_umap_optimize_layout_euclidean_f32(
            embedding, n_samples, dim, head, tail, n_edges, epochs_per_sample,
            n_epochs, a, b, gamma, initial_alpha, negative_sample_rate, seed);
        return -1;
    }

    for (int64_t i = 0; i < n_edges; ++i) {
        double eps = (double)epochs_per_sample[i];
        epn[i]  = eps / (double)negative_sample_rate;
        eons[i] = eps;
        eonn[i] = epn[i];
    }

    scblas_barrier_t barrier;
    scblas_barrier_init(&barrier, eff_threads);

    int spawned = 0;
    int spawn_failed = 0;
    for (int t = 0; t < eff_threads; ++t) {
        args[t].tid = t;
        args[t].n_threads = eff_threads;
        args[t].embedding = embedding;
        args[t].dim = dim;
        args[t].n_vertices = n_vertices;
        args[t].head = head;
        args[t].tail = tail;
        args[t].n_edges = n_edges;
        args[t].epochs_per_sample = epochs_per_sample;
        args[t].eons = eons;
        args[t].eonn = eonn;
        args[t].epn = epn;
        args[t].gbuf = gbuf;
        args[t].emb_len = emb_len;
        args[t].n_epochs = n_epochs;
        args[t].n_substeps = n_substeps;
        args[t].a = a;
        args[t].b = b;
        args[t].gamma = gamma;
        args[t].initial_alpha = initial_alpha;
        args[t].negative_sample_rate = negative_sample_rate;
        args[t].seed = seed;
        args[t].barrier = &barrier;
        if (pthread_create(&threads[t], 0, layout_worker, &args[t]) != 0) {
            spawn_failed = 1;
            break;
        }
        spawned++;
    }

    if (spawn_failed) {
        /* The full thread count will never reach the barrier. Signal abort and
         * release the spawned workers; each exits at the post-accumulate
         * barrier BEFORE writing the embedding, so the input is untouched and a
         * serial fallback is correct. pthread_create failure is OOM-rare. */
        scblas_barrier_abort(&barrier);
        for (int t = 0; t < spawned; ++t) pthread_join(threads[t], 0);
        scblas_barrier_destroy(&barrier);
        free(gbuf); free(eons); free(eonn); free(epn); free(threads); free(args);
        scblas_umap_optimize_layout_euclidean_f32(
            embedding, n_samples, dim, head, tail, n_edges, epochs_per_sample,
            n_epochs, a, b, gamma, initial_alpha, negative_sample_rate, seed);
        return -1;
    }

    for (int t = 0; t < eff_threads; ++t) {
        pthread_join(threads[t], 0);
    }

    scblas_barrier_destroy(&barrier);
    free(gbuf);
    free(eons);
    free(eonn);
    free(epn);
    free(threads);
    free(args);
    return 0;
}
