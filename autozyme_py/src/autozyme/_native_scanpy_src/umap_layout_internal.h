/* umap_layout_internal.h — PRIVATE inline helpers shared by the serial and
 * OpenMP UMAP layout/SGD translation units. Not installed; not public ABI.
 *
 * Math mirrors umap.layouts.optimize_layout_euclidean (the attractive/
 * repulsive gradient of the fuzzy-set cross-entropy used by UMAP SGD).
 *
 * RNG: a 3-component LFSR113-style tausworthe generator. umap.utils.tau_rand_int
 * uses the same masked-shift structure on a signed int64 state; we run it on
 * unsigned 32-bit lanes, which is the canonical (and cleaner) form. We do NOT
 * reproduce umap's exact negative-sampling stream: the embedding is float32
 * and our RNG is seeded statelessly per (edge, epoch), so the result is
 * validated by embedding quality, not coordinate identity. See
 * docs/umap_layout.md.
 */

#ifndef SCBLAS_UMAP_LAYOUT_INTERNAL_H
#define SCBLAS_UMAP_LAYOUT_INTERNAL_H

#include <stdint.h>
#include <math.h>

/* SplitMix64 — used only to expand a scalar seed into well-distributed
 * tausworthe seed words. */
static inline uint64_t scblas_umap_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

/* Seed a 3-word tausworthe state deterministically from (seed, edge, epoch).
 * Stateless across calls: identical (seed, edge, epoch) always yields the same
 * stream regardless of thread scheduling. LFSR113 requires the components to
 * exceed 1, 7, 15 respectively, which the low-bit OR-masks enforce. */
static inline void scblas_umap_seed_rng(uint64_t seed, int64_t edge, int epoch,
                                        uint32_t s[3]) {
    uint64_t z = scblas_umap_splitmix64(
        seed ^ (0x9E3779B97F4A7C15ull * (uint64_t)(edge + 1))
             ^ (0xD1B54A32D192ED03ull * (uint64_t)(epoch + 1)));
    s[0] = (uint32_t)(z & 0xFFFFFFFFu) | 2u;
    z = scblas_umap_splitmix64(z);
    s[1] = (uint32_t)(z & 0xFFFFFFFFu) | 8u;
    z = scblas_umap_splitmix64(z);
    s[2] = (uint32_t)(z & 0xFFFFFFFFu) | 16u;
}

/* LFSR113 step (umap tau_rand_int masked-shift form). Returns a 32-bit value. */
static inline uint32_t scblas_umap_tau_rand(uint32_t s[3]) {
    s[0] = (((s[0] & 4294967294u) << 12) & 0xFFFFFFFFu)
         ^ ((((s[0] << 13) & 0xFFFFFFFFu) ^ s[0]) >> 19);
    s[1] = (((s[1] & 4294967288u) << 4) & 0xFFFFFFFFu)
         ^ ((((s[1] << 2) & 0xFFFFFFFFu) ^ s[1]) >> 25);
    s[2] = (((s[2] & 4294967280u) << 17) & 0xFFFFFFFFu)
         ^ ((((s[2] << 3) & 0xFFFFFFFFu) ^ s[2]) >> 11);
    return s[0] ^ s[1] ^ s[2];
}

/* umap clip: clamp a per-coordinate gradient term to [-4, 4]. */
static inline float scblas_umap_clip(float v) {
    if (v > 4.0f) return 4.0f;
    if (v < -4.0f) return -4.0f;
    return v;
}

/* Reduced (squared) Euclidean distance. */
static inline float scblas_umap_rdist(const float *x, const float *y, int dim) {
    float r = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float diff = x[d] - y[d];
        r += diff * diff;
    }
    return r;
}

/* Attractive-edge gradient coefficient. Returns 0 when d2 == 0.
 * grad = -2ab d2^(b-1) / (a d2^b + 1). We pass pdb = d2^b back via *pdb so the
 * caller can reuse it (it equals d2^(b-1) * d2). */
static inline float scblas_umap_attractive_coeff(float d2, float a, float b) {
    if (d2 <= 0.0f) return 0.0f;
    float pb1 = powf(d2, b - 1.0f);
    float pb = pb1 * d2;            /* d2^b */
    float gc = -2.0f * a * b * pb1;
    gc /= (a * pb + 1.0f);
    return gc;
}

/* Repulsive (negative-sample) gradient coefficient. Returns 0 when d2 == 0. */
static inline float scblas_umap_repulsive_coeff(float d2, float a, float b,
                                                float gamma) {
    if (d2 <= 0.0f) return 0.0f;
    float pb = powf(d2, b);
    float gc = 2.0f * gamma * b;
    gc /= (0.001f + d2) * (a * pb + 1.0f);
    return gc;
}

#endif /* SCBLAS_UMAP_LAYOUT_INTERNAL_H */
