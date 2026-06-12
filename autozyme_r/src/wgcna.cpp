// C++ kernel for autozyme's WGCNA patch.
//
// Lifted from test_general_bio/test_wgcna_blockwise_real/pipeline/cpp/
// accelerate_kernels.cpp (was Rcpp::sourceCpp'd inline in pipeline/run.R).
//
// The single exported kernel `accelerate_crossprod` computes t(X) %*% X via
// Apple Accelerate's threaded BLAS dgemm. R's libRblas on macOS is the
// single-threaded reference BLAS, so the cor/TOM matmul steps inside
// WGCNA::blockwiseModules (~3-10 s on the dev tiers) are the dominant
// bottleneck. Apple Accelerate's threaded cblas_dgemm reaches the same answer
// in ~0.04 s on 5000x2000 (250x speedup over reference BLAS).
//
// macOS-only: the function is compiled only when __APPLE__ is defined and
// the autozyme R-side patch (`inst/patches/wgcna.R`) detects the result via
// `Sys.info()[["sysname"]] == "Darwin"`. On Linux / Windows the patch falls
// through to a parallel-chunked mclapply crossprod (still in the patch).
//
// IMPORTANT: we resolve cblas_dgemm via dlopen/dlsym at first call rather
// than link-time `-framework Accelerate`. Link-time framework dependency
// pulls every BLAS symbol in autozyme.so into Apple Accelerate's resolution
// path (even for other kernels' Armadillo / Rcpp matrix ops that intend to
// use R's reference libRblas). Accelerate's BLAS thread pool is NOT fork-safe
// — patches that call mclapply (monocle3, ...) then segfault at address
// ~0x110 in the forked child the moment any BLAS symbol is touched. By
// loading Accelerate dynamically, only this kernel (which is never called
// inside a fork) touches it; the rest of autozyme.so stays on R's single-
// threaded reference BLAS and is fork-safe by default. See
// `inst/patches/monocle3.R` for the bug class this prevents.

// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>

#ifdef __APPLE__
#include <dlfcn.h>

extern "C" {
typedef enum { CblasRowMajor=101, CblasColMajor=102 } CBLAS_ORDER;
typedef enum { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 } CBLAS_TRANSPOSE;
using cblas_dgemm_fn = void (*)(const CBLAS_ORDER, const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE,
                                const int, const int, const int,
                                const double, const double *, const int,
                                const double *, const int,
                                const double, double *, const int);
}

// Resolve Accelerate's cblas_dgemm lazily on first call; cache the function
// pointer for subsequent calls. dlopen with RTLD_LAZY|RTLD_GLOBAL adds the
// framework's symbols to the global lookup namespace, but only after this
// kernel is invoked — so monocle3's mclapply path (which is never reached
// for wgcna) stays untouched.
static cblas_dgemm_fn resolve_cblas_dgemm() {
  static cblas_dgemm_fn cached = nullptr;
  if (cached) return cached;
  void* handle = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate",
                        RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    Rcpp::stop("wgcna.accelerate_crossprod: failed to dlopen Accelerate framework: %s",
               dlerror());
  }
  cached = reinterpret_cast<cblas_dgemm_fn>(dlsym(handle, "cblas_dgemm"));
  if (!cached) {
    Rcpp::stop("wgcna.accelerate_crossprod: dlsym('cblas_dgemm') failed: %s",
               dlerror());
  }
  return cached;
}
#endif

using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix accelerate_crossprod(NumericMatrix X) {
#ifdef __APPLE__
  static cblas_dgemm_fn dgemm = resolve_cblas_dgemm();
  int n = X.nrow();
  int p = X.ncol();
  NumericMatrix C(p, p);
  dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
        p, p, n,
        1.0, REAL(X), n,
        REAL(X), n,
        0.0, REAL(C), p);
  return C;
#else
  // Linux / Windows: should not be called — the R-side patch detects
  // non-Darwin platforms in advance and uses the parallel-chunked
  // mclapply fallback path. Return an error if it ever is.
  stop("accelerate_crossprod: Apple Accelerate kernel called on non-Darwin platform");
  return NumericMatrix(0, 0);
#endif
}
