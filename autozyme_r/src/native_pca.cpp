// native_pca.cpp - Python-free native RunPCA + RunCCA cross-product, on a
// runtime-dlopen'd fast BLAS/LAPACK. Removes the reticulate -> numpy/scipy
// dependency from the Seurat patch's RunPCA / RunCCA fast paths.
//
// FORK-SAFETY / LINKING: like wgcna.cpp and native_blas.cpp, we NEVER link a
// fast BLAS at build time. Link-time `-framework Accelerate` (or -lopenblas)
// poisons every BLAS symbol in autozyme.so to route through that backend,
// breaking fork-safety (Accelerate's BLAS thread pool segfaults in mclapply
// children). Instead we resolve cblas_dgemm / cblas_dsyrk / dsyevr_ lazily on
// first use, RTLD_LOCAL, cached - scoping the fast backend to these kernels only.
//
// BACKEND: macOS -> Apple Accelerate (always present). Linux/Windows -> OpenBLAS
// (libopenblas.so/.dll), incl. env override AUTOZYME_NATIVE_BLAS /
// AUTOZYME_OPENBLAS_DLL. If none resolves (LP64 cblas_dgemm + cblas_dsyrk +
// dsyevr_), native_pca_available() is FALSE and patch.R falls back to scipy.
// Master off-switch: AUTOZYME_NATIVE_PCA=0.
//
// PCA: Gram G = X X^T (dsyrk) -> top-npcs eig (dsyevr, partial) -> embeddings
//      X^T V (dgemm). Bit-identical to the scipy Gram+eigh path up to sign.
// CCA: form A = X1^T X2 (dgemm); the truncated top-k SVD is done by irlba in R.

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace Rcpp;

namespace {

enum { CblasColMajor = 102, CblasNoTrans = 111, CblasTrans = 112,
       CblasUpper = 121, CblasLower = 122 };

using dgemm_fn = void (*)(int, int, int, int, int, int, double,
                          const double*, int, const double*, int,
                          double, double*, int);
using dsyrk_fn = void (*)(int, int, int, int, int, double,
                          const double*, int, double, double*, int);
// LAPACK dsyevr_ (Fortran, column-major). LP64 ints.
using dsyevr_fn = void (*)(const char*, const char*, const char*, const int*,
  double*, const int*, const double*, const double*, const int*, const int*,
  const double*, int*, double*, double*, const int*, int*, double*,
  const int*, int*, const int*, int*);

struct Backend {
  void* handle = nullptr;
  dgemm_fn  gemm  = nullptr;
  dsyrk_fn  syrk  = nullptr;
  dsyevr_fn syevr = nullptr;
  bool ok() const { return gemm && syrk && syevr; }
};
static Backend g_be;
static bool g_tried = false;

static std::string env_var(const char* n) {
  const char* v = std::getenv(n); return v ? std::string(v) : std::string();
}
#ifdef _WIN32
static void* open_lib(const std::string& p) { return (void*)LoadLibraryA(p.c_str()); }
static void* sym(void* h, const char* n) { return (void*)GetProcAddress((HMODULE)h, n); }
#else
static void* open_lib(const std::string& p) { return dlopen(p.c_str(), RTLD_LAZY | RTLD_LOCAL); }
static void* sym(void* h, const char* n) { return dlsym(h, n); }
#endif

static std::vector<std::string> candidates() {
  std::vector<std::string> c;
#ifdef __APPLE__
  c.push_back("/System/Library/Frameworks/Accelerate.framework/Accelerate");
#endif
  for (const char* e : {"AUTOZYME_NATIVE_BLAS", "AUTOZYME_OPENBLAS_DLL", "OPENBLAS_DLL"}) {
    std::string v = env_var(e); if (!v.empty()) c.push_back(v);
  }
#ifdef _WIN32
  c.push_back("libopenblas.dll"); c.push_back("openblas.dll");
#elif defined(__APPLE__)
  c.push_back("libopenblas.dylib");
#else
  c.push_back("libopenblas.so"); c.push_back("libopenblas.so.0");
#endif
  return c;
}

static Backend* resolve() {
  if (g_tried) return g_be.ok() ? &g_be : nullptr;
  g_tried = true;
  for (const auto& p : candidates()) {
    void* h = open_lib(p);
    if (!h) continue;
    Backend b;
    b.handle = h;
    b.gemm  = (dgemm_fn)  sym(h, "cblas_dgemm");
    b.syrk  = (dsyrk_fn)  sym(h, "cblas_dsyrk");
    b.syevr = (dsyevr_fn) sym(h, "dsyevr_");
    if (b.ok()) { g_be = b; return &g_be; }
  }
  return nullptr;
}

} // namespace

// [[Rcpp::export]]
bool native_pca_available() {
  if (env_var("AUTOZYME_NATIVE_PCA") == "0") return false;   // live master off-switch
  return resolve() != nullptr;
}

// Full native RunPCA. X: p features x n cells, already centered (scale.data).
// Returns embeddings (n x npcs), loadings (p x npcs), sdev. Matches the patch.
// [[Rcpp::export]]
List native_pca_run(const arma::mat& X, int npcs, bool weight_by_var = true) {
  Backend* be = resolve();
  if (!be) stop("native_pca_run: no fast BLAS/LAPACK backend");
  int p = X.n_rows, n = X.n_cols;
  npcs = std::min(npcs, std::min(p, n - 1));

  // Gram G = X X^T (lower triangle), p x p.
  arma::mat G(p, p, arma::fill::zeros);
  be->syrk(CblasColMajor, CblasLower, CblasNoTrans, p, n, 1.0,
           X.memptr(), p, 0.0, G.memptr(), p);

  // Top-npcs eigenpairs of lower-tri G via dsyevr (RANGE='I'), descending.
  int il = p - npcs + 1, iu = p, m = 0, info = 0, ldz = p;
  double vl = 0, vu = 0, abstol = -1.0;
  std::vector<double> w(p), z((size_t)p * npcs);
  std::vector<int> isuppz(2 * npcs);
  double wq; int iwq, lwork = -1, liwork = -1;
  be->syevr("V","I","L", &p, G.memptr(), &p, &vl,&vu,&il,&iu,&abstol,&m,
            w.data(), z.data(), &ldz, isuppz.data(), &wq,&lwork,&iwq,&liwork,&info);
  lwork = (int)wq; liwork = iwq;
  std::vector<double> work(lwork); std::vector<int> iwork(liwork);
  be->syevr("V","I","L", &p, G.memptr(), &p, &vl,&vu,&il,&iu,&abstol,&m,
            w.data(), z.data(), &ldz, isuppz.data(), work.data(),&lwork,iwork.data(),&liwork,&info);
  if (info != 0) stop("native_pca_run: dsyevr failed (info=%d)", info);
  arma::vec ev(npcs); arma::mat V(p, npcs);                 // descending
  for (int j = 0; j < npcs; ++j) {
    ev(j) = w[npcs - 1 - j];
    std::copy(z.begin()+(size_t)(npcs-1-j)*p, z.begin()+(size_t)(npcs-j)*p, V.colptr(j));
  }

  // Cell embeddings = X^T V (n x npcs) via dgemm.
  arma::mat emb(n, npcs);
  be->gemm(CblasColMajor, CblasTrans, CblasNoTrans, n, npcs, p, 1.0,
           X.memptr(), p, V.memptr(), p, 0.0, emb.memptr(), n);

  arma::vec d = arma::sqrt(arma::clamp(ev, 0.0, arma::datum::inf));
  arma::vec sdev = d / std::sqrt((double)std::max(1, n - 1));
  if (!weight_by_var) for (int j = 0; j < npcs; ++j) if (d(j) > 0) emb.col(j) /= d(j);
  return List::create(_["embeddings"]=emb, _["loadings"]=V, _["sdev"]=sdev, _["eigvals"]=ev);
}

// CCA cross-product A = X1^T X2 (cells1 x cells2) via dgemm. The truncated
// top-k SVD is computed by irlba in R (native_RunCCA in patch.R).
// [[Rcpp::export]]
arma::mat native_cca_formA(const arma::mat& X1, const arma::mat& X2) {
  Backend* be = resolve();
  if (!be) stop("native_cca_formA: no fast BLAS backend");
  int n1 = X1.n_cols, n2 = X2.n_cols, f = X1.n_rows;
  arma::mat A(n1, n2);
  be->gemm(CblasColMajor, CblasTrans, CblasNoTrans, n1, n2, f, 1.0,
           X1.memptr(), f, X2.memptr(), f, 0.0, A.memptr(), n1);
  return A;
}
