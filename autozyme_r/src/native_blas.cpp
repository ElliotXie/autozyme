// Cross-platform native BLAS dispatch for autozyme patches.
//
// This file intentionally loads BLAS backends at runtime instead of linking
// against them. That keeps the package installable on machines without
// OpenBLAS/BLIS/MKL while allowing hot paths to opt into a fast DGEMM backend
// when one is available. Patches should use the R helpers `.az_gemm()`,
// `.az_crossprod()`, and `.az_xtx()` rather than adding package-local platform
// branches.

// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>

#include <cstdlib>
#include <cstdint>
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

enum { CblasRowMajor = 101, CblasColMajor = 102 };
enum { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

using dgemm_lp64_fn = void (*)(const int, const int, const int,
                               const int, const int, const int,
                               const double, const double *, const int,
                               const double *, const int,
                               const double, double *, const int);
using dgemm_ilp64_fn = void (*)(const int64_t, const int64_t, const int64_t,
                                const int64_t, const int64_t, const int64_t,
                                const double, const double *, const int64_t,
                                const double *, const int64_t,
                                const double, double *, const int64_t);
using openblas_set_threads_fn = void (*)(int);
using openblas_get_threads_fn = int (*)();

struct BlasBackend {
  void* handle = nullptr;
  std::string path;
  std::string abi;
  std::string dgemm_symbol;
  dgemm_lp64_fn dgemm_lp64 = nullptr;
  dgemm_ilp64_fn dgemm_ilp64 = nullptr;
  openblas_set_threads_fn set_threads = nullptr;
  openblas_get_threads_fn get_threads = nullptr;
};

static BlasBackend g_backend;

static std::string env_var(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) return std::string();
  return std::string(value);
}

static void append_unique(std::vector<std::string>& xs, const std::string& x) {
  if (x.empty()) return;
  for (const auto& old : xs) {
    if (old == x) return;
  }
  xs.push_back(x);
}

static std::vector<std::string> make_candidates(CharacterVector paths) {
  std::vector<std::string> out;
  for (R_xlen_t i = 0; i < paths.size(); ++i) {
    if (paths[i] == NA_STRING) continue;
    append_unique(out, as<std::string>(paths[i]));
  }
  append_unique(out, env_var("AUTOZYME_OPENBLAS_DLL"));
  append_unique(out, env_var("AUTOZYME_BLAS_DLL"));
  append_unique(out, env_var("OPENBLAS_DLL"));

#ifdef _WIN32
  append_unique(out, "libopenblas.dll");
  append_unique(out, "openblas.dll");
  append_unique(out, "libblis.dll");
  append_unique(out, "blis.dll");
#elif defined(__APPLE__)
  append_unique(out, "libopenblas.dylib");
  append_unique(out, "libblis.dylib");
#else
  append_unique(out, "libopenblas.so");
  append_unique(out, "libopenblas.so.0");
  append_unique(out, "libblis.so");
  append_unique(out, "libblis.so.4");
#endif
  return out;
}

#ifdef _WIN32
static void* open_library(const std::string& path) {
  return reinterpret_cast<void*>(LoadLibraryA(path.c_str()));
}

static void* load_symbol(void* handle, const char* name) {
  return reinterpret_cast<void*>(
    GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
}
#else
static void* open_library(const std::string& path) {
  return dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
}

static void* load_symbol(void* handle, const char* name) {
  return dlsym(handle, name);
}
#endif

template <typename Fn>
static Fn symbol_as(void* handle, const char* name) {
  return reinterpret_cast<Fn>(load_symbol(handle, name));
}

static openblas_set_threads_fn first_set_threads(void* handle,
                                                 const char* const* names,
                                                 int n_names) {
  for (int i = 0; i < n_names; ++i) {
    auto fn = symbol_as<openblas_set_threads_fn>(handle, names[i]);
    if (fn != nullptr) return fn;
  }
  return nullptr;
}

static openblas_get_threads_fn first_get_threads(void* handle,
                                                 const char* const* names,
                                                 int n_names) {
  for (int i = 0; i < n_names; ++i) {
    auto fn = symbol_as<openblas_get_threads_fn>(handle, names[i]);
    if (fn != nullptr) return fn;
  }
  return nullptr;
}

static bool bind_backend(void* handle, const std::string& path,
                         BlasBackend& backend) {
  static const char* set_lp64[] = {
    "openblas_set_num_threads",
    "goto_set_num_threads"
  };
  static const char* get_lp64[] = {
    "openblas_get_num_threads",
    "goto_get_num_threads"
  };
  static const char* set_ilp64[] = {
    "openblas_set_num_threads64_",
    "openblas_set_num_threads_64_",
    "scipy_openblas_set_num_threads64_",
    "scipy_openblas_set_num_threads_64_"
  };
  static const char* get_ilp64[] = {
    "openblas_get_num_threads64_",
    "openblas_get_num_threads_64_",
    "scipy_openblas_get_num_threads64_",
    "scipy_openblas_get_num_threads_64_"
  };

  auto lp64 = symbol_as<dgemm_lp64_fn>(handle, "cblas_dgemm");
  if (lp64 != nullptr) {
    backend.handle = handle;
    backend.path = path;
    backend.abi = "lp64";
    backend.dgemm_symbol = "cblas_dgemm";
    backend.dgemm_lp64 = lp64;
    backend.set_threads = first_set_threads(handle, set_lp64, 2);
    backend.get_threads = first_get_threads(handle, get_lp64, 2);
    return true;
  }

  struct Ilp64Candidate {
    const char* name;
    dgemm_ilp64_fn fn;
  };
  Ilp64Candidate ilp64[] = {
    {"cblas_dgemm64_", symbol_as<dgemm_ilp64_fn>(handle, "cblas_dgemm64_")},
    {"cblas_dgemm_64_", symbol_as<dgemm_ilp64_fn>(handle, "cblas_dgemm_64_")},
    {"scipy_cblas_dgemm64_", symbol_as<dgemm_ilp64_fn>(handle, "scipy_cblas_dgemm64_")},
    {"scipy_cblas_dgemm_64_", symbol_as<dgemm_ilp64_fn>(handle, "scipy_cblas_dgemm_64_")}
  };
  for (const auto& cand : ilp64) {
    if (cand.fn == nullptr) continue;
    backend.handle = handle;
    backend.path = path;
    backend.abi = "ilp64";
    backend.dgemm_symbol = cand.name;
    backend.dgemm_ilp64 = cand.fn;
    backend.set_threads = first_set_threads(handle, set_ilp64, 4);
    backend.get_threads = first_get_threads(handle, get_ilp64, 4);
    return true;
  }
  return false;
}

static BlasBackend* resolve_backend(CharacterVector paths) {
  if (g_backend.handle != nullptr) return &g_backend;

  auto candidates = make_candidates(paths);
  for (const auto& path : candidates) {
    void* handle = open_library(path);
    if (handle == nullptr) continue;
    BlasBackend candidate;
    if (bind_backend(handle, path, candidate)) {
      g_backend = candidate;
      return &g_backend;
    }
  }
  return nullptr;
}

static void call_dgemm(BlasBackend* backend,
                       bool transA, bool transB,
                       int m, int n, int k,
                       const double* a, int lda,
                       const double* b, int ldb,
                       double* c, int ldc) {
  const int ta = transA ? CblasTrans : CblasNoTrans;
  const int tb = transB ? CblasTrans : CblasNoTrans;
  if (backend->dgemm_lp64 != nullptr) {
    backend->dgemm_lp64(
      CblasColMajor, ta, tb,
      m, n, k,
      1.0, a, lda,
      b, ldb,
      0.0, c, ldc);
    return;
  }

  backend->dgemm_ilp64(
    static_cast<int64_t>(CblasColMajor),
    static_cast<int64_t>(ta),
    static_cast<int64_t>(tb),
    static_cast<int64_t>(m),
    static_cast<int64_t>(n),
    static_cast<int64_t>(k),
    1.0, a, static_cast<int64_t>(lda),
    b, static_cast<int64_t>(ldb),
    0.0, c, static_cast<int64_t>(ldc));
}

} // namespace

// [[Rcpp::export]]
NumericMatrix az_blas_gemm(NumericMatrix A, NumericMatrix B,
                           bool transA = false, bool transB = false,
                           int threads = 0,
                           CharacterVector dll_paths = CharacterVector()) {
  BlasBackend* backend = resolve_backend(dll_paths);
  if (backend == nullptr) {
    stop("az_blas_gemm: no compatible BLAS backend found. Set "
         "AUTOZYME_OPENBLAS_DLL to an OpenBLAS/BLIS DLL exporting cblas_dgemm "
         "(or scipy_cblas_dgemm64_ for NumPy/SciPy OpenBLAS).");
  }
  if (threads > 0 && backend->set_threads != nullptr) {
    backend->set_threads(threads);
  }

  const int a_rows = A.nrow();
  const int a_cols = A.ncol();
  const int b_rows = B.nrow();
  const int b_cols = B.ncol();

  const int m = transA ? a_cols : a_rows;
  const int k_a = transA ? a_rows : a_cols;
  const int k_b = transB ? b_cols : b_rows;
  const int n = transB ? b_rows : b_cols;
  if (k_a != k_b) {
    stop("az_blas_gemm: non-conformable arguments");
  }
  if (m < 0 || n < 0 || k_a < 0) {
    stop("az_blas_gemm: invalid matrix dimensions");
  }

  NumericMatrix C(m, n);
  if (m == 0 || n == 0) return C;
  if (k_a == 0) return C;

  call_dgemm(backend, transA, transB, m, n, k_a,
             REAL(A), a_rows,
             REAL(B), b_rows,
             REAL(C), m);
  return C;
}

// [[Rcpp::export]]
NumericMatrix az_blas_crossprod(NumericMatrix X, int threads = 0,
                                CharacterVector dll_paths = CharacterVector()) {
  return az_blas_gemm(X, X, true, false, threads, dll_paths);
}

// [[Rcpp::export]]
List az_blas_info(CharacterVector dll_paths = CharacterVector()) {
  BlasBackend* backend = resolve_backend(dll_paths);
  if (backend == nullptr) {
    return List::create(
      _["available"] = false,
      _["backend"] = CharacterVector::create(NA_STRING),
      _["path"] = CharacterVector::create(NA_STRING),
      _["abi"] = CharacterVector::create(NA_STRING),
      _["dgemm_symbol"] = CharacterVector::create(NA_STRING),
      _["thread_control"] = false,
      _["threads"] = NA_INTEGER
    );
  }
  int n_threads = NA_INTEGER;
  if (backend->get_threads != nullptr) {
    n_threads = backend->get_threads();
  }
  return List::create(
    _["available"] = true,
    _["backend"] = "dynamic-blas",
    _["path"] = backend->path,
    _["abi"] = backend->abi,
    _["dgemm_symbol"] = backend->dgemm_symbol,
    _["thread_control"] = backend->set_threads != nullptr,
    _["threads"] = n_threads
  );
}
