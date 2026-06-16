/* version.h — scBLAS version macros and the version-string accessor.
 *
 * `SCBLAS_VERSION_STRING` is the compile-time version; `scblas_version()`
 * returns the runtime version of the library that was actually linked.
 * Differences between the two indicate header/library mismatch.
 */

#ifndef SCBLAS_VERSION_H
#define SCBLAS_VERSION_H

#define SCBLAS_VERSION_MAJOR 0
#define SCBLAS_VERSION_MINOR 1
#define SCBLAS_VERSION_PATCH 0
#define SCBLAS_VERSION_STRING "0.1.0"

/* ---- symbol visibility -------------------------------------------------- */

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(SCBLAS_BUILDING)
    #define SCBLAS_API __declspec(dllexport)
  #else
    #define SCBLAS_API __declspec(dllimport)
  #endif
#elif defined(__GNUC__) || defined(__clang__)
  #define SCBLAS_API __attribute__((visibility("default")))
#else
  #define SCBLAS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Returns the runtime library version as a NUL-terminated string,
 * e.g. "0.1.0". The returned pointer is owned by the library; do not free. */
SCBLAS_API const char *scblas_version(void);

#ifdef __cplusplus
}
#endif

#endif /* SCBLAS_VERSION_H */
