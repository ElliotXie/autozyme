/* az_pthread_compat.h — minimal pthread compatibility shim.
 *
 * POSIX (macOS/Linux): a no-op passthrough — just pull in <pthread.h> +
 * <unistd.h>. The vendored scanpy kernels use pthreads natively, so the POSIX
 * build is byte-for-byte unchanged.
 *
 * Windows (MSVC): map the small pthread subset the kernels actually use onto
 * Win32 primitives, so the SAME .c sources compile without winpthreads. The
 * subset is exactly:
 *   - knn.c                  : pthread_create / pthread_join
 *   - umap_layout_parallel.c : pthread_create / pthread_join + mutex + cond
 * No barriers / TLS / rwlocks are used (the layout kernel rolls its own
 * counting barrier from mutex+cond), so this header stays small and auditable.
 *
 * Also provides scblas_native_online_cpus() so knn.c's CPU detection is
 * portable (umap_layout_parallel.c already has its own #ifdef _WIN32 guard).
 */
#ifndef AZ_PTHREAD_COMPAT_H
#define AZ_PTHREAD_COMPAT_H

#ifndef _WIN32
/* ===== POSIX: native pthreads, unchanged ================================= */
#include <pthread.h>
#include <unistd.h>

static inline long scblas_native_online_cpus(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? n : 1L;
}

#else
/* ===== Windows (MSVC): pthread subset over Win32 ========================== */
#include <windows.h>
#include <process.h>
#include <stdlib.h>
#include <stdint.h>

static inline long scblas_native_online_cpus(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (si.dwNumberOfProcessors > 0) ? (long)si.dwNumberOfProcessors : 1L;
}

/* ---- threads ---- */
typedef HANDLE pthread_t;
typedef void   pthread_attr_t;        /* ignored; every caller passes NULL */

typedef void *(*az_pthread_fn)(void *);
typedef struct { az_pthread_fn fn; void *arg; } az_pthread_start;

/* Win32 thread entry is `unsigned __stdcall`; adapt the POSIX
 * `void *(*)(void *)` worker signature through a heap-allocated trampoline. */
static unsigned __stdcall az_pthread_trampoline(void *p) {
    az_pthread_start s = *(az_pthread_start *)p;
    free(p);
    (void)s.fn(s.arg);
    return 0u;
}

static inline int pthread_create(pthread_t *thr, const pthread_attr_t *attr,
                                 az_pthread_fn fn, void *arg) {
    (void)attr;
    az_pthread_start *s = (az_pthread_start *)malloc(sizeof(*s));
    if (s == NULL) return -1;
    s->fn = fn;
    s->arg = arg;
    uintptr_t h = _beginthreadex(NULL, 0, az_pthread_trampoline, s, 0, NULL);
    if (h == 0) { free(s); return -1; }
    *thr = (HANDLE)h;
    return 0;
}

static inline int pthread_join(pthread_t thr, void **retval) {
    (void)retval;
    WaitForSingleObject(thr, INFINITE);
    CloseHandle(thr);
    return 0;
}

/* ---- mutex (CRITICAL_SECTION) ---- */
typedef CRITICAL_SECTION pthread_mutex_t;
typedef void             pthread_mutexattr_t;

static inline int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a) {
    (void)a; InitializeCriticalSection(m); return 0;
}
static inline int pthread_mutex_destroy(pthread_mutex_t *m) { DeleteCriticalSection(m); return 0; }
static inline int pthread_mutex_lock(pthread_mutex_t *m)    { EnterCriticalSection(m);  return 0; }
static inline int pthread_mutex_unlock(pthread_mutex_t *m)  { LeaveCriticalSection(m);  return 0; }

/* ---- condition variable (CONDITION_VARIABLE) ---- */
typedef CONDITION_VARIABLE pthread_cond_t;
typedef void               pthread_condattr_t;

static inline int pthread_cond_init(pthread_cond_t *c, const pthread_condattr_t *a) {
    (void)a; InitializeConditionVariable(c); return 0;
}
static inline int pthread_cond_destroy(pthread_cond_t *c) { (void)c; return 0; }
static inline int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
    return SleepConditionVariableCS(c, m, INFINITE) ? 0 : -1;
}
static inline int pthread_cond_broadcast(pthread_cond_t *c) { WakeAllConditionVariable(c); return 0; }

#endif /* _WIN32 */
#endif /* AZ_PTHREAD_COMPAT_H */
