// omp_compat.cpp — OpenMP runtime compatibility shim for macOS.
//
// Why this exists
// ---------------
// On macOS arm64 the R process already contains R's own LLVM OpenMP runtime
// (`$(R_HOME)/lib/libomp.dylib`). If this package links a *different* libomp
// (e.g. Homebrew's), the process ends up with TWO LLVM OpenMP runtimes. The
// moment one of them spawns a worker thread, a worker looks up its
// `kmp_info_t` in the other runtime's `__kmp_threads[]` table, gets NULL, and
// crashes dereferencing it inside `__kmp_suspend_initialize_thread`
// (EXC_BAD_ACCESS at address 0x540). `KMP_DUPLICATE_LIB_OK=TRUE` does not help
// (it only suppresses the libiomp5 duplicate *abort*, not this crash).
//
// The fix (see src/Makevars) is to link against R's own libomp so the whole
// process shares a single OpenMP runtime. R's libomp is slightly older and
// lacks `__kmpc_dispatch_deinit`, a symbol Apple clang 17 emits after every
// `#pragma omp parallel for schedule(dynamic|guided, ...)` loop. We supply that
// one missing entry point here as a no-op, which is exactly the older-libomp
// behaviour: the per-thread dispatch buffer allocated by __kmpc_dispatch_init
// is simply reclaimed at thread teardown instead of at end-of-loop.
//
// macOS only. On Linux/Windows the system OpenMP runtime provides the real
// symbol, so the shim is compiled out and never participates in linking.
#if defined(__APPLE__)
extern "C" void __kmpc_dispatch_deinit(void * /*loc*/, int /*gtid*/) {}
#endif
