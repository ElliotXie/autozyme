# autozyme — cross-patch traps registry

Living list of non-obvious failure modes hit while lifting tasks into patches.
**Before lifting a new task**, scan for upstream pkgs that match any pattern
below and apply the relevant fix preemptively. Append a new entry whenever a
fresh trap costs more than ~30 minutes to diagnose.

Format: **Symptom — Trigger — Fix — Seen in**.

---

## 1. numba parallel JIT deadlocks when TensorFlow is co-loaded

- **Symptom**: `@njit(parallel=True)` first-call hangs forever (or segfaults
  in TBB) when both `tensorflow` and `numba` are imported in the same
  process. Single-thread numba (no parallel=True) is fine. Either lib alone
  is fine.
- **Trigger**: A patch is `activate()`d after another patch has already
  imported TF (typical case: `autozyme.activate("sccoda")` then
  `autozyme.activate("xclim")` in the same process).
- **Fix**: At the top of the offending patch's `__init__.py`, BEFORE
  `import numba`, set `os.environ.setdefault("NUMBA_NUM_THREADS", "1")` if
  `"tensorflow" in sys.modules`. The prange kernel still gets the JIT
  speedup; only parallelism is sacrificed. Must run before numba import —
  numba reads the env var only at module-init.
- **Seen in**: `autozyme/xclim/__init__.py` lines 18–24.

## 2. obspy `_ENTRY_POINT_CACHE` holds stale function refs across patch toggle

- **Symptom**: `verify_patch("obspy")` shows the patched timing identical to
  baseline. setattr-rebinding `obspy.signal.filter.bandpass` had no effect.
  Restart the Python process and re-run patched alone — it works. Toggle
  baseline → patched in same process — broken.
- **Trigger**: Upstream caches function references at first lookup
  (`obspy.core.util.misc._ENTRY_POINT_CACHE` is the specific case). If the
  baseline call ran first under restore, the cache holds upstream
  `bandpass`; subsequent activate's setattr is invisible to `Trace.filter`
  which goes through the cache.
- **Fix**: Inside the fast-replacement function (NOT at register-time),
  `from obspy.core.util.misc import _ENTRY_POINT_CACHE; _ENTRY_POINT_CACHE.clear()`
  on entry. Doing it at register / activate is too early — verify_patch
  toggles restore→activate within one process.
- **Seen in**: `autozyme/obspy/__init__.py::fast_stream_filter` lines 100–112.
- **General pattern**: any upstream that uses entry points
  (importlib.metadata, pkg_resources, plugin registries) likely has a
  similar cache. grep upstream src for `_CACHE`, `entry_points`, or
  `lru_cache` on lookup paths.

## 4. scvelo activation pins parent-process BLAS to 1 thread

- **Symptom**: After `autozyme.activate("scvelo")`, later activations like
  `autozyme.activate("scanpy")` see PCA / WLS / matmul running
  single-threaded — speedup numbers for those patches drop vs. running them
  in a fresh process.
- **Trigger**: scvelo's loky workers spawn-inherit `os.environ`, so the
  patch sets `OMP_NUM_THREADS=1` (and 4 siblings) at submodule-import time
  to prevent N-workers × M-BLAS oversubscription. The same env mutation
  also rebinds the parent process's BLAS thread count.
- **Fix**: Either activate BLAS-heavy patches BEFORE scvelo, run scvelo in
  its own process, or set `AUTOZYME_SCVELO_NO_BLAS_PIN=1` before
  activation (accepts oversubscription in exchange for un-pinned BLAS in
  the parent). The patch prints a one-shot stderr warning when it pins.
- **Seen in**: `autozyme/scvelo/__init__.py` lines ~41-77.

## 3. loky-pool + numba JIT first-call startup dilutes patches that parallelize via joblib/Parallel

- **Symptom**: `verify_patch` reports e.g. 7.5x on medium tier; the task's
  own `results.tsv` shows ~17x for the same matrix cell. The metrics still
  pass at pearson=1.0 — purely a timing-window mismatch.
- **Trigger**: The patch parallelizes per-unit work across a `loky` /
  joblib worker pool AND uses `@njit` numba kernels. The first call in a
  fresh Python process pays (a) loky worker fork + scvelo re-import per
  worker, and (b) numba on-disk JIT cache miss / compile in each worker.
  The task's `pipeline/run.py` pre-warms the pool OUTSIDE the timed
  window, which `verify_patch` doesn't reproduce — it times one
  `smoke.call` invocation, startup included.
- **Fix**: Don't paper over it. The verify number is the **honest user-
  observed first-call speedup**; the `results.tsv` number is steady-state
  per-call. Both are correct, measuring different things. In the patch's
  README, quote `verify_patch` as the cold-start figure and note that
  subsequent calls in the same process see the steady-state ratio
  (which is what results.tsv reports). Pre-warming in `smoke.load` would
  hide the cost only for the patched run (baseline n_jobs=1 doesn't use
  the pool), violating fair-comparison.
- **Seen in**: `autozyme/scvelo/__init__.py` — 4 loky workers + 5 numba
  kernels.

---

For R-side equivalents (callName reflection, S4 method install, mclapply
namespace scoping, Rcpp inline GC) see `../autozyme_r/CAVEATS.md`.
