"""bench_profile.py — fixture-matrix benchmark for `zyme profile`.

Drives N fixtures × 4 backends × R reps. For each cell, captures wall
time and validates per-backend assertions (hotspot recall, expected
ranking). Outputs a clean summary table.

This is NOT a pytest test — profile runs are slow (seconds to minutes
per cell). Run on demand:

    python tests/bench_profile.py                       # full matrix, 3 reps
    python tests/bench_profile.py --reps 1              # quick smoke
    python tests/bench_profile.py --fixtures synthetic  # one fixture only
    python tests/bench_profile.py --include-slow        # include long real tasks
    python tests/bench_profile.py --strict              # nonzero on failures
    python tests/bench_profile.py --json results.json   # dump structured

Design priorities:
  - Minimal test ceremony: each fixture is a dict in FIXTURES; add/remove
    by editing the list.
  - Per-backend assertions: each fixture declares what each backend SHOULD
    capture (or skip). Lets us bake in known limitations honestly
    (e.g. native on pure-Python sees interpreter frames, not user funcs).
  - Safe-to-rerun: --no-archive on every profile run; backend artifacts stay
    under each fixture's pipeline/ directory.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Repo root — used to locate fixtures + autozyme_cli's tests/ dir.
_HERE = Path(__file__).resolve().parent           # tests/
_AUTOZYME_CLI = _HERE.parent                       # autozyme_cli/
_WORKSPACE = _AUTOZYME_CLI.parent.parent           # autozyme_mac/

# Fixture matrix. Each entry is a per-task spec; bench iterates the list.
#
# Schema:
#   name:           short label (used in output)
#   task:           absolute path to task.yaml's directory
#   tier:           dataset tier name (must exist in task.yaml)
#   lang:           "py" | "R" — informational (bench infers from task)
#   skip_backends:  backends known to produce no useful output, or backend/env
#                   combinations intentionally omitted from the default matrix
#   skip_reasons:   optional {backend: reason} printed for skipped cells
#   slow:           optional bool; excluded by default unless --include-slow
#                   is passed or the fixture is explicitly named
#   checks:         per-backend validation rules. Empty {} = no validation
#                   (just record wall + hotspots, useful while exploring).
#   max_overhead_x: per-backend acceptable overhead vs baseline. Reported
#                   always; enforced when the script runs with --strict.
#
# Check types (combine freely):
#   any_of: [str]              - >=1 substring in any top-5 hotspot label/raw
#   all_of: [str]              - every substring appears somewhere in top-5
#   not_any_of: [str]          - no substring appears in top-5
#   top3_all: [str]            - all listed substrings present somewhere in top-3
#   rank1_contains: str        - the rank=1 hotspot label/raw contains substring
#   min_hotspots: int          - profile.json must have at least this many hotspots
#   max_hotspots: int          - profile.json must have at most this many hotspots
#   notes_any_of: [str]        - >=1 substring appears in profile notes
#   artifact_contains: str     - artifacts.raw contains substring
#   override_present: [str]    - override_summary contains each name substring
#   override_marker_present: [str] - override_markers contains each name substring
#   override_min_workers: dict - override aggregation covers at least N pids
#   actionable_any_of: [str]   - >=1 substring in actionable_hotspots top-5
#   actionable_rank1_contains: str - actionable rank=1 contains substring
#
FIXTURES = [
    {
        "name": "synthetic",
        "task": str(_HERE / "fixtures" / "synthetic_groundtruth"),
        "tier": "tiny",
        "lang": "py",
        "skip_backends": [],  # all 4 backends must work — fixture has with_profile()
        "checks": {
            "cpu":    {
                "top3_all": ["cpu_a", "cpu_b", "cpu_c"],
                "rank1_contains": "cpu_a",
                "artifact_contains": "profile.out",
            },
            "full":   {
                "top3_all": ["cpu_a", "cpu_b", "cpu_c"],
                "rank1_contains": "cpu_a",
                "artifact_contains": "scalene.json",
            },
            "mem":    {
                "any_of": ["cpu_c"],  # only cpu_c allocates (list.append)
                "not_any_of": ["dataclasses.py", "_compile_bytecode", "importlib"],
                "notes_any_of": ["scope=with_profile"],
                "artifact_contains": "memray.bin",
            },
            "native": {
                "any_of": ["_PyEval_EvalFrameDefault"],  # pure-Python → interpreter
                "artifact_contains": "native_sample",
            },
        },
        "max_overhead_x": {"cpu": 1.5, "full": 6.0, "mem": 3.0, "native": 2.0},
    },
    {
        "name": "lifelines_cox",
        "task": str(_WORKSPACE / "test_general_bio" / "test_lifelines_cox"),
        "tier": "tiny",
        "lang": "py",
        "skip_backends": ["cpu"],  # no with_profile() in pipeline/run.py
        "checks": {
            "full":   {"any_of": ["fast_get_efron", "_take_nd", "fast_preprocess"]},
            "mem":    {"any_of": ["_take_nd_ndarray", "fast_get_efron", "normalize"]},
            "native": {
                "any_of": ["_PyEval_EvalFrameDefault", "_multiarray_umath", "prepare_index"],
                "not_any_of": ["pthread_cond_wait"],
            },
        },
        "max_overhead_x": {"full": 15.0, "mem": 3.0, "native": 2.0},
    },
    {
        "name": "astropy_lombscargle",
        "task": str(_WORKSPACE / "test_non_bio" / "test_astropy_lombscargle"),
        "tier": "tiny",
        "lang": "py",
        # full skipped: Scalene SIGSEGVs on this workload (OpenMP interaction
        # between scalene's signal handler and pocketfft's libomp. Confirmed
        # via stderr "Scalene error: received signal SIGSEGV". Not fixable
        # from our side — Scalene upstream bug. Use --backend native or cpu
        # for Lombscargle / FFT-heavy tasks.
        "skip_backends": ["cpu", "full"],
        "checks": {
            "mem":    {"any_of": ["zeros", "empty", "lombscargle", "pypocketfft"]},
            # Native sees pocketfft directly (it's the modern numpy FFT engine).
            # Old fixture expected `fftpack` — that's the legacy scipy module name,
            # not what NumPy 2.x uses. Real frames: pocketfft::detail::cfftp<>::pass*.
            "native": {"any_of": ["pypocketfft", "_PyEval_EvalFrameDefault", "_multiarray"]},
        },
        "max_overhead_x": {"mem": 3.0, "native": 2.0},
    },
    {
        "name": "fgsea",
        "task": str(_WORKSPACE / "test_general_bio" / "test_fgsea"),
        "tier": "tiny",
        "lang": "R",
        "skip_backends": [],  # has with_profile in run.R
        # fgsea is the known hard case for profilers: workers exit too fast
        # to see fgseaMultilevel via any backend (parent-side IPC noise
        # only). HOWEVER override_summary (option 3) DOES see it because
        # the timing wrapper runs in-worker and emits a per-pid summary
        # line that mclapply forwards to parent. This is the breakthrough
        # signal the bench should verify on this fixture.
        "checks": {
            "cpu":    {
                "any_of": ["unserialize", "mcfork", "lazyLoadDBfetch", "readChild"],
                "override_present": ["fgsea::fgseaMultilevel"],
                "override_min_workers": {"name": "fgsea::fgseaMultilevel", "min": 4},
                "actionable_rank1_contains": "fgsea::fgseaMultilevel",
            },
            "full":   {"any_of": ["unserialize", "mcfork", "lazyLoadDBfetch", "readChild"]},
            "mem":    {"any_of": ["unserialize", "mcfork", "readChild"]},
            "native": {"any_of": ["RunGenCollect", "bcEval_loop", "Rf_mkCharLenCE"]},
        },
        "max_overhead_x": {"cpu": 1.5, "full": 1.5, "mem": 1.5, "native": 1.5},
    },
    {
        "name": "clusterprofiler",
        "task": str(_WORKSPACE / "test_general_bio" / "test_clusterprofiler"),
        "tier": "tiny",
        "lang": "R",
        "skip_backends": [],
        "checks": {
            # R-Rcpp single-process — native should see inside fastmatch / data.table C
            "cpu":    {
                "min_hotspots": 1,
                # task uses install_override on compareCluster
                "override_present": ["compareCluster"],
            },
            "full":   {"min_hotspots": 1},
            "native": {"any_of": ["libR.dylib", "fastmatch", "data.table", ".so"]},
        },
        "max_overhead_x": {"cpu": 1.5, "full": 1.5, "mem": 1.5, "native": 1.5},
    },
    # ---- Allocation-heavy synthetic: validates mem backend on real
    # NumPy buffer allocations (large 800MB array + churn patterns) ----
    {
        "name": "alloc_heavy",
        "task": str(_HERE / "fixtures" / "alloc_heavy_synthetic"),
        "tier": "tiny",
        "lang": "py",
        "skip_backends": [],  # all 4 backends should capture
        "checks": {
            "cpu":  {
                "any_of": ["alloc_big", "alloc_medium", "alloc_small"],
                "rank1_contains": "alloc_big",  # biggest function dominates
            },
            "full": {
                "any_of": ["alloc_big", "alloc_medium", "alloc_small"],
                "rank1_contains": "alloc_big",
            },
            "mem":  {
                # memray must rank alloc_big first (~800MB array dwarfs the others).
                "any_of": ["alloc_big"],
                "rank1_contains": "alloc_big",
            },
            "native": {
                # Compute dominated by BLAS matmul on the big array.
                "any_of": ["openblas", "DOUBLE", "matmul", "_PyEval"],
            },
        },
        "max_overhead_x": {"cpu": 1.5, "full": 4.0, "mem": 2.0, "native": 2.0},
    },
    # ---- Synthetic multiprocessing: worker-side override timing should
    # survive even when profilers mostly see parent Pool wait/IPC frames. ----
    {
        "name": "multiprocess_override",
        "task": str(_HERE / "fixtures" / "multiprocess_override_synthetic"),
        "tier": "tiny",
        "lang": "py",
        "skip_backends": ["full", "mem"],
        "skip_reasons": {
            "full": "Scalene + forked short workers is noisy; this fixture tests override aggregation",
            "mem": "scoped memray in the parent does not trace forked worker CPU work",
        },
        "checks": {
            "cpu": {
                "min_hotspots": 1,
                "override_present": ["worker_target.worker_payload"],
                "override_min_workers": {
                    "name": "worker_target.worker_payload",
                    "min": 4,
                },
                "actionable_rank1_contains": "worker_target.worker_payload",
                "notes_any_of": ["promoted override_summary"],
            },
            "native": {
                "any_of": ["_PyEval_EvalFrameDefault", "PyLong", "binary_op1"],
                "not_any_of": ["__psynch_cvwait", "libsystem_kernel.dylib:read", "poll"],
                "override_present": ["worker_target.worker_payload"],
                "override_min_workers": {
                    "name": "worker_target.worker_payload",
                    "min": 4,
                },
                "actionable_rank1_contains": "worker_target.worker_payload",
                "notes_any_of": ["aggregated", "idle/wait frames filtered out"],
            },
        },
        "max_overhead_x": {"cpu": 3.0, "native": 5.0},
    },
    # ---- Synthetic idle-only native profile: validates that wait/syscall
    # samples become notes instead of actionable hotspots. ----
    {
        "name": "idle_sleep_native",
        "task": str(_HERE / "fixtures" / "idle_sleep_synthetic"),
        "tier": "tiny",
        "lang": "py",
        "skip_backends": ["cpu", "full", "mem"],
        "skip_reasons": {
            "cpu": "idle-only workload is specifically a native wait-frame filter test",
            "full": "idle-only workload is specifically a native wait-frame filter test",
            "mem": "idle-only workload has no meaningful allocation target",
        },
        "checks": {
            "native": {
                "max_hotspots": 0,
                "notes_any_of": [
                    "idle/wait frames filtered out",
                    "no sample output captured",
                ],
                "not_any_of": ["__semwait_signal", "mach_msg2_trap", "__psynch_cvwait"],
            },
        },
        "max_overhead_x": {"native": 8.0},
    },
    # ---- Lightweight native numerical kernels: BLAS / FFT / sort without
    # alloc_heavy's large resident-memory footprint. ----
    {
        "name": "numpy_native_kernels",
        "task": str(_HERE / "fixtures" / "numpy_native_synthetic"),
        "tier": "tiny",
        "lang": "py",
        "skip_backends": ["cpu", "full", "mem"],
        "skip_reasons": {
            "cpu": "this fixture is meant to validate native kernel attribution",
            "full": "native-only fixture; Scalene signal overlaps synthetic/alloc fixtures",
            "mem": "not memory-focused; alloc_heavy_synthetic covers memray ranking",
        },
        "checks": {
            "native": {
                "any_of": ["openblas", "dgemm", "pypocketfft", "_multiarray"],
                "not_any_of": [
                    "pthread_cond_wait",
                    "__read_nocancel",
                    "__write_nocancel",
                    "__wait4",
                    "sem_wait",
                    "stat",
                    "madvise",
                ],
                "notes_any_of": ["aggregated", "native frames"],
            },
        },
        "max_overhead_x": {"native": 4.0},
    },
    # ---- Medium-tier: Python task at scale, longer runtime, exercises
    # variance / stability behavior. lifelines_cox medium (4M rows × 50d)
    # runs ~10-30s baseline, well above tiny's 1.7s. ----
    {
        "name": "lifelines_cox_medium",
        "task": str(_WORKSPACE / "test_general_bio" / "test_lifelines_cox"),
        "tier": "medium",
        "lang": "py",
        "skip_backends": ["cpu"],  # no with_profile in pipeline/run.py
        "checks": {
            "full":   {"any_of": ["fast_get_efron", "fast_preprocess", "_take_nd"]},
            "mem":    {"any_of": ["_take_nd_ndarray", "fast_get_efron", "normalize"]},
            "native": {
                "any_of": ["_PyEval_EvalFrameDefault", "_multiarray", "prepare_index"],
                "not_any_of": ["pthread_cond_wait"],
            },
        },
        "max_overhead_x": {"full": 15.0, "mem": 3.0, "native": 2.0},
    },
    {
        "name": "wgcna_blockwise_real",
        "task": str(_WORKSPACE / "test_general_bio" / "test_wgcna_blockwise_real"),
        "tier": "tiny",
        "lang": "R",
        "skip_backends": [],
        "checks": {
            "cpu":    {
                "min_hotspots": 1,
                "override_present": ["WGCNA::"],
            },
            "full":   {"min_hotspots": 1},
            # WGCNA does heavy correlation — expect either WGCNA C++ or BLAS
            "native": {"any_of": ["libR.dylib", "WGCNA", "openblas", "matrixStats"]},
        },
        "max_overhead_x": {"cpu": 1.5, "full": 1.5, "mem": 1.5, "native": 1.5},
    },
    # ---- Real task expansion: R/Bioconductor multi-process DESeq2.
    # Exercises Rprof on fork/IPC noise plus override_summary on DESeq2 internals.
    {
        "name": "deseq2_pseudobulk",
        "task": str(_WORKSPACE / "test_general_bio" / "test_deseq2_pseudobulk"),
        "tier": "tiny",
        "lang": "R",
        "skip_backends": [],
        "checks": {
            "cpu": {
                "any_of": ["unserialize", "readChild", "mcfork", "lazyLoadDBfetch"],
                "override_present": [
                    "DESeq2::fitDispWrapper",
                    "DESeq2::fitNbinomGLMs",
                ],
                "actionable_rank1_contains": "DESeq2::fitDispWrapper",
            },
            "full": {
                "min_hotspots": 1,
                "override_present": ["DESeq2::fitDispWrapper"],
                "actionable_rank1_contains": "DESeq2::fitDispWrapper",
            },
            "mem": {
                "min_hotspots": 1,
                "override_present": ["DESeq2::fitDispWrapper"],
                "actionable_rank1_contains": "DESeq2::fitDispWrapper",
            },
            "native": {
                "any_of": [
                    "RunGenCollect",
                    "bcEval_loop",
                    "libsystem_kernel.dylib:write",
                    "libsystem_kernel.dylib:read",
                ],
                "not_any_of": [
                    "__read_nocancel",
                    "__write_nocancel",
                    "__wait4",
                    "sem_wait",
                    "libsystem_kernel.dylib:read",
                    "libsystem_kernel.dylib:write",
                ],
                "override_present": ["DESeq2::fitDispWrapper"],
                "actionable_rank1_contains": "DESeq2::fitDispWrapper",
            },
        },
        "max_overhead_x": {"cpu": 2.5, "full": 2.5, "mem": 2.5, "native": 2.5},
    },
    # ---- Real task expansion: short R/CellChat communication probability.
    # Very small target region; verifies profile still returns schema + override
    # signal even when profiler hotspots are mostly dispatch/runtime frames.
    {
        "name": "cellchat_commprob",
        "task": str(_WORKSPACE / "test_core_singlecell" / "test_cellchat"),
        "tier": "tiny",
        "lang": "R",
        "skip_backends": [],
        "checks": {
            "cpu": {
                "min_hotspots": 1,
                "any_of": [".Call", "exists", "%in%", "computeCommunProb"],
                "override_present": ["CellChat::computeCommunProb"],
                "actionable_rank1_contains": "CellChat::computeCommunProb",
            },
            "full": {
                "min_hotspots": 1,
                "override_present": ["CellChat::computeCommunProb"],
                "actionable_rank1_contains": "CellChat::computeCommunProb",
            },
            "mem": {
                "min_hotspots": 1,
                "override_present": ["CellChat::computeCommunProb"],
                "actionable_rank1_contains": "CellChat::computeCommunProb",
            },
            "native": {
                "min_hotspots": 1,
                "any_of": ["RunGenCollect", "bcEval_loop", "__vfprintf"],
                "not_any_of": [
                    "__read_nocancel",
                    "__write_nocancel",
                    "__wait4",
                    "sem_wait",
                    "libsystem_kernel.dylib:read",
                    "libsystem_kernel.dylib:write",
                ],
                "override_present": ["CellChat::computeCommunProb"],
                "actionable_rank1_contains": "CellChat::computeCommunProb",
            },
        },
        "max_overhead_x": {"cpu": 3.0, "full": 3.0, "mem": 3.0, "native": 3.0},
    },
    # ---- Real task expansion: Seurat marker discovery with RcppParallel.
    # Native should see the injected Wilcoxon worker C++ symbols; actionable
    # ranking should point back to the Seurat API method.
    {
        "name": "seurat_find_all_markers",
        "task": str(_HERE / "fixtures" / "find_all_markers_profile_real"),
        "tier": "tiny",
        "lang": "R",
        "skip_backends": [],
        "checks": {
            "cpu": {
                "min_hotspots": 1,
                "any_of": [".Call", "readRDS", "new_func", "FindAllMarkers"],
                "override_present": ["Seurat::FindAllMarkers"],
                "actionable_rank1_contains": "Seurat::FindAllMarkers",
            },
            "full": {
                "min_hotspots": 1,
                "override_present": ["Seurat::FindAllMarkers"],
                "actionable_rank1_contains": "Seurat::FindAllMarkers",
            },
            "mem": {
                "min_hotspots": 1,
                "override_present": ["Seurat::FindAllMarkers"],
                "actionable_rank1_contains": "Seurat::FindAllMarkers",
            },
            "native": {
                "any_of": ["WilcoxPvalWorker", "sourceCpp", "RunGenCollect", "bcEval_loop"],
                "not_any_of": [
                    "pthread_cond_wait",
                    "__read_nocancel",
                    "__write_nocancel",
                    "__wait4",
                    "sem_wait",
                    "stat",
                    "madvise",
                ],
                "override_present": ["Seurat::FindAllMarkers"],
                "actionable_rank1_contains": "Seurat::FindAllMarkers",
            },
        },
        "max_overhead_x": {"cpu": 3.0, "full": 3.0, "mem": 3.0, "native": 3.0},
    },
    # ---- Real task expansion: MAST GLM / empirical Bayes hot path.
    # Rprof sees transpose/aperm and fork noise; override/actionable keeps
    # the editable MAST entry point at the top.
    {
        "name": "mast_lrtest",
        "task": str(_WORKSPACE / "test_core_singlecell" / "test_mast"),
        "tier": "tiny",
        "lang": "R",
        "skip_backends": [],
        "checks": {
            "cpu": {
                "any_of": ["t.default", "aperm.default", "ebayes", "lrTest"],
                "override_present": ["MAST::lrTest", "MAST::.bayesglm.fit"],
                "actionable_rank1_contains": "MAST::lrTest",
            },
            "full": {
                "any_of": ["t.default", "aperm.default", "ebayes", "lrTest"],
                "override_present": ["MAST::lrTest", "MAST::.bayesglm.fit"],
                "actionable_rank1_contains": "MAST::lrTest",
            },
            "mem": {
                "any_of": ["t.default", "aperm.default", "ebayes", "lrTest"],
                "override_present": ["MAST::lrTest", "MAST::.bayesglm.fit"],
                "actionable_rank1_contains": "MAST::lrTest",
            },
            "native": {
                "any_of": ["do_transpose", "do_aperm", "RunGenCollect", "bcEval_loop"],
                "not_any_of": [
                    "pthread_cond_wait",
                    "__read_nocancel",
                    "__write_nocancel",
                    "__wait4",
                    "sem_wait",
                ],
                "override_present": ["MAST::lrTest", "MAST::.bayesglm.fit"],
                "actionable_rank1_contains": "MAST::lrTest",
            },
        },
        "max_overhead_x": {"cpu": 3.0, "full": 3.0, "mem": 3.0, "native": 3.0},
    },
    # ---- Real task expansion: trajectory inference / principal curves.
    # This one intentionally has no override_summary; native must make the
    # Rcpp princurve kernel visible directly.
    {
        "name": "slingshot_curves",
        "task": str(_WORKSPACE / "test_core_singlecell" / "test_slingshot"),
        "tier": "tiny",
        "lang": "R",
        "skip_backends": [],
        "checks": {
            "cpu": {"any_of": [".Call", "as.double", ".local", "xy.coords"]},
            "full": {"any_of": [".Call", "as.double", "xy.coords", "lazyLoadDBfetch"]},
            "mem": {"any_of": [".Call", "as.double", ".local", "xy.coords"]},
            "native": {
                "any_of": ["project_to_curve", "princurve", "RunGenCollect", "bcEval_loop"],
                "not_any_of": [
                    "pthread_cond_wait",
                    "__read_nocancel",
                    "__write_nocancel",
                    "__wait4",
                    "sem_wait",
                ],
            },
        },
        "max_overhead_x": {"cpu": 2.5, "full": 2.5, "mem": 2.5, "native": 2.5},
    },
    # ---- Real task expansion: PyTorch/Pyro native kernels.
    # cProfile/memray are not the useful first signal here; native must surface
    # torch vector kernels while override_summary points back to the Python API.
    {
        "name": "cell2location_torch_native",
        "task": str(_WORKSPACE / "test_core_singlecell" / "test_cell2location"),
        "tier": "tiny",
        "lang": "py",
        "skip_backends": ["cpu", "full", "mem"],
        "skip_reasons": {
            "cpu": "torch/Pyro workload: cProfile mostly sees Python trainer dispatch",
            "full": "scalene is too expensive for this torch fixture in the default matrix",
            "mem": "cell2loc executor env does not currently include memray",
        },
        "checks": {
            "native": {
                "any_of": ["libtorch_cpu", "Sleef_logf4", "mul_kernel", "div_true_kernel"],
                "not_any_of": ["pthread_cond_wait"],
                "override_present": [
                    "LocationModelLinearDependent",
                    "PyroOptim",
                ],
                "actionable_rank1_contains": "LocationModelLinearDependent",
            },
        },
        "max_overhead_x": {"native": 3.0},
    },
    # ---- Real task expansion: MDAnalysis hydrogen bonds.
    # Useful unresolved case: native sees mixed file I/O / C-extension runtime
    # frames, while override_summary identifies the user-level method.
    {
        "name": "mdanalysis_hbonds_native",
        "task": str(_WORKSPACE / "test_general_bio" / "test_mdanalysis_hbonds"),
        "tier": "tiny",
        "lang": "py",
        "skip_backends": ["cpu", "full", "mem"],
        "skip_reasons": {
            "cpu": "legacy task lacks with_profile(); native is the useful signal here",
            "full": "kept out of the default matrix until a scoped region is added",
            "mem": "mda executor env does not currently include memray",
        },
        "checks": {
            "native": {
                "min_hotspots": 1,
                "not_any_of": [
                    "pthread_cond_wait",
                    "__read_nocancel",
                    "__write_nocancel",
                    "__wait4",
                    "sem_wait",
                    "libsystem_kernel.dylib:read",
                    "libsystem_kernel.dylib:write",
                ],
                "override_present": ["HydrogenBondAnalysis.run"],
                "actionable_rank1_contains": "HydrogenBondAnalysis.run",
            },
        },
        "max_overhead_x": {"native": 3.0},
    },
    # ---- Real task expansion: ObsPy signal filtering.
    # Native should expose scipy/numpy signal kernels instead of generic Python
    # frames; the task's override identifies Stream.filter.
    {
        "name": "obspy_filter_native",
        "task": str(_WORKSPACE / "test_non_bio" / "test_obspy"),
        "tier": "tiny",
        "lang": "py",
        "skip_backends": ["cpu", "full", "mem"],
        "skip_reasons": {
            "cpu": "legacy task lacks with_profile(); native is the useful signal here",
            "full": "kept out of the default matrix until a scoped region is added",
            "mem": "obspy executor env does not currently include memray",
        },
        "checks": {
            "native": {
                "any_of": ["sosfilt", "random_standard_normal", "_multiarray", "pcg64"],
                "not_any_of": [
                    "pthread_cond_wait",
                    "__read_nocancel",
                    "__write_nocancel",
                    "__wait4",
                    "sem_wait",
                ],
                "override_present": ["Stream.filter"],
                "actionable_rank1_contains": "Stream.filter",
            },
        },
        "max_overhead_x": {"native": 3.0},
    },
    # ---- Real task expansion: DIPY diffusion tensor fitting / imaging.
    # Captures Cython/scipy-image/native decompression and math kernels.
    {
        "name": "dipy_dti_native",
        "task": str(_WORKSPACE / "test_general_bio" / "test_dipy_dti"),
        "tier": "tiny",
        "lang": "py",
        "skip_backends": ["cpu", "full", "mem"],
        "skip_reasons": {
            "cpu": "legacy task lacks with_profile(); native is the useful signal here",
            "full": "kept out of the default matrix until a scoped region is added",
            "mem": "default Python executor env does not currently include memray",
        },
        "checks": {
            "native": {
                "any_of": ["NI_RankFilter", "inflate_fast", "logf", "_multiarray"],
                "not_any_of": [
                    "pthread_cond_wait",
                    "__read_nocancel",
                    "__write_nocancel",
                    "__wait4",
                    "sem_wait",
                ],
                "override_present": ["TensorModel.fit", "wls_fit_tensor"],
                "actionable_rank1_contains": "TensorModel.fit",
            },
        },
        "max_overhead_x": {"native": 3.0},
    },
    # ---- Real task expansion: structural biology ANM / sparse eigensolve.
    # Native should surface scipy sparse/eigensolver kernels; actionable top
    # remains the ProDy solve/build methods that the task patches.
    {
        "name": "prody_anm_native",
        "task": str(_WORKSPACE / "test_general_bio" / "test_prody"),
        "tier": "tiny",
        "lang": "py",
        "skip_backends": ["cpu", "full", "mem"],
        "skip_reasons": {
            "cpu": "legacy task lacks with_profile(); native is the useful sparse-kernel signal",
            "full": "kept out of default matrix until a scoped region is added",
            "mem": "whole-process memray would mostly test import/allocation noise here",
        },
        "checks": {
            "native": {
                "any_of": ["csr_matvecs", "_sparsetools", "FLOAT_subtract", "longest_match", "_multiarray"],
                "not_any_of": [
                    "pthread_cond_wait",
                    "__read_nocancel",
                    "__write_nocancel",
                    "__wait4",
                    "sem_wait",
                    "stat",
                    "madvise",
                ],
                "override_present": ["prody.dynamics.anm.solveEig", "ANMBase.buildHessian"],
                "actionable_rank1_contains": "prody.dynamics.anm.solveEig",
            },
        },
        "max_overhead_x": {"native": 3.0},
    },
    # ---- Real task expansion: FiPy finite-volume PDE solve.
    # Native should expose SuperLU/OpenBLAS work; override timing points at
    # Term.solve and matrix assembly/solve internals.
    {
        "name": "fipy_heat_native",
        "task": str(_WORKSPACE / "test_non_bio" / "test_fipy"),
        "tier": "tiny",
        "lang": "py",
        "skip_backends": ["cpu", "full", "mem"],
        "skip_reasons": {
            "cpu": "legacy task lacks with_profile(); native is the useful solver-kernel signal",
            "full": "kept out of default matrix until a scoped region is added",
            "mem": "whole-process memray would mostly test import/allocation noise here",
        },
        "checks": {
            "native": {
                "any_of": ["dgstrs", "superlu", "openblas", "_PyEval"],
                "not_any_of": [
                    "pthread_cond_wait",
                    "__read_nocancel",
                    "__write_nocancel",
                    "__wait4",
                    "sem_wait",
                    "stat",
                    "madvise",
                    "__commpage_gettimeofday",
                ],
                "override_present": [
                    "fipy.terms.term.Term.solve",
                    "_BinaryTerm._buildAndAddMatrices",
                    "LinearLUSolver._solve_",
                ],
                "actionable_rank1_contains": "fipy.terms.term.Term.solve",
            },
        },
        "max_overhead_x": {"native": 3.0},
    },
    # ---- Optional slow real tasks: these mirror longer agent-optimized
    # methods from the workspace. They are excluded from the default matrix
    # because each tiny-tier baseline is ~1 minute or more, but they are
    # valuable when intentionally stress-testing profile on real init targets.
    {
        "name": "vegan_adonis2_slow",
        "task": str(_WORKSPACE / "test_general_bio" / "test_vegan_adonis2"),
        "tier": "tiny",
        "lang": "R",
        "slow": True,
        "skip_backends": [],
        "checks": {
            "cpu": {
                "min_hotspots": 1,
                "any_of": [
                    "adonis2",
                    "permutest.cca",
                    ".Call",
                    ".Fortran",
                    "aperm.default",
                    "t.default",
                    "matrix",
                ],
                "override_present": [
                    "vegan::adonis2",
                    "vegan::adonis0",
                    "vegan::permutest.cca",
                ],
                "actionable_rank1_contains": "vegan::adonis2",
            },
            "full": {
                "min_hotspots": 1,
                "override_present": ["vegan::adonis2"],
                "actionable_any_of": ["vegan::adonis2", "vegan::permutest.cca"],
            },
            "mem": {
                "min_hotspots": 1,
                "override_present": ["vegan::adonis2"],
                "actionable_any_of": ["vegan::adonis2", "vegan::permutest.cca"],
            },
            "native": {
                "not_any_of": [
                    "pthread_cond_wait",
                    "__read_nocancel",
                    "__write_nocancel",
                    "__wait4",
                    "sem_wait",
                ],
                "notes_any_of": ["aggregated", "no sample output captured"],
                "override_present": ["vegan::adonis2"],
                "actionable_any_of": ["vegan::adonis2", "vegan::permutest.cca"],
            },
        },
        "max_overhead_x": {"cpu": 2.0, "full": 2.5, "mem": 2.5, "native": 2.5},
    },
    {
        "name": "maftools_read_maf_slow",
        "task": str(_WORKSPACE / "test_general_bio" / "test_maftools"),
        "tier": "tiny",
        "lang": "R",
        "slow": True,
        "skip_backends": [],
        "checks": {
            "cpu": {
                "min_hotspots": 1,
                "any_of": [
                    "read.maf",
                    "summarizeMaf",
                    "validateMaf",
                    "forderv",
                    "data.table",
                    ".Call",
                    "unique.default",
                ],
                "override_present": [
                    "maftools::read.maf",
                    "maftools::summarizeMaf",
                    "maftools::validateMaf",
                ],
                "actionable_rank1_contains": "maftools::read.maf",
            },
            "full": {
                "min_hotspots": 1,
                "override_present": ["maftools::read.maf"],
                "actionable_any_of": ["maftools::read.maf", "maftools::summarizeMaf"],
            },
            "mem": {
                "min_hotspots": 1,
                "override_present": ["maftools::read.maf"],
                "actionable_any_of": ["maftools::read.maf", "maftools::summarizeMaf"],
            },
            "native": {
                "min_hotspots": 1,
                "not_any_of": [
                    "pthread_cond_wait",
                    "__read_nocancel",
                    "__write_nocancel",
                    "__wait4",
                    "sem_wait",
                ],
                "override_present": ["maftools::read.maf"],
                "actionable_any_of": ["maftools::read.maf", "maftools::summarizeMaf"],
            },
        },
        "max_overhead_x": {"cpu": 2.0, "full": 2.5, "mem": 2.5, "native": 2.5},
    },
    {
        "name": "rctd_spacexr_slow",
        "task": str(_WORKSPACE / "test_core_singlecell" / "test_RCTD"),
        "tier": "tiny",
        "lang": "R",
        "slow": True,
        "skip_backends": ["full", "mem"],
        "skip_reasons": {
            "full": "tiny tier is already ~1 minute; keep this slow fixture to CPU/native signals first",
            "mem": "custom global overrides are the interesting signal; R memory attribution is not the first bottleneck here",
        },
        "checks": {
            "cpu": {
                "min_hotspots": 1,
                "any_of": [
                    "run.RCTD",
                    "fitPixels",
                    "process_bead_doublet",
                    "solveWLS",
                    "calc_Q_all",
                ],
                "override_marker_present": [
                    "spacexr::process_bead_doublet",
                    "spacexr::solveWLS",
                ],
                "actionable_any_of": [
                    "spacexr::process_bead_doublet",
                    "spacexr::solveWLS",
                ],
            },
            "native": {
                "min_hotspots": 1,
                "any_of": [
                    "RunGenCollect",
                    "bcEval_loop",
                    "solve",
                    "arma",
                    "openblas",
                    "libR.dylib",
                ],
                "not_any_of": [
                    "pthread_cond_wait",
                    "__read_nocancel",
                    "__write_nocancel",
                    "__wait4",
                    "sem_wait",
                ],
                "override_marker_present": [
                    "spacexr::process_bead_doublet",
                    "spacexr::solveWLS",
                ],
                "actionable_any_of": [
                    "spacexr::process_bead_doublet",
                    "spacexr::solveWLS",
                ],
            },
        },
        "max_overhead_x": {"cpu": 2.0, "native": 2.5},
    },
]

ALL_BACKENDS = ("cpu", "full", "mem", "native")

REQUIRED_TOP_LEVEL = {
    "schema_version", "backend", "lang", "tier", "hypothesis", "timestamp",
    "totals", "hotspots", "actionable_hotspots", "notes", "artifacts",
    "override_summary", "override_markers", "call_chains",
}
REQUIRED_HOTSPOT_FIELDS = {
    "rank", "label", "self_time_s", "total_time_s", "self_pct", "calls", "raw",
}
REQUIRED_OVERRIDE_FIELDS = {
    "name", "calls", "total_s", "mean_s", "min_s", "max_s", "n_workers",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    fixture: str
    backend: str
    walls: list[float] = field(default_factory=list)
    rc: int = 0
    err: str = ""
    skipped: bool = False
    skip_reason: str = ""
    profile_data: dict | None = None
    check_pass: bool | None = None
    check_detail: str = ""
    overhead_x: float | None = None
    overhead_limit_x: float | None = None
    overhead_pass: bool | None = None

    # Variance: stddev + CV computed across reps. None when reps == 1 (no
    # variance possible). CV > 20% flags the cell as unstable — usually
    # means the workload is too small or the host is loaded.
    wall_stddev_s: float | None = None
    wall_cv_pct: float | None = None
    unstable: bool = False

    def compute_variance(self) -> None:
        if len(self.walls) < 2:
            return
        med = statistics.median(self.walls)
        sd = statistics.stdev(self.walls)
        self.wall_stddev_s = sd
        self.wall_cv_pct = (sd / med * 100.0) if med > 0 else None
        if self.wall_cv_pct is not None and self.wall_cv_pct > 20.0:
            self.unstable = True

    def compute_overhead(self, baseline_s: float | None,
                         limit_x: float | None) -> None:
        """Record overhead against baseline and whether it exceeds limit_x."""
        self.overhead_limit_x = limit_x
        if not self.walls or not baseline_s:
            return
        self.overhead_x = statistics.median(self.walls) / baseline_s
        if limit_x is not None:
            self.overhead_pass = self.overhead_x <= limit_x


# ---------------------------------------------------------------------------
# Run primitives
# ---------------------------------------------------------------------------

def run_zyme(args: list[str], cwd: Path) -> tuple[float, str, str, int]:
    """Invoke `python -m zyme <args>`, return (wall, stdout, stderr, rc)."""
    cmd = [sys.executable, "-m", "zyme", *args]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return time.perf_counter() - t0, proc.stdout, proc.stderr, proc.returncode


def baseline_wall(task_dir: Path, tier: str) -> float | None:
    """One `zyme dryrun` invocation. Returns None on failure (logged)."""
    wall, _stdout, stderr, rc = run_zyme(
        ["dryrun", "--dataset", tier], cwd=task_dir,
    )
    if rc != 0:
        print(f"[bench]   baseline FAILED rc={rc}: {stderr[-200:]!r}", file=sys.stderr)
        return None
    return wall


def profile_run(task_dir: Path, tier: str, backend: str) -> tuple[float, dict | None, str]:
    """One `zyme profile --backend <b> --json` invocation."""
    wall, stdout, stderr, rc = run_zyme(
        ["profile", "--backend", backend, "--dataset", tier,
         "--no-archive", "--json"],
        cwd=task_dir,
    )
    if rc != 0:
        return wall, None, f"rc={rc} stderr_tail={stderr[-200:]!r}"
    try:
        return wall, json.loads(stdout), ""
    except json.JSONDecodeError as e:
        return wall, None, f"non-JSON stdout: {e} | head: {stdout[:200]!r}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def collect_searchable(profile_data: dict, top_n: int = 5,
                       key: str = "hotspots") -> list[tuple[int, str]]:
    """Return [(rank, lower-cased searchable text)] for the top-N hotspots.

    Searchable text combines label + every string field in raw — covers the
    'function name lives in raw.func not label' cases (memray, native).
    """
    out = []
    for h in (profile_data.get(key) or [])[:top_n]:
        bits = [str(h.get("label") or "")]
        for v in (h.get("raw") or {}).values():
            if isinstance(v, (str, int, float)):
                bits.append(str(v))
        out.append((int(h.get("rank") or 0), " ".join(bits).lower()))
    return out


def validate_schema(profile_data: dict | None) -> tuple[bool, str]:
    """Validate the canonical profile.json shape emitted by `zyme profile`.

    This is intentionally structural, not semantic: an empty hotspot list is
    valid for backends/workloads where capture produced no active samples, but
    missing keys or wrong container types are benchmark failures.
    """
    if profile_data is None:
        return False, "profile_data=None"
    if not isinstance(profile_data, dict):
        return False, f"profile_data is {type(profile_data).__name__}, expected dict"

    missing = REQUIRED_TOP_LEVEL - set(profile_data)
    if missing:
        return False, f"schema: missing top-level fields {sorted(missing)}"
    if profile_data.get("schema_version") != "1":
        return False, f"schema: schema_version={profile_data.get('schema_version')!r}"
    if profile_data.get("backend") not in ALL_BACKENDS:
        return False, f"schema: unknown backend {profile_data.get('backend')!r}"
    if profile_data.get("lang") not in ("py", "R"):
        return False, f"schema: unknown lang {profile_data.get('lang')!r}"

    for key in ("totals", "artifacts"):
        if not isinstance(profile_data.get(key), dict):
            return False, f"schema: {key} must be dict"
    for key in (
        "hotspots",
        "actionable_hotspots",
        "notes",
        "override_summary",
        "override_markers",
        "call_chains",
    ):
        if not isinstance(profile_data.get(key), list):
            return False, f"schema: {key} must be list"

    for i, h in enumerate(profile_data.get("hotspots") or []):
        if not isinstance(h, dict):
            return False, f"schema: hotspot[{i}] must be dict"
        missing_hotspot = REQUIRED_HOTSPOT_FIELDS - set(h)
        if missing_hotspot:
            return False, (
                f"schema: hotspot[{i}] missing fields {sorted(missing_hotspot)}"
            )
        if not isinstance(h.get("rank"), int) or h["rank"] < 1:
            return False, f"schema: hotspot[{i}].rank must be positive int"
        if not isinstance(h.get("label"), str):
            return False, f"schema: hotspot[{i}].label must be str"
        if not isinstance(h.get("raw"), dict):
            return False, f"schema: hotspot[{i}].raw must be dict"

    for i, o in enumerate(profile_data.get("override_summary") or []):
        if not isinstance(o, dict):
            return False, f"schema: override_summary[{i}] must be dict"
        missing_override = REQUIRED_OVERRIDE_FIELDS - set(o)
        if missing_override:
            return False, (
                f"schema: override_summary[{i}] missing fields "
                f"{sorted(missing_override)}"
            )
        if not isinstance(o.get("name"), str):
            return False, f"schema: override_summary[{i}].name must be str"

    return True, "schema ok"


def validate(profile_data: dict | None, checks: dict) -> tuple[bool, str]:
    """Run the per-backend assertion bundle. Returns (pass, detail_string)."""
    schema_ok, schema_detail = validate_schema(profile_data)
    if not schema_ok:
        return False, schema_detail
    if not checks:
        return True, "schema ok; no checks declared"

    haystacks = collect_searchable(profile_data, top_n=5)
    full_text = " | ".join(t for _, t in haystacks)
    actionable_haystacks = collect_searchable(
        profile_data, top_n=5, key="actionable_hotspots",
    )
    actionable_text = " | ".join(t for _, t in actionable_haystacks)
    notes_text = " | ".join(str(n).lower() for n in profile_data.get("notes") or [])
    failures: list[str] = []

    if "min_hotspots" in checks:
        n = len(profile_data.get("hotspots") or [])
        if n < checks["min_hotspots"]:
            failures.append(f"min_hotspots: {n} < {checks['min_hotspots']}")

    if "max_hotspots" in checks:
        n = len(profile_data.get("hotspots") or [])
        if n > checks["max_hotspots"]:
            failures.append(f"max_hotspots: {n} > {checks['max_hotspots']}")

    if "any_of" in checks:
        needles = [n.lower() for n in checks["any_of"]]
        if not any(n in full_text for n in needles):
            failures.append(f"any_of: none of {checks['any_of']} found")

    if "all_of" in checks:
        needles = [n.lower() for n in checks["all_of"]]
        missing = [n for n in needles if n not in full_text]
        if missing:
            failures.append(f"all_of: missing {missing}")

    if "not_any_of" in checks:
        needles = [n.lower() for n in checks["not_any_of"]]
        found = [n for n in needles if n in full_text]
        if found:
            failures.append(f"not_any_of: found forbidden {found}")

    if "top3_all" in checks:
        needles = [n.lower() for n in checks["top3_all"]]
        top3_text = " | ".join(t for _, t in haystacks[:3])
        missing = [n for n in needles if n not in top3_text]
        if missing:
            failures.append(f"top3_all: missing {missing} in top-3")

    if "rank1_contains" in checks:
        needle = checks["rank1_contains"].lower()
        if not haystacks:
            failures.append(f"rank1_contains: no hotspots")
        elif needle not in haystacks[0][1]:
            failures.append(f"rank1_contains: top hotspot lacks {needle!r}")

    if "actionable_any_of" in checks:
        needles = [n.lower() for n in checks["actionable_any_of"]]
        if not any(n in actionable_text for n in needles):
            failures.append(
                f"actionable_any_of: none of {checks['actionable_any_of']} found"
            )

    if "actionable_rank1_contains" in checks:
        needle = checks["actionable_rank1_contains"].lower()
        if not actionable_haystacks:
            failures.append("actionable_rank1_contains: no actionable_hotspots")
        elif needle not in actionable_haystacks[0][1]:
            failures.append(
                f"actionable_rank1_contains: top actionable target lacks {needle!r}"
            )

    if "notes_any_of" in checks:
        needles = [n.lower() for n in checks["notes_any_of"]]
        if not any(n in notes_text for n in needles):
            failures.append(f"notes_any_of: none of {checks['notes_any_of']} found")

    if "artifact_contains" in checks:
        artifact_text = " ".join(
            str(v).lower() for v in (profile_data.get("artifacts") or {}).values()
        )
        needle = checks["artifact_contains"].lower()
        if needle not in artifact_text:
            failures.append(
                f"artifact_contains: {needle!r} not found in artifacts "
                f"{profile_data.get('artifacts')}"
            )

    if "override_present" in checks:
        # The fixture's pipeline/run uses install_override on at least one
        # of these names — verify override_summary scraped from log.
        needles = [n.lower() for n in checks["override_present"]]
        overrides = profile_data.get("override_summary") or []
        names = " ".join((o.get("name") or "").lower() for o in overrides)
        missing = [n for n in needles if n not in names]
        if missing:
            failures.append(f"override_present: missing {missing} "
                            f"(saw {[o.get('name') for o in overrides]})")

    if "override_marker_present" in checks:
        needles = [n.lower() for n in checks["override_marker_present"]]
        markers = profile_data.get("override_markers") or []
        names = " ".join((m.get("name") or "").lower() for m in markers)
        missing = [n for n in needles if n not in names]
        if missing:
            failures.append(f"override_marker_present: missing {missing} "
                            f"(saw {[m.get('name') for m in markers]})")

    if "override_min_workers" in checks:
        # Multi-process override aggregation check. For mclapply tasks,
        # this verifies the cross-pid summing actually fired.
        needle_name = checks["override_min_workers"]["name"].lower()
        min_workers = checks["override_min_workers"]["min"]
        overrides = profile_data.get("override_summary") or []
        match = next((o for o in overrides
                      if needle_name in (o.get("name") or "").lower()), None)
        if match is None:
            failures.append(f"override_min_workers: no override matching "
                            f"{needle_name!r}")
        elif match.get("n_workers", 0) < min_workers:
            failures.append(f"override_min_workers: {needle_name} only "
                            f"aggregated across {match.get('n_workers')} "
                            f"workers, expected >= {min_workers}")

    if failures:
        return False, "; ".join(failures)
    return True, "ok"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def bench_fixture(fx: dict, reps: int, only_backends: list[str] | None = None
                  ) -> tuple[float | None, list[CellResult]]:
    name = fx["name"]
    task_dir = Path(fx["task"])
    tier = fx["tier"]
    skip_backends = set(fx.get("skip_backends") or [])
    skip_reasons = fx.get("skip_reasons") or {}
    checks_per_backend = fx.get("checks") or {}
    overhead_limits = fx.get("max_overhead_x") or {}

    print(f"\n{'─' * 78}")
    print(f"FIXTURE: {name}  ({fx['lang']}, tier={tier})")
    print(f"  path: {task_dir}")
    if not task_dir.exists() or not (task_dir / "task.yaml").exists():
        print(f"  ✗ task.yaml not found — fixture skipped")
        return None, []

    # Baseline (1 rep — overhead ratios don't need rep variance for this scale).
    print(f"\n  [baseline] zyme dryrun ...")
    base = baseline_wall(task_dir, tier)
    if base is not None:
        print(f"    wall: {base:.3f}s")
    else:
        print(f"    skipped (failed)")

    cells: list[CellResult] = []
    backends = only_backends if only_backends else ALL_BACKENDS
    for backend in backends:
        cell = CellResult(fixture=name, backend=backend)
        if backend in skip_backends:
            cell.skipped = True
            cell.skip_reason = skip_reasons.get(
                backend,
                "declared in fixture.skip_backends (known to produce no useful output)",
            )
            cells.append(cell)
            print(f"  [{backend}] SKIP — {cell.skip_reason}")
            continue
        print(f"\n  [{backend}] profile × {reps} reps ...")
        last_data = None
        for i in range(reps):
            wall, data, err = profile_run(task_dir, tier, backend)
            cell.walls.append(wall)
            if data is not None:
                last_data = data
            if err:
                cell.err = err
            status = "ok" if data is not None else f"FAIL({err[:50]})"
            print(f"    rep {i+1}/{reps}: {wall:.3f}s [{status}]")
        cell.profile_data = last_data
        cell.check_pass, cell.check_detail = validate(
            last_data, checks_per_backend.get(backend) or {}
        )
        cell.compute_variance()
        cell.compute_overhead(base, overhead_limits.get(backend))
        cells.append(cell)
    return base, cells


def select_fixtures(
    pick_fixtures: set[str] | None,
    *,
    include_slow: bool = False,
) -> list[dict]:
    """Return fixture specs matching CLI selection.

    Slow fixtures are excluded from the implicit default matrix, but an
    explicit --fixtures selection should always win so targeted smoke runs
    do not need an extra flag.
    """
    selected = []
    for fx in FIXTURES:
        explicitly_picked = (
            pick_fixtures is not None and fx["name"] in pick_fixtures
        )
        if pick_fixtures is not None and not explicitly_picked:
            continue
        if fx.get("slow") and not include_slow and not explicitly_picked:
            continue
        selected.append(fx)
    return selected


def result_counts(all_results: list[tuple[dict, float | None, list[CellResult]]]) -> dict[str, int]:
    """Count pass/fail/skip/variance/overhead outcomes for summaries and strict mode."""
    counts = {
        "pass": 0,
        "fail": 0,
        "skip": 0,
        "unstable": 0,
        "overhead_fail": 0,
    }
    for _, _, cells in all_results:
        for c in cells:
            if c.skipped:
                counts["skip"] += 1
            elif c.check_pass is True:
                counts["pass"] += 1
            elif c.check_pass is False:
                counts["fail"] += 1
            if c.unstable:
                counts["unstable"] += 1
            if c.overhead_pass is False:
                counts["overhead_fail"] += 1
    return counts


def emit_summary(all_results: list[tuple[dict, float | None, list[CellResult]]]) -> dict[str, int]:
    print(f"\n{'═' * 92}")
    print("MATRIX SUMMARY")
    print(f"{'═' * 92}")
    has_variance = any(c.wall_stddev_s is not None
                       for _, _, cells in all_results for c in cells)
    if has_variance:
        print(f"  {'fixture':<24} {'backend':<8} {'wall':>9} {'CV%':>6} {'overhead':>9} {'check':>20}")
        print(f"  {'─' * 24} {'─' * 8} {'─' * 9} {'─' * 6} {'─' * 9} {'─' * 20}")
    else:
        print(f"  {'fixture':<24} {'backend':<8} {'wall':>10} {'overhead':>10} {'check':>20}")
        print(f"  {'─' * 24} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 20}")
    counts = result_counts(all_results)
    for fx, base, cells in all_results:
        for c in cells:
            if c.skipped:
                check_str = "SKIP"
            elif c.check_pass is True:
                check_str = "✓ pass"
            elif c.check_pass is False:
                check_str = f"✗ {c.check_detail[:18]}"
            else:
                check_str = "?"
            wall_str = f"{statistics.median(c.walls):.2f}s" if c.walls else "—"
            if c.overhead_x is not None:
                overhead_str = f"{c.overhead_x:.2f}x"
                if c.overhead_pass is False and c.overhead_limit_x is not None:
                    overhead_str += f"!>{c.overhead_limit_x:.1f}"
            else:
                overhead_str = "—"
            if has_variance:
                if c.wall_cv_pct is not None:
                    cv_str = f"{c.wall_cv_pct:.1f}%" + ("⚠" if c.unstable else "")
                else:
                    cv_str = "—"
                print(f"  {c.fixture:<24} {c.backend:<8} {wall_str:>9} "
                      f"{cv_str:>6} {overhead_str:>9} {check_str:>20}")
            else:
                print(f"  {c.fixture:<24} {c.backend:<8} {wall_str:>10} "
                      f"{overhead_str:>10} {check_str:>20}")
    print(f"\n  totals: {counts['pass']} pass, {counts['fail']} fail, "
          f"{counts['skip']} skip"
          + (f", {counts['unstable']} unstable (CV>20%)" if has_variance else "")
          + (f", {counts['overhead_fail']} overhead over limit"
             if counts["overhead_fail"] else ""))
    print()
    if counts["fail"]:
        print("DETAILS (failures):")
        for fx, base, cells in all_results:
            for c in cells:
                if c.check_pass is False:
                    print(f"\n  ✗ {c.fixture} [{c.backend}]")
                    print(f"      reason: {c.check_detail}")
                    if c.profile_data:
                        for h in (c.profile_data.get("hotspots") or [])[:5]:
                            print(f"      top: {h.get('label')[:70]}")
    if counts["overhead_fail"]:
        print("\nDETAILS (overhead over limit):")
        for fx, base, cells in all_results:
            for c in cells:
                if c.overhead_pass is False:
                    print(
                        f"  ! {c.fixture} [{c.backend}] overhead={c.overhead_x:.2f}x "
                        f"limit={c.overhead_limit_x:.2f}x"
                    )
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=3,
                    help="Reps per cell (default 3, takes median for overhead)")
    ap.add_argument("--fixtures", default=None,
                    help="Comma-separated fixture names (default: all)")
    ap.add_argument("--backends", default=None,
                    help="Comma-separated backends (default: cpu,full,mem,native)")
    ap.add_argument("--include-slow", action="store_true",
                    help="Include slow real-task fixtures in the default selection")
    ap.add_argument("--json", default=None,
                    help="Dump structured matrix results to this JSON path")
    ap.add_argument("--strict", action="store_true",
                    help="Exit nonzero if validation fails, overhead exceeds "
                         "a fixture limit, or CV marks a cell unstable")
    args = ap.parse_args()

    pick_fixtures = (set(s.strip() for s in args.fixtures.split(","))
                     if args.fixtures else None)
    pick_backends = (list(s.strip() for s in args.backends.split(","))
                     if args.backends else None)

    selected = select_fixtures(pick_fixtures, include_slow=args.include_slow)
    if not selected:
        print("error: no fixtures matched", file=sys.stderr)
        return 2

    excluded_slow = [
        fx["name"] for fx in FIXTURES
        if fx.get("slow") and pick_fixtures is None and not args.include_slow
    ]
    print(f"running {len(selected)} fixture(s) × "
          f"{len(pick_backends or ALL_BACKENDS)} backend(s) × {args.reps} reps")
    if excluded_slow:
        print(
            "slow fixtures excluded by default: "
            f"{', '.join(excluded_slow)} (use --include-slow or --fixtures)"
        )

    all_results: list[tuple[dict, float | None, list[CellResult]]] = []
    for fx in selected:
        base, cells = bench_fixture(fx, args.reps, only_backends=pick_backends)
        all_results.append((fx, base, cells))

    counts = emit_summary(all_results)

    if args.json:
        out = []
        for fx, base, cells in all_results:
            out.append({
                "fixture": fx["name"], "task": fx["task"], "tier": fx["tier"],
                "lang": fx["lang"], "slow": bool(fx.get("slow")),
                "baseline_wall_s": base,
                "cells": [
                    {
                        "backend": c.backend,
                        "skipped": c.skipped, "skip_reason": c.skip_reason,
                        "walls": c.walls,
                        "wall_median_s": statistics.median(c.walls) if c.walls else None,
                        "wall_stddev_s": c.wall_stddev_s,
                        "wall_cv_pct": c.wall_cv_pct,
                        "unstable": c.unstable,
                        "overhead_x": c.overhead_x,
                        "overhead_limit_x": c.overhead_limit_x,
                        "overhead_pass": c.overhead_pass,
                        "check_pass": c.check_pass,
                        "check_detail": c.check_detail,
                        "err": c.err,
                        "top_5_hotspots": [
                            h.get("label") for h in
                            ((c.profile_data or {}).get("hotspots") or [])[:5]
                        ],
                    }
                    for c in cells
                ],
            })
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\n[bench] wrote structured results to {args.json}")
    if args.strict and (
        counts["fail"] or counts["overhead_fail"] or counts["unstable"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
