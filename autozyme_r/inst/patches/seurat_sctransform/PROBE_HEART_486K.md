# Probe: sctransform patched on heart_adult_486k (full)

Standalone probe run 2026-06-10 on Windows 11 / Ryzen 9 7950X / 127 GB.
NOT yet instrumented through attest -- numbers below are wall-clock from
`Sys.time()` only, peak_mb was not captured properly (R `gc()` estimate
was nonsensical). Re-run via the standard attest harness recommended
before any paper integration.

## Setup

- Dataset: `D:/autosearch/datasets/heart_adult.rds` (Seurat object,
  486,134 cells x 32,383 genes; 1.4 GB on disk, 2.8 GB as `.h5ad`).
- Call: `Seurat::SCTransform(obj, verbose = FALSE, vst.flavor = "v2")`
  after `autozyme::activate("seurat")`.
- Threads: `ZYME_THREADS = 1` (user-level); patched code chose 8 PSOCK
  workers itself ("[autozyme] SCTransform PSOCK ready (8 workers / 16 cores)").

## Result

| step | time |
|---|---|
| RDS load | 13.5 sec |
| **SCTransform patched** | **108.9 sec** |

Output: 486,134 cells x 29,222 SCT genes.

## Why this matters

The sctransform v1 task currently uses `heart_adult_60k.rds` (a subset)
as the `large` tier because earlier full-heart attempts OOM'd or were
not measurable. With the post-reboot 113 GB free baseline, patched now
runs the full 486k object in under 2 minutes -- on the same order as
the existing `medium = pbmc200k_glaucoma` cell (208k cells, ~115 sec
patched). The bottleneck is gene count, not cell count, which is why
2.34x more cells did not 2.34x the time.

## Suggested next step

If Mac team wants to incorporate this as a real benchmark tier, the
clean path is:

1. Re-run through `zyme attest --tiers <heart_full> --reps 3
   --patched-only` after wiring a new tier entry in
   `optimized_task/test_seurat_scanpy/sctransform/v1/task.yaml`
   pointing at `./data/heart_adult.rds` (full).
2. Verify peak_rss via attest's RSS instrumentation (the gc-based
   estimate in the probe script was useless).
3. Decide whether to *replace* the current `large = heart_adult_60k`
   with the full version, or keep both as `large` and `large_full`.

The probe script lives at
`testing/scripts/sct_heart_486k_patched_probe.R` in this repo (copied
from `D:/autosearch/testing/scripts/`).
