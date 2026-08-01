# SoupX Patch Changelog

## 2026-08-01

- Added the `soupx` patch lifted from `test_adjustcounts`.
- Registered fast replacements for `SoupX::adjustCounts()` and `SoupX::expandClusters()` with a narrow default-subtraction contract and upstream fallbacks for unsupported branches.
- Added native CSC water-filling kernels in `src/soupx.cpp`.
- Preserved upstream `...` fallback behavior, aligned cell weights by cell name,
  and hardened native water-filling against zero-weight and near-saturation
  denominator edge cases.
- Attested numerical output parity on PBMC10k and PBMC20k: 12.1x and 17.0x faster, with peak RSS reduced by 67.1% and 80.1%, respectively. A post-hardening PBMC10k rerun remained 11.5x faster with maximum absolute drift `9.1e-12` and relative Frobenius drift `2.5e-16`.
- Validated the task implementation on six tiers, including bounded real heart and neuron out-of-distribution inputs; no dataset exceeds 20,000 scored cells or 11,843 OOD cells.
