# SoupX Patch Changelog

## 2026-08-01

- Added the `soupx` patch lifted from `test_adjustcounts`.
- Registered fast replacements for `SoupX::adjustCounts()` and `SoupX::expandClusters()` with a narrow default-subtraction contract and upstream fallbacks for unsupported branches.
- Added native CSC water-filling kernels in `src/soupx.cpp`.
- Attested exact output parity on PBMC10k and PBMC20k: 12.1x and 17.0x faster, with peak RSS reduced by 67.1% and 80.1%, respectively.
- Validated the task implementation on six tiers, including bounded real heart and neuron out-of-distribution inputs; no dataset exceeds 20,000 scored cells or 11,843 OOD cells.
