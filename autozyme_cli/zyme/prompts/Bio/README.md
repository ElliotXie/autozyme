# prompts/Bio/ — biology / single-cell prompt set (default)

The active prompt set for single-cell / bioinformatics targets. `zyme init` reads from here by default. For non-biology targets, use `prompts/OtherField/` (`zyme init --field OtherField`).

| # | Prompt | Agent role | When | Context |
|---|---|---|---|---|
| 1 | [1_init.md](1_init.md) | Init agent | Once per task, when user runs `zyme init` | Long (reads target repo) |
| 2 | [2_iterate.md](2_iterate.md) | Main agent | Long-running optimization loop. May be re-entered after expand | Long (accumulates over rounds) |
| 2.5 | [2.5_iterate_memory.md](2.5_iterate_memory.md) | Memory-optimization agent | Optional pass after speed converges in phase 2; drives `peak_mb` down within a small speed budget (~0-10% slowdown) | Long |
| 3 | [3_validate_scaling.md](3_validate_scaling.md) | Production-scale validator | Once per task, after dev-loop converges. Adds OOD + xlarge tiers, runs Phase A / Phase B gates, fix loop only on failure. **Dev platform only.** | Medium |
| 3.5 | [3.5_portability.md](3.5_portability.md) | Cross-platform portability agent | Only when `zyme scan --portability` sets `run_3_5: true` in `.zyme/portability_scan.json`. Dev platform only. | Medium |
| 4 | [4_package.md](4_package.md) | Packaging agent | Lift patch, dev smoke, cross-platform `zyme attest`. Reads 3.5 outcome or inline hazard scan. | Medium |
| 5 | [5_reflect.md](5_reflect.md) | Post-task reflection agent | Once per task, after the loop ends. Writes `<framework_root>/reflections/<task>.md` and commits/pushes to `ElliotXie/autozyme-framework` main | Short (reads task artifacts only) |

## Main path

```
1_init  →  2_iterate  →  [2.5_iterate_memory]  →  3_validate_scaling  →  [3.5_portability]  →  4_package  →  zyme bench  →  5_reflect
```

## Design lineage

These prompts encode lessons from autozyme runs against Seurat / Scanpy / celda / DoubletFinder / cellchat targets. Updates land here. See `../../../PROMPT_PHILOSOPHY.md` for editing rules.
