# prompts/OtherField/ — generic prompt set for non-biology targets

For targets outside single-cell / bioinformatics — e.g. ML training kernels, web frameworks, scientific computing libraries in any field. Spawn this set with `zyme init --field OtherField`.

These prompts are **a near-copy of `../Bio/`**, with biology-specific role framing replaced by generic performance-engineer framing. Examples in the prompt body still sometimes reference Seurat / Scanpy because the framework matured against that domain — substitute the analogues from your target's domain. The structural advice (override pattern, three-zone editable scope, hypothesis tags, profile-driven iteration, concordance budgets) is fully domain-agnostic.

| # | Prompt | Agent role | When |
|---|---|---|---|
| 1 | [1_init.md](1_init.md) | Init agent | Once per task, when user runs `zyme init` |
| 2 | [2_iterate.md](2_iterate.md) | Main agent | Long-running optimization loop |
| 2.5 | [2.5_iterate_memory.md](2.5_iterate_memory.md) | Memory-optimization agent | Optional pass after speed converges in phase 2; drives `peak_mb` down within a small speed budget (~0-10% slowdown) |
| 3 | [3_expand_scaling.md](3_expand_scaling.md) | Scaling-expansion agent | Once per task, after dev-loop converges. Dev platform only. |
| 3.5 | [3.5_portability.md](3.5_portability.md) | Cross-platform portability agent | Optional after scaling; dev platform only. See Bio `3.5_portability.md` for same rules. |
| 4 | [4_package.md](4_package.md) | Packaging agent | Lift patch, dev smoke, cross-platform attest |
| 5 | [5_reflect.md](5_reflect.md) | Post-task reflection agent | Once per task, after the loop ends |

## Main path

```
1_init  →  2_iterate  →  [2.5_iterate_memory]  →  3_expand_scaling  →  [3.5_portability]  →  4_package  →  zyme bench  →  5_reflect
```

## Drift policy

`Bio/` is the reference. When `Bio/` gets a structural update (new tag, new helper, new flow), copy the change here too — but skip the biology-specific examples / role framing. If the two diverge structurally beyond a few role lines, that's a signal to refactor: extract the shared body into a single source and keep this dir small.

See `../../../PROMPT_PHILOSOPHY.md` for editing rules.
