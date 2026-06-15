# <TASK_NAME>

<ONE-PARAGRAPH WHAT THIS TASK OPTIMIZES + WHY IT'S A BOTTLENECK>

> Structured facts (target_repo, target_function, signature, datasets, metrics) live in `task.yaml`; round-0 baselines in `results.tsv`. This file is for narrative reasoning that doesn't fit a YAML cell. Replace every `<...>` placeholder with concrete content; a finished README has none left.

## Upstream source

- **Local clone:** `<task_dir>/upstream_repo/` (or absolute path if init agent used a pre-existing clone)
- **Commit SHA:** `<SHA>` — patches are measured against this exact state.
- **Version tag (if applicable):** `<v2.0.6 / 0.4.0 / etc.>`

## Call chain (top callees)

`<FN>` → `<FN_LEVEL_2>` → `<FN_LEVEL_3>`

## Why this is worth optimizing

<Profile snippet (function-level wall-time table) + one-paragraph headroom estimate. Iterate agent's main reference for picking angles — be concrete.>

## Dev dataset

<Prose explanation of dataset choices, complementing `task.yaml::datasets`: why these datasets, why this size progression, what regeneration script exists. Skip the section if choices are obvious from `task.yaml`.>

## Anticipated angles

<3 init-agent leads, each tagged `[conservative]` or `[algorithmic]`. Iterate agent expands or invalidates as it goes.>

1. **[conservative|algorithmic]** <ANGLE — profile-backed reason>
2. **[conservative|algorithmic]** <ANGLE>
3. **[conservative|algorithmic]** <ANGLE>
