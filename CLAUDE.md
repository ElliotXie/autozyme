# autozyme

Autonomous CPU-side speed optimization of scientific computing functions. One target function per task; iterate via the `zyme` CLI.

## Layout

- `autozyme_cli/` — the CLI framework. Install with `pip install -e autozyme_cli/`.
- `autozyme_py/` — Python accelerator package (ships optimized patches).
- `autozyme_r/` — R accelerator package (ships optimized patches).
- `datasets/` — README + setup scripts. Actual data on HuggingFace: `elliotxie/autozyme-datasets`.

## How to use

1. `pip install -e autozyme_cli/`
2. `zyme --version` to verify
3. Follow `autozyme_cli/zyme/prompts/Bio/0_bootstrap.md` to start a new optimization task

## Environment

- R tasks (Seurat, Bioconductor): need R + relevant packages installed.
- Python tasks (Scanpy, etc.): need conda env with the target package.
- Data: downloaded from HuggingFace by the init agent when needed.

## CLI quick reference

```
zyme init <repo> [<function>]    # scaffold a task
zyme run "<hypothesis>"          # run an optimization attempt
zyme accept / reject             # keep or discard
zyme status                      # task state summary
zyme verify                      # thread x tier validation matrix
```
