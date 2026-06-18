# AutoZyme

**AutoZyme** is an autonomous multi-agent framework that speeds up scientific software on CPUs while preserving the original results. Install **AutoZyme-Library** for drop-in accelerators of Seurat, Scanpy, and 30+ packages, or run the framework to optimize a function it doesn't ship yet.

<!-- TODO: point docs badge -> docs site once a stable docs URL is chosen (currently placeholder #) -->
[![Preprint](https://img.shields.io/badge/preprint-bioRxiv-brightgreen)](https://www.biorxiv.org/content/10.64898/2026.06.12.731250v1)
[![Docs](https://img.shields.io/badge/docs-available-brightgreen)](#)
[![Website](https://img.shields.io/badge/website-autozyme.com-1f6feb)](https://www.autozyme.com)
[![Datasets](https://img.shields.io/badge/datasets-HuggingFace-ffce1c?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/elliotxie/autozyme-datasets)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Using pre-built patches

Three steps in either language: **install**, **activate** a package, then call its functions exactly as before.

### R

```r
# install
remotes::install_github("ElliotXie/autozyme", subdir = "autozyme_r")

# activate  (Seurat, InferCNV, cellchat, and more)
library(autozyme)
autozyme::activate("seurat")

# run your existing code, now faster
markers <- FindAllMarkers(obj)
```

> The R package compiles C++ kernels at install. This only really trips up **Windows** (no built-in compiler): install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) matching your R version first, or the build fails. macOS (Xcode Command Line Tools) and Linux (gcc) normally already have a compiler.

### Python

```python
# install
pip install autozyme

# activate  (Scanpy, Squidpy, CellPhoneDB, and more)
import autozyme
autozyme.activate("scanpy")

# run your existing code, now faster
sc.tl.rank_genes_groups(adata, "leiden")
```

Activate as many packages as you need. Run `autozyme::dashboard()` (R) or `python -m autozyme` (Python) to list every available patch.

## Optimizing a new function

AutoZyme can also optimize functions it doesn't ship yet, driven by any coding agent that can run shell commands ([Claude Code](https://claude.ai/code), [Codex](https://chatgpt.com/codex), [Cursor](https://cursor.com), and others). The examples below target Seurat's `FindAllMarkers`; swap in your own function and repo. Run it two ways.

### Autonomous (one session)

The manager agent runs the whole pipeline for you: it launches workers, waits for them, checks progress every 15 minutes, and advances through scaffold, init, validate, iterate, scaling, and package. Paste this to your agent:

```
Clone https://github.com/ElliotXie/autozyme.git,
install the CLI (pip install -e autozyme/autozyme_cli/).

I want to optimize FindAllMarkers from Seurat.
Repo: https://github.com/satijalab/seurat

Read and follow autozyme/autozyme_cli/zyme/prompts/manager/0_pipeline.md
```

Typical wall time is 4 to 10 hours, depending on the target function.

### Manual (phase by phase)

Scaffold the task, either by pasting the bootstrap prompt to your agent:

```
I want to speed up FindAllMarkers from Seurat using AutoZyme.
Clone https://github.com/ElliotXie/autozyme.git,
install the CLI (pip install -e autozyme/autozyme_cli/),
then read and follow autozyme/autozyme_cli/zyme/prompts/Bio/0_bootstrap.md
```

or by running the CLI yourself:

```bash
git clone https://github.com/ElliotXie/autozyme.git
cd autozyme && pip install -e autozyme_cli/

mkdir ../test_findallmarkers && cd ../test_findallmarkers
zyme init https://github.com/satijalab/seurat FindAllMarkers
```

Then paste each phase prompt in a fresh agent session:

1. **Init**: `prompts/1_init.md` fills the scaffold, picks datasets, and records baselines. Then run `zyme validate init`.
2. **Iterate**: `prompts/2_iterate.md` runs the 50-round optimization loop. Then run `zyme validate iterate`.
3. **Scaling validation**: `prompts/3_validate_scaling.md` checks OOD and threading robustness.
4. **Package**: `prompts/4_package.md` lifts the converged patch into `autozyme_r` or `autozyme_py`.

## Contributing

AutoZyme-Library grows by community demand, and contributions of every kind are welcome:

- **Request or vote** on what we optimize next at **[autozyme.com](https://www.autozyme.com)**: nominate a function or package, upvote the bottlenecks you hit most, and track requests through optimization, validation, and release.
- **Contribute a patch**: optimize a function yourself with the framework above, then open a pull request to add it to AutoZyme-Library.
- **Share feedback or report bugs**: open a [GitHub issue](https://github.com/ElliotXie/autozyme/issues) with ideas, comments, or problems.

## Datasets

Benchmark datasets are hosted on HuggingFace: [`elliotxie/autozyme-datasets`](https://huggingface.co/datasets/elliotxie/autozyme-datasets)

```bash
hf download elliotxie/autozyme-datasets \
  --repo-type dataset --include "single_cell/raw/pbmc68k.rds"
```

## Citation

```bibtex
@article{xie2026autozyme,
  title   = {AutoZyme: An Autonomous Agentic Framework to Optimize Bioinformatics Software},
  author  = {Xie, Elliot and Cheng, Lingxin and Cai, Yujia and Shireman, Jack and Kendziorski, Christina},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.06.12.731250},
  url     = {https://www.biorxiv.org/content/10.64898/2026.06.12.731250v1}
}
```
