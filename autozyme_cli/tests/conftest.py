"""Shared fixtures for the zyme test suite.

Most tests need a temporary directory that looks like a zyme task: a
task.yaml + a results.tsv + a .zyme/ state dir. Building these by hand
in every test is noisy, so we centralize the scaffolding here.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# --------------------------------------------------------------------------
# task.yaml templates — small, readable, exercising real conventions.
# --------------------------------------------------------------------------

MINIMAL_TASK_YAML = """\
target_repo: https://example.com/foo
target_function: foo

datasets:
  - {tier: tiny, name: tiny_a, path: data/tiny.h5ad}

metrics:
  - {name: speedup, comparator: gte, threshold: 1.0}
"""

FULL_TASK_YAML = """\
target_repo: https://example.com/foo
target_function: foo

datasets:
  - {tier: tiny, name: tiny_a, path: data/tiny.h5ad}
  - {tier: medium, name: medium_a, path: /abs/path/medium.h5ad, params: {n_cells: 1000, n_genes: 500}}
  - {tier: ood_large, name: ood_large_a, path: data/ood.h5ad}

metrics:
  - {name: speedup, comparator: gte, threshold: 1.0}
  - {name: max_diff, comparator: lte, noise_multiplier: 2.0, absolute_floor: 0.05}

threading: default

algorithm_class: stochastic

random_seeds: {primary: 42, noise_calibration: [43, 44, 45]}

intrinsic_noise:
  tiny: {max_diff: 0.0008}
  medium: {max_diff: 0.0024}

scaling_tax_thresholds:
  ood_large_soft: 1.5
  hard_fail: 5.0

executor:
  python: myenv
  rscript: /opt/R-4.4/bin/Rscript

active_mode: default

modes:
  default: {description: "serial", reference_script: reference.R}
  parallel_t8: {description: "mclapply mc.cores=8", reference_script: reference.parallel_t8.R, threads: 8}
"""


# --------------------------------------------------------------------------
# results.tsv content fixtures.
# --------------------------------------------------------------------------

# v0 schema: 11 cols (round..phase). Pre-mode, pre-bench.
RESULTS_TSV_V0 = """\
round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tphase
0\tabc1234\ttiny_a\t10.0\t0.0\t512.0\tbaseline\t{}\tupstream\t\toptimize
1\tdef5678\ttiny_a\t8.0\t20.0\t500.0\tkeep\t{"cpu_sec": 7.5}\t[conservative] X\tnotes\toptimize
1.1\tdef5678\ttiny_a\t8.1\t19.0\t500.0\trerun\t{"cpu_sec": 7.6}\t[conservative] X\trerun\toptimize
2\tghi9012\ttiny_a\t9.0\t10.0\t500.0\tdiscard\t{"cpu_sec": 8.5}\t[algorithmic] Y\trolling back\toptimize
"""

# v2 schema: 12 cols (added thread_mode after phase).
RESULTS_TSV_V2 = """\
round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tphase\tthread_mode
0\tabc1234\ttiny_a\t10.0\t0.0\t512.0\tbaseline\t{}\tupstream\t\toptimize\tdefault
1\tdef5678\ttiny_a\t8.0\t20.0\t500.0\tkeep\t{"cpu_sec": 7.5}\t[conservative] X\tnotes\toptimize\tdefault
0\tabc1234\ttiny_a\t12.0\t0.0\t520.0\tbaseline\t{}\tupstream\t\toptimize\tparallel_t8
1\tdef5678\ttiny_a\t6.0\t50.0\t500.0\tkeep\t{"cpu_sec": 5.5}\t[conservative] X\tnotes\toptimize\tparallel_t8
"""


# --------------------------------------------------------------------------
# Fixtures.
# --------------------------------------------------------------------------

@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    """Empty task dir scaffold: just task.yaml + .zyme/. Tests fill rest."""
    (tmp_path / "task.yaml").write_text(MINIMAL_TASK_YAML)
    (tmp_path / ".zyme").mkdir()
    return tmp_path


@pytest.fixture
def task_dir_full(tmp_path: Path) -> Path:
    """Task dir with the FULL task.yaml — covers modes, executor,
    intrinsic_noise, scaling_tax_thresholds, etc."""
    (tmp_path / "task.yaml").write_text(FULL_TASK_YAML)
    (tmp_path / ".zyme").mkdir()
    return tmp_path


@pytest.fixture
def task_dir_with_results_v0(task_dir: Path) -> Path:
    """task_dir + a v0-schema results.tsv (no thread_mode column)."""
    (task_dir / "results.tsv").write_text(RESULTS_TSV_V0)
    return task_dir


@pytest.fixture
def task_dir_with_results_v2(task_dir_full: Path) -> Path:
    """task_dir_full + a v2-schema results.tsv (with thread_mode)."""
    (task_dir_full / "results.tsv").write_text(RESULTS_TSV_V2)
    return task_dir_full
