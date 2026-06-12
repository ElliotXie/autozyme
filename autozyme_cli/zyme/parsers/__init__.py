"""zyme.parsers — task.yaml + results.tsv parsing/queries.

Submodules:
  task_yaml.py   — parsing + writing task.yaml fields (datasets, metrics,
                   modes, executor, intrinsic noise, etc.)
  results_tsv.py — reading + writing results.tsv (and verify.tsv columns
                   that share its schema), plus the per-(dataset, mode)
                   history queries used by the CLI commands.

Importers should reach for the specific submodule:

    from zyme.parsers.task_yaml import parse_datasets, parse_metrics
    from zyme.parsers.results_tsv import update_last_status, parse_log

zyme.utils retains a back-compat shim that re-exports the parser names
unchanged, so legacy `from zyme.utils import parse_*` callsites still work.
"""
