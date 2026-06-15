"""zyme.commands — implementations of all CLI subcommands.

Each subcommand lives in its own module; this __init__ re-exports them
so cli.py can do `from zyme.commands import cmd_*` unchanged.
"""
from zyme.commands.init import cmd_init
from zyme.commands.init_check import cmd_init_check
from zyme.commands.init_attest import cmd_init_attest
from zyme.commands.baseline import (
    cmd_record_baseline, cmd_reference, cmd_record_noise, cmd_promote_baseline,
    cmd_baseline_list, cmd_baseline_show,
)
from zyme.commands.baseline_rebench import cmd_baseline_rebench
from zyme.commands.dispatch import cmd_dispatch_wait
from zyme.commands.run import cmd_run, cmd_dryrun, cmd_accept, cmd_reject, cmd_rollback
from zyme.commands.iterate import cmd_iterate
from zyme.commands.plot import cmd_plot
from zyme.commands.verify import cmd_verify
from zyme.commands.attest import cmd_attest
from zyme.commands.attest_sweep import cmd_attest_sweep
from zyme.commands.backfill import cmd_backfill
from zyme.commands.publish_speedups import cmd_publish_speedups
from zyme.commands.inspect_parallelism import cmd_inspect_parallelism
from zyme.commands.status import cmd_status
from zyme.commands.scan import cmd_scan
from zyme.commands.registry import (
    cmd_registry_rebuild, cmd_registry_query, cmd_registry_suggest, cmd_registry_list,
)
from zyme.commands.dispatch import (
    cmd_dispatch, cmd_dispatch_status, cmd_dispatch_usage,
    cmd_dispatch_prices, cmd_dispatch_resume, cmd_dispatch_logs, cmd_dispatch_stop,
)
from zyme.commands.prompt import (
    cmd_prompt_save, cmd_prompt_list, cmd_prompt_show,
    cmd_prompt_diff, cmd_prompt_use, cmd_prompt_annotate,
)
from zyme.commands.bench import (
    cmd_bench_register_template, cmd_bench_list_templates,
    cmd_bench_doctor, cmd_bench_status, cmd_bench_init, cmd_bench_start,
    cmd_bench_usage, cmd_bench_prices, cmd_bench_list,
)
from zyme.commands.audit import cmd_audit
from zyme.commands.cost import cmd_cost, cmd_cost_capture
from zyme.commands.validate import cmd_validate_init, cmd_validate_iterate
from zyme.commands.profile import cmd_profile
from zyme.commands.report import cmd_report
from zyme.commands.datasets import cmd_datasets_migrate, add_datasets_subparser
from zyme.commands.package import (
    cmd_package_lint,
    cmd_package_check_intercept,
    cmd_package_check_versions,
    cmd_package_preflight,
    cmd_package_smoke_parity,
    cmd_package_sync_manifests,
)

__all__ = [
    "cmd_init", "cmd_init_check", "cmd_init_attest",
    "cmd_record_baseline", "cmd_reference", "cmd_record_noise", "cmd_promote_baseline",
    "cmd_baseline_list", "cmd_baseline_show", "cmd_baseline_rebench",
    "cmd_run", "cmd_dryrun", "cmd_accept", "cmd_reject", "cmd_rollback", "cmd_iterate",
    "cmd_plot", "cmd_verify", "cmd_attest", "cmd_attest_sweep", "cmd_backfill",
    "cmd_publish_speedups", "cmd_inspect_parallelism",
    "cmd_status", "cmd_scan",
    "cmd_registry_rebuild", "cmd_registry_query", "cmd_registry_suggest", "cmd_registry_list",
    "cmd_dispatch", "cmd_dispatch_status", "cmd_dispatch_usage",
    "cmd_dispatch_prices", "cmd_dispatch_resume", "cmd_dispatch_logs", "cmd_dispatch_stop",
    "cmd_prompt_save", "cmd_prompt_list", "cmd_prompt_show",
    "cmd_prompt_diff", "cmd_prompt_use", "cmd_prompt_annotate",
    "cmd_bench_register_template", "cmd_bench_list_templates",
    "cmd_bench_doctor", "cmd_bench_status", "cmd_bench_init", "cmd_bench_start",
    "cmd_bench_usage", "cmd_bench_prices", "cmd_bench_list",
    "cmd_audit",
    "cmd_cost", "cmd_cost_capture",
    "cmd_validate_init", "cmd_validate_iterate",
    "cmd_profile",
    "cmd_report",
    "cmd_datasets_migrate", "add_datasets_subparser",
    "cmd_package_lint", "cmd_package_check_intercept",
    "cmd_package_check_versions",
    "cmd_package_preflight",
    "cmd_package_smoke_parity", "cmd_package_sync_manifests",
]
