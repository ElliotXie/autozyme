"""zyme package — patch packaging utilities.

Subcommands:
  lint            Static checks against CAVEATS-derived rules.
  check-intercept Run smoke under instrumented dispatcher; assert intercepts fired.
  check-versions  Flag patches whose tested_against drifted from installed upstream.
  preflight       Run lint + portability scan + smoke-parity (default attest gate).
  smoke-parity    Compare pipeline/run output vs smoke output via task evaluate.
  sync-manifests  Reconcile UPSTREAMS / .zyme_upstreams against register_patch calls.

Each subcommand exports ``cmd_package_<name>(args)``; the dispatcher lives in
cli.py and is wired through ``commands/__init__.py``.
"""
from zyme.commands.package.lint import cmd_package_lint
from zyme.commands.package.check_intercept import cmd_package_check_intercept
from zyme.commands.package.check_versions import cmd_package_check_versions
from zyme.commands.package.preflight import cmd_package_preflight
from zyme.commands.package.smoke_parity import cmd_package_smoke_parity
from zyme.commands.package.sync_manifests import cmd_package_sync_manifests

__all__ = [
    "cmd_package_lint",
    "cmd_package_check_intercept",
    "cmd_package_check_versions",
    "cmd_package_preflight",
    "cmd_package_smoke_parity",
    "cmd_package_sync_manifests",
]
