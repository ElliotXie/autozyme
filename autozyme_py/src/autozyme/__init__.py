"""autozyme: drop-in accelerators for scientific Python packages.

Lazy by default: `import autozyme` enumerates available patches but does
not import or activate any. Call `autozyme.activate("subset_or_patch")`
(or pass a list) to activate.
"""
from __future__ import annotations

import os
import sys

from autozyme._benchmark import benchmark
from autozyme._core import (
    _populate_available,
    activate,
    disabled,
    env_snapshot,
    inspect,
    is_disabled,
    list_patches,
    list_subsets,
    register_patch,
    deactivate,
    deactivate_all,
    status,
    subset,
)
from autozyme._threads import auto_threads, set_threads
from autozyme._utils import resolve_dataset_path
from autozyme._verify import verify_patch
from autozyme._speedups import speedups

__version__ = "0.3.0"
__all__ = [
    "__version__",
    "activate",
    "auto_threads",
    "benchmark",
    "disabled",
    "env_snapshot",
    "inspect",
    "is_disabled",
    "list_patches",
    "list_subsets",
    "register_patch",
    "resolve_dataset_path",
    "deactivate",
    "deactivate_all",
    "set_threads",
    "speedups",
    "status",
    "subset",
    "verify_patch",
]


def _banner() -> None:
    if os.environ.get("AUTOZYME_DISABLED"):
        return
    n_patches = len(list_patches())
    n_subsets = len(list_subsets())
    if n_patches == 0:
        return
    print(
        f"autozyme {__version__}: {n_patches} patches available, "
        f"{n_subsets} subsets — autozyme.activate(<name>) to enable",
        file=sys.stderr,
    )


_populate_available(list(sys.modules[__name__].__path__))
_banner()
