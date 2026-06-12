"""`python -m autozyme` entry point — one-screen environment dashboard.

Prints which patches autozyme ships, which of them have their upstream
installed, the version drift, and the curated subsets. No arguments;
just a snapshot.
"""
from __future__ import annotations

import sys

from autozyme import __version__
from autozyme._core import (
    _REGISTRY,
    _installed_version,
    _probe_patch_installed,
    list_patches,
    list_subsets,
)
from autozyme._subsets import SUBSETS, UPSTREAMS


def _patch_line(name: str) -> tuple[str, str]:
    """Return (status_char, descriptor) for a patch. Cheap probe only —
    does not import the patch submodule."""
    installed, err = _probe_patch_installed(name)
    if not installed:
        return ("x", err or "upstream not installed")
    upstream_pkgs = UPSTREAMS.get(name, [name])
    actual_versions = {pkg: _installed_version(pkg) for pkg in upstream_pkgs}
    ver_str = ", ".join(
        f"{pkg} {v}" if v else f"{pkg} (version unknown)"
        for pkg, v in actual_versions.items()
    )
    # Drift info is only known after the patch has been imported (tested_against
    # lives in register_patch). Dashboard doesn't force-import, so we just show
    # installed versions; drift surfaces at activate() time via the marker.
    return ("v", ver_str)


def main() -> int:
    patches = list_patches()
    subsets = list_subsets()
    print(f"autozyme {__version__}")
    print(f"{len(patches)} patches discovered:")
    if patches:
        name_width = max(len(n) for n in patches)
        installed_count = 0
        for name in patches:
            ch, desc = _patch_line(name)
            if ch == "v":
                installed_count += 1
            print(f"  {ch} {name.ljust(name_width)}  {desc}")
        print(f"  ({installed_count}/{len(patches)} with upstream installed)")
    else:
        print("  (none — this autozyme install ships zero patches)")
    print()
    print(f"{len(subsets)} subsets:")
    for s in subsets:
        members = SUBSETS[s]
        installed_in_subset = sum(
            1 for m in members if _probe_patch_installed(m)[0]
        )
        print(f"  {s}: {', '.join(members)}  ({installed_in_subset}/{len(members)} installed)")
    print()
    print("Activate with: autozyme.activate('<name>') / autozyme.activate('<subset>')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
