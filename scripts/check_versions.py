#!/usr/bin/env python3
"""Verify autozyme_py and autozyme_r declare the same version.

Three places hold the version literal:
  - autozyme_py/pyproject.toml          [project.version]
  - autozyme_py/src/autozyme/__init__.py  __version__
  - autozyme_r/DESCRIPTION              Version:

Run from CI on every push, and locally before tagging a release.
Exits 0 on match, 1 on mismatch with a diff-style report.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent


def read_pyproject() -> str:
    data = tomllib.loads((ROOT / "autozyme_py/pyproject.toml").read_text())
    return data["project"]["version"]


def read_init() -> str:
    text = (ROOT / "autozyme_py/src/autozyme/__init__.py").read_text()
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError("__version__ not found in autozyme/__init__.py")
    return m.group(1)


def read_description() -> str:
    text = (ROOT / "autozyme_r/DESCRIPTION").read_text()
    m = re.search(r"^Version:\s*(\S+)", text, re.MULTILINE)
    if not m:
        raise RuntimeError("Version: not found in autozyme_r/DESCRIPTION")
    return m.group(1)


def main() -> int:
    versions = {
        "autozyme_py/pyproject.toml":         read_pyproject(),
        "autozyme_py/src/autozyme/__init__.py": read_init(),
        "autozyme_r/DESCRIPTION":             read_description(),
    }
    unique = set(versions.values())
    width = max(len(k) for k in versions)
    for path, ver in versions.items():
        print(f"  {path.ljust(width)}  {ver}")
    if len(unique) == 1:
        print(f"\nOK: all three pin {unique.pop()}")
        return 0
    print(f"\nMISMATCH: {sorted(unique)} — bump them in lock-step before release")
    return 1


if __name__ == "__main__":
    sys.exit(main())
