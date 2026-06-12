"""Public API signature snapshot.

Every callable in `autozyme.__all__` is pinned by its `inspect.signature`
string. Refactors that accidentally rename a parameter, change a default,
drop a kwarg, or alter a type annotation will fail this test loudly in PR
CI — forcing an explicit snapshot update + reviewer attention.

Intentional API breakage:
  1. Update tests/api_signatures.json (regen with the snippet below)
  2. Bump autozyme.__version__ if breaking (PEP 440 semver)
  3. Note the change in the PR description

Regenerate the snapshot:
  python -c "import json, inspect, autozyme; \
    print(json.dumps({n: str(inspect.signature(getattr(autozyme, n))) \
      for n in autozyme.__all__ \
      if callable(getattr(autozyme, n))}, indent=2, sort_keys=True))" \
    > autozyme_py/tests/api_signatures.json
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import autozyme

_SNAPSHOT_PATH = Path(__file__).parent / "api_signatures.json"


def _current_signatures() -> dict[str, str]:
    return {
        name: str(inspect.signature(getattr(autozyme, name)))
        for name in autozyme.__all__
        if callable(getattr(autozyme, name))
    }


def test_public_api_signatures_match_snapshot():
    current = _current_signatures()
    snapshot = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))

    added = sorted(set(current) - set(snapshot))
    removed = sorted(set(snapshot) - set(current))
    changed = sorted(
        n for n in set(current) & set(snapshot)
        if current[n] != snapshot[n]
    )

    if added or removed or changed:
        msg_lines = ["Public API signatures drifted from snapshot."]
        if added:
            msg_lines.append(f"  ADDED:   {added}")
        if removed:
            msg_lines.append(f"  REMOVED: {removed}")
        for n in changed:
            msg_lines.append(f"  CHANGED  {n}:")
            msg_lines.append(f"    snapshot: {snapshot[n]}")
            msg_lines.append(f"    current:  {current[n]}")
        msg_lines.append(
            "\nIf intentional: regenerate autozyme_py/tests/api_signatures.json"
            " (see module docstring for snippet) and bump __version__ if "
            "breaking."
        )
        raise AssertionError("\n".join(msg_lines))


def test_all_exports_are_importable():
    """Every name in __all__ must actually exist on the module — catches
    typos and removed exports that __all__ still references."""
    missing = [n for n in autozyme.__all__ if not hasattr(autozyme, n)]
    assert not missing, f"names in __all__ but not on module: {missing}"
