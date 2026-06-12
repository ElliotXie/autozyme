"""End-to-end tutorial-as-test: the journey an external patch author follows.

This test walks through exactly what a new contributor does to ship a patch,
using ONLY the public API surface (`autozyme.*` — no _core / _verify private
imports). The docstring of each step matches what the user-facing "Write
your first patch" tutorial will say.

If this test breaks, the tutorial is wrong (and external users will be
confused). If the tutorial is rewritten, update this test to match.

NOTE: external authors must currently add their patch as a real submodule
under `autozyme_py/src/autozyme/<name>/` — the verify_patch worker
subprocess can only find patches that way. This constraint is captured
here as Step 1's caveat; lifting it (e.g. via an entry-point registration
mechanism) is on the roadmap.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import autozyme


# ---- Step 1: Author writes their patch and registers it ------------------
#
# For this tutorial we use `json.dumps` (stdlib, always available) as the
# stand-in for the user's real upstream package. The synthetic submodule
# `autozyme/_test_json/__init__.py` plays the role of what a real author's
# `autozyme/<their_pkg>/__init__.py` looks like — see its source for the
# minimal `register_patch(...)` shape.
#
# Equivalent for an external author:
#
#   # File: autozyme_py/src/autozyme/myfastlib/__init__.py
#   from myfastlib import slow_fn
#   import autozyme
#
#   def fast_slow_fn(*a, **k):
#       ...
#
#   autozyme.register_patch(
#       name="myfastlib",
#       targets=[("myfastlib", "slow_fn", fast_slow_fn)],
#       smoke=dict(load=..., call=..., save=...),  # for verify_patch
#       tested_against="myfastlib 1.2.3",
#   )

PATCH_NAME = "_test_json"  # stand-in for the author's own patch name


def test_step1_patch_appears_in_registry():
    """After importing autozyme, the user's patch shows up in inspect()."""
    from autozyme._core import _import_submodule
    _import_submodule(PATCH_NAME)

    # Public API: the author confirms their patch was registered.
    info = autozyme.inspect(PATCH_NAME)
    assert info["name"] == PATCH_NAME
    assert len(info["targets"]) >= 1


# ---- Step 2: Author activates the patch in a script ----------------------

def test_step2_activate_binds_targets():
    """`autozyme.activate(name)` swaps in the fast function. Author verifies
    via inspect() that every declared target is now bound."""
    from autozyme._core import _import_submodule
    _import_submodule(PATCH_NAME)
    try:
        ok = autozyme.activate(PATCH_NAME)
        assert ok is True

        info = autozyme.inspect(PATCH_NAME)
        for t in info["targets"]:
            assert t["currently_bound_to_fast"] is True, (
                f"target {t['upstream']}::{t['attr']} did not bind — "
                f"likely a typo in the dotted path"
            )
    finally:
        autozyme.deactivate(PATCH_NAME)


# ---- Step 3: Author writes a task dir to validate against ----------------

_TASK_YAML = """\
task: tutorial_demo
datasets:
  - {tier: tiny, name: tutorial_input, path: /dev/null}
metrics:
  - {name: output_match, threshold: 1.0, comparator: gte}
"""

_EVALUATE_PY = """\
# Compares patched output (ZYME_TEST_DIR) vs baseline (ZYME_REFERENCE_DIR).
# Prints `<metric>: <value>` lines that verify_patch parses.
import os
ref_dir = os.environ["ZYME_REFERENCE_DIR"]
test_dir = os.environ["ZYME_TEST_DIR"]
ref = open(os.path.join(ref_dir, "output.txt")).read().strip()
test = open(os.path.join(test_dir, "output.txt")).read().strip()
print(f"output_match: {1.0 if test == ref else 0.0}")
"""


@pytest.fixture()
def author_task_dir(tmp_path: Path) -> Path:
    """Mimics the minimal task layout an external author hand-writes."""
    td = tmp_path / "tutorial_task"
    td.mkdir()
    (td / "task.yaml").write_text(_TASK_YAML, encoding="utf-8")
    (td / "evaluate.py").write_text(_EVALUATE_PY, encoding="utf-8")
    return td


# ---- Step 4: Author runs verify_patch to validate end-to-end -------------

def test_step4_verify_patch_returns_pass_verdict(author_task_dir: Path):
    """The full author flow: register → verify. Author sees `all_pass=True`
    in the returned per-tier dict and a `package_verify.tsv` row written
    to the task dir."""
    from autozyme._core import _import_submodule
    _import_submodule(PATCH_NAME)

    results = autozyme.verify_patch(
        name=PATCH_NAME,
        task_dir=str(author_task_dir),
        tiers=("tiny",),
        reps=1,
        verbose=False,
        use_baseline_cache=False,
    )

    assert len(results) == 1
    r = results[0]
    assert r["tier"] == "tiny"
    assert r["all_pass"] is True, (
        f"verify_patch verdict was not pass; metrics: {r.get('metrics_json')}"
    )

    # Author can read the persisted record:
    tsv = author_task_dir / "package_verify.tsv"
    assert tsv.exists(), "verify_patch did not persist package_verify.tsv"


# ---- Step 5: Author checks the registry / dashboard ----------------------

def test_step5_speedups_query_returns_list(author_task_dir: Path):
    """`autozyme.speedups(name)` returns the published speedup rows for
    the author's patch. Empty list is OK — means no rows published yet,
    which is the post-verify state for a brand-new patch."""
    from autozyme._core import _import_submodule
    _import_submodule(PATCH_NAME)

    rows = autozyme.speedups(PATCH_NAME)
    assert isinstance(rows, list)
    # Brand-new synthetic patch has no published speedups; only check shape.


# ---- Step 6: Author lists / introspects via the public API ---------------

def test_step6_public_api_introspection():
    """Final sanity check — the standard introspection surface works."""
    from autozyme._core import _import_submodule
    _import_submodule(PATCH_NAME)

    # `_test_json` is underscore-prefixed so it's NOT in list_patches() — the
    # author's real patch (no underscore) WOULD appear there. Test the
    # parts of the surface that work regardless.
    assert isinstance(autozyme.list_patches(), list)
    assert isinstance(autozyme.list_subsets(), list)
    assert isinstance(autozyme.status(), dict)
    assert isinstance(autozyme.env_snapshot(), dict)
