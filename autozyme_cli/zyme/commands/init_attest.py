"""`zyme init-attest` — scaffold a post-publication attest task.

Clones the structure of an existing postpublication task (default: find_markers)
into ``postpublication/<name>/``, substitutes the task name, replicates the data
symlink, and prints the short checklist of fields to fill. The cloned
``attest/smoke.R`` / ``evaluate.R`` are working code (not blank stubs), so
adapting a kernel-reuse function (the common case) is a few edits rather than a
from-scratch write.

The scaffold deliberately lands under ``postpublication/`` and is never wired
into the seurat/scanpy attest manifest or the lifted-from index, so
``zyme attest``'s publish step finds no destination and auto-skips: paper
numbers and package speedups stay frozen. See ``postpublication/README.md`` for
the full post-publication workflow (write override in the shipped patch →
``init-attest`` → fill → attest → sync the patch code to release).
"""
import os
import re
import shutil
import sys
from pathlib import Path

from zyme.utils import die

# commands/init_attest.py -> commands -> zyme -> autozyme_cli -> autozyme-framework
_FRAMEWORK = Path(__file__).resolve().parents[3]

# The canonical files that define an attest task. package_verify.tsv (the
# measurement log), reference_output_*/ (cached baselines) and the data symlink
# content are intentionally NOT cloned.
_CLONE_FILES = ("task.yaml", "attest/smoke.R", "evaluate.R", ".gitignore")


def _replace_yaml_field(text: str, field: str, value: str) -> str:
    """Replace the first ``field: ...`` line's value (function repl avoids
    backslash-escape surprises from values like ``Seurat::FindMarkers``)."""
    pattern = re.compile(rf"(?m)^({re.escape(field)}:\s*).*$")
    if not pattern.search(text):
        return text
    return pattern.sub(lambda m: m.group(1) + value, text, count=1)


def _print_next_steps(name: str, like: str, dst: Path, patch: str) -> None:
    rel = dst.relative_to(_FRAMEWORK) if dst.is_relative_to(_FRAMEWORK) else dst
    print(f"[init-attest] scaffolded {rel}/ (cloned from {like})\n")
    print("Fill these, then attest:")
    print(f"  1. {rel}/task.yaml          — target_function, signature, datasets, metrics")
    print(f"  2. {rel}/attest/smoke.R     — the `call` that attest times (and `save` schema)")
    print(f"  3. {rel}/evaluate.R         — metric logic for THIS function vs reference")
    print(f"  4. add the override to the shipped patch (e.g. autozyme_r/inst/patches/{patch}/patch.R),")
    print(f"     then reinstall it (patch-only: copy into the installed package; src change: R CMD INSTALL)")
    print()
    print("  Then:")
    print(f"    KMP_DUPLICATE_LIB_OK=TRUE AUTOZYME_SKIP_RLIBS_CHECK=1 \\")
    print(f"      python -m zyme attest {rel} --name {patch} --lang R \\")
    print(f"      --tiers small,medium,large --threads 1,4,8 --no-preflight")
    print()
    print("  publish auto-skips (not in any manifest); measurement lands in")
    print(f"  {rel}/package_verify.tsv. Paper numbers stay frozen.")


def cmd_init_attest(args):
    name = args.name.strip()
    if not name or "/" in name or name.startswith("."):
        die(f"invalid task name: {args.name!r}")

    # --like always resolves against the canonical postpublication/ (the source
    # of clonable tasks); --dest only controls where the new task is written.
    canonical = _FRAMEWORK / "postpublication"
    dst_parent = Path(args.dest).resolve() if args.dest else canonical
    src = canonical / args.like
    dst = dst_parent / name

    if not (src / "task.yaml").is_file():
        avail = (
            ", ".join(p.name for p in canonical.iterdir() if (p / "task.yaml").is_file())
            if canonical.is_dir() else ""
        )
        die(
            f"--like source has no task.yaml: {src}\n"
            f"   (available under {canonical}: {avail or 'none'})"
        )
    if dst.exists():
        if not getattr(args, "force", False):
            die(f"{dst} already exists; pass --force to overwrite")
        shutil.rmtree(dst)

    for rel in _CLONE_FILES:
        s = src / rel
        if not s.is_file():
            continue
        d = dst / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)

    # Substitute the task name (+ target_function if given) and prepend a TODO
    # banner so the fields to fill are unmissable.
    ty = dst / "task.yaml"
    text = ty.read_text(encoding="utf-8")
    text = _replace_yaml_field(text, "task", name)
    if getattr(args, "target", None):
        text = _replace_yaml_field(text, "target_function", args.target)
    banner = (
        "# TODO(init-attest): fill before attesting —\n"
        "#   target_function, signature (the exact call attest times),\n"
        "#   datasets (tier -> dataset), metrics (thresholds for THIS function).\n"
        f"#   Cloned from '{args.like}'; attest/smoke.R `call` and evaluate.R\n"
        "#   metric logic likely need adapting too. See postpublication/README.md.\n"
    )
    ty.write_text(banner + text, encoding="utf-8")

    # Replicate the data symlink, pointing at the SAME shared data dir the
    # source task resolves to. Copying the source's relative target verbatim
    # would dangle whenever --dest puts the new task at a different depth, so
    # recompute the relative path from this task dir to the resolved target
    # (stays portable, and is identical to the source link in the default case).
    src_data = src / "data"
    if src_data.is_symlink() and not (dst / "data").exists():
        target = os.path.realpath(src_data)
        if not os.path.exists(target):
            print(
                f"[init-attest] WARNING: {args.like}'s data symlink resolves to a "
                f"missing target ({target}); the new data/ link will dangle. "
                "Point it at a real dataset dir before attesting.",
                file=sys.stderr,
            )
        (dst / "data").symlink_to(os.path.relpath(target, start=dst))

    _print_next_steps(name, args.like, dst, args.patch)
    return 0
