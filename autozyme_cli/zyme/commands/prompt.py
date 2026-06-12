"""`zyme prompt` family — versioned snapshots of phase prompts via zyme.registry."""

import sys
import subprocess
from pathlib import Path

from zyme.utils import die, info
from zyme.commands._shared import FRAMEWORK_ROOT




# ============================================================================
# Prompt registry
# ============================================================================
# Background: see autozyme_cli/zyme/registry.py docstring. The registry
# snapshots live phase prompts (autozyme_cli/zyme/prompts/<field>/<N>_<slot>.md)
# into _registry/<field>/<slot>/snapshots/<prompt_id>/, each carrying a
# card.yaml with hypothesis + free-form labels (aggressiveness, thread_breadth,
# rerun_budget, ...). active.lock per slot tracks which snapshot's content is
# currently in the live file.

def _parse_label_kvs(specs):
    """`['aggressiveness=10', 'name=foo']` → `{'aggressiveness': 10.0, 'name': 'foo'}`.

    Numeric values are coerced to float; non-numeric stay as strings.
    """
    out = {}
    for spec in specs or []:
        if "=" not in spec:
            die(f"--label expected k=v, got '{spec}'")
        k, _, v = spec.partition("=")
        k = k.strip()
        v = v.strip()
        if not k:
            die(f"--label key is empty: '{spec}'")
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out




def cmd_prompt_save(args):
    from zyme import registry
    live_path = Path(args.live_path).resolve()
    labels = _parse_label_kvs(args.labels)
    try:
        card = registry.save_snapshot(
            FRAMEWORK_ROOT, live_path, args.name,
            args.message or "", labels, notes=args.notes or "",
        )
    except (registry.RegistryError, ValueError) as e:
        die(str(e))
    info(f"saved: {card['id']}")
    info(f"  field/slot: {card['field']}/{card['slot']}")
    info(f"  parent:     {card.get('parent') or '(none — first snapshot for this slot)'}")
    info(f"  sha256[:8]: {card['content_sha256'][:8]}")
    if labels:
        info(f"  labels:     {' '.join(f'{k}={v}' for k, v in sorted(labels.items()))}")
    if args.message:
        info(f"  hypothesis: {args.message.splitlines()[0][:80]}{'...' if len(args.message) > 80 else ''}")
    info(f"active.lock now → {card['id']}")




def cmd_prompt_list(args):
    from zyme import registry
    filters = []
    for expr in args.label_filters or []:
        try:
            filters.append(registry.parse_label_filter(expr))
        except ValueError as e:
            die(str(e))

    rows = []
    for f, s, pid, card in registry.iter_snapshots(
        FRAMEWORK_ROOT, field=args.field, slot=args.slot
    ):
        labels = card.get("labels") or {}
        if any(not registry.label_matches(labels, k, op, w) for (k, op, w) in filters):
            continue
        active = registry.read_active_lock(FRAMEWORK_ROOT, f, s)
        rows.append((f, s, pid, card, pid == active))

    if not rows:
        info("(no snapshots match)")
        return

    print(f"{'A':1} {'field':<6} {'slot':<14} {'name':<26} {'created':<20}  labels")
    print("-" * 100)
    for f, s, pid, card, is_active in rows:
        marker = "*" if is_active else " "
        name = (card.get("name") or "")[:26]
        created = (card.get("created_at") or "")[:19]
        labels = card.get("labels") or {}
        labstr = " ".join(f"{k}={v}" for k, v in sorted(labels.items())) if labels else "-"
        print(f"{marker} {f:<6} {s:<14} {name:<26} {created:<20}  {labstr}")
        print(f"  id: {pid}")
    print()
    print(f"({len(rows)} snapshot(s); * = active in live file)")




def cmd_prompt_show(args):
    from zyme import registry
    try:
        found = registry.find_snapshot(FRAMEWORK_ROOT, args.prompt_id)
    except ValueError as e:
        die(str(e))
    if found is None:
        die(f"snapshot not found: {args.prompt_id}")
    f, s, pid, card = found
    snap_dir = registry.snapshot_dir_for(FRAMEWORK_ROOT, f, s, pid)
    print(f"# === card.yaml ({pid}) ===")
    print((snap_dir / "card.yaml").read_text(encoding="utf-8").rstrip())
    if not args.card_only:
        print()
        print(f"# === prompt.md ({pid}) ===")
        print((snap_dir / "prompt.md").read_text(encoding="utf-8").rstrip())




def cmd_prompt_diff(args):
    from zyme import registry
    try:
        a = registry.find_snapshot(FRAMEWORK_ROOT, args.id_a)
        b = registry.find_snapshot(FRAMEWORK_ROOT, args.id_b)
    except ValueError as e:
        die(str(e))
    if a is None:
        die(f"snapshot not found: {args.id_a}")
    if b is None:
        die(f"snapshot not found: {args.id_b}")
    pa = registry.snapshot_dir_for(FRAMEWORK_ROOT, a[0], a[1], a[2]) / "prompt.md"
    pb = registry.snapshot_dir_for(FRAMEWORK_ROOT, b[0], b[1], b[2]) / "prompt.md"
    # Use git diff --no-index for colored output. Exit 0 (same) or 1 (differ);
    # both are non-error from our POV.
    rc = subprocess.call(
        ["git", "--no-pager", "diff", "--no-index", "--", str(pa), str(pb)]
    )
    if rc not in (0, 1):
        sys.exit(rc)




def cmd_prompt_use(args):
    from zyme import registry
    try:
        card = registry.install_snapshot(
            FRAMEWORK_ROOT, args.prompt_id, force=args.force
        )
    except (registry.RegistryError, ValueError) as e:
        die(str(e))
    live = registry.live_prompts_dir(FRAMEWORK_ROOT, card["field"]) / Path(card["source_path"]).name
    info(f"installed: {card['id']}")
    info(f"  → {live}")
    info(f"active.lock now → {card['id']}")




def cmd_prompt_annotate(args):
    from zyme import registry
    labels = _parse_label_kvs(args.labels) or None
    if labels is None and args.note is None:
        die("nothing to do — pass --label k=v and/or --note '...'")
    try:
        card = registry.annotate_snapshot(
            FRAMEWORK_ROOT, args.prompt_id, labels=labels, notes=args.note,
        )
    except (registry.RegistryError, ValueError) as e:
        die(str(e))
    info(f"annotated: {card['id']}")
    if labels:
        info(f"  labels merged: {' '.join(f'{k}={v}' for k, v in sorted(labels.items()))}")
    if args.note is not None:
        info(f"  notes: {'(cleared)' if args.note == '' else args.note[:80]}")
