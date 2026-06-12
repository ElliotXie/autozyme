"""Prompt registry — versioned snapshots of phase prompts with semantic labels.

Primary layout when a workspace PromptLab exists:

    PromptLab/
      prompt_snapshots/
        Bio/
          iterate/
            index.yaml
            active.lock
            snapshots/
              zyme_iterate_<date>_<name>_<sha8>/
                prompt.md
                card.yaml

Legacy layout under FRAMEWORK_ROOT (= autozyme_cli/zyme/) is still readable:

    prompts/
      Bio/                       # live prompts (untouched)
      OtherField/
      _registry/
        Bio/
          iterate/
            index.yaml           # ordered prompt_id list
            active.lock          # the prompt_id whose content == live file
            snapshots/
              p_iterate_<date>_<name>_<sha8>/
                prompt.md
                card.yaml

Slot derivation: strip the leading `N_` (or `N.k_`) and the `.md` extension
from a live prompt filename. e.g. `2_iterate.md` -> `iterate`,
`6.1_transfer_init.md` -> `transfer_init`. `6.2_iterate.md` is treated as
`transfer_iterate` so it does not collide with the main `2_iterate.md` slot.
Letter-prefixed helper prompts such as `M_thread_baseline_fairness.md` are
also accepted.

This module is pure file I/O. CLI dispatch lives in commands.py.
"""
import hashlib
import re
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------------

def registry_root(framework_root: Path) -> Path:
    """Primary write root for prompt snapshots.

    Prompt experiments belong with PromptLab when that workspace exists. The
    old framework-local registry remains readable through `_registry_roots()`
    so existing p_* snapshots keep working.
    """
    promptlab = framework_root.parent.parent.parent / "PromptLab"
    if promptlab.is_dir():
        return promptlab / "prompt_snapshots"
    return _legacy_registry_root(framework_root)


def _legacy_registry_root(framework_root: Path) -> Path:
    return framework_root / "prompts" / "_registry"


def _registry_roots(framework_root: Path) -> list[Path]:
    roots = [registry_root(framework_root), _legacy_registry_root(framework_root)]
    out = []
    seen = set()
    for root in roots:
        key = str(root.resolve() if root.exists() else root)
        if key not in seen:
            out.append(root)
            seen.add(key)
    return out


def live_prompts_dir(framework_root: Path, field: str) -> Path:
    return framework_root / "prompts" / field


def slot_dir(framework_root: Path, field: str, slot: str) -> Path:
    return registry_root(framework_root) / field / slot


def snapshots_dir(framework_root: Path, field: str, slot: str) -> Path:
    return slot_dir(framework_root, field, slot) / "snapshots"


def index_path(framework_root: Path, field: str, slot: str) -> Path:
    return slot_dir(framework_root, field, slot) / "index.yaml"


def active_lock_path(framework_root: Path, field: str, slot: str) -> Path:
    return slot_dir(framework_root, field, slot) / "active.lock"


def snapshot_dir_for(framework_root: Path, field: str, slot: str, prompt_id: str) -> Path:
    for root in _registry_roots(framework_root):
        p = root / field / slot / "snapshots" / prompt_id
        if p.exists():
            return p
    return snapshots_dir(framework_root, field, slot) / prompt_id


# ----------------------------------------------------------------------------
# Slot / field detection
# ----------------------------------------------------------------------------

_SLOT_RE = re.compile(r"^(?:[\d.]+|[A-Za-z]+)_(.+)\.md$")


def slot_from_filename(filename: str) -> str:
    """`2_iterate.md` -> `iterate`. `6.1_transfer_init.md` -> `transfer_init`."""
    m = _SLOT_RE.match(filename)
    if not m:
        raise ValueError(
            f"prompt filename '{filename}' doesn't match `<N>_<slot>.md`, "
            f"`<N.k>_<slot>.md`, or `<Letter>_<slot>.md`; cannot derive slot "
            f"for registry."
        )
    slot = m.group(1)
    if filename.startswith("6.2_") and slot == "iterate":
        return "transfer_iterate"
    return slot


def field_from_path(prompt_path: Path, framework_root: Path) -> str:
    """Given .../prompts/Bio/2_iterate.md return 'Bio'."""
    prompts_root = (framework_root / "prompts").resolve()
    try:
        rel = prompt_path.resolve().relative_to(prompts_root)
    except ValueError:
        raise ValueError(
            f"prompt path '{prompt_path}' is not under {prompts_root}; "
            f"can only register prompts that live in the framework's prompts/ tree."
        )
    parts = rel.parts
    if len(parts) < 2 or parts[0] == "_registry":
        raise ValueError(
            f"prompt path '{prompt_path}' must be like prompts/<field>/<file>.md"
        )
    return parts[0]


# ----------------------------------------------------------------------------
# IDs
# ----------------------------------------------------------------------------

def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SLUG_BAD = re.compile(r"[^a-zA-Z0-9_]+")


def slugify(name: str) -> str:
    s = _SLUG_BAD.sub("_", name).strip("_")
    return (s[:40] if s else "unnamed").lower()


def make_prompt_id(slot: str, name: str, content_sha: str) -> str:
    """`zyme_<slot>_<YYYYMMDD>_<name_slug>_<sha8>`."""
    date = datetime.now().strftime("%Y%m%d")
    return f"zyme_{slot}_{date}_{slugify(name)}_{content_sha[:8]}"


# ----------------------------------------------------------------------------
# Mini YAML — handles only the card.yaml shape we emit.
# ----------------------------------------------------------------------------
# Supported shapes:
#   key: scalar
#   key: |               (literal block)
#     line1
#     line2
#   key:                 (nested mapping)
#     subkey1: scalar
#     subkey2: scalar
#   key: {}              (empty mapping)
#   key: null            (None)
# Comments (#) and blank lines ignored. No flow-style sequences.

_NEEDS_QUOTE = re.compile(r"[:#\{\}\[\],\n]")


def _yaml_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    s = str(v)
    if s == "":
        return '""'
    if _NEEDS_QUOTE.search(s) or s.lower() in ("true", "false", "null", "yes", "no"):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def _parse_scalar(s: str):
    s = s.strip()
    if s == "" or s == '""':
        return ""
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        return s[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if s == "null":
        return None
    if s == "true":
        return True
    if s == "false":
        return False
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def _emit_block(key: str, text: str, lines: list) -> None:
    if not text:
        lines.append(f'{key}: ""')
        return
    lines.append(f"{key}: |")
    for ln in text.splitlines() or [""]:
        lines.append(f"  {ln}")


def write_card_yaml(snap_dir: Path, card: dict) -> None:
    """Emit card.yaml in a fixed key order."""
    lines = []
    for key in ("id", "slot", "field", "source_path", "parent",
                "content_sha256", "name", "created_at"):
        lines.append(f"{key}: {_yaml_scalar(card.get(key))}")
    _emit_block("hypothesis", card.get("hypothesis", "") or "", lines)
    labels = card.get("labels") or {}
    if labels:
        lines.append("labels:")
        for lk in sorted(labels):
            lines.append(f"  {lk}: {_yaml_scalar(labels[lk])}")
    else:
        lines.append("labels: {}")
    _emit_block("notes", card.get("notes", "") or "", lines)
    (snap_dir / "card.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_card_yaml(card_path: Path) -> dict:
    text = card_path.read_text(encoding="utf-8")
    out: dict = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if line[:1] == " ":
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "|":
            i += 1
            buf = []
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith("  "):
                    buf.append(nxt[2:])
                    i += 1
                elif nxt.strip() == "":
                    buf.append("")
                    i += 1
                else:
                    break
            while buf and buf[-1] == "":
                buf.pop()
            out[key] = "\n".join(buf)
            continue
        if val == "":
            i += 1
            sub: dict = {}
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith("  "):
                    s = nxt.strip()
                    if not s or s.startswith("#"):
                        i += 1
                        continue
                    sk, _, sv = s.partition(":")
                    sub[sk.strip()] = _parse_scalar(sv)
                    i += 1
                elif nxt.strip() == "":
                    i += 1
                else:
                    break
            out[key] = sub
            continue
        if val == "{}":
            out[key] = {}
            i += 1
            continue
        out[key] = _parse_scalar(val)
        i += 1
    return out


# ----------------------------------------------------------------------------
# active.lock + index.yaml I/O
# ----------------------------------------------------------------------------

def read_active_lock(framework_root: Path, field: str, slot: str):
    for root in _registry_roots(framework_root):
        p = root / field / slot / "active.lock"
        if p.exists():
            val = p.read_text(encoding="utf-8").strip()
            return val or None
    return None


def write_active_lock(framework_root: Path, field: str, slot: str, prompt_id: str) -> None:
    p = active_lock_path(framework_root, field, slot)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(prompt_id + "\n", encoding="utf-8")


def append_to_index(framework_root: Path, field: str, slot: str, prompt_id: str) -> None:
    p = index_path(framework_root, field, slot)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"- {prompt_id}\n")


def read_index(framework_root: Path, field: str, slot: str) -> list:
    p = index_path(framework_root, field, slot)
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln.startswith("- "):
            out.append(ln[2:].strip())
    return out


# ----------------------------------------------------------------------------
# Listing / lookup
# ----------------------------------------------------------------------------

def iter_snapshots(framework_root: Path, field: str = None, slot: str = None):
    """Yield (field, slot, prompt_id, card_dict) tuples."""
    seen = set()
    for root in _registry_roots(framework_root):
        if not root.exists():
            continue
        fields = [field] if field else sorted(
            p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        for f in fields:
            f_dir = root / f
            if not f_dir.is_dir():
                continue
            slots = [slot] if slot else sorted(
                p.name for p in f_dir.iterdir() if p.is_dir()
            )
            for s in slots:
                sd = root / f / s / "snapshots"
                if not sd.exists():
                    continue
                for entry in sorted(sd.iterdir()):
                    card_path = entry / "card.yaml"
                    if not (entry.is_dir() and card_path.exists()):
                        continue
                    key = (f, s, entry.name)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield (f, s, entry.name, read_card_yaml(card_path))


def find_snapshot(framework_root: Path, ident: str):
    """Look up a snapshot by exact prompt_id, exact name, or unique substring.

    Returns (field, slot, prompt_id, card) or None. Raises ValueError on ambiguity.
    """
    exact = []
    name_match = []
    substr = []
    for f, s, pid, card in iter_snapshots(framework_root):
        if pid == ident:
            exact.append((f, s, pid, card))
        elif card.get("name") == ident:
            name_match.append((f, s, pid, card))
        elif ident in pid or ident in (card.get("name") or ""):
            substr.append((f, s, pid, card))
    for bucket in (exact, name_match, substr):
        if len(bucket) == 1:
            return bucket[0]
        if len(bucket) > 1:
            ids = [m[2] for m in bucket]
            raise ValueError(f"identifier '{ident}' is ambiguous: {ids}")
    return None


# ----------------------------------------------------------------------------
# Save / install
# ----------------------------------------------------------------------------

class RegistryError(Exception):
    pass


def save_snapshot(framework_root: Path, live_path: Path, name: str,
                  hypothesis: str, labels: dict, notes: str = "") -> dict:
    """Snapshot live_path content into the registry. Returns the new card."""
    if not live_path.exists():
        raise RegistryError(f"live prompt not found: {live_path}")
    content = live_path.read_text(encoding="utf-8")
    sha = sha256_of(content)
    field = field_from_path(live_path, framework_root)
    slot = slot_from_filename(live_path.name)

    parent = read_active_lock(framework_root, field, slot)
    if parent is not None:
        parent_card_path = snapshot_dir_for(framework_root, field, slot, parent) / "card.yaml"
        if parent_card_path.exists():
            parent_card = read_card_yaml(parent_card_path)
            if parent_card.get("content_sha256") == sha:
                raise RegistryError(
                    f"live file content is identical to current active snapshot "
                    f"({parent}). Edit the file before saving a new version."
                )

    prompt_id = make_prompt_id(slot, name, sha)
    snap_dir = snapshot_dir_for(framework_root, field, slot, prompt_id)
    if snap_dir.exists():
        raise RegistryError(
            f"snapshot directory already exists: {snap_dir}. "
            f"Pick a different --as name or wait one second (id contains date)."
        )
    snap_dir.mkdir(parents=True, exist_ok=False)
    (snap_dir / "prompt.md").write_text(content, encoding="utf-8")

    framework_repo_root = framework_root.parent.parent  # autozyme-framework/
    try:
        rel_source = str(live_path.resolve().relative_to(framework_repo_root.resolve()))
    except ValueError:
        rel_source = str(live_path)

    card = {
        "id": prompt_id,
        "slot": slot,
        "field": field,
        "source_path": rel_source,
        "parent": parent,
        "content_sha256": sha,
        "name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hypothesis": hypothesis or "",
        "labels": labels or {},
        "notes": notes or "",
    }
    write_card_yaml(snap_dir, card)
    append_to_index(framework_root, field, slot, prompt_id)
    write_active_lock(framework_root, field, slot, prompt_id)
    return card


def install_snapshot(framework_root: Path, prompt_id: str, force: bool = False) -> dict:
    """Copy a snapshot's prompt.md back to its live path. Returns the card."""
    found = find_snapshot(framework_root, prompt_id)
    if found is None:
        raise RegistryError(f"snapshot not found: {prompt_id}")
    field, slot, pid, card = found
    snap_dir = snapshot_dir_for(framework_root, field, slot, pid)

    live_path = live_prompts_dir(framework_root, field) / Path(card["source_path"]).name
    if not live_path.exists():
        raise RegistryError(
            f"live target not found: {live_path}. Did the prompt filename change?"
        )

    if not force:
        live_sha = sha256_of(live_path.read_text(encoding="utf-8"))
        active = read_active_lock(framework_root, field, slot)
        if active and active != pid:
            active_card_path = snapshot_dir_for(framework_root, field, slot, active) / "card.yaml"
            if active_card_path.exists():
                active_card = read_card_yaml(active_card_path)
                if active_card.get("content_sha256") != live_sha:
                    raise RegistryError(
                        f"live file '{live_path.name}' has unsaved changes "
                        f"(sha mismatch with active snapshot {active}). "
                        f"Save first with `zyme prompt save`, or pass --force to overwrite."
                    )

    src_md = snap_dir / "prompt.md"
    live_path.write_text(src_md.read_text(encoding="utf-8"), encoding="utf-8")
    write_active_lock(framework_root, field, slot, pid)
    return card


def annotate_snapshot(framework_root: Path, prompt_id: str,
                      labels: dict = None, notes: str = None) -> dict:
    """Update labels (merged) and/or notes (replaced) on an existing snapshot."""
    found = find_snapshot(framework_root, prompt_id)
    if found is None:
        raise RegistryError(f"snapshot not found: {prompt_id}")
    field, slot, pid, card = found
    snap_dir = snapshot_dir_for(framework_root, field, slot, pid)

    if labels:
        merged = dict(card.get("labels") or {})
        merged.update(labels)
        card["labels"] = merged
    if notes is not None:
        card["notes"] = notes
    write_card_yaml(snap_dir, card)
    return card


# ----------------------------------------------------------------------------
# Label filter parsing — supports k=v, k>=v, k<=v, k>v, k<v
# ----------------------------------------------------------------------------

_LABEL_OP_RE = re.compile(r"^([^<>=]+?)\s*(>=|<=|==|=|>|<)\s*(.+)$")


def parse_label_filter(expr: str):
    """Parse 'aggressiveness>=8' into ('aggressiveness', '>=', 8).

    Numeric RHS is parsed as float; non-numeric stays a string (only `=`/`==` valid).
    """
    m = _LABEL_OP_RE.match(expr)
    if not m:
        raise ValueError(f"can't parse label filter '{expr}' (expected k=v / k>=v / etc.)")
    key, op, val = m.group(1).strip(), m.group(2), m.group(3).strip()
    if op == "==":
        op = "="
    try:
        val_parsed = float(val)
    except ValueError:
        val_parsed = val
        if op != "=":
            raise ValueError(
                f"label filter '{expr}': operator '{op}' requires numeric RHS"
            )
    return key, op, val_parsed


def label_matches(labels: dict, key: str, op: str, want) -> bool:
    if labels is None or key not in labels:
        return False
    have = labels[key]
    if op == "=":
        if isinstance(want, str) or isinstance(have, str):
            return str(have) == str(want)
        return float(have) == float(want)
    try:
        h = float(have)
        w = float(want)
    except (TypeError, ValueError):
        return False
    return {">=": h >= w, "<=": h <= w, ">": h > w, "<": h < w}[op]
