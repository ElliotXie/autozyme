"""task.yaml parsing + writing.

Pure I/O on the task.yaml file. No pyyaml dependency — every function is a
hand-written line walker tuned to the inline-flow conventions in the
task.yaml template (block headers + `- {key: val, ...}` entries).
"""
import re
import sys
from pathlib import Path

from zyme.utils import LEGACY_THREAD


def _strip_trailing_yaml_comment(s: str) -> str:
    """Strip ` # comment` from the tail of an inline-flow yaml line.

    Conservative: only strips when a `}` appears before the `#`, so values
    that legitimately contain `#` (e.g. a future hex color or hashtag in a
    description) aren't mangled. Whitespace before `#` is also dropped.

    Why this exists: the parsers below check `inner.endswith('}')` to detect
    a `- {...}` flow entry, but the prompt examples and many real task.yaml
    files have `- {name: foo, ...}  # one-line reason` — the trailing comment
    breaks the structural check and the entry silently drops.
    """
    hash_pos = s.rfind("#")
    if hash_pos != -1 and "}" in s[:hash_pos]:
        return s[:hash_pos].rstrip()
    return s


def parse_dataset_name(task_yaml: Path) -> str:
    """Best-effort dataset name extraction (legacy: first dataset listed).

    Prefer `parse_datasets()` for tier-aware resolution.
    """
    entries = parse_datasets(task_yaml)
    return entries[0]["name"] if entries else ""


def _split_top_level_commas(s: str):
    """Split `s` on commas only at brace-depth zero. Used to walk inline-flow
    YAML entries that may contain nested `{...}` values (e.g. `params:`)."""
    parts = []
    depth = 0
    buf = []
    for ch in s:
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def _coerce_scalar(v: str):
    """Strip surrounding quotes; coerce numeric literals to int/float; else string."""
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
        return v[1:-1]
    # Try int first (5000), then float (1e-3), else leave as string.
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_inline_value(v: str):
    """Parse a value that may be a scalar or a nested inline dict."""
    v = v.strip()
    if v.startswith("{") and v.endswith("}"):
        inner = v[1:-1]
        out = {}
        for kv in _split_top_level_commas(inner):
            kv = kv.strip()
            if not kv:
                continue
            # Accept dotted keys (burn.in, na.rm, BPPARAM) — R packages
            # routinely expose them. Bare \w+ silently dropped these.
            m = re.match(r"\s*([A-Za-z0-9_.]+)\s*:\s*(.+)$", kv)
            if not m:
                import sys as _sys
                _sys.stderr.write(
                    f"[parse_params] WARN: dropping malformed entry "
                    f"{kv!r} (no `key: value` shape)\n"
                )
                continue
            out[m.group(1)] = _parse_inline_value(m.group(2))
        return out
    return _coerce_scalar(v)


def parse_datasets(task_yaml: Path):
    """Parse the datasets list from task.yaml without requiring pyyaml.

    Returns list of {"tier": str, "name": str, "path": str, "params": dict}.
    `params` is `{}` when the entry omits it. Skips commented lines. If a
    `tier:` field is absent on a dataset entry, defaults to "tiny" (legacy
    single-dataset compatibility).

    Handles inline-flow style entries: `- {tier: X, name: Y, path: Z}` and
    `- {tier: X, name: Y, path: Z, params: {n_cells: 1000, n_genes: 500}}`.
    Block style isn't supported — task.yaml template uses inline flow.
    """
    if not task_yaml.exists():
        return []
    text = task_yaml.read_text()

    in_datasets = False
    out = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if re.match(r"^datasets:\s*$", line):
            in_datasets = True
            continue
        if in_datasets:
            # End of datasets block: a non-indented top-level key
            if re.match(r"^\S", line) and not line.startswith("- "):
                in_datasets = False
                continue
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            inner = stripped[2:].strip()
            inner = _strip_trailing_yaml_comment(inner)
            if inner.startswith("{") and inner.endswith("}"):
                inner = inner[1:-1]
            entry = {}
            for kv in _split_top_level_commas(inner):
                kv = kv.strip()
                if not kv:
                    continue
                m = re.match(r"\s*([A-Za-z0-9_.]+)\s*:\s*(.+)$", kv)
                if not m:
                    import sys as _sys
                    _sys.stderr.write(
                        f"[parse_datasets] WARN: dropping malformed entry "
                        f"{kv!r} in datasets list (no `key: value` shape)\n"
                    )
                    continue
                key = m.group(1)
                val = m.group(2).strip()
                if val.startswith("{") and val.endswith("}"):
                    # Nested dict (params:) — fully coerced via recursion.
                    entry[key] = _parse_inline_value(val)
                else:
                    # Top-level scalars stay as strings (existing behavior:
                    # tier, name, path are downstream-stringly-typed).
                    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                        val = val[1:-1]
                    entry[key] = val
            if "name" in entry and "path" in entry:
                entry.setdefault("tier", "tiny")
                entry.setdefault("params", {})
                # Resolve relative dataset paths against the task directory.
                # The runner passes ZYME_DATA_PATH verbatim and runs the
                # pipeline with cwd=task_dir/pipeline, so a relative path
                # like ./data/foo.rds breaks. Resolve here once.
                p = entry["path"]
                if p and not Path(p).is_absolute():
                    entry["path"] = str((task_yaml.parent / p).resolve())
                out.append(entry)
    return out


def parse_scaling_tax_thresholds(task_yaml: Path) -> dict:
    """Parse `scaling_tax_thresholds:` block from task.yaml. Returns dict with
    `ood_large_soft`, `ood_xlarge_soft`, `hard_fail`, `super_linear_max`
    (each a positive float). The first three are tax multipliers (tax >
    threshold triggers the corresponding flag) used by `zyme verify` to
    grade OOD speedup factor against dev-tier geometric mean. Defaults
    1.5 / 2 / 5 reflect a healthy data-shape-variance band: OOD getting
    less than 1/2 of dev's speedup is already cause to investigate, less
    than 1/5 means the patch isn't generalizing. `super_linear_max` is
    a multiplier on the thread count (default 1.5) — at any tier, factor
    at thread=N may not exceed `super_linear_max * N * factor at thread=1`.
    Catches the symmetric pathology of thread=1 path being broken (fast
    path only firing when threaded). 1.5N allows cache-level super-linear
    while flagging the rest.
    """
    defaults = {
        "ood_large_soft": 1.5,
        "ood_xlarge_soft": 2.0,
        "hard_fail": 5.0,
        "super_linear_max": 1.5,
    }
    if not task_yaml.exists():
        return defaults
    text = task_yaml.read_text()
    in_block = False
    out = dict(defaults)
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if re.match(r"^scaling_tax_thresholds:\s*$", line):
            in_block = True
            continue
        if in_block:
            if re.match(r"^\S", line):
                break
            m = re.match(r"\s+(\w+)\s*:\s*([\d.]+)", line)
            if m and m.group(1) in defaults:
                try:
                    out[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    return out


def parse_threading_mode(task_yaml: Path) -> str:
    """Parse top-level `threading:` field from task.yaml. Returns "not_applicable"
    or "default".

    Escape hatch for tasks where the converged optimization stack is
    fundamentally sequential (sequential algorithm, no parallelism opportunity,
    or wiring threading would require risky rewrites of the thread=1 path that
    delivered the dev-tier speedup). Setting:

        threading: not_applicable

    tells `zyme verify` to (a) skip the threading-wired probe, (b) not enforce
    the no-multi-thread-regression / no-super-linear cell-level rules, and (c)
    drop the implicit "thread > 1 must improve" requirement. Packaging-time
    `zyme attest` / `zyme publish-speedups` enforce this more strictly: only
    thread=1 rows are run or published for final speedup snapshots.

    Should be used SPARINGLY — declared at Setup time with a one-line rationale
    in `memory/discoveries.md`. Most tasks should wire threading.
    """
    if not task_yaml.exists():
        return "default"
    text = task_yaml.read_text()
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^threading\s*:\s*(\S+)\s*$", line)
        if m:
            val = m.group(1).strip().strip('"').strip("'")
            if val == "not_applicable":
                return "not_applicable"
    return "default"


# ---- thread axis (K2) -----------------------------------------------------
#
# K2 dropped `mode` as an axis — every "different upstream config" use case
# (HMM on/off, GPU, different solvers) maps to "open a new task" rather than
# "switch mode within one task". Thread count is the only baseline axis.
#
# task.yaml format:
#
#     baseline_threads: [1, 4, 8]    # thread points to maintain baselines at
#
# The single `reference.{R,py}` per task reads ZYME_THREADS and branches as
# needed; baselines for each thread point are recorded as separate (tier,
# thread) rows in results.tsv.
#
# V1/K1 may have left behind `active_mode:` and `modes:` blocks in some
# task.yaml files — K2 ignores them on read.


def parse_active_mode(task_yaml: Path) -> str:
    """Backward-compatible parser for pre-K2 `active_mode:`.

    New K2 tasks use `baseline_threads` instead of modes. This reader remains
    intentionally small so old task.yaml files and old tests can still be
    inspected without pulling in PyYAML.
    """
    if not task_yaml.exists():
        return "legacy"
    text = task_yaml.read_text()
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^active_mode\s*:\s*(.+?)\s*$", line)
        if m:
            return m.group(1).strip().strip('"').strip("'") or "legacy"
    return "legacy"


def parse_modes_block(task_yaml: Path) -> dict:
    """Backward-compatible parser for the old `modes:` block.

    Returns `{mode_name: {field: value_as_string}}`. Values are left as
    strings because the old callers used this block for path templates and
    human-readable descriptions, not numeric computation.
    """
    if not task_yaml.exists():
        return {}
    text = task_yaml.read_text()
    in_modes = False
    out = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if re.match(r"^modes:\s*$", line):
            in_modes = True
            continue
        if in_modes:
            if re.match(r"^\S", line):
                break
            m = re.match(r"\s+([A-Za-z0-9_.-]+)\s*:\s*\{(.*)\}\s*$", line)
            if not m:
                continue
            name, inner = m.group(1), m.group(2)
            entry = {}
            for kv in _split_top_level_commas(inner):
                kv = kv.strip()
                if not kv:
                    continue
                km = re.match(r"\s*([A-Za-z0-9_.]+)\s*:\s*(.*?)\s*$", kv)
                if not km:
                    continue
                val = km.group(2).strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1]
                entry[km.group(1)] = val
            out[name] = entry
    return out


def _render_mode_value(value) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(ch in text for ch in [",", "{", "}", "#"]) or text != text.strip():
        return '"' + text.replace('"', '\\"') + '"'
    return text


def _render_mode_entry(name: str, fields: dict) -> str:
    body = ", ".join(f"{k}: {_render_mode_value(v)}" for k, v in fields.items())
    return f"  {name}: {{{body}}}"


def write_mode_entry(task_yaml: Path, name: str, fields: dict) -> None:
    """Insert or replace one entry in the old `modes:` block."""
    text = task_yaml.read_text() if task_yaml.exists() else ""
    lines = text.splitlines()
    new_line = _render_mode_entry(name, fields)

    modes_idx = None
    entry_idx = None
    block_end = None
    for i, line in enumerate(lines):
        if modes_idx is None and re.match(r"^modes:\s*$", line):
            modes_idx = i
            continue
        if modes_idx is not None:
            if re.match(r"^\S", line):
                block_end = i
                break
            if re.match(rf"\s+{re.escape(name)}\s*:", line):
                entry_idx = i
                break

    if entry_idx is not None:
        lines[entry_idx] = new_line
    elif modes_idx is not None:
        insert_at = block_end if block_end is not None else len(lines)
        lines.insert(insert_at, new_line)
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("modes:")
        lines.append(new_line)

    task_yaml.write_text("\n".join(lines) + "\n")


def remove_mode_entry(task_yaml: Path, name: str) -> bool:
    """Remove one old-style mode entry. Returns True if a line changed."""
    if not task_yaml.exists():
        return False
    lines = task_yaml.read_text().splitlines()
    modes_idx = None
    entry_idx = None
    block_end = len(lines)
    for i, line in enumerate(lines):
        if modes_idx is None and re.match(r"^modes:\s*$", line):
            modes_idx = i
            continue
        if modes_idx is not None:
            if re.match(r"^\S", line):
                block_end = i
                break
            if re.match(rf"\s+{re.escape(name)}\s*:", line):
                entry_idx = i
                break
    if entry_idx is None or modes_idx is None:
        return False
    del lines[entry_idx]

    block_lines = lines[modes_idx + 1:block_end - 1 if entry_idx < block_end else block_end]
    has_entry = any(re.match(r"\s+[A-Za-z0-9_.-]+\s*:", line) for line in block_lines)
    if not has_entry:
        del lines[modes_idx]

    task_yaml.write_text("\n".join(lines) + ("\n" if lines else ""))
    return True


def write_active_mode(task_yaml: Path, mode: str) -> None:
    """Set or insert the old `active_mode:` field."""
    text = task_yaml.read_text() if task_yaml.exists() else ""
    lines = text.splitlines()
    new_line = f"active_mode: {mode}"
    for i, line in enumerate(lines):
        if re.match(r"^active_mode\s*:", line):
            lines[i] = new_line
            task_yaml.write_text("\n".join(lines) + "\n")
            return
    insert_at = None
    for i, line in enumerate(lines):
        if re.match(r"^modes:\s*$", line):
            insert_at = i
            break
    if insert_at is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(new_line)
    else:
        lines.insert(insert_at, new_line)
    task_yaml.write_text("\n".join(lines) + "\n")


def parse_baseline_threads(task_yaml: Path) -> list[int]:
    """Parse top-level `baseline_threads: [N1, N2, ...]` flow-list from task.yaml.

    K1: declares which thread points the task maintains baselines at —
    `mode-set` and `verify` use this to compute per-(mode × tier × thread)
    baseline-completeness matrices. Defaults to `[LEGACY_THREAD]` (=`[1]`)
    when the field is absent — single-threaded baseline only, equivalent
    to pre-K1 behavior.

    Format (inline flow list):
        baseline_threads: [1, 4, 8]
    """
    if not task_yaml.exists():
        return [LEGACY_THREAD]
    text = task_yaml.read_text()
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^baseline_threads\s*:\s*\[(.*)\]\s*$", line)
        if not m:
            continue
        inner = m.group(1)
        out: list[int] = []
        for tok in inner.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(int(tok))
            except ValueError:
                continue
        return out or [LEGACY_THREAD]
    return [LEGACY_THREAD]


def write_baseline_threads(task_yaml: Path, threads: list[int]) -> None:
    """Set/insert top-level `baseline_threads: [N1, N2, ...]`.

    Replaces an existing line in place; otherwise inserts after `active_mode:`
    (or before `modes:` block, whichever comes first), or at end of file.
    """
    if not threads:
        threads = [LEGACY_THREAD]
    text = task_yaml.read_text()
    lines = text.splitlines()
    rendered = ", ".join(str(int(t)) for t in threads)
    new_line = f"baseline_threads: [{rendered}]"

    bt_idx = None
    active_idx = None
    modes_idx = None
    for i, line in enumerate(lines):
        if bt_idx is None and re.match(r"^baseline_threads\s*:", line):
            bt_idx = i
        if active_idx is None and re.match(r"^active_mode\s*:", line):
            active_idx = i
        if modes_idx is None and re.match(r"^modes\s*:\s*$", line):
            modes_idx = i

    if bt_idx is not None:
        lines[bt_idx] = new_line
    elif active_idx is not None:
        lines.insert(active_idx + 1, new_line)
    elif modes_idx is not None:
        lines.insert(modes_idx, new_line)
    else:
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.append(new_line)

    task_yaml.write_text("\n".join(lines) + "\n")


def parse_metrics(task_yaml: Path):
    """Parse the metrics list from task.yaml without requiring pyyaml.

    Each entry is one of two forms:
      Deterministic: {name, threshold, comparator}
      Stochastic:    {name, comparator, noise_multiplier, absolute_floor}

    Detection: presence of `threshold` -> deterministic; presence of
    `noise_multiplier` AND `absolute_floor` -> stochastic. Mixed schemas
    (e.g. all three keys at once) raise nothing here but downstream gate
    code prefers the deterministic `threshold` if it exists.

    Mirrors `parse_datasets` parsing strategy (inline-flow `- {...}`).
    """
    if not task_yaml.exists():
        return []
    text = task_yaml.read_text()

    in_metrics = False
    out = []
    rejected = []  # [(raw_line, reason)] — surfaced loudly at end if non-empty
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if re.match(r"^metrics:\s*$", line):
            in_metrics = True
            continue
        if in_metrics:
            if re.match(r"^\S", line) and not line.startswith("- "):
                in_metrics = False
                continue
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            inner = stripped[2:].strip()
            inner = _strip_trailing_yaml_comment(inner)
            if inner.startswith("{") and inner.endswith("}"):
                inner = inner[1:-1]
            else:
                rejected.append((raw, "entry doesn't open with `{` or close with `}` "
                                       "(after stripping trailing `# comment`)"))
                continue
            entry = {}
            for kv in re.split(r",\s*(?=\w+:)", inner):
                m = re.match(r"\s*(\w+)\s*:\s*(.+?)\s*$", kv)
                if m:
                    entry[m.group(1)] = m.group(2).strip()
            if "name" not in entry or "comparator" not in entry:
                rejected.append((raw, f"missing required key (name + comparator); got {sorted(entry)}"))
                continue
            if entry["comparator"] not in ("gte", "lte"):
                rejected.append((raw, f"comparator='{entry['comparator']}' not in {{gte, lte}}"))
                continue
            # Determine schema: deterministic (threshold) vs stochastic
            # (noise_multiplier + absolute_floor).
            has_threshold = "threshold" in entry
            has_stochastic = "noise_multiplier" in entry and "absolute_floor" in entry
            try:
                if has_threshold:
                    entry["threshold"] = float(entry["threshold"])
                if has_stochastic:
                    entry["noise_multiplier"] = float(entry["noise_multiplier"])
                    entry["absolute_floor"] = float(entry["absolute_floor"])
            except ValueError:
                rejected.append((raw, "non-numeric threshold / noise_multiplier / absolute_floor"))
                continue
            if not (has_threshold or has_stochastic):
                rejected.append((raw, "needs `threshold` (deterministic) OR "
                                       "`noise_multiplier`+`absolute_floor` (stochastic)"))
                continue
            out.append(entry)

    # Silent-empty was the worst behavior — caller saw "no metrics block"
    # while staring at one. Be loud whenever entries were rejected, regardless
    # of whether others survived. stderr so it doesn't pollute parsed-stdout
    # consumers; format mirrors `zyme:` info-style for grep familiarity.
    if rejected:
        n_acc, n_rej = len(out), len(rejected)
        plural = "y" if n_rej == 1 else "ies"
        sys.stderr.write(
            f"zyme: [parse_metrics] WARN: rejected {n_rej} metric entr{plural} "
            f"in {task_yaml} ({n_acc} accepted)\n"
        )
        for raw, reason in rejected[:3]:
            sys.stderr.write(f"  rejected: {raw.strip()}\n    reason: {reason}\n")
        if n_rej > 3:
            sys.stderr.write(f"  ... and {n_rej - 3} more.\n")
    return out


def parse_target_function(task_yaml: Path) -> str:
    """Parse top-level `target_function:` field. Returns the raw value or "".

    Format examples:
        cellphonedb.src.core.methods.cpdb_statistical_analysis_method::call
        spacexr::run.RCTD
        MDAnalysis.analysis.rms.RMSD.run
    """
    if not task_yaml.exists():
        return ""
    text = task_yaml.read_text()
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^target_function:\s*(.+?)\s*$", line)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    return ""


def parse_upstream_parallelism(task_yaml: Path) -> list[str]:
    """Parse `upstream_parallelism: [name1, name2, ...]` flow-list. Returns [] if absent."""
    if not task_yaml.exists():
        return []
    text = task_yaml.read_text()
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^upstream_parallelism:\s*\[(.*)\]\s*$", line)
        if m:
            inner = m.group(1).strip()
            if not inner:
                return []
            parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
            return [p for p in parts if p]
    return []


def parse_synthesis(task_yaml: Path) -> str | None:
    """Parse top-level `synthesis: <reason>` field. Returns the reason string
    or None if absent.

    Declared when the tier inputs are intrinsically synthesis-driven (PDE
    solver, Lomb-Scargle on generated signals, license-blocked data). When
    set AND `--accept-synthesis` is passed to a baseline command, the
    fingerprint check on `reference.{py,R}` is bypassed.
    """
    if not task_yaml.exists():
        return None
    text = task_yaml.read_text()
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^synthesis:\s*(.+?)\s*$", line)
        if m:
            val = m.group(1).strip().strip('"').strip("'")
            return val if val else None
    return None


def parse_algorithm_class(task_yaml: Path) -> str:
    """Parse top-level `algorithm_class:` field. Returns "stochastic" or
    "deterministic" (the default if absent).

    Stochastic algorithms pair with `random_seeds:` + per-tier
    `intrinsic_noise:` + per-metric `noise_multiplier`/`absolute_floor`
    schema. Deterministic algorithms keep the original fixed-threshold
    schema.
    """
    if not task_yaml.exists():
        return "deterministic"
    text = task_yaml.read_text()
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^algorithm_class:\s*(\w+)\s*$", line)
        if m:
            val = m.group(1)
            return val if val in ("stochastic", "deterministic") else "deterministic"
    return "deterministic"


def _parse_int_list(raw: str) -> list[int]:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return vals


def parse_random_seeds(task_yaml: Path) -> dict:
    """Parse top-level random seeds.

    Preferred form:
        random_seeds: {primary: 42, noise_calibration: [43, 44, 45]}

    Legacy scalar form remains accepted:
        random_seeds: {primary: 42, noise_calibration: 43}

    Returns dict with `primary` (default 42) and `noise_calibration`
    (default [43, 44, 45]) keys. `noise_calibration` is always a list.
    Used by `zyme baseline noise` to pick calibration seeds.
    """
    defaults = {"primary": 42, "noise_calibration": [43, 44, 45]}
    if not task_yaml.exists():
        return defaults
    text = task_yaml.read_text()
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        m = re.match(r"^random_seeds:\s*\{(.+)\}\s*$", line)
        if m:
            inner = m.group(1)
            out = dict(defaults)
            for kv in re.split(r",\s*(?=\w+:)", inner):
                kv_m = re.match(r"\s*(\w+)\s*:\s*(.+?)\s*$", kv)
                if kv_m:
                    try:
                        key = kv_m.group(1)
                        val = kv_m.group(2).strip()
                        if key == "noise_calibration":
                            out[key] = _parse_int_list(val)
                        else:
                            out[key] = int(val)
                    except ValueError:
                        pass
            return out
    return defaults


def parse_intrinsic_noise(task_yaml: Path) -> dict:
    """Parse `intrinsic_noise:` block from task.yaml.

    Format:
        intrinsic_noise:
          tiny: {pearson_beta_mean: 0.0008, max_abs_diff_beta_mean: 0.0031}
          medium: {pearson_beta_mean: 0.0024, max_abs_diff_beta_mean: 0.0089}
          ood_xlarge: {pearson_beta_mean: 0.985, max_abs_diff_beta_mean: 0.12}

    Returns nested dict {tier_name: {metric_name: float_value}}. Empty dict
    if the block is missing — caller falls back to absolute_floor or
    threshold accordingly.
    """
    if not task_yaml.exists():
        return {}
    text = task_yaml.read_text()
    in_block = False
    out = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if re.match(r"^intrinsic_noise:\s*$", line):
            in_block = True
            continue
        if in_block:
            if re.match(r"^\S", line):
                break
            m = re.match(r"\s+(\w+)\s*:\s*\{(.+)\}\s*$", line)
            if not m:
                continue
            tier = m.group(1)
            inner = m.group(2)
            metrics = {}
            for kv in re.split(r",\s*(?=\w+:)", inner):
                kv_m = re.match(r"\s*(\w+)\s*:\s*([\d.eE+-]+)\s*$", kv)
                if kv_m:
                    try:
                        metrics[kv_m.group(1)] = float(kv_m.group(2))
                    except ValueError:
                        pass
            if metrics:
                out[tier] = metrics
    return out


def write_intrinsic_noise(task_yaml: Path, tier: str, metrics: dict) -> None:
    """Write/update `intrinsic_noise[tier]` in task.yaml.

    If `intrinsic_noise:` block doesn't exist, append it at end of file.
    If it exists but `<tier>:` line doesn't, append `<tier>: {...}` after
    the block header. If `<tier>:` line exists, replace it in place. All
    metric values are written with 6-decimal float formatting.
    """
    text = task_yaml.read_text()
    lines = text.splitlines()

    inline = ", ".join(f"{k}: {v:.10g}" for k, v in metrics.items())
    new_tier_line = f"  {tier}: {{{inline}}}"

    block_idx = None
    tier_line_idx = None
    block_end = None
    for i, line in enumerate(lines):
        if re.match(r"^intrinsic_noise:\s*$", line):
            block_idx = i
            continue
        if block_idx is not None and tier_line_idx is None:
            if re.match(r"^\S", line):
                block_end = i
                break
            m = re.match(r"^\s+(\w+)\s*:\s*\{", line)
            if m and m.group(1) == tier:
                tier_line_idx = i

    if tier_line_idx is not None:
        lines[tier_line_idx] = new_tier_line
    elif block_idx is not None:
        # Append within existing block — at block_end (or end of file).
        insert_at = block_end if block_end is not None else len(lines)
        lines.insert(insert_at, new_tier_line)
    else:
        # No block; append a new one at end of file.
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.append("intrinsic_noise:")
        lines.append(new_tier_line)

    task_yaml.write_text("\n".join(lines) + "\n")


def effective_threshold(metric: dict, tier: str, intrinsic_noise: dict):
    """Compute effective concordance threshold for one (metric, tier) pair.

    Returns (threshold_value, label_for_logging).

    Deterministic metrics (have `threshold`): returns the fixed threshold.

    Stochastic metrics (have `noise_multiplier` + `absolute_floor`):
        For lte (smaller = better):
            effective = max(absolute_floor, multiplier × intrinsic_noise[tier][name])
        For gte (larger = better, e.g. correlation):
            effective = max(absolute_floor, 1 - multiplier × (1 - intrinsic_noise[tier][name]))

    `intrinsic_noise[tier][name]` is the worst raw metric value when comparing
    the primary reference seed to one or more calibration seeds. For lte
    (e.g. max_abs_diff = 0.05),
    the relaxed gate is multiplier × 0.05 (we allow that much diff). For gte
    (e.g. pearson = 0.9994), distance from perfect is 1-0.9994 = 0.0006, and
    relaxed gate allows multiplier × 0.0006 distance, so effective threshold
    = 1 - multiplier × 0.0006 = 0.9988. `absolute_floor` is a SAFETY: never
    relax the gate below this value (lte: never accept value larger than
    floor; gte: never accept value smaller than floor).

    If intrinsic_noise[tier][name] is missing (e.g. `zyme baseline noise`
    wasn't run for this tier), falls back to absolute_floor.
    """
    if "threshold" in metric:
        return float(metric["threshold"]), "absolute"
    floor = float(metric["absolute_floor"])
    multiplier = float(metric.get("noise_multiplier", 2.0))
    name = metric["name"]
    raw = (intrinsic_noise.get(tier) or {}).get(name)
    if raw is None:
        return floor, "absolute_floor (no intrinsic_noise calibrated)"
    raw = float(raw)
    if metric["comparator"] == "lte":
        relaxed = multiplier * raw
        eff = max(floor, relaxed)
        label = (f"max(floor={floor}, {multiplier}×noise={relaxed:.6g})"
                 if relaxed > floor else f"absolute_floor (noise×{multiplier}={relaxed:.6g} ≤ floor)")
    else:
        relaxed = 1.0 - multiplier * (1.0 - raw)
        eff = max(floor, relaxed)
        label = (f"max(floor={floor}, 1−{multiplier}×(1−noise)={relaxed:.6g})"
                 if relaxed > floor else f"absolute_floor (relaxed={relaxed:.6g} ≤ floor)")
    return eff, label


def parse_executor(task_yaml: Path):
    """Parse the optional `executor:` block from task.yaml.

    Returns dict possibly containing keys:
      - "python":  conda env name OR absolute path to a python interpreter
      - "rscript": absolute path to an Rscript binary

    Block format (inline-flow, like datasets):
        executor:
          python: myenv            # conda env name
          # OR
          python: /opt/envs/xyz/bin/python
          rscript: /opt/R-4.4/bin/Rscript

    Returns {} if the block is absent. Lines are quote-stripped to match the
    datasets parser. Unknown keys are kept verbatim (forward-compatible).
    """
    if not task_yaml.exists():
        return {}
    text = task_yaml.read_text()

    in_executor = False
    out = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if re.match(r"^executor:\s*$", line):
            in_executor = True
            continue
        if in_executor:
            if re.match(r"^\S", line):
                # New top-level key -> end of block
                in_executor = False
                continue
            m = re.match(r"^\s+([A-Za-z_][\w]*)\s*:\s*(.+?)\s*$", line)
            if not m:
                continue
            key = m.group(1)
            val = m.group(2).strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            out[key] = val
    return out


def resolve_tiers(task_yaml: Path, requested):
    """Resolve a `--dataset` / `--extra-tiers` argument to ordered (tier,name,path) entries.

    Args:
        task_yaml: Path to task.yaml.
        requested: None (= use first listed) or list of tier-or-name strings.

    Returns: list of dicts {tier, name, path}, in the order requested.
    Raises ValueError on missing tier / unknown name.
    """
    available = parse_datasets(task_yaml)
    if not available:
        raise ValueError(f"task.yaml has no datasets: {task_yaml}")
    if not requested:
        return [available[0]]

    by_tier = {e["tier"]: e for e in available}
    by_name = {e["name"]: e for e in available}
    chosen = []
    for token in requested:
        e = by_tier.get(token) or by_name.get(token)
        if e is None:
            raise ValueError(
                f"requested dataset '{token}' not found. Available tiers: "
                f"{sorted(by_tier.keys())}; names: {sorted(by_name.keys())}"
            )
        chosen.append(e)
    return chosen
