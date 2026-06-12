"""Cross-platform hazard scan for converged task pipelines.

Backs `zyme scan --portability`. Statically scans `pipeline/run.{R,py}` and
optional sibling sources, classifies portability debt, and writes
`<task_dir>/.zyme/portability_scan.json`.

The scan is mechanical triage — it decides whether to dispatch the 3.5
portability agent. Exploration and fixes stay in the agent prompt.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from zyme.scan import _find_patch, _read_task_name

SCHEMA_VERSION = 1
PORTABILITY_SCAN_REL = ".zyme/portability_scan.json"

_SKIP_DIRS = {
    ".git", "node_modules", "vendor", "build", "dist", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".tox", ".venv", "venv", "env",
    "upstream_repo", "artifacts", "reference_outputs", "data",
}

# (regex, hit_kind, severity, human label)
# hit_kind: crash_on_win | harmful_default | review | portable | mitigation
_RULES: list[tuple[re.Pattern, str, str, str]] = [
    (re.compile(r"\.zyme_mclapply\s*\("), "mitigation", "info", "zyme_mclapply"),
    (re.compile(r"parallel::mclapply\s*\("), "crash_on_win", "high", "raw_parallel_mclapply"),
    (re.compile(r"(?<![.\w])mclapply\s*\("), "crash_on_win", "high", "raw_mclapply"),
    (re.compile(r"MulticoreParam\s*\("), "crash_on_win", "high", "multicore_param"),
    (re.compile(r"future::multicore\b"), "crash_on_win", "high", "future_multicore"),
    (re.compile(r"\bbplapply\s*\("), "review", "medium", "bplapply"),
    (re.compile(r"\bbpmapply\s*\("), "review", "medium", "bpmapply"),
    (re.compile(r"makePSOCKcluster\s*\("), "harmful_default", "high", "psock_cluster"),
    (re.compile(r"parLapply(?:LB)?\s*\("), "harmful_default", "high", "parLapply"),
    (re.compile(r"clusterApply(?:LB)?\s*\("), "harmful_default", "medium", "clusterApply"),
    (re.compile(r"makeCluster\s*\("), "harmful_default", "medium", "makeCluster"),
    (re.compile(r"SnowParam\s*\("), "harmful_default", "medium", "snow_param"),
    (re.compile(r"%dopar%"), "review", "medium", "dopar"),
    (re.compile(r"joblib\.Parallel\s*\("), "review", "medium", "joblib_parallel"),
    (re.compile(r"multiprocessing\.Pool\s*\("), "review", "medium", "mp_pool"),
    (re.compile(r"ProcessPoolExecutor\s*\("), "review", "medium", "process_pool"),
    (re.compile(r"RcppParallel::"), "portable", "info", "rcpp_parallel"),
    (re.compile(r"#pragma\s+omp\s+"), "portable", "info", "openmp"),
    (re.compile(r"std::thread\b"), "portable", "info", "std_thread"),
    (re.compile(r"sourceCpp\s*\("), "portable", "info", "sourceCpp"),
    (re.compile(r"@(?:numba\.)?(?:jit|njit|prange)\b"), "portable", "info", "numba"),
    (re.compile(r"\.Platform\$OS\.type"), "mitigation", "info", "os_type_guard"),
    (re.compile(r"sys\.platform\b|platform\.system\s*\("), "mitigation", "info", "py_os_guard"),
]

_VERDICT_SKIP = frozenset({"CLEAN", "MAC_BONUS_ONLY"})


@dataclass
class Hit:
    kind: str
    severity: str
    pattern: str
    file: str
    line: int
    text: str
    mac_only_guess: bool = False
    note: str = ""


@dataclass
class ScanResult:
    schema_version: int = SCHEMA_VERSION
    scanned_at: str = ""
    task_dir: str = ""
    sources: list[str] = field(default_factory=list)
    verdict: str = "CLEAN"
    run_3_5: bool = False
    action: str = "skip_3_5"
    hits: list[Hit] = field(default_factory=list)
    portable_signals: list[dict] = field(default_factory=list)
    mitigations: list[dict] = field(default_factory=list)
    summary: str = ""

    def to_json_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "scanned_at": self.scanned_at,
            "task_dir": self.task_dir,
            "sources": self.sources,
            "verdict": self.verdict,
            "run_3_5": self.run_3_5,
            "action": self.action,
            "hits": [asdict(h) for h in self.hits],
            "portable_signals": self.portable_signals,
            "mitigations": self.mitigations,
            "summary": self.summary,
        }


def portability_scan_path(task_dir: Path) -> Path:
    return Path(task_dir) / PORTABILITY_SCAN_REL


def load_portability_scan(task_dir: Path) -> dict | None:
    p = portability_scan_path(task_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_portability_scan(task_dir: Path, result: ScanResult) -> Path:
    p = portability_scan_path(task_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(result.to_json_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return p


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _is_comment(line: str, rel: str) -> bool:
    stripped = line.lstrip()
    if rel.endswith((".R", ".r")):
        return stripped.startswith("#")
    if rel.endswith((".py", ".pyx")):
        return stripped.startswith("#")
    if rel.endswith((".cpp", ".cc", ".cxx", ".c", ".h", ".hpp")):
        return stripped.startswith("//") or stripped.startswith("/*")
    return False


def _mac_only_guess(lines: list[str], lineno: int) -> bool:
    """Heuristic: non-Windows guard within 8 lines above this hit."""
    start = max(0, lineno - 9)
    window = "\n".join(lines[start:lineno])
    if re.search(r'\.Platform\$OS\.type\s*!=\s*["\']windows["\']', window):
        return True
    if re.search(r'\.Platform\$OS\.type\s*==\s*["\']unix["\']', window):
        return True
    if re.search(r'sys\.platform\s*(!=|==)\s*["\']win', window):
        return True
    if re.search(r'platform\.system\s*\(\s*\)\s*!=\s*["\']Windows', window):
        return True
    return False


def _collect_sources(task_dir: Path, framework_root: Path | None) -> list[Path]:
    task_dir = Path(task_dir)
    out: list[Path] = []
    pipeline = task_dir / "pipeline"
    for name in ("run.R", "run.py"):
        p = pipeline / name
        if p.is_file():
            out.append(p)
    if pipeline.is_dir():
        for ext in ("*.cpp", "*.cc", "*.cxx", "*.pyx"):
            out.extend(sorted(pipeline.glob(ext)))

    if framework_root is not None:
        task_name = _read_task_name(task_dir / "task.yaml")
        dir_name = task_dir.name
        candidates = [task_name, dir_name]
        for s in (task_name, dir_name):
            if s.startswith("test_"):
                candidates.append(s[len("test_"):])
        seen: set[str] = set()
        unique = [c for c in candidates if not (c in seen or seen.add(c))]
        patch_path = _find_patch(Path(framework_root), *unique)
        if patch_path:
            pp = Path(patch_path)
            if pp.is_file():
                out.append(pp)
            elif pp.is_dir():
                for cand in (pp / "patch.R", pp.parent / f"{pp.name}.R"):
                    if cand.is_file():
                        out.append(cand)
                        break

    # Dedupe preserving order
    seen_abs: set[str] = set()
    deduped: list[Path] = []
    for p in out:
        key = str(p.resolve())
        if key not in seen_abs:
            seen_abs.add(key)
            deduped.append(p)
    return deduped


def _rel_to_task(path: Path, task_dir: Path) -> str:
    try:
        return path.resolve().relative_to(task_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _scan_file(path: Path, task_dir: Path) -> tuple[list[Hit], list[dict], list[dict]]:
    rel = _rel_to_task(path, task_dir)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], [], []
    if len(text) > 2_000_000:
        return [], [], []

    lines = text.splitlines()
    hits: list[Hit] = []
    portable: list[dict] = []
    mitigations: list[dict] = []

    for lineno, line in enumerate(lines, start=1):
        if len(line) > 2000 or _is_comment(line, rel):
            continue
        matched_kinds: set[str] = set()
        for rx, kind, severity, label in _RULES:
            if not rx.search(line):
                continue
            # raw_mclapply rule also matches parallel::mclapply — skip duplicate
            if label == "raw_mclapply" and "parallel::mclapply" in line:
                continue
            if label == "raw_mclapply" and ".zyme_mclapply" in line:
                continue
            if kind in matched_kinds:
                continue
            matched_kinds.add(kind)
            entry = {
                "pattern": label,
                "file": rel,
                "line": lineno,
                "text": line.strip()[:200],
            }
            if kind == "portable":
                portable.append(entry)
                continue
            if kind == "mitigation":
                mitigations.append(entry)
                continue
            mac_only = _mac_only_guess(lines, lineno - 1)
            note = ""
            if mac_only and kind == "crash_on_win":
                note = "Likely Unix-only branch (OS guard within 8 lines above)."
            hits.append(Hit(
                kind=kind,
                severity=severity,
                pattern=label,
                file=rel,
                line=lineno,
                text=line.strip()[:200],
                mac_only_guess=mac_only,
                note=note,
            ))
    return hits, portable, mitigations


def _compute_verdict(
    hits: list[Hit],
    mitigations: list[dict],
    portable: list[dict],
) -> tuple[str, bool, str]:
    harmful = [h for h in hits if h.kind == "harmful_default"]
    crash = [h for h in hits if h.kind == "crash_on_win"]
    review = [h for h in hits if h.kind == "review"]

    if harmful:
        return (
            "HARMFUL-DEFAULT",
            True,
            f"{len(harmful)} harmful-default pattern(s) — remove or gate before package.",
        )

    unguarded_crash = [h for h in crash if not h.mac_only_guess]
    if unguarded_crash:
        return (
            "CRASH-ON-WIN",
            True,
            f"{len(unguarded_crash)} Unix-only parallel pattern(s) on a path that will ship.",
        )

    if review:
        return (
            "NEEDS-KERNEL",
            True,
            f"{len(review)} BiocParallel/worker-pool pattern(s) need portable review.",
        )

    if crash and all(h.mac_only_guess for h in crash):
        return (
            "MAC_BONUS_ONLY",
            False,
            "Fork parallel is Unix-guarded; Windows uses another path.",
        )

    has_zyme = any(m.get("pattern") == "zyme_mclapply" for m in mitigations)
    if has_zyme and not hits:
        return (
            "MAC_BONUS_ONLY",
            False,
            ".zyme_mclapply present with no unguarded crash patterns.",
        )

    if portable and not hits:
        return (
            "CLEAN",
            False,
            f"Portable kernel/backends only ({len(portable)} signal(s)).",
        )

    if not hits and not review:
        return ("CLEAN", False, "No portability hazards detected.")

    return ("CLEAN", False, "No action required.")


def scan_task(task_dir: Path, framework_root: Path | None = None) -> ScanResult:
    task_dir = Path(task_dir).resolve()
    sources = _collect_sources(task_dir, framework_root)
    all_hits: list[Hit] = []
    all_portable: list[dict] = []
    all_mitigations: list[dict] = []

    for src in sources:
        hits, portable, mitigations = _scan_file(src, task_dir)
        all_hits.extend(hits)
        all_portable.extend(portable)
        all_mitigations.extend(mitigations)

    verdict, run_35, summary = _compute_verdict(all_hits, all_mitigations, all_portable)
    action = "run_3_5" if run_35 else "skip_3_5"

    rel_sources = []
    for s in sources:
        try:
            rel_sources.append(s.relative_to(task_dir).as_posix())
        except ValueError:
            rel_sources.append(s.as_posix())

    return ScanResult(
        scanned_at=_utc_now(),
        task_dir=str(task_dir),
        sources=rel_sources,
        verdict=verdict,
        run_3_5=run_35,
        action=action,
        hits=all_hits,
        portable_signals=all_portable,
        mitigations=all_mitigations,
        summary=summary,
    )


def render_table(rows: list[ScanResult]) -> str:
    if not rows:
        return "(no tasks scanned)"
    name_w = min(36, max(20, max(len(Path(r.task_dir).name) for r in rows)))
    lines = [
        f"  {'task':<{name_w}}  {'verdict':<18}  {'3.5?':<6}  hits  summary",
    ]
    for r in rows:
        name = Path(r.task_dir).name
        if len(name) > name_w:
            name = name[: name_w - 3] + "..."
        flag = "yes" if r.run_3_5 else "no"
        n_hits = len([h for h in r.hits if h.kind != "mitigation"])
        summary = r.summary
        if len(summary) > 48:
            summary = summary[:45] + "..."
        lines.append(
            f"  {name:<{name_w}}  {r.verdict:<18}  {flag:<6}  {n_hits:>4}  {summary}"
        )
    run_n = sum(1 for r in rows if r.run_3_5)
    lines.append("")
    lines.append(f"Totals: tasks={len(rows)}  run_3_5={run_n}  skip_3_5={len(rows) - run_n}")
    lines.append(f"Written: {PORTABILITY_SCAN_REL} per task")
    return "\n".join(lines)
