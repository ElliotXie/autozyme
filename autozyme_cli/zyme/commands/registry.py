"""`zyme registry` — searchable index of optimized functions and hot engines."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from zyme.scan import detect_phase, find_framework_root, find_tasks, find_workspace_root


FIELDNAMES = [
    "symbol",
    "kind",
    "status",
    "task",
    "task_dir",
    "phase",
    "round",
    "speedup_pct",
    "patch_path",
    "active_opts",
    "discoveries",
    "detail_path",
    "tags",
]

WORKFLOW_PRIMITIVES = {
    "seurat::normalizedata",
    "seurat::scaledata",
    "seurat::runpca",
    "seurat::findvariablefeatures",
    "seurat::findneighbors",
    "seurat::findclusters",
    "seurat::runumap",
    "seurat::sctransform",
    "seurat::findmarkers",
    "seurat::findallmarkers",
    "scanpy.pp.normalize_total",
    "scanpy.pp.log1p",
    "scanpy.pp.scale",
    "scanpy.pp.pca",
    "scanpy.tl.leiden",
    "scanpy.tl.rank_genes_groups",
}

DEPENDENCY_PATTERNS = [
    (r"\bmgcv::gam\b|\bgam\.fit[34]\b|\bestimate\.gam\b", "mgcv::gam", "numerical_engine", "mgcv,gam,trajectory"),
    (r"\bpresto::wilcoxauc\b|\brank_matrix(?:\.dgCMatrix)?\b", "presto::wilcoxauc", "workflow_primitive", "presto,wilcoxon,markers"),
    (r"\bSeurat::NormalizeData\b|\bNormalizeData\b", "Seurat::NormalizeData", "workflow_primitive", "seurat,normalize,lognormalize"),
    (r"\bSeurat::ScaleData\b|\bScaleData\b", "Seurat::ScaleData", "workflow_primitive", "seurat,scale"),
    (r"\bSeurat::RunPCA\b|\bRunPCA\b|\birlba\b", "Seurat::RunPCA", "workflow_primitive", "seurat,pca,irlba"),
    (r"\bFindVariableFeatures\b|\bvst\b", "Seurat::FindVariableFeatures", "workflow_primitive", "seurat,hvg,vst"),
    (r"\bFindNeighbors\b|\bRANN::nn2\b|\bRcppHNSW\b|\bHNSW\b", "Seurat::FindNeighbors", "workflow_primitive", "seurat,knn,hnsw"),
    (r"\bscipy\.signal\.sosfilt\b|\bsosfilt\b", "scipy.signal.sosfilt", "numerical_engine", "scipy,signal,filter"),
    (r"\bscipy\.fft\.ifft\b|\bFFT\b|\bifft\b", "scipy.fft.ifft", "numerical_engine", "scipy,fft"),
    (r"\bsklearn\b.*\bGradientBoostingRegressor\b|\bGradientBoostingRegressor\b", "sklearn.GradientBoostingRegressor", "numerical_engine", "sklearn,gbm,tree"),
    (r"\bIDAKLU\b|\bSUNDIALS\b", "PyBaMM.IDAKLU", "numerical_engine", "pybamm,sundials,ode"),
    (r"\bARPACK\b|\bscipy\.sparse\.linalg\b|\blobpcg\b", "ARPACK/LOBPCG", "numerical_engine", "eigen,arpack,lobpcg,blas"),
    (r"\bBLAS\b|\bdgemm\b|\bdgemv\b|\bApple Accelerate\b", "BLAS", "numerical_engine", "blas,linear-algebra"),
    (r"\brasterio\.warp\.transform\b", "rasterio.warp.transform", "workflow_primitive", "rasterio,geo,transform"),
    (r"\bpyproj\.Transformer\b|\bpyproj\b", "pyproj.Transformer", "workflow_primitive", "pyproj,geo,transform"),
    (r"\bxarray\b|\bapply_ufunc\b", "xarray.apply_ufunc", "workflow_primitive", "xarray,dispatch"),
    (r"\buwot::umap\b|\bumap\b", "uwot::umap", "workflow_primitive", "umap,knn"),
]

LEGACY_SYMBOLS = {
    "normalize_data": "Seurat::NormalizeData",
    "scale_data": "Seurat::ScaleData",
    "run_pca": "Seurat::RunPCA",
    "find_variable_features": "Seurat::FindVariableFeatures",
    "find_neighbors": "Seurat::FindNeighbors",
    "find_clusters": "Seurat::FindClusters",
    "run_umap": "Seurat::RunUMAP",
    "sctransform": "Seurat::SCTransform",
    "find_all_markers": "Seurat::FindAllMarkers",
    "integrate_cca": "Seurat::IntegrateData",
    "sc_normalize": "scanpy.pp.normalize_total",
    "sc_scale": "scanpy.pp.scale",
    "sc_pca": "scanpy.tl.pca",
    "sc_leiden": "scanpy.tl.leiden",
    "sc_rank_genes": "scanpy.tl.rank_genes_groups",
    "sc_regress_out": "scanpy.pp.regress_out",
    "sc_highly_variable": "scanpy.pp.highly_variable_genes",
}

STATUS_RANK = {
    "packaged": 70,
    "scaled": 60,
    "optimized_dependency": 55,
    "optimized": 50,
    "needs_review": 30,
    "initialized": 20,
    "dependency_bottleneck": 10,
    "scaffold": 0,
}


def _clean_yaml_value(raw: str) -> str:
    value = re.sub(r"\s+#.*$", "", raw).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value.strip()


def _read_task_fields(task_yaml: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    try:
        lines = task_yaml.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return fields
    for line in lines:
        if ":" not in line or line.startswith((" ", "\t", "#")):
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        if key in {"task", "target_repo", "target_function", "signature"}:
            fields[key] = _clean_yaml_value(raw)
    return fields


def _is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    v = value.strip()
    return (
        not v
        or "<" in v
        or ">" in v
        or "e.g." in v.lower()
        or v.upper() in {"NA", "NULL"}
    )


def _symbol_from_task(task_dir: Path, fields: dict[str, str]) -> str:
    target = fields.get("target_function", "")
    if not _is_placeholder(target):
        return target
    task = fields.get("task", "")
    if not _is_placeholder(task):
        return task
    return task_dir.name


def _slug(symbol: str) -> str:
    s = symbol.strip().replace("::", "__")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s or "unknown"


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9_.:-]{2,}", text.lower())}


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _kind_for(symbol: str, task_dir: Path, fields: dict[str, str]) -> str:
    low = symbol.lower()
    name = task_dir.name.lower()
    repo = fields.get("target_repo", "").lower()
    if low in WORKFLOW_PRIMITIVES or name in {
        "normalize_data",
        "scale_data",
        "run_pca",
        "find_variable_features",
        "find_neighbors",
        "find_clusters",
        "run_umap",
        "sctransform",
        "sc_normalize",
        "sc_scale",
        "sc_pca",
        "sc_leiden",
        "sc_rank_genes",
        "sc_regress_out",
        "sc_highly_variable",
    }:
        return "workflow_primitive"
    if any(x in low or x in repo for x in ("mgcv", "blas", "scipy", "sklearn", "pybamm", "arpack", "fipy")):
        return "numerical_engine"
    if "fit_false" in name:
        return "numerical_engine"
    return "target_function"


def _status_for(row: dict, latest: dict | None) -> str:
    if latest:
        pct = latest.get("speedup_pct")
        if isinstance(pct, (int, float)) and pct < 0:
            return "needs_review"
    if row["phases"]["package"]["done"]:
        return "packaged"
    if row["phases"]["scaling"]["done"]:
        return "scaled"
    if row["phases"]["iterate"]["done"] and latest:
        return "optimized"
    if row["phases"]["init"]["done"]:
        return "initialized"
    return "scaffold"


def _read_limited(path: Path, max_chars: int = 160_000) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except OSError:
        return ""


def _headings(path: Path, limit: int = 10) -> list[str]:
    out: list[str] = []
    for line in _read_limited(path, 80_000).splitlines():
        if line.startswith("## "):
            out.append(line[3:].strip())
        if len(out) >= limit:
            break
    return out


def _entry_tags(symbol: str, task_dir: Path, fields: dict[str, str], kind: str) -> str:
    parts = [symbol, task_dir.name, fields.get("task", ""), fields.get("target_repo", ""), kind]
    return ",".join(sorted(t for t in _tokens(" ".join(parts)) if len(t) <= 40))


def _is_bench_task(task_dir: Path) -> bool:
    parts = set(task_dir.parts)
    return "bench_runs" in parts


def _resolve_roots(args) -> tuple[list[Path], Path | None, Path | None]:
    cwd = Path.cwd()
    workspace = find_workspace_root(cwd)
    framework = Path(args.framework_root).resolve() if getattr(args, "framework_root", None) else (
        (workspace / "autozyme-framework").resolve() if workspace else find_framework_root(cwd)
    )
    roots = [Path(p).resolve() for p in getattr(args, "paths", []) if Path(p).is_dir()]
    if not roots:
        roots = [workspace] if workspace else [cwd]
    return roots, workspace, framework


def _registry_root(args, framework: Path | None) -> Path:
    if getattr(args, "registry_root", None):
        return Path(args.registry_root).resolve()
    if framework:
        return framework / "registry"
    return Path.cwd() / "registry"


def _make_target_entry(task_dir: Path, row: dict) -> dict:
    fields = _read_task_fields(task_dir / "task.yaml")
    symbol = _symbol_from_task(task_dir, fields)
    latest = row.get("latest_keep")
    kind = _kind_for(symbol, task_dir, fields)
    active = task_dir / "memory" / "active_opts.md"
    discoveries = task_dir / "memory" / "discoveries.md"
    return {
        "symbol": symbol,
        "kind": kind,
        "status": _status_for(row, latest),
        "task": fields.get("task") or row.get("task_name") or task_dir.name,
        "task_dir": str(task_dir),
        "phase": row.get("phase") or "",
        "round": str((latest or {}).get("round") or ""),
        "speedup_pct": "" if not latest or latest.get("speedup_pct") is None else str(latest.get("speedup_pct")),
        "patch_path": row["phases"]["package"].get("patch_path") or "",
        "active_opts": str(active) if active.is_file() else "",
        "discoveries": str(discoveries) if discoveries.is_file() else "",
        "detail_path": "",
        "tags": _entry_tags(symbol, task_dir, fields, kind),
    }


def _latest_keep_from_results(results_tsv: Path) -> dict | None:
    if not results_tsv.is_file():
        return None
    try:
        lines = results_tsv.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if len(lines) < 2:
        return None
    header = lines[0].split("\t")
    col = {n: i for i, n in enumerate(header)}
    if "status" not in col:
        return None
    for line in reversed(lines[1:]):
        parts = line.split("\t")
        if len(parts) <= col["status"] or parts[col["status"]] != "keep":
            continue
        def get(name: str) -> str:
            i = col.get(name)
            return parts[i] if i is not None and i < len(parts) else ""
        try:
            pct = float(get("speedup_pct"))
        except ValueError:
            pct = None
        return {
            "round": get("round"),
            "dataset": get("dataset"),
            "speedup_pct": pct,
        }
    return None


def _legacy_result_dirs(roots: list[Path], task_dirs: list[Path], include_bench: bool) -> list[Path]:
    task_reals = {p.resolve() for p in task_dirs}
    out: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for results in root.glob("*/*/results.tsv"):
            d = results.parent
            if d.resolve() in task_reals or (d / "task.yaml").exists():
                continue
            if not include_bench and _is_bench_task(d):
                continue
            if d.name in LEGACY_SYMBOLS:
                out.append(d)
        for results in root.glob("*/results.tsv"):
            d = results.parent
            if d.resolve() in task_reals or (d / "task.yaml").exists():
                continue
            if not include_bench and _is_bench_task(d):
                continue
            if d.name in LEGACY_SYMBOLS:
                out.append(d)
    return sorted(set(out), key=lambda p: p.as_posix())


def _make_legacy_entry(task_dir: Path) -> dict:
    symbol = LEGACY_SYMBOLS.get(task_dir.name, task_dir.name)
    latest = _latest_keep_from_results(task_dir / "results.tsv")
    active = task_dir / "memory" / "active_opts.md"
    discoveries = task_dir / "memory" / "discoveries.md"
    pct = latest.get("speedup_pct") if latest else None
    status = "optimized" if latest and not (isinstance(pct, (int, float)) and pct < 0) else ("needs_review" if latest else "scaffold")
    return {
        "symbol": symbol,
        "kind": "workflow_primitive",
        "status": status,
        "task": task_dir.name,
        "task_dir": str(task_dir),
        "phase": "legacy",
        "round": str((latest or {}).get("round") or ""),
        "speedup_pct": "" if pct is None else str(pct),
        "patch_path": "",
        "active_opts": str(active) if active.is_file() else "",
        "discoveries": str(discoveries) if discoveries.is_file() else "",
        "detail_path": "",
        "tags": ",".join(sorted(_tokens(f"{symbol} {task_dir.name} workflow primitive legacy"))),
    }


def _dependency_hits(task_dir: Path) -> list[tuple[str, str, str, str]]:
    corpus = "\n".join([
        _read_limited(task_dir / "README.md", 120_000),
        _read_limited(task_dir / "memory" / "discoveries.md", 120_000),
        _read_limited(task_dir / "memory" / "active_opts.md", 80_000),
    ])
    hits: list[tuple[str, str, str, str]] = []
    for pattern, symbol, kind, tags in DEPENDENCY_PATTERNS:
        m = re.search(pattern, corpus, flags=re.IGNORECASE | re.DOTALL)
        if m:
            snippet_start = max(0, m.start() - 90)
            snippet_end = min(len(corpus), m.end() + 140)
            snippet = re.sub(r"\s+", " ", corpus[snippet_start:snippet_end]).strip()
            hits.append((symbol, kind, tags, snippet))
    return hits


def _write_detail(root: Path, entry: dict, related: list[dict] | None = None, evidence: list[str] | None = None) -> str:
    detail_dir = root / "functions"
    detail_dir.mkdir(parents=True, exist_ok=True)
    path = detail_dir / f"{_slug(entry['symbol'])}.md"
    lines = [
        f"# {entry['symbol']}",
        "",
        f"- Kind: `{entry['kind']}`",
        f"- Status: `{entry['status']}`",
        f"- Task: `{entry['task']}`",
        f"- Phase: `{entry['phase']}`",
    ]
    if entry.get("speedup_pct"):
        lines.append(f"- Latest speedup: `{entry['speedup_pct']}%`")
    if entry.get("round"):
        lines.append(f"- Latest keep round: `{entry['round']}`")
    if entry.get("task_dir"):
        lines.append(f"- Task dir: `{entry['task_dir']}`")
    if entry.get("patch_path"):
        lines.append(f"- Package patch: `{entry['patch_path']}`")
    if entry.get("active_opts"):
        lines.append(f"- Active opts: `{entry['active_opts']}`")
    if entry.get("discoveries"):
        lines.append(f"- Discoveries: `{entry['discoveries']}`")
    lines.append("")
    active_path = Path(entry.get("active_opts") or "")
    heads = _headings(active_path)
    if heads:
        lines.extend(["## Active Optimization Headings", ""])
        lines.extend(f"- {h}" for h in heads)
        lines.append("")
    if evidence:
        lines.extend(["## Dependency Evidence", ""])
        for ev in evidence[:8]:
            lines.append(f"- {ev}")
        lines.append("")
    if related:
        lines.extend(["## Related Tasks", ""])
        for rel in related[:20]:
            desc = f"- `{rel['task']}` [{rel['status']}]"
            if rel.get("speedup_pct"):
                desc += f" speedup={rel['speedup_pct']}%"
            if rel.get("active_opts"):
                desc += f" active_opts=`{rel['active_opts']}`"
            lines.append(desc)
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _load_entries(root: Path) -> list[dict]:
    tsv = root / "functions.tsv"
    if not tsv.is_file():
        raise FileNotFoundError(f"registry not found at {tsv}; run `zyme registry rebuild` first")
    with tsv.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _write_entries(root: Path, entries: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "functions.tsv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for e in entries:
            writer.writerow({k: e.get(k, "") for k in FIELDNAMES})


def cmd_registry_rebuild(args) -> None:
    roots, _workspace, framework = _resolve_roots(args)
    registry_root = _registry_root(args, framework)
    task_dirs = find_tasks(roots, max_depth=args.max_depth, framework_root=framework)
    if not args.include_bench:
        task_dirs = [p for p in task_dirs if not _is_bench_task(p)]

    target_entries: list[dict] = []
    dependency_map: dict[str, dict] = {}
    related_by_symbol: dict[str, list[dict]] = defaultdict(list)
    evidence_by_symbol: dict[str, list[str]] = defaultdict(list)

    for task_dir in task_dirs:
        row = detect_phase(task_dir, framework)
        entry = _make_target_entry(task_dir, row)
        target_entries.append(entry)
        for symbol, kind, tags, snippet in _dependency_hits(task_dir):
            dep = dependency_map.setdefault(symbol, {
                "symbol": symbol,
                "kind": kind,
                "status": "dependency_bottleneck",
                "task": "",
                "task_dir": "",
                "phase": "",
                "round": "",
                "speedup_pct": "",
                "patch_path": "",
                "active_opts": "",
                "discoveries": "",
                "detail_path": "",
                "tags": tags,
            })
            if entry["symbol"].lower() == symbol.lower() and entry["status"] in {"packaged", "scaled", "optimized"}:
                dep["status"] = "optimized_dependency"
                dep["task"] = entry["task"]
                dep["task_dir"] = entry["task_dir"]
                dep["phase"] = entry["phase"]
                dep["round"] = entry["round"]
                dep["speedup_pct"] = entry["speedup_pct"]
                dep["patch_path"] = entry["patch_path"]
                dep["active_opts"] = entry["active_opts"]
                dep["discoveries"] = entry["discoveries"]
            related_by_symbol[symbol].append(entry)
            if snippet:
                evidence_by_symbol[symbol].append(f"{task_dir.name}: {snippet}")

    legacy_entries = [_make_legacy_entry(d) for d in _legacy_result_dirs(roots, task_dirs, args.include_bench)]
    target_entries.extend(legacy_entries)

    direct_symbols = {
        e["symbol"].lower()
        for e in target_entries
        if e["status"] in {"packaged", "scaled", "optimized", "optimized_dependency", "needs_review"}
    }
    dependency_entries = [
        dep for sym, dep in dependency_map.items()
        if sym.lower() not in direct_symbols
    ]
    all_entries = target_entries + dependency_entries

    by_symbol: dict[str, list[dict]] = defaultdict(list)
    for entry in all_entries:
        by_symbol[entry["symbol"].lower()].append(entry)

    for entries_for_symbol in by_symbol.values():
        primary = max(
            entries_for_symbol,
            key=lambda e: (STATUS_RANK.get(e.get("status", ""), 0), bool(e.get("active_opts")), e.get("speedup_pct") or ""),
        )
        related = list(entries_for_symbol) + related_by_symbol.get(primary["symbol"], [])
        evidence = evidence_by_symbol.get(primary["symbol"], [])
        detail = _write_detail(registry_root, primary, related=related, evidence=evidence)
        for entry in entries_for_symbol:
            entry["detail_path"] = detail

    all_entries.sort(key=lambda e: (e["kind"], e["symbol"].lower(), e["task"]))
    _write_entries(registry_root, all_entries)
    print(f"registry: wrote {len(all_entries)} entries from {len(task_dirs)} task(s) + {len(legacy_entries)} legacy result dir(s)")
    print(f"root    : {registry_root}")
    print(f"index   : {registry_root / 'functions.tsv'}")


def _score(entry: dict, query_tokens: set[str], raw_query: str) -> int:
    text = " ".join(str(entry.get(k, "")) for k in ("symbol", "kind", "status", "task", "tags")).lower()
    score = 0
    raw = raw_query.lower()
    if raw and raw in text:
        score += 20
    compact_text = _compact(text)
    compact_raw = _compact(raw_query)
    if compact_raw and compact_raw in compact_text:
        score += 30
    raw_terms = [
        t for t in re.findall(r"[a-z0-9][a-z0-9_.:-]{1,}", raw)
        if t not in {"the", "and", "for", "with", "from", "this", "that"}
    ][:12]
    term_hits = 0
    for term in raw_terms:
        compact_term = _compact(term)
        if (term and term in text) or (compact_term and compact_term in compact_text):
            term_hits += 1
            score += 8
    if raw_terms and term_hits == len(raw_terms):
        score += 40
    toks = _tokens(text)
    score += 3 * len(query_tokens & toks)
    if entry.get("active_opts"):
        score += 1
    if entry.get("status") in {"packaged", "scaled", "optimized", "optimized_dependency"}:
        score += 2
    return score


def _print_entry(entry: dict, *, show_paths: bool = True) -> None:
    speed = f" speedup={entry['speedup_pct']}%" if entry.get("speedup_pct") else ""
    print(f"- {entry['symbol']} [{entry['kind']} / {entry['status']}]{speed}")
    if show_paths:
        if entry.get("task_dir"):
            print(f"  task: {entry['task']}  {entry['task_dir']}")
        if entry.get("active_opts"):
            print(f"  active_opts: {entry['active_opts']}")
        if entry.get("detail_path"):
            print(f"  detail: {entry['detail_path']}")


def cmd_registry_query(args) -> None:
    registry_root = _registry_root(args, find_framework_root(Path.cwd()))
    entries = _load_entries(registry_root)
    raw = " ".join(args.terms)
    q = _tokens(raw)
    ranked = [(e, _score(e, q, raw)) for e in entries]
    ranked = [(e, s) for e, s in ranked if s > 0]
    ranked.sort(key=lambda x: (-x[1], x[0]["symbol"].lower()))
    for entry, _s in ranked[:args.limit]:
        _print_entry(entry)
    if not ranked:
        print(f"registry: no match for {raw!r}", file=sys.stderr)


def _extract_suggest_query(task_path: Path | None, profile_path: Path | None) -> str:
    chunks: list[str] = []
    if task_path:
        task_path = task_path.resolve()
        if task_path.is_dir():
            chunks.append(_read_limited(task_path / "task.yaml", 20_000))
            chunks.append(_read_limited(task_path / "README.md", 80_000))
            chunks.append(_read_limited(task_path / "memory" / "discoveries.md", 60_000))
        elif task_path.is_file():
            chunks.append(_read_limited(task_path, 40_000))
            if task_path.name == "task.yaml":
                chunks.append(_read_limited(task_path.parent / "README.md", 80_000))
    if profile_path and profile_path.is_file():
        text = _read_limited(profile_path, 120_000)
        try:
            data = json.loads(text)
            chunks.append(json.dumps(data.get("actionable_hotspots", data), default=str)[:80_000])
        except Exception:
            chunks.append(text)
    return "\n".join(chunks)


def cmd_registry_suggest(args) -> None:
    registry_root = _registry_root(args, find_framework_root(Path.cwd()))
    entries = _load_entries(registry_root)
    query_text = _extract_suggest_query(
        Path(args.task).resolve() if args.task else None,
        Path(args.profile).resolve() if args.profile else None,
    )
    if not query_text:
        raise SystemExit("registry suggest requires --task and/or --profile")
    q = _tokens(query_text)
    ranked = [(e, _score(e, q, query_text[:200])) for e in entries]
    ranked = [(e, s) for e, s in ranked if s > 0]
    ranked.sort(key=lambda x: (-x[1], x[0]["symbol"].lower()))

    direct = [e for e, _ in ranked if e["status"] in {"packaged", "scaled", "optimized", "optimized_dependency"}]
    deps = [e for e, _ in ranked if e["status"] == "dependency_bottleneck" or e["kind"] == "numerical_engine"]
    similar = [e for e, _ in ranked if e.get("active_opts")]

    print("Registry Suggestions")
    print("====================")
    print()
    print("Optimized / Reusable Matches")
    for entry in direct[:args.limit]:
        _print_entry(entry)
    if not direct:
        print("- none")
    print()
    print("Low-Level / Dependency Candidates")
    for entry in deps[:args.limit]:
        _print_entry(entry)
    if not deps:
        print("- none")
    print()
    print("Similar Task Active Opts")
    seen: set[str] = set()
    count = 0
    for entry in similar:
        key = entry.get("active_opts") or entry.get("task_dir")
        if not key or key in seen:
            continue
        seen.add(key)
        _print_entry(entry)
        count += 1
        if count >= args.limit:
            break
    if count == 0:
        print("- none")


def cmd_registry_list(args) -> None:
    registry_root = _registry_root(args, find_framework_root(Path.cwd()))
    entries = _load_entries(registry_root)
    rows = [e for e in entries if not args.kind or e.get("kind") == args.kind]
    rows.sort(key=lambda e: (e["kind"], e["symbol"].lower(), e["status"]))
    for entry in rows[:args.limit]:
        _print_entry(entry, show_paths=False)
