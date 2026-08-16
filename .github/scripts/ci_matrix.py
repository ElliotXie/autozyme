#!/usr/bin/env python3
"""Generate the GitHub Actions matrix for Tier B / Tier C CI.

Walks both packages, reads each patch's ``tested_upstream_versions``
declaration, joins with the canonical task_dir for that patch, and emits
a JSON object the workflows feed into ``strategy.matrix`` via fromJson().

Tier B  — one matrix entry per plugin (latest declared version only).
Tier C  — cartesian product: one entry per (plugin, declared_version).

Usage (CI):
    python .github/scripts/ci_matrix.py --tier=B
    python .github/scripts/ci_matrix.py --tier=C --output=$GITHUB_OUTPUT

Usage (local debug):
    python .github/scripts/ci_matrix.py --tier=B --pretty

Python plugins are discovered by statically scanning ``register_patch`` calls.
R plugins are discovered by parsing ``autozyme_r/inst/patches/*/patch.R`` for the
``tested_upstream_versions = list(Pkg = "X.Y.Z", ...)`` literal — no R
interpreter required.

Plugins with no ``tested_upstream_versions`` declaration are skipped
with a stderr warning. Plugins with no canonical task_dir (see
PLUGIN_TASK_DIR below) are also skipped — declare one to enroll.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Canonical task_dir for each plugin. verify_patch() reads tier datasets
# from <task_dir>/reference_output_{tier}/. Add an entry when you wire a
# new plugin into Tier B/C — until then the plugin is skipped.
PLUGIN_TASK_DIR: dict[str, str] = {
    # --- Python plugins ---
    "cell2location":         "optimized_task/test_core_singlecell/test_cell2location",
    "cellphonedb":           "optimized_task/test_core_singlecell/test_cellphonedb_v5",
    "dipy":                  "optimized_task/test_general_bio/test_dipy_dti",
    "fipy":                  "optimized_task/test_non_bio/test_fipy",
    "lifelines":             "optimized_task/test_general_bio/test_lifelines_cox",
    "mdanalysis":            "optimized_task/test_general_bio/test_mdanalysis_hbonds",
    "mdanalysis_rmsd":       "optimized_task/test_non_bio/test_mdanalysis",
    "obspy":                 "optimized_task/test_non_bio/test_obspy",
    "prody":                 "optimized_task/test_general_bio/test_prody",
    "sarsen":                "optimized_task/test_non_bio/test_sarsen",
    "scvelo":                "optimized_task/test_core_singlecell/test_scvelo_recover_dynamics",
    "squidpy_cooccurrence":  "optimized_task/test_general_bio/test_squidpy_cooccurrence",
    "statsmodels":           "optimized_task/test_non_bio/test_statsmodels",
    "scanpy":                "optimized_task/test_seurat_scanpy/scanpy_pipeline",
    "xclim":                 "optimized_task/test_non_bio/test_xclim",
    "astropy_boxleastsquares": "optimized_task/test_non_bio/test_astropy_boxleastsquares",

    # --- R plugins ---
    "bayesspace":            "optimized_task/test_general_bio/test_bayesspace",
    "cellchat":              "optimized_task/test_core_singlecell/test_cellchat",
    "clusterprofiler":       "optimized_task/test_general_bio/test_clusterprofiler",
    "decontx":               "optimized_task/test_core_singlecell/test_decontx",
    "fgsea":                 "optimized_task/test_general_bio/test_fgsea",
    "infercnv":              "optimized_task/test_core_singlecell/test_infercnv_hmm",
    "maftools":              "optimized_task/test_general_bio/test_maftools",
    "mast":                  "optimized_task/test_core_singlecell/test_mast",
    "nichenetr":             "optimized_task/test_core_singlecell/test_nichenet",
    "rctd":                  "optimized_task/test_core_singlecell/test_RCTD",
    "scriabin":              "optimized_task/test_core_singlecell/test_scriabin",
    "seurat":                "optimized_task/test_seurat_scanpy/find_all_markers/v3",
    "slingshot":             "optimized_task/test_core_singlecell/test_slingshot",
    "tradeseq":              "optimized_task/test_core_singlecell/test_tradeseq_fitgam",
    "ucell":                 "optimized_task/test_general_bio/test_ucell",
    "vegan":                 "optimized_task/test_general_bio/test_vegan_adonis2",
    "wgcna":                 "optimized_task/test_general_bio/test_wgcna_blockwise_real",
}

# Per-patch install specs for upstreams that were lifted against an installable
# git commit rather than a released PyPI version. Keys are
# (plugin, upstream package, tested_upstream_version). Values are accepted by
# pip install exactly as emitted here.
PY_PIP_SPEC_OVERRIDES: dict[tuple[str, str, str], str] = {
    (
        "sarsen",
        "sarsen",
        "0.9.6.dev5+g6c5e37d1d",
    ): (
        "sarsen @ "
        "git+https://github.com/bopen/sarsen@"
        "6c5e37d1d3d2f124a051209c8a8a6b15ba51d8d7"
    ),
}


def _warn(msg: str) -> None:
    print(f"::warning::{msg}", file=sys.stderr)


def _extract_py_versions(path: Path) -> dict[str, list[str]] | None:
    """AST-scan a plugin __init__.py for the literal
    ``tested_upstream_versions={...}`` kwarg of register_patch.

    Pure static parse — no plugin import, no upstream dependency. Returns
    None if the field isn't declared or isn't a plain dict literal.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "register_patch"):
            continue
        for kw in node.keywords:
            if kw.arg != "tested_upstream_versions":
                continue
            if isinstance(kw.value, ast.Constant) and kw.value.value is None:
                return None
            if not isinstance(kw.value, ast.Dict):
                return None
            try:
                out: dict[str, list[str]] = {}
                for k_node, v_node in zip(kw.value.keys, kw.value.values):
                    if not isinstance(k_node, ast.Constant) or not isinstance(k_node.value, str):
                        return None
                    if not isinstance(v_node, ast.List):
                        return None
                    vers = []
                    for e in v_node.elts:
                        if not isinstance(e, ast.Constant) or not isinstance(e.value, str):
                            return None
                        vers.append(e.value)
                    out[k_node.value] = vers
                return out
            except Exception:
                return None
    return None


def discover_py_plugins() -> list[dict]:
    """Static-scan each Python plugin __init__.py for tested_upstream_versions.

    Does NOT import any plugin — keeps matrix-gen lightweight so CI doesn't
    have to install 17 heavy upstream packages just to enumerate the matrix.
    """
    plugins_root = ROOT / "autozyme_py" / "src" / "autozyme"
    if not plugins_root.is_dir():
        return []
    rows = []
    for child in sorted(plugins_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_") or child.name.endswith(".egg-info"):
            continue
        init = child / "__init__.py"
        if not init.is_file():
            continue
        versions = _extract_py_versions(init)
        if versions is None:
            _warn(f"py plugin {child.name!r}: no tested_upstream_versions declared; skipped")
            continue
        rows.append({
            "lang": "py",
            "plugin": child.name,
            "upstream_versions": versions,
        })
    return rows


# Match the literal R declaration:
#   tested_upstream_versions = list(BayesSpace = "1.21.2")
#   tested_upstream_versions = list(Seurat = c("5.0.3", "5.1.0"),
#                                    SeuratObject = "5.0.2")
_R_LIST_RE = re.compile(
    r"tested_upstream_versions\s*=\s*list\((?P<body>.*?)\)\s*[,)]",
    re.DOTALL,
)
_R_PAIR_RE = re.compile(
    r"(?P<pkg>\w[\w.]*)\s*=\s*(?P<val>c\([^)]*\)|\"[^\"]*\")",
    re.DOTALL,
)


def _parse_r_versions(body: str) -> dict[str, list[str]]:
    """Turn the inside of list(...) into a dict[str, list[str]]."""
    out: dict[str, list[str]] = {}
    for m in _R_PAIR_RE.finditer(body):
        pkg = m.group("pkg")
        raw = m.group("val").strip()
        if raw.startswith("c("):
            inner = raw[2:-1]
            vers = [v.strip().strip('"') for v in inner.split(",") if v.strip()]
        else:
            vers = [raw.strip('"')]
        out[pkg] = vers
    return out


def discover_r_plugins() -> list[dict]:
    """Text-scan each R patch file for tested_upstream_versions."""
    patches_dir = ROOT / "autozyme_r" / "inst" / "patches"
    if not patches_dir.is_dir():
        return []
    rows = []
    for path in sorted(patches_dir.glob("*/patch.R")):
        name = path.parent.name
        text = path.read_text(encoding="utf-8")
        m = _R_LIST_RE.search(text)
        if not m:
            _warn(f"r plugin {name!r}: no tested_upstream_versions declared; skipped")
            continue
        versions = _parse_r_versions(m.group("body"))
        if not versions:
            _warn(f"r plugin {name!r}: tested_upstream_versions parsed to empty; "
                  f"check syntax in {path.relative_to(ROOT)}")
            continue
        rows.append({
            "lang": "r",
            "plugin": name,
            "upstream_versions": versions,
        })
    return rows


def build_matrix(rows: list[dict], tier: str) -> list[dict]:
    """Turn discovery rows into one matrix entry per cell to run.

    Tier B: pick the latest declared version per upstream-pkg (single entry per plugin).
    Tier C: one entry per (plugin, every declared version).
    """
    out = []
    for row in rows:
        task_dir = PLUGIN_TASK_DIR.get(row["plugin"])
        if task_dir is None:
            _warn(f"{row['lang']} plugin {row['plugin']!r}: no entry in "
                  f"PLUGIN_TASK_DIR; add one to .github/scripts/ci_matrix.py to enroll")
            continue
        if not (ROOT / task_dir).is_dir():
            _warn(f"plugin {row['plugin']!r}: task_dir {task_dir!r} does not "
                  f"exist on disk; skipping")
            continue
        for pkg, versions in row["upstream_versions"].items():
            sorted_vers = _sort_versions(versions)
            picked = [sorted_vers[-1]] if tier == "B" else sorted_vers
            for v in picked:
                pip_spec = ""
                if row["lang"] == "py":
                    pip_spec = PY_PIP_SPEC_OVERRIDES.get(
                        (row["plugin"], pkg, v),
                        f"{pkg}=={v}",
                    )
                out.append({
                    "lang": row["lang"],
                    "plugin": row["plugin"],
                    "task_dir": task_dir,
                    "task_name": Path(task_dir).name,
                    "upstream_pkg": pkg,
                    "upstream_version": v,
                    # Convenience strings for the install step:
                    "pip_spec": pip_spec,
                })
    return out


def _sort_versions(versions: list[str]) -> list[str]:
    """Sort by PEP 440 ordering when possible; fall back to lexical."""
    try:
        from packaging.version import Version
        return sorted(versions, key=Version)
    except Exception:
        return sorted(versions)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=["B", "C"], required=True)
    p.add_argument("--output", default="",
                   help="Path to a file (e.g. $GITHUB_OUTPUT). "
                        "Empty = print JSON to stdout.")
    p.add_argument("--pretty", action="store_true",
                   help="Pretty-print JSON to stdout (debug; ignores --output).")
    args = p.parse_args()

    if not (ROOT / "optimized_task").is_dir():
        _warn("optimized_task/ is not present; Tier B/C verify matrix is empty "
              "in release-only checkouts")
        matrix = []
    else:
        rows = discover_py_plugins() + discover_r_plugins()
        matrix = build_matrix(rows, args.tier)

    payload = {"include": matrix}

    if args.pretty:
        print(json.dumps(payload, indent=2))
        print(f"\n{len(matrix)} matrix entries (tier {args.tier})", file=sys.stderr)
        return 0

    blob = json.dumps(payload, separators=(",", ":"))
    if args.output:
        with open(args.output, "a", encoding="utf-8") as fp:
            fp.write(f"matrix={blob}\n")
    else:
        print(blob)
    print(f"matrix entries: {len(matrix)} (tier {args.tier})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
