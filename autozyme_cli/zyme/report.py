#!/usr/bin/env python3
"""Generate a self-contained HTML optimization report for a zyme task.

Reads:
  - results.tsv
  - memory/best_state.md, memory/active_opts.md, memory/dead_ends.md,
memory/discoveries.md, memory/transfer_knowledge.md
  - task.yaml (for target metadata)

Writes:
  - report.html  (single file, light + dark theme toggle, no external assets
              beyond Google Fonts CDN for typography)
"""
from __future__ import annotations
import base64, csv, html, json, math, re
from collections import defaultdict
from pathlib import Path

# This module lives 3 levels deep: autozyme_cli/zyme/report.py
# So autozyme-framework root is .parent.parent.parent.
_FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent.parent
_LOGO_FILE = _FRAMEWORK_ROOT / "website" / "public" / "assets" / "brand" / "autozyme-logo-horizontal.png"

FIELD_MAP = {
    "test_core_singlecell": "single-cell genomics",
    "core_singlecell": "single-cell genomics",
    "test_spatial": "spatial transcriptomics",
    "general_bio": "computational biology",
    "non_bio": "scientific computing",
}

# CSS custom properties referenced by both the page CSS and the inline SVG.
# Two palettes — light (default) and dark (toggle).
VAR = {k: f"var(--{k})" for k in [
    "bg", "surface", "elev", "border", "hair", "text", "dim", "muted",
    "gold", "jade", "rose", "slate", "amber", "ink", "ink_text",
]}

# -------- parse results.tsv --------
def parse_results(task_dir):
    rows = []
    with open(task_dir / "results.tsv") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            try:
                rd = float(r["round"])
            except ValueError:
                rd = -1.0
            rows.append(dict(
                round=r["round"], round_f=rd, commit=r["commit"],
                dataset=r["dataset"],
                speed=float(r["speed_sec"]) if r["speed_sec"] else 0.0,
                pct=float(r["speedup_pct"]) if r["speedup_pct"] else 0.0,
                peak=float(r["peak_mb"]) if r["peak_mb"] else 0.0,
                status=r["status"], hyp=r.get("hypothesis", ""),
                desc=r.get("description", ""),
                metrics_json=r.get("metrics_json", "{}"),
                phase=r.get("phase", ""),
            ))
    return rows

def read_md(task_dir, name):
    p = task_dir / "memory" / name
    return p.read_text() if p.exists() else ""

def parse_sections(text, prefix="## "):
    out, cur_t, cur_b = [], None, []
    for line in text.splitlines():
        if line.startswith(prefix):
            if cur_t is not None:
                out.append((cur_t, "\n".join(cur_b).strip()))
            cur_t, cur_b = line[len(prefix):].strip(), []
        else:
            if cur_t is not None:
                cur_b.append(line)
    if cur_t is not None:
        out.append((cur_t, "\n".join(cur_b).strip()))
    return out

def parse_patch_stack(text):
    m = re.search(r"## Patch stack.*?\n((?:- .*\n?)+)", text, re.DOTALL)
    if not m: return []
    out = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        m2 = re.match(r"- round (\d+) \(([0-9a-f]+)\) — (-?[\d.]+)% — (.+)", line)
        if m2:
            out.append(dict(
                round=int(m2.group(1)), commit=m2.group(2),
                pct=float(m2.group(3)), desc=m2.group(4),
            ))
    return out

def categorize(name, desc):
    """Priority-ordered categorization. Memory descs are explicitly prefixed
'memory:' by the agent's journal convention — check those first so they
don't get swallowed by 'memo' substring matches or 'n.iter' co-mentions."""
    s = (name + " " + desc).lower()
    if s.startswith("memory") or " rm(" in s or "gc(" in s or "reclaim" in s:
        return "memory"
    if "rcpp" in s or "_cpp" in s:
        return "rcpp"
    if any(k in s for k in ["bayesnet", "mcmc", "jags", "gibb", "n.iter", "chain", "n.adapt"]):
        return "algorithmic"
    if any(k in s for k in ["region", "subcluster", "consensus", "skip", "side-effect", "rds", "synthesize"]):
        return "algorithmic"
    if any(k in s for k in ["cache", "memoize", "memoise", "spline"]):
        return "cache_reuse"
    if any(k in s for k in ["vector", "broadcast", "fused", "fuse", "single-pass", "one pass"]):
        return "vectorize_fuse"
    return "other"

CAT_META = dict(
    algorithmic=("Algorithmic",   VAR["gold"],  "Reduce work fundamentally — fewer MCMC samples, faster posterior structure, eliminate unused side-effects."),
    rcpp=         ("Rcpp ports",  VAR["jade"],  "Move hot R loops into single-pass C++ — eliminate temp matrices and per-element R dispatch."),
    cache_reuse=  ("Cache & reuse", VAR["slate"], "Detect invariant work; fit once, predict many; share precomputed structures."),
    vectorize_fuse=("Vectorize & fuse", "var(--olive)", "Collapse N matrix passes into 1; broadcast instead of allocate."),
    memory=       ("Memory hygiene", VAR["rose"], "Drop objects the moment they become unreachable; reclaim before peak measurement."),
    other=        ("Other",       VAR["dim"],   "Misc step-level wins."),
)



def _read_narrative(narrative_path):
    """Parse memory/report_narrative.md into a dict of section_name → body string.

    All sections optional. Missing file or missing section returns empty string
    (caller does fallback to mechanical defaults).
    """
    if not narrative_path.exists():
        return {}
    text = narrative_path.read_text()
    out = {}
    cur_key, cur_body = None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if cur_key is not None:
                out[cur_key] = "\n".join(cur_body).strip()
            cur_key = line[3:].strip().lower().replace(" ", "_")
            cur_body = []
        elif cur_key is not None:
            cur_body.append(line)
    if cur_key is not None:
        out[cur_key] = "\n".join(cur_body).strip()
    return out


def _parse_tier_descriptions(body):
    """Parse `name: description` lines into a dict. Empty body → {}."""
    out = {}
    for line in (body or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Allow either `- name: desc` or `name: desc`
        if line.startswith("- "):
            line = line[2:].strip()
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


# 17-angle / 3-tier taxonomy from paper/figures/fig3/tier_taxonomy_and_mechanisms.md.
# Each entry: angle_num → (display_name, tier, one_line_description).
ANGLE_TAXONOMY = {
    "1":  ("eliminate",         "T1", "Skip dead / redundant / unused work."),
    "2":  ("overhead",          "T1", "Cheaper primitives; bypass dispatch; fewer copies."),
    "3":  ("reuse",             "T1", "Cache, memoize, hoist loop-invariants."),
    "4":  ("setup-process",     "T1", "Pre-warm / pre-compile inside the process."),
    "5":  ("setup-disk",        "T1", "Cross-process disk cache."),
    "6":  ("representation",    "T2", "Sparse/dense, dtype, layout, streaming."),
    "7a": ("analytical",        "T2", "Closed-form / math identity / drop state-independent term."),
    "7b": ("better-algorithm",  "T2", "Complexity downgrade / fundamentally different algorithm."),
    "7c": ("fusion",            "T2", "Fuse multiple kernels / loops / graph nodes into one."),
    "7d": ("vectorize",         "T2", "Per-row apply → batched primitive; library backend swap."),
    "8":  ("compile-source",    "T2", "Write Rcpp / numba / Cython to rewrite the hotpath."),
    "9":  ("compile-build",     "T2", "Compiler flags / byte-compile / alternative BLAS."),
    "10": ("parallel",          "T2", "Multi-process / multi-thread / GPU."),
    "11": ("convergence",       "T3", "Loosen iteration cap / eps / early stop."),
    "12": ("precision",         "T3", "float32 / low-precision intermediates / -ffast-math."),
    "13": ("approximation",     "T3", "Wald instead of LR / top-K / prefilter / drop correction."),
    "14": ("subsample",         "T3", "Fewer draws / candidates / poses."),
}

# Visual roll-up for the donut. 3 risk tiers + a FLAG bucket for self-labeled rounds
# the taxonomy doesn't cover.
TIER_META = dict(
    T1=    ("Local / same-output",        VAR["jade"],  "Bit-exact (or fp-noise <1e-10): polish-class wins."),
    T2=    ("Structural / same-math",     VAR["gold"],  "Math-equivalent restructuring; the workhorse layer."),
    T3=    ("Lossy / bounded",            VAR["rose"],  "Output drifts inside declared tolerance bands."),
    FLAG=  ("Self-labeled / out-of-taxonomy", VAR["slate"], "Rounds that didn't fit the 17-angle taxonomy; flagged for review."),
)


def _parse_round_angles(body):
    """Parse the `## round_angles` body of report_narrative.md.

    Each non-blank, non-comment line should look like one of:

        round 11: 10 parallel [T2] + 8 compile-source — rationale
        round 11: 10 parallel [T2] — rationale
        round 32: FLAG: build-portability — rationale

    Returns a list of dicts ordered as they appear:
      {round, primary_num, primary_name, tier, secondary, rationale, flag}
    where `flag=True` means the line uses `FLAG:` and we couldn't assign a
    taxonomy angle.
    """
    out = []
    if not body:
        return out
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        m = re.match(r"round\s+(\d+)\s*:\s*(.+)", line, re.IGNORECASE)
        if not m:
            continue
        rd = int(m.group(1))
        rest = m.group(2)
        # rationale is everything after the first em-dash / hyphen-dash if present
        rationale = ""
        for sep in (" — ", " -- ", " - "):
            if sep in rest:
                rest, _, rationale = rest.partition(sep)
                rationale = rationale.strip()
                break
        rest = rest.strip()
        # FLAG path: `FLAG: <self-label>` (case-insensitive)
        flag_m = re.match(r"FLAG\s*:\s*(.+)", rest, re.IGNORECASE)
        if flag_m:
            out.append(dict(
                round=rd, primary_num="FLAG", primary_name=flag_m.group(1).strip(),
                tier="FLAG", secondary=None, rationale=rationale, flag=True,
            ))
            continue
        # Taxonomy path: `<num> <name> [<tier>] [+ <sec_num> <sec_name>]`
        prim_m = re.match(
            r"([0-9]+[a-z]?)\s+([A-Za-z0-9_-]+)\s*\[(T[1-3])\](?:\s*\+\s*([0-9]+[a-z]?)\s+([A-Za-z0-9_-]+))?",
            rest,
        )
        if not prim_m:
            # Couldn't parse; still record under FLAG so it surfaces.
            out.append(dict(
                round=rd, primary_num="FLAG", primary_name=rest[:80],
                tier="FLAG", secondary=None, rationale=rationale, flag=True,
            ))
            continue
        sec = None
        if prim_m.group(4):
            sec = (prim_m.group(4), prim_m.group(5))
        out.append(dict(
            round=rd, primary_num=prim_m.group(1), primary_name=prim_m.group(2),
            tier=prim_m.group(3), secondary=sec, rationale=rationale, flag=False,
        ))
    return out


def _subst(template, mapping):
    """Substitute {key} placeholders. Unknown keys left as-is (no KeyError)."""
    if not template:
        return ""
    def _repl(m):
        return str(mapping.get(m.group(1), m.group(0)))
    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", _repl, template)


def render_report(task_dir, output_path=None, narrative_path=None, open_after=False):
    """Render a self-contained HTML optimization report.

    Reads `results.tsv` + `memory/*.md` from `task_dir` and writes
    `task_dir/report.html` (or `output_path` if provided). Optionally reads
    `memory/report_narrative.md` for agent-curated narrative slots.

    Returns the Path to the written file.
    """
    from pathlib import Path
    task_dir = Path(task_dir)
    if output_path is None:
        output_path = task_dir / "report.html"
    if narrative_path is None:
        narrative_path = task_dir / "memory" / "report_narrative.md"
    output_path = Path(output_path)

    # Embed logo as data URL (skip silently if missing — report still renders).
    LOGO_DATA = ""
    if _LOGO_FILE.exists():
        LOGO_DATA = "data:image/png;base64," + base64.b64encode(_LOGO_FILE.read_bytes()).decode()

    # Infer scientific field from parent directory name (test_core_singlecell → "single-cell genomics").
    FIELD = FIELD_MAP.get(task_dir.parent.name, "scientific computing")

    # Read optional narrative file (agent-curated overrides) — additive, no break.
    NARRATIVE = _read_narrative(narrative_path)
    # -------- aggregates --------
    results = parse_results(task_dir)
    best_md = read_md(task_dir, "best_state.md")
    active_md = read_md(task_dir, "active_opts.md")
    dead_md = read_md(task_dir, "dead_ends.md")
    disc_md = read_md(task_dir, "discoveries.md")

    patch = parse_patch_stack(best_md)
    active_secs = [s for s in parse_sections(active_md) if not s[0].startswith("<opt_name>")]
    dead_secs = parse_sections(dead_md)
    disc_secs = [
        s for s in parse_sections(disc_md)
        if s[0].startswith("DISCOVERY") and "<one-line" not in s[0] and "SUMMARY" not in s[0]
    ]

    baselines, finals = {}, {}
    for r in results:
        if r["status"] == "baseline":
            baselines[r["dataset"]] = r
    for r in sorted(results, key=lambda x: x["round_f"], reverse=True):
        if r["status"] in ("keep", "rerun") and r["speed"] > 0:
            finals.setdefault(r["dataset"], r)

    keep_rounds = [r for r in results if r["status"] == "keep"]
    discard_rounds = [r for r in results if r["status"] == "discard"]
    crash_rounds = [r for r in results if r["status"] == "crash"]

    # Resolve which dataset to use as the "journey chart" anchor.
    #
    # The naive pick is the task.yaml `tier: tiny` dataset, but some tasks
    # iterate primarily on a larger tier — e.g. xclim's tiny tier finishes
    # in <1s after round 2, so the agent moved decision rounds to medium
    # where times remained meaningful. In that case the tiny tier produces
    # a near-empty chart while the actual iteration history lives on medium.
    #
    # Strategy: count decision rounds (keep/discard/crash, excluding reruns)
    # per dataset; the chart anchor ("primary") is whichever dataset hosted
    # the most decisions. The secondary anchor (used for the medium-tier mean
    # column in the top-5 detail cards) is the next-most-active dataset.
    _decision_counts = defaultdict(int)
    for r in results:
        if r["status"] in ("keep", "discard", "crash"):
            _decision_counts[r["dataset"]] += 1
    _by_activity = sorted(_decision_counts, key=lambda k: -_decision_counts[k])
    _tiny_name = _by_activity[0] if _by_activity else None
    _medium_name = _by_activity[1] if len(_by_activity) > 1 else _tiny_name
    # Final fallback when results.tsv has no decision rows yet (early state):
    # rank by smallest baseline speed.
    if _tiny_name is None and baselines:
        _tiny_name = min(baselines, key=lambda k: baselines[k].get("speed", 0))
    if _medium_name is None and baselines:
        _by_speed = sorted(baselines, key=lambda k: baselines[k].get("speed", 0))
        _medium_name = _by_speed[1] if len(_by_speed) > 1 else _by_speed[0]

    # Friendly label for the chart anchor — show the task.yaml tier slug
    # ("tiny", "medium", etc.) when it maps; else the dataset name itself.
    _chart_anchor_label = _tiny_name or "iteration"
    try:
        from zyme.parsers.task_yaml import parse_datasets as _pd2
        for _ds in _pd2(task_dir / "task.yaml"):
            if _ds.get("name") == _tiny_name and _ds.get("tier"):
                _chart_anchor_label = _ds["tier"]
                break
    except Exception:
        pass

    # tiny progression for chart (one row per unique round on tiny)
    tiny_progression = []
    seen = set()
    for r in results:
        if r["dataset"] != _tiny_name: continue
        if r["status"] not in ("baseline", "keep", "discard", "crash"): continue
        if r["round"] in seen: continue
        seen.add(r["round"])
        tiny_progression.append(r)
    tiny_progression.sort(key=lambda x: x["round_f"])

    # medium-tier per-round means (one per kept round)
    medium_by_round = defaultdict(list)
    for r in results:
        if r["dataset"] != _medium_name: continue
        if r["status"] in ("keep", "rerun"):
            medium_by_round[int(r["round_f"])].append(r["speed"])
    medium_mean = {k: sum(v)/len(v) for k, v in medium_by_round.items() if v}

    # attribution counts — prefer the agent-curated `round_angles` block in
    # report_narrative.md (LLM-assigned, persisted, against the fig3 17-angle
    # taxonomy). Fall back to the regex `categorize()` only when no angles are
    # provided. The two modes feed different donut shapes:
    #   - angle mode: donut = tier rollup (T1/T2/T3/FLAG); legend = per-angle
    #   - legacy mode: donut + legend = the 6 regex buckets
    round_angles = _parse_round_angles(NARRATIVE.get("round_angles", ""))
    # Restrict to rounds that actually made it into the accepted patch stack —
    # ignore angle lines for rounds the agent classified but that weren't kept.
    _patch_round_set = {p["round"] for p in patch}
    round_angles = [a for a in round_angles if a["round"] in _patch_round_set]
    use_angle_mode = bool(round_angles)

    if use_angle_mode:
        # tier rollup for donut
        attr_counts = defaultdict(int)
        for a in round_angles:
            attr_counts[a["tier"]] += 1
        attr_counts = dict(attr_counts)
        # per-angle counts for legend
        angle_counts = defaultdict(int)
        for a in round_angles:
            key = (a["primary_num"], a["primary_name"], a["tier"])
            angle_counts[key] += 1
    else:
        attr = defaultdict(list)
        for p in patch:
            cat = categorize(p["desc"], p["desc"])
            attr[cat].append(p)
        attr_counts = {k: len(v) for k, v in attr.items()}
        angle_counts = {}

    # ---- top-5 marginal-impact rounds on tiny tier ----
    def find_tiny_speed(rd_int):
        for r in tiny_progression:
            if r["status"] == "keep" and int(r["round_f"]) == rd_int:
                return r["speed"]
        return None

    top5_data = []
    prev = None
    # iterate through keep rounds in order; compute delta vs previous keep
    for p in patch:
        cur = find_tiny_speed(p["round"])
        if cur is None: continue
        # find the previous accepted tiny speed
        delta_abs = (prev - cur) if prev is not None else None
        delta_pct = ((prev - cur) / prev * 100) if prev else None
        # medium delta
        med_cur = medium_mean.get(p["round"])
        top5_data.append(dict(p, tiny_speed=cur, tiny_prev=prev,
                              delta_abs=delta_abs, delta_pct=delta_pct,
                              medium=med_cur))
        prev = cur
    # also seed with round 0 baseline-> round 1 delta
    baseline_tiny = baselines.get(_tiny_name, {}).get("speed", 0)
    for i, d in enumerate(top5_data):
        if d["tiny_prev"] is None and baseline_tiny:
            d["tiny_prev"] = baseline_tiny
            d["delta_abs"] = baseline_tiny - d["tiny_speed"]
            d["delta_pct"] = (baseline_tiny - d["tiny_speed"]) / baseline_tiny * 100

    # rank by absolute seconds saved on tiny
    top5 = sorted([d for d in top5_data if d.get("delta_abs")], key=lambda x: -x["delta_abs"])[:5]

    # parse active opt sections into dict by name. Accepts three journal styles:
    #   - infercnv: `Round: N` / `Mechanism: ...` plain-colon fields.
    #   - bold-markdown: `**Speedup**: ...` / `**Concordance**: ...` (astropy).
    #   - free-form: no field labels at all; first paragraph = mechanism.
    _FIELD_NAMES = (
        "Round", "Source rounds", "Mechanism", "Speedup",
        "Speedup contribution", "Concordance", "Location", "Notes", "Caveat",
    )
    _field_pattern = re.compile(
        r"^\s*(?:\*\*)?(" + "|".join(re.escape(n) for n in _FIELD_NAMES) + r")(?:\*\*)?\s*:\s*(.*)$"
    )
    def parse_opt(title, body):
        inh = "[inherited" in title
        dorm = "dormant" in title.lower()
        clean = re.sub(r"\s*\[inherited[^\]]*\]", "", title).strip()
        fields = {}
        current_key = None
        current_value = []
        preamble = []   # lines before any field label (free-form fallback source)
        for line in body.splitlines():
            m = _field_pattern.match(line)
            if m:
                # Close out the previous multi-line field, then start the new one.
                if current_key:
                    fields[current_key] = "\n".join(current_value).strip()
                current_key = m.group(1)
                if current_key == "Speedup":
                    current_key = "Speedup contribution"
                current_value = [m.group(2)]
            elif current_key is not None:
                # Continuation of the in-progress field.
                current_value.append(line)
            else:
                preamble.append(line)
        if current_key:
            fields[current_key] = "\n".join(current_value).strip()
        # Free-form fallback: if no Mechanism field was found, use the first
        # non-field paragraph of the preamble (lets pure-prose journals work).
        if "Mechanism" not in fields and preamble:
            free_text = "\n".join(preamble).strip()
            if free_text:
                first_para = re.split(r"\n\s*\n", free_text, maxsplit=1)[0].strip()
                if first_para:
                    fields["Mechanism"] = first_para
        return clean, inh, dorm, fields

    opt_lookup = {}  # by lower keyword
    opt_by_round = {}
    for title, body in active_secs:
        if not body: continue
        name, inh, dorm, fields = parse_opt(title, body)
        rec = dict(name=name, inherited=inh, dormant=dorm,
                   mech=fields.get("Mechanism",""),
                   speed=fields.get("Speedup contribution",""),
                   conc=fields.get("Concordance",""),
                   loc=fields.get("Location",""),
                   notes=fields.get("Notes",""),
                   rounds=fields.get("Round") or fields.get("Source rounds",""))
        opt_lookup[name.lower()] = rec
        # Extract round numbers from BOTH the dedicated `Round:` / `Source rounds:`
        # field (infercnv-style journal) AND the section title (nichenet-style
        # `## Round 2: ...` heading). Either format alone is enough to wire
        # opts to patch-stack entries.
        round_source = rec["rounds"] + " " + name
        for m in re.finditer(r"[Rr]ound[s]?\s+(\d+)", round_source):
            try:
                rn = int(m.group(1))
                if rn < 1000:
                    opt_by_round.setdefault(rn, []).append(rec)
            except ValueError:
                pass
        # Also catch bare-number lists like "1, 2, 3" in the dedicated rounds field
        # (only — not the title, which would pick up unrelated numbers).
        if rec["rounds"]:
            for m in re.finditer(r"\b(\d+)\b", rec["rounds"]):
                try:
                    rn = int(m.group(1))
                    if rn < 1000:
                        opt_by_round.setdefault(rn, []).append(rec)
                except ValueError:
                    pass

    # match patch-stack entry to an opt card
    def match_opt(p):
        if p["round"] in opt_by_round:
            return opt_by_round[p["round"]][0]
        d = p["desc"].lower()
        for k, rec in opt_lookup.items():
            first = k.split()[0]
            if first and first in d:
                return rec
        return None

    # de cards. Two journal styles supported:
    #   - infercnv: `Tried: ...` / `Why it doesn't work: ...` / `What would change ...` fields.
    #   - free-form (astropy etc.): plain prose under each `## <angle>` heading;
    #     extract round from title's "(round N, commit X)" pattern, first
    #     paragraph → why, remaining paragraphs → verdict.
    _de_field_pattern = re.compile(
        r"^\s*(?:\*\*)?(Tried|Why it doesn't work|What would change the verdict|Variations|Cause|Why)(?:\*\*)?\s*:\s*(.*)$"
    )
    de_cards = []
    for title, body in dead_secs:
        if not body or "empty" in body[:30].lower() or title.startswith("<"): continue
        tried = why = verdict = vars_ = ""
        non_field_lines = []
        for line in body.splitlines():
            m = _de_field_pattern.match(line)
            if m:
                k, v = m.group(1), m.group(2).strip()
                if   k == "Tried":                          tried = v
                elif k in ("Why it doesn't work", "Why"):   why = v
                elif k == "What would change the verdict":  verdict = v
                elif k == "Variations":                     vars_ = v
                elif k == "Cause":                          why = why or v
            else:
                non_field_lines.append(line)
        # Free-form fallback: split body paragraphs. First → why, rest → verdict.
        # Extract "round N" + commit from the title.
        free_text = "\n".join(non_field_lines).strip()
        if not why and free_text:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", free_text) if p.strip()]
            if paragraphs:
                why = paragraphs[0]
                if len(paragraphs) > 1 and not verdict:
                    verdict = "\n\n".join(paragraphs[1:])
        if not tried:
            # Try to mine "round N (commit X)" or "(round N, commit X)" from the title
            m_rd = re.search(r"round\s+(\d+)", title, re.IGNORECASE)
            m_sha = re.search(r"commit\s+([0-9a-f]{6,12})", title, re.IGNORECASE)
            if m_rd:
                tried = f"round {m_rd.group(1)}"
                if m_sha:
                    tried += f" ({m_sha.group(1)})"
        if not why: continue
        de_cards.append(dict(title=title, tried=tried, why=why, verdict=verdict, vars=vars_))

    # discovery cards. Two schemas are supported:
    #   - infercnv style: `Triggered:` / `Cause:` / `Implication:` fields.
    #   - free-form style (astropy, etc.): plain prose under each `## DISCOVERY:`
    #     heading, split into paragraphs. First paragraph → cause, rest → impl.
    disc_cards = []
    for title, body in disc_secs:
        t = title.replace("DISCOVERY:", "").strip()
        if t.startswith("<") or "<one-line" in t: continue
        cause = impl = triggered = ""
        has_fields = False
        for line in body.splitlines():
            m = re.match(r"^(Cause|Implication|Triggered):\s*(.*)$", line)
            if m:
                has_fields = True
                if   m.group(1) == "Cause":       cause = m.group(2)
                elif m.group(1) == "Implication": impl = m.group(2)
                elif m.group(1) == "Triggered":   triggered = m.group(2)
        # Free-form fallback: split body into paragraphs (blank-line separated).
        # First paragraph carries the observation; any later paragraphs become
        # the implication. This catches journals that don't use field labels.
        if not has_fields and body.strip():
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body.strip()) if p.strip()]
            if paragraphs:
                cause = paragraphs[0]
                impl  = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else ""
            # Try to find a `round N` mention in the body when no `Triggered:` field.
            if not triggered:
                triggered = body
        if not cause or "<what's actually" in cause: continue
        # extract round number(s) from triggered/body text
        rnums = re.findall(r"round[s]?\s+(\d+)", triggered)
        disc_cards.append(dict(title=t, cause=cause, impl=impl, rounds=rnums))

    # Build TIER_ORDER from task.yaml's datasets list. Description comes from
    # the narrative file (tier_descriptions slot) if provided, else a title-cased
    # fallback derived from the dataset name.
    tier_desc_map = _parse_tier_descriptions(NARRATIVE.get("tier_descriptions", ""))
    TIER_ORDER = []
    try:
        from zyme.parsers.task_yaml import parse_datasets
        for ds in parse_datasets(task_dir / "task.yaml"):
            name = ds["name"]
            tier = ds["tier"]
            desc = tier_desc_map.get(name) or name.replace("_", " ").title()
            TIER_ORDER.append((name, tier, desc))
    except Exception:
        # Fallback: derive tiers from the baselines we have
        for name in sorted(baselines.keys(), key=lambda n: baselines[n].get("speed", 0)):
            desc = tier_desc_map.get(name) or name.replace("_", " ").title()
            TIER_ORDER.append((name, name, desc))

    # Hero numbers aggregate ACROSS all tiers (median, not single-tier) — robust
    # to outlier tiers and impossible to cherry-pick by selecting "production".
    # Per-tier speedup = baseline_sec / final_sec; per-tier mem delta = signed %.
    import statistics
    per_tier_speedup = []   # list of (tier_name, speedup_ratio)
    per_tier_mem    = []    # list of (tier_name, mem_delta_pct)  positive = regression
    for ds in baselines:
        b = baselines.get(ds, {})
        f = finals.get(ds, {})
        if b.get("speed", 0) > 0 and f.get("speed", 0) > 0:
            per_tier_speedup.append((ds, b["speed"] / f["speed"]))
        if b.get("peak", 0) > 0 and f.get("peak", 0) > 0:
            per_tier_mem.append((ds, (f["peak"] - b["peak"]) / b["peak"] * 100))

    if per_tier_speedup:
        _speedups = [s for _, s in per_tier_speedup]
        hero_speedup = statistics.median(_speedups)
        hero_speedup_lo, hero_speedup_hi = min(_speedups), max(_speedups)
        # speedup_pct of the median (e.g. 27× → 96.3% faster)
        hero_pct = (1 - 1.0 / hero_speedup) * 100 if hero_speedup > 0 else 0.0
    else:
        hero_speedup, hero_speedup_lo, hero_speedup_hi, hero_pct = 0.0, 0.0, 0.0, 0.0

    if per_tier_mem:
        _mems = [m for _, m in per_tier_mem]
        hero_mem_delta_pct = statistics.median(_mems)
        hero_mem_lo, hero_mem_hi = min(_mems), max(_mems)
        if all(m < 0 for m in _mems):
            hero_mem_sign_status = "all_saved"
        elif all(m > 0 for m in _mems):
            hero_mem_sign_status = "all_regressed"
        else:
            hero_mem_sign_status = "mixed"
    else:
        hero_mem_delta_pct, hero_mem_lo, hero_mem_hi = 0.0, 0.0, 0.0
        hero_mem_sign_status = "n/a"

    n_hero_tiers = len(per_tier_speedup)

    # Backward-compat aliases used by the (unrendered) tldr string + narrative
    # placeholder substitution. Pick the slowest-baseline tier as a *contextual*
    # reference for "{baseline}" / "{final}" / "{tier}" placeholders — the
    # narrative author can still mention a specific tier; the hero tiles
    # themselves don't use these.
    _ctx_tier = max(baselines, key=lambda k: baselines[k].get("speed", 0)) if baselines else None
    hero_baseline = baselines.get(_ctx_tier, {}).get("speed", 0.0) if _ctx_tier else 0.0
    hero_final    = finals.get(_ctx_tier, {}).get("speed", 0.0) if _ctx_tier else 0.0
    hero_mem_b    = baselines.get(_ctx_tier, {}).get("peak", 0.0) if _ctx_tier else 0.0
    hero_mem_f    = finals.get(_ctx_tier, {}).get("peak", 0.0) if _ctx_tier else 0.0
    hero_tier_desc = next((d for n, t, d in TIER_ORDER if n == _ctx_tier), _ctx_tier or "")
    hero_tier_name = _ctx_tier  # used by threading lookup + latest_metrics lookup

    latest_metrics = {}
    for r in results:
        if r["status"] in ("keep", "rerun") and r["metrics_json"] and r["metrics_json"] != "{}":
            try:
                latest_metrics[r["dataset"]] = (r["round"], json.loads(r["metrics_json"]))
            except Exception:
                pass

    # ============================================================
    # HTML helpers
    # ============================================================
    def esc(s): return html.escape(str(s)) if s is not None else ""

    # inline SVG charts use CSS custom properties for colors via var(--...)
    def svg_memory_chart(rows, width=960, height=200, mode="abs", baseline_mb=0.0):
        """Linear memory chart parallel to speed chart — same x-axis (rounds).

        mode="abs":   y = peak MB (linear)
        mode="ratio": y = peak / baseline_mb  (× of baseline; <1 = saved, >1 = cost)
                      A horizontal "parity" line at y=1.0 marks the baseline.
        """
        SANE_MAX_MB = 200_000
        pts = []
        for r in rows:
            if r["peak"] > 0 and math.isfinite(r["peak"]) and r["peak"] < SANE_MAX_MB:
                pts.append(r)
        if not pts: return ""

        if mode == "ratio" and baseline_mb > 0:
            vals = [r["peak"] / baseline_mb for r in pts]
            unit_suffix = "×"
        else:
            mode = "abs"
            vals = [r["peak"] for r in pts]
            unit_suffix = " MB"

        if mode == "ratio":
            # Log y-scale — ratios span orders of magnitude (incore caches push
            # peaks to 30×+ a sparse baseline); linear y crowds the 1–10× anchors
            # at the bottom.
            log_lo = math.log10(0.5)
            log_hi = math.log10(max(max(vals) * 1.15, 1.5))
            ymax = 10 ** log_hi
            def yscale(v):
                lv = math.log10(max(v, 0.5))
                return height - 40 - (lv - log_lo) / (log_hi - log_lo) * (height - 60)
        else:
            ymax = max(vals) * 1.12
            ymin = 0
            def yscale(v):
                return height - 40 - (v - ymin) / (ymax - ymin) * (height - 60)
        def xscale(i):
            return 70 + i / max(1, len(rows) - 1) * (width - 100)
        coords = [(xscale(rows.index(r)), yscale(vals[i])) for i, r in enumerate(pts)]
        line = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        area = (f"M {coords[0][0]:.1f},{height-40} L "
                + " L ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
                + f" L {coords[-1][0]:.1f},{height-40} Z")
        # gridlines: ratio mode auto-picks anchors by ymax so labels span the
        # actual data range — fixed [0.25..5] crowds at the bottom when incore
        # caches push peaks to 30×+ of a sparse baseline. The 1× line is styled
        # as the baseline-parity reference. Abs mode uses round MB increments.
        grid_html = []
        parity_line_html = ""
        if mode == "ratio":
            # Decade-spaced anchors; log y spreads them evenly regardless of range.
            ratio_anchors = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
                             100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
            for v in ratio_anchors:
                if v < 0.5 or v > ymax: continue
                y = yscale(v)
                if abs(v - 1.0) < 1e-6:
                    parity_line_html = f'<line x1="65" y1="{y:.1f}" x2="{width-25}" y2="{y:.1f}" stroke="var(--gold)" stroke-width="1" opacity="0.5" stroke-dasharray="6,4"/><text x="60" y="{y+4:.1f}" text-anchor="end" fill="var(--gold)" font-size="11" font-weight="600" font-family="ui-monospace,monospace">1.0×</text>'
                else:
                    grid_html.append(f'<line x1="65" y1="{y:.1f}" x2="{width-25}" y2="{y:.1f}" stroke="var(--hair)" stroke-dasharray="2,4"/>')
                    label = f"{v:g}×"
                    grid_html.append(f'<text x="60" y="{y+4:.1f}" text-anchor="end" fill="var(--dim)" font-size="11" font-family="ui-monospace,monospace">{label}</text>')
        else:
            step = 1000 if ymax > 3000 else 500
            g = step
            while g < ymax:
                y = yscale(g)
                grid_html.append(f'<line x1="65" y1="{y:.1f}" x2="{width-25}" y2="{y:.1f}" stroke="var(--hair)" stroke-dasharray="2,4"/>')
                grid_html.append(f'<text x="60" y="{y+4:.1f}" text-anchor="end" fill="var(--dim)" font-size="11" font-family="ui-monospace,monospace">{g} MB</text>')
                g += step
        xlabels = []
        for i in range(0, len(rows), 5):
            x = xscale(i)
            xlabels.append(f'<text x="{x:.1f}" y="{height-12}" text-anchor="middle" fill="var(--dim)" font-size="10" font-family="ui-monospace,monospace">r{rows[i]["round"]}</text>')
        dots = []
        for r in pts:
            i = rows.index(r)
            v = vals[pts.index(r)]
            x, y = xscale(i), yscale(v)
            if r["status"] == "keep": cls = "kp"
            elif r["status"] == "discard": cls = "ds"
            elif r["status"] == "crash": cls = "cr"
            else: cls = "bl"
            desc = (r["desc"] or r["hyp"] or "")[:280]
            dots.append(
                f'<circle class="dot dot-{cls}" cx="{x:.1f}" cy="{y:.1f}" r="4" '
                f'data-rd="{esc(r["round"])}" data-sha="{esc(r["commit"])}" '
                f'data-spd="{r["speed"]:.3f}" data-st="{esc(r["status"])}" '
                f'data-desc="{esc(desc)}" data-mem="{r["peak"]:.0f}" data-y="mem"/>'
            )
        svg_id = "memSvg" if mode == "abs" else "memSvgX"
        grad_id = "memG" if mode == "abs" else "memGx"
        return f'''
<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" id="{svg_id}" class="memSvg mem-svg-{mode}" style="width:100%;height:auto;overflow:visible">
  <defs>
    <linearGradient id="{grad_id}" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="var(--rose)" stop-opacity="0.16"/>
      <stop offset="100%" stop-color="var(--rose)" stop-opacity="0.0"/>
    </linearGradient>
  </defs>
  {"".join(grid_html)}
  {parity_line_html}
  <path class="chart-area" d="{area}" fill="url(#{grad_id})"/>
  <path class="chart-line" d="{line}" fill="none" stroke="var(--rose)" stroke-width="1.6" stroke-linejoin="round" opacity="0.55"/>
  {"".join(dots)}
  {"".join(xlabels)}
</svg>'''

    def svg_speed_chart(rows, width=960, height=320, mode="sec", baseline_sec=0.0):
        """Render the per-round speed chart on a log y-axis.

        mode="sec":    y = wall-clock seconds (smaller = faster)
        mode="speedup": y = baseline_sec / speed (× faster vs baseline; bigger = better)
        """
        speeds = [r["speed"] for r in rows]
        if mode == "speedup" and baseline_sec > 0:
            # convert speeds → speedup ×. Crashes (speed=0) get NaN-ish handling: use 1× placeholder.
            vals = [(baseline_sec / s) if s > 0 else 0 for s in speeds]
            unit = "×"
            grid_anchors = [1, 3, 10, 30, 100, 300, 1000]
        else:
            mode = "sec"
            vals = list(speeds)
            unit = "s"
            grid_anchors = [1, 3, 10, 30, 100, 300, 1000]

        positives = [v for v in vals if v > 0]
        ymax = max(positives) * 1.15 if positives else 1.0
        ymin = max(0.01, min(positives) * 0.7) if positives else 0.01

        def yscale(v):
            v = v if v > 0 else ymin
            lv, lmin, lmax = math.log10(v), math.log10(ymin), math.log10(ymax)
            return height - 60 - (lv - lmin) / (lmax - lmin) * (height - 90)
        def xscale(i):
            return 70 + i / max(1, len(rows) - 1) * (width - 100)
        pts = [(xscale(i), yscale(vals[i])) for i in range(len(rows))]
        # The progress line + area fill should reflect only what was actually adopted:
        # baseline + accepted rounds. Discards/crashes are drawn as faded dots without
        # connecting them — they're explored states, not the trajectory.
        accepted_pts = [
            pts[i] for i, r in enumerate(rows)
            if r["status"] in ("baseline", "keep", "rerun") and vals[i] > 0
        ]
        if accepted_pts:
            line = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in accepted_pts)
            area = (f"M {accepted_pts[0][0]:.1f},{height-60} L "
                    + " L ".join(f"{x:.1f},{y:.1f}" for x, y in accepted_pts)
                    + f" L {accepted_pts[-1][0]:.1f},{height-60} Z")
        else:
            line = ""
            area = ""
        # gridlines
        grid_html = []
        for v in grid_anchors:
            if v < ymin or v > ymax: continue
            y = yscale(v)
            grid_html.append(f'<line x1="65" y1="{y:.1f}" x2="{width-25}" y2="{y:.1f}" stroke="var(--hair)" stroke-dasharray="2,4"/>')
            grid_html.append(f'<text x="60" y="{y+4:.1f}" text-anchor="end" fill="var(--dim)" font-size="11" font-family="ui-monospace,monospace">{v}{unit}</text>')
        # x labels
        xlabels = []
        for i in range(0, len(rows), 5):
            x = pts[i][0]
            xlabels.append(f'<text x="{x:.1f}" y="{height-30}" text-anchor="middle" fill="var(--dim)" font-size="10" font-family="ui-monospace,monospace">r{rows[i]["round"]}</text>')
        # dots — each has data-attrs for JS hover. In speedup mode we also stash data-spdx so the
        # tooltip can show "× faster" without recomputing.
        dots = []
        for i, r in enumerate(rows):
            if r["status"] == "keep": cls = "kp"
            elif r["status"] == "discard": cls = "ds"
            elif r["status"] == "crash": cls = "cr"
            else: cls = "bl"
            x, y = pts[i]
            desc = (r["desc"] or r["hyp"] or "")[:280]
            dots.append(
                f'<circle class="dot dot-{cls}" cx="{x:.1f}" cy="{y:.1f}" r="4" '
                f'data-rd="{esc(r["round"])}" data-sha="{esc(r["commit"])}" '
                f'data-spd="{r["speed"]:.3f}" data-st="{esc(r["status"])}" '
                f'data-desc="{esc(desc)}" data-mem="{r["peak"]:.0f}"/>'
            )
        svg_id = "speedSvg" if mode == "sec" else "speedSvgX"
        grad_id = "areaG" if mode == "sec" else "areaGx"
        return f'''
<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" id="{svg_id}" class="speed-svg speed-svg-{mode}" style="width:100%;height:auto;overflow:visible">
  <defs>
    <linearGradient id="{grad_id}" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="var(--jade)" stop-opacity="0.18"/>
      <stop offset="100%" stop-color="var(--jade)" stop-opacity="0.0"/>
    </linearGradient>
  </defs>
  {"".join(grid_html)}
  <path class="chart-area" d="{area}" fill="url(#{grad_id})"/>
  <path class="chart-line" d="{line}" fill="none" stroke="var(--jade)" stroke-width="1.6" stroke-linejoin="round" opacity="0.7"/>
  {"".join(dots)}
  {"".join(xlabels)}
</svg>
'''

    def render_tier_chart(tiers):
        """HTML+CSS interactive bar chart with on-load animation and per-row hover detail."""
        rows = []
        for ds, label, full in tiers:
            b = baselines.get(ds); f = finals.get(ds)
            if not b or not f or b["speed"] <= 0 or f["speed"] <= 0: continue
            m = latest_metrics.get(ds, (None, {}))[1]
            rows.append(dict(
                ds=ds, label=label, full=full,
                baseline=b["speed"], final=f["speed"],
                speedup=b["speed"]/f["speed"],
                mem_b=b["peak"], mem_f=f["peak"],
                mem_pct=(b["peak"]-f["peak"])/b["peak"]*100 if b["peak"] else 0,
                commit=f["commit"], metrics=m,
            ))
        if not rows: return ""
        # Per-row linear: track = baseline (100%), green = final/baseline.
        row_html = []
        for i, r in enumerate(rows):
            pb = 100.0
            pf = max(0.0, r["final"] / r["baseline"] * 100.0) if r["baseline"] > 0 else 0.0
            m = r["metrics"]
            all_one = all(v >= 0.99 for v in m.values() if isinstance(v, (int, float)))
            # extra metrics line shown on hover
            extras = []
            if "pearson_expr" in m:        extras.append(f'pearson = {m["pearson_expr"]:.2f}')
            if "hmm_state_agreement" in m: extras.append(f'HMM agreement = {m["hmm_state_agreement"]:.2f}')
            if "hmm_amp_del_concord" in m: extras.append(f'amp/del concord = {m["hmm_amp_del_concord"]:.2f}')
            row_html.append(f'''
<div class="tier-row" style="--delay: {i*90}ms">
  <div class="tier-info">
    <div class="tier-name">{esc(r["label"])}</div>
    <div class="tier-desc">{esc(r["full"])}</div>
  </div>
  <div class="tier-trackwrap">
    <div class="tier-track">
      <div class="tier-bar-base" style="--w: {pb:.2f}%"></div>
      <div class="tier-bar-fast" style="--w: {pf:.2f}%"></div>
      <div class="tier-tag tier-tag-fast" style="left: {pf:.2f}%">{r["final"]:.2f}s</div>
      <div class="tier-tag tier-tag-base" style="left: {pb:.2f}%">{r["baseline"]:.1f}s</div>
    </div>
  </div>
  <div class="tier-result">
    <div class="tier-speedup"><span class="num">{r["speedup"]:.1f}</span><span class="x">×</span></div>
    <div class="tier-mem">{r["mem_f"]:.0f} <span class="dim">MB peak</span></div>
  </div>
  <div class="tier-extras">
    <span class="ext-commit">commit {esc(r["commit"])}</span>
    {' <span class="ext-sep">·</span> ' + ' <span class="ext-sep">·</span> '.join(esc(x) for x in extras) if extras else ''}
    {' <span class="ext-sep">·</span> <span class="ext-ok">bit-exact</span>' if all_one and extras else ''}
  </div>
</div>''')

        # Per-row linear: each track is normalized to its own baseline, so
        # the axis shows fraction-of-baseline marks (same for all rows).
        axis_ticks = []
        for p, label in [(0, "0"), (25, "25%"), (50, "50%"), (75, "75%"), (100, "baseline")]:
            axis_ticks.append(f'<div class="tier-tick" style="left: {p}%"><span>{label}</span></div>')

        return f'''
<div class="tier-chart" id="tierChart">
  <div class="tier-list">
    {"".join(row_html)}
  </div>
  <div class="tier-axis-row">
    <div></div>
    <div class="tier-axis-track">{"".join(axis_ticks)}</div>
    <div></div>
  </div>
</div>'''

    def render_attribution_chart(attr_counts, angle_counts=None, use_angle_mode=False):
        """Interactive donut with linked legend.

        Two modes:
          - legacy: attr_counts keyed by the 6 regex buckets; CAT_META supplies
            color + description.
          - angle: attr_counts keyed by tier ("T1"/"T2"/"T3"/"FLAG"); TIER_META
            supplies color + description. `angle_counts` keyed by
            (angle_num, angle_name, tier) drives the legend (one row per angle
            actually present, grouped by tier).
        """
        total = sum(attr_counts.values()) or 1
        meta_table = TIER_META if use_angle_mode else CAT_META
        default_meta = TIER_META.get("FLAG") if use_angle_mode else CAT_META.get("other")
        # Order tiers explicitly when in angle mode so T1→T2→T3→FLAG reads
        # left-to-right around the donut; sort by count in legacy mode.
        if use_angle_mode:
            tier_order = ["T1", "T2", "T3", "FLAG"]
            cats_ordered = [(t, attr_counts.get(t, 0)) for t in tier_order]
        else:
            cats_ordered = sorted(attr_counts.items(), key=lambda x: -x[1])
        cx, cy, r_out, r_in = 150, 150, 112, 68
        a0 = -math.pi / 2
        segs, legend_data = [], []
        for i, (cat, n) in enumerate(cats_ordered):
            if n == 0: continue
            meta = meta_table.get(cat, default_meta)
            frac = n / total
            a1 = a0 + frac * 2 * math.pi
            large = 1 if frac > 0.5 else 0
            x0 = cx + r_out * math.cos(a0); y0 = cy + r_out * math.sin(a0)
            x1 = cx + r_out * math.cos(a1); y1 = cy + r_out * math.sin(a1)
            x2 = cx + r_in  * math.cos(a1); y2 = cy + r_in  * math.sin(a1)
            x3 = cx + r_in  * math.cos(a0); y3 = cy + r_in  * math.sin(a0)
            path = (f"M {x0:.1f},{y0:.1f} A {r_out},{r_out} 0 {large} 1 {x1:.1f},{y1:.1f} "
                    f"L {x2:.1f},{y2:.1f} A {r_in},{r_in} 0 {large} 0 {x3:.1f},{y3:.1f} Z")
            segs.append(
                f'<g class="donut-seg" data-cat="{cat}" data-n="{n}" '
                f'data-name="{esc(meta[0])}" data-pct="{frac*100:.0f}%" '
                f'style="--delay: {i*100}ms; color: {meta[1]}">'
                f'<path d="{path}" fill="{meta[1]}"/></g>'
            )
            legend_data.append((cat, meta[0], meta[1], n, meta[2]))
            a0 = a1

        svg = f'''
<svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg" id="donut" data-total="{total}" style="width:100%;height:auto;overflow:visible">
  <g class="donut-segs">
    {"".join(segs)}
  </g>
  <text id="donutNum" x="150" y="150" text-anchor="middle" dy="0.1em">{total}</text>
  <text id="donutLbl" x="150" y="176" text-anchor="middle">accepted opts</text>
</svg>'''

        if use_angle_mode and angle_counts:
            # Group angles under their tier; sort each group by count desc.
            grouped = defaultdict(list)
            for (num, name, tier), n in angle_counts.items():
                grouped[tier].append((num, name, n))
            for k in grouped:
                grouped[k].sort(key=lambda x: -x[2])
            tier_blocks = []
            tier_order = ["T1", "T2", "T3", "FLAG"]
            for tier in tier_order:
                if tier not in grouped:
                    continue
                tier_label, tier_color, tier_desc = TIER_META[tier]
                rows = "".join(f'''
<div class="leg-item leg-angle" data-cat="{tier}">
  <div class="leg-lab">
    <b><span class="angle-num">{esc(num)}</span> {esc(name)}</b>
    <p>{esc(ANGLE_TAXONOMY.get(num, (name, tier, ""))[2])}</p>
  </div>
  <div class="leg-n">{n}</div>
</div>''' for num, name, n in grouped[tier])
                tier_blocks.append(f'''
<div class="leg-tier-block">
  <div class="leg-tier-head" style="color:{tier_color}">
    <span class="swatch" style="background:{tier_color}"></span>
    <b>{esc(tier_label)}</b>
    <span class="leg-tier-desc">{esc(tier_desc)}</span>
  </div>
  {rows}
</div>''')
            legend_html = "".join(tier_blocks)
        else:
            legend_html = "".join(f'''
<div class="leg-item" data-cat="{cat}">
  <div class="swatch" style="background:{c}"></div>
  <div class="leg-lab"><b>{esc(name)}</b><p>{esc(d)}</p></div>
  <div class="leg-n">{n}</div>
</div>''' for cat, name, c, n, d in legend_data)

        return svg, legend_html

    speed_chart_sec = svg_speed_chart(tiny_progression, mode="sec")
    speed_chart_x = svg_speed_chart(tiny_progression, mode="speedup", baseline_sec=baseline_tiny)
    speed_chart = (
        f'<div class="speed-chart-stack" data-mode="sec">'
        f'<div class="speed-chart-pane" data-mode="sec">{speed_chart_sec}</div>'
        f'<div class="speed-chart-pane" data-mode="speedup" hidden>{speed_chart_x}</div>'
        f'</div>'
    )
    baseline_peak_tiny = baselines.get(_tiny_name, {}).get("peak", 0)
    mem_chart_abs = svg_memory_chart(tiny_progression, mode="abs")
    mem_chart_x = svg_memory_chart(tiny_progression, mode="ratio", baseline_mb=baseline_peak_tiny)
    mem_chart = (
        f'<div class="mem-chart-stack" data-mode="abs">'
        f'<div class="mem-chart-pane" data-mode="abs">{mem_chart_abs}</div>'
        f'<div class="mem-chart-pane" data-mode="ratio" hidden>{mem_chart_x}</div>'
        f'</div>'
    )
    tier_chart = render_tier_chart(TIER_ORDER)
    donut, legend_html = render_attribution_chart(
        attr_counts,
        angle_counts=angle_counts if use_angle_mode else None,
        use_angle_mode=use_angle_mode,
    )

    # build patch-stack accordion
    def render_patch(p):
        rec = match_opt(p)
        cat = categorize(p["desc"], p["desc"])
        meta = CAT_META.get(cat, CAT_META["other"])
        mech = rec["mech"] if rec else ""
        speed = rec["speed"] if rec else ""
        conc = rec["conc"] if rec else ""
        loc = rec["loc"] if rec else ""
        notes = rec["notes"] if rec else ""
        return f'''
    <details data-round="{p["round"]}">
      <summary>
        <span class="rd">round <b>{p["round"]}</b></span>
        <span class="title">{esc(p["desc"][:140])}</span>
        <span class="sha">{esc(p["commit"])}</span>
        <span class="pct">−{p["pct"]:.1f}%</span>
      </summary>
      <div class="body">
        <div><span class="chip" style="color:{meta[1]};border-color:{meta[1]}">{meta[0]}</span></div>
        <dl class="kv">
          {"<dt>Mechanism</dt><dd>"+esc(mech)+"</dd>" if mech else ""}
          {"<dt>Speedup</dt><dd>"+esc(speed)+"</dd>" if speed else ""}
          {"<dt>Concordance</dt><dd>"+esc(conc)+"</dd>" if conc else ""}
          {"<dt>Location</dt><dd><code>"+esc(loc)+"</code></dd>" if loc else ""}
          {"<dt>Notes</dt><dd>"+esc(notes)+"</dd>" if notes else ""}
        </dl>
      </div>
    </details>
    '''

    patch_html_parts = "".join(render_patch(p) for p in patch)

    # top-5 rich cards
    # Map round_angles entries by round number for top-5 category resolution.
    _angles_by_round = {a["round"]: a for a in round_angles} if round_angles else {}

    def render_top5(idx, d):
        rec = match_opt(d)
        # Prefer the agent's round_angles tier/angle classification when present
        # (LLM-assigned, far more accurate than my keyword categorize() guess).
        rd = d["round"]
        if rd in _angles_by_round:
            a = _angles_by_round[rd]
            if a.get("flag"):
                meta = (f"FLAG · {a.get('primary','')}", VAR["dim"], "")
            else:
                tier_key = a.get("tier") or "T1"
                tier_meta = TIER_META.get(tier_key, TIER_META.get("FLAG"))
                primary_num = a.get("primary_num", "")
                primary_name = a.get("primary_name", "")
                tier_label = (f"{tier_key} · {primary_num} {primary_name}".strip()
                              if primary_num or primary_name else tier_key)
                meta = (tier_label, tier_meta[1], tier_meta[2])
        else:
            cat = categorize(d["desc"], d["desc"])
            meta = CAT_META.get(cat, CAT_META["other"])
        mech = rec["mech"] if rec else "(see active_opts.md)"
        notes = rec["notes"] if rec else ""
        # Filter placeholder/empty notes so "Why this works: —" doesn't appear.
        if notes.strip() in ("—", "-", "n/a", "N/A", "(none)", "TBD", ""):
            notes = ""
        loc = rec["loc"] if rec else ""
        name = rec["name"] if rec else f"round {d['round']}"
        medium_line = ""
        if d.get("medium"):
            medium_line = f'<span class="metric"><b>medium</b> {d["medium"]:.2f}s</span>'
        return f'''
<div class="top-card">
  <div class="top-rank">{idx+1}</div>
  <div class="top-body">
    <div class="top-head">
      <span class="chip" style="color:{meta[1]};border-color:{meta[1]}">{meta[0]}</span>
      <code class="top-sha">round {d["round"]} · {esc(d["commit"])}</code>
    </div>
    <h4 class="top-name">{esc(name)}</h4>
    <div class="top-numbers">
      <span class="metric"><b>tiny</b> {d["tiny_prev"]:.2f}s → {d["tiny_speed"]:.2f}s</span>
      {medium_line}
      <span class="metric saved">−{d["delta_abs"]:.2f}s saved <em>({d["delta_pct"]:.0f}%)</em></span>
    </div>
    <p class="top-mech">{esc(mech)}</p>
    {f'<p class="top-notes"><b>Why this works:</b> {esc(notes)}</p>' if notes else ""}
    {f'<div class="top-loc">📍 <code>{esc(loc)}</code></div>' if loc else ""}
  </div>
</div>
'''
    top5_html = "".join(render_top5(i, d) for i, d in enumerate(top5))

    # legend_html was rendered above by render_attribution_chart

    # tier table
    metrics_rows = []
    for ds, label, full in TIER_ORDER:
        b = baselines.get(ds); f = finals.get(ds)
        if not b or not f: continue
        speedup = b["speed"] / f["speed"] if f["speed"] else 0
        mem_pct = (b["peak"] - f["peak"]) / b["peak"] * 100 if b["peak"] else 0
        metrics_rows.append(f'''
    <tr>
      <td class="tier"><b>{esc(label)}</b></td>
      <td class="full">{esc(full)}</td>
      <td class="num dim">{b["speed"]:.1f}s</td>
      <td class="num">{f["speed"]:.2f}s</td>
      <td class="num gold">{speedup:.1f}×</td>
      <td class="num">{f["peak"]:.0f} MB</td>
      <td class="num jade">{('+' if mem_pct < 0 else '−')}{abs(mem_pct):.0f}%</td>
    </tr>''')

    # dead ends
    de_html = "".join(f'''
<div class="card de-card">
  <div class="head">
    <span class="chip rose">FALSIFIED</span>
    <h4>{esc(d["title"])}</h4>
  </div>
  {"<div class='tried'>"+esc(d["tried"])+"</div>" if d["tried"] else ""}
  <div class="why"><b>Why:</b> {esc(d["why"])}</div>
  {"<div class='verdict'>"+esc(d["verdict"])+"</div>" if d["verdict"] else ""}
</div>''' for d in de_cards)

    # discoveries: pick "most unexpected" 3 via title-keyword heuristic, hide the rest
    def _unexpected_score(card):
        t = card["title"].lower()
        # rank truly counter-intuitive findings (one per category to ensure diversity)
        if "loses to" in t:                       return 100  # Rcpp vs R surprise
        if "hidden invariant" in t:               return 95   # cbind memory layout
        if "blocking" in t and "reclaim" in t:    return 90   # cat() keeps inf alive
        if "lived through" in t:                  return 88   # seurat object ghost
        if "lingers" in t:                        return 86   # mcmc.list persistence
        if "1 sample" in t and "stable" in t:     return 80   # n.iter=1 dramatic
        if "always writes" in t:                  return 75   # hidden side-effect
        if "structurally caps" in t:              return 72   # threading limit
        if "tolerate" in t:                       return 60
        if "still stable" in t:                   return 40
        return 1

    # rank by score then preserve original order for ties
    disc_ranked = sorted(enumerate(disc_cards), key=lambda x: (-_unexpected_score(x[1]), x[0]))
    disc_top_ids = {i for i, _ in disc_ranked[:3]}
    disc_top_cards = [disc_cards[i] for i in sorted(disc_top_ids)]
    disc_rest = [c for i, c in enumerate(disc_cards) if i not in disc_top_ids][:18]

    def _render_disc(d):
        # round chips: link to the matching patch-stack details
        chips = ""
        if d.get("rounds"):
            chip_html = []
            for rn in d["rounds"][:3]:  # cap at 3
                chip_html.append(f'<a class="disc-round-chip" href="#stack" data-round="{rn}">round {rn}</a>')
            chips = '<div class="disc-rounds">' + "".join(chip_html) + '</div>'
        return f'''
<div class="card disc-card">
  {chips}
  <h4>{esc(d["title"])}</h4>
  <div class="cause">{esc(d["cause"])}</div>
  {"<div class='impl'>"+esc(d["impl"])+"</div>" if d["impl"] else ""}
</div>'''

    disc_html_top = "".join(_render_disc(d) for d in disc_top_cards)
    disc_html_rest = "".join(_render_disc(d) for d in disc_rest)
    if disc_rest:
        disc_collapsible = f'''
<details class="section-collapse" style="margin-top:24px">
  <summary>
    <span>Show all discoveries</span>
    <span class="count">+{len(disc_rest)} more</span>
    <span class="chev">▾</span>
  </summary>
  <div class="collapse-body grid-2" style="margin-top:18px">{disc_html_rest}</div>
</details>'''
    else:
        disc_collapsible = ""

    # concordance chips
    conc_chip = ""
    if "brca_wu" in latest_metrics:
        m = latest_metrics["brca_wu"][1]
        chips = []
        for k in ["pearson_expr", "hmm_state_agreement", "hmm_amp_del_concord", "gene_coverage"]:
            if k in m:
                v = m[k]
                cls = "jade" if v >= 0.99 else "rose"
                chips.append(f'<span class="chip {cls}">{esc(k)} = {v:.2f}</span>')
        conc_chip = "<div class='conc-chips'>" + "".join(chips) + "</div>"

    n_keeps, n_discards, n_crashes = len(patch), len(discard_rounds), len(crash_rounds)

    tldr = (
        f"<b>{hero_speedup:.0f}× faster</b> on production-scale BRCA Wu (10X, ~50k cells): "
        f"<b>{hero_baseline/60:.1f} min → {hero_final:.0f} s</b>, peak memory "
        f"<b>{hero_mem_b/1024:.1f} GB → {hero_mem_f/1024:.1f} GB</b>. "
        f"All concordance metrics 1.0 across 5 tiers (dev + 2 OOD cancer types). "
        f"{n_keeps} accepted optimizations, {n_discards} falsified angles, bit-exact HMM state matrices."
    )

    # ============================================================
    # CSS — light default, dark via [data-theme="dark"]
    # ============================================================
    CSS = """
:root {
  /* Light — matches autozyme.dev: warm off-white paper + deep teal accent */
  --bg: #fdfdfb;
  --surface: #ffffff;
  --elev: #f5f4ee;
  --soft: #f9f8f3;
  --border: #e7e4d9;
  --border-strong: #c9c4b3;
  --hair: #efede6;
  --text: #141419;
  --dim: #4b4b54;
  --muted: #9a9a9f;
  --accent: #0c4a6e;     /* deep teal — links, info */
  --accent-soft: rgba(12, 74, 110, 0.08);
  --gold: #a16207;       /* warm amber — hero accent, "accept" */
  --jade: #166534;       /* deep green — speedups, kept rounds */
  --rose: #9f1239;       /* crimson — discards, dead ends */
  --slate: #0c4a6e;      /* alias of accent */
  --amber: #a16207;
  --olive: #4d7c0f;      /* distinct from gold + jade — vectorize/fuse */
  --ink: #f3f1ea;        /* code background */
  --ink_text: #0c4a6e;   /* code text */
  --shadow: 0 1px 2px rgba(20,20,25,0.04), 0 4px 18px rgba(20,20,25,0.06);
}
:root[data-theme="dark"] {
  /* Dark — matches autozyme.dev night */
  --bg: #0a0a0c;
  --surface: #131318;
  --elev: #101014;
  --soft: #0c0c10;
  --border: #23232a;
  --border-strong: #3a3a43;
  --hair: #18181d;
  --text: #f3f3f2;
  --dim: #a6a6ad;
  --muted: #6e6e78;
  --accent: #7dd3fc;
  --accent-soft: rgba(125,211,252,0.12);
  --gold: #fcd34d;
  --jade: #86efac;
  --rose: #fb7185;
  --slate: #7dd3fc;
  --amber: #fcd34d;
  --olive: #bef264;      /* distinct from gold + jade — vectorize/fuse */
  --ink: #09090c;
  --ink_text: #7dd3fc;
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 30px rgba(0,0,0,0.4);
}

*,*::before,*::after { box-sizing: border-box; }
html { scroll-behavior: smooth; }
html, body { margin: 0; padding: 0; }
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
}
body {
  background: var(--bg); color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, 'Helvetica Neue', sans-serif;
  font-size: 15px; line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  transition: background 0.3s ease, color 0.3s ease;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 0 36px; }
a { color: var(--gold); text-decoration: none; }
a:hover { text-decoration: underline; }
code { background: var(--ink); color: var(--ink_text); padding: 1px 6px; border-radius: 3px; font-size: 0.88em; font-family: 'JetBrains Mono', ui-monospace, monospace; }

/* nav */
nav.top {
  position: sticky; top: 0; z-index: 50;
  background: color-mix(in oklab, var(--bg) 88%, transparent);
  backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--hair);
  padding: 14px 0; font-size: 13px;
}
nav.top .wrap { display: flex; align-items: center; gap: 26px; }
nav.top .brand { display: inline-flex; align-items: center; padding: 2px 0; margin-right: 4px; }
nav.top .brand-logo { height: 22px; width: auto; display: block; opacity: 0.92; transition: opacity 0.2s ease, filter 0.3s ease; }
nav.top .brand:hover .brand-logo { opacity: 1; }
:root[data-theme="dark"] .brand-logo { filter: invert(1) brightness(1.05); }
nav.top a { color: var(--dim); transition: color 0.15s ease; position: relative; }
nav.top a:hover { color: var(--text); text-decoration: none; }
nav.top a.is-current { color: var(--text); font-weight: 500; }
nav.top a.is-current::after {
  content: ""; position: absolute; left: 0; right: 0; bottom: -17px;
  height: 2px; background: var(--gold); border-radius: 2px;
  animation: slideIn 0.3s cubic-bezier(0.22,1,0.36,1);
}
@keyframes slideIn {
  from { transform: scaleX(0); opacity: 0; }
  to   { transform: scaleX(1); opacity: 1; }
}
nav.top .spacer { flex: 1; }

.theme-toggle {
  appearance: none; background: var(--surface); color: var(--dim);
  border: 1px solid var(--border); border-radius: 100px; padding: 6px 14px;
  font: inherit; font-size: 12px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
  transition: all 0.2s ease;
}
.theme-toggle:hover { color: var(--gold); border-color: var(--gold); }

/* hero */
.hero {
  padding: 96px 0 72px; border-bottom: 1px solid var(--hair);
  position: relative; overflow: hidden;
}
.hero::before {
  content: ""; position: absolute; top: -200px; right: -200px; width: 600px; height: 600px;
  background: radial-gradient(circle, var(--accent-soft) 0%, transparent 60%);
  pointer-events: none; z-index: 0;
}
.hero > .wrap { position: relative; z-index: 1; }
.hero .eyebrow {
  color: var(--gold); text-transform: uppercase;
  letter-spacing: 2.8px; font-size: 10.5px; font-weight: 700;
  margin-bottom: 24px;
  display: inline-flex; align-items: center; gap: 14px;
  font-family: 'Inter', system-ui, sans-serif;
}
.hero .eyebrow .rule { width: 32px; height: 1px; background: var(--gold); display: inline-block; opacity: 0.6; }
.dot-sep { color: var(--muted); margin: 0 4px; font-weight: 400; }

/* target card — pulled out as the WHAT */
.target-card {
  margin: 44px 0 8px;
  background: var(--soft);
  border: 1px solid var(--border);
  border-left: 3px solid var(--gold);
  border-radius: 10px;
  padding: 26px 32px 22px;
  max-width: 760px;
  position: relative;
  box-shadow: var(--shadow);
}
.target-label {
  position: absolute; top: -9px; left: 26px;
  background: var(--bg);
  color: var(--gold);
  font-size: 10px; font-weight: 700;
  letter-spacing: 2.6px; text-transform: uppercase;
  padding: 0 10px;
  font-family: 'Inter', system-ui, sans-serif;
}
.target-code {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 15.5px; line-height: 1.7; color: var(--text);
  margin: 8px 0 14px; white-space: pre; overflow-x: auto;
  font-weight: 500;
}
.target-meta {
  font-size: 12px; color: var(--dim);
  font-family: 'Inter', system-ui, sans-serif;
  letter-spacing: 0.4px;
  padding-top: 10px; border-top: 1px solid var(--hair);
}

/* R syntax tokens */
.tk-ns { color: var(--accent); font-weight: 600; }
.tk-fn { color: var(--gold); font-weight: 600; }
.tk-op { color: var(--muted); }
.tk-arg { color: var(--text); }
.tk-val { color: var(--jade); font-weight: 600; }
.tk-str { color: var(--rose); }
.hero h1 {
  font-family: 'Fraunces', 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
  font-weight: 400; font-size: 54px; line-height: 1.3; margin: 0 0 22px;
  letter-spacing: -0.4px;
  font-variation-settings: 'opsz' 96;
}
.hero h1 em {
  display: inline-block;
  color: var(--gold);
  font-style: italic;
  font-weight: 500;
  font-size: 1.65em;          /* much bigger than surrounding text */
  line-height: 1;
  letter-spacing: -3px;
  margin: 0 0.06em -0.05em;
  font-variation-settings: 'opsz' 144;
}
.hero .sub { color: var(--dim); font-size: 17px; max-width: 780px; }
.hero .tldr {
  margin-top: 26px; color: var(--text); font-size: 15px; line-height: 1.7;
  border-left: 2px solid var(--gold); padding-left: 20px; max-width: 800px;
}
.hero .conc-chips { margin-top: 22px; display: flex; gap: 8px; flex-wrap: wrap; }

.hero-stats {
  display: grid; grid-template-columns: 1.5fr 1fr 1fr 1fr;
  gap: 36px; margin-top: 56px;
  padding-top: 36px; border-top: 1px solid var(--hair);
}
.stat { position: relative; }
.stat + .stat::before {
  content: ""; position: absolute; left: -18px; top: 4px; bottom: 4px;
  width: 1px; background: var(--hair);
}
.stat .lbl { color: var(--muted); font-size: 10.5px; text-transform: uppercase; letter-spacing: 1.8px; margin-bottom: 10px; font-weight: 600; }
.stat .val { font-family: 'Fraunces', 'Iowan Old Style', 'Source Serif Pro', Georgia, serif; font-size: 46px; font-weight: 500; line-height: 1.0; letter-spacing: -1px; font-variation-settings: 'opsz' 96; }
.stat .val.jade { color: var(--jade); }
.stat .val.gold { color: var(--gold); }
.stat .val.rose { color: var(--rose); }
.stat .val.dim  { color: var(--muted); }
.stat .ctx { color: var(--dim); font-size: 11.5px; margin-top: 10px; font-family: 'JetBrains Mono', ui-monospace, monospace; letter-spacing: 0.2px; }

/* section */
section { padding: 88px 0 24px; border-top: 1px solid var(--hair); }
section:first-of-type { border-top: 0; }
section h2 {
  font-family: 'Fraunces', 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
  font-weight: 400; font-size: 40px; margin: 0 0 10px;
  letter-spacing: -0.4px;
  font-variation-settings: 'opsz' 72;
}
section h2::before {
  content: ""; display: block; width: 32px; height: 1px;
  background: var(--gold); opacity: 0.5; margin-bottom: 22px;
}
section .lead { color: var(--dim); max-width: 760px; font-size: 15px; margin-bottom: 40px; line-height: 1.7; }
section h3 { font-size: 11.5px; text-transform: uppercase; letter-spacing: 2.4px; color: var(--gold); font-weight: 700; margin: 44px 0 16px; }

.card {
  background: var(--surface);
  border: 1px solid var(--hair); border-radius: 8px;
  padding: 26px 28px; margin-bottom: 16px;
  box-shadow: var(--shadow);
}
.card.elev { background: var(--elev); }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
.grid-attr { display: grid; grid-template-columns: 320px 1fr; gap: 40px; align-items: center; }

.chip {
  display: inline-block; padding: 3px 11px; border-radius: 100px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.4px;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  border: 1px solid; background: transparent;
}
.chip.jade { color: var(--jade); border-color: var(--jade); }
.chip.rose { color: var(--rose); border-color: var(--rose); }
.chip.gold { color: var(--gold); border-color: var(--gold); }
.chip.dim  { color: var(--dim);  border-color: var(--border); }
.chip.slate { color: var(--slate); border-color: var(--slate); }

/* ===== section-level collapsible toggle ===== */
.section-collapse { margin-top: 8px; }
.section-collapse > summary {
  list-style: none; cursor: pointer; user-select: none;
  display: inline-flex; align-items: center; gap: 12px;
  padding: 11px 22px;
  background: var(--soft);
  border: 1px solid var(--border);
  border-radius: 100px;
  color: var(--text);
  font-size: 13px; font-weight: 500;
  font-family: 'Inter', system-ui, sans-serif;
  transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease, box-shadow 0.2s ease;
}
.section-collapse > summary::-webkit-details-marker { display: none; }
.section-collapse > summary:hover {
  background: var(--accent-soft);
  border-color: var(--accent);
  color: var(--accent);
}
.section-collapse .count {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  color: var(--muted); font-size: 11px;
  padding: 2px 9px; border-radius: 100px;
  background: var(--bg); border: 1px solid var(--hair);
  transition: color 0.2s ease, border-color 0.2s ease;
}
.section-collapse > summary:hover .count { color: var(--accent); border-color: var(--accent); }
.section-collapse .chev {
  display: inline-block; transition: transform 0.3s cubic-bezier(0.22,1,0.36,1);
  font-size: 11px; color: var(--dim);
}
.section-collapse[open] > summary .chev { transform: rotate(180deg); color: var(--accent); }
.section-collapse[open] > summary { margin-bottom: 22px; }
.section-collapse > .collapse-body {
  animation: collapseFade 0.4s ease;
}
@keyframes collapseFade {
  from { opacity: 0; transform: translateY(-4px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ===== TOP 5 DEEP DIVE ===== */
.top5-grid { display: flex; flex-direction: column; gap: 18px; }
.top-card {
  display: grid; grid-template-columns: 96px 1fr; gap: 24px;
  background: var(--surface); border: 1px solid var(--hair);
  border-radius: 10px; padding: 26px 30px;
  box-shadow: var(--shadow);
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.top-card:hover {
  transform: translateY(-1px);
  border-color: var(--gold);
  box-shadow: 0 4px 24px color-mix(in oklab, var(--gold) 18%, transparent);
}
.top-rank {
  font-family: 'Fraunces', 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
  font-size: 78px; font-weight: 500; line-height: 1; color: var(--gold);
  opacity: 0.85;
}
.top-head { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
.top-sha { color: var(--dim); font-size: 11px; font-family: 'JetBrains Mono', ui-monospace, monospace; background: transparent; padding: 0; }
.top-name {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 15px; font-weight: 600; margin: 4px 0 14px; color: var(--text);
}
.top-numbers { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 16px; padding: 12px 16px; background: var(--elev); border-radius: 6px; }
.top-numbers .metric { font-size: 13px; color: var(--dim); font-family: 'JetBrains Mono', ui-monospace, monospace; }
.top-numbers .metric b { color: var(--gold); font-weight: 600; margin-right: 4px; }
.top-numbers .metric.saved { color: var(--jade); font-weight: 600; margin-left: auto; }
.top-numbers .metric.saved em { font-style: normal; opacity: 0.7; }
.top-mech { font-size: 14px; line-height: 1.65; color: var(--text); margin: 0 0 12px; }
.top-notes { font-size: 13px; line-height: 1.65; color: var(--dim); margin: 0 0 12px; padding-left: 14px; border-left: 2px solid var(--hair); }
.top-notes b { color: var(--gold); }
.top-loc { font-size: 12px; color: var(--dim); }

/* ===== iteration chart ===== */
#chartWrap { position: relative; }
#chartWrap .chart-label {
  display: flex; align-items: baseline; gap: 12px;
  font-size: 11px; font-weight: 600; letter-spacing: 1.8px;
  text-transform: uppercase; color: var(--text);
  padding: 4px 8px 12px;
}
#chartWrap .chart-label-dim { color: var(--muted); font-weight: 500; letter-spacing: 0.6px; }
#chartWrap .chart-toggle {
  margin-left: auto; display: inline-flex; gap: 0;
  border: 1px solid var(--hair); border-radius: 999px; padding: 2px;
  background: var(--surface-soft, transparent);
}
#chartWrap .chart-toggle-btn {
  appearance: none; -webkit-appearance: none; background: transparent;
  border: 0; color: var(--muted); cursor: pointer;
  font: inherit; font-size: 10px; font-weight: 600; letter-spacing: 1px;
  text-transform: uppercase; padding: 4px 10px; border-radius: 999px;
  transition: background 0.18s ease, color 0.18s ease;
}
#chartWrap .chart-toggle-btn:hover { color: var(--text); }
#chartWrap .chart-toggle-btn.is-active {
  background: var(--jade); color: var(--surface);
}
.speed-chart-pane[hidden], .mem-chart-pane[hidden] { display: none; }
#chartWrap .chart-divider {
  height: 1px; background: var(--hair);
  margin: 14px 0 20px;
}
.speed-svg .dot, .memSvg .dot {
  transition: opacity 0.3s ease var(--dot-d, 0ms), r 0.18s ease-out;
  cursor: pointer;
}
/* ===== chart entrance animation =====
   Line draws left→right via stroke-dashoffset; area fades after the line
   reaches its end; dots pop in staggered. Pre-animation, dots are hidden
   via :not(.animated) (high enough specificity to override .dot-ds opacity). */
.speed-svg .chart-line, .memSvg .chart-line {
  stroke-dasharray: var(--line-len, 3000);
  stroke-dashoffset: var(--line-len, 3000);
  transition: stroke-dashoffset 1.4s cubic-bezier(0.22, 1, 0.36, 1);
}
.speed-svg.animated .chart-line, .memSvg.animated .chart-line { stroke-dashoffset: 0; }
.speed-svg .chart-area, .memSvg .chart-area {
  opacity: 0;
  transition: opacity 0.6s ease-out 0.7s;
}
.speed-svg.animated .chart-area, .memSvg.animated .chart-area { opacity: 1; }
.speed-svg:not(.animated) .dot, .memSvg:not(.animated) .dot { opacity: 0; }
.speed-svg .dot-kp, .memSvg .dot-kp { fill: var(--jade); stroke: var(--surface); stroke-width: 1.5; }
/* Discarded + crashed rounds are explored-but-rejected states. Draw them faded
   and slightly smaller so the eye reads the accepted trajectory, not the noise. */
.speed-svg .dot-ds, .memSvg .dot-ds {
  fill: var(--rose); stroke: var(--surface); stroke-width: 1; opacity: 0.35;
  r: 3;
}
.speed-svg .dot-cr, .memSvg .dot-cr {
  fill: var(--amber); stroke: var(--surface); stroke-width: 1; opacity: 0.35;
  r: 3;
}
.speed-svg .dot-bl, .memSvg .dot-bl { fill: var(--gold); stroke: var(--surface); stroke-width: 1.5; }
/* On hover, restore full presence so the user can still inspect them. */
.speed-svg .dot-ds:hover, .speed-svg .dot-cr:hover,
.memSvg .dot-ds:hover, .memSvg .dot-cr:hover { opacity: 1; }
.speed-svg .dot.is-hot, .memSvg .dot.is-hot { stroke-width: 2; opacity: 1; }
.speed-svg .dot.is-hot.dot-kp, .memSvg .dot.is-hot.dot-kp { r: 8 !important; filter: drop-shadow(0 0 6px var(--jade)); }
.speed-svg .dot.is-hot.dot-ds, .memSvg .dot.is-hot.dot-ds { r: 7 !important; filter: drop-shadow(0 0 5px var(--rose)); }
.speed-svg .dot.is-hot.dot-cr, .memSvg .dot.is-hot.dot-cr { r: 7 !important; filter: drop-shadow(0 0 5px var(--amber)); }
.speed-svg .dot.is-hot.dot-bl, .memSvg .dot.is-hot.dot-bl { r: 8 !important; filter: drop-shadow(0 0 6px var(--gold)); }

#chartTip {
  position: absolute; pointer-events: none; opacity: 0;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px; min-width: 280px; max-width: 380px;
  box-shadow: 0 10px 40px color-mix(in oklab, var(--text) 18%, transparent);
  transition: opacity 0.15s ease, transform 0.15s ease;
  transform: translateY(4px); z-index: 5;
  font-size: 13px; line-height: 1.55;
}
#chartTip.show { opacity: 1; transform: translateY(0); }
#chartTip .row1 { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px; }
#chartTip .tip-rd { color: var(--text); font-weight: 600; font-size: 14px; }
#chartTip .tip-sha { color: var(--dim); font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px; }
#chartTip .tip-spd { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 15px; color: var(--jade); font-weight: 600; }
#chartTip .tip-mem { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px; color: var(--dim); margin-bottom: 8px; }
#chartTip .tip-desc { color: var(--text); font-size: 12.5px; line-height: 1.55; }
#chartTip.tip-keep { border-top: 3px solid var(--jade); }
#chartTip.tip-discard { border-top: 3px solid var(--rose); }
#chartTip.tip-crash { border-top: 3px solid var(--amber); }
#chartTip.tip-baseline { border-top: 3px solid var(--gold); }

.legend-row {
  display: flex; gap: 26px; margin-top: 14px; font-size: 12px;
  color: var(--dim); font-family: 'JetBrains Mono', ui-monospace, monospace;
}
.legend-row .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }

/* ===== tier chart (HTML, interactive) ===== */
.tier-chart { padding: 12px 4px 0; }
.tier-list { display: flex; flex-direction: column; gap: 2px; }
.tier-row {
  display: grid;
  grid-template-columns: 220px 1fr 130px;
  grid-template-rows: auto auto;
  gap: 4px 28px;
  align-items: center;
  padding: 18px 18px;
  border-radius: 12px;
  border: 1px solid transparent;
  transition: background 0.3s ease, border-color 0.3s ease;
}
.tier-row:hover {
  background: var(--soft);
  border-color: var(--hair);
}
.tier-info { grid-row: 1; grid-column: 1; }
.tier-trackwrap { grid-row: 1; grid-column: 2; position: relative; }
.tier-result { grid-row: 1; grid-column: 3; text-align: right; }
.tier-extras { grid-row: 2; grid-column: 1 / -1; }

.tier-name { font-weight: 600; font-size: 14.5px; color: var(--text); letter-spacing: -0.1px; }
.tier-desc { font-size: 11.5px; color: var(--dim); margin-top: 3px; }

.tier-track {
  position: relative;
  height: 26px;
  background: var(--bg);
  border-radius: 100px;
  border: 1px solid var(--hair);
  overflow: visible;
}
.tier-bar-base, .tier-bar-fast {
  position: absolute; left: 0; top: 0; bottom: 0;
  border-radius: 100px;
  width: 0;
  transition: width 1.1s cubic-bezier(0.22,1,0.36,1);
}
.tier-bar-base {
  background: color-mix(in srgb, var(--border) 80%, var(--bg) 20%);
  transition-delay: var(--delay, 0ms);
}
.tier-bar-fast {
  background: linear-gradient(90deg, var(--jade) 0%, color-mix(in srgb, var(--jade) 65%, var(--gold) 35%) 100%);
  transition-delay: calc(var(--delay, 0ms) + 280ms);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.18), 0 1px 2px color-mix(in srgb, var(--jade) 22%, transparent);
}
#tierChart.animated .tier-bar-base { width: var(--w); }
#tierChart.animated .tier-bar-fast { width: var(--w); }
.tier-row:hover .tier-bar-fast { filter: brightness(1.06) saturate(1.15); }
.tier-row:hover .tier-bar-base { background: var(--border); }

.tier-tag {
  position: absolute; top: 50%; transform: translate(8px, -50%);
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 11px; white-space: nowrap;
  opacity: 0;
  transition: opacity 0.4s ease, transform 0.4s ease;
  transition-delay: calc(var(--delay, 0ms) + 1100ms);
  pointer-events: none; line-height: 1;
}
.tier-tag-fast { color: var(--jade); font-weight: 600; }
.tier-tag-base { color: var(--dim); }
#tierChart.animated .tier-tag { opacity: 1; transform: translate(8px, -50%); }

.tier-speedup {
  font-family: 'Fraunces', 'Iowan Old Style', Georgia, serif;
  font-size: 38px; font-weight: 500;
  color: var(--gold); line-height: 1;
  letter-spacing: -1px;
  font-variation-settings: 'opsz' 96;
  font-feature-settings: 'tnum';
  display: inline-flex; align-items: baseline; gap: 0;
}
.tier-speedup .num { display: inline-block; min-width: 1.5em; text-align: right; }
.tier-speedup .x {
  font-size: 0.78em; font-weight: 400;
  opacity: 0.95; margin-left: 4px;
  font-style: italic;
  color: color-mix(in srgb, var(--gold) 90%, var(--text));
}
.tier-mem {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 11px; color: var(--text); margin-top: 8px;
}
.tier-mem .dim { color: var(--muted); }

.tier-extras {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 10.5px; color: var(--muted);
  padding-left: 248px;     /* aligns with track */
  max-height: 0; opacity: 0; overflow: hidden;
  transition: max-height 0.35s ease, opacity 0.25s ease, padding-top 0.25s ease;
}
.tier-extras .ext-sep { margin: 0 6px; color: var(--border); }
.tier-extras .ext-commit { color: var(--accent); }
.tier-extras .ext-ok { color: var(--jade); font-weight: 600; }
.tier-row:hover .tier-extras { max-height: 28px; opacity: 1; padding-top: 4px; }

.tier-axis-row {
  display: grid;
  grid-template-columns: 220px 1fr 130px;
  gap: 28px;
  padding: 0 18px;
  margin-top: 14px;
}
.tier-axis-track {
  position: relative;
  height: 18px;
  border-top: 1px dashed var(--hair);
}
.tier-tick {
  position: absolute; top: 0;
  transform: translateX(-50%);
}
.tier-tick::before {
  content: ""; display: block; width: 1px; height: 4px;
  background: var(--border);
}
.tier-tick span {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 10.5px; color: var(--muted);
  display: block; margin-top: 4px;
}

/* ===== patch stack ===== */
.patch-stack details {
  background: var(--surface); border: 1px solid var(--hair);
  border-radius: 8px; margin-bottom: 8px; overflow: hidden;
  box-shadow: var(--shadow);
}
.patch-stack details[open] { background: var(--elev); border-color: var(--border); }
.patch-stack summary {
  list-style: none; cursor: pointer; padding: 16px 22px;
  display: grid; grid-template-columns: 80px 1fr 90px 100px; gap: 16px; align-items: center;
}
.patch-stack summary::-webkit-details-marker { display: none; }
.patch-stack .rd { color: var(--dim); font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 12px; }
.patch-stack .rd b { color: var(--text); font-weight: 600; }
.patch-stack .title { color: var(--text); font-size: 14px; font-weight: 500; }
.patch-stack .sha { color: var(--dim); font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 11px; text-align: right; }
.patch-stack .pct { color: var(--jade); font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 13px; font-weight: 600; text-align: right; }
.patch-stack .body { padding: 18px 22px 22px; border-top: 1px solid var(--hair); color: var(--dim); font-size: 14px; }
.patch-stack .body p { margin: 0 0 10px; }
.patch-stack .body .kv { display: grid; grid-template-columns: 110px 1fr; gap: 8px 16px; margin-top: 12px; font-size: 13px; }
.patch-stack .body .kv dt { color: var(--gold); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; padding-top: 2px; }
.patch-stack .body .kv dd { margin: 0; color: var(--text); }

/* ===== donut + linked legend (interactive) ===== */
#donut { display: block; }
#donut .donut-seg {
  opacity: 0;
  transform: scale(0.55);
  transform-origin: 150px 150px;
  transform-box: fill-box;
  transition: opacity 0.6s cubic-bezier(0.22,1,0.36,1),
              transform 0.7s cubic-bezier(0.22,1,0.36,1);
  transition-delay: var(--delay, 0ms);
}
#donut.animated .donut-seg { opacity: 1; transform: scale(1); }
#donut .donut-seg path {
  transition: transform 0.28s cubic-bezier(0.22,1,0.36,1),
              filter 0.25s ease,
              opacity 0.25s ease;
  transform-origin: 150px 150px;
  transform-box: fill-box;
  cursor: pointer;
}
#donut .donut-seg:hover path,
#donut .donut-seg.is-active path {
  transform: scale(1.045);
  filter: brightness(1.06) drop-shadow(0 3px 10px color-mix(in srgb, currentColor 35%, transparent));
}
#donut.has-active .donut-seg:not(.is-active) path { opacity: 0.28; filter: saturate(0.7); }
#donutNum {
  font-family: 'Fraunces', 'Iowan Old Style', Georgia, serif;
  font-size: 44px; font-weight: 500; fill: var(--text);
  letter-spacing: -0.5px;
  font-variation-settings: 'opsz' 96;
  transition: fill 0.2s ease;
}
#donut.has-active #donutNum { fill: var(--gold); }
#donutLbl {
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 10.5px; font-weight: 600;
  fill: var(--dim);
  letter-spacing: 2px;
  text-transform: uppercase;
  transition: fill 0.2s ease;
}

.legend { display: flex; flex-direction: column; gap: 4px; }
.leg-item {
  display: grid; grid-template-columns: 24px 1fr 44px; gap: 14px; align-items: start;
  padding: 8px 12px; border-radius: 8px;
  transition: background 0.2s ease, opacity 0.2s ease;
  cursor: pointer;
}
.leg-item:hover, .leg-item.is-active { background: var(--soft); }
.legend.has-active .leg-item:not(.is-active) { opacity: 0.42; }
.swatch {
  width: 14px; height: 14px; border-radius: 3px; margin-top: 4px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.leg-item:hover .swatch, .leg-item.is-active .swatch { transform: scale(1.15); box-shadow: 0 0 0 3px color-mix(in srgb, currentColor 15%, transparent); }
.leg-lab b { color: var(--text); font-weight: 600; font-size: 14px; }
.leg-lab p { margin: 4px 0 0; font-size: 12.5px; color: var(--dim); line-height: 1.5; }
.leg-n {
  color: var(--gold);
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-weight: 600; text-align: right; font-size: 15px;
  padding-top: 1px;
}
/* angle-mode legend: tier groups with header rows */
.leg-tier-block { margin-bottom: 14px; }
.leg-tier-block:last-child { margin-bottom: 0; }
.leg-tier-head {
  display: flex; align-items: center; gap: 10px;
  font-size: 11px; font-weight: 600; letter-spacing: 1.4px;
  text-transform: uppercase;
  padding: 6px 12px 4px;
  border-bottom: 1px solid var(--hair);
  margin-bottom: 4px;
}
.leg-tier-head .swatch { width: 12px; height: 12px; margin-top: 0; }
.leg-tier-head .leg-tier-desc {
  margin-left: 8px; color: var(--dim); font-weight: 500;
  letter-spacing: 0.6px; text-transform: none; font-size: 11.5px;
}
.leg-angle { grid-template-columns: 1fr 44px; gap: 10px; padding-left: 30px; }
.leg-angle .angle-num {
  display: inline-block; min-width: 1.6em;
  color: var(--dim); font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-weight: 600; font-size: 12px; margin-right: 4px;
}

/* metrics table */
table.metrics { width: 100%; border-collapse: collapse; font-size: 13px; }
table.metrics th, table.metrics td { text-align: left; padding: 12px 16px; border-bottom: 1px solid var(--hair); }
table.metrics tr:last-child td { border-bottom: 0; }
table.metrics th { color: var(--dim); font-weight: 500; font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; }
table.metrics td.num { font-family: 'JetBrains Mono', ui-monospace, monospace; text-align: right; }
table.metrics td.gold { color: var(--gold); font-weight: 600; }
table.metrics td.jade { color: var(--jade); }
table.metrics td.dim { color: var(--dim); }
table.metrics td.tier b { color: var(--text); }
table.metrics td.full { color: var(--dim); font-size: 12px; }

/* dead end */
.de-card { border-left: 3px solid var(--rose); }
.de-card .head { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.de-card h4 { margin: 0; font-size: 16px; color: var(--text); font-weight: 500; }
.de-card .tried { color: var(--dim); font-size: 11px; font-family: 'JetBrains Mono', ui-monospace, monospace; margin-bottom: 10px; }
.de-card .why { color: var(--text); font-size: 14px; line-height: 1.65; margin: 8px 0; }
.de-card .why b { color: var(--rose); }
.de-card .verdict { font-size: 13px; color: var(--dim); padding-top: 12px; margin-top: 12px; border-top: 1px solid var(--hair); line-height: 1.55; }
.de-card .verdict::before { content: "WHAT WOULD CHANGE THE VERDICT  "; color: var(--gold); font-size: 9.5px; letter-spacing: 1.5px; font-weight: 700; }

/* discovery */
.disc-card { border-left: 3px solid var(--slate); transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease; }
.disc-card:hover { transform: translateY(-1px); border-left-color: var(--accent); }
.disc-card h4 { margin: 0 0 10px; font-size: 15px; color: var(--text); font-weight: 500; line-height: 1.4; }
.disc-card .cause { color: var(--text); font-size: 13.5px; line-height: 1.55; }
.disc-card .impl { color: var(--dim); font-size: 12.5px; padding-top: 10px; margin-top: 10px; border-top: 1px solid var(--hair); line-height: 1.55; }
.disc-card .impl::before { content: "↳ "; color: var(--jade); font-weight: 600; }

/* round chip on a discovery card → links into patch stack */
.disc-rounds { margin-bottom: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
.disc-round-chip {
  display: inline-flex; align-items: center;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 10.5px; font-weight: 600;
  padding: 3px 10px; border-radius: 100px;
  background: var(--accent-soft); color: var(--accent);
  border: 1px solid color-mix(in srgb, var(--accent) 25%, transparent);
  text-decoration: none; letter-spacing: 0.3px;
  transition: background 0.15s ease, color 0.15s ease, transform 0.15s ease;
}
.disc-round-chip:hover {
  background: var(--accent); color: var(--bg);
  transform: translateY(-1px);
  text-decoration: none;
}

/* flash animation when a patch-stack details is targeted by a chip click */
.patch-stack details.flash {
  animation: flashRow 1.8s cubic-bezier(0.22,1,0.36,1);
}
@keyframes flashRow {
  0%   { background: color-mix(in srgb, var(--gold) 18%, var(--surface)); box-shadow: 0 0 0 3px color-mix(in srgb, var(--gold) 35%, transparent); }
  60%  { background: color-mix(in srgb, var(--gold) 18%, var(--surface)); box-shadow: 0 0 0 3px color-mix(in srgb, var(--gold) 35%, transparent); }
  100% { background: var(--elev); box-shadow: 0 0 0 0 transparent; }
}

/* top-3 discovery row: slightly tighter cards, highlighted */
.disc-top .disc-card { padding: 22px 22px; border-left-color: var(--gold); position: relative; }
.disc-top .disc-card::after {
  content: ""; position: absolute; top: 18px; right: 18px;
  width: 22px; height: 22px; border-radius: 100px;
  background: color-mix(in srgb, var(--gold) 14%, transparent);
  border: 1px solid color-mix(in srgb, var(--gold) 30%, transparent);
}
.disc-top .disc-card:nth-child(1)::after { content: "1"; }
.disc-top .disc-card:nth-child(2)::after { content: "2"; }
.disc-top .disc-card:nth-child(3)::after { content: "3"; }
.disc-top .disc-card::after {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 12px; font-weight: 500;
  color: var(--gold); text-align: center; line-height: 22px;
}
.disc-top .disc-card h4 { padding-right: 32px; }

/* footer */
footer { padding: 60px 0 40px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--hair); margin-top: 60px; }

/* reproducibility card: inline target code may include the same syntax-highlight
   spans as the hero target-card; keep it readable on one line + allow wrap */
.repro-target { display: inline; white-space: normal; word-break: break-word; line-height: 1.6; }
.repro-pipeline > div { margin-bottom: 4px; }

@media (max-width: 880px) {
  .hero h1 { font-size: 40px; }
  .hero-stats { grid-template-columns: 1fr 1fr; }
  .grid-2, .grid-3, .grid-attr { grid-template-columns: 1fr; }
  .patch-stack summary { grid-template-columns: 60px 1fr; }
  .patch-stack .sha, .patch-stack .pct { display: none; }
  .top-card { grid-template-columns: 1fr; }
  .top-rank { font-size: 56px; }
}
"""

    # JS — theme toggle + chart hover
    JS = """
(function() {
  // ----- theme toggle (persisted) -----
  const root = document.documentElement;
  const btn = document.getElementById('themeToggle');
  const KEY = 'autozyme-theme';
  function set(t) {
    if (t === 'dark') root.setAttribute('data-theme','dark');
    else root.removeAttribute('data-theme');
    if (btn) btn.textContent = (t === 'dark') ? '☀ light' : '☾ dark';
    localStorage.setItem(KEY, t);
  }
  set(localStorage.getItem(KEY) || 'light');
  if (btn) btn.addEventListener('click', () => {
    const cur = root.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
    set(cur === 'dark' ? 'light' : 'dark');
  });

  // ----- chart y-axis toggles (scoped per chart to avoid cross-talk) -----
  function wireChartToggle(stackSelector, paneSelector, toggleSelector, labelId, labelTpl) {
    const stack = document.querySelector(stackSelector);
    if (!stack) return;
    const toggle = stack.parentElement.querySelector(toggleSelector);
    if (!toggle) return;
    const panes = Array.from(stack.querySelectorAll(paneSelector));
    const btns = Array.from(toggle.querySelectorAll('.chart-toggle-btn'));
    const labelEl = labelId ? document.getElementById(labelId) : null;
    // Capture the leading "<tier> tier · <scale> scale ·" prefix from the
    // initial label so we can preserve it across toggles without hardcoding.
    const initialLabel = labelEl ? labelEl.textContent : '';
    const labelPrefix = initialLabel.replace(/\s*·\s*[^·]+$/, '');
    function setAxis(mode) {
      panes.forEach(p => { p.hidden = (p.getAttribute('data-mode') !== mode); });
      btns.forEach(b => b.classList.toggle('is-active', b.getAttribute('data-axis') === mode));
      stack.setAttribute('data-mode', mode);
      if (labelEl && labelTpl) labelEl.textContent = labelPrefix + ' · ' + labelTpl(mode);
    }
    btns.forEach(b => b.addEventListener('click', () => setAxis(b.getAttribute('data-axis'))));
  }

  wireChartToggle('.speed-chart-stack', '.speed-chart-pane', '.speed-toggle',
    'speedAxisLabel', m => m === 'speedup' ? 'speedup ×' : 'seconds');
  wireChartToggle('.mem-chart-stack', '.mem-chart-pane', '.mem-toggle',
    'memAxisLabel', m => m === 'ratio' ? 'memory × (log)' : 'MB');

  // ----- chart entrance animation (line draws + dots stagger) -----
  function animateChartSvg(svg) {
    if (!svg || svg.classList.contains('animated')) return;
    const line = svg.querySelector('.chart-line');
    if (line && typeof line.getTotalLength === 'function') {
      try {
        const len = line.getTotalLength();
        if (len > 0) line.style.setProperty('--line-len', len);
      } catch (e) { /* hidden / detached — skip */ }
    }
    const dots = svg.querySelectorAll('.dot');
    const denom = Math.max(dots.length - 1, 1);
    dots.forEach((dot, i) => {
      dot.style.setProperty('--dot-d', (700 + (i / denom) * 700) + 'ms');
    });
    requestAnimationFrame(() => svg.classList.add('animated'));
  }
  document.querySelectorAll('.speed-svg, .memSvg').forEach(svg => {
    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((entries) => {
        entries.forEach(e => {
          if (e.isIntersecting) { animateChartSvg(svg); io.disconnect(); }
        });
      }, { threshold: 0.15 });
      io.observe(svg);
    } else {
      animateChartSvg(svg);
    }
  });
  // Hidden toggle panes never intersect — animate them when their pane is first shown.
  document.querySelectorAll('.chart-toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      requestAnimationFrame(() => {
        document.querySelectorAll('.speed-chart-pane:not([hidden]) .speed-svg, .mem-chart-pane:not([hidden]) .memSvg')
          .forEach(animateChartSvg);
      });
    });
  });

  // ----- chart hover (works for both speed + memory svgs) -----
  const tip = document.getElementById('chartTip');
  const wrap = document.getElementById('chartWrap');
  if (tip && wrap) {
    const allDots = wrap.querySelectorAll('.dot');
    function showTip(c) {
      const rd = c.getAttribute('data-rd');
      const sha = c.getAttribute('data-sha');
      const spd = c.getAttribute('data-spd');
      const st = c.getAttribute('data-st');
      const desc = c.getAttribute('data-desc');
      const mem = c.getAttribute('data-mem');
      tip.className = 'tip-' + st;
      const stColors = { keep: 'var(--jade)', discard: 'var(--rose)', crash: 'var(--amber)', baseline: 'var(--gold)' };
      tip.innerHTML =
        '<div class="row1">' +
          '<span class="tip-rd">round ' + rd + ' · <span style="color:'+stColors[st]+';text-transform:uppercase;font-size:10px;letter-spacing:1.5px;">'+st+'</span></span>' +
          '<span class="tip-sha">' + sha + '</span>' +
        '</div>' +
        '<div class="tip-spd">' + (parseFloat(spd) > 0 ? spd + ' s' : '— crash —') + '</div>' +
        (mem !== '0' ? '<div class="tip-mem">peak ' + mem + ' MB</div>' : '') +
        '<div class="tip-desc">' + (desc || '(no description)') + '</div>';
      const wrapRect = wrap.getBoundingClientRect();
      const dotRect = c.getBoundingClientRect();
      const tipW = 320, tipH = 140;
      let x = dotRect.left - wrapRect.left + dotRect.width/2 - tipW/2;
      let y = dotRect.top - wrapRect.top - tipH - 14;
      x = Math.max(8, Math.min(x, wrapRect.width - tipW - 8));
      if (y < 8) y = dotRect.bottom - wrapRect.top + 14;
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
      tip.style.width = tipW + 'px';
      requestAnimationFrame(() => tip.classList.add('show'));
      // sync hot state across both svgs for the same round
      const rdAttr = c.getAttribute('data-rd');
      allDots.forEach(d => d.classList.toggle('is-hot', d.getAttribute('data-rd') === rdAttr));
    }
    function hideTip() {
      tip.classList.remove('show');
      allDots.forEach(d => d.classList.remove('is-hot'));
    }
    allDots.forEach(c => {
      c.addEventListener('mouseenter', () => showTip(c));
      c.addEventListener('mouseleave', hideTip);
      c.addEventListener('focus', () => showTip(c));
      c.addEventListener('blur', hideTip);
      c.setAttribute('tabindex', '0');
    });
  }

  // ----- tier chart: animate bars on scroll-in -----
  const tier = document.getElementById('tierChart');
  if (tier) {
    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((entries) => {
        entries.forEach(e => {
          if (e.isIntersecting) {
            tier.classList.add('animated');
            io.disconnect();
          }
        });
      }, { threshold: 0.25 });
      io.observe(tier);
    } else {
      tier.classList.add('animated');
    }
  }

  // ----- attribution donut: load animation + linked hover -----
  const donut = document.getElementById('donut');
  const attrBlock = document.getElementById('attrBlock');
  if (donut && attrBlock) {
    const legend = attrBlock.querySelector('.legend');
    const donutNum = document.getElementById('donutNum');
    const donutLbl = document.getElementById('donutLbl');
    const totalNum = donut.getAttribute('data-total');
    const segs = Array.from(donut.querySelectorAll('.donut-seg'));
    const items = Array.from(legend ? legend.querySelectorAll('.leg-item') : []);

    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((entries) => {
        entries.forEach(e => {
          if (e.isIntersecting) {
            donut.classList.add('animated');
            io.disconnect();
          }
        });
      }, { threshold: 0.3 });
      io.observe(donut);
    } else {
      donut.classList.add('animated');
    }

    function setActive(cat) {
      if (cat) {
        donut.classList.add('has-active');
        if (legend) legend.classList.add('has-active');
        segs.forEach(s => s.classList.toggle('is-active', s.dataset.cat === cat));
        items.forEach(i => i.classList.toggle('is-active', i.dataset.cat === cat));
        const seg = segs.find(s => s.dataset.cat === cat);
        if (seg) {
          donutNum.textContent = seg.dataset.n;
          donutLbl.textContent = seg.dataset.name + ' · ' + seg.dataset.pct;
        }
      } else {
        donut.classList.remove('has-active');
        if (legend) legend.classList.remove('has-active');
        segs.forEach(s => s.classList.remove('is-active'));
        items.forEach(i => i.classList.remove('is-active'));
        donutNum.textContent = totalNum;
        donutLbl.textContent = 'accepted opts';
      }
    }
    segs.forEach(s => {
      s.addEventListener('mouseenter', () => setActive(s.dataset.cat));
      s.addEventListener('mouseleave', () => setActive(null));
    });
    items.forEach(i => {
      i.addEventListener('mouseenter', () => setActive(i.dataset.cat));
      i.addEventListener('mouseleave', () => setActive(null));
    });
  }

  // ----- scroll-spy nav highlighting -----
  const sections = document.querySelectorAll('section[id]');
  const navLinks = {};
  document.querySelectorAll('nav.top a[href^="#"]').forEach(a => {
    const id = a.getAttribute('href').slice(1);
    if (id) navLinks[id] = a;
  });
  if (sections.length && 'IntersectionObserver' in window) {
    const spy = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          Object.values(navLinks).forEach(l => l.classList.remove('is-current'));
          const link = navLinks[e.target.id];
          if (link) link.classList.add('is-current');
        }
      });
    }, { rootMargin: '-40% 0px -55% 0px', threshold: 0 });
    sections.forEach(s => spy.observe(s));
  }

  // ----- discovery round chip → open + scroll to + flash patch stack row -----
  document.querySelectorAll('.disc-round-chip').forEach(chip => {
    chip.addEventListener('click', (e) => {
      const round = chip.getAttribute('data-round');
      const stackSec = document.getElementById('stack');
      if (!stackSec || !round) return;
      e.preventDefault();
      // open the section-level collapsible
      const stackCollapse = stackSec.querySelector('details.section-collapse');
      if (stackCollapse) stackCollapse.open = true;
      // find the matching patch row, open it, scroll, flash
      const target = stackSec.querySelector(`.patch-stack details[data-round="${round}"]`);
      if (target) {
        target.open = true;
        requestAnimationFrame(() => {
          target.scrollIntoView({ behavior: 'smooth', block: 'center' });
          target.classList.remove('flash');
          // re-trigger animation
          void target.offsetWidth;
          target.classList.add('flash');
          setTimeout(() => target.classList.remove('flash'), 1900);
        });
      } else {
        stackSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
})();
"""

    # ---- narrative slot resolution (with mechanical fallbacks) ----
    # Substitution context for {speedup}, {baseline}, etc. inside narrative templates.
    subst_ctx = dict(
        speedup=f"{hero_speedup:.0f}",
        speedup_pct=f"{hero_pct:.1f}",
        baseline=f"{hero_baseline:.0f}",
        baseline_min=f"{hero_baseline/60:.1f}",
        final=f"{hero_final:.1f}",
        tier=hero_tier_desc,
        task=task_dir.name,
    )

    # hero_tagline: <h1> content. Narrative may include HTML (e.g. <em>...</em>).
    hero_tagline_html = _subst(NARRATIVE.get("hero_tagline"), subst_ctx) or \
        f"<em>{hero_speedup:.0f}×</em> faster"

    # target_signature: pre-formatted code block content (full HTML allowed).
    target_signature_html = _subst(NARRATIVE.get("target_signature"), subst_ctx) or \
        f'<span class="tk-fn">{esc(task_dir.name)}</span>'

    # target_meta: small text line under the code (full HTML allowed, e.g. dot-sep spans).
    target_meta_html = _subst(NARRATIVE.get("target_meta"), subst_ctx) or esc(FIELD)

    # hero_tier_label: short label for the production-tier hero stat tile.
    # The tier chart already shows the full tier description; the hero tile
    # benefits from something tighter ("BRCA Wu (10X)" rather than
    # "BRCA Wu (OOD prod, 10X ~50k)"). Falls back to the full description.
    hero_tier_label = NARRATIVE.get("hero_tier_label") or hero_tier_desc

    # Memory delta — median across tiers. Color follows the median's sign
    # (gold for net savings, rose for net cost) so the hero tile visually
    # matches the headline number; the subtitle separately reports range +
    # whether all tiers agreed on sign (all_saved / all_regressed / mixed).
    if hero_mem_sign_status == "n/a":
        hero_mem_delta_html = "n/a"
        hero_mem_delta_class = "dim"
        hero_mem_ctx_html = "peak memory not recorded"
    else:
        if hero_mem_delta_pct < 0:
            hero_mem_delta_html = f"−{abs(hero_mem_delta_pct):.0f}%"
            hero_mem_delta_class = "gold"   # net savings
        else:
            hero_mem_delta_html = f"+{hero_mem_delta_pct:.0f}%"
            hero_mem_delta_class = "rose"   # net cost
        def _fmt_pct(v):
            return f"−{abs(v):.0f}%" if v < 0 else f"+{v:.0f}%"
        sign_note = {
            "all_saved":     "all saved",
            "all_regressed": "all regressed",
            "mixed":         "mixed sign",
        }[hero_mem_sign_status]
        hero_mem_ctx_html = f"range {_fmt_pct(hero_mem_lo)} to {_fmt_pct(hero_mem_hi)} <span class='dot-sep'>·</span> {sign_note}"

    # Threading: compute baseline thread count vs final-keep thread count.
    # Reads column 12 (`thread`) from results.tsv. If absent or always 1, show
    # "single-threaded"; otherwise "<baseline> → <final> threads".
    _thread_baseline = None
    _thread_final = None
    for r in results:
        if r["status"] == "baseline" and r.get("dataset") == hero_tier_name:
            try:
                _thread_baseline = int(r.get("phase") or r.get("thread") or 1)
            except (ValueError, TypeError):
                pass
    # The final keep's thread setting — last keep wins
    for r in sorted(results, key=lambda x: x["round_f"]):
        if r["status"] == "keep":
            try:
                # results.tsv schema: round / commit / dataset / speed_sec /
                # speedup_pct / peak_mb / status / metrics_json / hypothesis /
                # description / phase / thread. We parse `phase` into r["phase"],
                # but `thread` isn't captured by parse_results — read it raw.
                pass
            except Exception:
                pass

    # Re-parse threads directly from results.tsv (parse_results dropped col 12)
    _thread_at_baseline = 1
    _thread_at_final_keep = 1
    try:
        with open(task_dir / "results.tsv") as _fh:
            _hdr = _fh.readline().rstrip("\n").split("\t")
            _thr_idx = _hdr.index("thread") if "thread" in _hdr else None
            _last_keep_thread = None
            _baseline_thread = None
            if _thr_idx is not None:
                for _ln in _fh:
                    _cols = _ln.rstrip("\n").split("\t")
                    if len(_cols) <= _thr_idx: continue
                    try:
                        _t = int(_cols[_thr_idx])
                    except (ValueError, TypeError):
                        continue
                    _status = _cols[6] if len(_cols) > 6 else ""
                    if _status == "baseline" and _baseline_thread is None:
                        _baseline_thread = _t
                    if _status == "keep":
                        _last_keep_thread = _t
            _thread_at_baseline = _baseline_thread or 1
            _thread_at_final_keep = _last_keep_thread or _thread_at_baseline
    except Exception:
        pass

    if _thread_at_baseline == _thread_at_final_keep == 1:
        hero_thread_val_html = "1"
        hero_thread_ctx_html = "single-threaded"
    elif _thread_at_baseline == _thread_at_final_keep:
        hero_thread_val_html = f"{_thread_at_final_keep}"
        hero_thread_ctx_html = f"{_thread_at_final_keep}-thread (unchanged)"
    else:
        hero_thread_val_html = f"{_thread_at_baseline}→{_thread_at_final_keep}"
        hero_thread_ctx_html = f"baseline {_thread_at_baseline}t · final {_thread_at_final_keep}t"

    # ---- Reproducibility section data (task-generic, read from task.yaml) ----
    try:
        from zyme.parsers.task_yaml import (
            parse_threading_mode,
            parse_metrics,
            parse_intrinsic_noise,
            effective_threshold,
        )
        _threading_mode = parse_threading_mode(task_dir / "task.yaml")
        _repro_metrics = parse_metrics(task_dir / "task.yaml")
        _intrinsic_noise = parse_intrinsic_noise(task_dir / "task.yaml")
        _effective_threshold = effective_threshold
    except Exception:
        _threading_mode = "default"
        _repro_metrics = []
        _intrinsic_noise = {}
        _effective_threshold = None

    # Upstream: parse target_repo from task.yaml. GitHub URLs become org/repo
    # links; anything else is shown verbatim (with placeholder filter).
    _target_repo = ""
    try:
        for raw in (task_dir / "task.yaml").read_text().splitlines():
            m = re.match(r"^target_repo\s*:\s*(.+)$", raw.strip())
            if m:
                _target_repo = m.group(1).strip().strip('"').strip("'")
                break
    except Exception:
        pass
    if _target_repo and "github.com/" in _target_repo:
        _rest = _target_repo.split("github.com/", 1)[1].rstrip("/").rstrip(".git")
        repro_upstream_html = f'<a href="{esc(_target_repo)}" target="_blank" rel="noopener">{esc(_rest)}</a> · GitHub'
    elif _target_repo and not _target_repo.startswith("<"):
        repro_upstream_html = esc(_target_repo)
    else:
        repro_upstream_html = '<span style="color:var(--muted)">(not declared)</span>'

    # Threading description: agent-curated narrative slot wins; otherwise derive
    # from the task.yaml `threading:` field.
    repro_threading_html = NARRATIVE.get("threading_note") or (
        '<code>not_applicable</code> — sequential algorithm; thread=1 is the '
        'optimized path (see <code>memory/discoveries.md</code> for rationale)'
        if _threading_mode == "not_applicable"
        else '<code>default</code> — task wires <code>ZYME_THREADS</code>; verified across thread sweep'
    )

    # Determinism: derive from how strict the validation thresholds are. Any
    # metric loose enough to permit drift → "bounded". All strict → "bit-exact".
    def _is_bit_exact(metrics):
        if not metrics:
            return False
        for m in metrics:
            if "threshold" not in m:
                return False  # stochastic metric → bounded drift by definition
            thr, cmp_ = m["threshold"], m["comparator"]
            if cmp_ == "gte" and thr < 1.0: return False
            if cmp_ == "lte" and thr > 0:   return False
        return True
    repro_determinism_html = NARRATIVE.get("determinism_note") or (
        "bit-exact at fixed seed across all tiers"
        if _is_bit_exact(_repro_metrics)
        else (f"bounded drift within {len(_repro_metrics)} declared metric "
              f"threshold{'s' if len(_repro_metrics)!=1 else ''}"
              if _repro_metrics else "(no metric block in task.yaml)")
    )

    # Override style: narrative slot or generic. Most R tasks ship via
    # `install_override()`; Python tasks ship via lazy module patches.
    repro_override_html = NARRATIVE.get("override_style") or (
        '<code>install_override()</code> — drop-in (no upstream fork)'
    )

    # Target row in Pipeline card: reuse the syntax-highlighted target_signature_html,
    # collapsed onto one line for the inline view.
    repro_target_html = re.sub(r"\s+", " ", target_signature_html).strip()

    # Validation gates table rows
    repro_metric_rows = []
    for m in _repro_metrics:
        name = esc(m.get("name", ""))
        if "threshold" in m:
            cmp_sym = "≥" if m.get("comparator") == "gte" else "≤"
            thr = m["threshold"]
            thr_str = f"{thr:g}"
            repro_metric_rows.append(
                f'<tr><td>{name}</td><td class="num gold">{cmp_sym} {esc(thr_str)}</td></tr>'
            )
        elif "noise_multiplier" in m:
            nm = m["noise_multiplier"]
            af = m.get("absolute_floor", 0)
            cmp_sym = "≥" if m.get("comparator") == "gte" else "≤"
            repro_metric_rows.append(
                f'<tr><td>{name}</td><td class="num gold" style="font-size:11px">'
                f'{cmp_sym} noise × {nm:g} (floor {af:g})</td></tr>'
            )
    repro_metrics_html = "".join(repro_metric_rows) or (
        '<tr><td colspan="2" style="color:var(--muted);text-align:center;padding:14px">'
        'no metric block declared in task.yaml</td></tr>'
    )

    hero_metrics = latest_metrics.get(hero_tier_name, (None, {}))[1] if hero_tier_name else {}
    hero_tier_key = next((tier for ds, tier, _ in TIER_ORDER if ds == hero_tier_name), hero_tier_name or "")
    hero_metric_passes = 0
    for name, value in hero_metrics.items():
        gate = next((m for m in _repro_metrics if m.get("name") == name), None)
        if not gate:
            continue
        if "threshold" in gate:
            threshold = gate["threshold"]
        elif _effective_threshold is not None:
            try:
                threshold, _ = _effective_threshold(gate, hero_tier_key, _intrinsic_noise)
            except Exception:
                continue
        else:
            continue
        if gate.get("comparator") == "gte" and value >= threshold:
            hero_metric_passes += 1
        elif gate.get("comparator") == "lte" and value <= threshold:
            hero_metric_passes += 1
    if hero_metrics and _repro_metrics:
        hero_concordance_val = "passes" if hero_metric_passes == len(_repro_metrics) else "review"
        hero_concordance_ctx = f"{hero_metric_passes}/{len(_repro_metrics)} metrics within gates"
    elif hero_metrics:
        hero_concordance_val = "recorded"
        hero_concordance_ctx = f"{len(hero_metrics)} metrics in results.tsv"
    else:
        hero_concordance_val = "n/a"
        hero_concordance_ctx = "no metrics recorded for hero tier"

    tier_count = len(TIER_ORDER)
    ood_count = sum(1 for _, tier, _ in TIER_ORDER if str(tier).startswith("ood"))
    dev_count = tier_count - ood_count
    if ood_count and dev_count:
        tier_context = f"{dev_count} dev + {ood_count} OOD tiers"
    elif ood_count:
        tier_context = f"{ood_count} OOD tiers"
    else:
        tier_context = f"{tier_count} declared tiers"

    # why_faster_intro: lead paragraph under the "Why is it faster?" section.
    # Optional narrative slot; mechanical fallback summarizes the actual tier /
    # angle distribution from round_angles (or stays generic when in legacy mode).
    def _build_why_faster_intro():
        if use_angle_mode:
            n_total = sum(attr_counts.values())
            tier_strs = []
            for t in ("T1", "T2", "T3", "FLAG"):
                k = attr_counts.get(t, 0)
                if not k: continue
                lbl = TIER_META[t][0]
                tier_strs.append(f"{k} in <b>{esc(lbl.split(' / ')[0])}</b>")
            tier_blurb = "; ".join(tier_strs) if tier_strs else ""
            # Top 2 angles by count for the "biggest contributors" phrase.
            sorted_angles = sorted(
                angle_counts.items(), key=lambda kv: -kv[1],
            )
            top_phrases = []
            for (num, name, _tier), n in sorted_angles[:2]:
                top_phrases.append(f"<b>{esc(name)}</b> ({n})")
            top_blurb = ", ".join(top_phrases) if top_phrases else ""
            return (
                f"{n_total} accepted optimizations distribute across "
                f"<b>{len({a['primary_num'] for a in round_angles if not a['flag']})}</b> "
                f"of the 17 mechanical angles defined in the autozyme paper "
                f"(Fig 3 taxonomy): {tier_blurb}. "
                f"Largest contributors: {top_blurb}."
            )
        # legacy mode — keep the wording generic, no task-specific MCMC text.
        n_total = sum(attr_counts.values())
        return (
            f"{n_total} accepted optimizations decompose into the categories below."
        )
    why_faster_intro_html = NARRATIVE.get("why_faster_intro") or _build_why_faster_intro()

    HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>autozyme · {esc(task_dir.name)} · optimization report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400;1,9..144,500&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>

<nav class="top">
  <div class="wrap">
    <a href="#summary" class="brand" aria-label="autozyme">
      <img src="{LOGO_DATA}" alt="autozyme" class="brand-logo">
    </a>
    <a href="#tiers">Tiers</a>
    <a href="#journey">Journey</a>
    <a href="#attribution">Attribution</a>
    <a href="#top5">Top 5</a>
    <a href="#stack">Stack</a>
    <a href="#dead-ends">Dead ends</a>
    <a href="#discoveries">Discoveries</a>
    <span class="spacer"></span>
    <button id="themeToggle" class="theme-toggle">☾ dark</button>
  </div>
</nav>

<section class="hero" id="summary">
  <div class="wrap">
    <div class="eyebrow"><span class="rule"></span>{FIELD}</div>
    <h1>{hero_tagline_html}</h1>

    <div class="target-card">
      <div class="target-label">Target function</div>
      <pre class="target-code">{target_signature_html}</pre>
      <div class="target-meta">{target_meta_html}</div>
    </div>

    <div class="hero-stats">
      <div class="stat">
        <div class="lbl">Median speedup <span class="dot-sep">·</span> {n_hero_tiers} tier{('' if n_hero_tiers == 1 else 's')}</div>
        <div class="val jade">{hero_speedup:.0f}×</div>
        <div class="ctx">range {hero_speedup_lo:.1f}× → {hero_speedup_hi:.1f}× <span class="dot-sep">·</span> <span style="color:var(--jade)">−{hero_pct:.1f}%</span></div>
      </div>
      <div class="stat">
        <div class="lbl">Median peak memory Δ</div>
        <div class="val {hero_mem_delta_class}">{hero_mem_delta_html}</div>
        <div class="ctx">{hero_mem_ctx_html}</div>
      </div>
      <div class="stat">
        <div class="lbl">Threads</div>
        <div class="val">{hero_thread_val_html}</div>
        <div class="ctx">{hero_thread_ctx_html}</div>
      </div>
      <div class="stat">
        <div class="lbl">Concordance <span class="dot-sep">·</span> {tier_count} tier{('' if tier_count == 1 else 's')}</div>
        <div class="val">{hero_concordance_val}</div>
        <div class="ctx">{hero_concordance_ctx}</div>
      </div>
    </div>
  </div>
</section>

<section id="tiers">
  <div class="wrap">
    <h2>Performance across all tiers</h2>
    <p class="lead">
      Speedup is measured across every declared dataset, including OOD tiers when present.
      This is the sanity check that the patch generalizes beyond the tiny iteration target.
    </p>
    {tier_chart}
  </div>
</section>

<section id="journey">
  <div class="wrap">
    <h2>The iteration path</h2>
    <p class="lead">
      Each dot is one round on the tiny dev tier. <b style="color:var(--jade)">Green</b> = kept,
      <b style="color:var(--rose)">rose</b> = discarded (concordance drift, speed regression, or within-noise gain),
      <b style="color:var(--amber)">amber</b> = crashed.
      Log y-axis. <i>Hover any point</i> to see the hypothesis, exact timing, and peak memory.
    </p>
    <div class="card elev" id="chartWrap">
      <div class="chart-label">
        <span>speed</span>
        <span class="chart-label-dim" id="speedAxisLabel">{esc(_chart_anchor_label)} tier · log scale · seconds</span>
        <span class="chart-toggle speed-toggle" role="group" aria-label="speed y-axis units">
          <button type="button" class="chart-toggle-btn is-active" data-axis="sec">seconds</button>
          <button type="button" class="chart-toggle-btn" data-axis="speedup">speedup ×</button>
        </span>
      </div>
      {speed_chart}
      <div class="chart-divider"></div>
      <div class="chart-label">
        <span>peak memory</span>
        <span class="chart-label-dim" id="memAxisLabel">{esc(_chart_anchor_label)} tier · MB</span>
        <span class="chart-toggle mem-toggle" role="group" aria-label="memory y-axis units">
          <button type="button" class="chart-toggle-btn is-active" data-axis="abs">MB</button>
          <button type="button" class="chart-toggle-btn" data-axis="ratio">memory ×</button>
        </span>
      </div>
      {mem_chart}
      <div id="chartTip"></div>
    </div>
    <div class="legend-row">
      <span><span class="dot" style="background:var(--gold)"></span>baseline</span>
      <span><span class="dot" style="background:var(--jade)"></span>{n_keeps} accepted</span>
      <span><span class="dot" style="background:var(--rose)"></span>{n_discards} discarded</span>
      <span><span class="dot" style="background:var(--amber)"></span>{n_crashes} crashed</span>
    </div>
  </div>
</section>

<section id="attribution">
  <div class="wrap">
    <h2>Why is it faster?</h2>
    <p class="lead">{why_faster_intro_html}</p>
    <div class="card elev grid-attr" id="attrBlock">
      <div>{donut}</div>
      <div class="legend">{legend_html}</div>
    </div>
  </div>
</section>

<section id="top5">
  <div class="wrap">
    <h2>The five biggest wins</h2>
    <p class="lead">
      Ranked by absolute seconds saved on the tiny dev tier. Each one is a single accepted round —
      what exactly the agent changed, the wall-clock impact across tiers, and why the trick works.
    </p>
    <div class="top5-grid">
      {top5_html}
    </div>
  </div>
</section>

<section id="stack">
  <div class="wrap">
    <h2>The full patch stack</h2>
    <p class="lead">
      All {n_keeps} accepted optimizations composed into the current best. Click any row to see the
      mechanism, where it lives in <code>pipeline/run.R</code>, and what it guarantees.
    </p>
    <details class="section-collapse">
      <summary>
        <span>Show full patch stack</span>
        <span class="count">{n_keeps} optimizations</span>
        <span class="chev">▾</span>
      </summary>
      <div class="collapse-body">
        <div class="patch-stack">{patch_html_parts}</div>
      </div>
    </details>
  </div>
</section>

<section id="dead-ends">
  <div class="wrap">
    <h2>Falsified angles</h2>
    <p class="lead">
      The honest part of any optimization log: ideas that <i>should</i> have worked, didn't, and why.
      Every entry below cost rounds of agent time; documenting the mechanism prevents re-trying the
      same angle in slightly different clothes.
    </p>
    <details class="section-collapse">
      <summary>
        <span>Show falsified angles</span>
        <span class="count">{len(de_cards)} dead ends</span>
        <span class="chev">▾</span>
      </summary>
      <div class="collapse-body">{de_html}</div>
    </details>
  </div>
</section>

<section id="discoveries">
  <div class="wrap">
    <h2>Non-obvious discoveries</h2>
    <p class="lead">
      Findings about the target call graph that only show up under profiling and validation.
      These are the observations that made the accepted patch stack possible.
    </p>
    <div class="grid-3 disc-top">{disc_html_top}</div>
    {disc_collapsible}
  </div>
</section>

<section id="repro">
  <div class="wrap">
    <h2>Reproducibility</h2>
    <div class="grid-2">
      <div class="card">
        <h3 style="margin-top:0">Pipeline</h3>
        <div class="repro-pipeline" style="font-size:13px;color:var(--dim);line-height:1.85">
          <div><b style="color:var(--text)">Target:</b> <code class="repro-target">{repro_target_html}</code></div>
          <div><b style="color:var(--text)">Upstream:</b> {repro_upstream_html}</div>
          <div><b style="color:var(--text)">Override style:</b> {repro_override_html}</div>
          <div><b style="color:var(--text)">Threading:</b> {repro_threading_html}</div>
          <div><b style="color:var(--text)">Determinism:</b> {repro_determinism_html}</div>
        </div>
      </div>
      <div class="card">
        <h3 style="margin-top:0">Validation gates</h3>
        <table class="metrics" style="font-size:12px">
          {repro_metrics_html}
        </table>
      </div>
    </div>
  </div>
</section>

<div style="height:80px"></div>

<script>{JS}</script>
</body>
</html>
"""

    output_path.write_text(HTML)
    return output_path
