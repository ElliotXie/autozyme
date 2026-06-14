"""Parse backend-native profile artifacts into a normalized profile dict.

Schema (returned by normalize):
    {
      "schema_version": "1",
      "backend": "cpu" | "full" | "mem",
      "lang": "py" | "R",
      "tier": <str>,
      "hypothesis": <str>,
      "timestamp": <ISO8601 UTC>,
      "totals": {
        "wall_s": <float|None>,
        "cpu_s": <float|None>,
        "peak_mb": <float|None>,
      },
      "hotspots": [
        {
          "rank": <int, 1-based>,
          "label": "<file>:<line>:<func>",
          "self_time_s": <float|None>,
          "total_time_s": <float|None>,
          "self_pct": <float|None>,
          "calls": <int|None>,
          "raw": { backend-specific fields }
        },
        ...
      ],
      "notes": [<str>, ...],   # caveats agent should know (e.g. "BLAS opaque")
      "artifacts": {
        "raw": "<relpath to backend-native file>",
        "viewer": "<relpath, optional, e.g. profvis.html>",
      }
    }

Each parser is best-effort — a missing or malformed artifact yields an
empty hotspots list with an explanatory note rather than a crash.
"""
import json
import os
import pstats
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TOP_N = 15


# ---------------------------------------------------------------------------
# Actionable ranking: pure profiler hotspots (override_summary pipeline
# previously fed into this was removed when install_override was shimmed; use
# `zyme profile` raw hotspots / call_chains for the same information).
# ---------------------------------------------------------------------------

# Frames that are useful diagnostic evidence but usually poor optimization
# targets — demoted out of actionable_hotspots so the agent doesn't chase
# runtime/IPC frames.
_LOW_ACTIONABILITY_PATTERNS = [
    re.compile(r"\bunserialize\b", re.IGNORECASE),
    re.compile(r"\breadchild\b", re.IGNORECASE),
    re.compile(r"\bmcfork\b", re.IGNORECASE),
    re.compile(r"\blazyloaddbfetch\b", re.IGNORECASE),
    re.compile(r"\bselectchildren\b", re.IGNORECASE),
    re.compile(r"\brecvdata\b", re.IGNORECASE),
    re.compile(r"\bsendmaster\b", re.IGNORECASE),
    re.compile(r"\b__?wait4\b", re.IGNORECASE),
    re.compile(r"\b__?read(?:_nocancel)?\b", re.IGNORECASE),
    re.compile(r"\b__?write(?:_nocancel)?\b", re.IGNORECASE),
    re.compile(r"\bsem_wait\b", re.IGNORECASE),
    re.compile(r"\bpthread_(?:cond|mutex)", re.IGNORECASE),
]


def build_actionable_hotspots(profile_data: dict, top_n: int = TOP_N
                              ) -> tuple[list[dict], list[str]]:
    """Build a target-oriented ranking for agents from profiler hotspots,
    demoting low-actionability runtime/IPC frames so the agent doesn't chase
    fork waits or pthread sync.
    """
    raw_hotspots = profile_data.get("hotspots") or []
    actionable: list[dict] = []
    notes: list[str] = []

    skipped_noise: list[str] = []
    for h in raw_hotspots:
        if len(actionable) >= top_n:
            break
        if _is_low_actionability_hotspot(h):
            skipped_noise.append(str(h.get("label") or "?"))
            continue
        copied = dict(h)
        raw = dict(copied.get("raw") or {})
        raw.setdefault("source", "profiler_hotspot")
        raw.setdefault("source_rank", h.get("rank"))
        copied["raw"] = raw
        copied["rank"] = len(actionable) + 1
        actionable.append(copied)

    # If every raw hotspot was classified as low-actionability, keep the raw
    # list. Empty actionable output would be less useful than showing the
    # "blocked on runtime/IO" evidence.
    if not actionable and raw_hotspots:
        for h in raw_hotspots[:top_n]:
            copied = dict(h)
            raw = dict(copied.get("raw") or {})
            raw.setdefault("source", "profiler_hotspot")
            raw.setdefault("source_rank", h.get("rank"))
            copied["raw"] = raw
            copied["rank"] = len(actionable) + 1
            actionable.append(copied)

    if skipped_noise:
        preview = ", ".join(s[:60] for s in skipped_noise[:5])
        if len(skipped_noise) > 5:
            preview += ", ..."
        notes.append(
            f"actionable ranking demoted {len(skipped_noise)} profiler "
            f"runtime/IPC frame(s): {preview}"
        )
    return actionable, notes


def annotate_call_chains(profile_data: dict) -> list[dict]:
    """Pass-through. Previously annotated chains with nearest override-owner
    hints, but the override_summary pipeline that fed owner names has been
    removed (see Phase 5 of the install_override → patch_namespace refactor).
    Kept as a function so callers don't break; returns chains unmodified.
    """
    return [dict(c) for c in (profile_data.get("call_chains") or [])]


def _is_low_actionability_hotspot(hotspot: dict) -> bool:
    bits = [str(hotspot.get("label") or "")]
    for value in (hotspot.get("raw") or {}).values():
        if isinstance(value, (str, int, float)):
            bits.append(str(value))
    text = " ".join(bits)
    return any(p.search(text) for p in _LOW_ACTIONABILITY_PATTERNS)


def normalize(backend: str, lang: str, artifact_dir: Path,
              hypothesis: str, tier: str,
              artifact_prefix: str,
              executor: dict | None = None,
              totals: dict | None = None) -> dict:
    """Parse the appropriate raw artifact and return the normalized dict."""
    hotspots: list[dict] = []
    notes: list[str] = []
    artifacts: dict[str, str] = {}
    call_chains: list[dict] = []

    if backend == "cpu":
        if lang == "py":
            raw = artifact_dir / "profile.out"
            artifacts["raw"] = f"{artifact_prefix}/profile.out"
            hotspots, n, call_chains = _parse_cprofile(raw)
            notes.extend(n)
        else:
            raw = artifact_dir / "Rprof.out"
            artifacts["raw"] = f"{artifact_prefix}/Rprof.out"
            hotspots, n, call_chains = _parse_rprof(raw, executor)
            notes.extend(n)
    elif backend == "full":
        if lang == "py":
            raw = artifact_dir / "scalene.json"
            artifacts["raw"] = f"{artifact_prefix}/scalene.json"
            hotspots, n = _parse_scalene(raw)
            notes.extend(n)
        else:
            raw = artifact_dir / "Rprof.out"
            artifacts["raw"] = f"{artifact_prefix}/Rprof.out"
            viewer = artifact_dir / "profvis.html"
            if viewer.exists():
                artifacts["viewer"] = f"{artifact_prefix}/profvis.html"
            hotspots, n, call_chains = _parse_rprof(raw, executor)
            notes.extend(n)
    elif backend == "mem":
        if lang == "py":
            raw = artifact_dir / "memray.bin"
            artifacts["raw"] = f"{artifact_prefix}/memray.bin"
            hotspots, n = _parse_memray(raw, executor)
            notes.extend(n)
        else:
            raw = artifact_dir / "Rprof.out"
            artifacts["raw"] = f"{artifact_prefix}/Rprof.out"
            hotspots, n, call_chains = _parse_rprof(raw, executor, mem_focus=True)
            notes.extend(n)
    else:
        raise ValueError(f"unknown backend: {backend}")

    return {
        "schema_version": "1",
        "backend": backend,
        "lang": lang,
        "tier": tier,
        "hypothesis": hypothesis,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "totals": totals or {},
        "hotspots": hotspots,
        "actionable_hotspots": [],
        "call_chains": call_chains,
        "notes": notes,
        "artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# cProfile (deterministic, function-level, CPU+overhead time)
# ---------------------------------------------------------------------------

def _parse_cprofile(prof_path: Path) -> tuple[list[dict], list[str], list[dict]]:
    if not prof_path.exists():
        return [], [f"profile.out missing at {prof_path} — pipeline may have crashed before profile dump"], []
    try:
        stats = pstats.Stats(str(prof_path))
    except Exception as e:
        return [], [f"failed to load profile.out: {e}"], []
    stats.sort_stats("tottime")
    # stats.fcn_list is populated after sort; tuples of (file, line, name).
    keys = stats.fcn_list[:TOP_N] if stats.fcn_list else list(stats.stats.keys())[:TOP_N]
    total_self = sum(stats.stats[k][2] for k in stats.stats)
    hotspots = []
    for rank, key in enumerate(keys, 1):
        cc, nc, tt, ct, _callers = stats.stats[key]
        file, line, name = key
        self_pct = (tt / total_self * 100.0) if total_self > 0 else None
        hotspots.append({
            "rank": rank,
            "label": _short_label(file, line, name),
            "self_time_s": round(tt, 6),
            "total_time_s": round(ct, 6),
            "self_pct": round(self_pct, 2) if self_pct is not None else None,
            "calls": int(cc),
            "raw": {"file": file, "line": line, "func": name, "ncalls": int(nc)},
        })
    notes = [
        "kind=deterministic unit=cpu_seconds (cProfile measures CPU time + tracing overhead)",
        "native code (numpy, BLAS, Cython) appears as a single line — invisible to cProfile internals",
    ]
    call_chains = _cprofile_call_chains(stats, keys, hotspots)
    return hotspots, notes, call_chains


def _cprofile_call_chains(
    stats: pstats.Stats,
    keys: list[tuple[str, int, str]],
    hotspots: list[dict],
) -> list[dict]:
    chains: list[dict] = []
    hotspot_by_key = {key: hotspots[i] for i, key in enumerate(keys[:len(hotspots)])}
    for key in keys[:TOP_N]:
        h = hotspot_by_key.get(key)
        if not h:
            continue
        chain_keys = _best_cprofile_chain(stats, key)
        chain = [_short_label(file, line, name) for file, line, name in chain_keys]
        file, line, name = key
        chains.append({
            "rank": len(chains) + 1,
            "hotspot_rank": h.get("rank"),
            "hotspot": h.get("label"),
            "leaf": _short_label(file, line, name),
            "chain": chain,
            "evidence": "cProfile callers",
            "samples": h.get("calls"),
            "defined_at": f"{file}:{line}" if file and line else None,
            "raw": {"func": name, "file": file, "line": line},
        })
    return chains


def _best_cprofile_chain(
    stats: pstats.Stats,
    leaf: tuple[str, int, str],
    max_depth: int = 8,
) -> list[tuple[str, int, str]]:
    """Choose a representative caller chain by walking the hottest caller."""
    chain = [leaf]
    seen = {leaf}
    cur = leaf
    for _ in range(max_depth - 1):
        callers = (stats.stats.get(cur) or (None, None, None, None, {}))[4]
        if not callers:
            break
        ranked = []
        for caller, vals in callers.items():
            if caller in seen:
                continue
            try:
                # pstats caller tuple: (callcount, reccallcount, tottime, cumtime)
                score = float(vals[3]) if len(vals) > 3 else float(vals[0])
            except (TypeError, ValueError, IndexError):
                score = 0.0
            ranked.append((score, caller))
        if not ranked:
            break
        _score, parent = max(ranked, key=lambda x: x[0])
        chain.append(parent)
        seen.add(parent)
        cur = parent
    return list(reversed(chain))


# ---------------------------------------------------------------------------
# Rprof (sampling, function or line-level, optional memory)
# ---------------------------------------------------------------------------

_RPROF_SUMMARY_R = r'''
suppressWarnings(suppressMessages({
  if (!requireNamespace("jsonlite", quietly=TRUE)) {
    cat("{\"error\": \"jsonlite not installed\"}")
    quit(status=0)
  }
  prof_path <- "%PROF_PATH%"
  if (!file.exists(prof_path)) {
    cat(jsonlite::toJSON(list(error=paste("missing:", prof_path)), auto_unbox=TRUE))
    quit(status=0)
  }
  # `memory="both"` errors when memory.profiling=FALSE was used at capture
  # time. Fall back to plain summary so cpu-mode profiles still parse.
  summ <- tryCatch(summaryRprof(prof_path, memory="both"),
                   error=function(e) NULL)
  if (is.null(summ)) {
    summ <- tryCatch(summaryRprof(prof_path),
                     error=function(e) NULL)
  }
  if (is.null(summ) || is.null(summ$by.self) || nrow(summ$by.self) == 0) {
    cat(jsonlite::toJSON(list(hotspots=list(), notes="empty profile"), auto_unbox=TRUE))
    quit(status=0)
  }
  top <- utils::head(summ$by.self, %TOP_N%)
  has_mem <- !is.null(top$mem.total)
  hotspots <- lapply(seq_len(nrow(top)), function(i) {
    row <- top[i, ]
    list(
      rank=i,
      label=rownames(top)[i],
      self_time_s=row$self.time,
      total_time_s=row$total.time,
      self_pct=row$self.pct,
      calls=NA,
      raw=list(
        total_pct=row$total.pct,
        mem_total_mb=if (has_mem && !is.na(row$mem.total)) row$mem.total else NA
      )
    )
  })
  total_time <- if (!is.null(summ$sampling.time)) summ$sampling.time else NA
  cat(jsonlite::toJSON(list(
    hotspots=hotspots,
    sampling_interval_s=if (!is.null(summ$sample.interval)) summ$sample.interval else NA,
    total_sampled_s=total_time,
    has_memory=has_mem
  ), auto_unbox=TRUE, na="null"))
}))
'''


def _parse_rprof(prof_path: Path, executor: dict | None,
                 mem_focus: bool = False) -> tuple[list[dict], list[str], list[dict]]:
    if not prof_path.exists():
        return [], [f"Rprof.out missing at {prof_path} — pipeline may have crashed before profile dump"], []
    rscript = (executor or {}).get("rscript") or "Rscript"
    # Inject prof_path directly into the snippet — passing it through
    # `--args` doesn't work cleanly when Rscript is launched with `-e`.
    # Path is one we control (resolved Path object), but escape quotes
    # defensively in case of unusual filenames.
    safe_path = str(prof_path).replace("\\", "\\\\").replace('"', '\\"')
    snippet = (_RPROF_SUMMARY_R
               .replace("%TOP_N%", str(TOP_N))
               .replace("%PROF_PATH%", safe_path))
    try:
        result = subprocess.run(
            [rscript, "-e", snippet],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return [], [f"Rscript invocation failed: {e}"], []
    if result.returncode != 0:
        return [], [f"Rscript returned {result.returncode}: {result.stderr.strip()[:200]}"], []
    raw_out = result.stdout.strip()
    if not raw_out:
        return [], ["Rscript produced no output parsing Rprof.out"], []
    try:
        data = json.loads(raw_out)
    except json.JSONDecodeError as e:
        return [], [f"Rscript output not JSON: {e} (output={raw_out[:200]!r})"], []
    if "error" in data:
        return [], [f"R parser error: {data['error']}"], []
    hotspots = data.get("hotspots") or []
    # Normalize: jsonlite emits NA as null after our na='null', but ensure
    # numeric fields are properly typed.
    cleaned = []
    for h in hotspots[:TOP_N]:
        cleaned.append({
            "rank": int(h.get("rank", 0)),
            "label": h.get("label") or "?",
            "self_time_s": _maybe_float(h.get("self_time_s")),
            "total_time_s": _maybe_float(h.get("total_time_s")),
            "self_pct": _maybe_float(h.get("self_pct")),
            "calls": None,
            "raw": h.get("raw") or {},
        })
    notes = [
        f"kind=sampling unit=wall_seconds interval={data.get('sampling_interval_s', '?')}s",
        "Rcpp / BLAS / dispatch internals appear as opaque frames — invisible to Rprof",
    ]
    if mem_focus and data.get("has_memory"):
        notes.append("memory.profiling=TRUE: mem_total_mb in raw is approximate (sampling-based)")
    elif mem_focus:
        notes.append("memory column unavailable — backend=mem requires memory.profiling=TRUE in helper")
    call_chains = _rprof_call_chains(prof_path, cleaned)
    if call_chains:
        notes.append(
            "call_chains are representative Rprof stacks (root -> leaf); "
            "they show dynamic ownership, not static source definitions."
        )
    return cleaned, notes, call_chains


_RPROF_FRAME_RE = re.compile(r'"([^"]+)"')


def _rprof_call_chains(prof_path: Path, hotspots: list[dict]) -> list[dict]:
    stacks = _read_rprof_stacks(prof_path)
    if not stacks:
        return []

    chains: list[dict] = []
    for h in hotspots[:TOP_N]:
        target = _clean_rprof_label(h.get("label"))
        if not target:
            continue
        counter: Counter[tuple[str, ...]] = Counter()
        for stack in stacks:
            if target not in stack:
                continue
            idx = stack.index(target)
            # Rprof stack order is leaf -> root. Store root -> leaf.
            chain = tuple(reversed(stack[idx:]))
            counter[chain] += 1
        if not counter:
            continue
        chain, count = counter.most_common(1)[0]
        chains.append({
            "rank": len(chains) + 1,
            "hotspot_rank": h.get("rank"),
            "hotspot": h.get("label"),
            "leaf": target,
            "chain": list(chain),
            "evidence": "Rprof stack samples",
            "samples": count,
            "defined_at": None,
            "raw": {"stack_samples": count},
        })
    return chains


def _strip_rprof_lineinfo(frame: str) -> str:
    """Strip #file#line suffix from line-profiling Rprof frames.

    With line.profiling=TRUE, frames look like 'La.svd#/path/svd.R#42'.
    summaryRprof aggregates by function name only, so hotspot labels are
    just 'La.svd'. Strip the suffix so call-chain matching works.
    """
    idx = frame.find("#")
    return frame[:idx] if idx > 0 else frame


def _read_rprof_stacks(prof_path: Path) -> list[list[str]]:
    try:
        text = prof_path.read_text(errors="replace")
    except OSError:
        return []
    stacks: list[list[str]] = []
    for line in text.splitlines():
        frames = _RPROF_FRAME_RE.findall(line)
        if frames:
            stacks.append([_strip_rprof_lineinfo(f) for f in frames])
    return stacks


def _clean_rprof_label(label) -> str:
    text = str(label or "").strip()
    while len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return _strip_rprof_lineinfo(text)


# ---------------------------------------------------------------------------
# Scalene (line-level CPU+mem+native split, JSON output)
# ---------------------------------------------------------------------------

def _parse_scalene(json_path: Path) -> tuple[list[dict], list[str]]:
    if not json_path.exists():
        return [], [f"scalene.json missing at {json_path} — Scalene wrapper may have failed"]
    try:
        data = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [], [f"failed to read scalene.json: {e}"]

    elapsed = data.get("elapsed_time_sec")
    files = data.get("files") or {}

    # Scalene's `n_cpu_percent_total` is None in 2.x JSON (verified via the
    # synthetic fixture). The actual signal lives in n_cpu_percent_python /
    # _c / _sys. Compute total = py + c + sys ourselves.
    #
    # **Ranking decision**: we rank by FUNCTIONS only, even though Scalene
    # also reports lines[]. Reason: lines and functions cover the same code
    # but with slightly different attribution (Scalene's per-line counts
    # don't sum exactly to the function's count due to sampling boundaries).
    # Including both led to the synthetic fixture surfacing "cpu_b" then
    # "L47" (cpu_b's loop body) ranked above cpu_c — making the agent
    # think there are 4 hotspots when there are 3. Functions-only gives
    # one entry per logical hotspot.
    #
    # Limitation: module-level code (top-level statements, with-blocks
    # outside any def) is captured in lines[] only — not functions[] —
    # so it's invisible to our ranking. For typical autozyme tasks where
    # the work is inside functions, this is fine. We keep the top
    # non-function lines in `notable_lines` (in notes) for awareness.
    func_rows: list[dict] = []
    line_rows: list[dict] = []
    for file_path, finfo in files.items():
        for fn in (finfo.get("functions") or []):
            cpu_py = _scalene_pct(fn.get("n_cpu_percent_python"))
            cpu_c = _scalene_pct(fn.get("n_cpu_percent_c"))
            cpu_sys = _scalene_pct(fn.get("n_sys_percent"))
            mem_peak = _scalene_pct(fn.get("n_peak_mb"))
            mem_growth = _scalene_pct(fn.get("n_growth_mb"))
            cpu_total = cpu_py + cpu_c + cpu_sys
            score = cpu_total + (mem_peak / 100.0) + (mem_growth / 100.0)
            if score <= 0:
                continue
            label_func = fn.get("line") or fn.get("name") or "?"
            func_rows.append({
                "file": file_path, "label": label_func, "score": score,
                "cpu_total_pct": cpu_total, "cpu_python_pct": cpu_py,
                "cpu_native_pct": cpu_c, "sys_pct": cpu_sys,
                "mem_peak_mb": mem_peak, "mem_growth_mb": mem_growth,
                "mem_avg_mb": _scalene_pct(fn.get("n_avg_mb")),
            })
        for ln in (finfo.get("lines") or []):
            cpu_py = _scalene_pct(ln.get("n_cpu_percent_python"))
            cpu_c = _scalene_pct(ln.get("n_cpu_percent_c"))
            cpu_sys = _scalene_pct(ln.get("n_sys_percent"))
            mem_peak = _scalene_pct(ln.get("n_peak_mb"))
            mem_growth = _scalene_pct(ln.get("n_growth_mb"))
            cpu_total = cpu_py + cpu_c + cpu_sys
            score = cpu_total + (mem_peak / 100.0) + (mem_growth / 100.0)
            if score <= 0:
                continue
            lineno = int(ln.get("lineno") or ln.get("line") or 0)
            line_rows.append({
                "file": file_path, "lineno": lineno, "score": score,
                "cpu_total_pct": cpu_total, "cpu_python_pct": cpu_py,
                "cpu_native_pct": cpu_c, "sys_pct": cpu_sys,
                "mem_peak_mb": mem_peak, "mem_growth_mb": mem_growth,
                "mem_avg_mb": _scalene_pct(ln.get("n_avg_mb")),
            })

    func_rows.sort(key=lambda r: -r["score"])
    # Fallback: when Scalene has 0 functions (workload entirely in C
    # extensions or module-level code), we MUST surface lines or we'd
    # return an empty hotspot list — the astropy_lombscargle bench
    # failure mode caught this exact case. In that fallback, we
    # promote line_rows to hotspots.
    rows_for_ranking = func_rows
    promoted_from_lines = False
    if not func_rows and line_rows:
        rows_for_ranking = sorted(line_rows, key=lambda r: -r["score"])
        promoted_from_lines = True

    hotspots = []
    for rank, r in enumerate(rows_for_ranking[:TOP_N], 1):
        if promoted_from_lines:
            display = f"{os.path.basename(r['file'])}:{r['lineno']}"
            raw_extra = {"kind": "line", "file": r["file"],
                         "lineno": r["lineno"], "func": None}
        else:
            display = f"{os.path.basename(r['file'])}:{r['label']}()"
            raw_extra = {"kind": "func", "file": r["file"],
                         "lineno": None, "func": r["label"]}
        hotspots.append({
            "rank": rank,
            "label": display,
            "self_time_s": (r["cpu_total_pct"] / 100.0 * elapsed) if elapsed else None,
            "total_time_s": None,
            "self_pct": round(r["cpu_total_pct"], 2),
            "calls": None,
            "raw": {
                **raw_extra,
                "cpu_python_pct": round(r["cpu_python_pct"], 2),
                "cpu_native_pct": round(r["cpu_native_pct"], 2),
                "sys_pct": round(r["sys_pct"], 2),
                "mem_avg_mb": round(r["mem_avg_mb"], 2),
                "mem_peak_mb": round(r["mem_peak_mb"], 2),
                "mem_growth_mb": round(r["mem_growth_mb"], 2),
            },
        })

    # Surface top non-function lines as notable_lines in notes — useful when
    # the agent wants to spot module-level setup or with-block costs that
    # aren't visible in the function-ranked hotspots.
    notable_lines: list[str] = []
    if not promoted_from_lines and line_rows:
        # Only surface lines whose CPU pct is meaningful AND whose file
        # has at least one function (i.e. we have function ranking covered).
        line_rows_sorted = sorted(line_rows, key=lambda r: -r["score"])
        for ln in line_rows_sorted[:5]:
            if ln["cpu_total_pct"] >= 5.0:
                notable_lines.append(
                    f"{os.path.basename(ln['file'])}:{ln['lineno']} "
                    f"(cpu_pct={ln['cpu_total_pct']:.1f})"
                )

    notes = [
        f"kind=sampling+alloc unit=mixed total_elapsed={elapsed}s" if elapsed
            else "kind=sampling+alloc unit=mixed",
        "cpu_python_pct vs cpu_native_pct: python-time wins benefit from algorithmic / vectorization changes; native-time wins point at BLAS / Cython / NumPy internals (try different math, not python-level rewrites)",
    ]
    if promoted_from_lines:
        notes.append(
            "no functions[] entries in scalene.json — workload runs entirely in "
            "C extensions or module-level code. Showing line-level ranking instead. "
            "(this is honest output, not a parser bug)"
        )
    elif notable_lines:
        notes.append(
            f"notable non-function lines (cpu_pct >= 5%): "
            + ", ".join(notable_lines)
        )
    if data.get("max_footprint_mb"):
        notes.append(f"peak process memory: {data['max_footprint_mb']:.1f} MB")
    return hotspots, notes


def _scalene_pct(v) -> float:
    """Coerce Scalene's percentage values (None | int | float | str) to float.

    Scalene 2.x emits None for inactive lines (not 0). Defensive.
    """
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# memray (allocation tracking with native attribution)
# ---------------------------------------------------------------------------

def _parse_memray(bin_path: Path, executor: dict | None) -> tuple[list[dict], list[str]]:
    if not bin_path.exists():
        return [], [f"memray.bin missing at {bin_path} — memray Tracker may have failed"]
    # Use the task's python (where memray was installed) to invoke memray stats.
    from zyme.commands.profile.backends import _resolve_python_bin
    py_bin = _resolve_python_bin(executor)
    if not py_bin:
        return [], ["could not resolve task python interpreter for memray analysis"]

    # memray stats --json writes to a file (no stdout option). Use a sibling
    # path next to memray.bin and clean up afterward. -n controls top-N count.
    json_out = bin_path.with_suffix(".stats.json")
    try:
        result = subprocess.run(
            [py_bin, "-m", "memray", "stats", "--json",
             "-o", str(json_out), "-f",
             "-n", str(TOP_N), str(bin_path)],
            capture_output=True, text=True, timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return [], [f"memray stats invocation failed: {e}"]
    if result.returncode != 0:
        return [], [f"memray stats returned {result.returncode}: {result.stderr.strip()[:300]}"]
    if not json_out.exists():
        return [], [f"memray stats finished but produced no JSON at {json_out}"]

    try:
        data = json.loads(json_out.read_text())
    except json.JSONDecodeError as e:
        return [], [f"memray --json output not parseable: {e}"]
    finally:
        # Sidecar JSON is a build artifact of the parse step; not interesting
        # to keep alongside the canonical profile.json.
        try:
            json_out.unlink()
        except OSError:
            pass

    # memray's top_allocations_by_size schema:
    #   { "location": "<func>:<file>:<line>", "size": <bytes> }
    # `location` is a single colon-delimited string (not a dict). Split it
    # tolerantly — locations like "<module>:<string>:8" or "fn:/abs/path:42"
    # both occur.
    by_size = data.get("top_allocations_by_size") or []
    by_count = data.get("top_allocations_by_count") or []
    # Index by_count by location for cross-lookup so we can show alloc count
    # alongside size when available.
    count_index = {entry.get("location"): entry.get("count") or entry.get("n_allocations") or 0
                   for entry in by_count}

    hotspots: list[dict] = []
    for rank, entry in enumerate(by_size[:TOP_N], 1):
        size = int(entry.get("size") or 0)
        loc_str = str(entry.get("location") or "?")
        func, file, lineno = _split_memray_location(loc_str)
        count = int(count_index.get(loc_str) or 0)
        hotspots.append({
            "rank": rank,
            "label": _short_label(file, lineno, func),
            "self_time_s": None,
            "total_time_s": None,
            "self_pct": None,
            "calls": count or None,
            "raw": {
                "alloc_bytes": size,
                "alloc_count": count,
                "file": file, "lineno": lineno, "func": func,
                "location": loc_str,
            },
        })

    metadata = data.get("metadata") or {}
    total_allocs = data.get("total_num_allocations")
    total_bytes = data.get("total_bytes_allocated")
    peak = metadata.get("peak_memory") or data.get("peak_memory")
    notes = [
        "kind=allocation_trace unit=bytes (sorted by allocation size; high-water-mark not directly exposed by memray stats — use `memray flamegraph` for HWM analysis)",
        "memray captures both Python and native (NumPy, torch buffers) allocations — distinct from tracemalloc which sees only Python-level",
    ]
    if total_allocs is not None and total_bytes is not None:
        notes.append(f"totals: {total_allocs:,} allocations, {total_bytes:,} bytes "
                     f"({total_bytes / (1024**3):.3f} GB)")
    if peak:
        notes.append(f"peak resident: {int(peak):,} bytes ({int(peak) / (1024**3):.2f} GB)")
    return hotspots, notes


def _split_memray_location(loc: str) -> tuple[str, str, int]:
    """Best-effort split of memray's `<func>:<file>:<line>` location string.

    The file part may itself contain a colon (a Windows drive letter, e.g.
    ``C:\\proj\\a.py``), so split ``func`` off the LEFT (before the first colon)
    and ``line`` off the RIGHT (after the last colon); everything in between is
    the file path, keeping any drive-letter colon intact.
    """
    func, sep, rest = loc.partition(":")
    if not sep:
        return loc, "?", 0          # no colon at all -> not a location
    file, sep2, line_s = rest.rpartition(":")
    if not sep2:
        return loc, "?", 0          # only one colon -> not a full func:file:line
    try:
        line = int(line_s)
    except ValueError:
        line = 0
    return func, file, line


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _short_label(file: str, line: int | str, name: str) -> str:
    """Compact identifier; trim absolute paths to basename for readability."""
    try:
        base = os.path.basename(str(file)) if file else "?"
    except Exception:
        base = str(file)[:60]
    return f"{base}:{line}:{name}"


def _maybe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
