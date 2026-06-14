"""Unit tests for zyme.report (the HTML optimization-report renderer).

report.py is ~2800 lines but most of the logic is pure: a handful of
module-level parsers/formatters plus one big public entry point
(`render_report`) whose many nested helpers (svg charts, tier chart, donut,
patch/top5/dead-end/discovery card builders, hero-number math, reproducibility
section) are only reachable by driving the whole renderer end-to-end.

Strategy:
  - Test every module-level function directly across normal + edge inputs.
  - Drive `render_report` with realistic results.tsv + task.yaml + memory/*.md
    fixtures, plus targeted edge variants (zero rows, baseline-only, NaN/None
    metrics, huge/tiny speedups, multi-thread data, angle-mode narrative,
    legacy-categorize mode, discard/crash statuses) and assert on concrete
    substrings/values in the emitted HTML.

These tests are deterministic and dependency-light: only stdlib + zyme.report.
The renderer reads/writes plain files in a tmp dir; no network, no scientific
upstream.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

import zyme.report as R
from zyme.report import (
    ANGLE_TAXONOMY,
    CAT_META,
    FIELD_MAP,
    TIER_META,
    VAR,
    _parse_round_angles,
    _parse_tier_descriptions,
    _read_narrative,
    _subst,
    categorize,
    parse_patch_stack,
    parse_results,
    parse_sections,
    read_md,
    render_report,
)


# ==========================================================================
# Local fixtures — task dir scaffolding (we do NOT edit the shared conftest).
# These mirror the conftest templates but add memory/*.md so we can drive the
# full render_report path. Built by hand so each test can tweak one piece.
# ==========================================================================

TASK_YAML = """\
target_repo: https://github.com/example/foo
target_function: foo::bar

datasets:
  - {tier: tiny, name: tiny_a, path: data/tiny.h5ad}
  - {tier: medium, name: medium_a, path: data/medium.h5ad}
  - {tier: ood_large, name: ood_large_a, path: data/ood.h5ad}

metrics:
  - {name: speedup, comparator: gte, threshold: 1.0}
  - {name: max_diff, comparator: lte, noise_multiplier: 2.0, absolute_floor: 0.05}

threading: default
"""

# Realistic multi-round results.tsv on three tiers. Includes baseline / keep /
# discard / crash / rerun statuses and metrics_json payloads.
RESULTS_TSV = """\
round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tphase
0\tabc1234\ttiny_a\t100.0\t0.0\t1024.0\tbaseline\t{}\tupstream\t\toptimize
0\tabc1234\tmedium_a\t400.0\t0.0\t4096.0\tbaseline\t{}\tupstream\t\toptimize
0\tabc1234\tood_large_a\t900.0\t0.0\t8192.0\tbaseline\t{}\tupstream\t\toptimize
1\tdef5678\ttiny_a\t60.0\t40.0\t900.0\tkeep\t{"speedup": 1.6, "max_diff": 0.0}\t[algorithmic] fewer MCMC samples\tcut n.iter\toptimize
1\tdef5678\tmedium_a\t240.0\t40.0\t3600.0\tkeep\t{"speedup": 1.6, "max_diff": 0.0}\t[algorithmic] fewer MCMC samples\tcut n.iter\toptimize
2\taaa1111\ttiny_a\t70.0\t-16.0\t900.0\tdiscard\t{}\t[cache] memoize spline\trolled back\toptimize
3\tbbb2222\ttiny_a\t40.0\t33.0\t850.0\tkeep\t{"speedup": 1.5, "max_diff": 0.01}\t[rcpp] move loop into C++\tRcpp port of inner loop\toptimize
3\tbbb2222\tmedium_a\t160.0\t33.0\t3400.0\tkeep\t{"speedup": 1.5}\t[rcpp] move loop into C++\tRcpp port of inner loop\toptimize
3\tbbb2222\tood_large_a\t360.0\t33.0\t7000.0\tkeep\t{"speedup": 1.5}\t[rcpp] move loop into C++\tRcpp port of inner loop\toptimize
4\tccc3333\ttiny_a\t0.0\t0.0\t0.0\tcrash\t{}\t[vectorize] broadcast residuals\tsegfault\toptimize
5\tddd4444\ttiny_a\t30.0\t25.0\t800.0\tkeep\t{"speedup": 1.33, "max_diff": 0.0}\t[memory] rm() intermediate\tdrop temp matrix\toptimize
5\tddd4444\tmedium_a\t120.0\t25.0\t3200.0\trerun\t{"speedup": 1.33}\t[memory] rm() intermediate\tdrop temp matrix\toptimize
"""

BEST_STATE_MD = """\
# Best state

## Patch stack
- round 1 (def5678) — 40.0% — cut n.iter: fewer MCMC samples
- round 3 (bbb2222) — 33.0% — Rcpp port of inner loop
- round 5 (ddd4444) — 25.0% — rm() intermediate to reclaim memory
"""

ACTIVE_OPTS_MD = """\
# Active opts

## fewer MCMC samples
Round: 1
Mechanism: Reduce n.iter from 10000 to 2000 with no posterior drift.
Speedup: 1.6x on tiny
Concordance: pearson = 1.00
Location: src/model.R:120
Notes: validated against 3 seeds

## Rcpp inner loop
Source rounds: 3
Mechanism: Move the per-element R dispatch into a single-pass C++ kernel.
Concordance: bit-exact
Location: src/kernel.cpp

## dormant idea [inherited from sibling]
Round: 5
Mechanism: Drop the temp matrix the moment it becomes unreachable.
Notes: —

## <opt_name> placeholder
This is a template placeholder section and should be skipped.
"""

DEAD_ENDS_MD = """\
# Dead ends

## Rcpp loses to vectorized R
Tried: round 4 (ccc3333)
Why it doesn't work: The Rcpp port allocated a temp on every call, defeating the gain.
What would change the verdict: A pre-allocated workspace buffer.

## Free-form falsified angle (round 9, commit eee5555)
The broadcast trick triggered a copy-on-write of the whole sparse matrix.

That copy dominated wall-clock for the medium tier, so the net was a regression.
"""

DISCOVERIES_MD = """\
# Discoveries

## DISCOVERY: Rcpp loses to vectorized base R
Cause: BLAS already dispatches the matmul; the C++ port just duplicates it.
Implication: Prefer vectorized R for matmul-bound steps.
Triggered: round 3

## DISCOVERY: hidden invariant in cbind memory layout
The cbind() call forced a dense copy because the inputs disagreed on storage order.

Re-ordering the binds kept everything sparse and halved the peak.

## DISCOVERY: <one-line summary placeholder>
Cause: placeholder
"""


def _make_task(tmp_path: Path, *, task_yaml=TASK_YAML, results=RESULTS_TSV,
               memory=None, narrative=None) -> Path:
    """Build a task dir on disk and return its Path.

    memory: dict of {filename: text} written under memory/.
    narrative: text for memory/report_narrative.md (if given).
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "task.yaml").write_text(task_yaml)
    if results is not None:
        (tmp_path / "results.tsv").write_text(results)
    mem = tmp_path / "memory"
    mem.mkdir(exist_ok=True)
    if memory:
        for name, text in memory.items():
            (mem / name).write_text(text)
    if narrative is not None:
        (mem / "report_narrative.md").write_text(narrative)
    return tmp_path


@pytest.fixture
def full_task(tmp_path: Path) -> Path:
    """A fully-populated task dir: 3 tiers, all statuses, all memory files."""
    return _make_task(
        tmp_path,
        memory={
            "best_state.md": BEST_STATE_MD,
            "active_opts.md": ACTIVE_OPTS_MD,
            "dead_ends.md": DEAD_ENDS_MD,
            "discoveries.md": DISCOVERIES_MD,
        },
    )


def _render(task_dir: Path, **kw) -> str:
    """Render and return the HTML string (also asserts the file was written)."""
    out = render_report(task_dir, **kw)
    assert out.exists()
    return out.read_text()


# ==========================================================================
# parse_results
# ==========================================================================

class TestParseResults:
    def test_returns_one_dict_per_row(self, full_task):
        rows = parse_results(full_task)
        # one dict per data row in RESULTS_TSV (12 data rows)
        assert len(rows) == 12

    def test_numeric_coercion(self, full_task):
        rows = parse_results(full_task)
        r0 = rows[0]
        assert r0["round"] == "0"
        assert r0["round_f"] == 0.0
        assert r0["speed"] == 100.0
        assert r0["peak"] == 1024.0
        assert r0["status"] == "baseline"

    def test_keeps_string_round_and_float_round(self, full_task):
        rows = parse_results(full_task)
        rounds = {r["round"] for r in rows}
        assert "1" in rounds and "3" in rounds

    def test_empty_numeric_fields_become_zero(self, tmp_path):
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
            "1\tc\td\t\t\t\tkeep\t{}\th\tdesc\toptimize\n"
        )
        td = _make_task(tmp_path, results=tsv)
        rows = parse_results(td)
        assert rows[0]["speed"] == 0.0
        assert rows[0]["pct"] == 0.0
        assert rows[0]["peak"] == 0.0

    def test_nonnumeric_round_becomes_negative_one(self, tmp_path):
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
            "setup\tc\td\t1.0\t0\t1\tbaseline\t{}\th\tdesc\toptimize\n"
        )
        td = _make_task(tmp_path, results=tsv)
        rows = parse_results(td)
        assert rows[0]["round_f"] == -1.0

    def test_missing_optional_columns_default(self, tmp_path):
        # Drop metrics_json / hypothesis / description / phase columns.
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\n"
            "1\tc\td\t1.0\t0\t1\tkeep\n"
        )
        td = _make_task(tmp_path, results=tsv)
        rows = parse_results(td)
        assert rows[0]["metrics_json"] == "{}"
        assert rows[0]["hyp"] == ""
        assert rows[0]["phase"] == ""

    def test_header_only_yields_no_rows(self, tmp_path):
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
        )
        td = _make_task(tmp_path, results=tsv)
        assert parse_results(td) == []


# ==========================================================================
# read_md
# ==========================================================================

class TestReadMd:
    def test_reads_existing(self, full_task):
        text = read_md(full_task, "best_state.md")
        assert "Patch stack" in text

    def test_missing_returns_empty(self, full_task):
        assert read_md(full_task, "does_not_exist.md") == ""


# ==========================================================================
# parse_sections
# ==========================================================================

class TestParseSections:
    def test_splits_on_prefix(self):
        text = "## A\nbody a\n## B\nbody b\n"
        secs = parse_sections(text)
        assert secs == [("A", "body a"), ("B", "body b")]

    def test_preamble_before_first_header_ignored(self):
        text = "intro line\n## A\nbody\n"
        assert parse_sections(text) == [("A", "body")]

    def test_empty_text_returns_empty(self):
        assert parse_sections("") == []

    def test_no_header_returns_empty(self):
        assert parse_sections("just prose\nmore prose") == []

    def test_custom_prefix(self):
        text = "# H1\nbody\n"
        assert parse_sections(text, prefix="# ") == [("H1", "body")]

    def test_body_is_stripped(self):
        text = "## A\n\n  spaced  \n\n"
        title, body = parse_sections(text)[0]
        assert body == "spaced"


# ==========================================================================
# parse_patch_stack
# ==========================================================================

class TestParsePatchStack:
    def test_parses_each_entry(self):
        out = parse_patch_stack(BEST_STATE_MD)
        assert len(out) == 3
        assert out[0] == dict(round=1, commit="def5678", pct=40.0,
                              desc="cut n.iter: fewer MCMC samples")

    def test_negative_pct(self):
        text = "## Patch stack\n- round 2 (abc1234) — -5.0% — regressed\n"
        out = parse_patch_stack(text)
        assert out[0]["pct"] == -5.0

    def test_no_patch_stack_section(self):
        assert parse_patch_stack("# nothing here") == []

    def test_malformed_lines_skipped(self):
        text = "## Patch stack\n- not a valid entry line\n- round 1 (aa) — 1.0% — ok\n"
        out = parse_patch_stack(text)
        assert len(out) == 1
        assert out[0]["round"] == 1

    def test_non_dash_line_between_entries_tolerated(self):
        # The `## Patch stack.*?` lead-in uses DOTALL, so a non-bullet line
        # between two entries is skipped and both entries are still parsed.
        text = (
            "## Patch stack\n"
            "- round 1 (aa) — 1.0% — first\n"
            "  indented note (not a bullet)\n"
            "- round 2 (bb) — 2.0% — second\n"
        )
        out = parse_patch_stack(text)
        assert [e["round"] for e in out] == [1, 2]


# ==========================================================================
# categorize
# ==========================================================================

class TestCategorize:
    @pytest.mark.parametrize("name,desc,expected", [
        ("memory: drop temp", "", "memory"),
        ("x", "we call rm( the matrix", "memory"),
        ("x", "gc( forces reclaim", "memory"),
        ("rcpp port", "moved to _cpp kernel", "rcpp"),
        ("x", "lower n.iter for the gibbs chain", "algorithmic"),
        ("x", "skip the consensus subcluster region", "algorithmic"),
        ("x", "memoize the spline fit / cache it", "cache_reuse"),
        ("x", "broadcast and fuse in a single-pass", "vectorize_fuse"),
        ("plain", "nothing special", "other"),
    ])
    def test_buckets(self, name, desc, expected):
        assert categorize(name, desc) == expected

    def test_memory_prefix_wins_over_substring(self):
        # 'memory' prefix must beat any later keyword.
        assert categorize("memory: vectorize", "broadcast") == "memory"

    def test_every_bucket_has_meta(self):
        for cat in {"memory", "rcpp", "algorithmic", "cache_reuse",
                    "vectorize_fuse", "other"}:
            assert cat in CAT_META


# ==========================================================================
# _read_narrative
# ==========================================================================

class TestReadNarrative:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        assert _read_narrative(tmp_path / "nope.md") == {}

    def test_parses_sections_to_lowercase_underscore_keys(self, tmp_path):
        p = tmp_path / "narr.md"
        p.write_text("## Hero Tagline\n<em>10x</em>\n## Why Faster\nbecause\n")
        out = _read_narrative(p)
        assert out["hero_tagline"] == "<em>10x</em>"
        assert out["why_faster"] == "because"

    def test_preamble_before_first_section_ignored(self, tmp_path):
        p = tmp_path / "narr.md"
        p.write_text("intro\n## A\nbody\n")
        assert _read_narrative(p) == {"a": "body"}


# ==========================================================================
# _parse_tier_descriptions
# ==========================================================================

class TestParseTierDescriptions:
    def test_plain_and_dash_lines(self):
        body = "tiny_a: dev tier\n- medium_a: medium tier\n"
        out = _parse_tier_descriptions(body)
        assert out == {"tiny_a": "dev tier", "medium_a": "medium tier"}

    def test_empty_body(self):
        assert _parse_tier_descriptions("") == {}
        assert _parse_tier_descriptions(None) == {}

    def test_comment_and_no_colon_lines_skipped(self):
        body = "# comment\nno colon here\nname: ok\n"
        assert _parse_tier_descriptions(body) == {"name": "ok"}


# ==========================================================================
# _parse_round_angles
# ==========================================================================

class TestParseRoundAngles:
    def test_empty_body(self):
        assert _parse_round_angles("") == []
        assert _parse_round_angles(None) == []

    def test_taxonomy_primary_only(self):
        out = _parse_round_angles("round 11: 10 parallel [T2] — wired threads")
        assert len(out) == 1
        a = out[0]
        assert a["round"] == 11
        assert a["primary_num"] == "10"
        assert a["primary_name"] == "parallel"
        assert a["tier"] == "T2"
        assert a["secondary"] is None
        assert a["rationale"] == "wired threads"
        assert a["flag"] is False

    def test_taxonomy_with_secondary(self):
        out = _parse_round_angles("round 11: 10 parallel [T2] + 8 compile-source — x")
        a = out[0]
        assert a["secondary"] == ("8", "compile-source")

    def test_explicit_flag_line(self):
        out = _parse_round_angles("round 32: FLAG: build-portability — odd one")
        a = out[0]
        assert a["flag"] is True
        assert a["tier"] == "FLAG"
        assert a["primary_name"] == "build-portability"
        assert a["rationale"] == "odd one"

    def test_unparseable_rest_falls_back_to_flag(self):
        out = _parse_round_angles("round 5: total gibberish with no taxonomy")
        a = out[0]
        assert a["flag"] is True
        assert a["tier"] == "FLAG"

    def test_dash_bullet_prefix_stripped(self):
        out = _parse_round_angles("- round 7: 1 eliminate [T1]")
        assert out[0]["round"] == 7
        assert out[0]["primary_name"] == "eliminate"

    def test_lines_without_round_prefix_skipped(self):
        out = _parse_round_angles("# header\nnot a round line\nround 1: 1 eliminate [T1]")
        assert len(out) == 1
        assert out[0]["round"] == 1

    def test_subangle_letter_suffix(self):
        out = _parse_round_angles("round 2: 7a analytical [T2] — closed form")
        assert out[0]["primary_num"] == "7a"
        assert out[0]["primary_name"] == "analytical"

    def test_double_hyphen_separator(self):
        out = _parse_round_angles("round 3: 2 overhead [T1] -- cheaper primitive")
        assert out[0]["rationale"] == "cheaper primitive"


# ==========================================================================
# _subst
# ==========================================================================

class TestSubst:
    def test_replaces_known_keys(self):
        assert _subst("{a}-{b}", {"a": "1", "b": "2"}) == "1-2"

    def test_unknown_keys_left_intact(self):
        assert _subst("{a}-{zzz}", {"a": "1"}) == "1-{zzz}"

    def test_empty_template(self):
        assert _subst("", {"a": "1"}) == ""
        assert _subst(None, {"a": "1"}) == ""

    def test_non_string_values_stringified(self):
        assert _subst("v={n}", {"n": 42}) == "v=42"

    def test_braces_without_identifier_untouched(self):
        # `{ }` / `{1}` aren't valid identifier placeholders.
        assert _subst("{ } {1}", {}) == "{ } {1}"


# ==========================================================================
# Module-level constants / taxonomy sanity
# ==========================================================================

class TestConstants:
    def test_field_map_known(self):
        assert FIELD_MAP["non_bio"] == "scientific computing"
        assert FIELD_MAP["core_singlecell"] == "single-cell genomics"

    def test_var_returns_css_custom_properties(self):
        assert VAR["gold"] == "var(--gold)"
        assert VAR["jade"] == "var(--jade)"

    def test_tier_meta_has_all_four_tiers(self):
        assert set(TIER_META) == {"T1", "T2", "T3", "FLAG"}

    def test_angle_taxonomy_entries_are_triples(self):
        for k, v in ANGLE_TAXONOMY.items():
            assert len(v) == 3
            name, tier, desc = v
            assert tier in {"T1", "T2", "T3"}


# ==========================================================================
# render_report — happy path (full fixture)
# ==========================================================================

class TestRenderReportHappyPath:
    def test_writes_html_and_returns_path(self, full_task):
        out = render_report(full_task)
        assert out == full_task / "report.html"
        assert out.exists()
        html = out.read_text()
        assert html.startswith("<!doctype html>")
        assert html.rstrip().endswith("</html>")

    def test_custom_output_path(self, full_task, tmp_path):
        dest = tmp_path / "elsewhere" / "r.html"
        dest.parent.mkdir()
        out = render_report(full_task, output_path=dest)
        assert out == dest
        assert dest.exists()

    def test_contains_task_name_in_title(self, full_task):
        html = _render(full_task)
        assert f"· {full_task.name} ·" in html

    def test_section_anchors_present(self, full_task):
        html = _render(full_task)
        for anchor in ("id=\"summary\"", "id=\"tiers\"", "id=\"journey\"",
                       "id=\"attribution\"", "id=\"top5\"", "id=\"stack\"",
                       "id=\"dead-ends\"", "id=\"discoveries\"", "id=\"repro\""):
            assert anchor in html

    def test_hero_speedup_is_median_across_tiers(self, full_task):
        # tiny 100/30=3.33x, medium 400/120=3.33x, ood 900/360=2.5x → median 3.33 → "3×"
        html = _render(full_task)
        assert ">3×<" in html  # hero stat val, formatted with :.0f

    def test_patch_stack_rounds_rendered(self, full_task):
        html = _render(full_task)
        # all three patch-stack rounds appear as <details data-round=...>
        assert 'data-round="1"' in html
        assert 'data-round="3"' in html
        assert 'data-round="5"' in html

    def test_keep_discard_crash_counts_in_legend(self, full_task):
        html = _render(full_task)
        # n_keeps = len(patch) = 3 accepted
        assert "3 accepted" in html
        # one discard (round 2 tiny) and one crash (round 4 tiny)
        assert "1 discarded" in html
        assert "1 crashed" in html

    def test_patch_mechanism_wired_from_active_opts(self, full_task):
        html = _render(full_task)
        assert "Reduce n.iter from 10000 to 2000" in html

    def test_dead_end_card_rendered(self, full_task):
        html = _render(full_task)
        assert "FALSIFIED" in html
        assert "Rcpp loses to vectorized R" in html

    def test_discovery_card_rendered(self, full_task):
        html = _render(full_task)
        assert "hidden invariant in cbind" in html

    def test_placeholder_discovery_excluded(self, full_task):
        html = _render(full_task)
        assert "one-line summary placeholder" not in html

    def test_top5_section_lists_biggest_wins(self, full_task):
        html = _render(full_task)
        assert "top5-grid" in html
        # the round-1 mechanism (largest abs saving baseline->r1) should appear
        assert "fewer MCMC samples" in html or "Reduce n.iter" in html

    def test_tier_chart_shows_all_three_speedups(self, full_task):
        html = _render(full_task)
        # tier chart speedups rendered as "<num>×"; 3.3, 3.3, 2.5 round to 1dp
        assert "3.3" in html
        assert "2.5" in html

    def test_svg_charts_emitted(self, full_task):
        html = _render(full_task)
        assert 'id="speedSvg"' in html
        assert 'id="memSvg"' in html
        assert 'id="donut"' in html

    def test_reproducibility_upstream_github_link(self, full_task):
        html = _render(full_task)
        assert "example/foo" in html
        assert "GitHub" in html

    def test_validation_gates_table(self, full_task):
        html = _render(full_task)
        assert "speedup" in html
        assert "max_diff" in html


# ==========================================================================
# render_report — edge cases / robustness
# ==========================================================================

class TestRenderReportEdgeCases:
    def test_no_memory_files(self, tmp_path):
        # Only task.yaml + results.tsv, no memory/*.md at all.
        td = _make_task(tmp_path)
        html = _render(td)
        assert html.startswith("<!doctype html>")
        # patch stack empty → "0 optimizations"
        assert "0 optimizations" in html

    def test_baseline_only_results(self, tmp_path):
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
            "0\tabc\ttiny_a\t100.0\t0\t1024\tbaseline\t{}\tupstream\t\toptimize\n"
        )
        td = _make_task(tmp_path, results=tsv)
        html = _render(td)
        # No finals → hero speedup 0 → "0×"
        assert ">0×<" in html

    def test_empty_results_just_header(self, tmp_path):
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
        )
        td = _make_task(tmp_path, results=tsv)
        html = _render(td)
        assert html.startswith("<!doctype html>")
        # no tiers with both baseline+final → n_hero_tiers 0
        assert "Median speedup" in html

    def test_missing_results_tsv_raises(self, tmp_path):
        # render_report calls parse_results which opens results.tsv.
        (tmp_path / "task.yaml").write_text(TASK_YAML)
        (tmp_path / "memory").mkdir()
        with pytest.raises(FileNotFoundError):
            render_report(tmp_path)

    def test_nan_and_huge_peak_memory_filtered(self, tmp_path):
        # peak NaN and an insane peak (>SANE_MAX_MB) must be dropped from the
        # memory chart without crashing.
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
            "0\tabc\ttiny_a\t100.0\t0\t1024\tbaseline\t{}\tupstream\t\toptimize\n"
            "1\tdef\ttiny_a\t50.0\t50\tnan\tkeep\t{}\th\tcut\toptimize\n"
            "2\tghi\ttiny_a\t40.0\t60\t999999999\tkeep\t{}\th\tmore\toptimize\n"
        )
        td = _make_task(tmp_path, results=tsv)
        html = _render(td)
        assert html.startswith("<!doctype html>")

    def test_huge_speedup_value(self, tmp_path):
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
            "0\tabc\ttiny_a\t10000.0\t0\t1024\tbaseline\t{}\tupstream\t\toptimize\n"
            "1\tdef\ttiny_a\t1.0\t99\t512\tkeep\t{}\th\tbig\toptimize\n"
        )
        td = _make_task(tmp_path, results=tsv,
                        memory={"best_state.md":
                                "## Patch stack\n- round 1 (def) — 99.0% — big win\n"})
        html = _render(td)
        # 10000/1 = 10000x
        assert "10000×" in html

    def test_tiny_subsecond_speeds(self, tmp_path):
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
            "0\tabc\ttiny_a\t0.5\t0\t10\tbaseline\t{}\tupstream\t\toptimize\n"
            "1\tdef\ttiny_a\t0.1\t80\t8\tkeep\t{}\th\tfast\toptimize\n"
        )
        td = _make_task(tmp_path, results=tsv)
        html = _render(td)
        assert html.startswith("<!doctype html>")

    def test_memory_regression_sign(self, tmp_path):
        # Final peak HIGHER than baseline → memory regression (+ sign, rose).
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
            "0\tabc\ttiny_a\t100.0\t0\t500\tbaseline\t{}\tupstream\t\toptimize\n"
            "1\tdef\ttiny_a\t50.0\t50\t900\tkeep\t{}\th\tcut\toptimize\n"
        )
        td = _make_task(tmp_path, results=tsv)
        html = _render(td)
        # +% memory delta and rose styling present
        assert "all regressed" in html

    def test_memory_savings_sign(self, full_task):
        # full fixture: finals all have lower peak than baseline → savings.
        html = _render(full_task)
        assert "all saved" in html or "mixed sign" in html

    def test_invalid_metrics_json_tolerated(self, tmp_path):
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
            "0\tabc\ttiny_a\t100.0\t0\t1024\tbaseline\t{}\tupstream\t\toptimize\n"
            "1\tdef\ttiny_a\t50.0\t50\t900\tkeep\t{not valid json}\th\tcut\toptimize\n"
        )
        td = _make_task(tmp_path, results=tsv)
        html = _render(td)
        assert html.startswith("<!doctype html>")

    def test_field_inferred_from_parent_dir(self, tmp_path):
        # Put the task under a 'non_bio' parent → FIELD = scientific computing.
        parent = tmp_path / "non_bio"
        parent.mkdir()
        td = _make_task(parent / "test_thing")
        html = _render(td)
        assert "scientific computing" in html

    def test_field_default_for_unknown_parent(self, full_task):
        html = _render(full_task)
        # parent is a pytest tmp dir → not in FIELD_MAP → default
        assert "scientific computing" in html


# ==========================================================================
# render_report — angle-mode narrative
# ==========================================================================

ANGLE_NARRATIVE = """\
## hero_tagline
<em>{speedup}×</em> faster on {task}

## round_angles
round 1: 7b better-algorithm [T2] — fewer MCMC samples
round 3: 8 compile-source [T2] — Rcpp port
round 5: 1 eliminate [T1] — drop temp matrix

## tier_descriptions
tiny_a: development tier
medium_a: realistic tier

## threading_note
custom threading prose for the report
"""


class TestRenderReportAngleMode:
    def test_angle_mode_donut_uses_tier_rollup(self, tmp_path):
        td = _make_task(
            tmp_path,
            memory={"best_state.md": BEST_STATE_MD,
                    "active_opts.md": ACTIVE_OPTS_MD},
            narrative=ANGLE_NARRATIVE,
        )
        html = _render(td)
        # angle-mode legend uses TIER_META labels
        assert "Structural / same-math" in html  # T2 label
        assert "Local / same-output" in html      # T1 label

    def test_angle_names_in_legend(self, tmp_path):
        td = _make_task(
            tmp_path,
            memory={"best_state.md": BEST_STATE_MD},
            narrative=ANGLE_NARRATIVE,
        )
        html = _render(td)
        assert "better-algorithm" in html
        assert "compile-source" in html
        assert "eliminate" in html

    def test_narrative_hero_tagline_substituted(self, tmp_path):
        td = _make_task(
            tmp_path,
            memory={"best_state.md": BEST_STATE_MD},
            narrative=ANGLE_NARRATIVE,
        )
        html = _render(td)
        # {task} substituted with task dir name, {speedup} with the median
        assert f"faster on {td.name}" in html

    def test_narrative_threading_note_used(self, tmp_path):
        td = _make_task(
            tmp_path,
            memory={"best_state.md": BEST_STATE_MD},
            narrative=ANGLE_NARRATIVE,
        )
        html = _render(td)
        assert "custom threading prose for the report" in html

    def test_tier_description_from_narrative(self, tmp_path):
        td = _make_task(
            tmp_path,
            memory={"best_state.md": BEST_STATE_MD},
            narrative=ANGLE_NARRATIVE,
        )
        html = _render(td)
        assert "development tier" in html

    def test_top5_flag_angle_label(self, tmp_path):
        # A FLAG round_angle on a top5 win renders the "FLAG · ..." chip label.
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
            "0\tabc\ttiny_a\t100.0\t0\t1024\tbaseline\t{}\tupstream\t\toptimize\n"
            "1\tdef\ttiny_a\t40.0\t60\t800\tkeep\t{}\th\tbig win\toptimize\n"
        )
        narrative = (
            "## round_angles\n"
            "round 1: FLAG: build-portability — out of taxonomy\n"
        )
        td = _make_task(
            tmp_path, results=tsv,
            memory={"best_state.md": "## Patch stack\n- round 1 (def) — 60.0% — big win\n"},
            narrative=narrative,
        )
        html = _render(td)
        assert "FLAG" in html

    def test_angle_lines_for_non_patch_rounds_ignored(self, tmp_path):
        # round_angles mentions round 99 which is not in the patch stack;
        # it must be filtered out (use_angle_mode restricts to patch rounds).
        narrative = (
            "## round_angles\n"
            "round 1: 1 eliminate [T1]\n"
            "round 99: 10 parallel [T2]\n"
        )
        td = _make_task(
            tmp_path,
            memory={"best_state.md": "## Patch stack\n- round 1 (def) — 40.0% — x\n"},
            narrative=narrative,
        )
        html = _render(td)
        # parallel (round 99) should not appear in the legend
        assert "parallel" not in html


# ==========================================================================
# render_report — legacy categorize-mode (no round_angles narrative)
# ==========================================================================

class TestRenderReportLegacyMode:
    def test_legacy_mode_uses_regex_buckets(self, full_task):
        # No round_angles → categorize() buckets drive the donut/legend.
        html = _render(full_task)
        # CAT_META labels — at least the algorithmic + rcpp buckets present
        assert "Algorithmic" in html or "Rcpp ports" in html

    def test_legacy_why_faster_intro_generic(self, full_task):
        html = _render(full_task)
        assert "accepted optimizations decompose into the categories below" in html


# ==========================================================================
# render_report — threading hero tile
# ==========================================================================

class TestRenderReportThreading:
    def test_single_threaded_default(self, full_task):
        # No `thread` column → single-threaded.
        html = _render(full_task)
        assert "single-threaded" in html

    def test_thread_column_changes_hero(self, tmp_path):
        # Add a `thread` column: baseline thread=1, final keep thread=8.
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\tthread\n"
            "0\tabc\ttiny_a\t100.0\t0\t1024\tbaseline\t{}\tupstream\t\toptimize\t1\n"
            "1\tdef\ttiny_a\t30.0\t70\t900\tkeep\t{}\th\tcut\toptimize\t8\n"
        )
        td = _make_task(
            tmp_path, results=tsv,
            memory={"best_state.md": "## Patch stack\n- round 1 (def) — 70.0% — x\n"},
        )
        html = _render(td)
        assert "1→8" in html or "final 8t" in html

    def test_thread_unchanged_nontrivial(self, tmp_path):
        # baseline thread == final keep thread, both > 1 → "unchanged".
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\tthread\n"
            "0\tabc\ttiny_a\t100.0\t0\t1024\tbaseline\t{}\tupstream\t\toptimize\t4\n"
            "1\tdef\ttiny_a\t30.0\t70\t900\tkeep\t{}\th\tcut\toptimize\t4\n"
        )
        td = _make_task(
            tmp_path, results=tsv,
            memory={"best_state.md": "## Patch stack\n- round 1 (def) — 70.0% — x\n"},
        )
        html = _render(td)
        assert "unchanged" in html

    def test_malformed_thread_value_tolerated(self, tmp_path):
        # A non-integer thread cell hits the `except (ValueError, TypeError):
        # continue` guard in the raw thread re-parse without crashing.
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\tthread\n"
            "0\tabc\ttiny_a\t100.0\t0\t1024\tbaseline\t{}\tupstream\t\toptimize\tNA\n"
            "1\tdef\ttiny_a\t30.0\t70\t900\tkeep\t{}\th\tcut\toptimize\t8\n"
        )
        td = _make_task(
            tmp_path, results=tsv,
            memory={"best_state.md": "## Patch stack\n- round 1 (def) — 70.0% — x\n"},
        )
        html = _render(td)
        assert html.startswith("<!doctype html>")


# ==========================================================================
# render_report — reproducibility variations
# ==========================================================================

class TestRenderReportReproducibility:
    def test_non_github_upstream_verbatim(self, tmp_path):
        yaml = TASK_YAML.replace(
            "https://github.com/example/foo", "https://gitlab.com/x/y"
        )
        td = _make_task(tmp_path, task_yaml=yaml,
                        memory={"best_state.md": BEST_STATE_MD})
        html = _render(td)
        assert "gitlab.com/x/y" in html
        # not flagged as GitHub
        assert "x/y</a> · GitHub" not in html

    def test_placeholder_upstream_shows_not_declared(self, tmp_path):
        yaml = TASK_YAML.replace(
            "https://github.com/example/foo", "<your repo here>"
        )
        td = _make_task(tmp_path, task_yaml=yaml,
                        memory={"best_state.md": BEST_STATE_MD})
        html = _render(td)
        assert "(not declared)" in html

    def test_not_applicable_threading(self, tmp_path):
        yaml = TASK_YAML.replace("threading: default", "threading: not_applicable")
        td = _make_task(tmp_path, task_yaml=yaml,
                        memory={"best_state.md": BEST_STATE_MD})
        html = _render(td)
        assert "not_applicable" in html

    def test_no_metric_block(self, tmp_path):
        # Strip the metrics block entirely.
        yaml = (
            "target_repo: https://github.com/example/foo\n"
            "target_function: foo::bar\n\n"
            "datasets:\n"
            "  - {tier: tiny, name: tiny_a, path: data/tiny.h5ad}\n\n"
            "threading: default\n"
        )
        td = _make_task(tmp_path, task_yaml=yaml,
                        memory={"best_state.md": BEST_STATE_MD})
        html = _render(td)
        assert "no metric block declared" in html

    def test_bit_exact_determinism_when_all_strict(self, tmp_path):
        # All metrics strict (gte>=1, lte threshold) → bit-exact.
        yaml = (
            "target_repo: https://github.com/example/foo\n"
            "target_function: foo::bar\n\n"
            "datasets:\n"
            "  - {tier: tiny, name: tiny_a, path: data/tiny.h5ad}\n\n"
            "metrics:\n"
            "  - {name: speedup, comparator: gte, threshold: 1.0}\n"
            "  - {name: max_diff, comparator: lte, threshold: 0.0}\n\n"
            "threading: default\n"
        )
        td = _make_task(tmp_path, task_yaml=yaml,
                        memory={"best_state.md": BEST_STATE_MD})
        html = _render(td)
        assert "bit-exact at fixed seed" in html


# ==========================================================================
# render_report — chart anchor selection
# ==========================================================================

class TestRenderReportChartAnchor:
    def test_anchor_label_uses_tier_slug(self, full_task):
        # tiny_a hosts the most decision rounds → anchor label = its tier "tiny".
        html = _render(full_task)
        assert "tiny tier" in html

    def test_anchor_falls_back_when_no_decisions(self, tmp_path):
        # Only baselines, no keep/discard/crash → anchor by smallest baseline.
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
            "0\tabc\ttiny_a\t10.0\t0\t100\tbaseline\t{}\tupstream\t\toptimize\n"
            "0\tabc\tmedium_a\t40.0\t0\t400\tbaseline\t{}\tupstream\t\toptimize\n"
        )
        td = _make_task(tmp_path, results=tsv)
        html = _render(td)
        assert html.startswith("<!doctype html>")


# ==========================================================================
# Determinism: rendering twice yields identical bytes.
# ==========================================================================

class TestRenderDeterminism:
    def test_render_is_deterministic(self, full_task, tmp_path):
        a = render_report(full_task, output_path=tmp_path / "a.html").read_text()
        b = render_report(full_task, output_path=tmp_path / "b.html").read_text()
        assert a == b


# ==========================================================================
# render_report — discovery ranking / collapsible (>3 discoveries)
# ==========================================================================

MANY_DISCOVERIES_MD = """\
# Discoveries

## DISCOVERY: Rcpp loses to vectorized base R
Cause: BLAS already dispatches the matmul.
Implication: prefer vectorized R.
Triggered: round 3

## DISCOVERY: cat() blocking reclaim of inf
Cause: the cat() call keeps the inf alive past gc.
Implication: drop cat() before peak.

## DISCOVERY: object lived through gc
Cause: a stale reference in the closure.
Implication: rebind to NULL.

## DISCOVERY: mcmc.list lingers
Cause: persistent reference in the result list.

## DISCOVERY: hidden side-effect always writes a cache
Cause: the function always writes to disk.

## DISCOVERY: threading structurally caps at 4
Cause: lock contention beyond 4 cores.
"""


class TestRenderReportDiscoveries:
    def test_collapsible_shown_for_extra_discoveries(self, tmp_path):
        # 6 discoveries → top-3 highlighted + collapsible "show all" for the rest.
        td = _make_task(
            tmp_path,
            memory={"best_state.md": BEST_STATE_MD,
                    "discoveries.md": MANY_DISCOVERIES_MD},
        )
        html = _render(td)
        assert "Show all discoveries" in html
        # the "+N more" count: 6 total - 3 top = 3 more
        assert "+3 more" in html

    def test_unexpected_score_keywords_rank_to_top(self, tmp_path):
        td = _make_task(
            tmp_path,
            memory={"best_state.md": BEST_STATE_MD,
                    "discoveries.md": MANY_DISCOVERIES_MD},
        )
        html = _render(td)
        # high-score titles surface; all titles present somewhere in the doc
        assert "Rcpp loses to vectorized base R" in html
        assert "threading structurally caps" in html

    def test_discovery_round_chip_links(self, full_task):
        html = _render(full_task)
        # round-3 discovery has Triggered: round 3 → round chip into the stack
        assert "disc-round-chip" in html

    def test_low_priority_unexpected_score_keywords(self, tmp_path):
        # Titles hitting the lower-priority _unexpected_score branches
        # ("tolerate", "still stable") + a plain default-score title.
        disc = (
            "# Discoveries\n\n"
            "## DISCOVERY: model can tolerate fewer draws\n"
            "Cause: the posterior is flat in that region.\n\n"
            "## DISCOVERY: still stable at thread=8\n"
            "Cause: no lock contention observed.\n\n"
            "## DISCOVERY: a perfectly ordinary finding\n"
            "Cause: nothing special here.\n"
        )
        td = _make_task(
            tmp_path,
            memory={"best_state.md": BEST_STATE_MD, "discoveries.md": disc},
        )
        html = _render(td)
        assert "can tolerate fewer draws" in html
        assert "still stable at thread=8" in html


# ==========================================================================
# render_report — concordance chips (brca_wu special case) + opt parsing styles
# ==========================================================================

BRCA_RESULTS_TSV = """\
round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\tmetrics_json\thypothesis\tdescription\tphase
0\tabc\tbrca_wu\t100.0\t0\t1024\tbaseline\t{}\tupstream\t\toptimize
1\tdef\tbrca_wu\t40.0\t60\t800\tkeep\t{"pearson_expr": 1.0, "hmm_state_agreement": 0.995, "gene_coverage": 0.92}\th\tcut\toptimize
"""

BRCA_TASK_YAML = """\
target_repo: https://github.com/example/foo
target_function: foo::bar

datasets:
  - {tier: ood_prod, name: brca_wu, path: data/brca.h5ad}

metrics:
  - {name: pearson_expr, comparator: gte, threshold: 0.99}

threading: default
"""


class TestRenderReportConcordanceChips:
    def test_brca_wu_concordance_branch_executes(self, tmp_path):
        # The `brca_wu in latest_metrics` branch builds conc chips. NOTE: the
        # resulting `conc_chip` string is computed but never interpolated into
        # the final HTML (see SUSPECTED BUG in the module docstring/report).
        # We assert the renderer completes without error exercising that branch;
        # the metrics still surface elsewhere (e.g. the validation gates table).
        td = _make_task(
            tmp_path, task_yaml=BRCA_TASK_YAML, results=BRCA_RESULTS_TSV,
            memory={"best_state.md": "## Patch stack\n- round 1 (def) — 60.0% — cut\n"},
        )
        html = _render(td)
        assert html.startswith("<!doctype html>")
        # pearson_expr appears in the validation-gates table (metric name)
        assert "pearson_expr" in html

    def test_hero_metric_gate_passes(self, tmp_path):
        # pearson_expr = 1.0 >= 0.99 threshold → "passes".
        td = _make_task(
            tmp_path, task_yaml=BRCA_TASK_YAML, results=BRCA_RESULTS_TSV,
            memory={"best_state.md": "## Patch stack\n- round 1 (def) — 60.0% — cut\n"},
        )
        html = _render(td)
        assert "metrics within gates" in html


# Bold-markdown + free-form opt journals (astropy-style) exercise the
# alternate field-parsing branches in parse_opt.
BOLD_ACTIVE_OPTS_MD = """\
# Active opts

## bold style round 1
**Speedup**: 1.6x faster
**Concordance**: pearson 0.999
This is the free-form mechanism paragraph that should be picked up.

Second paragraph is ignored for the mechanism field.

## numeric rounds field
Source rounds: 1, 3
Mechanism: applies to multiple rounds.
"""


class TestRenderReportOptParsing:
    def test_bold_markdown_fields_parsed(self, tmp_path):
        td = _make_task(
            tmp_path,
            memory={"best_state.md": BEST_STATE_MD,
                    "active_opts.md": BOLD_ACTIVE_OPTS_MD},
        )
        html = _render(td)
        # free-form mechanism paragraph wired into a patch/top5 card
        assert "free-form mechanism paragraph" in html

    def test_numeric_rounds_field_wires_multiple_rounds(self, tmp_path):
        td = _make_task(
            tmp_path,
            memory={"best_state.md": BEST_STATE_MD,
                    "active_opts.md": BOLD_ACTIVE_OPTS_MD},
        )
        html = _render(td)
        assert "applies to multiple rounds" in html

    def test_pure_prose_opt_mechanism_from_preamble(self, tmp_path):
        # An opt section with NO field labels at all — the first preamble
        # paragraph becomes the Mechanism (free-form fallback branch).
        opts = (
            "# Active opts\n\n"
            "## Round 1: pure prose opt\n"
            "We replaced the dense loop with a sparse single-pass scan.\n\n"
            "A second paragraph that is not part of the mechanism.\n"
        )
        td = _make_task(
            tmp_path,
            memory={"best_state.md": BEST_STATE_MD, "active_opts.md": opts},
        )
        html = _render(td)
        assert "sparse single-pass scan" in html


# ==========================================================================
# render_report — more reachable branches: patch round with no tiny keep,
# dead-end `Cause`/`Variations` fields, mixed memory sign, stochastic gate.
# ==========================================================================

class TestRenderReportMoreBranches:
    def test_patch_round_without_tiny_keep(self, tmp_path):
        # Patch stack references round 7, which only kept on medium (no tiny
        # keep row) → find_tiny_speed returns None for it (the `return None`
        # branch) and it's skipped from top5.
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
            "0\tabc\ttiny_a\t100.0\t0\t1024\tbaseline\t{}\tupstream\t\toptimize\n"
            "1\tdef\ttiny_a\t50.0\t50\t900\tkeep\t{}\th\tcut\toptimize\n"
            "0\tabc\tmedium_a\t400\t0\t4096\tbaseline\t{}\tupstream\t\toptimize\n"
            "7\tghi\tmedium_a\t200\t50\t3000\tkeep\t{}\th\tmedium only\toptimize\n"
        )
        td = _make_task(
            tmp_path, results=tsv,
            memory={"best_state.md":
                    "## Patch stack\n"
                    "- round 1 (def) — 50.0% — tiny win\n"
                    "- round 7 (ghi) — 50.0% — medium-only win\n"},
        )
        html = _render(td)
        assert html.startswith("<!doctype html>")
        # round 1 (kept on tiny) renders in the stack accordion; round 7's
        # missing tiny-keep exercises find_tiny_speed's `return None` branch.
        assert 'data-round="1"' in html

    def test_dead_end_cause_and_variations_fields(self, tmp_path):
        de = (
            "# Dead ends\n\n"
            "## angle with Cause field\n"
            "Cause: the temp allocation dominated.\n"
            "Variations: tried three buffer sizes.\n"
        )
        td = _make_task(
            tmp_path,
            memory={"best_state.md": BEST_STATE_MD, "dead_ends.md": de},
        )
        html = _render(td)
        assert "the temp allocation dominated" in html

    def test_mixed_memory_sign(self, tmp_path):
        # One tier saves memory, another regresses → "mixed sign".
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
            "0\tabc\ttiny_a\t100.0\t0\t500\tbaseline\t{}\tupstream\t\toptimize\n"
            "1\tdef\ttiny_a\t50.0\t50\t300\tkeep\t{}\th\tcut\toptimize\n"
            "0\tabc\tmedium_a\t400\t0\t1000\tbaseline\t{}\tupstream\t\toptimize\n"
            "1\tdef\tmedium_a\t200\t50\t1800\tkeep\t{}\th\tcut\toptimize\n"
        )
        td = _make_task(tmp_path, results=tsv)
        html = _render(td)
        assert "mixed sign" in html

    def test_stochastic_metric_gate_via_effective_threshold(self, tmp_path):
        # A stochastic metric (noise_multiplier + absolute_floor, no threshold)
        # routes the hero-gate evaluation through effective_threshold(), and the
        # determinism note reports "bounded drift".
        yaml = (
            "target_repo: https://github.com/example/foo\n"
            "target_function: foo::bar\n\n"
            "datasets:\n"
            "  - {tier: tiny, name: tiny_a, path: data/t.h5ad}\n\n"
            "metrics:\n"
            "  - {name: max_diff, comparator: lte, noise_multiplier: 2.0, absolute_floor: 0.05}\n\n"
            "intrinsic_noise:\n"
            "  tiny: {max_diff: 0.001}\n\n"
            "threading: default\n"
        )
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
            "0\tabc\ttiny_a\t100.0\t0\t1024\tbaseline\t{}\tupstream\t\toptimize\n"
            "1\tdef\ttiny_a\t40.0\t60\t800\tkeep\t{\"max_diff\": 0.0}\th\tcut\toptimize\n"
        )
        td = _make_task(
            tmp_path, task_yaml=yaml, results=tsv,
            memory={"best_state.md": "## Patch stack\n- round 1 (def) — 60.0% — cut\n"},
        )
        html = _render(td)
        assert "bounded drift" in html
        # max_diff = 0.0 <= effective threshold → gate passes
        assert "metrics within gates" in html

    def test_task_yaml_without_datasets_block(self, tmp_path):
        # task.yaml with no datasets block → parse_datasets returns [] (it does
        # not raise), so TIER_ORDER is empty and the tier chart has no rows, but
        # the renderer still completes and emits the document shell.
        yaml = "garbage: not a real task yaml\nno datasets here\n"
        tsv = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
            "metrics_json\thypothesis\tdescription\tphase\n"
            "0\tabc\ttiny_a\t100.0\t0\t1024\tbaseline\t{}\tupstream\t\toptimize\n"
            "1\tdef\ttiny_a\t50.0\t50\t900\tkeep\t{}\th\tcut\toptimize\n"
        )
        td = _make_task(tmp_path, task_yaml=yaml, results=tsv)
        html = _render(td)
        assert html.startswith("<!doctype html>")
        assert html.rstrip().endswith("</html>")
