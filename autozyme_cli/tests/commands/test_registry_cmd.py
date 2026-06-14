"""Unit tests for zyme.commands.registry — the searchable function-index command.

NOTE: this is the COMMAND module (`zyme.commands.registry`), entirely separate
from `zyme.registry` (the versioned prompt-snapshot store covered by
tests/test_registry_unit.py). No overlap.

Covers the pure helpers (yaml-field parsing, placeholder detection, symbol/slug
derivation, kind/status classification, tokenization, scoring, entry rendering,
TSV read/write, legacy result-dir discovery, suggest-query extraction) plus the
rebuild/query/list/suggest entrypoints driven over a fake workspace tree in
tmp_path with the scan boundary monkeypatched where needed.
"""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands import registry as reg


# --------------------------------------------------------------------------
# YAML field parsing + placeholder detection
# --------------------------------------------------------------------------

class TestReadTaskFields:
    def test_extracts_known_keys(self, tmp_path):
        ty = tmp_path / "task.yaml"
        ty.write_text(
            "task: my_task  # a comment\n"
            "target_repo: https://example.com/foo\n"
            'target_function: "mgcv::gam"\n'
            "signature: foo(x)\n"
            "unrelated: ignored\n"
            "  indented: skipped\n"
        )
        fields = reg._read_task_fields(ty)
        assert fields["task"] == "my_task"
        assert fields["target_repo"] == "https://example.com/foo"
        assert fields["target_function"] == "mgcv::gam"
        assert fields["signature"] == "foo(x)"
        assert "unrelated" not in fields
        assert "indented" not in fields

    def test_missing_file_returns_empty(self, tmp_path):
        assert reg._read_task_fields(tmp_path / "nope.yaml") == {}

    def test_strips_quotes_and_comments(self):
        assert reg._clean_yaml_value("  'val'  # c") == "val"
        assert reg._clean_yaml_value('"val"') == "val"
        assert reg._clean_yaml_value("bare # tail") == "bare"


class TestIsPlaceholder:
    @pytest.mark.parametrize("v", [None, "", "  ", "<PKG::FUNC>", "e.g. foo",
                                   "NA", "null", "NULL", "x>y"])
    def test_placeholder_values(self, v):
        assert reg._is_placeholder(v) is True

    @pytest.mark.parametrize("v", ["mgcv::gam", "real_func", "lifelines"])
    def test_real_values(self, v):
        assert reg._is_placeholder(v) is False


class TestSymbolFromTask:
    def test_prefers_target_function(self, tmp_path):
        d = tmp_path / "test_x"
        d.mkdir()
        fields = {"target_function": "mgcv::gam", "task": "t"}
        assert reg._symbol_from_task(d, fields) == "mgcv::gam"

    def test_falls_back_to_task(self, tmp_path):
        d = tmp_path / "test_x"
        d.mkdir()
        fields = {"target_function": "<PKG::FUNC>", "task": "mytask"}
        assert reg._symbol_from_task(d, fields) == "mytask"

    def test_falls_back_to_dir_name(self, tmp_path):
        d = tmp_path / "test_dir"
        d.mkdir()
        assert reg._symbol_from_task(d, {}) == "test_dir"


# --------------------------------------------------------------------------
# slug / tokens / compact
# --------------------------------------------------------------------------

class TestSlug:
    def test_double_colon(self):
        assert reg._slug("Seurat::NormalizeData") == "Seurat__NormalizeData"

    def test_special_chars_collapse(self):
        assert reg._slug("foo bar/baz") == "foo_bar_baz"

    def test_empty_to_unknown(self):
        assert reg._slug("!!!") == "unknown"


class TestTokensCompact:
    def test_tokens_min_length(self):
        toks = reg._tokens("ab abc abcd a1.2")
        assert "abc" in toks
        assert "ab" not in toks  # len 2 minimum is 3 chars total

    def test_compact_strips_nonalnum(self):
        assert reg._compact("Foo::Bar-1") == "foobar1"


# --------------------------------------------------------------------------
# kind classification
# --------------------------------------------------------------------------

class TestKindFor:
    def test_workflow_primitive_by_symbol(self, tmp_path):
        d = tmp_path / "anything"
        d.mkdir()
        assert reg._kind_for("scanpy.pp.normalize_total", d, {}) == "workflow_primitive"

    def test_workflow_primitive_by_dirname(self, tmp_path):
        d = tmp_path / "run_pca"
        d.mkdir()
        assert reg._kind_for("xyz", d, {}) == "workflow_primitive"

    def test_numerical_engine_by_keyword(self, tmp_path):
        d = tmp_path / "test_blas"
        d.mkdir()
        assert reg._kind_for("BLAS gemm", d, {}) == "numerical_engine"

    def test_numerical_engine_by_repo(self, tmp_path):
        d = tmp_path / "test_x"
        d.mkdir()
        assert reg._kind_for("foo", d, {"target_repo": "github.com/scipy"}) == "numerical_engine"

    def test_fit_false_dirname(self, tmp_path):
        d = tmp_path / "gam_fit_false"
        d.mkdir()
        assert reg._kind_for("x", d, {}) == "numerical_engine"

    def test_default_target_function(self, tmp_path):
        d = tmp_path / "test_x"
        d.mkdir()
        assert reg._kind_for("custom_thing", d, {}) == "target_function"


# --------------------------------------------------------------------------
# status classification
# --------------------------------------------------------------------------

def _phase_row(**done):
    phases = {p: {"done": False} for p in ("init", "iterate", "scaling", "package")}
    for p, v in done.items():
        phases[p] = {"done": v}
    return {"phases": phases}


class TestStatusFor:
    def test_needs_review_on_regression(self):
        row = _phase_row(package=True)
        assert reg._status_for(row, {"speedup_pct": -5.0}) == "needs_review"

    def test_packaged(self):
        assert reg._status_for(_phase_row(package=True), None) == "packaged"

    def test_scaled(self):
        assert reg._status_for(_phase_row(scaling=True), None) == "scaled"

    def test_optimized(self):
        assert reg._status_for(_phase_row(iterate=True), {"speedup_pct": 10}) == "optimized"

    def test_initialized(self):
        assert reg._status_for(_phase_row(init=True), None) == "initialized"

    def test_scaffold(self):
        assert reg._status_for(_phase_row(), None) == "scaffold"


# --------------------------------------------------------------------------
# results.tsv latest-keep extraction
# --------------------------------------------------------------------------

class TestLatestKeep:
    def test_missing_file(self, tmp_path):
        assert reg._latest_keep_from_results(tmp_path / "nope.tsv") is None

    def test_header_only(self, tmp_path):
        p = tmp_path / "results.tsv"
        p.write_text("round\tstatus\n")
        assert reg._latest_keep_from_results(p) is None

    def test_no_status_col(self, tmp_path):
        p = tmp_path / "results.tsv"
        p.write_text("a\tb\n1\t2\n")
        assert reg._latest_keep_from_results(p) is None

    def test_picks_last_keep(self, tmp_path):
        p = tmp_path / "results.tsv"
        p.write_text(
            "round\tdataset\tspeedup_pct\tstatus\n"
            "1\tt\t10.0\tkeep\n"
            "2\tt\t20.0\tdiscard\n"
            "3\tt\t30.0\tkeep\n"
        )
        out = reg._latest_keep_from_results(p)
        assert out == {"round": "3", "dataset": "t", "speedup_pct": 30.0}

    def test_bad_speedup_becomes_none(self, tmp_path):
        p = tmp_path / "results.tsv"
        p.write_text(
            "round\tdataset\tspeedup_pct\tstatus\n"
            "1\tt\tnotnum\tkeep\n"
        )
        out = reg._latest_keep_from_results(p)
        assert out["speedup_pct"] is None


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

class TestScore:
    def _entry(self, **kw):
        base = {"symbol": "mgcv::gam", "kind": "numerical_engine",
                "status": "optimized", "task": "test_gam", "tags": "mgcv,gam"}
        base.update(kw)
        return base

    def test_raw_substring_and_compact(self):
        e = self._entry()
        s = reg._score(e, reg._tokens("mgcv gam"), "mgcv gam")
        assert s > 0

    def test_no_text_match_keeps_only_priors(self):
        # A query that hits no symbol/tag text still scores >0 when the entry
        # carries status/active_opts priors: optimized status alone adds +2.
        # (Documents that _score never returns 0 for a high-status entry.)
        e = self._entry(status="optimized")
        s = reg._score(e, reg._tokens("zzz qqq"), "zzz qqq")
        assert s == 2  # status prior only, no term hits

    def test_zero_for_scaffold_no_match(self):
        # A scaffold entry with no priors and no text match scores exactly 0.
        e = self._entry(status="scaffold")
        s = reg._score(e, reg._tokens("zzz qqq"), "zzz qqq")
        assert s == 0

    def test_status_bonus(self):
        opt = self._entry(status="optimized")
        scaf = self._entry(status="scaffold")
        assert reg._score(opt, reg._tokens("mgcv"), "mgcv") > \
               reg._score(scaf, reg._tokens("mgcv"), "mgcv")

    def test_active_opts_bonus(self):
        e = self._entry(active_opts="/p/active_opts.md")
        e2 = self._entry()
        assert reg._score(e, reg._tokens("mgcv"), "mgcv") > \
               reg._score(e2, reg._tokens("mgcv"), "mgcv")


# --------------------------------------------------------------------------
# entry rendering
# --------------------------------------------------------------------------

class TestPrintEntry:
    def test_full(self, capsys):
        e = {"symbol": "mgcv::gam", "kind": "numerical_engine",
             "status": "optimized", "speedup_pct": "40.0",
             "task": "test_gam", "task_dir": "/p/test_gam",
             "active_opts": "/p/active.md", "detail_path": "/p/detail.md"}
        reg._print_entry(e)
        out = capsys.readouterr().out
        assert "mgcv::gam" in out
        assert "speedup=40.0%" in out
        assert "/p/test_gam" in out
        assert "/p/active.md" in out
        assert "/p/detail.md" in out

    def test_no_paths(self, capsys):
        e = {"symbol": "x", "kind": "k", "status": "s",
             "task": "t", "task_dir": "/d"}
        reg._print_entry(e, show_paths=False)
        out = capsys.readouterr().out
        assert "/d" not in out


# --------------------------------------------------------------------------
# TSV round trip
# --------------------------------------------------------------------------

class TestEntriesRoundTrip:
    def test_write_then_load(self, tmp_path):
        entries = [
            {"symbol": "a", "kind": "k", "status": "optimized",
             "task": "t", "tags": "x"},
        ]
        reg._write_entries(tmp_path, entries)
        loaded = reg._load_entries(tmp_path)
        assert loaded[0]["symbol"] == "a"
        # missing fields written as empty string
        assert loaded[0]["patch_path"] == ""
        # header complete
        with (tmp_path / "functions.tsv").open() as fh:
            header = fh.readline().rstrip("\n").split("\t")
        assert header == reg.FIELDNAMES

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="registry not found"):
            reg._load_entries(tmp_path)


# --------------------------------------------------------------------------
# legacy result dirs
# --------------------------------------------------------------------------

class TestLegacyResultDirs:
    def test_finds_legacy_symbol_dirs(self, tmp_path):
        # /root/cat/run_pca/results.tsv (depth-2 glob) but no task.yaml
        d = tmp_path / "cat" / "run_pca"
        d.mkdir(parents=True)
        (d / "results.tsv").write_text("round\tstatus\n1\tkeep\n")
        out = reg._legacy_result_dirs([tmp_path], [], include_bench=True)
        assert d.resolve() in {p.resolve() for p in out}

    def test_skips_when_task_yaml_present(self, tmp_path):
        d = tmp_path / "cat" / "run_pca"
        d.mkdir(parents=True)
        (d / "results.tsv").write_text("round\tstatus\n1\tkeep\n")
        (d / "task.yaml").write_text("x: 1\n")
        out = reg._legacy_result_dirs([tmp_path], [], include_bench=True)
        assert d.resolve() not in {p.resolve() for p in out}

    def test_skips_non_legacy_name(self, tmp_path):
        d = tmp_path / "cat" / "some_other_thing"
        d.mkdir(parents=True)
        (d / "results.tsv").write_text("round\tstatus\n1\tkeep\n")
        out = reg._legacy_result_dirs([tmp_path], [], include_bench=True)
        assert out == []


class TestMakeLegacyEntry:
    def test_optimized_when_positive(self, tmp_path):
        d = tmp_path / "run_pca"
        d.mkdir()
        (d / "results.tsv").write_text(
            "round\tdataset\tspeedup_pct\tstatus\n1\tt\t10\tkeep\n")
        e = reg._make_legacy_entry(d)
        assert e["symbol"] == "Seurat::RunPCA"
        assert e["status"] == "optimized"
        assert e["kind"] == "workflow_primitive"

    def test_scaffold_when_no_results(self, tmp_path):
        d = tmp_path / "run_pca"
        d.mkdir()
        e = reg._make_legacy_entry(d)
        assert e["status"] == "scaffold"


# --------------------------------------------------------------------------
# dependency hits
# --------------------------------------------------------------------------

class TestDependencyHits:
    def test_matches_pattern(self, tmp_path):
        d = tmp_path / "test_gam"
        d.mkdir()
        (d / "README.md").write_text("This task optimizes mgcv::gam internals.")
        hits = reg._dependency_hits(d)
        symbols = [h[0] for h in hits]
        assert "mgcv::gam" in symbols

    def test_no_corpus_no_hits(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert reg._dependency_hits(d) == []


# --------------------------------------------------------------------------
# detail writer
# --------------------------------------------------------------------------

class TestWriteDetail:
    def test_writes_markdown(self, tmp_path):
        entry = {"symbol": "mgcv::gam", "kind": "numerical_engine",
                 "status": "optimized", "task": "test_gam", "phase": "package",
                 "speedup_pct": "40", "round": "3", "task_dir": "/d"}
        related = [{"task": "sibling", "status": "optimized", "speedup_pct": "10"}]
        evidence = ["test_gam: some snippet"]
        path = reg._write_detail(tmp_path, entry, related=related, evidence=evidence)
        text = Path(path).read_text()
        assert "# mgcv::gam" in text
        assert "Dependency Evidence" in text
        assert "Related Tasks" in text
        assert Path(path).name == "mgcv__gam.md"


# --------------------------------------------------------------------------
# root resolution
# --------------------------------------------------------------------------

class TestRegistryRoot:
    def test_explicit_override(self, tmp_path):
        args = SimpleNamespace(registry_root=str(tmp_path / "reg"))
        assert reg._registry_root(args, None) == (tmp_path / "reg").resolve()

    def test_from_framework(self, tmp_path):
        args = SimpleNamespace(registry_root=None)
        assert reg._registry_root(args, tmp_path) == tmp_path / "registry"

    def test_fallback_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        args = SimpleNamespace(registry_root=None)
        assert reg._registry_root(args, None) == tmp_path / "registry"


# --------------------------------------------------------------------------
# suggest query extraction
# --------------------------------------------------------------------------

class TestExtractSuggestQuery:
    def test_task_dir(self, tmp_path):
        d = tmp_path / "test_x"
        d.mkdir()
        (d / "task.yaml").write_text("target_function: mgcv::gam\n")
        (d / "README.md").write_text("optimize gam")
        q = reg._extract_suggest_query(d, None)
        assert "mgcv::gam" in q
        assert "optimize gam" in q

    def test_profile_json(self, tmp_path):
        prof = tmp_path / "profile.json"
        prof.write_text('{"actionable_hotspots": ["gam.fit"]}')
        q = reg._extract_suggest_query(None, prof)
        assert "gam.fit" in q

    def test_profile_non_json(self, tmp_path):
        prof = tmp_path / "profile.txt"
        prof.write_text("raw text hotspot")
        q = reg._extract_suggest_query(None, prof)
        assert "raw text hotspot" in q

    def test_task_yaml_file_directly(self, tmp_path):
        ty = tmp_path / "task.yaml"
        ty.write_text("target_function: foo\n")
        (tmp_path / "README.md").write_text("readme body")
        q = reg._extract_suggest_query(ty, None)
        assert "foo" in q
        assert "readme body" in q


# --------------------------------------------------------------------------
# command entrypoints driven over fake scan boundary
# --------------------------------------------------------------------------

class TestRebuildAndQuery:
    def _make_task(self, root, name, *, target_function, status_keep=True):
        d = root / name
        d.mkdir(parents=True)
        (d / "task.yaml").write_text(
            f"task: {name}\n"
            f"target_function: {target_function}\n"
            "target_repo: https://example.com/x\n"
        )
        if status_keep:
            (d / "results.tsv").write_text(
                "round\tdataset\tspeedup_pct\tstatus\n1\tt\t25\tkeep\n")
        return d

    @pytest.fixture
    def fake_scan(self, tmp_path, monkeypatch):
        """Build a workspace with two tasks; stub scan helpers used by rebuild."""
        ws = tmp_path / "ws"
        fw = ws / "autozyme-framework"
        fw.mkdir(parents=True)
        t1 = self._make_task(ws, "test_gam", target_function="mgcv::gam")
        t2 = self._make_task(ws, "run_pca", target_function="Seurat::RunPCA")
        monkeypatch.chdir(ws)
        monkeypatch.setattr(reg, "find_workspace_root", lambda cwd: ws)
        monkeypatch.setattr(reg, "find_framework_root", lambda cwd: fw)
        monkeypatch.setattr(reg, "find_tasks",
                            lambda roots, max_depth, framework_root: [t1, t2])

        def fake_detect_phase(task_dir, framework):
            kept = (task_dir / "results.tsv").exists()
            return {
                "phases": {
                    "init": {"done": True},
                    "iterate": {"done": kept},
                    "scaling": {"done": False},
                    "package": {"done": False, "patch_path": ""},
                },
                "latest_keep": reg._latest_keep_from_results(task_dir / "results.tsv"),
                "phase": "iterate",
                "task_name": task_dir.name,
            }
        monkeypatch.setattr(reg, "detect_phase", fake_detect_phase)
        return SimpleNamespace(ws=ws, fw=fw, registry_root=fw / "registry")

    def test_rebuild_then_query_list(self, fake_scan, capsys):
        rebuild_args = SimpleNamespace(
            framework_root=str(fake_scan.fw), paths=[], registry_root=None,
            max_depth=3, include_bench=False)
        reg.cmd_registry_rebuild(rebuild_args)
        out = capsys.readouterr().out
        assert "registry: wrote" in out
        assert (fake_scan.registry_root / "functions.tsv").is_file()

        # query
        query_args = SimpleNamespace(
            registry_root=str(fake_scan.registry_root), terms=["mgcv"], limit=10)
        reg.cmd_registry_query(query_args)
        out = capsys.readouterr().out
        assert "mgcv::gam" in out

        # list filtered by kind
        list_args = SimpleNamespace(
            registry_root=str(fake_scan.registry_root),
            kind="numerical_engine", limit=10)
        reg.cmd_registry_list(list_args)
        out = capsys.readouterr().out
        assert "mgcv::gam" in out

    def test_query_no_match(self, tmp_path, monkeypatch, capsys):
        # The "no match" branch only fires when ALL entries score 0, which
        # requires scaffold-status entries with no priors (optimized status
        # always adds +2). Build such a registry directly.
        root = tmp_path / "registry"
        reg._write_entries(root, [
            {"symbol": "thing", "kind": "target_function",
             "status": "scaffold", "task": "t", "tags": "thing"},
        ])
        monkeypatch.setattr(reg, "find_framework_root", lambda cwd: None)
        query_args = SimpleNamespace(
            registry_root=str(root), terms=["zzz_no_such_token"], limit=10)
        reg.cmd_registry_query(query_args)
        err = capsys.readouterr().err
        assert "no match" in err

    def test_suggest(self, fake_scan, capsys):
        rebuild_args = SimpleNamespace(
            framework_root=str(fake_scan.fw), paths=[], registry_root=None,
            max_depth=3, include_bench=False)
        reg.cmd_registry_rebuild(rebuild_args)
        capsys.readouterr()
        # Point --task at the gam task dir so the query text mentions mgcv::gam.
        suggest_args = SimpleNamespace(
            registry_root=str(fake_scan.registry_root),
            task=str(fake_scan.ws / "test_gam"), profile=None, limit=5)
        reg.cmd_registry_suggest(suggest_args)
        out = capsys.readouterr().out
        assert "Registry Suggestions" in out
        assert "Optimized / Reusable Matches" in out

    def test_suggest_requires_input(self, fake_scan):
        rebuild_args = SimpleNamespace(
            framework_root=str(fake_scan.fw), paths=[], registry_root=None,
            max_depth=3, include_bench=False)
        reg.cmd_registry_rebuild(rebuild_args)
        suggest_args = SimpleNamespace(
            registry_root=str(fake_scan.registry_root),
            task=None, profile=None, limit=5)
        with pytest.raises(SystemExit):
            reg.cmd_registry_suggest(suggest_args)
