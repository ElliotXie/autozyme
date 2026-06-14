"""Wave-4 mop-up for zyme.commands.registry — the searchable function index.

(NOT zyme.registry, the prompt-snapshot store.) Wave-2 (test_registry_cmd.py)
covered the pure helpers + the rebuild/query/list/suggest happy paths. This
file fills the remaining REACHABLE branches:

  - _read_limited / _headings: OSError fallback + the "## " heading scan with
    a real headings file.
  - _latest_keep_from_results: short-row continue, non-keep rows skipped,
    no-keep-found returns None.
  - _legacy_result_dirs: the depth-1 (`*/results.tsv`) glob and the
    include_bench=False skip.
  - _write_detail: patch_path / active_opts / discoveries detail lines, the
    Active-Optimization-Headings section, and the related-task active_opts
    annotation.
  - cmd_registry_rebuild: the dependency-map "optimized_dependency" promotion
    (a task whose target_function IS the matched dependency symbol).
  - cmd_registry_suggest: the empty-bucket "- none" branches and the
    similar-task dedup loop.

Everything runs over tmp_path fixtures with the scan boundary monkeypatched.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands import registry as reg


# --------------------------------------------------------------------------
# _read_limited / _headings
# --------------------------------------------------------------------------

class TestReadLimitedAndHeadings:
    def test_read_limited_missing_returns_empty(self, tmp_path):
        assert reg._read_limited(tmp_path / "nope.md") == ""

    def test_read_limited_oserror_returns_empty(self, tmp_path, monkeypatch):
        p = tmp_path / "x.md"
        p.write_text("body")

        def boom(*a, **k):
            raise OSError("io error")
        monkeypatch.setattr(Path, "read_text", boom)
        assert reg._read_limited(p) == ""

    def test_headings_extracts_h2(self, tmp_path):
        p = tmp_path / "active.md"
        p.write_text(
            "# Title\n"
            "## First heading\n"
            "body\n"
            "## Second heading\n"
            "### subheading ignored\n"
        )
        heads = reg._headings(p)
        assert heads == ["First heading", "Second heading"]

    def test_headings_respects_limit(self, tmp_path):
        p = tmp_path / "many.md"
        p.write_text("\n".join(f"## H{i}" for i in range(20)))
        heads = reg._headings(p, limit=3)
        assert len(heads) == 3

    def test_headings_empty_when_missing(self, tmp_path):
        assert reg._headings(tmp_path / "gone.md") == []


# --------------------------------------------------------------------------
# _latest_keep_from_results — remaining edges
# --------------------------------------------------------------------------

class TestLatestKeepEdges:
    def test_oserror_returns_none(self, tmp_path, monkeypatch):
        p = tmp_path / "results.tsv"
        p.write_text("round\tstatus\n1\tkeep\n")

        def boom(*a, **k):
            raise OSError("io")
        monkeypatch.setattr(Path, "read_text", boom)
        assert reg._latest_keep_from_results(p) is None

    def test_no_keep_rows_returns_none(self, tmp_path):
        p = tmp_path / "results.tsv"
        p.write_text(
            "round\tdataset\tspeedup_pct\tstatus\n"
            "1\tt\t10\tdiscard\n"
            "2\tt\t20\tpending\n"
        )
        assert reg._latest_keep_from_results(p) is None

    def test_short_row_skipped(self, tmp_path):
        # A row shorter than the status column index is skipped (continue),
        # and the next valid keep row is returned.
        p = tmp_path / "results.tsv"
        p.write_text(
            "round\tdataset\tspeedup_pct\tstatus\n"
            "1\tt\t10\tkeep\n"
            "short\n"  # too few columns -> continue
        )
        out = reg._latest_keep_from_results(p)
        assert out["round"] == "1"


# --------------------------------------------------------------------------
# _legacy_result_dirs — depth-1 glob + bench skip
# --------------------------------------------------------------------------

class TestLegacyResultDirsExtra:
    def test_depth1_glob_finds_legacy(self, tmp_path):
        # /root/run_pca/results.tsv (depth-1, `*/results.tsv`) with no task.yaml.
        d = tmp_path / "run_pca"
        d.mkdir()
        (d / "results.tsv").write_text("round\tstatus\n1\tkeep\n")
        out = reg._legacy_result_dirs([tmp_path], [], include_bench=True)
        assert d.resolve() in {p.resolve() for p in out}

    def test_bench_dir_skipped_when_excluded(self, tmp_path):
        # A legacy-named dir under bench_runs/ is dropped when include_bench=False.
        d = tmp_path / "bench_runs" / "run_pca"
        d.mkdir(parents=True)
        (d / "results.tsv").write_text("round\tstatus\n1\tkeep\n")
        out = reg._legacy_result_dirs([tmp_path], [], include_bench=False)
        assert d.resolve() not in {p.resolve() for p in out}

    def test_bench_dir_skipped_depth1_glob(self, tmp_path):
        # Pass the bench_runs dir itself as a root so `*/results.tsv` (depth-1)
        # matches `bench_runs/run_pca/results.tsv`; "bench_runs" in parts ->
        # _is_bench_task True -> skipped at the depth-1 glob's bench guard.
        bench_root = tmp_path / "bench_runs"
        d = bench_root / "run_pca"
        d.mkdir(parents=True)
        (d / "results.tsv").write_text("round\tstatus\n1\tkeep\n")
        out = reg._legacy_result_dirs([bench_root], [], include_bench=False)
        assert d.resolve() not in {p.resolve() for p in out}
        # included -> appears
        out2 = reg._legacy_result_dirs([bench_root], [], include_bench=True)
        assert d.resolve() in {p.resolve() for p in out2}

    def test_bench_dir_kept_when_included(self, tmp_path):
        d = tmp_path / "bench_runs" / "run_pca"
        d.mkdir(parents=True)
        (d / "results.tsv").write_text("round\tstatus\n1\tkeep\n")
        out = reg._legacy_result_dirs([tmp_path], [], include_bench=True)
        assert d.resolve() in {p.resolve() for p in out}

    def test_skips_dir_in_task_reals(self, tmp_path):
        # A legacy dir that IS a known task (passed via task_dirs) is excluded.
        d = tmp_path / "run_pca"
        d.mkdir()
        (d / "results.tsv").write_text("round\tstatus\n1\tkeep\n")
        out = reg._legacy_result_dirs([tmp_path], [d], include_bench=True)
        assert d.resolve() not in {p.resolve() for p in out}

    def test_non_dir_root_skipped(self, tmp_path):
        # A root that doesn't exist as a dir is skipped without error.
        assert reg._legacy_result_dirs([tmp_path / "ghost"], [],
                                       include_bench=True) == []


# --------------------------------------------------------------------------
# _write_detail — conditional detail lines + headings + related annotation
# --------------------------------------------------------------------------

class TestWriteDetailBranches:
    def test_all_optional_lines_and_headings(self, tmp_path):
        active = tmp_path / "active_opts.md"
        active.write_text("## Vectorized wilcoxon\nbody\n## Sparse path\n")
        entry = {
            "symbol": "Seurat::FindMarkers", "kind": "workflow_primitive",
            "status": "packaged", "task": "test_fam", "phase": "package",
            "speedup_pct": "30", "round": "5", "task_dir": "/d",
            "patch_path": "/d/patch.R",
            "active_opts": str(active),
            "discoveries": "/d/memory/discoveries.md",
        }
        related = [{"task": "sibling", "status": "optimized",
                    "speedup_pct": "12", "active_opts": "/s/active_opts.md"}]
        path = reg._write_detail(tmp_path, entry, related=related, evidence=None)
        text = Path(path).read_text()
        assert "Package patch:" in text
        assert "Active opts:" in text
        assert "Discoveries:" in text
        assert "Active Optimization Headings" in text
        assert "Vectorized wilcoxon" in text
        # related-task active_opts annotation (line 422)
        assert "active_opts=`/s/active_opts.md`" in text


# --------------------------------------------------------------------------
# cmd_registry_rebuild — dependency-map optimized_dependency promotion
# --------------------------------------------------------------------------

class TestRebuildDependencyPromotion:
    def _make_task(self, root, name, target_function, *, readme=""):
        d = root / name
        d.mkdir(parents=True)
        (d / "task.yaml").write_text(
            f"task: {name}\n"
            f"target_function: {target_function}\n"
            "target_repo: https://example.com/x\n"
        )
        (d / "results.tsv").write_text(
            "round\tdataset\tspeedup_pct\tstatus\n1\tt\t25\tkeep\n")
        if readme:
            (d / "README.md").write_text(readme)
        return d

    @pytest.fixture
    def fake_scan(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        fw = ws / "autozyme-framework"
        fw.mkdir(parents=True)
        # A task whose target_function IS the dependency symbol mgcv::gam, and
        # whose README mentions mgcv::gam so _dependency_hits fires. This makes
        # the dependency-map promotion branch (entry.symbol == dep symbol and
        # status in {packaged,scaled,optimized}) execute.
        t1 = self._make_task(ws, "test_gam", "mgcv::gam",
                             readme="This optimizes mgcv::gam fit internals.")
        monkeypatch.chdir(ws)
        monkeypatch.setattr(reg, "find_workspace_root", lambda cwd: ws)
        monkeypatch.setattr(reg, "find_framework_root", lambda cwd: fw)
        monkeypatch.setattr(reg, "find_tasks",
                            lambda roots, max_depth, framework_root: [t1])

        def fake_detect_phase(task_dir, framework):
            return {
                "phases": {
                    "init": {"done": True},
                    "iterate": {"done": True},
                    "scaling": {"done": False},
                    "package": {"done": False, "patch_path": ""},
                },
                "latest_keep": reg._latest_keep_from_results(
                    task_dir / "results.tsv"),
                "phase": "iterate",
                "task_name": task_dir.name,
            }
        monkeypatch.setattr(reg, "detect_phase", fake_detect_phase)
        return SimpleNamespace(ws=ws, fw=fw, registry_root=fw / "registry",
                               task=t1)

    def test_rebuild_promotes_dependency_to_optimized(self, fake_scan, capsys):
        args = SimpleNamespace(
            framework_root=str(fake_scan.fw), paths=[], registry_root=None,
            max_depth=3, include_bench=False)
        reg.cmd_registry_rebuild(args)
        out = capsys.readouterr().out
        assert "registry: wrote" in out
        entries = reg._load_entries(fake_scan.registry_root)
        gam = [e for e in entries if e["symbol"] == "mgcv::gam"]
        assert gam, "mgcv::gam entry should exist"
        # The direct target_function entry is optimized; because it equals the
        # dependency symbol AND is optimized, the dependency map promoted it
        # (and the dedup at the end drops the redundant dependency_bottleneck
        # row). The surviving entry carries an optimized-family status.
        assert any(e["status"] in {"optimized", "optimized_dependency"}
                   for e in gam)


# --------------------------------------------------------------------------
# cmd_registry_suggest — empty-bucket "none" + similar dedup
# --------------------------------------------------------------------------

class TestSuggestBranches:
    def test_suggest_all_none_buckets(self, tmp_path, monkeypatch, capsys):
        # A registry with one scaffold entry that matches the query text but is
        # neither optimized (direct) nor a dependency/engine nor has active_opts
        # -> all three buckets print "- none".
        root = tmp_path / "registry"
        reg._write_entries(root, [
            {"symbol": "widget_fn", "kind": "target_function",
             "status": "scaffold", "task": "test_widget",
             "tags": "widget,fn"},
        ])
        monkeypatch.setattr(reg, "find_framework_root", lambda cwd: None)
        # task dir provides the query text mentioning "widget".
        task = tmp_path / "test_widget"
        task.mkdir()
        (task / "task.yaml").write_text("target_function: widget_fn\n")
        (task / "README.md").write_text("optimize the widget_fn hotspot")
        args = SimpleNamespace(registry_root=str(root), task=str(task),
                               profile=None, limit=5)
        reg.cmd_registry_suggest(args)
        out = capsys.readouterr().out
        # Direct bucket is empty (scaffold), dep bucket empty (target_function,
        # not bottleneck/engine), but similar bucket: scaffold has no active_opts
        # -> "- none" there too.
        assert out.count("- none") == 3

    def test_suggest_similar_dedup(self, tmp_path, monkeypatch, capsys):
        # Two entries sharing the same active_opts path: only the first should
        # be printed under "Similar Task Active Opts" (dedup via `seen`).
        root = tmp_path / "registry"
        reg._write_entries(root, [
            {"symbol": "fn_a", "kind": "target_function", "status": "scaffold",
             "task": "ta", "tags": "shared", "active_opts": "/p/active.md"},
            {"symbol": "fn_b", "kind": "target_function", "status": "scaffold",
             "task": "tb", "tags": "shared", "active_opts": "/p/active.md"},
        ])
        monkeypatch.setattr(reg, "find_framework_root", lambda cwd: None)
        task = tmp_path / "t"
        task.mkdir()
        (task / "task.yaml").write_text("target_function: shared\n")
        (task / "README.md").write_text("the shared hotspot")
        args = SimpleNamespace(registry_root=str(root), task=str(task),
                               profile=None, limit=5)
        reg.cmd_registry_suggest(args)
        out = capsys.readouterr().out
        sim_section = out.split("Similar Task Active Opts", 1)[1]
        # Exactly one of fn_a/fn_b is printed (dedup on active_opts).
        printed = sum(1 for s in ("fn_a", "fn_b") if s in sim_section)
        assert printed == 1

    def test_suggest_similar_limit_break(self, tmp_path, monkeypatch, capsys):
        # Three entries each with a DISTINCT active_opts; --limit 2 means the
        # similar loop breaks after printing 2 (the `count >= limit` break).
        root = tmp_path / "registry"
        reg._write_entries(root, [
            {"symbol": f"fn_{i}", "kind": "target_function",
             "status": "scaffold", "task": f"t{i}", "tags": "shared",
             "active_opts": f"/p/active_{i}.md"}
            for i in range(3)
        ])
        monkeypatch.setattr(reg, "find_framework_root", lambda cwd: None)
        task = tmp_path / "t"
        task.mkdir()
        (task / "task.yaml").write_text("target_function: shared\n")
        (task / "README.md").write_text("the shared hotspot here")
        args = SimpleNamespace(registry_root=str(root), task=str(task),
                               profile=None, limit=2)
        reg.cmd_registry_suggest(args)
        out = capsys.readouterr().out
        sim_section = out.split("Similar Task Active Opts", 1)[1]
        printed = sum(1 for i in range(3) if f"fn_{i}" in sim_section)
        assert printed == 2  # capped by --limit, loop broke
