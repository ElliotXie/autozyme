"""Wave-4 coverage tests for zyme.scan_portability.

Targets REACHABLE lines not covered by test_scan_portability.py /
test_scan_portability_deep.py:
  - _collect_sources framework-root branch: packaged patch discovery (file
    layout -> Py __init__.py; dir layout -> patch.R) + test_ prefix stripping
  - _scan_file matched_kinds dedup (two same-kind rules on one line) +
    .zyme_mclapply skip on the raw_mclapply rule
  - _rel_to_task ValueError fallback (path outside task_dir)
  - scan_task rel_sources ValueError fallback (a source outside task_dir)

NOTE on the unreachable `_compute_verdict` line 325 ("No action required"):
by the time control reaches it, harmful/unguarded-crash/review have all
returned, all-guarded-crash returned MAC_BONUS_ONLY, and the preceding
`if not hits and not review` (line 322) is always true there. It is dead code
and documented, not tested.
"""
from __future__ import annotations

from pathlib import Path

import zyme.scan
from zyme.scan_portability import (
    _collect_sources,
    _rel_to_task,
    _scan_file,
    scan_task,
)


def _scaffold(tmp_path: Path, name: str = "test_foo") -> Path:
    task = tmp_path / name
    task.mkdir()
    (task / "task.yaml").write_text(f"task: {name}\n")
    return task


# --------------------------------------------------------------------------
# _collect_sources framework-root patch discovery
# --------------------------------------------------------------------------
class TestCollectSourcesFramework:
    def setup_method(self):
        zyme.scan._build_lifted_from_index.cache_clear()

    def test_py_patch_file_appended(self, tmp_path: Path):
        # task.yaml's `task:` is "test_foo" -> stripped to "foo"; a Py patch
        # at autozyme_py/.../foo/__init__.py is discovered and appended.
        ws = tmp_path / "ws"
        fw = ws / "autozyme-framework"
        task = ws / "test_foo"
        task.mkdir(parents=True)
        (task / "task.yaml").write_text("task: test_foo\n")
        (task / "pipeline").mkdir()
        (task / "pipeline" / "run.py").write_text("x = 1\n")
        d = fw / "autozyme_py" / "src" / "autozyme" / "foo"
        d.mkdir(parents=True)
        (d / "__init__.py").write_text("# joblib.Parallel(...)\n")
        srcs = _collect_sources(task, fw)
        assert any(p.name == "__init__.py" for p in srcs)

    def test_r_patch_via_lifted_index(self, tmp_path: Path):
        # Direct lookup `<patches>/bar.R` misses; the lifted-from index links
        # task "bar" to a folder-layout patch.R, which _find_patch returns as a
        # FILE path -> the is_file() branch appends it.
        ws = tmp_path / "ws"
        fw = ws / "autozyme-framework"
        task = ws / "test_bar"
        task.mkdir(parents=True)
        (task / "task.yaml").write_text("task: bar\n")
        (task / "pipeline").mkdir()
        (task / "pipeline" / "run.R").write_text("x <- 1\n")
        patch_dir = fw / "autozyme_r" / "inst" / "patches" / "bar"
        patch_dir.mkdir(parents=True)
        (patch_dir / "patch.R").write_text(
            "# Lifted from autozyme task `bar`\nx <- 1\n"
        )
        srcs = _collect_sources(task, fw)
        assert any(p.name == "patch.R" for p in srcs)

    def test_no_framework_patch_found(self, tmp_path: Path):
        # framework root present but no matching patch -> sources stay pipeline-only
        ws = tmp_path / "ws"
        fw = ws / "autozyme-framework"
        fw.mkdir(parents=True)
        task = ws / "test_none"
        task.mkdir(parents=True)
        (task / "task.yaml").write_text("task: none\n")
        (task / "pipeline").mkdir()
        (task / "pipeline" / "run.R").write_text("x <- 1\n")
        srcs = _collect_sources(task, fw)
        assert [p.name for p in srcs] == ["run.R"]


# --------------------------------------------------------------------------
# _scan_file matched_kinds dedup + .zyme_mclapply skip
# --------------------------------------------------------------------------
class TestScanFileDedup:
    def test_two_review_rules_one_line_dedup(self, tmp_path: Path):
        # A line matching BOTH joblib.Parallel( and multiprocessing.Pool(
        # (same `review` kind) yields exactly ONE hit (kind-dedup).
        f = tmp_path / "run.py"
        f.write_text("x = joblib.Parallel()(multiprocessing.Pool())\n")
        hits, _, _ = _scan_file(f, tmp_path)
        review = [h for h in hits if h.kind == "review"]
        assert len(review) == 1

    def test_zyme_mclapply_not_raw_mclapply(self, tmp_path: Path):
        # `.zyme_mclapply(` matches the mitigation rule but must NOT be flagged
        # by the raw_mclapply crash rule (explicit skip).
        f = tmp_path / "run.R"
        f.write_text(".zyme_mclapply(1:10, identity)\n")
        hits, _, mit = _scan_file(f, tmp_path)
        assert all(h.pattern != "raw_mclapply" for h in hits)
        assert any(m["pattern"] == "zyme_mclapply" for m in mit)

    def test_bare_mclapply_skipped_when_zyme_mclapply_on_line(self, tmp_path: Path):
        # A line where the bare-mclapply regex matches but `.zyme_mclapply` is
        # also present -> the raw_mclapply rule is explicitly skipped.
        f = tmp_path / "run.R"
        f.write_text("y <- mclapply(x); z <- .zyme_mclapply(w)\n")
        hits, _, mit = _scan_file(f, tmp_path)
        assert all(h.pattern != "raw_mclapply" for h in hits)
        assert any(m["pattern"] == "zyme_mclapply" for m in mit)

    def test_parallel_mclapply_not_double_counted(self, tmp_path: Path):
        # parallel::mclapply matches both the parallel-specific and the bare
        # raw_mclapply rule; the bare one is skipped, leaving one crash hit.
        f = tmp_path / "run.R"
        f.write_text("parallel::mclapply(1:10, identity, mc.cores=2)\n")
        hits, _, _ = _scan_file(f, tmp_path)
        crash = [h for h in hits if h.kind == "crash_on_win"]
        assert len(crash) == 1
        assert crash[0].pattern == "raw_parallel_mclapply"


# --------------------------------------------------------------------------
# _rel_to_task ValueError fallback
# --------------------------------------------------------------------------
class TestRelToTask:
    def test_inside_task(self, tmp_path: Path):
        task = tmp_path / "t"
        (task / "pipeline").mkdir(parents=True)
        f = task / "pipeline" / "run.R"
        f.write_text("x")
        assert _rel_to_task(f, task) == "pipeline/run.R"

    def test_outside_task_falls_back_to_name(self, tmp_path: Path):
        task = tmp_path / "t"
        task.mkdir()
        outside = tmp_path / "elsewhere" / "foo.R"
        outside.parent.mkdir()
        outside.write_text("x")
        # outside is not under task -> relative_to raises -> returns basename
        assert _rel_to_task(outside, task) == "foo.R"


# --------------------------------------------------------------------------
# scan_task rel_sources ValueError fallback (source outside task_dir)
# --------------------------------------------------------------------------
class TestScanTaskRelSourcesFallback:
    def setup_method(self):
        zyme.scan._build_lifted_from_index.cache_clear()

    def test_framework_patch_source_rendered_absolute(self, tmp_path: Path):
        # The packaged patch lives OUTSIDE the task dir, so its rel_source can't
        # be relative_to(task_dir) -> rendered as an absolute posix path.
        ws = tmp_path / "ws"
        fw = ws / "autozyme-framework"
        task = ws / "test_ext"
        task.mkdir(parents=True)
        (task / "task.yaml").write_text("task: ext\n")
        (task / "pipeline").mkdir()
        (task / "pipeline" / "run.py").write_text("x = 1\n")
        d = fw / "autozyme_py" / "src" / "autozyme" / "ext"
        d.mkdir(parents=True)
        (d / "__init__.py").write_text("x = 1\n")
        res = scan_task(task, framework_root=fw)
        # one source is the external patch, rendered as an absolute path
        assert any(s.startswith("/") and s.endswith("ext/__init__.py")
                   for s in res.sources)
