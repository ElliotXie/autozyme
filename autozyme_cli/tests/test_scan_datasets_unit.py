"""Unit tests for zyme.scan_datasets — dataset scope classification,
subdir/orphan accounting, cross-task aggregation, and the humanize/format
helpers. Filesystem scaffolds use tmp_path with a realistic workspace layout
(framework dir + datasets/ category dirs) so find_workspace_root resolves.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from zyme import scan_datasets as SD


# ---------------------------------------------------------------------------
# humanize_bytes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, expected", [
    (0, "0"),
    (-5, "0"),
    (500, "500B"),
    (1024, "1.0K"),
    (1536, "1.5K"),
    (10 * 1024, "10K"),
    (1024 ** 2, "1.0M"),
    (1024 ** 3, "1.0G"),
    (12 * 1024 ** 3, "12G"),
    (1024 ** 4, "1.0T"),
])
def test_humanize_bytes(n, expected):
    assert SD.humanize_bytes(n) == expected


# ---------------------------------------------------------------------------
# format_breakdown
# ---------------------------------------------------------------------------

def test_format_breakdown_all_zero():
    assert SD.format_breakdown({"data": 0, "reference_outputs": 0,
                                "upstream_repo": 0, "other": 0}) == "-"


def test_format_breakdown_skips_zeros():
    out = SD.format_breakdown({"data": 1024 ** 3, "reference_outputs": 0,
                              "upstream_repo": 1024 ** 2, "other": 0})
    assert out == "d:1.0G u:1.0M"


def test_format_breakdown_order():
    out = SD.format_breakdown({"data": 1024, "reference_outputs": 1024,
                              "upstream_repo": 1024, "other": 1024})
    assert out == "d:1.0K ro:1.0K u:1.0K o:1.0K"


# ---------------------------------------------------------------------------
# path_size_bytes
# ---------------------------------------------------------------------------

def test_path_size_bytes_file(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"\0" * 100)
    assert SD.path_size_bytes(f) == 100


def test_path_size_bytes_dir_recursive(tmp_path):
    d = tmp_path / "tree"
    (d / "sub").mkdir(parents=True)
    (d / "a.bin").write_bytes(b"\0" * 40)
    (d / "sub" / "b.bin").write_bytes(b"\0" * 60)
    assert SD.path_size_bytes(d) == 100


def test_path_size_bytes_missing_returns_zero(tmp_path):
    assert SD.path_size_bytes(tmp_path / "gone") == 0


def test_path_size_bytes_oserror_returns_zero(tmp_path, monkeypatch):
    # is_file raising OSError -> outer except -> 0 (lines 46-47).
    f = tmp_path / "x.bin"
    f.write_bytes(b"\0")
    monkeypatch.setattr(Path, "is_file",
                        lambda self: (_ for _ in ()).throw(OSError("stat fail")))
    assert SD.path_size_bytes(f) == 0


def test_path_size_bytes_dir_inner_stat_oserror(tmp_path, monkeypatch):
    # A leaf whose stat() raises is skipped (lines 43-44) -> partial sum.
    d = tmp_path / "tree"
    d.mkdir()
    (d / "good.bin").write_bytes(b"\0" * 10)
    (d / "bad.bin").write_bytes(b"\0" * 10)
    orig = Path.stat

    def flaky(self, *a, **k):
        if self.name == "bad.bin":
            raise OSError("no")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "stat", flaky)
    assert SD.path_size_bytes(d) == 10


# ---------------------------------------------------------------------------
# _is_inside
# ---------------------------------------------------------------------------

def test_is_inside_true(tmp_path):
    child = tmp_path / "a" / "b"
    assert SD._is_inside(child, tmp_path) is True


def test_is_inside_false(tmp_path):
    assert SD._is_inside(Path("/etc/passwd"), tmp_path) is False


def test_is_inside_same_path(tmp_path):
    assert SD._is_inside(tmp_path, tmp_path) is True


# ---------------------------------------------------------------------------
# _shared_roots
# ---------------------------------------------------------------------------

def test_shared_roots_none():
    assert SD._shared_roots(None) == []


def test_shared_roots_existing_only(tmp_path):
    (tmp_path / "single_cell").mkdir()
    (tmp_path / "bulk").mkdir()
    # spatial absent
    roots = SD._shared_roots(tmp_path)
    names = {r.name for r in roots}
    assert names == {"single_cell", "bulk"}


# ---------------------------------------------------------------------------
# _classify_scope
# ---------------------------------------------------------------------------

def test_classify_scope_missing(tmp_path):
    s = SD._classify_scope(None, exists=False, task_dir=tmp_path,
                           per_task_self=None, shared_dirs=[])
    assert s == "missing"


def test_classify_scope_local_inside_task(tmp_path):
    resolved = tmp_path / "data" / "x.h5ad"
    s = SD._classify_scope(resolved, exists=True, task_dir=tmp_path,
                           per_task_self=None, shared_dirs=[])
    assert s == "local"


def test_classify_scope_local_per_task(tmp_path):
    per_task = tmp_path / "per_task" / "mytask"
    resolved = per_task / "x.rds"
    s = SD._classify_scope(resolved, exists=True, task_dir=tmp_path / "other",
                           per_task_self=per_task, shared_dirs=[])
    assert s == "local"


def test_classify_scope_shared(tmp_path):
    shared = tmp_path / "single_cell"
    resolved = shared / "ifnb.rds"
    s = SD._classify_scope(resolved, exists=True, task_dir=tmp_path / "task",
                           per_task_self=None, shared_dirs=[shared])
    assert s == "shared"


def test_classify_scope_external(tmp_path):
    resolved = Path("/somewhere/else/data.h5ad")
    s = SD._classify_scope(resolved, exists=True, task_dir=tmp_path / "task",
                           per_task_self=None, shared_dirs=[tmp_path / "single_cell"])
    assert s == "external"


def test_classify_scope_task_beats_shared(tmp_path):
    # per_task_self check beats shared check (order matters).
    per_task = tmp_path / "datasets" / "per_task" / "t"
    shared = tmp_path / "datasets" / "single_cell"
    resolved = per_task / "x.rds"
    s = SD._classify_scope(resolved, exists=True, task_dir=tmp_path / "task",
                           per_task_self=per_task, shared_dirs=[shared])
    assert s == "local"


# ---------------------------------------------------------------------------
# _classify_subdir
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, label", [
    ("data", "data"),
    ("data_raw", "data"),
    ("reference_outputs", "reference_outputs"),
    ("reference_output_tiny", "reference_outputs"),
    ("upstream_repo", "upstream_repo"),
    ("memory", "other"),
    ("pipeline", "other"),
])
def test_classify_subdir(name, label):
    assert SD._classify_subdir(name) == label


# ---------------------------------------------------------------------------
# _is_python_venv
# ---------------------------------------------------------------------------

def test_is_python_venv_true(tmp_path):
    (tmp_path / "pyvenv.cfg").write_text("home = /usr\n")
    assert SD._is_python_venv(tmp_path) is True


def test_is_python_venv_false(tmp_path):
    assert SD._is_python_venv(tmp_path) is False


def test_is_python_venv_oserror_false(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "is_file",
                        lambda self: (_ for _ in ()).throw(OSError("x")))
    assert SD._is_python_venv(tmp_path) is False


# ---------------------------------------------------------------------------
# _task_subdir_breakdown
# ---------------------------------------------------------------------------

def test_task_subdir_breakdown_buckets(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.bin").write_bytes(b"\0" * 100)
    (tmp_path / "reference_outputs").mkdir()
    (tmp_path / "reference_outputs" / "r.bin").write_bytes(b"\0" * 50)
    (tmp_path / "upstream_repo").mkdir()
    (tmp_path / "upstream_repo" / "u.bin").write_bytes(b"\0" * 30)
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "m.bin").write_bytes(b"\0" * 20)
    (tmp_path / "task.yaml").write_bytes(b"\0" * 10)
    total, bd = SD._task_subdir_breakdown(tmp_path)
    assert bd["data"] == 100
    assert bd["reference_outputs"] == 50
    assert bd["upstream_repo"] == 30
    assert bd["other"] == 20 + 10  # memory dir + top-level file
    assert total == 210


def test_task_subdir_breakdown_skips_dotfiles(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "huge.bin").write_bytes(b"\0" * 9999)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.bin").write_bytes(b"\0" * 5)
    total, bd = SD._task_subdir_breakdown(tmp_path)
    assert total == 5
    assert bd["data"] == 5


def test_task_subdir_breakdown_data_symlink_followed(tmp_path):
    target = tmp_path / "store"
    target.mkdir()
    (target / "big.bin").write_bytes(b"\0" * 200)
    task = tmp_path / "task"
    task.mkdir()
    (task / "data").symlink_to(target)
    total, bd = SD._task_subdir_breakdown(task)
    assert bd["data"] == 200
    assert total == 200


def test_task_subdir_breakdown_iterdir_oserror(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "iterdir",
                        lambda self: (_ for _ in ()).throw(OSError("perm")))
    total, bd = SD._task_subdir_breakdown(tmp_path)
    assert total == 0
    assert all(v == 0 for v in bd.values())


def test_task_subdir_breakdown_file_stat_oserror(tmp_path, monkeypatch):
    (tmp_path / "top.txt").write_bytes(b"\0" * 10)
    orig = Path.stat

    def flaky(self, *a, **k):
        if self.name == "top.txt":
            raise OSError("no")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "stat", flaky)
    total, bd = SD._task_subdir_breakdown(tmp_path)
    # stat failed -> size 0 counted into "other"
    assert bd["other"] == 0


def test_task_subdir_breakdown_other_symlink_skipped(tmp_path):
    target = tmp_path / "fw"
    target.mkdir()
    (target / "big.bin").write_bytes(b"\0" * 200)
    task = tmp_path / "task"
    task.mkdir()
    (task / "autozyme-framework").symlink_to(target)
    total, bd = SD._task_subdir_breakdown(task)
    assert total == 0


# ---------------------------------------------------------------------------
# _detect_data_orphans
# ---------------------------------------------------------------------------

def test_detect_orphans_no_data_dir(tmp_path):
    assert SD._detect_data_orphans(tmp_path, set()) == []


def test_detect_orphans_file_and_dir(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "declared.h5ad").write_bytes(b"\0" * 10)
    (data / "orphan.bin").write_bytes(b"\0" * 100)
    extra = data / "extra_dir"
    extra.mkdir()
    (extra / "a.bin").write_bytes(b"\0" * 50)
    (extra / "b.bin").write_bytes(b"\0" * 50)
    declared = {str((data / "declared.h5ad").resolve())}
    orphans = SD._detect_data_orphans(tmp_path, declared)
    paths = {Path(o["path"]).name for o in orphans}
    assert "declared.h5ad" not in paths
    assert "orphan.bin" in paths
    assert "extra_dir" in paths
    # sorted by size desc: extra_dir (100) before orphan.bin (100)? equal -> stable.
    dir_rec = next(o for o in orphans if Path(o["path"]).name == "extra_dir")
    assert dir_rec["kind"] == "dir"
    assert dir_rec["n_files"] == 2
    assert dir_rec["size_bytes"] == 100
    file_rec = next(o for o in orphans if Path(o["path"]).name == "orphan.bin")
    assert file_rec["kind"] == "file"
    assert file_rec["n_files"] == 1


def test_detect_orphans_skips_raw_and_dotfiles(tmp_path):
    data = tmp_path / "data"
    (data / "_raw").mkdir(parents=True)
    (data / "_raw" / "big.bin").write_bytes(b"\0" * 999)
    (data / ".hidden").write_bytes(b"\0" * 50)
    (data / "orphan.bin").write_bytes(b"\0" * 10)
    orphans = SD._detect_data_orphans(tmp_path, set())
    names = {Path(o["path"]).name for o in orphans}
    assert names == {"orphan.bin"}


def test_detect_orphans_skips_symlinks(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "x.bin").write_bytes(b"\0" * 100)
    data = tmp_path / "data"
    data.mkdir()
    (data / "link").symlink_to(target)
    orphans = SD._detect_data_orphans(tmp_path, set())
    assert orphans == []


def test_detect_orphans_covers_declared_parent(tmp_path):
    # declared is data/foo/file.rds -> the foo/ dir is "covered", not an orphan.
    data = tmp_path / "data"
    foo = data / "foo"
    foo.mkdir(parents=True)
    (foo / "file.rds").write_bytes(b"\0" * 10)
    declared = {str((foo / "file.rds").resolve())}
    orphans = SD._detect_data_orphans(tmp_path, declared)
    assert orphans == []


def test_detect_orphans_iterdir_oserror(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(Path, "iterdir",
                        lambda self: (_ for _ in ()).throw(OSError("perm")))
    assert SD._detect_data_orphans(tmp_path, set()) == []


def test_detect_orphans_covers_resolve_oserror(tmp_path, monkeypatch):
    # _covers_a_declared: entry.resolve() raising is swallowed (lines 221-222).
    data = tmp_path / "data"
    data.mkdir()
    (data / "orphan.bin").write_bytes(b"\0" * 10)
    orig = Path.resolve

    def flaky(self, *a, **k):
        if self.name == "orphan.bin":
            raise OSError("loop")
        return orig(self, *a, **k)

    monkeypatch.setattr(Path, "resolve", flaky)
    orphans = SD._detect_data_orphans(tmp_path, set())
    # still reported (display falls back to str(entry))
    assert len(orphans) == 1


def test_detect_orphans_venv_kind(tmp_path):
    data = tmp_path / "data"
    venv = data / "myenv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home=/usr\n")
    (venv / "lib.bin").write_bytes(b"\0" * 5)
    orphans = SD._detect_data_orphans(tmp_path, set())
    rec = next(o for o in orphans if Path(o["path"]).name == "myenv")
    assert rec["kind"] == "venv"


# ---------------------------------------------------------------------------
# inspect_task_datasets — end to end on a scaffold workspace
# ---------------------------------------------------------------------------

def _make_workspace(tmp_path):
    """Build root/ with a real autozyme-framework dir + datasets/ category dirs.

    Returns (root, category_dir). find_workspace_root walks up from a task and
    needs root/autozyme-framework to be a real dir whose resolved parent == root.
    """
    root = tmp_path / "ws"
    (root / "autozyme-framework").mkdir(parents=True)
    datasets = root / "datasets"
    (datasets / "single_cell").mkdir(parents=True)
    (datasets / "spatial").mkdir()
    (datasets / "bulk").mkdir()
    category = root / "general_bio"
    category.mkdir()
    return root, category


def test_inspect_task_local_dataset(tmp_path):
    root, category = _make_workspace(tmp_path)
    task = category / "test_foo"
    (task / "data").mkdir(parents=True)
    (task / "data" / "tiny.h5ad").write_bytes(b"\0" * 100)
    (task / "task.yaml").write_text(
        "target_function: f\n\ndatasets:\n"
        "  - {tier: tiny, name: tiny_a, path: data/tiny.h5ad}\n"
        "metrics:\n  - {name: speedup, comparator: gte, threshold: 1.0}\n"
    )
    res = SD.inspect_task_datasets(task)
    assert res["dir_name"] == "test_foo"
    assert res["category"] == "general_bio"
    assert len(res["datasets"]) == 1
    ds = res["datasets"][0]
    assert ds["scope"] == "local"
    assert ds["exists"] is True
    assert ds["size_bytes"] == 100
    assert res["totals"]["local_count"] == 1
    assert res["totals"]["local_bytes"] == 100


def test_inspect_task_shared_dataset(tmp_path):
    root, category = _make_workspace(tmp_path)
    shared_file = root / "datasets" / "single_cell" / "ifnb.rds"
    shared_file.write_bytes(b"\0" * 250)
    task = category / "test_bar"
    task.mkdir(parents=True)
    (task / "task.yaml").write_text(
        "target_function: f\n\ndatasets:\n"
        f"  - {{tier: tiny, name: t, path: {shared_file}}}\n"
        "metrics:\n  - {name: speedup, comparator: gte, threshold: 1.0}\n"
    )
    res = SD.inspect_task_datasets(task)
    assert res["datasets"][0]["scope"] == "shared"
    assert res["totals"]["shared_count"] == 1
    assert res["totals"]["shared_bytes"] == 250


def test_inspect_task_missing_dataset(tmp_path):
    root, category = _make_workspace(tmp_path)
    task = category / "test_missing"
    task.mkdir(parents=True)
    (task / "task.yaml").write_text(
        "target_function: f\n\ndatasets:\n"
        "  - {tier: tiny, name: t, path: data/nope.h5ad}\n"
        "metrics:\n  - {name: speedup, comparator: gte, threshold: 1.0}\n"
    )
    res = SD.inspect_task_datasets(task)
    assert res["datasets"][0]["scope"] == "missing"
    assert res["datasets"][0]["exists"] is False
    assert res["totals"]["missing_count"] == 1


def test_inspect_task_external_dataset(tmp_path):
    root, category = _make_workspace(tmp_path)
    ext = tmp_path / "outside_ws"
    ext.mkdir()
    ext_file = ext / "weird.h5ad"
    ext_file.write_bytes(b"\0" * 70)
    task = category / "test_ext"
    task.mkdir(parents=True)
    (task / "task.yaml").write_text(
        "target_function: f\n\ndatasets:\n"
        f"  - {{tier: tiny, name: t, path: {ext_file}}}\n"
        "metrics:\n  - {name: speedup, comparator: gte, threshold: 1.0}\n"
    )
    res = SD.inspect_task_datasets(task)
    assert res["datasets"][0]["scope"] == "external"
    assert res["totals"]["external_count"] == 1


def test_inspect_task_detects_orphan(tmp_path):
    root, category = _make_workspace(tmp_path)
    task = category / "test_orphan"
    (task / "data").mkdir(parents=True)
    (task / "data" / "declared.h5ad").write_bytes(b"\0" * 10)
    (task / "data" / "forgotten.bin").write_bytes(b"\0" * 500)
    (task / "task.yaml").write_text(
        "target_function: f\n\ndatasets:\n"
        "  - {tier: tiny, name: t, path: data/declared.h5ad}\n"
        "metrics:\n  - {name: speedup, comparator: gte, threshold: 1.0}\n"
    )
    res = SD.inspect_task_datasets(task)
    assert res["totals"]["orphan_count"] == 1
    assert res["orphans"][0]["size_bytes"] == 500
    assert Path(res["orphans"][0]["path"]).name == "forgotten.bin"


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------

def _row(dir_name, category, datasets, orphans=None, totals=None):
    return {
        "dir_name": dir_name, "category": category,
        "datasets": datasets, "orphans": orphans or [],
        "totals": totals or {},
    }


def test_aggregate_shared_collapses_by_path():
    rows = [
        _row("t1", "bio", [{"scope": "shared", "path": "/d/ifnb.rds",
                            "size_bytes": 100, "tier": "tiny", "name": "a"}]),
        _row("t2", "bio", [{"scope": "shared", "path": "/d/ifnb.rds",
                            "size_bytes": 100, "tier": "medium", "name": "b"}]),
        _row("t3", "bio", [{"scope": "shared", "path": "/d/other.rds",
                            "size_bytes": 300, "tier": "tiny", "name": "c"}]),
    ]
    agg = SD.aggregate_shared(rows)
    # sorted by size desc: other.rds (300) first
    assert agg[0]["path"] == "/d/other.rds"
    ifnb = next(a for a in agg if a["path"] == "/d/ifnb.rds")
    assert len(ifnb["references"]) == 2
    assert {r["task"] for r in ifnb["references"]} == {"t1", "t2"}


def test_aggregate_external_filters_scope():
    rows = [
        _row("t1", "bio", [
            {"scope": "external", "path": "/x", "size_bytes": 10, "tier": "tiny", "name": "a"},
            {"scope": "shared", "path": "/y", "size_bytes": 10, "tier": "tiny", "name": "b"},
        ]),
    ]
    agg = SD.aggregate_external(rows)
    assert len(agg) == 1
    assert agg[0]["path"] == "/x"


def test_collect_missing():
    rows = [
        _row("t1", "bio", [
            {"scope": "missing", "path": "/m", "tier": "tiny", "name": "a"},
            {"scope": "local", "path": "/l", "tier": "tiny", "name": "b"},
        ]),
    ]
    miss = SD.collect_missing(rows)
    assert len(miss) == 1
    assert miss[0]["task"] == "t1"
    assert miss[0]["path"] == "/m"


def test_collect_orphans_sorted():
    rows = [
        _row("t1", "bio", [], orphans=[{"path": "/a", "size_bytes": 10}]),
        _row("t2", "bio", [], orphans=[{"path": "/b", "size_bytes": 99}]),
    ]
    orphans = SD.collect_orphans(rows)
    assert [o["size_bytes"] for o in orphans] == [99, 10]
    assert orphans[0]["task"] == "t2"


def test_aggregate_totals():
    rows = [
        _row("t1", "bio",
             [{"scope": "shared", "path": "/s", "size_bytes": 100, "tier": "tiny", "name": "a"}],
             totals={"n_entries": 1, "local_count": 0, "local_bytes": 0,
                     "shared_count": 1, "shared_bytes": 100,
                     "external_count": 0, "external_bytes": 0,
                     "missing_count": 0, "task_dir_bytes": 100,
                     "orphan_count": 0, "orphan_bytes": 0,
                     "subdir": {"data": 100, "reference_outputs": 0,
                                "upstream_repo": 0, "other": 0}}),
        _row("t2", "bio",
             [{"scope": "shared", "path": "/s", "size_bytes": 100, "tier": "tiny", "name": "b"}],
             totals={"n_entries": 1, "local_count": 0, "local_bytes": 0,
                     "shared_count": 1, "shared_bytes": 100,
                     "external_count": 0, "external_bytes": 0,
                     "missing_count": 0, "task_dir_bytes": 5,
                     "orphan_count": 0, "orphan_bytes": 0,
                     "subdir": {"data": 5, "reference_outputs": 0,
                                "upstream_repo": 0, "other": 0}}),
    ]
    out = SD.aggregate_totals(rows)
    assert out["n_tasks"] == 2
    assert out["shared_count"] == 2          # raw refs summed
    assert out["shared_bytes"] == 200        # raw refs summed
    assert out["shared_unique_paths"] == 1   # same /s path
    assert out["shared_unique_bytes"] == 100  # counted once
    assert out["subdir"]["data"] == 105


def test_aggregate_totals_empty():
    out = SD.aggregate_totals([])
    assert out["n_tasks"] == 0
    assert out["shared_unique_paths"] == 0
    assert out["subdir"]["data"] == 0
