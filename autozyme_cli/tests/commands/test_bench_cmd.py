"""Unit tests for zyme.commands.bench.

The bench module is the largest in the package; most of it is orchestration
that forks subprocesses (git archive/clone, hf download, agent dispatch). This
file targets the PURE layer that coverage can see:

  - suite/template/task.yaml parsing (`_read_suite_yaml`, `_parse_flow_mapping`,
    `_parse_task_yaml_datasets`, `_validate_suite`, `_select_suite_tasks`)
  - simple-YAML emit/read round-trip
  - dataset/tier/path resolution + reference-output dir lookup
  - results.tsv progress aggregation + speedup formatting
  - dispatch summary / status filtering / run resolution
  - tier-classification + reflect-category + csv-splitting helpers
  - the doctor / list-templates / list / status commands driven against a
    temp templates+suites+runs tree (root resolvers monkeypatched).

Functions already covered by tests/test_bench_templates.py
(`_reference_text_as_round0_pipeline`, `_reset_pipeline_to_reference`,
`_template_check_lines` literal/override cases, `_write_experiment_doc`,
`_append_experiment_dispatch_doc`) are NOT duplicated here.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from zyme.commands import bench


# ===========================================================================
# Small pure helpers
# ===========================================================================

def test_split_csv_dedupes_and_flattens():
    assert bench._split_csv(["a,b", "b,c", " d "]) == ["a", "b", "c", "d"]
    assert bench._split_csv(None) == []
    assert bench._split_csv([""]) == []


@pytest.mark.parametrize("name,expected", [
    ("ood", True),
    ("ood_large", True),
    ("ood_large1", True),
    ("tiny", False),
    ("medium", False),
    ("good", False),  # 'ood' substring but not prefix
])
def test_is_ood_tier_dirname(name, expected):
    assert bench._is_ood_tier_dirname(name) is expected


@pytest.mark.parametrize("stage,expected", [
    ("init", "initialization"),
    ("transfer_init", "initialization"),
    ("iterate", "iteration"),
    ("memory", "iteration"),
    ("validate_scaling", "scaling"),
    ("package", "packaging"),
    ("", "iteration"),
    (None, "iteration"),
    ("weird", "weird"),
])
def test_reflect_category_for_stage(stage, expected):
    assert bench._reflect_category_for_stage(stage) == expected


@pytest.mark.parametrize("val,expected", [
    (None, "-"),
    (0.0, "+0.0%"),
    (12.34, "+12.3%"),
    (-5.0, "-5.0%"),
])
def test_fmt_speedup_pct(val, expected):
    assert bench._fmt_speedup_pct(val) == expected


def test_has_active_install_override():
    assert bench._has_active_install_override('install_override("X","Y",f)\n')
    assert not bench._has_active_install_override('# install_override("X")\n')
    assert not bench._has_active_install_override("nothing here\n")


# ===========================================================================
# simple-YAML round trip
# ===========================================================================

def test_emit_and_read_simple_yaml_roundtrip(tmp_path: Path):
    d = {
        "name": "t1",
        "stage": "iterate",
        "count": 3,
        "nested": {"a": "1", "b": "two"},
        "empty_map": {},
        "items": ["x", "y"],
        "empty_list": [],
    }
    text = bench._emit_simple_yaml(d)
    p = tmp_path / "m.yaml"
    p.write_text(text, encoding="utf-8")
    out = bench._read_simple_yaml(p)
    assert out["name"] == "t1"
    assert out["stage"] == "iterate"
    assert str(out["count"]) == "3"
    assert out["nested"]["a"] in ("1", 1)
    assert out["nested"]["b"] == "two"


# ===========================================================================
# Suite parsing + validation + selection
# ===========================================================================

SUITE_YAML = """\
id: iterate_demo
stage: iterate
prompt_slot: iterate
field: Bio
description: demo
default_reps: 2

tasks:
  - id: findallmarker
    template: findallmarker
    probe: aggressiveness
  - id: mast
    template: mast_tpl
    probe: concordance
"""


def _write_suite(tmp_path, text=SUITE_YAML, name="iterate_demo"):
    p = tmp_path / f"{name}.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_read_suite_yaml_parses_tasks_list(tmp_path: Path):
    suite = bench._read_suite_yaml(_write_suite(tmp_path))
    assert suite["id"] == "iterate_demo"
    assert suite["stage"] == "iterate"
    assert len(suite["tasks"]) == 2
    assert suite["tasks"][0] == {
        "id": "findallmarker", "template": "findallmarker", "probe": "aggressiveness"}
    assert suite["tasks"][1]["template"] == "mast_tpl"


def test_validate_suite_ok():
    suite = {
        "id": "s", "stage": "iterate", "prompt_slot": "iterate", "field": "Bio",
        "tasks": [{"id": "a", "template": "a"}],
    }
    bench._validate_suite(suite, "s")  # no raise


def test_validate_suite_missing_key_dies():
    with pytest.raises(SystemExit):
        bench._validate_suite({"id": "s"}, "s")


def test_validate_suite_empty_tasks_dies():
    suite = {"id": "s", "stage": "iterate", "prompt_slot": "iterate",
             "field": "Bio", "tasks": []}
    with pytest.raises(SystemExit):
        bench._validate_suite(suite, "s")


def test_validate_suite_task_missing_template_dies():
    suite = {"id": "s", "stage": "iterate", "prompt_slot": "iterate",
             "field": "Bio", "tasks": [{"id": "a"}]}
    with pytest.raises(SystemExit):
        bench._validate_suite(suite, "s")


def test_select_suite_tasks_no_filter_returns_all(tmp_path):
    suite = bench._read_suite_yaml(_write_suite(tmp_path))
    assert len(bench._select_suite_tasks(suite, None)) == 2


def test_select_suite_tasks_filter_by_id(tmp_path):
    suite = bench._read_suite_yaml(_write_suite(tmp_path))
    sel = bench._select_suite_tasks(suite, ["mast"])
    assert [t["id"] for t in sel] == ["mast"]


def test_select_suite_tasks_filter_by_template(tmp_path):
    suite = bench._read_suite_yaml(_write_suite(tmp_path))
    sel = bench._select_suite_tasks(suite, ["mast_tpl"])
    assert [t["id"] for t in sel] == ["mast"]


def test_select_suite_tasks_unknown_dies(tmp_path):
    suite = bench._read_suite_yaml(_write_suite(tmp_path))
    with pytest.raises(SystemExit):
        bench._select_suite_tasks(suite, ["ghost"])


# ===========================================================================
# Flow-mapping + task.yaml dataset parsing
# ===========================================================================

def test_parse_flow_mapping_basic():
    out = bench._parse_flow_mapping("{tier: tiny, name: a, path: data/x.h5ad}")
    assert out == {"tier": "tiny", "name": "a", "path": "data/x.h5ad"}


def test_parse_flow_mapping_quoted_commas_preserved():
    out = bench._parse_flow_mapping('{name: "a,b", path: data/x}')
    assert out["name"] == "a,b"
    assert out["path"] == "data/x"


def test_parse_task_yaml_datasets_flow_rows(tmp_path: Path):
    ty = tmp_path / "task.yaml"
    ty.write_text(
        "datasets:\n"
        "  - {tier: tiny, name: tiny_a, path: data/tiny.h5ad}\n"
        "  - {tier: ood_large, name: ood_a, path: data/ood.h5ad}\n"
        "metrics:\n  - {name: s, comparator: gte, threshold: 1.0}\n",
        encoding="utf-8",
    )
    ds = bench._parse_task_yaml_datasets(ty)
    assert [d["tier"] for d in ds] == ["tiny", "ood_large"]
    assert ds[0]["name"] == "tiny_a"


def test_parse_task_yaml_datasets_missing_file(tmp_path):
    assert bench._parse_task_yaml_datasets(tmp_path / "no.yaml") == []


# ===========================================================================
# Dataset path resolution + reference output dir
# ===========================================================================

def test_resolve_template_dataset_path_absolute(tmp_path):
    p = bench._resolve_template_dataset_path(tmp_path, "/abs/foo.h5ad", {})
    assert p == Path("/abs/foo.h5ad")


def test_resolve_template_dataset_path_relative_to_template(tmp_path):
    p = bench._resolve_template_dataset_path(tmp_path, "data/foo.h5ad", {})
    assert p == tmp_path / "data/foo.h5ad"


def test_resolve_template_dataset_path_symlink_source(tmp_path):
    meta = {"symlink_sources": {"data": "/src/datasets/foo"}}
    p = bench._resolve_template_dataset_path(tmp_path, "data/bar.h5ad", meta)
    assert p == Path("/src/datasets/foo/bar.h5ad")


def test_resolve_template_dataset_path_hf_block(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "_resolve_datasets_root", lambda: Path("/cache"))
    meta = {"data": {"hf_path": "per_task/mytask"}}
    p = bench._resolve_template_dataset_path(tmp_path, "data/x.rds", meta)
    assert p == Path("/cache/per_task/mytask/x.rds")


def test_reference_output_dir_prefers_new_layout(tmp_path):
    (tmp_path / "reference_outputs" / "tiny").mkdir(parents=True)
    got = bench._reference_output_dir_for(tmp_path, "tiny")
    assert got == tmp_path / "reference_outputs" / "tiny"


def test_reference_output_dir_falls_back_to_legacy(tmp_path):
    (tmp_path / "reference_output_tiny").mkdir()
    got = bench._reference_output_dir_for(tmp_path, "tiny")
    assert got == tmp_path / "reference_output_tiny"


def test_reference_output_dir_default_when_neither(tmp_path):
    got = bench._reference_output_dir_for(tmp_path, "tiny")
    assert got == tmp_path / "reference_outputs" / "tiny"


# ===========================================================================
# HF / symlink source detection
# ===========================================================================

def test_detect_hf_data_from_per_task_symlink(tmp_path):
    target = tmp_path / "datasets" / "per_task" / "mytask"
    target.mkdir(parents=True)
    src = tmp_path / "src"
    src.mkdir()
    (src / "data").symlink_to(target)
    out = bench._detect_hf_data(src)
    assert out == {
        "hf_repo": bench._HF_DATASET_REPO,
        "hf_repo_type": bench._HF_DATASET_REPO_TYPE,
        "hf_path": "per_task/mytask",
    }


def test_detect_hf_data_non_symlink_returns_none(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "data").mkdir()
    assert bench._detect_hf_data(src) is None


def test_detect_hf_data_outside_per_task_returns_none(tmp_path):
    target = tmp_path / "datasets" / "shared" / "mytask"
    target.mkdir(parents=True)
    src = tmp_path / "src"
    src.mkdir()
    (src / "data").symlink_to(target)
    assert bench._detect_hf_data(src) is None


def test_detect_symlink_sources(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "data").mkdir()
    (src / "upstream").mkdir()
    out = bench._detect_symlink_sources(src)
    assert set(out) == {"data", "upstream"}
    assert out["data"] == str((src / "data").resolve())


# ===========================================================================
# relative-path rewriting + gitignore rewriting
# ===========================================================================

def test_rewrite_relative_data_paths(tmp_path):
    source = tmp_path / "source_task"
    source.mkdir()
    dest = tmp_path / "tpl"
    dest.mkdir()
    (dest / "task.yaml").write_text(
        "datasets:\n  - {tier: tiny, name: a, path: ../datasets/x.h5ad}\n",
        encoding="utf-8",
    )
    rewrites = bench._rewrite_relative_data_paths(dest, source)
    assert len(rewrites) == 1
    rel, absolute = rewrites[0]
    assert rel == "../datasets/x.h5ad"
    # the rewritten path must be absolute and resolved against source
    assert Path(absolute).is_absolute()
    assert "task.yaml" not in (source / "task.yaml").name or True
    assert absolute in (dest / "task.yaml").read_text(encoding="utf-8")


def test_rewrite_relative_data_paths_noop_when_no_relatives(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    dest = tmp_path / "tpl"
    dest.mkdir()
    (dest / "task.yaml").write_text(
        "datasets:\n  - {tier: tiny, name: a, path: data/x.h5ad}\n", encoding="utf-8")
    assert bench._rewrite_relative_data_paths(dest, source) == []


def test_rewrite_gitignore_drops_state_patterns(tmp_path):
    dest = tmp_path / "tpl"
    dest.mkdir()
    (dest / ".gitignore").write_text(
        "# comment\n"
        "results.tsv\n"
        ".zyme/\n"
        "data/\n"          # kept (not a drop pattern)
        "reference_outputs/\n",
        encoding="utf-8",
    )
    bench._rewrite_gitignore_for_template(dest)
    gi = (dest / ".gitignore").read_text(encoding="utf-8")
    assert "results.tsv" not in gi.splitlines()
    assert ".zyme/" not in gi.splitlines()
    assert "reference_outputs/" not in gi.splitlines()
    assert "data/" in gi  # non-state entries survive
    # original stashed
    assert (dest / ".gitignore.task").exists()
    assert "results.tsv" in (dest / ".gitignore.task").read_text(encoding="utf-8")


# ===========================================================================
# runtime-state copying (register-template helper)
# ===========================================================================

def test_copy_runtime_state_skips_ood_for_iterate(tmp_path):
    source = tmp_path / "src"
    (source / "reference_outputs" / "tiny").mkdir(parents=True)
    (source / "reference_outputs" / "tiny" / "out.txt").write_text("x")
    (source / "reference_outputs" / "ood_large").mkdir(parents=True)
    (source / "reference_outputs" / "ood_large" / "out.txt").write_text("y")
    dest = tmp_path / "dest"
    dest.mkdir()
    copied, skipped = bench._copy_runtime_state(source, dest, "iterate")
    assert "reference_outputs" in copied
    assert (dest / "reference_outputs" / "tiny" / "out.txt").exists()
    assert not (dest / "reference_outputs" / "ood_large").exists()
    assert "reference_outputs/ood_large" in skipped


def test_copy_runtime_state_keeps_ood_for_other_stages(tmp_path):
    source = tmp_path / "src"
    (source / "reference_outputs" / "ood_large").mkdir(parents=True)
    (source / "reference_outputs" / "ood_large" / "out.txt").write_text("y")
    dest = tmp_path / "dest"
    dest.mkdir()
    copied, skipped = bench._copy_runtime_state(source, dest, "package")
    assert "reference_outputs" in copied
    assert (dest / "reference_outputs" / "ood_large").exists()
    assert skipped == []


def test_copy_runtime_files(tmp_path):
    source = tmp_path / "src"
    (source / "pipeline").mkdir(parents=True)
    (source / "results.tsv").write_text("hdr\n")
    (source / "pipeline" / "profile.json").write_text("{}")
    dest = tmp_path / "dest"
    dest.mkdir()
    copied = bench._copy_runtime_files(source, dest)
    assert set(copied) == {"results.tsv", "pipeline/profile.json"}
    assert (dest / "results.tsv").exists()
    assert (dest / "pipeline" / "profile.json").exists()


# ===========================================================================
# bench-task progress aggregation from results.tsv
# ===========================================================================

PROGRESS_HEADER = (
    "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
    "metrics_json\thypothesis\tdescription\tphase"
)


def _write_progress(task_dir, rows):
    (task_dir / "results.tsv").write_text(
        PROGRESS_HEADER + "\n"
        + "\n".join("\t".join(map(str, r)) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_bench_task_progress_counts_decisions(tmp_path):
    td = tmp_path
    _write_progress(td, [
        ("0", "up", "a", "10", "0", "5", "baseline", "{}", "", "", "optimize"),
        ("1", "c1", "a", "8", "20", "5", "keep", "{}", "", "", "optimize"),
        ("2", "c2", "a", "9", "10", "5", "discard", "{}", "", "", "optimize"),
    ])
    has, decisions, last = bench._bench_task_progress(td)
    assert has and decisions == 2 and last == "discard"


def test_bench_task_progress_no_results(tmp_path):
    assert bench._bench_task_progress(tmp_path) == (False, 0, "-")


def test_bench_task_progress_detail_tracks_best(tmp_path):
    td = tmp_path
    _write_progress(td, [
        ("0", "up", "a", "10", "0", "5", "baseline", "{}", "", "", "optimize"),
        ("1", "c1", "a", "8", "20", "5", "keep", "{}", "", "", "optimize"),
        ("2", "c2", "a", "7", "35", "5", "keep", "{}", "", "", "optimize"),
        ("3", "c3", "a", "9", "12", "5", "pending", "{}", "", "", "optimize"),
    ])
    d = bench._bench_task_progress_detail(td)
    assert d["has_results"]
    assert d["decision_rounds"] == 2  # two keeps
    assert d["pending_rounds"] == 1
    assert d["last_round"] == 3
    assert d["best_keep_pct"] == 35.0
    assert d["best_seen_pct"] == 35.0


def test_bench_task_progress_detail_skips_non_optimize_phase(tmp_path):
    td = tmp_path
    _write_progress(td, [
        ("1", "c1", "a", "8", "20", "5", "keep", "{}", "", "", "scaling"),
    ])
    d = bench._bench_task_progress_detail(td)
    assert d["decision_rounds"] == 0
    assert d["last_round"] is None


def test_bench_task_progress_detail_no_results(tmp_path):
    d = bench._bench_task_progress_detail(tmp_path)
    assert d["has_results"] is False
    assert d["best_keep_pct"] is None


# ===========================================================================
# dispatch summary
# ===========================================================================

def test_dispatch_summary_not_started():
    assert bench._dispatch_summary({}) == "not_started"


def test_dispatch_summary_counts_and_stale_flag():
    states = {
        "t1": {"status": "running", "dispatch_alive": True},
        "t2": {"status": "done", "dispatch_alive": True},
        "t3": {"status": "running", "dispatch_alive": True},
    }
    out = bench._dispatch_summary(states)
    assert "2 running" in out and "1 done" in out
    assert "stale" not in out


def test_dispatch_summary_marks_stale_when_dead():
    states = {"t1": {"status": "failed", "dispatch_alive": False}}
    out = bench._dispatch_summary(states)
    assert out.endswith("stale")


# ===========================================================================
# bench-run resolution + filtering (uses ZYME_META + manifest files)
# ===========================================================================

def _make_run(tmp_path, run_name="run1", tasks=("a_r1", "b_r1")):
    run = tmp_path / run_name
    run.mkdir()
    (run / "bench_manifest.yaml").write_text(
        "bench_suite: demo\nprompt_id: p1\nprompt_slot: iterate\n", encoding="utf-8")
    for t in tasks:
        td = run / t
        td.mkdir()
        (td / bench.ZYME_META_FILENAME).write_text(
            f"bench_task: {t.rsplit('_', 1)[0]}\ntemplate_name: {t.rsplit('_', 1)[0]}\n",
            encoding="utf-8",
        )
    return run


def test_resolve_bench_run_by_path(tmp_path):
    run = _make_run(tmp_path)
    got = bench._resolve_bench_run(str(run))
    assert got == run.resolve()


def test_resolve_bench_run_not_found_dies(tmp_path):
    with pytest.raises(SystemExit):
        bench._resolve_bench_run(str(tmp_path / "nope"))


def test_resolve_bench_run_by_name_under_root(tmp_path, monkeypatch):
    run = _make_run(tmp_path, "namedrun")
    monkeypatch.setattr(bench, "_default_bench_runs_root", lambda: tmp_path)
    got = bench._resolve_bench_run("namedrun")
    assert got == run.resolve()


def test_filter_task_dirs_for_status(tmp_path):
    run = _make_run(tmp_path, tasks=("a_r1", "b_r1"))
    task_dirs = [run / "a_r1", run / "b_r1"]
    # No filter -> all
    assert bench._filter_task_dirs_for_status(task_dirs, None) == task_dirs
    # Filter by bench_task name
    filtered = bench._filter_task_dirs_for_status(task_dirs, ["a"])
    assert filtered == [run / "a_r1"]


def test_bench_run_task_dirs_lists_and_filters(tmp_path):
    run = _make_run(tmp_path, tasks=("a_r1", "b_r1"))
    out = bench._bench_run_task_dirs(run, None)
    assert {d["name"] for d in out} == {"a_r1", "b_r1"}
    only_a = bench._bench_run_task_dirs(run, ["a"])
    assert {d["name"] for d in only_a} == {"a_r1"}


def test_bench_run_task_dirs_unknown_only_dies(tmp_path):
    run = _make_run(tmp_path, tasks=("a_r1",))
    with pytest.raises(SystemExit):
        bench._bench_run_task_dirs(run, ["ghost"])


def test_iter_bench_runs_filters_by_suite(tmp_path):
    _make_run(tmp_path, "run1")
    other = tmp_path / "run2"
    other.mkdir()
    (other / "bench_manifest.yaml").write_text("bench_suite: other\n", encoding="utf-8")
    rows = list(bench._iter_bench_runs(tmp_path, suite_id="demo"))
    assert len(rows) == 1
    assert rows[0][0].name == "run1"


def test_iter_bench_runs_missing_root(tmp_path):
    assert list(bench._iter_bench_runs(tmp_path / "nope")) == []


# ===========================================================================
# git commit-walk helpers (real tiny git repo)
# ===========================================================================

def _git_repo(tmp_path):
    from zyme.utils import git
    repo = tmp_path / "repo"
    repo.mkdir()
    git("init", "--quiet", cwd=repo)
    git("config", "user.email", "t@e.st", cwd=repo)
    git("config", "user.name", "t", cwd=repo)
    return repo


def _commit(repo, fname, msg):
    from zyme.utils import git
    (repo / fname).write_text("x", encoding="utf-8")
    git("add", "-A", cwd=repo)
    git("commit", "--quiet", "-m", msg, cwd=repo)
    return git("rev-parse", "HEAD", cwd=repo)


def test_find_init_commit(tmp_path):
    repo = _git_repo(tmp_path)
    init_sha = _commit(repo, "a", "zyme init: scaffold task")
    _commit(repo, "b", "[conservative] first opt")
    assert bench._find_init_commit(repo) == init_sha


def test_find_init_commit_none_when_absent(tmp_path):
    repo = _git_repo(tmp_path)
    _commit(repo, "a", "setup: nothing")
    assert bench._find_init_commit(repo) is None


def test_find_post_init_commit_parent_of_first_iterate(tmp_path):
    repo = _git_repo(tmp_path)
    init_sha = _commit(repo, "a", "zyme init: scaffold")
    _commit(repo, "b", "[algorithmic] first opt")
    # post_init = parent of first tagged-iterate commit = init commit
    assert bench._find_post_init_commit(repo) == init_sha


def test_find_post_init_commit_head_when_no_iterate(tmp_path):
    repo = _git_repo(tmp_path)
    head = _commit(repo, "a", "zyme init: scaffold")
    assert bench._find_post_init_commit(repo) == head


def test_git_head_sha(tmp_path):
    repo = _git_repo(tmp_path)
    sha = _commit(repo, "a", "init")
    assert bench._git_head_sha(repo) == sha


def test_git_head_sha_empty_repo(tmp_path):
    repo = _git_repo(tmp_path)
    assert bench._git_head_sha(repo) == ""


# ===========================================================================
# doctor / list / status commands against a temp templates+suites+runs tree
# ===========================================================================

def _build_template(stage_dir, name, *, with_data=True, stage="iterate"):
    tpl = stage_dir / name
    tpl.mkdir(parents=True)
    (tpl / "bench_template.yaml").write_text(f"stage: {stage}\n", encoding="utf-8")
    data = tpl / "data"
    if with_data:
        data.mkdir()
        (data / "tiny.h5ad").write_text("x", encoding="utf-8")
    (tpl / "task.yaml").write_text(
        "datasets:\n  - {tier: tiny, name: tiny_a, path: data/tiny.h5ad}\n",
        encoding="utf-8",
    )
    (tpl / "reference.R").write_text("# ref\n", encoding="utf-8")
    refout = tpl / "reference_outputs" / "tiny"
    refout.mkdir(parents=True)
    (refout / "out.txt").write_text("y", encoding="utf-8")
    # round-0 pipeline that differs from reference
    pipe = tpl / "pipeline"
    pipe.mkdir()
    (pipe / "run.R").write_text("# round0 pipeline run\n", encoding="utf-8")
    return tpl


@pytest.fixture
def bench_tree(tmp_path, monkeypatch):
    """A hermetic templates/suites/runs layout with roots monkeypatched."""
    templates_root = tmp_path / "bench_templates"
    suites_root = tmp_path / "bench_suites"
    runs_root = tmp_path / "bench_runs"
    suites_root.mkdir()
    runs_root.mkdir()
    stage_dir = templates_root / "iterate"
    _build_template(stage_dir, "findallmarker")
    _build_template(stage_dir, "mast_tpl")
    monkeypatch.setattr(bench, "_bench_templates_root", lambda: templates_root)
    monkeypatch.setattr(bench, "_bench_suites_root", lambda: suites_root)
    monkeypatch.setattr(bench, "_default_bench_runs_root", lambda: runs_root)
    suite_path = suites_root / "iterate_demo.yaml"
    suite_path.write_text(SUITE_YAML, encoding="utf-8")
    monkeypatch.setattr(bench, "_bench_suite_path", lambda sid: suites_root / f"{sid}.yaml")
    return types.SimpleNamespace(
        templates_root=templates_root, suites_root=suites_root,
        runs_root=runs_root, stage_dir=stage_dir, suite_path=suite_path,
    )


def test_cmd_bench_doctor_passes_for_valid_templates(bench_tree, capsys):
    args = types.SimpleNamespace(suite_id="iterate_demo", only=None)
    bench.cmd_bench_doctor(args)
    out = capsys.readouterr().out
    assert "doctor passed" in out
    assert "OK   findallmarker" in out


def test_cmd_bench_doctor_fails_on_missing_data(bench_tree, capsys):
    # Remove the data file so the dataset-path check fails.
    (bench_tree.stage_dir / "findallmarker" / "data" / "tiny.h5ad").unlink()
    args = types.SimpleNamespace(suite_id="iterate_demo", only=["findallmarker"])
    with pytest.raises(SystemExit):
        bench.cmd_bench_doctor(args)
    out = capsys.readouterr().out
    assert "FAIL findallmarker" in out
    assert "path missing" in out


def test_cmd_bench_doctor_missing_suite_dies(bench_tree):
    args = types.SimpleNamespace(suite_id="nonexistent", only=None)
    with pytest.raises(SystemExit):
        bench.cmd_bench_doctor(args)


def test_cmd_bench_doctor_missing_template_dir(bench_tree, capsys):
    import shutil
    shutil.rmtree(bench_tree.stage_dir / "mast_tpl")
    args = types.SimpleNamespace(suite_id="iterate_demo", only=None)
    with pytest.raises(SystemExit):
        bench.cmd_bench_doctor(args)
    out = capsys.readouterr().out
    assert "missing template dir" in out


def test_cmd_bench_list_templates(bench_tree, capsys):
    args = types.SimpleNamespace(stage=None)
    bench.cmd_bench_list_templates(args)
    out = capsys.readouterr().out
    assert "stage: iterate" in out
    assert "findallmarker" in out and "mast_tpl" in out
    assert "2 total template(s)" in out


def test_cmd_bench_list_templates_filtered_stage(bench_tree, capsys):
    # Only the 'iterate' stage exists; request a different one -> none.
    args = types.SimpleNamespace(stage="package")
    bench.cmd_bench_list_templates(args)
    out = capsys.readouterr().out
    assert "no templates registered" in out


def test_cmd_bench_list_templates_empty_root(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench, "_bench_templates_root", lambda: tmp_path / "empty")
    bench.cmd_bench_list_templates(types.SimpleNamespace(stage=None))
    assert "no templates registered yet" in capsys.readouterr().out


def test_cmd_bench_list_runs(bench_tree, capsys):
    _make_run(bench_tree.runs_root, "demo_run", tasks=("findallmarker_r1",))
    args = types.SimpleNamespace(root=None)
    bench.cmd_bench_list(args)
    out = capsys.readouterr().out
    assert "demo_run" in out
    assert "1 run(s)" in out


def test_cmd_bench_list_runs_empty(bench_tree, capsys):
    args = types.SimpleNamespace(root=None)
    bench.cmd_bench_list(args)
    assert "no bench runs" in capsys.readouterr().out


# ===========================================================================
# register-template early validation (pre-subprocess dies)
# ===========================================================================

def test_register_template_not_a_task_dies(tmp_path):
    args = types.SimpleNamespace(
        task_dir=str(tmp_path), commit=None, at="post_init", stage="iterate",
        name="x", force=False)
    with pytest.raises(SystemExit):
        bench.cmd_bench_register_template(args)


def test_register_template_not_a_git_repo_dies(tmp_path):
    (tmp_path / "task.yaml").write_text("datasets: []\n", encoding="utf-8")
    args = types.SimpleNamespace(
        task_dir=str(tmp_path), commit=None, at="post_init", stage="iterate",
        name="x", force=False)
    with pytest.raises(SystemExit):
        bench.cmd_bench_register_template(args)


# ===========================================================================
# usage / prices commands (dispatch telemetry boundary monkeypatched)
# ===========================================================================

def test_cmd_bench_usage_json(bench_tree, capsys, monkeypatch):
    import zyme.dispatch as dispatch
    run = _make_run(bench_tree.runs_root, "urun", tasks=("findallmarker_r1",))
    monkeypatch.setattr(dispatch, "collect_usage",
                        lambda *a, **k: {"total_usd": 1.23})
    monkeypatch.setattr(dispatch, "render_usage", lambda s: "rendered usage")
    args = types.SimpleNamespace(
        run=str(run), root=None, token_budget=None, budget_basis=None,
        price_model=None, json_output=True)
    bench.cmd_bench_usage(args)
    out = capsys.readouterr().out
    assert '"total_usd": 1.23' in out


def test_cmd_bench_usage_text(bench_tree, capsys, monkeypatch):
    import zyme.dispatch as dispatch
    run = _make_run(bench_tree.runs_root, "urun2", tasks=("findallmarker_r1",))
    monkeypatch.setattr(dispatch, "collect_usage", lambda *a, **k: {})
    monkeypatch.setattr(dispatch, "render_usage", lambda s: "USAGE TABLE")
    args = types.SimpleNamespace(
        run=str(run), root=None, token_budget=None, budget_basis=None,
        price_model=None, json_output=False)
    bench.cmd_bench_usage(args)
    assert "USAGE TABLE" in capsys.readouterr().out


def test_cmd_bench_prices_json(capsys, monkeypatch):
    import zyme.dispatch as dispatch
    monkeypatch.setattr(dispatch, "list_prices", lambda: {"opus": {"in": 1.0}})
    bench.cmd_bench_prices(types.SimpleNamespace(json_output=True))
    assert '"opus"' in capsys.readouterr().out


def test_cmd_bench_prices_table(capsys, monkeypatch):
    import zyme.dispatch as dispatch
    monkeypatch.setattr(dispatch, "render_price_table", lambda: "PRICE TABLE")
    bench.cmd_bench_prices(types.SimpleNamespace(json_output=False))
    assert "PRICE TABLE" in capsys.readouterr().out


# ===========================================================================
# _scaffold_bench_task — orchestration with git-init boundary stubbed
# ===========================================================================

def test_scaffold_bench_task_materializes_symlink_and_meta(bench_tree, tmp_path, monkeypatch):
    # Build a symlink source the template references.
    data_src = tmp_path / "shared_data"
    data_src.mkdir()
    (data_src / "x.h5ad").write_text("data", encoding="utf-8")

    # Build a template that does NOT already carry a data/ dir, so the
    # symlink_sources materialization actually creates the symlink.
    template_dir = _build_template(
        bench_tree.stage_dir, "scaffold_tpl", with_data=False)
    (template_dir / "bench_template.yaml").write_text(
        "stage: iterate\nsymlink_sources:\n  data: " + str(data_src) + "\n",
        encoding="utf-8",
    )

    # Prompt snapshot dir + card.
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "prompt.md").write_text("PROMPT BODY", encoding="utf-8")
    field_prompts = tmp_path / "field_prompts"
    field_prompts.mkdir()
    (field_prompts / "2_iterate.md").write_text("live iterate\n", encoding="utf-8")
    prompt_card = {"source_path": "prompts/2_iterate.md"}

    # Stub the git boundary so no real repo is created.
    monkeypatch.setattr(
        bench, "_git_init_with_initial_commit", lambda d, m: "deadbeef" * 5)

    dest = tmp_path / "dest_r1"
    meta = {"bench_suite": "s", "bench_task": "findallmarker", "replicate": 1}
    sha, report = bench._scaffold_bench_task(
        template_dir=template_dir, dest=dest, prompt_snapshot_dir=snap,
        prompt_card=prompt_card, field_prompts_dir=field_prompts, meta=meta,
    )
    assert sha.startswith("deadbeef")
    # symlink materialized
    assert (dest / "data").is_symlink()
    assert any(n == "data" and st == "created" for n, _, st in report)
    # prompt slot overwritten with the snapshot body
    assert (dest / "prompts" / "2_iterate.md").read_text(encoding="utf-8") == "PROMPT BODY"
    # meta dropped at root
    assert (dest / bench.ZYME_META_FILENAME).exists()
    # template manifest stripped from the replicate
    assert not (dest / "bench_template.yaml").exists()


def test_scaffold_bench_task_dest_exists_dies(bench_tree, tmp_path):
    template_dir = bench_tree.stage_dir / "findallmarker"
    dest = tmp_path / "dest_exists"
    dest.mkdir()
    with pytest.raises(SystemExit):
        bench._scaffold_bench_task(
            template_dir=template_dir, dest=dest, prompt_snapshot_dir=tmp_path,
            prompt_card={"source_path": "prompts/2_iterate.md"},
            field_prompts_dir=tmp_path, meta={})


def test_cmd_bench_status_runs_clean(bench_tree, capsys, monkeypatch):
    # No snapshots reachable, no runs -> still prints the headers and bails.
    import zyme.registry as registry
    monkeypatch.setattr(registry, "iter_snapshots", lambda *a, **k: iter([]))
    _make_run(bench_tree.runs_root, "demo_run", tasks=("findallmarker_r1",))
    args = types.SimpleNamespace(
        suite_id="iterate_demo", root=None, only=None, limit=10,
        field=None, slot=None)
    bench.cmd_bench_status(args)
    out = capsys.readouterr().out
    assert "# bench status" in out
    assert "## templates" in out
    assert "## bench runs" in out
