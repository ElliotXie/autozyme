"""Wave-3 mop-up for zyme.commands.bench.

Wave-2 (tests/commands/test_bench_cmd.py) covered the parsing/selection/dispatch
helpers and the doctor/list/status commands. This file fills the remaining
REACHABLE in-process branches the orchestration commands left:

  - _template_check_lines error/warning branches (stage mismatch, missing files,
    runtime-state pollution, upstream_repo/data block validation, dataset path
    + reference-output checks)
  - _detect_upstream_clone (target_repo URL vs no-origin -> None)
  - _copy_memory_state
  - _ensure_bench_framework_link (created / already_exists / conflict)
  - _dispatch_task_states (no state -> {}; populated queue)
  - _prompt_file_for_bench_run (snapshot resolution + slot-glob fallback + die)
  - _bench_task_progress / _detail edge cases (missing status col, header-only)
  - _read_suite_yaml empty-mapping branch
  - cmd_bench_init pre-scaffold validation gates (slot/field mismatch,
    missing template, --name arity, out-exists, framework-link conflict)
    and a full single-name end-to-end scaffold with the git boundary stubbed.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from zyme.commands import bench


# ===========================================================================
# _template_check_lines — error / warning branches
# ===========================================================================

def _base_template(tmp_path, *, stage="iterate", with_ref=True,
                   with_task=True, with_data=True):
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "bench_template.yaml").write_text(f"stage: {stage}\n", encoding="utf-8")
    if with_task:
        (tpl / "task.yaml").write_text(
            "datasets:\n  - {tier: tiny, name: tiny_a, path: data/tiny.h5ad}\n",
            encoding="utf-8")
    if with_ref:
        (tpl / "reference.R").write_text("# ref\n", encoding="utf-8")
    if with_data:
        (tpl / "data").mkdir()
        (tpl / "data" / "tiny.h5ad").write_text("x", encoding="utf-8")
        ro = tpl / "reference_outputs" / "tiny"
        ro.mkdir(parents=True)
        (ro / "out.txt").write_text("y", encoding="utf-8")
    return tpl


_SUITE_ITER = {"id": "s", "stage": "iterate", "prompt_slot": "iterate",
               "field": "Bio"}


def test_template_check_missing_manifest(tmp_path):
    tpl = tmp_path / "bare"
    tpl.mkdir()
    errors, warnings = bench._template_check_lines(_SUITE_ITER, {}, tpl)
    assert any("missing bench_template.yaml" in e for e in errors)


def test_template_check_stage_mismatch(tmp_path):
    tpl = _base_template(tmp_path, stage="package")
    errors, _ = bench._template_check_lines(_SUITE_ITER, {}, tpl)
    assert any("manifest stage=package" in e for e in errors)


def test_template_check_missing_task_and_reference(tmp_path):
    tpl = _base_template(tmp_path, with_ref=False, with_task=False,
                         with_data=False)
    errors, _ = bench._template_check_lines(_SUITE_ITER, {}, tpl)
    assert any("missing task.yaml" in e for e in errors)
    assert any("missing reference" in e for e in errors)


def test_template_check_run_without_reference(tmp_path):
    tpl = _base_template(tmp_path, with_ref=False)
    # pipeline/run.R exists but reference.R is absent.
    (tpl / "pipeline").mkdir()
    (tpl / "pipeline" / "run.R").write_text("# round0\n", encoding="utf-8")
    errors, _ = bench._template_check_lines(_SUITE_ITER, {}, tpl)
    assert any("reference.R is missing" in e for e in errors)


def test_template_check_literal_reference_copy(tmp_path):
    tpl = _base_template(tmp_path)
    (tpl / "pipeline").mkdir()
    same = "# identical\n"
    (tpl / "reference.R").write_text(same, encoding="utf-8")
    (tpl / "pipeline" / "run.R").write_text(same, encoding="utf-8")
    errors, _ = bench._template_check_lines(_SUITE_ITER, {}, tpl)
    assert any("literal reference copy" in e for e in errors)


def test_template_check_active_install_override(tmp_path):
    tpl = _base_template(tmp_path)
    (tpl / "pipeline").mkdir()
    (tpl / "reference.R").write_text("# ref\n", encoding="utf-8")
    (tpl / "pipeline" / "run.R").write_text(
        'install_override("a","b",f)\n# round0\n', encoding="utf-8")
    errors, _ = bench._template_check_lines(_SUITE_ITER, {}, tpl)
    assert any("active install_override" in e for e in errors)


def test_template_check_runtime_state_pollution(tmp_path):
    tpl = _base_template(tmp_path)
    (tpl / ".zyme").mkdir()
    (tpl / ".zyme" / "best.ref").write_text("sha\n", encoding="utf-8")
    errors, _ = bench._template_check_lines(_SUITE_ITER, {}, tpl)
    assert any("runtime state .zyme/best.ref" in e for e in errors)


def test_template_check_pipeline_output_dirs(tmp_path):
    tpl = _base_template(tmp_path)
    out = tpl / "pipeline" / "output_tiny"
    out.mkdir(parents=True)
    errors, _ = bench._template_check_lines(_SUITE_ITER, {}, tpl)
    assert any("pipeline output dirs" in e for e in errors)


def test_template_check_symlink_source_missing(tmp_path):
    tpl = _base_template(tmp_path)
    (tpl / "bench_template.yaml").write_text(
        "stage: iterate\nsymlink_sources:\n  data: /no/such/path\n",
        encoding="utf-8")
    errors, _ = bench._template_check_lines(_SUITE_ITER, {}, tpl)
    assert any("symlink source missing" in e for e in errors)


def test_template_check_upstream_repo_block_errors(tmp_path):
    tpl = _base_template(tmp_path)
    (tpl / "bench_template.yaml").write_text(
        "stage: iterate\nupstream_repo:\n  notes: incomplete\n", encoding="utf-8")
    errors, _ = bench._template_check_lines(_SUITE_ITER, {}, tpl)
    assert any("missing clone_url" in e for e in errors)
    assert any("missing commit" in e for e in errors)


def test_template_check_data_block_errors(tmp_path):
    tpl = _base_template(tmp_path)
    (tpl / "bench_template.yaml").write_text(
        "stage: iterate\ndata:\n  hf_repo: only_this\n", encoding="utf-8")
    errors, _ = bench._template_check_lines(_SUITE_ITER, {}, tpl)
    assert any("data block missing hf_repo_type" in e for e in errors)
    assert any("data block missing hf_path" in e for e in errors)


def test_template_check_dataset_path_missing(tmp_path):
    tpl = _base_template(tmp_path)
    # remove the data file so the dataset path resolves but doesn't exist.
    (tpl / "data" / "tiny.h5ad").unlink()
    errors, _ = bench._template_check_lines(_SUITE_ITER, {}, tpl)
    assert any("path missing" in e for e in errors)


def test_template_check_reference_output_empty(tmp_path):
    tpl = _base_template(tmp_path)
    # empty out the reference_outputs/tiny dir.
    (tpl / "reference_outputs" / "tiny" / "out.txt").unlink()
    errors, _ = bench._template_check_lines(_SUITE_ITER, {}, tpl)
    assert any("reference output empty" in e for e in errors)


def test_template_check_passes_clean(tmp_path):
    tpl = _base_template(tmp_path)
    # A round-0 pipeline that differs from reference is required for iterate.
    (tpl / "pipeline").mkdir()
    (tpl / "pipeline" / "run.R").write_text(
        "# round0 pipeline (differs from reference)\n", encoding="utf-8")
    errors, warnings = bench._template_check_lines(_SUITE_ITER, {}, tpl)
    assert errors == []


# ===========================================================================
# _detect_upstream_clone
# ===========================================================================

def _git_repo(tmp_path, name="up"):
    from zyme.utils import git
    repo = tmp_path / name
    repo.mkdir()
    git("init", "--quiet", cwd=repo)
    git("config", "user.email", "t@e.st", cwd=repo)
    git("config", "user.name", "t", cwd=repo)
    (repo / "f.txt").write_text("x", encoding="utf-8")
    git("add", "-A", cwd=repo)
    git("commit", "--quiet", "-m", "init", cwd=repo)
    return repo


def test_detect_upstream_clone_none_when_no_upstream(tmp_path):
    src = tmp_path / "task"
    src.mkdir()
    assert bench._detect_upstream_clone(src) is None


def test_detect_upstream_clone_from_target_repo_url(tmp_path):
    src = tmp_path / "task"
    src.mkdir()
    upstream = _git_repo(src.parent, "scratch")
    # move the git repo into src/upstream_repo
    import shutil
    shutil.move(str(upstream), str(src / "upstream_repo"))
    (src / "task.yaml").write_text(
        'target_repo: "https://github.com/foo/bar"\n', encoding="utf-8")
    out = bench._detect_upstream_clone(src)
    assert out is not None
    assert out["clone_url"] == "https://github.com/foo/bar"
    assert len(out["commit"]) >= 7


def test_detect_upstream_clone_none_without_url(tmp_path):
    src = tmp_path / "task"
    src.mkdir()
    upstream = _git_repo(src.parent, "scratch2")
    import shutil
    shutil.move(str(upstream), str(src / "upstream_repo"))
    # task.yaml target_repo is a local path, not a URL; no git origin set.
    (src / "task.yaml").write_text("target_repo: /local/path\n", encoding="utf-8")
    assert bench._detect_upstream_clone(src) is None


# ===========================================================================
# _copy_memory_state
# ===========================================================================

def test_copy_memory_state_copies_tree(tmp_path):
    src = tmp_path / "src"
    (src / "memory").mkdir(parents=True)
    (src / "memory" / "discoveries.md").write_text("notes", encoding="utf-8")
    dest = tmp_path / "dest"
    dest.mkdir()
    copied = bench._copy_memory_state(src, dest)
    assert "memory/discoveries.md" in copied
    assert (dest / "memory" / "discoveries.md").exists()


def test_copy_memory_state_noop_without_memory(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    assert bench._copy_memory_state(src, dest) == []


# ===========================================================================
# _ensure_bench_framework_link
# ===========================================================================

def test_ensure_framework_link_created(tmp_path):
    out_root = tmp_path / "run"
    out_root.mkdir()
    status, target = bench._ensure_bench_framework_link(out_root)
    assert status == "created"
    assert (out_root / "autozyme-framework").is_symlink()


def test_ensure_framework_link_already_exists(tmp_path):
    out_root = tmp_path / "run"
    out_root.mkdir()
    bench._ensure_bench_framework_link(out_root)
    status, _ = bench._ensure_bench_framework_link(out_root)
    assert status == "already_exists"


def test_ensure_framework_link_conflict(tmp_path):
    out_root = tmp_path / "run"
    out_root.mkdir()
    # A symlink pointing somewhere else -> conflict.
    other = tmp_path / "other"
    other.mkdir()
    (out_root / "autozyme-framework").symlink_to(other, target_is_directory=True)
    status, _ = bench._ensure_bench_framework_link(out_root)
    assert status == "conflict"


# ===========================================================================
# _dispatch_task_states
# ===========================================================================

def test_dispatch_task_states_no_state(tmp_path):
    assert bench._dispatch_task_states(tmp_path) == {}


def test_dispatch_task_states_populated(tmp_path):
    from zyme.dispatch.state import (
        ensure_dispatch_dirs, state_path, write_state_atomic)
    ensure_dispatch_dirs(tmp_path)
    write_state_atomic(state_path(tmp_path), {
        "master_pid": 2_000_000_000,  # dead -> dispatch_alive False
        "agent": "claude",
        "model": "opus",
        "queue": [{"name": "task_a", "status": "running"}],
    })
    out = bench._dispatch_task_states(tmp_path)
    assert "task_a" in out
    assert out["task_a"]["agent"] == "claude"
    assert out["task_a"]["dispatch_alive"] is False


# ===========================================================================
# _prompt_file_for_bench_run
# ===========================================================================

def test_prompt_file_from_slot_glob(tmp_path, monkeypatch):
    run = tmp_path / "run"
    task = run / "task_r1"
    (task / "prompts").mkdir(parents=True)
    (task / bench.ZYME_META_FILENAME).write_text("bench_task: a\n", encoding="utf-8")
    (task / "prompts" / "2_iterate.md").write_text("body\n", encoding="utf-8")
    # No prompt_id -> falls to slot glob.
    out = bench._prompt_file_for_bench_run(run, {"prompt_slot": "iterate"})
    assert out == "prompts/2_iterate.md"


def test_prompt_file_unresolvable_dies(tmp_path):
    run = tmp_path / "run"
    task = run / "task_r1"
    (task / "prompts").mkdir(parents=True)
    (task / bench.ZYME_META_FILENAME).write_text("bench_task: a\n", encoding="utf-8")
    # slot with no matching prompt + no prompt_id -> die.
    with pytest.raises(SystemExit):
        bench._prompt_file_for_bench_run(run, {"prompt_slot": "ghostslot"})


# ===========================================================================
# _bench_task_progress / _detail edge cases
# ===========================================================================

def test_bench_task_progress_header_only(tmp_path):
    (tmp_path / "results.tsv").write_text(
        "round\tcommit\tdataset\tstatus\n", encoding="utf-8")
    has, decisions, last = bench._bench_task_progress(tmp_path)
    assert has is True and decisions == 0 and last == "-"


def test_bench_task_progress_no_status_column(tmp_path):
    # Header without a 'status' column -> status_idx None, no decisions counted.
    (tmp_path / "results.tsv").write_text(
        "round\tcommit\tdataset\n1\tabc\tds\n", encoding="utf-8")
    has, decisions, last = bench._bench_task_progress(tmp_path)
    assert has is True and decisions == 0


def test_bench_task_progress_detail_tracks_pending_and_best_seen(tmp_path):
    (tmp_path / "results.tsv").write_text(
        "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
        "metrics_json\thypothesis\tdescription\tphase\n"
        "0\tup\ta\t10\t0\t5\tbaseline\t{}\t\t\toptimize\n"
        "1\tc1\ta\t8\t20\t5\tpending\t{}\t\t\toptimize\n"
        "2\tc2\ta\t7\t30\t5\trerun\t{}\t\t\toptimize\n",
        encoding="utf-8")
    d = bench._bench_task_progress_detail(tmp_path)
    assert d["pending_rounds"] == 1
    # rerun + pending count toward best_seen but not best_keep.
    assert d["best_keep_pct"] is None
    assert d["best_seen_pct"] == 30.0


# ===========================================================================
# _read_suite_yaml empty-mapping branch
# ===========================================================================

def test_read_suite_yaml_empty_mapping(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        "id: demo\n"
        "stage: iterate\n"
        "extras:\n"          # key with no value and no list -> empty mapping
        "field: Bio\n",
        encoding="utf-8")
    out = bench._read_suite_yaml(p)
    assert out["extras"] == {}
    assert out["id"] == "demo"
    assert out["field"] == "Bio"


# ===========================================================================
# cmd_bench_init — validation gates (pre-scaffold) + single-name end-to-end
# ===========================================================================

SUITE_YAML = """\
id: iterate_demo
stage: iterate
prompt_slot: iterate
field: Bio
description: demo
default_reps: 1

tasks:
  - id: findallmarker
    template: findallmarker
  - id: mast
    template: mast_tpl
"""


def _build_template_tree(tmp_path, monkeypatch):
    templates_root = tmp_path / "bench_templates"
    suites_root = tmp_path / "bench_suites"
    runs_root = tmp_path / "bench_runs"
    suites_root.mkdir()
    runs_root.mkdir()
    stage_dir = templates_root / "iterate"
    for name in ("findallmarker", "mast_tpl"):
        tpl = stage_dir / name
        tpl.mkdir(parents=True)
        (tpl / "bench_template.yaml").write_text("stage: iterate\n", encoding="utf-8")
        (tpl / "task.yaml").write_text(
            "datasets:\n  - {tier: tiny, name: tiny_a, path: data/tiny.h5ad}\n",
            encoding="utf-8")
        (tpl / "reference.R").write_text("# ref\n", encoding="utf-8")
        (tpl / "prompts").mkdir()
        (tpl / "prompts" / "2_iterate.md").write_text("orig prompt\n", encoding="utf-8")
        (tpl / "memory").mkdir()
        (tpl / "memory" / "discoveries.md").write_text("# disc\n", encoding="utf-8")
    monkeypatch.setattr(bench, "_bench_templates_root", lambda: templates_root)
    monkeypatch.setattr(bench, "_bench_suites_root", lambda: suites_root)
    monkeypatch.setattr(bench, "_default_bench_runs_root", lambda: runs_root)
    suite_path = suites_root / "iterate_demo.yaml"
    suite_path.write_text(SUITE_YAML, encoding="utf-8")
    monkeypatch.setattr(bench, "_bench_suite_path",
                        lambda sid: suites_root / f"{sid}.yaml")
    return types.SimpleNamespace(
        templates_root=templates_root, suites_root=suites_root,
        runs_root=runs_root, stage_dir=stage_dir)


def _init_args(**over):
    base = dict(
        suite_id="iterate_demo", prompt_id="bio/iterate/p1", only=None,
        reps=None, name=None, out=None, force=False, purpose=None)
    base.update(over)
    return types.SimpleNamespace(**base)


def _stub_snapshot(tmp_path, monkeypatch, *, slot="iterate", field="Bio"):
    """Make registry.find_snapshot/snapshot_dir_for/live_prompts_dir hermetic."""
    from zyme import registry
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "prompt.md").write_text("SNAPSHOT BODY", encoding="utf-8")
    live = tmp_path / "live_prompts"
    live.mkdir()
    (live / "2_iterate.md").write_text("live iterate\n", encoding="utf-8")
    card = {"source_path": "prompts/2_iterate.md", "name": "p1",
            "content_sha256": "abc123"}
    monkeypatch.setattr(
        registry, "find_snapshot",
        lambda root, pid: (field, slot, "bio/iterate/p1", card))
    monkeypatch.setattr(registry, "snapshot_dir_for",
                        lambda root, f, s, pid: snap)
    monkeypatch.setattr(registry, "live_prompts_dir", lambda root, f: live)
    return card


def test_init_suite_missing_dies(tmp_path, monkeypatch):
    _build_template_tree(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        bench.cmd_bench_init(_init_args(suite_id="nonexistent"))


def test_init_slot_mismatch_dies(tmp_path, monkeypatch):
    _build_template_tree(tmp_path, monkeypatch)
    _stub_snapshot(tmp_path, monkeypatch, slot="package")
    with pytest.raises(SystemExit):
        bench.cmd_bench_init(_init_args())


def test_init_field_mismatch_dies(tmp_path, monkeypatch):
    _build_template_tree(tmp_path, monkeypatch)
    _stub_snapshot(tmp_path, monkeypatch, field="OtherField")
    with pytest.raises(SystemExit):
        bench.cmd_bench_init(_init_args())


def test_init_missing_template_dies(tmp_path, monkeypatch):
    tree = _build_template_tree(tmp_path, monkeypatch)
    _stub_snapshot(tmp_path, monkeypatch)
    import shutil
    shutil.rmtree(tree.stage_dir / "mast_tpl")
    with pytest.raises(SystemExit):
        bench.cmd_bench_init(_init_args())


def test_init_name_requires_single_task_dies(tmp_path, monkeypatch):
    _build_template_tree(tmp_path, monkeypatch)
    _stub_snapshot(tmp_path, monkeypatch)
    # --name with 2 tasks selected (no --only) -> dies.
    with pytest.raises(SystemExit):
        bench.cmd_bench_init(_init_args(name="solo"))


def test_init_name_reps_gt_one_dies(tmp_path, monkeypatch):
    _build_template_tree(tmp_path, monkeypatch)
    _stub_snapshot(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        bench.cmd_bench_init(_init_args(name="solo", only=["findallmarker"],
                                        reps=2))


def test_init_out_exists_without_force_dies(tmp_path, monkeypatch):
    _build_template_tree(tmp_path, monkeypatch)
    _stub_snapshot(tmp_path, monkeypatch)
    out = tmp_path / "preexisting"
    out.mkdir()
    with pytest.raises(SystemExit):
        bench.cmd_bench_init(_init_args(out=str(out)))


def test_init_single_name_end_to_end(tmp_path, monkeypatch):
    """Full --name scaffold path with only the git boundary stubbed."""
    _build_template_tree(tmp_path, monkeypatch)
    _stub_snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(
        bench, "_git_init_with_initial_commit", lambda d, m: "abcdef01" * 5)
    out = tmp_path / "drop"
    out.mkdir()
    bench.cmd_bench_init(_init_args(
        name="solo", only=["findallmarker"], out=str(out)))
    dest = out / "solo"
    assert dest.is_dir()
    # prompt slot overwritten with the snapshot body.
    assert (dest / "prompts" / "2_iterate.md").read_text(
        encoding="utf-8") == "SNAPSHOT BODY"
    # meta dropped at root; template manifest stripped.
    assert (dest / bench.ZYME_META_FILENAME).exists()
    assert not (dest / "bench_template.yaml").exists()
    # --name mode does NOT write a suite-level manifest.
    assert not (out / "bench_manifest.yaml").exists()
