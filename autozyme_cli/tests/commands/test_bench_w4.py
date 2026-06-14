"""Wave-4 mop-up for zyme.commands.bench — benchmark template/run scaffolding.

Wave-2 (test_bench_cmd.py) + wave-3 (test_bench_w3.py) covered the helpers,
doctor/list/usage/prices/scaffold, and the register-template early-validation
dies. This file drives the big in-process command bodies that wave-2/3 skipped:

  - cmd_bench_register_template END TO END over a real throwaway git task: the
    `git archive | tar` snapshot actually runs (local, no network), exercising
    path-rewrite, pipeline-reset, runtime/memory state copy, gitignore rewrite,
    bench_template.yaml emit, and the summary banner. (~200 lines.)
  - cmd_bench_start: dry-run banner + the validation gates (missing prompt,
    missing reflect prompt, already-running, bad ram-floor, explicit disk-floor)
    + the non-dry-run start_dispatch path with the dispatch boundary stubbed.
  - cmd_bench_status: the rich branches — prompt snapshots present, suite
    template OK/FAIL/missing status, no-suite stage counts, and bench-run +
    task-progress rows.

Only the dispatch daemon boundary (start_dispatch / experiment doc) is stubbed;
git is real.
"""
from __future__ import annotations

import subprocess
import types
from pathlib import Path

import pytest

from zyme.commands import bench


# ---------------------------------------------------------------------------
# Real throwaway git task for register-template
# ---------------------------------------------------------------------------

def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, text=True)


def _make_source_task(tmp_path, *, name="src_task", relative_data=False):
    d = tmp_path / name
    (d / "pipeline").mkdir(parents=True)
    (d / "memory").mkdir()
    (d / ".zyme").mkdir()
    data_line = (
        "  - {tier: tiny, name: tiny_a, path: ../shared/data/x.h5ad}\n"
        if relative_data
        else "  - {tier: tiny, name: tiny_a, path: data/x.h5ad}\n"
    )
    (d / "task.yaml").write_text(
        "target_repo: https://example.com/x\n"
        "target_function: foo\n"
        "datasets:\n" + data_line +
        "metrics:\n  - {name: j, comparator: gte, threshold: 0.9}\n"
    )
    (d / "reference.py").write_text("def reference():\n    return 1\n")
    (d / "pipeline" / "run.py").write_text("# round0 pipeline (differs from ref)\n")
    (d / ".gitignore").write_text("data/\nresults.tsv\n")
    # Tracked runtime state that the template should carry.
    (d / "results.tsv").write_text("round\tstatus\n0\tbaseline\n")
    (d / ".zyme" / "baselines_stash.tsv").write_text("tier\tspeed\ntiny\t1.0\n")
    (d / "memory" / "discoveries.md").write_text("## DISCOVERY: vectorize\n")
    _git("init", "-q", cwd=d)
    _git("config", "user.email", "t@t", cwd=d)
    _git("config", "user.name", "t", cwd=d)
    # Force-add the gitignored runtime state so git archive carries it.
    _git("add", "-A", "-f", cwd=d)
    _git("commit", "-q", "-m", "zyme init: scaffold", cwd=d)
    return d


@pytest.fixture
def templates_root(tmp_path, monkeypatch):
    root = tmp_path / "bench_templates"
    monkeypatch.setattr(bench, "_bench_templates_root", lambda: root)
    monkeypatch.setattr(bench, "_bench_template_path",
                        lambda stage, name: root / stage / name)
    return root


def _register_args(task_dir, **over):
    base = dict(task_dir=str(task_dir), commit=None, at="post_init",
                stage="iterate", name="tpl1", force=False)
    base.update(over)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# cmd_bench_register_template — end to end
# ---------------------------------------------------------------------------

def test_register_template_post_init_end_to_end(tmp_path, templates_root,
                                                capsys, monkeypatch):
    src = _make_source_task(tmp_path)
    bench.cmd_bench_register_template(_register_args(src))
    out = capsys.readouterr().out
    assert "template 'tpl1' registered" in out
    dest = templates_root / "iterate" / "tpl1"
    assert dest.is_dir()
    # The git-archive snapshot carried tracked files.
    assert (dest / "task.yaml").exists()
    assert (dest / "pipeline" / "run.py").exists()
    # Provenance manifest written.
    assert (dest / "bench_template.yaml").exists()
    meta_text = (dest / "bench_template.yaml").read_text()
    assert "stage: iterate" in meta_text
    assert "source_commit" in meta_text


def test_register_template_iterate_resets_pipeline_to_reference(
        tmp_path, templates_root, capsys):
    # The pipeline carries an active install_override() (a converged/optimized
    # state) -> iterate-stage register resets it to a round-0 baseline derived
    # from reference.py (the "reset pipeline" banner).
    src = _make_source_task(tmp_path)
    (src / "pipeline" / "run.py").write_text(
        "install_override(foo, fast_foo)\n# optimized round\n")
    _git("add", "-A", "-f", cwd=src)
    _git("commit", "-q", "-m", "[algorithmic] optimize foo", cwd=src)
    bench.cmd_bench_register_template(
        _register_args(src, stage="iterate", commit="HEAD"))
    out = capsys.readouterr().out
    assert "reset pipeline" in out
    dest = templates_root / "iterate" / "tpl1"
    run_text = (dest / "pipeline" / "run.py").read_text()
    assert "install_override" not in run_text  # reset away from the optimized state


def test_register_template_explicit_commit(tmp_path, templates_root, capsys):
    src = _make_source_task(tmp_path)
    bench.cmd_bench_register_template(
        _register_args(src, commit="HEAD", stage="package", name="pkg"))
    out = capsys.readouterr().out
    assert "registered" in out
    meta = (templates_root / "package" / "pkg" / "bench_template.yaml").read_text()
    assert "explicit_commit:HEAD" in meta


def test_register_template_bad_commit_dies(tmp_path, templates_root, capsys):
    src = _make_source_task(tmp_path)
    with pytest.raises(SystemExit):
        bench.cmd_bench_register_template(
            _register_args(src, commit="nonexistent_ref_xyz"))


def test_register_template_init_no_commit_dies(tmp_path, templates_root, capsys):
    # A task whose first commit does NOT start with 'zyme init:' -> --at init
    # can't find a tagged commit -> die.
    d = tmp_path / "untagged"
    (d / "pipeline").mkdir(parents=True)
    (d / "task.yaml").write_text("datasets: []\n")
    (d / "pipeline" / "run.py").write_text("x\n")
    _git("init", "-q", cwd=d)
    _git("config", "user.email", "t@t", cwd=d)
    _git("config", "user.name", "t", cwd=d)
    _git("add", "-A", cwd=d)
    _git("commit", "-q", "-m", "just a commit", cwd=d)
    with pytest.raises(SystemExit):
        bench.cmd_bench_register_template(_register_args(d, at="init"))


def test_register_template_exists_without_force_dies(tmp_path, templates_root,
                                                     capsys):
    src = _make_source_task(tmp_path)
    bench.cmd_bench_register_template(_register_args(src))
    capsys.readouterr()
    # Second registration of the same name without --force -> die.
    with pytest.raises(SystemExit):
        bench.cmd_bench_register_template(_register_args(src))


def test_register_template_force_overwrites(tmp_path, templates_root, capsys):
    src = _make_source_task(tmp_path)
    bench.cmd_bench_register_template(_register_args(src))
    capsys.readouterr()
    bench.cmd_bench_register_template(_register_args(src, force=True))
    out = capsys.readouterr().out
    assert "registered" in out


def test_register_template_relative_path_rewrite(tmp_path, templates_root,
                                                 capsys):
    # A task.yaml with a relative `../shared/...` dataset path gets rewritten
    # to an absolute path in the template (path-rewrite banner).
    src = _make_source_task(tmp_path, relative_data=True)
    bench.cmd_bench_register_template(_register_args(src, stage="package"))
    out = capsys.readouterr().out
    assert "rewrote" in out and "relative data path" in out


# ---------------------------------------------------------------------------
# cmd_bench_start — dry-run + gates + start path
# ---------------------------------------------------------------------------

def _make_bench_run(tmp_path, *, with_reflect=True, name="run1"):
    run = tmp_path / "runs" / name
    run.mkdir(parents=True)
    (run / "bench_manifest.yaml").write_text(
        "bench_suite: demo\nprompt_id: \nprompt_slot: iterate\n")
    task = run / "fam_r1"
    (task / "prompts").mkdir(parents=True)
    (task / bench.ZYME_META_FILENAME).write_text(
        "bench_task: fam\ntemplate_name: fam\n")
    (task / "prompts" / "2_iterate.md").write_text("PROMPT\n")
    if with_reflect:
        (task / "reflect.md").write_text("REFLECT\n")
    return run, task


def _start_args(run, **over):
    base = dict(
        run=str(run), root=None, prompt="prompts/2_iterate.md",
        reflect_prompt=None, reflect_category=None, only=None,
        agent="claude", model="m", effort="high", max_rounds=10,
        force_mode=False, reflect=False, ram_floor="8G", disk_floor="auto",
        detach=False, dry_run=True, stall_threshold=900)
    base.update(over)
    return types.SimpleNamespace(**base)


def _patch_dispatch(monkeypatch, *, prior_pid=None, alive=False, parse=None,
                    start=None):
    import zyme.dispatch as zd
    import zyme.dispatch.state as zds
    import zyme.dispatch.resources as zdr
    monkeypatch.setattr(zd, "find_agent_binary",
                        lambda a: "/bin/" + a, raising=False)
    monkeypatch.setattr(zdr, "parse_size_gb",
                        parse or (lambda s: None if s == "auto" else 8.0))
    monkeypatch.setattr(zds, "read_pid", lambda p: prior_pid)
    monkeypatch.setattr(zds, "pid_alive", lambda pid: alive)
    monkeypatch.setattr(zds, "pid_path", lambda ws: ws / "pid")
    monkeypatch.setattr(zds, "state_path", lambda ws: ws / "state.json")
    monkeypatch.setattr(zds, "master_log_path", lambda ws: ws / "log")
    if start is not None:
        monkeypatch.setattr(zd, "start_dispatch", start, raising=False)


def test_bench_start_dry_run(tmp_path, monkeypatch, capsys):
    run, _ = _make_bench_run(tmp_path)
    _patch_dispatch(monkeypatch)
    bench.cmd_bench_start(_start_args(run))
    out = capsys.readouterr().out
    assert "bench run :" in out
    assert "queue" in out
    assert "fam_r1" in out


def test_bench_start_missing_prompt_dies(tmp_path, monkeypatch):
    run, _ = _make_bench_run(tmp_path)
    _patch_dispatch(monkeypatch)
    with pytest.raises(SystemExit):
        bench.cmd_bench_start(_start_args(run, prompt="prompts/ghost.md"))


def test_bench_start_missing_reflect_prompt_dies(tmp_path, monkeypatch):
    run, _ = _make_bench_run(tmp_path)
    _patch_dispatch(monkeypatch)
    with pytest.raises(SystemExit):
        bench.cmd_bench_start(_start_args(
            run, reflect=True, reflect_prompt="prompts/no_reflect.md"))


def test_bench_start_already_running_dies(tmp_path, monkeypatch):
    run, _ = _make_bench_run(tmp_path)
    _patch_dispatch(monkeypatch, prior_pid=4321, alive=True)
    with pytest.raises(SystemExit):
        bench.cmd_bench_start(_start_args(run))


def test_bench_start_bad_ram_floor_dies(tmp_path, monkeypatch):
    run, _ = _make_bench_run(tmp_path)

    def parse(s):
        if s == "garbage":
            raise ValueError("bad size")
        return None
    _patch_dispatch(monkeypatch, parse=parse)
    with pytest.raises(SystemExit):
        bench.cmd_bench_start(_start_args(run, ram_floor="garbage"))


def test_bench_start_explicit_disk_floor(tmp_path, monkeypatch, capsys):
    run, _ = _make_bench_run(tmp_path)
    _patch_dispatch(monkeypatch, parse=lambda s: 32.0)
    bench.cmd_bench_start(_start_args(run, disk_floor="32G"))
    assert "disk floor: 32.0 GB" in capsys.readouterr().out


def test_bench_start_reflect_dry_run(tmp_path, monkeypatch, capsys):
    run, _ = _make_bench_run(tmp_path)
    _patch_dispatch(monkeypatch)
    bench.cmd_bench_start(_start_args(
        run, reflect=True, reflect_prompt="reflect.md"))
    out = capsys.readouterr().out
    assert "reflect   : True" in out


def test_bench_start_non_dry_run_starts(tmp_path, monkeypatch, capsys):
    run, _ = _make_bench_run(tmp_path)
    captured = {}

    def fake_start(**kw):
        captured.update(kw)
        return 1234
    _patch_dispatch(monkeypatch, start=fake_start)
    # Stub the experiment-doc writer to avoid touching unrelated machinery.
    monkeypatch.setattr(bench, "_append_experiment_dispatch_doc",
                        lambda *a, **k: None)
    bench.cmd_bench_start(_start_args(run, dry_run=False, detach=True))
    out = capsys.readouterr().out
    assert "bench dispatch started (PID 1234" in out
    assert captured["prompt"] == "prompts/2_iterate.md"


# ---------------------------------------------------------------------------
# cmd_bench_status — rich branches
# ---------------------------------------------------------------------------

SUITE_YAML = """\
id: iterate_demo
stage: iterate
prompt_slot: iterate
field: Bio
description: demo
default_reps: 2

tasks:
  - id: fam
    template: fam_tpl
    probe: aggressiveness
"""


def _status_tree(tmp_path, monkeypatch):
    templates_root = tmp_path / "bench_templates"
    suites_root = tmp_path / "bench_suites"
    runs_root = tmp_path / "bench_runs"
    suites_root.mkdir()
    runs_root.mkdir()
    # A valid iterate template named fam_tpl.
    tpl = templates_root / "iterate" / "fam_tpl"
    (tpl / "data").mkdir(parents=True)
    (tpl / "data" / "x.h5ad").write_text("x")
    (tpl / "bench_template.yaml").write_text("stage: iterate\n")
    (tpl / "task.yaml").write_text(
        "datasets:\n  - {tier: tiny, name: tiny_a, path: data/x.h5ad}\n")
    (tpl / "reference.R").write_text("# ref\n")
    refout = tpl / "reference_outputs" / "tiny"
    refout.mkdir(parents=True)
    (refout / "out.txt").write_text("y")
    (tpl / "pipeline").mkdir()
    (tpl / "pipeline" / "run.R").write_text("# round0\n")

    monkeypatch.setattr(bench, "_bench_templates_root", lambda: templates_root)
    monkeypatch.setattr(bench, "_bench_suites_root", lambda: suites_root)
    monkeypatch.setattr(bench, "_default_bench_runs_root", lambda: runs_root)
    monkeypatch.setattr(bench, "_bench_suite_path",
                        lambda sid: suites_root / f"{sid}.yaml")
    (suites_root / "iterate_demo.yaml").write_text(SUITE_YAML)
    return types.SimpleNamespace(templates_root=templates_root,
                                 suites_root=suites_root, runs_root=runs_root,
                                 tpl=tpl)


def _make_status_run(runs_root, name="fam_run"):
    run = runs_root / name
    (run / "fam_r1").mkdir(parents=True)
    (run / "bench_manifest.yaml").write_text(
        "bench_suite: iterate_demo\nprompt_id: p1\nprompt_slot: iterate\n")
    task = run / "fam_r1"
    (task / bench.ZYME_META_FILENAME).write_text(
        "bench_task: fam\ntemplate_name: fam\n")
    # A results.tsv with a keep so best_seen renders.
    (task / "results.tsv").write_text(
        "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
        "metrics_json\thypothesis\tdescription\tphase\tthread\n"
        "0\tabc\ttiny_a\t10\t0\t100\tbaseline\t{}\tu\t\toptimize\t1\n"
        "1\tdef\ttiny_a\t6\t40\t100\tkeep\t{}\th\tk\toptimize\t1\n")
    return run


def test_status_with_suite_and_snapshots(tmp_path, monkeypatch, capsys):
    tree = _status_tree(tmp_path, monkeypatch)
    _make_status_run(tree.runs_root)
    import zyme.registry as registry
    # One snapshot present so the snapshot-table branch (1863-1870) renders.
    card = {"name": "snap_v1", "created_at": "2026-01-01T00:00:00"}
    monkeypatch.setattr(registry, "iter_snapshots",
                        lambda *a, **k: iter([("Bio", "iterate", "pid123", card)]))
    monkeypatch.setattr(registry, "read_active_lock",
                        lambda fr, f, s: "pid123")
    args = types.SimpleNamespace(
        suite_id="iterate_demo", root=None, only=None, limit=10,
        field=None, slot=None)
    bench.cmd_bench_status(args)
    out = capsys.readouterr().out
    assert "## prompt snapshots" in out
    assert "snap_v1" in out
    # template status line (OK) for the valid fam_tpl template.
    assert "fam" in out
    # bench-run row + task-progress table.
    assert "## bench runs" in out
    assert "## task progress" in out
    assert "fam_run" in out


def test_status_template_missing(tmp_path, monkeypatch, capsys):
    tree = _status_tree(tmp_path, monkeypatch)
    # Remove the template so the suite reports it "missing" (1882-1883).
    import shutil
    shutil.rmtree(tree.tpl)
    import zyme.registry as registry
    monkeypatch.setattr(registry, "iter_snapshots", lambda *a, **k: iter([]))
    args = types.SimpleNamespace(
        suite_id="iterate_demo", root=None, only=None, limit=10,
        field=None, slot=None)
    bench.cmd_bench_status(args)
    out = capsys.readouterr().out
    assert "missing" in out


def test_status_template_fail(tmp_path, monkeypatch, capsys):
    tree = _status_tree(tmp_path, monkeypatch)
    # Break the data file so the template check returns errors -> FAIL line.
    (tree.tpl / "data" / "x.h5ad").unlink()
    import zyme.registry as registry
    monkeypatch.setattr(registry, "iter_snapshots", lambda *a, **k: iter([]))
    args = types.SimpleNamespace(
        suite_id="iterate_demo", root=None, only=None, limit=10,
        field=None, slot=None)
    bench.cmd_bench_status(args)
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_status_no_suite_stage_counts(tmp_path, monkeypatch, capsys):
    tree = _status_tree(tmp_path, monkeypatch)
    import zyme.registry as registry
    monkeypatch.setattr(registry, "iter_snapshots", lambda *a, **k: iter([]))
    # No suite_id -> the no-suite stage-count branch (1893-1900) lists stages.
    args = types.SimpleNamespace(
        suite_id=None, root=None, only=None, limit=10, field=None, slot=None)
    bench.cmd_bench_status(args)
    out = capsys.readouterr().out
    assert "iterate" in out
    assert "template(s)" in out


def test_status_no_runs_message(tmp_path, monkeypatch, capsys):
    tree = _status_tree(tmp_path, monkeypatch)
    import zyme.registry as registry
    monkeypatch.setattr(registry, "iter_snapshots", lambda *a, **k: iter([]))
    args = types.SimpleNamespace(
        suite_id="iterate_demo", root=None, only=None, limit=10,
        field=None, slot=None)
    bench.cmd_bench_status(args)
    out = capsys.readouterr().out
    assert "no bench runs" in out


# ---------------------------------------------------------------------------
# cmd_bench_init — full SUITE mode (multi-rep, non-name) end to end
# ---------------------------------------------------------------------------

INIT_SUITE_YAML = """\
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


def _init_template_tree(tmp_path, monkeypatch):
    templates_root = tmp_path / "bench_templates"
    suites_root = tmp_path / "bench_suites"
    runs_root = tmp_path / "bench_runs"
    suites_root.mkdir()
    runs_root.mkdir()
    stage_dir = templates_root / "iterate"
    for name in ("findallmarker", "mast_tpl"):
        tpl = stage_dir / name
        tpl.mkdir(parents=True)
        (tpl / "bench_template.yaml").write_text("stage: iterate\n")
        (tpl / "task.yaml").write_text(
            "datasets:\n  - {tier: tiny, name: tiny_a, path: data/tiny.h5ad}\n")
        (tpl / "reference.R").write_text("# ref\n")
        (tpl / "prompts").mkdir()
        (tpl / "prompts" / "2_iterate.md").write_text("orig prompt\n")
        (tpl / "memory").mkdir()
        (tpl / "memory" / "discoveries.md").write_text("# disc\n")
    monkeypatch.setattr(bench, "_bench_templates_root", lambda: templates_root)
    monkeypatch.setattr(bench, "_bench_suites_root", lambda: suites_root)
    monkeypatch.setattr(bench, "_default_bench_runs_root", lambda: runs_root)
    (suites_root / "iterate_demo.yaml").write_text(INIT_SUITE_YAML)
    monkeypatch.setattr(bench, "_bench_suite_path",
                        lambda sid: suites_root / f"{sid}.yaml")
    return types.SimpleNamespace(templates_root=templates_root,
                                 suites_root=suites_root, runs_root=runs_root,
                                 stage_dir=stage_dir)


def _stub_snapshot(tmp_path, monkeypatch, *, slot="iterate", field="Bio"):
    from zyme import registry
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "prompt.md").write_text("SNAPSHOT BODY")
    live = tmp_path / "live_prompts"
    live.mkdir()
    (live / "2_iterate.md").write_text("live iterate\n")
    card = {"source_path": "prompts/2_iterate.md", "name": "p1",
            "content_sha256": "abc123"}
    monkeypatch.setattr(registry, "find_snapshot",
                        lambda root, pid: (field, slot, "bio/iterate/p1", card))
    monkeypatch.setattr(registry, "snapshot_dir_for",
                        lambda root, f, s, pid: snap)
    monkeypatch.setattr(registry, "live_prompts_dir", lambda root, f: live)
    return card


def _init_args(**over):
    base = dict(suite_id="iterate_demo", prompt_id="bio/iterate/p1", only=None,
                reps=None, name=None, out=None, force=False, purpose=None)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_init_suite_mode_multi_rep_end_to_end(tmp_path, monkeypatch, capsys):
    _init_template_tree(tmp_path, monkeypatch)
    _stub_snapshot(tmp_path, monkeypatch)
    # Real git init per replicate (local); framework link materialization runs.
    out = tmp_path / "run_out"
    args = _init_args(only=["findallmarker"], reps=2, out=str(out),
                      purpose="bench the iterate prompt")
    bench.cmd_bench_init(args)
    text = capsys.readouterr().out
    assert "scaffolding 1 task(s) × 2 rep(s)" in text
    assert "purpose:" in text
    # Suite-level manifest + EXPERIMENT.md written at out_root.
    assert (out / "bench_manifest.yaml").exists()
    assert (out / bench.EXPERIMENT_FILENAME).exists()
    # Two replicate dirs scaffolded, each with the snapshot prompt + a .git.
    for r in (1, 2):
        rep = out / f"findallmarker_r{r}"
        assert rep.is_dir()
        assert (rep / bench.ZYME_META_FILENAME).exists()
        assert (rep / "prompts" / "2_iterate.md").read_text() == "SNAPSHOT BODY"
        assert (rep / ".git").exists()


def test_init_suite_mode_out_exists_without_force_dies(tmp_path, monkeypatch):
    _init_template_tree(tmp_path, monkeypatch)
    _stub_snapshot(tmp_path, monkeypatch)
    out = tmp_path / "preexisting"
    out.mkdir()
    args = _init_args(only=["findallmarker"], out=str(out))
    with pytest.raises(SystemExit):
        bench.cmd_bench_init(args)


def test_init_prompt_snapshot_not_found_dies(tmp_path, monkeypatch):
    _init_template_tree(tmp_path, monkeypatch)
    from zyme import registry
    monkeypatch.setattr(registry, "find_snapshot", lambda root, pid: None)
    with pytest.raises(SystemExit):
        bench.cmd_bench_init(_init_args(only=["findallmarker"]))
