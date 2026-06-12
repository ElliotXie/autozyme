"""End-to-end CLI tests — actually invoke `python -m zyme ...` against a
real fake task in a tmp git repo. Asserts on results.tsv rows, .zyme state
files, and HEAD/git invariants.

Ported from the old `test/smoke_test.sh` shell suite — the bash script and
pytest split was duplicated maintenance, and shell assertions are harder to
debug than pytest's introspection. Each shell test_* function is one
function below; tier 18+19 (the dropped `--datasets` flag) became a single
`--datasets is rejected` regression plus the `--extra-tiers` replacement.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


# --------------------------------------------------------------------------
# Subprocess helper
# --------------------------------------------------------------------------

def zyme(*argv: str, cwd: Path | None = None, check: bool = False,
         input_text: str | None = None) -> subprocess.CompletedProcess:
    """Invoke `python -m zyme <argv>`. Returns CompletedProcess.

    `check=True` asserts rc==0; default returns whatever rc came back so
    the caller can assert on failure paths.
    """
    cp = subprocess.run(
        [sys.executable, "-m", "zyme", *argv],
        cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, timeout=60,
        input=input_text,
    )
    if check and cp.returncode != 0:
        raise AssertionError(
            f"zyme {' '.join(argv)} exit {cp.returncode}\n"
            f"stdout: {cp.stdout}\nstderr: {cp.stderr}"
        )
    return cp


def _git(*argv: str, cwd: Path) -> str:
    """Run git, return stdout (raises on non-zero)."""
    cp = subprocess.run(
        ["git", *argv], cwd=str(cwd),
        capture_output=True, text=True, check=True,
    )
    return cp.stdout.strip()


def _row(tsv: Path, n: int) -> list[str]:
    """Return TSV row n (1-indexed; header is row 1, first data row is 2)."""
    lines = tsv.read_text().splitlines()
    return lines[n - 1].split("\t")


def _data_rows(tsv: Path) -> int:
    """Count data rows in TSV (excludes header)."""
    return max(0, len(tsv.read_text().splitlines()) - 1)


# --------------------------------------------------------------------------
# Single-tier task fixture (mirrors shell setup_task)
# --------------------------------------------------------------------------

_REFERENCE_PY = """\
import os, time
os.makedirs("reference_output", exist_ok=True)
start = time.perf_counter()
time.sleep(0.005)
with open("reference_output/value.txt", "w") as f:
    f.write("42\\n")
print(f"speed_sec: {time.perf_counter() - start:.6f}")
"""

_EVALUATE_PY = """\
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ref = open(os.path.join(SCRIPT_DIR, "reference_output", "value.txt")).read().strip()
test = open(os.path.join(SCRIPT_DIR, "pipeline", "value.txt")).read().strip()
print(f"value_match: {1.0 if test == ref else 0.0}")
"""

_PIPELINE_RUN_PY = """\
import time
start = time.perf_counter()
time.sleep(0.005)
with open("value.txt", "w") as f:
    f.write("42\\n")
print(f"speed_sec: {time.perf_counter() - start:.6f}")
"""


def _pipeline_fixed_speed(speed: float, cpu: float | None = None) -> str:
    cpu_line = f"print('cpu_sec: {cpu}')\n" if cpu is not None else ""
    return f"""\
with open("value.txt", "w") as f:
    f.write("42\\n")
print("speed_sec: {speed}")
{cpu_line}"""

_TASK_YAML_SINGLE = """\
task: smoke
datasets:
  - {name: dummy, path: /dev/null}
metrics:
  - {name: value_match, threshold: 1.0, comparator: gte}
"""

_GITIGNORE = """\
.zyme/
reference_output/
artifacts/
results.tsv
discoveries.md
active_opts.md
dead_ends.md
round_log.md
round_log_archive.md
run.log
pipeline/value.txt
__pycache__/
"""


@pytest.fixture
def task(tmp_path: Path) -> Path:
    """A fresh tmp task with single-tier (`dummy`) dataset, populated
    reference_output, and an initial git commit. Returns the task dir."""
    td = tmp_path / "smoke"
    (td / "pipeline").mkdir(parents=True)
    (td / "task.yaml").write_text(_TASK_YAML_SINGLE)
    (td / "reference.py").write_text(_REFERENCE_PY)
    (td / "evaluate.py").write_text(_EVALUATE_PY)
    (td / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY)
    (td / ".gitignore").write_text(_GITIGNORE)

    # Run reference once to populate reference_output/, before init commit.
    subprocess.run([sys.executable, "reference.py"], cwd=str(td),
                   capture_output=True, check=True)
    _git("init", "-q", cwd=td)
    _git("config", "user.email", "smoke@test", cwd=td)
    _git("config", "user.name", "Smoke Test", cwd=td)
    _git("add", ".", cwd=td)
    _git("commit", "-q", "-m", "initial: smoke task scaffold", cwd=td)
    return td


# --------------------------------------------------------------------------
# Multi-tier task fixture (mirrors shell setup_multitier_task)
# --------------------------------------------------------------------------

_TASK_YAML_MULTI = """\
task: smoke_mt
datasets:
  - {tier: tiny,   name: ds_tiny,   path: ./data/tiny.txt}
  - {tier: medium, name: ds_medium, path: ./data/medium.txt}
metrics:
  - {name: value_match, threshold: 1.0, comparator: gte}
"""

_REFERENCE_MT_PY = """\
import os, time
tier = os.environ.get("ZYME_TIER", "tiny")
ref_dir = os.environ.get("ZYME_REFERENCE_DIR", "reference_output")
data_path = os.environ.get("ZYME_DATA_PATH", "./data/tiny.txt")
os.makedirs(ref_dir, exist_ok=True)
start = time.perf_counter()
time.sleep(0.005)
content = open(data_path).read().strip()
with open(os.path.join(ref_dir, "value.txt"), "w") as f:
    f.write(content + "\\n")
print(f"speed_sec: {time.perf_counter() - start:.6f}")
"""

_EVALUATE_MT_PY = """\
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ref_dir = os.environ.get("ZYME_REFERENCE_DIR") or os.path.join(SCRIPT_DIR, "reference_output")
ref = open(os.path.join(ref_dir, "value.txt")).read().strip()
test = open(os.path.join(SCRIPT_DIR, "pipeline", "value.txt")).read().strip()
print(f"value_match: {1.0 if test == ref else 0.0}")
"""

_PIPELINE_RUN_MT_PY = """\
import os, time
data_path = os.environ.get("ZYME_DATA_PATH", "../data/tiny.txt")
if not os.path.isabs(data_path):
    data_path = os.path.join("..", data_path)
start = time.perf_counter()
time.sleep(0.005)
content = open(data_path).read().strip()
with open("value.txt", "w") as f:
    f.write(content + "\\n")
print(f"speed_sec: {time.perf_counter() - start:.6f}")
"""

_GITIGNORE_MT = """\
.zyme/
reference_output/
reference_output_tiny/
reference_output_medium/
reference_outputs/
artifacts/
results.tsv
discoveries.md
active_opts.md
dead_ends.md
round_log.md
round_log_archive.md
run.log
pipeline/value.txt
__pycache__/
"""


@pytest.fixture
def multitier_task(tmp_path: Path) -> Path:
    """Two-tier task (tiny + medium) with per-tier reference outputs and a
    pipeline that varies on ZYME_DATA_PATH. Mirrors shell setup_multitier_task."""
    td = tmp_path / "smoke_mt"
    (td / "pipeline").mkdir(parents=True)
    (td / "data").mkdir()
    (td / "data" / "tiny.txt").write_text("tiny_data\n")
    (td / "data" / "medium.txt").write_text("medium_data\n")
    (td / "task.yaml").write_text(_TASK_YAML_MULTI)
    (td / "reference.py").write_text(_REFERENCE_MT_PY)
    (td / "evaluate.py").write_text(_EVALUATE_MT_PY)
    (td / "pipeline" / "run.py").write_text(_PIPELINE_RUN_MT_PY)
    (td / ".gitignore").write_text(_GITIGNORE_MT)

    # Populate reference_outputs/<tier>/value.txt for both tiers (K2 layout).
    for tier_name, src in (("tiny", "./data/tiny.txt"),
                            ("medium", "./data/medium.txt")):
        ref_dir = td / "reference_outputs" / tier_name
        env = {**os.environ,
               "ZYME_TIER": tier_name,
               "ZYME_DATA_PATH": src,
               "ZYME_REFERENCE_DIR": str(ref_dir)}
        subprocess.run([sys.executable, "reference.py"], cwd=str(td),
                       env=env, capture_output=True, check=True)

    _git("init", "-q", cwd=td)
    _git("config", "user.email", "smoke@test", cwd=td)
    _git("config", "user.name", "Smoke Test", cwd=td)
    _git("add", ".", cwd=td)
    _git("commit", "-q", "-m", "initial: multitier smoke task", cwd=td)
    return td


# --------------------------------------------------------------------------
# Tests — happy path & decision flow
# --------------------------------------------------------------------------

def test_happy_path(task: Path):
    """edit → run → accept → edit → run → reject."""
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r1\n")
    zyme("run", "--task-dir", str(task), "r1: baseline", check=True)
    zyme("accept", "--task-dir", str(task), "-m", "baseline", check=True)

    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r2\n")
    zyme("run", "--task-dir", str(task), "r2: tweak", check=True)
    zyme("reject", "--task-dir", str(task), "-m", "no gain",
         "--force", check=True)

    tsv = task / "results.tsv"
    assert _row(tsv, 2)[6] == "keep"
    assert _row(tsv, 3)[6] == "discard"
    assert any((task / "artifacts").glob("001*"))
    assert any((task / "artifacts").glob("002*"))


def test_pipeline_crash(task: Path):
    """Pipeline that raises → status=crash row."""
    (task / "pipeline" / "run.py").write_text(
        'raise RuntimeError("intentional pipeline boom")\n'
    )
    zyme("run", "--task-dir", str(task), "intentional crash")
    assert _row(task / "results.tsv", 2)[6] == "crash"


def test_evaluate_crash(task: Path):
    """Pipeline runs but doesn't write expected output → evaluate fails → crash."""
    (task / "pipeline" / "run.py").write_text(
        'print("speed_sec: 0.001")\n'
    )
    zyme("run", "--task-dir", str(task), "pipeline misses output file")
    assert _row(task / "results.tsv", 2)[6] == "crash"


def test_missing_reference_output(task: Path):
    """Removing reference_output → run fails cleanly, status=crash."""
    import shutil
    shutil.rmtree(task / "reference_output")
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r1\n")
    cp = zyme("run", "--task-dir", str(task), "r1")
    assert "reference_output empty" in (cp.stdout + cp.stderr)
    assert _row(task / "results.tsv", 2)[6] == "crash"


def test_no_changes_blocks_run(task: Path):
    """zyme run with no edits to pipeline → must error."""
    cp = zyme("run", "--task-dir", str(task), "empty")
    assert cp.returncode != 0
    assert "unchanged" in (cp.stdout + cp.stderr)


def test_outside_task_dir(tmp_path: Path):
    """zyme outside a task dir (no task.yaml) → must error."""
    cp = zyme("run", "--task-dir", str(tmp_path), "x")
    assert cp.returncode != 0
    assert "task.yaml" in (cp.stdout + cp.stderr)


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------

def test_init_command(tmp_path: Path):
    """zyme init scaffolds task flat in cwd (cwd basename = task name)."""
    task_root = tmp_path / "scrublet_doublets"
    task_root.mkdir()
    _git("init", "-q", cwd=task_root)
    _git("config", "user.email", "smoke@test", cwd=task_root)
    _git("config", "user.name", "Smoke Test", cwd=task_root)

    zyme("init", "scrublet", "scrublet.simulate_doublets",
         "--no-clone", "--no-bench-snapshot",
         cwd=task_root, check=True)

    assert (task_root / "task.yaml").exists()
    assert (task_root / "README.md").exists()
    assert (task_root / "pipeline" / "run.py.template").exists()
    assert not (task_root / "tasks").exists()


def test_init_scaffolds_split_notes(tmp_path: Path):
    """init scaffolds memory/{discoveries,active_opts,dead_ends}.md (not legacy notes.md)."""
    task_root = tmp_path / "scrublet_doublets"
    task_root.mkdir()
    _git("init", "-q", cwd=task_root)
    _git("config", "user.email", "smoke@test", cwd=task_root)
    _git("config", "user.name", "Smoke Test", cwd=task_root)

    zyme("init", "scrublet", "--no-clone", "--no-bench-snapshot",
         cwd=task_root, check=True)

    assert (task_root / "memory" / "discoveries.md").exists()
    assert (task_root / "memory" / "active_opts.md").exists()
    assert (task_root / "memory" / "dead_ends.md").exists()
    assert not (task_root / "notes.md").exists()
    # results.tsv is created lazily on first `zyme run`, not by init.


# --------------------------------------------------------------------------
# accept / reject behavior
# --------------------------------------------------------------------------

def test_multiple_accepts(task: Path):
    """Three consecutive accepts: best.ref advances each round."""
    prev_best = None
    for i in range(1, 4):
        (task / "pipeline" / "run.py").write_text(
            _PIPELINE_RUN_PY + f"# round {i}\n"
        )
        zyme("run", "--task-dir", str(task), f"r{i}", check=True)
        zyme("accept", "--task-dir", str(task), "-m", f"r{i} ok", check=True)
        best = (task / ".zyme" / "best.ref").read_text().strip()
        if prev_best is not None:
            assert best != prev_best, f"best didn't advance on round {i}"
        prev_best = best
    assert _data_rows(task / "results.tsv") == 3


def test_reject_auto_restores(task: Path):
    """reject restores pipeline working tree to best (= round 1 state)."""
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r1 baseline\n")
    zyme("run", "--task-dir", str(task), "r1", check=True)
    zyme("accept", "--task-dir", str(task), "-m", "r1 keep", check=True)

    pipeline_at_best = (task / "pipeline" / "run.py").read_text()

    (task / "pipeline" / "run.py").write_text(
        _PIPELINE_RUN_PY + "# r1 baseline\n# r2 garbage\n"
    )
    zyme("run", "--task-dir", str(task), "r2 broken", check=True)
    zyme("reject", "--task-dir", str(task), "-m", "discard",
         "--force", check=True)
    assert (task / "pipeline" / "run.py").read_text() == pipeline_at_best


def test_accept_writes_best_state(task: Path):
    """accept writes memory/best_state.md; updates after second keep."""
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r1\n")
    zyme("run", "--task-dir", str(task), "r1: first hypothesis", check=True)
    zyme("accept", "--task-dir", str(task), "-m",
         "first keep, baseline-tweak +5%", check=True)

    bs = task / "memory" / "best_state.md"
    assert bs.exists()
    c = bs.read_text()
    assert "# Current best state" in c
    assert "Latest keep: round 1" in c
    assert "Patch stack" in c
    assert "baseline-tweak" in c

    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r2\n")
    zyme("run", "--task-dir", str(task), "r2: second hypothesis", check=True)
    zyme("accept", "--task-dir", str(task), "-m",
         "second keep, another +3%", check=True)
    c2 = bs.read_text()
    assert "Latest keep: round 2" in c2
    assert "2 keeps" in c2


# --------------------------------------------------------------------------
# Wall clock fallback + schema
# --------------------------------------------------------------------------

def test_wall_clock_fallback(task: Path):
    """No speed_sec print → runner falls back to wall-clock time."""
    (task / "pipeline" / "run.py").write_text(textwrap.dedent("""
        import time
        time.sleep(0.05)
        with open("value.txt", "w") as f:
            f.write("42\\n")
    """))
    zyme("run", "--task-dir", str(task), "no speed_sec print", check=True)
    speed = float(_row(task / "results.tsv", 2)[3])
    assert 0.01 < speed < 30.0, f"speed_sec wall fallback wrong: {speed}"


def test_hypothesis_column_populated(task: Path):
    """Header has hypothesis at col 9, description at col 10; row stores it."""
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r1\n")
    zyme("run", "--task-dir", str(task), "vectorize the inner loop", check=True)

    header = _row(task / "results.tsv", 1)
    assert header[8] == "hypothesis"
    assert header[9] == "description"
    assert _row(task / "results.tsv", 2)[8] == "vectorize the inner loop"


def test_generated_pipeline_output_not_committed_or_audited(task: Path):
    """pipeline/output_<tier>/ files are generated artifacts, not source edits."""
    out_dir = task / "pipeline" / "output_tiny"
    out_dir.mkdir()
    (out_dir / "result.rds").write_text("generated\n")
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r1\n")

    cp = zyme("run", "--task-dir", str(task), "r1", check=True)

    committed = _git(
        "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD", cwd=task
    )
    assert "pipeline/output_tiny/result.rds" not in committed.splitlines()
    assert "output_tiny" not in (cp.stdout + cp.stderr)


def test_run_rejects_task_definition_changes(task: Path):
    """Ordinary optimization rounds cannot smuggle evaluator edits."""
    (task / "evaluate.py").write_text(_EVALUATE_PY + "\n# repair attempt\n")
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r1\n")

    cp = zyme("run", "--task-dir", str(task), "r1")

    assert cp.returncode != 0
    out = cp.stdout + cp.stderr
    assert "task-definition files changed outside --setup" in out
    assert "evaluate.py" in out


def test_setup_preserves_evaluator_across_reject(task: Path):
    """A setup evaluator repair becomes the reject reset floor."""
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# best\n")
    zyme("run", "--task-dir", str(task), "r1 best", check=True)
    zyme("accept", "--task-dir", str(task), "-m", "r1 keep", check=True)

    marker = "# setup evaluator repair\n"
    (task / "evaluate.py").write_text(_EVALUATE_PY + "\n" + marker)

    cp_setup = zyme(
        "run", "--task-dir", str(task), "--setup",
        "repair evaluator for generated outputs",
        check=True,
    )
    assert "task truth surface changed" in (cp_setup.stdout + cp_setup.stderr)
    assert (task / ".zyme" / "setup_audit.log").exists()

    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# reject me\n")
    zyme("run", "--task-dir", str(task), "r2", check=True)
    zyme("reject", "--task-dir", str(task), "-m", "discard",
         "--force", check=True)

    assert marker in (task / "evaluate.py").read_text()


def test_setup_refuses_pending_round(task: Path):
    """Do not change truth surface between measurement and accept/reject."""
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# pending\n")
    zyme("run", "--task-dir", str(task), "r1", check=True)
    (task / "evaluate.py").write_text(_EVALUATE_PY + "\n# late repair\n")

    cp = zyme("run", "--task-dir", str(task), "--setup", "late evaluator repair")

    assert cp.returncode != 0
    assert "pending decision round" in (cp.stdout + cp.stderr)


# --------------------------------------------------------------------------
# --rerun
# --------------------------------------------------------------------------

def test_rerun_no_new_commit(task: Path):
    """--rerun --n 1 doesn't create a new commit; row shares the same SHA."""
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r1\n")
    zyme("run", "--task-dir", str(task), "r1", check=True)
    zyme("accept", "--task-dir", str(task), "-m", "r1 keep", check=True)

    sha_before = _git("rev-parse", "HEAD", cwd=task)
    n_before = int(_git("rev-list", "--count", "HEAD", cwd=task))

    zyme("run", "--rerun", "--n", "1", "--task-dir", str(task), check=True)

    assert _git("rev-parse", "HEAD", cwd=task) == sha_before
    assert int(_git("rev-list", "--count", "HEAD", cwd=task)) == n_before
    tsv = task / "results.tsv"
    assert _data_rows(tsv) == 2
    assert _row(tsv, 2)[1] == _row(tsv, 3)[1]
    assert _row(tsv, 3)[6] == "rerun"


def test_rerun_rejects_hypothesis(task: Path):
    """--rerun + positional hypothesis → error."""
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r1\n")
    zyme("run", "--task-dir", str(task), "r1", check=True)
    zyme("accept", "--task-dir", str(task), "-m", "r1 keep", check=True)

    cp = zyme("run", "--rerun", "--task-dir", str(task), "should not pass hyp")
    assert cp.returncode != 0
    assert "rerun" in (cp.stdout + cp.stderr)


def test_rerun_rejects_task_definition_changes(task: Path):
    """Rerun also requires the committed task definition."""
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r1\n")
    zyme("run", "--task-dir", str(task), "r1", check=True)
    zyme("accept", "--task-dir", str(task), "-m", "r1 keep", check=True)
    (task / "evaluate.py").write_text(_EVALUATE_PY + "\n# dirty evaluator\n")

    cp = zyme("run", "--rerun", "--n", "1", "--task-dir", str(task))

    assert cp.returncode != 0
    assert "task-definition files changed outside --setup" in (cp.stdout + cp.stderr)


def test_rerun_default_n1(task: Path):
    """`zyme run --rerun` (no --n) defaults to one quick re-check."""
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r1\n")
    zyme("run", "--task-dir", str(task), "r1", check=True)
    zyme("accept", "--task-dir", str(task), "-m", "r1 keep", check=True)

    cp = zyme("run", "--rerun", "--task-dir", str(task), check=True)
    tsv = task / "results.tsv"
    assert _data_rows(tsv) == 2  # 1 keep + 1 rerun
    assert _row(tsv, 3)[6] == "rerun"
    assert "1 rerun row(s)" in cp.stdout


def test_accept_after_rerun_flips_pending(task: Path):
    """accept must skip status=rerun rows and flip the original pending row."""
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r1\n")
    zyme("run", "--task-dir", str(task), "r1: hypothesis", check=True)

    # Two invocations give two rerun rows without relying on default reps.
    zyme("run", "--rerun", "--n", "1", "--task-dir", str(task), check=True)
    zyme("run", "--rerun", "--n", "1", "--task-dir", str(task), check=True)

    zyme("accept", "--task-dir", str(task), "-m",
         "kept after stability check", check=True)

    tsv = task / "results.tsv"
    assert _row(tsv, 2)[6] == "keep"
    assert _row(tsv, 3)[6] == "rerun"
    assert _row(tsv, 4)[6] == "rerun"
    assert "stability check" in _row(tsv, 2)[9]


def test_decision_rounds_excludes_reruns(task: Path):
    """rerun rows don't count toward the 50-round budget."""
    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r1\n")
    zyme("run", "--task-dir", str(task), "r1", check=True)
    zyme("accept", "--task-dir", str(task), "-m", "ok", check=True)
    zyme("run", "--rerun", "--n", "1", "--task-dir", str(task), check=True)
    zyme("run", "--rerun", "--n", "1", "--task-dir", str(task), check=True)

    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# r2\n")
    cp = zyme("run", "--task-dir", str(task), "r2", check=True)
    assert "decision rounds: 2/50" in cp.stdout.lower() + cp.stderr.lower()


def test_wall_cpu_low_ratio_reports_parallelism_not_host_load(task: Path):
    """CPU > wall is expected for parallel kernels and should not say host load."""
    (task / "pipeline" / "run.py").write_text(_pipeline_fixed_speed(1.0, cpu=1.0))
    zyme("run", "--task-dir", str(task), "r1", check=True)
    zyme("accept", "--task-dir", str(task), "-m", "r1 keep", check=True)
    zyme("run", "--rerun", "--n", "1", "--task-dir", str(task), check=True)

    (task / "pipeline" / "run.py").write_text(_pipeline_fixed_speed(0.2, cpu=1.0))
    cp = zyme("run", "--task-dir", str(task), "r2 parallel", check=True)
    out = cp.stdout + cp.stderr
    assert "parallelism detected" in out
    assert "host load suspect" not in out


def test_rerun_estimate_prefers_current_head_measurements(task: Path):
    """A fast pending HEAD should estimate from HEAD, not stale baseline rows."""
    (task / "pipeline" / "run.py").write_text(_pipeline_fixed_speed(60.0))
    zyme("run", "--task-dir", str(task), "r1 slow", check=True)
    zyme("accept", "--task-dir", str(task), "-m", "r1 keep", check=True)

    (task / "pipeline" / "run.py").write_text(_pipeline_fixed_speed(0.4))
    zyme("run", "--task-dir", str(task), "r2 fast", check=True)
    cp = zyme("run", "--rerun", "--n", "2", "--task-dir", str(task), check=True)
    out = cp.stdout + cp.stderr
    assert "current HEAD median, n=1" in out
    assert "~0.4s" in out


# --------------------------------------------------------------------------
# dryrun
# --------------------------------------------------------------------------

def test_dryrun_no_state_change(task: Path):
    """dryrun runs pipeline but doesn't commit, write results.tsv, or counter."""
    sha_before = _git("rev-parse", "HEAD", cwd=task)
    n_before = int(_git("rev-list", "--count", "HEAD", cwd=task))

    (task / "pipeline" / "run.py").write_text(_PIPELINE_RUN_PY + "# dryrun probe\n")
    zyme("dryrun", "--task-dir", str(task), check=True)

    assert _git("rev-parse", "HEAD", cwd=task) == sha_before
    assert int(_git("rev-list", "--count", "HEAD", cwd=task)) == n_before
    assert not (task / "results.tsv").exists()
    assert not (task / ".zyme" / "round.counter").exists()


# --------------------------------------------------------------------------
# Multi-tier: --dataset / --extra-tiers
# --------------------------------------------------------------------------

def test_dataset_flag_picks_tier(multitier_task: Path):
    """--dataset medium routes to the medium tier."""
    (multitier_task / "pipeline" / "run.py").write_text(
        _PIPELINE_RUN_MT_PY + "# tweak\n"
    )
    zyme("run", "--task-dir", str(multitier_task),
         "r1: medium tier", "--dataset", "medium", check=True)
    assert _row(multitier_task / "results.tsv", 2)[2] == "ds_medium"


def test_extra_tiers_writes_two_rows(multitier_task: Path):
    """--dataset tiny + --extra-tiers medium → 1 pending + 1 rerun row, same commit."""
    (multitier_task / "pipeline" / "run.py").write_text(
        _PIPELINE_RUN_MT_PY + "# tweak\n"
    )
    zyme("run", "--task-dir", str(multitier_task), "r1: both tiers",
         "--dataset", "tiny", "--extra-tiers", "medium", check=True)

    tsv = multitier_task / "results.tsv"
    assert _data_rows(tsv) == 2
    assert _row(tsv, 2)[2] == "ds_tiny"
    assert _row(tsv, 3)[2] == "ds_medium"
    assert _row(tsv, 2)[6] == "pending"
    assert _row(tsv, 3)[6] == "rerun"
    assert _row(tsv, 2)[1] == _row(tsv, 3)[1]  # same commit

    zyme("accept", "--task-dir", str(multitier_task), "-m",
         "validated at both tiers", check=True)
    assert _row(tsv, 2)[6] == "keep"
    assert _row(tsv, 3)[6] == "rerun"


def test_datasets_flag_rejected(multitier_task: Path):
    """The old `--datasets a,b` flag was removed in favor of --dataset + --extra-tiers."""
    cp = zyme("run", "--task-dir", str(multitier_task), "x",
              "--datasets", "tiny,medium")
    assert cp.returncode != 0
    assert "unrecognized arguments" in cp.stderr or "--datasets" in cp.stderr


def test_unknown_tier_errors(multitier_task: Path):
    """--dataset <unknown> errors cleanly with the missing name surfaced."""
    (multitier_task / "pipeline" / "run.py").write_text(
        _PIPELINE_RUN_MT_PY + "# tweak\n"
    )
    cp = zyme("run", "--task-dir", str(multitier_task), "x", "--dataset", "huge")
    assert cp.returncode != 0
    assert "huge" in (cp.stdout + cp.stderr)
