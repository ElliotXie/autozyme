"""Unit tests for zyme.commands.baseline_rebench.

Covers the pure helpers (verify.tsv migration + baseline-divisor rewriting)
directly, and drives cmd_baseline_rebench end-to-end with the reference
subprocess boundary monkeypatched so the orchestration / record / verify-update
logic runs without launching anything.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from zyme.commands import baseline_rebench as br


# ---------------------------------------------------------------------------
# _migrate_verify_tsv_for_k2
# ---------------------------------------------------------------------------

def test_migrate_verify_tsv_noop_when_missing(tmp_path: Path):
    # Should silently return for a non-existent path.
    br._migrate_verify_tsv_for_k2(tmp_path / "nope.tsv")


def test_migrate_verify_tsv_noop_when_thread_column_present(tmp_path: Path):
    vt = tmp_path / "verify.tsv"
    original = (
        "timestamp\tthread\ttier\tspeed_sec\tbaseline_speed\tspeedup_pct\n"
        "2026-01-01\t1\ttiny\t5.0\t10.0\t50.0\n"
    )
    vt.write_text(original, encoding="utf-8")
    br._migrate_verify_tsv_for_k2(vt)
    # Unchanged; no .prek2.bak created.
    assert vt.read_text(encoding="utf-8") == original
    assert not (tmp_path / "verify.tsv.prek2.bak").exists()


def test_migrate_verify_tsv_backfills_thread_and_snapshots(tmp_path: Path):
    vt = tmp_path / "verify.tsv"
    vt.write_text(
        "timestamp\ttier\tspeed_sec\tbaseline_speed\tspeedup_pct\n"
        "2026-01-01\ttiny\t5.0\t10.0\t50.0\n"
        "\n"  # blank line preserved
        "2026-01-02\tmedium\t8.0\t16.0\t50.0\n",
        encoding="utf-8",
    )
    br._migrate_verify_tsv_for_k2(vt)

    lines = vt.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == [
        "timestamp", "thread", "tier", "speed_sec", "baseline_speed", "speedup_pct",
    ]
    assert lines[1].split("\t")[:3] == ["2026-01-01", "1", "tiny"]
    assert lines[2] == ""  # blank passthrough
    assert lines[3].split("\t")[:3] == ["2026-01-02", "1", "medium"]
    # .prek2.bak snapshot saved with original content.
    bak = tmp_path / "verify.tsv.prek2.bak"
    assert bak.exists()
    assert "tier\tspeed_sec" in bak.read_text(encoding="utf-8")


def test_migrate_verify_tsv_empty_file(tmp_path: Path):
    vt = tmp_path / "verify.tsv"
    vt.write_text("", encoding="utf-8")
    br._migrate_verify_tsv_for_k2(vt)  # no crash, no .bak
    assert not (tmp_path / "verify.tsv.prek2.bak").exists()


# ---------------------------------------------------------------------------
# _update_verify_tsv_baselines
# ---------------------------------------------------------------------------

VERIFY_HEADER = "tier\tthread\tspeed_sec\tbaseline_speed\tspeedup_pct"


def _write_verify(tmp_path, rows):
    vt = tmp_path / "verify.tsv"
    body = "\n".join(["\t".join(map(str, r)) for r in rows])
    vt.write_text(VERIFY_HEADER + "\n" + body + "\n", encoding="utf-8")
    return vt


def test_update_verify_baselines_recomputes_speedup(tmp_path: Path):
    vt = _write_verify(tmp_path, [
        ("tiny", 1, 5.0, 10.0, 50.0),   # will be repointed to baseline=20
        ("tiny", 4, 4.0, 8.0, 50.0),    # not in new_baselines -> untouched
    ])
    out = br._update_verify_tsv_baselines(vt, {("tiny", 1): 20.0})

    assert out == [("tiny", 1, 1)]
    lines = vt.read_text(encoding="utf-8").splitlines()
    repointed = lines[1].split("\t")
    assert repointed[3] == "20.000"  # baseline_speed
    # speedup = (1 - 5/20)*100 = 75.0
    assert repointed[4] == "75.0"
    # second row untouched
    assert lines[2].split("\t")[3] == "8.0"


def test_update_verify_baselines_zero_baseline_gives_zero_pct(tmp_path: Path):
    vt = _write_verify(tmp_path, [("tiny", 1, 5.0, 10.0, 50.0)])
    br._update_verify_tsv_baselines(vt, {("tiny", 1): 0.0})
    parts = vt.read_text(encoding="utf-8").splitlines()[1].split("\t")
    assert parts[3] == "0.000"
    assert parts[4] == "0.0"


def test_update_verify_baselines_skips_unparseable_thread_and_speed(tmp_path: Path):
    vt = _write_verify(tmp_path, [
        ("tiny", "x", 5.0, 10.0, 50.0),     # bad thread -> passthrough
        ("medium", 1, "NaNish", 16.0, 50.0),  # bad speed -> passthrough
    ])
    out = br._update_verify_tsv_baselines(
        vt, {("tiny", 1): 20.0, ("medium", 1): 30.0})
    # tiny row never matched (thread unparseable); medium matched key but speed bad.
    assert out == []
    lines = vt.read_text(encoding="utf-8").splitlines()
    assert lines[1].split("\t")[3] == "10.0"  # unchanged
    assert lines[2].split("\t")[3] == "16.0"  # unchanged


def test_update_verify_baselines_missing_columns_returns_empty(tmp_path: Path):
    vt = tmp_path / "verify.tsv"
    vt.write_text("tier\tspeed_sec\n tiny\t5.0\n", encoding="utf-8")
    assert br._update_verify_tsv_baselines(vt, {("tiny", 1): 20.0}) == []


def test_update_verify_baselines_missing_file(tmp_path: Path):
    assert br._update_verify_tsv_baselines(tmp_path / "no.tsv", {}) == []


# ---------------------------------------------------------------------------
# cmd_baseline_rebench end-to-end (reference subprocess monkeypatched)
# ---------------------------------------------------------------------------

REBENCH_TASK_YAML = """\
target_repo: https://example.com/foo
target_function: foo

baseline_threads: [1, 4]

datasets:
  - {tier: tiny, name: tiny_a, path: data/tiny.bin}
  - {tier: medium, name: medium_a, path: data/medium.bin}

metrics:
  - {name: speedup, comparator: gte, threshold: 1.0}
  - {name: max_diff, comparator: lte, absolute_floor: 0.05}
"""


@pytest.fixture
def rebench_task(tmp_path: Path) -> Path:
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(REBENCH_TASK_YAML, encoding="utf-8")
    # A reference script must exist so resolve_reference_script's existence
    # check passes (it's never actually executed: we monkeypatch the runner).
    (td / "reference.py").write_text("# reference\n", encoding="utf-8")
    return td


def _args(task_dir, **over):
    base = dict(
        task_dir=str(task_dir), threads=None, tiers=None, reps=None,
        replicated=False, verify_tsv=None, no_plot=True,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _fake_run_reference(seq):
    """Return a stub for _run_reference_once that yields (speed, peak) pairs.

    seq is a dict keyed by (tier, thread) -> (speed, peak); calls outside the
    map fall back to a deterministic value derived from thread.
    """
    calls = []

    def stub(task_dir, entry, thread, ref_path, ref_out_dir):
        calls.append((entry["tier"], thread))
        return seq.get((entry["tier"], thread), (10.0 / thread, 100.0))

    stub.calls = calls
    return stub


def test_rebench_records_baselines_for_each_tier_thread(rebench_task, monkeypatch):
    stub = _fake_run_reference({
        ("tiny", 1): (5.0, 50.0), ("tiny", 4): (2.0, 60.0),
        ("medium", 1): (9.0, 90.0), ("medium", 4): (4.0, 95.0),
    })
    monkeypatch.setattr(br, "_run_reference_once", stub)

    br.cmd_baseline_rebench(_args(rebench_task))

    # 2 tiers × 2 threads = 4 reference runs.
    assert sorted(stub.calls) == [
        ("medium", 1), ("medium", 4), ("tiny", 1), ("tiny", 4),
    ]
    results = (rebench_task / "results.tsv").read_text(encoding="utf-8")
    # Every (dataset, thread) baseline row landed.
    assert "tiny_a" in results and "medium_a" in results
    # Audit history written.
    hist = (rebench_task / ".zyme" / "baselines_history.tsv").read_text(encoding="utf-8")
    assert "baseline-rebench" in hist
    assert hist.count("tiny_a") + hist.count("medium_a") == 4


def test_rebench_explicit_threads_and_tiers_filter(rebench_task, monkeypatch):
    stub = _fake_run_reference({("tiny", 8): (3.0, 30.0)})
    monkeypatch.setattr(br, "_run_reference_once", stub)

    br.cmd_baseline_rebench(_args(rebench_task, threads="8", tiers="tiny"))

    assert stub.calls == [("tiny", 8)]


def test_rebench_replicated_runs_once_copies_to_all_threads(rebench_task, monkeypatch):
    # In replicated mode the reference runs exactly once per tier (at thread=1)
    # and the value is copied to every requested thread.
    stub = _fake_run_reference({("tiny", 1): (7.0, 70.0), ("medium", 1): (11.0, 110.0)})
    monkeypatch.setattr(br, "_run_reference_once", stub)

    br.cmd_baseline_rebench(_args(rebench_task, replicated=True))

    # Only thread=1 runs (once per tier), despite baseline_threads=[1,4].
    assert sorted(stub.calls) == [("medium", 1), ("tiny", 1)]
    hist = (rebench_task / ".zyme" / "baselines_history.tsv").read_text(encoding="utf-8")
    assert "baseline-rebench (replicated)" in hist
    # Replicated description tags thread>1 rows.
    results = (rebench_task / "results.tsv").read_text(encoding="utf-8")
    assert "replicated from thread=1" in results


def test_rebench_median_over_reps(rebench_task, monkeypatch):
    # With --reps 3, the median of the three measured speeds is recorded.
    seq_per_call = {
        ("tiny", 1): iter([(6.0, 60.0), (4.0, 50.0), (5.0, 55.0)]),
    }

    def stub(task_dir, entry, thread, ref_path, ref_out_dir):
        return next(seq_per_call[(entry["tier"], thread)])

    monkeypatch.setattr(br, "_run_reference_once", stub)

    br.cmd_baseline_rebench(
        _args(rebench_task, threads="1", tiers="tiny", reps=3))

    results = (rebench_task / "results.tsv").read_text(encoding="utf-8")
    # median(4,5,6) = 5.000
    assert "5.000" in results


def test_rebench_bad_threads_token_dies(rebench_task, monkeypatch):
    monkeypatch.setattr(br, "_run_reference_once", _fake_run_reference({}))
    with pytest.raises(SystemExit):
        br.cmd_baseline_rebench(_args(rebench_task, threads="1,notint"))


def test_rebench_tiers_no_match_dies(rebench_task, monkeypatch):
    monkeypatch.setattr(br, "_run_reference_once", _fake_run_reference({}))
    with pytest.raises(SystemExit):
        br.cmd_baseline_rebench(_args(rebench_task, tiers="bogus_tier"))


def test_rebench_missing_reference_script_dies(tmp_path, monkeypatch):
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(REBENCH_TASK_YAML, encoding="utf-8")
    # No reference.py / reference.R present.
    monkeypatch.setattr(br, "_run_reference_once", _fake_run_reference({}))
    with pytest.raises(SystemExit):
        br.cmd_baseline_rebench(_args(td))


def test_rebench_verify_tsv_no_matching_rows(rebench_task, monkeypatch, capsys):
    stub = _fake_run_reference({("tiny", 1): (5.0, 50.0)})
    monkeypatch.setattr(br, "_run_reference_once", stub)
    # verify.tsv has rows for a (tier, thread) that won't be re-benched.
    vt = rebench_task / "verify.tsv"
    vt.write_text(
        "tier\tthread\tspeed_sec\tbaseline_speed\tspeedup_pct\n"
        "medium\t8\t2.5\t100.0\t97.5\n",
        encoding="utf-8",
    )
    br.cmd_baseline_rebench(_args(rebench_task, threads="1", tiers="tiny"))
    assert "no rows matched" in capsys.readouterr().out


def test_rebench_verify_tsv_absent(rebench_task, monkeypatch, capsys):
    stub = _fake_run_reference({("tiny", 1): (5.0, 50.0)})
    monkeypatch.setattr(br, "_run_reference_once", stub)
    br.cmd_baseline_rebench(
        _args(rebench_task, threads="1", tiers="tiny", verify_tsv="missing.tsv"))
    assert "verify.tsv not found" in capsys.readouterr().out


def test_rebench_rerenders_verify_figure(rebench_task, monkeypatch, capsys):
    stub = _fake_run_reference({("tiny", 1): (5.0, 50.0)})
    monkeypatch.setattr(br, "_run_reference_once", stub)
    vt = rebench_task / "verify.tsv"
    vt.write_text(
        "tier\tthread\tspeed_sec\tbaseline_speed\tspeedup_pct\n"
        "tiny\t1\t2.5\t100.0\t97.5\n",
        encoding="utf-8",
    )
    # Stub the figure renderer so we hit the re-render branch without
    # depending on matplotlib output.
    rendered = []
    import zyme.commands.verify_render as vr
    monkeypatch.setattr(vr, "_render_verify_matrix",
                        lambda td, p: rendered.append(p))
    br.cmd_baseline_rebench(
        _args(rebench_task, threads="1", tiers="tiny", no_plot=False))
    assert rendered  # renderer invoked
    assert "re-rendered verify figure" in capsys.readouterr().out


def test_rebench_updates_verify_tsv_in_place(rebench_task, monkeypatch):
    stub = _fake_run_reference({("tiny", 1): (5.0, 50.0)})
    monkeypatch.setattr(br, "_run_reference_once", stub)
    # Pre-existing verify.tsv with a tiny/thread=1 row carrying a stale baseline.
    vt = rebench_task / "verify.tsv"
    vt.write_text(
        "tier\tthread\tspeed_sec\tbaseline_speed\tspeedup_pct\n"
        "tiny\t1\t2.5\t100.0\t97.5\n",
        encoding="utf-8",
    )
    br.cmd_baseline_rebench(_args(rebench_task, threads="1", tiers="tiny"))

    parts = vt.read_text(encoding="utf-8").splitlines()[1].split("\t")
    # baseline repointed to the freshly measured 5.0; speedup = (1-2.5/5)*100=50.
    assert parts[3] == "5.000"
    assert parts[4] == "50.0"
