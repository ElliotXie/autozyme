"""Wave-4 mop-up coverage for zyme.commands.baseline_rebench.

Wave-2 (test_baseline_rebench_cmd.py) covered _migrate_verify_tsv_for_k2,
_update_verify_tsv_baselines, and drove cmd_baseline_rebench end-to-end with
the reference subprocess (`_run_reference_once`) monkeypatched.

This file targets the REACHABLE branches wave-2 left:

  - _update_verify_tsv_baselines: header-only file (< 2 lines) -> [];
    blank-line passthrough and short-row (<= speedup_pct col) passthrough.
  - cmd_baseline_rebench --threads: an empty token between commas is skipped
    (continue), and an all-empty `--threads ","` parses to the empty list ->
    die("--threads: empty list").
  - cmd_baseline_rebench: the UPDATE-existing-row path in BOTH replicated and
    non-replicated modes (results.tsv already carries a (dataset, thread)
    baseline -> _update_baseline_row_in_results rather than a fresh write).
  - the re-render exception branch: _render_verify_matrix raising is caught and
    reported, not propagated.

The reference subprocess body (_run_reference_once, lines 57-95) is the
documented unreachable boundary and stays monkeypatched throughout.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from zyme.commands import baseline_rebench as br


VERIFY_HEADER = "tier\tthread\tspeed_sec\tbaseline_speed\tspeedup_pct"


# ==========================================================================
# _update_verify_tsv_baselines — header-only / blank / short-row branches
# ==========================================================================

def test_update_verify_header_only_returns_empty(tmp_path: Path):
    vt = tmp_path / "verify.tsv"
    vt.write_text(VERIFY_HEADER + "\n", encoding="utf-8")  # no data rows
    assert br._update_verify_tsv_baselines(vt, {("tiny", 1): 20.0}) == []


def test_update_verify_blank_and_short_rows_passthrough(tmp_path: Path):
    vt = tmp_path / "verify.tsv"
    vt.write_text(
        VERIFY_HEADER + "\n"
        + "\n"  # blank line -> passthrough (line 160-161)
        + "tiny\t1\t5.0\n"  # short row (<= speedup_pct col) -> passthrough (164-165)
        + "tiny\t1\t5.0\t10.0\t50.0\n",  # full, matched + repointed
        encoding="utf-8",
    )
    out = br._update_verify_tsv_baselines(vt, {("tiny", 1): 20.0})
    assert out == [("tiny", 1, 1)]
    lines = vt.read_text(encoding="utf-8").splitlines()
    # blank preserved, short row preserved, full row repointed.
    assert lines[1] == ""
    assert lines[2] == "tiny\t1\t5.0"
    assert lines[3].split("\t")[3] == "20.000"


# ==========================================================================
# cmd_baseline_rebench end-to-end pieces
# ==========================================================================

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
    calls = []

    def stub(task_dir, entry, thread, ref_path, ref_out_dir):
        calls.append((entry["tier"], thread))
        return seq.get((entry["tier"], thread), (10.0 / thread, 100.0))

    stub.calls = calls
    return stub


def test_rebench_threads_skips_empty_token(rebench_task, monkeypatch):
    # "1, ,4" -> the empty middle token is skipped (continue), leaving [1, 4].
    stub = _fake_run_reference({})
    monkeypatch.setattr(br, "_run_reference_once", stub)
    br.cmd_baseline_rebench(_args(rebench_task, threads="1, ,4", tiers="tiny"))
    assert sorted({t for _, t in stub.calls}) == [1, 4]


def test_rebench_threads_all_empty_dies(rebench_task, monkeypatch):
    stub = _fake_run_reference({})
    monkeypatch.setattr(br, "_run_reference_once", stub)
    # "," parses to no integers -> empty list -> die.
    with pytest.raises(SystemExit):
        br.cmd_baseline_rebench(_args(rebench_task, threads=","))


def test_rebench_updates_existing_results_row_nonreplicated(rebench_task, monkeypatch):
    # Pre-seed results.tsv with a (tiny_a, thread=1) baseline so the rebench
    # takes the UPDATE-existing branch (line 339) rather than a fresh write.
    from zyme.commands import baseline as bl
    bl.cmd_record_baseline(types.SimpleNamespace(
        task_dir=str(rebench_task), tier="tiny", name=None, speed_sec=9.0,
        peak_mb=90.0, thread=1, metrics="{}", from_log=None, oom=False,
        force=False, accept_synthesis=False,
    ))
    stub = _fake_run_reference({("tiny", 1): (5.0, 50.0)})
    monkeypatch.setattr(br, "_run_reference_once", stub)
    br.cmd_baseline_rebench(_args(rebench_task, threads="1", tiers="tiny"))

    # The single baseline row was updated in place to the new 5.000 value.
    results = (rebench_task / "results.tsv").read_text(encoding="utf-8")
    assert "5.000" in results
    # Only one tiny_a baseline row (updated, not duplicated).
    assert results.count("\ttiny_a\t") == 1


def test_rebench_updates_existing_results_row_replicated(rebench_task, monkeypatch):
    from zyme.commands import baseline as bl
    # Pre-seed BOTH thread=1 and thread=4 baselines for tiny so replicated mode
    # updates both in place (line 291).
    for th in (1, 4):
        bl.cmd_record_baseline(types.SimpleNamespace(
            task_dir=str(rebench_task), tier="tiny", name=None, speed_sec=9.0,
            peak_mb=90.0, thread=th, metrics="{}", from_log=None, oom=False,
            force=True, accept_synthesis=False,
        ))
    stub = _fake_run_reference({("tiny", 1): (7.0, 70.0)})
    monkeypatch.setattr(br, "_run_reference_once", stub)
    br.cmd_baseline_rebench(_args(rebench_task, threads="1,4", tiers="tiny",
                                  replicated=True))

    results = (rebench_task / "results.tsv").read_text(encoding="utf-8")
    # Both threads repointed to 7.000; tiny ran exactly once (thread=1).
    assert results.count("7.000") >= 2
    assert stub.calls == [("tiny", 1)]
    # Replicated-from-thread=1 tag present on the thread=4 row.
    assert "replicated from thread=1" in results


def test_rebench_rerender_exception_is_caught(rebench_task, monkeypatch, capsys):
    stub = _fake_run_reference({("tiny", 1): (5.0, 50.0)})
    monkeypatch.setattr(br, "_run_reference_once", stub)
    vt = rebench_task / "verify.tsv"
    vt.write_text(
        VERIFY_HEADER + "\n" + "tiny\t1\t2.5\t100.0\t97.5\n",
        encoding="utf-8",
    )
    import zyme.commands.verify_render as vr

    def boom(td, p):
        raise RuntimeError("matplotlib exploded")

    monkeypatch.setattr(vr, "_render_verify_matrix", boom)
    # Must not propagate; the except-branch reports and continues.
    br.cmd_baseline_rebench(_args(rebench_task, threads="1", tiers="tiny",
                                  no_plot=False))
    assert "could not re-render verify figure" in capsys.readouterr().out
