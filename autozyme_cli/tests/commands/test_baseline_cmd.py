"""Unit tests for zyme.commands.baseline.

Pure helpers (history append, OOM-row rewrite, baseline lookups, show-row
extraction, OOM heuristic) are tested directly. cmd_record_baseline /
cmd_baseline_list / cmd_baseline_show / cmd_promote_baseline are driven with
crafted temp task dirs; cmd_record_baseline launches no subprocess, so it runs
end-to-end. cmd_reference / cmd_record_noise are subprocess-bound and only
their no-subprocess validation/branch logic is exercised.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from zyme.commands import baseline as bl


# ---------------------------------------------------------------------------
# Fixtures: a realistic two-tier task dir.
# ---------------------------------------------------------------------------

TASK_YAML = """\
target_repo: https://example.com/foo
target_function: foo

baseline_threads: [1, 4]

datasets:
  - {tier: tiny, name: tiny_a, path: data/tiny.bin}
  - {tier: medium, name: medium_a, path: data/medium.bin}

metrics:
  - {name: corr, comparator: gte, threshold: 0.99}
  - {name: max_diff, comparator: lte, noise_multiplier: 2.0, absolute_floor: 0.05}
"""


@pytest.fixture
def btask(tmp_path: Path) -> Path:
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(TASK_YAML, encoding="utf-8")
    (td / ".zyme").mkdir()
    return td


def _rec_args(task_dir, **over):
    base = dict(
        task_dir=str(task_dir), tier=None, name=None, speed_sec=None,
        peak_mb=0.0, thread=None, metrics="{}", from_log=None, oom=False,
        force=False, accept_synthesis=False,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# _append_baseline_history
# ---------------------------------------------------------------------------

def test_append_baseline_history_creates_header_then_appends(btask):
    bl._append_baseline_history(btask, "tiny", "tiny_a", 5.0, 50.0, "manual")
    bl._append_baseline_history(
        btask, "tiny", "tiny_a", 4.0, 48.0, "rebench",
        prior_speed_sec=5.0, prior_peak_mb=50.0, thread=4)
    p = bl._baseline_history_path(btask)
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == list(bl._BASELINE_HISTORY_FIELDS)
    assert len(lines) == 3
    # First row: thread defaults to LEGACY_THREAD (=1), no prior columns.
    first = lines[1].split("\t")
    assert first[1:3] == ["tiny", "tiny_a"]
    assert first[-1] == "1"  # thread
    assert first[6] == "" and first[7] == ""  # prior_* empty
    # Second row: prior values rendered, thread=4.
    second = lines[2].split("\t")
    assert second[6] == "5.000" and second[7] == "50.0"
    assert second[-1] == "4"


# ---------------------------------------------------------------------------
# _looks_like_oom
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("log", [
    "boom\ncannot allocate vector of size 9.9 Gb\n",
    "terminate called after throwing std::bad_alloc",
    "MemoryError: unable to allocate array",
    "R session: vector memory exhausted",
    "Killed",
    "out of memory while running",
])
def test_looks_like_oom_true(log):
    assert bl._looks_like_oom(log)


def test_looks_like_oom_false_for_normal_error():
    assert not bl._looks_like_oom("Traceback: ValueError: bad shape\n")


def test_looks_like_oom_only_scans_tail():
    # OOM marker only in the head of a >4096 char log is ignored.
    head = "cannot allocate vector of size 5Gb\n"
    log = head + ("x" * 5000)
    assert not bl._looks_like_oom(log)


# ---------------------------------------------------------------------------
# _update_baseline_row_to_oom
# ---------------------------------------------------------------------------

RESULTS_HEADER = (
    "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
    "metrics_json\thypothesis\tdescription\tphase\tthread"
)


def _write_results(task_dir, rows):
    p = task_dir / "results.tsv"
    body = "\n".join("\t".join(map(str, r)) for r in rows)
    p.write_text(RESULTS_HEADER + "\n" + body + "\n", encoding="utf-8")
    return p


def test_update_baseline_row_to_oom_rewrites_in_place(btask):
    rt = _write_results(btask, [
        ("0", "upstream", "tiny_a", "5.000", "0.0", "50.0", "baseline",
         "{}", "", "upstream reference baseline", "optimize", "1"),
    ])
    updated = bl._update_baseline_row_to_oom(rt, "tiny_a", thread=1)
    assert updated
    row = rt.read_text(encoding="utf-8").splitlines()[1].split("\t")
    assert row[6] == "oom"
    assert row[3] == "0" and row[5] == "0"
    assert "OOM override" in row[9]


def test_update_baseline_row_to_oom_thread_mismatch(btask):
    rt = _write_results(btask, [
        ("0", "upstream", "tiny_a", "5.000", "0.0", "50.0", "baseline",
         "{}", "", "d", "optimize", "1"),
    ])
    # Asking for thread=4 -> no row matches -> not updated.
    assert not bl._update_baseline_row_to_oom(rt, "tiny_a", thread=4)


def test_update_baseline_row_to_oom_missing_file(btask):
    assert not bl._update_baseline_row_to_oom(btask / "no.tsv", "tiny_a")


def test_update_baseline_row_to_oom_header_only(btask):
    p = btask / "results.tsv"
    p.write_text(RESULTS_HEADER + "\n", encoding="utf-8")
    assert not bl._update_baseline_row_to_oom(p, "tiny_a")


# ---------------------------------------------------------------------------
# _baseline_show_rows
# ---------------------------------------------------------------------------

def test_baseline_show_rows_filters_by_tier(btask):
    _write_results(btask, [
        ("0", "upstream", "tiny_a", "5.0", "0.0", "50.0", "baseline",
         "{}", "", "d", "optimize", "1"),
        ("0", "upstream", "medium_a", "9.0", "0.0", "90.0", "baseline",
         "{}", "", "d", "optimize", "4"),
        ("1", "abc", "tiny_a", "4.0", "20.0", "48.0", "keep",
         "{}", "", "d", "optimize", "1"),  # not a baseline -> excluded
    ])
    rows = bl._baseline_show_rows(btask, "tiny")
    assert len(rows) == 1
    assert rows[0]["name"] == "tiny_a"
    assert rows[0]["thread"] == "1"
    assert rows[0]["status"] == "baseline"


def test_baseline_show_rows_empty_when_no_results(btask):
    assert bl._baseline_show_rows(btask, "tiny") == []


# ---------------------------------------------------------------------------
# _existing_baseline + _update_baseline_row_in_results
# ---------------------------------------------------------------------------

def test_existing_baseline_lookup(btask):
    _write_results(btask, [
        ("0", "upstream", "tiny_a", "5.000", "0.0", "50.0", "baseline",
         "{}", "", "d", "optimize", "1"),
    ])
    speed, peak = bl._existing_baseline(btask, "tiny_a", thread=1)
    assert speed == pytest.approx(5.0)
    assert peak == pytest.approx(50.0)


def test_existing_baseline_none_when_absent(btask):
    _write_results(btask, [])
    assert bl._existing_baseline(btask, "tiny_a", thread=1) == (None, None)


def test_update_baseline_row_in_results_in_place(btask):
    rt = _write_results(btask, [
        ("0", "upstream", "tiny_a", "5.000", "0.0", "50.0", "baseline",
         "{}", "", "old", "optimize", "1"),
    ])
    bl._update_baseline_row_in_results(
        btask, {"name": "tiny_a", "tier": "tiny"}, thread=1,
        speed_sec=4.0, peak_mb=40.0, metrics_json='{"corr": 1.0}',
        description="re-recorded")
    row = rt.read_text(encoding="utf-8").splitlines()[1].split("\t")
    assert row[3] == "4.000" and row[5] == "40.0"
    assert row[7] == '{"corr": 1.0}'
    assert row[9] == "re-recorded"


def test_update_baseline_row_in_results_thread_mismatch_noop(btask):
    rt = _write_results(btask, [
        ("0", "upstream", "tiny_a", "5.000", "0.0", "50.0", "baseline",
         "{}", "", "old", "optimize", "1"),
    ])
    bl._update_baseline_row_in_results(
        btask, {"name": "tiny_a", "tier": "tiny"}, thread=4,
        speed_sec=4.0, peak_mb=40.0, metrics_json="{}", description="x")
    # thread=1 row left intact (no thread=4 row to update).
    row = rt.read_text(encoding="utf-8").splitlines()[1].split("\t")
    assert row[3] == "5.000"


def test_baseline_show_rows_includes_oom_status(btask):
    _write_results(btask, [
        ("0", "upstream", "tiny_a", "0", "0.0", "0", "oom",
         "{}", "", "(OOM)", "optimize", "1"),
    ])
    rows = bl._baseline_show_rows(btask, "tiny")
    assert rows and rows[0]["status"] == "oom"


# ---------------------------------------------------------------------------
# cmd_record_baseline — the core no-subprocess entry point.
# ---------------------------------------------------------------------------

def test_record_baseline_writes_row_and_history(btask):
    bl.cmd_record_baseline(_rec_args(btask, tier="tiny", speed_sec=5.0, peak_mb=50.0))
    results = (btask / "results.tsv").read_text(encoding="utf-8")
    assert "tiny_a" in results
    assert "baseline" in results
    hist = bl._baseline_history_path(btask).read_text(encoding="utf-8")
    assert "tiny_a" in hist


def test_record_baseline_overwrites_existing_same_thread(btask):
    bl.cmd_record_baseline(_rec_args(btask, tier="tiny", speed_sec=5.0))
    # Re-record within the 2x sanity band -> updated in place, one baseline row.
    bl.cmd_record_baseline(_rec_args(btask, tier="tiny", speed_sec=6.0))
    lines = (btask / "results.tsv").read_text(encoding="utf-8").splitlines()
    baseline_rows = [ln for ln in lines[1:] if "\tbaseline\t" in ln and "tiny_a" in ln]
    assert len(baseline_rows) == 1
    assert "6.000" in baseline_rows[0]


def test_record_baseline_2x_gate_blocks_without_force(btask):
    bl.cmd_record_baseline(_rec_args(btask, tier="tiny", speed_sec=5.0))
    with pytest.raises(SystemExit):
        bl.cmd_record_baseline(_rec_args(btask, tier="tiny", speed_sec=50.0))
    # With --force it goes through.
    bl.cmd_record_baseline(_rec_args(btask, tier="tiny", speed_sec=50.0, force=True))
    results = (btask / "results.tsv").read_text(encoding="utf-8")
    assert "50.000" in results


def test_record_baseline_unknown_tier_dies(btask):
    with pytest.raises(SystemExit):
        bl.cmd_record_baseline(_rec_args(btask, tier="nope", speed_sec=5.0))


def test_record_baseline_missing_tier_dies(btask):
    with pytest.raises(SystemExit):
        bl.cmd_record_baseline(_rec_args(btask, tier=None, speed_sec=5.0))


def test_record_baseline_missing_speed_dies(btask):
    with pytest.raises(SystemExit):
        bl.cmd_record_baseline(_rec_args(btask, tier="tiny", speed_sec=None))


def test_record_baseline_name_mismatch_dies(btask):
    with pytest.raises(SystemExit):
        bl.cmd_record_baseline(
            _rec_args(btask, tier="tiny", speed_sec=5.0, name="wrong_name"))


def test_record_baseline_bad_metrics_json_dies(btask):
    with pytest.raises(SystemExit):
        bl.cmd_record_baseline(
            _rec_args(btask, tier="tiny", speed_sec=5.0, metrics="{not json"))


def test_record_baseline_autofills_identity_metrics(btask):
    # metrics="{}" + task.yaml has metrics -> identity metrics auto-filled.
    bl.cmd_record_baseline(_rec_args(btask, tier="tiny", speed_sec=5.0))
    row = [
        ln for ln in (btask / "results.tsv").read_text(encoding="utf-8").splitlines()
        if "tiny_a" in ln
    ][0]
    # corr gte -> 1.0, max_diff lte -> 0.0
    assert '"corr": 1.0' in row
    assert '"max_diff": 0.0' in row


def test_record_baseline_from_log(btask, tmp_path):
    log = tmp_path / "ref.log"
    log.write_text("speed_sec: 7.250\npeak_mb: 123.4\n", encoding="utf-8")
    bl.cmd_record_baseline(
        _rec_args(btask, tier="tiny", speed_sec=None, from_log=str(log)))
    results = (btask / "results.tsv").read_text(encoding="utf-8")
    assert "7.250" in results
    assert "123.4" in results


def test_record_baseline_from_log_missing_file_dies(btask):
    with pytest.raises(SystemExit):
        bl.cmd_record_baseline(
            _rec_args(btask, tier="tiny", from_log="/no/such/log"))


def test_record_baseline_from_log_no_speed_dies(btask, tmp_path):
    log = tmp_path / "ref.log"
    log.write_text("nothing useful here\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        bl.cmd_record_baseline(_rec_args(btask, tier="tiny", from_log=str(log)))


def test_record_baseline_oom_mode(btask):
    bl.cmd_record_baseline(_rec_args(btask, tier="tiny", oom=True))
    row = [
        ln for ln in (btask / "results.tsv").read_text(encoding="utf-8").splitlines()
        if "tiny_a" in ln
    ][0]
    assert "\toom\t" in row


def test_record_baseline_oom_refuses_over_existing_without_force(btask):
    bl.cmd_record_baseline(_rec_args(btask, tier="tiny", speed_sec=5.0))
    with pytest.raises(SystemExit):
        bl.cmd_record_baseline(_rec_args(btask, tier="tiny", oom=True))
    # --force allows the OOM override.
    bl.cmd_record_baseline(_rec_args(btask, tier="tiny", oom=True, force=True))
    row = [
        ln for ln in (btask / "results.tsv").read_text(encoding="utf-8").splitlines()
        if "tiny_a" in ln and "\toom\t" in ln
    ]
    assert row


def test_record_baseline_oom_requires_tier(btask):
    with pytest.raises(SystemExit):
        bl.cmd_record_baseline(_rec_args(btask, tier=None, oom=True))


def test_record_baseline_respects_thread_arg(btask):
    bl.cmd_record_baseline(_rec_args(btask, tier="tiny", speed_sec=5.0, thread=4))
    row = [
        ln for ln in (btask / "results.tsv").read_text(encoding="utf-8").splitlines()
        if "tiny_a" in ln
    ][0]
    assert row.split("\t")[-1] == "4"


# ---------------------------------------------------------------------------
# cmd_baseline_list / cmd_baseline_show
# ---------------------------------------------------------------------------

def test_baseline_list_no_results(btask, capsys):
    bl.cmd_baseline_list(_rec_args(btask, history=False))
    assert "No results.tsv yet" in capsys.readouterr().out


def test_baseline_list_shows_current_rows(btask, capsys):
    _write_results(btask, [
        ("0", "upstream", "tiny_a", "5.0", "0.0", "50.0", "baseline",
         "{}", "", "d", "optimize", "1"),
    ])
    bl.cmd_baseline_list(_rec_args(btask, history=False))
    out = capsys.readouterr().out
    assert "current baselines" in out
    assert "tiny_a" in out
    assert "tiny" in out  # tier label resolved from task.yaml


def test_baseline_list_history(btask, capsys):
    bl._append_baseline_history(btask, "tiny", "tiny_a", 5.0, 50.0, "manual")
    bl.cmd_baseline_list(_rec_args(btask, history=True))
    out = capsys.readouterr().out
    assert "audit history" in out
    assert "tiny_a" in out


def test_baseline_list_history_empty(btask, capsys):
    bl.cmd_baseline_list(_rec_args(btask, history=True))
    assert "No audit history yet" in capsys.readouterr().out


def test_baseline_show_no_results(btask, capsys):
    bl.cmd_baseline_show(_rec_args(btask, tier="tiny"))
    assert "no results.tsv yet" in capsys.readouterr().out


def test_baseline_show_prints_rows(btask, capsys):
    _write_results(btask, [
        ("0", "upstream", "tiny_a", "5.0", "0.0", "50.0", "baseline",
         "{}", "", "d", "optimize", "1"),
    ])
    bl.cmd_baseline_show(_rec_args(btask, tier="tiny"))
    out = capsys.readouterr().out
    assert "tier=tiny" in out and "name=tiny_a" in out and "thread=1" in out


def test_baseline_show_no_matching_tier(btask, capsys):
    _write_results(btask, [
        ("0", "upstream", "tiny_a", "5.0", "0.0", "50.0", "baseline",
         "{}", "", "d", "optimize", "1"),
    ])
    bl.cmd_baseline_show(_rec_args(btask, tier="medium"))
    assert "no baseline row" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# cmd_promote_baseline (stash drain escape hatch)
# ---------------------------------------------------------------------------

def test_promote_baseline_no_stash(btask, capsys):
    bl.cmd_promote_baseline(_rec_args(btask))
    assert "no stash entries to drain" in capsys.readouterr().out


def test_promote_baseline_drains_stash(btask):
    from zyme.utils import write_baseline_stash
    write_baseline_stash(btask, [
        {"name": "tiny_a", "thread": "1", "speed_sec": "5.0",
         "peak_mb": "50.0", "status": "baseline", "metrics_json": "{}"},
    ])
    bl.cmd_promote_baseline(_rec_args(btask))
    results = (btask / "results.tsv").read_text(encoding="utf-8")
    assert "tiny_a" in results and "drained from stash" in results
    hist = bl._baseline_history_path(btask).read_text(encoding="utf-8")
    assert "promote-baseline (drain)" in hist


def test_promote_baseline_skips_unknown_dataset(btask, capsys):
    from zyme.utils import write_baseline_stash
    write_baseline_stash(btask, [
        {"name": "ghost", "thread": "1", "speed_sec": "5.0", "peak_mb": "50.0"},
    ])
    bl.cmd_promote_baseline(_rec_args(btask))
    out = capsys.readouterr().out
    assert "unknown dataset" in out


# ---------------------------------------------------------------------------
# cmd_record_noise — validation branches (no subprocess reached).
# ---------------------------------------------------------------------------

def test_record_noise_requires_stochastic(btask):
    # btask's task.yaml has no algorithm_class -> defaults non-stochastic.
    with pytest.raises(SystemExit):
        bl.cmd_record_noise(_rec_args(btask, tier="tiny", seeds=None))


def test_record_noise_bad_seeds_format_dies(tmp_path):
    td = tmp_path / "task"
    td.mkdir()
    (td / "task.yaml").write_text(
        TASK_YAML + "\nalgorithm_class: stochastic\n", encoding="utf-8")
    (td / ".zyme").mkdir()
    with pytest.raises(SystemExit):
        bl.cmd_record_noise(_rec_args(td, tier="tiny", seeds="1,bad,3"))


# ---------------------------------------------------------------------------
# cmd_reference — pre-subprocess validation branches.
# ---------------------------------------------------------------------------

def test_reference_unknown_tier_dies(btask):
    (btask / "reference.py").write_text("# ref\n", encoding="utf-8")
    args = _rec_args(btask, tier="bogus", reps=1)
    with pytest.raises(SystemExit):
        bl.cmd_reference(args)


def test_reference_missing_script_dies(btask):
    # No reference.{py,R} present -> dies before any subprocess.
    args = _rec_args(btask, tier="tiny", reps=1)
    with pytest.raises(SystemExit):
        bl.cmd_reference(args)
