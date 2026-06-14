"""Unit tests for zyme.progress_guard — plateau_state classification (Tier-1
regime + Tier-2 bands), the reminder dispatcher, cooldown gating in
maybe_print_plateau_reminder, the silent-log appender, and state roundtrip.

results.tsv rows are built directly so we control round numbers, statuses,
and speedup_pct exactly.
"""
from __future__ import annotations

import json

import pytest

from zyme import progress_guard as PG


HEADER = ("round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\t"
          "status\tmetrics_json\thypothesis\tdescription\tphase")


def _row(round_num, status, speedup_pct, phase="optimize"):
    return (f"{round_num}\tc{round_num}\ttiny_a\t1.0\t{speedup_pct}\t10.0\t"
            f"{status}\t{{}}\thyp\tdesc\t{phase}")


def _write_results(task_dir, rows):
    (task_dir / "results.tsv").write_text(HEADER + "\n" + "\n".join(rows) + "\n")


@pytest.fixture
def tdir(tmp_path):
    (tmp_path / ".zyme").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# plateau_state — None paths
# ---------------------------------------------------------------------------

def test_plateau_state_no_results(tdir):
    assert PG.plateau_state(tdir) is None


def test_plateau_state_no_keep_rows(tdir):
    _write_results(tdir, [
        _row(0, "baseline", 0.0),
        _row(1, "discard", 5.0),
    ])
    assert PG.plateau_state(tdir) is None


def test_plateau_state_keep_without_speedup(tdir):
    # keep row but blank speedup_pct -> best_keep_pct stays None -> None.
    rows = [HEADER.replace("\t", "\t"),
            "1\tc1\ttiny_a\t1.0\t\t10.0\tkeep\t{}\thyp\tdesc\toptimize"]
    (tdir / "results.tsv").write_text("\n".join([HEADER] + rows[1:]) + "\n")
    assert PG.plateau_state(tdir) is None


def test_plateau_state_keep_but_no_decisions_with_round_gt0(tdir):
    # keep exists (round 0 baseline-like) but no decision row with round>0.
    _write_results(tdir, [_row(0, "keep", 30.0)])
    assert PG.plateau_state(tdir) is None


# ---------------------------------------------------------------------------
# plateau_state — Tier-1 regimes
# ---------------------------------------------------------------------------

def test_regime_early_stuck(tdir):
    rows = [_row(1, "keep", 30.0)]
    rows += [_row(r, "discard", 10.0) for r in range(2, 8)]
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    assert st["regime"] == "early_stuck"
    assert st["best_keep_pct"] == 30.0
    assert st["tail_len"] == 6  # rounds 2..7 discards


def test_regime_middle(tdir):
    rows = [_row(1, "keep", 70.0)]
    rows += [_row(r, "discard", 10.0) for r in range(2, 8)]
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    assert st["regime"] == "middle"


def test_regime_late_plateau(tdir):
    rows = [_row(1, "keep", 96.0)]
    rows += [_row(r, "discard", 10.0) for r in range(2, 8)]
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    assert st["regime"] == "late_plateau"


def test_best_keep_pct_is_max_over_keeps(tdir):
    rows = [_row(1, "keep", 20.0), _row(2, "keep", 45.0), _row(3, "keep", 30.0)]
    rows += [_row(r, "discard", 5.0) for r in range(4, 6)]
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    assert st["best_keep_pct"] == 45.0


def test_tail_len_resets_at_keep(tdir):
    # most recent keep at round 5; discards after.
    rows = [_row(1, "keep", 60.0), _row(2, "discard", 5.0),
            _row(5, "keep", 65.0), _row(6, "discard", 5.0), _row(7, "discard", 5.0)]
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    assert st["tail_len"] == 2
    assert st["latest_keep_round"] == 5


def test_only_terminal_statuses_count_as_decisions(tdir):
    # 'rerun'/'pending' are not terminal; they don't count toward tail.
    rows = [_row(1, "keep", 30.0),
            _row(2, "rerun", 30.0),
            _row(3, "discard", 5.0)]
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    # decisions = round1 keep + round3 discard; tail = [discard]
    assert st["tail_len"] == 1


def test_non_optimize_phase_ignored(tdir):
    rows = [_row(1, "keep", 30.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 7)]
    rows += [_row(8, "keep", 99.0, phase="scaling")]  # scaling phase ignored
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    assert st["best_keep_pct"] == 30.0  # scaling keep not counted


# ---------------------------------------------------------------------------
# plateau_state — Tier-2 bands
# ---------------------------------------------------------------------------

def test_tier2_escalate(tdir):
    # best < 50, tail >= 10 -> escalate.
    rows = [_row(1, "keep", 30.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 13)]  # 11 discards
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    assert st["tier2_band"] == "escalate"
    assert st["tier2_triggered"] is True


def test_tier2_decide_by_tail(tdir):
    # best in [50,90), tail >= 10 -> decide.
    rows = [_row(1, "keep", 70.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 13)]
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    assert st["tier2_band"] == "decide"


def test_tier2_terminate_by_tail(tdir):
    # best >= 90, tail >= 10 -> terminate.
    rows = [_row(1, "keep", 95.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 13)]
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    assert st["tier2_band"] == "terminate"


def test_tier2_decide_by_low_gain(tdir):
    # best 70, tail < 10 but >=10 decisions with <2pp gain over last 10 -> decide.
    # rounds 1..6 are keeps (so tail is small), but plenty of decisions and
    # recent best does not exceed prior best by >=2pp.
    rows = [_row(r, "keep", 70.0) for r in range(1, 12)]  # 11 keeps all at 70
    # tail_len = 0 (last is keep) so the low-gain path must drive the band.
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    # 11 decisions; recent_best 70, prior_best 70 -> gain 0 < 2 -> decide.
    assert st["gain_pct_over_10"] == pytest.approx(0.0)
    assert st["tier2_band"] == "decide"


def test_tier2_no_band_when_healthy(tdir):
    # best 70, only a few rounds -> no tier2.
    rows = [_row(1, "keep", 70.0), _row(2, "discard", 5.0)]
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    assert st["tier2_band"] is None
    assert st["tier2_triggered"] is False


def test_gain_pct_over_10_none_with_few_decisions(tdir):
    rows = [_row(1, "keep", 70.0), _row(2, "discard", 5.0)]
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    assert st["gain_pct_over_10"] is None


def test_escalate_band_ignores_low_gain(tdir):
    # best < 50 with tail < 10 -> NOT escalate even if many low-gain decisions.
    rows = [_row(r, "keep", 30.0) for r in range(1, 12)]  # tail 0, best 30
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    # escalate requires tail >= 10; gain signal does not apply in escalate band.
    assert st["tier2_band"] is None


# ---------------------------------------------------------------------------
# _reminder_for / current_plateau_reminder dispatch
# ---------------------------------------------------------------------------

def test_current_reminder_none_without_results(tdir):
    assert PG.current_plateau_reminder(tdir) is None


def test_current_reminder_middle_band_silent(tdir):
    rows = [_row(1, "keep", 70.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 8)]
    _write_results(tdir, rows)
    assert PG.current_plateau_reminder(tdir) is None


def test_current_reminder_tier1_too_few_rounds_silent(tdir):
    # late_plateau but tail < 5 -> silent at tier1.
    rows = [_row(1, "keep", 96.0), _row(2, "discard", 5.0)]
    _write_results(tdir, rows)
    assert PG.current_plateau_reminder(tdir) is None


def test_current_reminder_late_plateau_text(tdir):
    rows = [_row(1, "keep", 96.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 8)]
    _write_results(tdir, rows)
    msg = PG.current_plateau_reminder(tdir)
    assert "progress plateau reminder" in msg


def test_current_reminder_early_stuck_text(tdir):
    rows = [_row(1, "keep", 30.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 8)]
    _write_results(tdir, rows)
    msg = PG.current_plateau_reminder(tdir)
    assert "early-stuck reminder" in msg


def test_current_reminder_tier2_escalate_text(tdir):
    rows = [_row(1, "keep", 30.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 13)]
    _write_results(tdir, rows)
    msg = PG.current_plateau_reminder(tdir)
    assert "plateau escalation" in msg


def test_current_reminder_tier2_decide_text(tdir):
    rows = [_row(1, "keep", 70.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 13)]
    _write_results(tdir, rows)
    msg = PG.current_plateau_reminder(tdir)
    assert "decision window" in msg


def test_current_reminder_tier2_terminate_text(tdir):
    rows = [_row(1, "keep", 95.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 13)]
    _write_results(tdir, rows)
    msg = PG.current_plateau_reminder(tdir)
    assert "plateau terminate" in msg


# ---------------------------------------------------------------------------
# Reminder formatters — direct state input
# ---------------------------------------------------------------------------

def _state(**kw):
    base = {"regime": "middle", "best_keep_pct": 70.0, "tail_len": 6,
            "current_round": 12, "latest_keep_round": 1, "latest_keep_pct": 70.0,
            "gain_pct_over_10": None, "tier2_band": None, "tier2_triggered": False}
    base.update(kw)
    return base


def test_format_plateau_reminder(monkeypatch):
    msg = PG.format_plateau_reminder(_state())
    assert "Best kept speedup is +70.0%" in msg
    assert "latest keep was round 1" in msg


def test_format_plateau_reminder_no_latest_pct():
    msg = PG.format_plateau_reminder(_state(latest_keep_pct=None))
    assert "round 1" in msg
    assert "at +" not in msg


def test_format_early_stuck_reminder():
    msg = PG.format_early_stuck_reminder(_state(best_keep_pct=20.0))
    assert "only +20.0%" in msg


def test_format_tier2_escalate():
    msg = PG.format_tier2_escalate(_state(best_keep_pct=30.0))
    assert "stop being timid" in msg


def test_format_tier2_decide():
    msg = PG.format_tier2_decide(_state(best_keep_pct=70.0))
    assert "structural budget" in msg
    assert str(PG.TIER2_DECIDE_BUDGET_ROUNDS) in msg


def test_format_tier2_terminate_with_gain_clause():
    msg = PG.format_tier2_terminate(_state(best_keep_pct=95.0, gain_pct_over_10=0.5))
    assert "marginal value exhausted" in msg
    assert "cumulative gain" in msg


def test_format_tier2_terminate_no_gain_clause():
    msg = PG.format_tier2_terminate(_state(best_keep_pct=95.0, gain_pct_over_10=None))
    assert "cumulative gain" not in msg


def test_reminder_for_dispatch():
    assert "escalation" in PG._reminder_for(_state(tier2_band="escalate"))
    assert "decision window" in PG._reminder_for(_state(tier2_band="decide"))
    assert "terminate" in PG._reminder_for(_state(tier2_band="terminate"))
    assert "early-stuck" in PG._reminder_for(_state(regime="early_stuck"))
    assert "progress plateau" in PG._reminder_for(_state(regime="middle"))


# ---------------------------------------------------------------------------
# state file roundtrip + helpers
# ---------------------------------------------------------------------------

def test_state_roundtrip(tdir):
    assert PG._read_state(tdir) is None
    PG._write_state(tdir, {"last_tier1_warned_round": 5})
    assert PG._read_state(tdir) == {"last_tier1_warned_round": 5}


def test_read_state_malformed(tdir):
    (tdir / ".zyme" / PG.STATE_FILE_NAME).write_text("{bad")
    assert PG._read_state(tdir) is None


def test_float_helper():
    assert PG._float("3.5") == 3.5
    assert PG._float(None) is None
    assert PG._float("x") is None


def test_round_num_helper():
    assert PG._round_num({"round": "5"}) == 5
    assert PG._round_num({"round": "2.1"}) == 2
    assert PG._round_num({"round": ""}) == 0
    assert PG._round_num({}) == 0
    assert PG._round_num({"round": "bad"}) == 0


# ---------------------------------------------------------------------------
# maybe_print_plateau_reminder — cooldown + side effects
# ---------------------------------------------------------------------------

def test_maybe_print_no_state_silent(tdir, capsys):
    PG.maybe_print_plateau_reminder(tdir)
    assert capsys.readouterr().out == ""


def test_maybe_print_tier1_fires_and_records(tdir, capsys):
    rows = [_row(1, "keep", 30.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 8)]
    _write_results(tdir, rows)
    PG.maybe_print_plateau_reminder(tdir)
    out = capsys.readouterr().out
    assert "early-stuck" in out
    st = PG._read_state(tdir)
    assert st["last_tier1_warned_round"] == 7  # current_round (latest decision)


def test_maybe_print_tier1_cooldown_suppresses(tdir, capsys):
    rows = [_row(1, "keep", 30.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 8)]  # current_round = 7
    _write_results(tdir, rows)
    PG._write_state(tdir, {"last_tier1_warned_round": 5})  # 7-5=2 < 5 cooldown
    PG.maybe_print_plateau_reminder(tdir)
    assert capsys.readouterr().out == ""


def test_maybe_print_tier1_backcompat_last_warned_round(tdir, capsys):
    rows = [_row(1, "keep", 30.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 8)]
    _write_results(tdir, rows)
    # old-format key
    PG._write_state(tdir, {"last_warned_round": 5})
    PG.maybe_print_plateau_reminder(tdir)
    assert capsys.readouterr().out == ""  # cooldown via legacy key


def test_maybe_print_tier1_short_tail_silent(tdir, capsys):
    # early_stuck regime but only 2 discards since last keep -> tail < 5 -> silent.
    rows = [_row(1, "keep", 30.0), _row(2, "discard", 5.0), _row(3, "discard", 5.0)]
    _write_results(tdir, rows)
    st = PG.plateau_state(tdir)
    assert st["regime"] == "early_stuck" and st["tail_len"] == 2
    PG.maybe_print_plateau_reminder(tdir)
    assert capsys.readouterr().out == ""


def test_maybe_print_tier2_non_int_cooldown_state_prints(tdir, capsys):
    # last_tier2_warned_round is non-int -> cooldown check skipped -> prints.
    rows = [_row(1, "keep", 30.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 13)]
    _write_results(tdir, rows)
    PG._write_state(tdir, {"last_tier2_warned_round": "garbage"})
    PG.maybe_print_plateau_reminder(tdir)
    assert "plateau escalation" in capsys.readouterr().out


def test_maybe_print_middle_band_silent(tdir, capsys):
    rows = [_row(1, "keep", 70.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 8)]
    _write_results(tdir, rows)
    PG.maybe_print_plateau_reminder(tdir)
    assert capsys.readouterr().out == ""


def test_maybe_print_tier2_fires_and_logs(tdir, capsys):
    rows = [_row(1, "keep", 30.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 13)]
    _write_results(tdir, rows)
    PG.maybe_print_plateau_reminder(tdir)
    out = capsys.readouterr().out
    assert "plateau escalation" in out
    # silent log appended
    log = tdir / ".zyme" / PG.SILENT_LOG_NAME
    assert log.exists()
    entry = json.loads(log.read_text().strip())
    assert entry["reason"] == "tier2_consecutive_discards"
    st = PG._read_state(tdir)
    assert "last_tier2_warned_round" in st


def test_maybe_print_tier2_cooldown_still_logs(tdir, capsys):
    rows = [_row(1, "keep", 30.0)]
    rows += [_row(r, "discard", 5.0) for r in range(2, 13)]  # current_round 12
    _write_results(tdir, rows)
    PG._write_state(tdir, {"last_tier2_warned_round": 10})  # 12-10=2 < 5 cooldown
    PG.maybe_print_plateau_reminder(tdir)
    out = capsys.readouterr().out
    assert out == ""  # console suppressed
    # but the audit log still got an append
    log = tdir / ".zyme" / PG.SILENT_LOG_NAME
    assert log.exists()
    assert len(log.read_text().strip().splitlines()) == 1


def test_silent_log_reason_low_gain(tdir):
    # decide-by-low-gain (tail < 10) -> reason tier2_low_gain_over_10.
    rows = [_row(r, "keep", 70.0) for r in range(1, 12)]
    _write_results(tdir, rows)
    PG.maybe_print_plateau_reminder(tdir)
    log = tdir / ".zyme" / PG.SILENT_LOG_NAME
    entry = json.loads(log.read_text().strip().splitlines()[0])
    assert entry["reason"] == "tier2_low_gain_over_10"
