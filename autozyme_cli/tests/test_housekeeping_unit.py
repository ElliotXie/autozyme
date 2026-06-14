"""Unit tests for zyme.housekeeping — the mechanical dead-code scanner
(Findings), cooldown gating, state file roundtrip + legacy migration, and the
reminder formatter.
"""
from __future__ import annotations

import json

import pytest

from zyme import housekeeping as HK


# ---------------------------------------------------------------------------
# _find_pipeline
# ---------------------------------------------------------------------------

def test_find_pipeline_none(tmp_path):
    assert HK._find_pipeline(tmp_path) is None


def test_find_pipeline_prefers_R(tmp_path):
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "run.R").write_text("x <- 1\n")
    (tmp_path / "pipeline" / "run.py").write_text("x = 1\n")
    p = HK._find_pipeline(tmp_path)
    assert p.name == "run.R"


def test_find_pipeline_py(tmp_path):
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "run.py").write_text("x = 1\n")
    p = HK._find_pipeline(tmp_path)
    assert p.name == "run.py"


# ---------------------------------------------------------------------------
# state file: read / write / legacy migration
# ---------------------------------------------------------------------------

def test_read_state_missing(tmp_path):
    assert HK._read_state(tmp_path) is None


def test_write_then_read_state(tmp_path):
    HK._write_state(tmp_path, {"last_warned_round": 7})
    assert HK._read_state(tmp_path) == {"last_warned_round": 7}
    # canonical location
    assert (tmp_path / ".zyme" / HK.STATE_FILE_NAME).exists()


def test_read_state_malformed(tmp_path):
    (tmp_path / ".zyme").mkdir()
    (tmp_path / ".zyme" / HK.STATE_FILE_NAME).write_text("{broken")
    assert HK._read_state(tmp_path) is None


def test_state_path_migrates_legacy(tmp_path):
    legacy = tmp_path / HK.LEGACY_STATE_FILE
    legacy.write_text(json.dumps({"last_warned_round": 3}))
    canonical = HK._state_path(tmp_path)
    assert canonical == tmp_path / ".zyme" / HK.STATE_FILE_NAME
    assert canonical.exists()
    assert not legacy.exists()
    assert json.loads(canonical.read_text())["last_warned_round"] == 3


def test_state_path_legacy_not_clobbered_when_canonical_exists(tmp_path):
    (tmp_path / ".zyme").mkdir()
    canonical = tmp_path / ".zyme" / HK.STATE_FILE_NAME
    canonical.write_text(json.dumps({"last_warned_round": 99}))
    legacy = tmp_path / HK.LEGACY_STATE_FILE
    legacy.write_text(json.dumps({"last_warned_round": 1}))
    result = HK._state_path(tmp_path)
    assert result == canonical
    # legacy stays in place (canonical already had the truth)
    assert legacy.exists()
    assert json.loads(canonical.read_text())["last_warned_round"] == 99


# ---------------------------------------------------------------------------
# _scan — dead fast_* helpers (M2)
# ---------------------------------------------------------------------------

def _scan_src(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(src)
    return HK._scan(p)


def test_scan_dead_fast_py(tmp_path):
    src = (
        "def fast_helper(x):\n"
        "    return x\n"
        "\n"
        "def main():\n"
        "    return 1\n"
    )
    f = _scan_src(tmp_path, "run.py", src)
    assert len(f.dead_fast) == 1
    assert f.dead_fast[0][1] == "fast_helper"
    assert f.fires()


def test_scan_fast_installed_not_dead(tmp_path):
    # fast_norm referenced in install_override -> not dead.
    src = (
        "def fast_norm(x):\n"
        "    return x\n"
        "install_override('mod', 'fn', fast_norm)\n"
    )
    f = _scan_src(tmp_path, "run.py", src)
    assert f.dead_fast == []


def test_scan_fast_called_not_dead(tmp_path):
    src = (
        "def fast_norm(x):\n"
        "    return x\n"
        "y = fast_norm(3)\n"
        "z = fast_norm(4)\n"
    )
    f = _scan_src(tmp_path, "run.py", src)
    assert f.dead_fast == []


def test_scan_fast_attribute_assignment_not_dead(tmp_path):
    src = (
        "def fast_pca(x):\n"
        "    return x\n"
        "obj.method = fast_pca\n"
    )
    f = _scan_src(tmp_path, "run.py", src)
    assert f.dead_fast == []


def test_scan_dead_fast_r(tmp_path):
    src = (
        "fast_norm <- function(x) x\n"
        "main <- function() 1\n"
    )
    f = _scan_src(tmp_path, "run.R", src)
    assert len(f.dead_fast) == 1
    assert f.dead_fast[0][1] == "fast_norm"


def test_scan_r_fast_installed_not_dead(tmp_path):
    src = (
        "fast_norm <- function(x) x\n"
        "install_override('NormalizeData', 'Seurat', fast_norm)\n"
    )
    f = _scan_src(tmp_path, "run.R", src)
    assert f.dead_fast == []


# ---------------------------------------------------------------------------
# _scan — dead orig captures (M5)
# ---------------------------------------------------------------------------

def test_scan_dead_orig_py(tmp_path):
    src = "orig_fn = something\n"
    f = _scan_src(tmp_path, "run.py", src)
    assert len(f.dead_orig) == 1
    assert f.dead_orig[0][1] == "orig_fn"


def test_scan_orig_referenced_not_dead(tmp_path):
    src = (
        "orig_fn = capture()\n"
        "result = orig_fn(data)\n"
    )
    f = _scan_src(tmp_path, "run.py", src)
    assert f.dead_orig == []


def test_scan_dead_orig_r(tmp_path):
    src = ".orig_initZ <- getFromNamespace('.x', 'celda')\n"
    f = _scan_src(tmp_path, "run.R", src)
    assert len(f.dead_orig) == 1
    assert f.dead_orig[0][1] == ".orig_initZ"


# ---------------------------------------------------------------------------
# _scan — stale round-history comments (M10)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("comment", [
    "# removed in round 5",
    "# reverted in round 12",
    "# deprecated in round 3",
    "# dead code in round 7",
    "# see active_opts",
    "# REMOVED IN ROUND 9 (case-insensitive)",
])
def test_scan_stale_comments(tmp_path, comment):
    src = f"x = 1  {comment}\n"
    f = _scan_src(tmp_path, "run.py", src)
    assert len(f.stale) == 1


def test_scan_no_stale_for_plain_comment(tmp_path):
    src = "x = 1  # just a normal comment\n"
    f = _scan_src(tmp_path, "run.py", src)
    assert f.stale == []


# ---------------------------------------------------------------------------
# _scan — counters + line threshold
# ---------------------------------------------------------------------------

def test_scan_counts_install_and_cpp(tmp_path):
    src = (
        "install_override('a', 'b', c)\n"
        "install_global_override('d', 'e', f)\n"
        "getFromNamespace('.x', 'pkg')\n"
        "sourceCpp('k.cpp')\n"
        "cppFunction('int z(){return 0;}')\n"
    )
    f = _scan_src(tmp_path, "run.R", src)
    assert f.n_install == 2
    assert f.n_getfromns == 1
    assert f.n_cpp_blocks == 2


def test_scan_total_lines(tmp_path):
    src = "a = 1\nb = 2\nc = 3\n"
    f = _scan_src(tmp_path, "run.py", src)
    # 3 newlines + 1 = 4 (trailing newline counts a final empty line).
    assert f.total_lines == 4


def test_fires_on_line_threshold_only(tmp_path):
    src = "\n".join(f"x{i} = {i}" for i in range(HK.LINE_THRESHOLD + 5)) + "\n"
    f = _scan_src(tmp_path, "run.py", src)
    assert not f.dead_fast and not f.dead_orig and not f.stale
    assert f.total_lines > HK.LINE_THRESHOLD
    assert f.fires()


def test_clean_file_does_not_fire(tmp_path):
    src = "def main():\n    return 1\n"
    f = _scan_src(tmp_path, "run.py", src)
    assert not f.fires()


# ---------------------------------------------------------------------------
# Findings.format_reminder
# ---------------------------------------------------------------------------

def test_format_reminder_includes_findings(tmp_path):
    src = (
        "def fast_dead(x):\n"
        "    return x\n"
        "orig_cap = capture()\n"
        "y = 1  # removed in round 4\n"
    )
    p = tmp_path / "run.py"
    p.write_text(src)
    f = HK._scan(p)
    msg = f.format_reminder(p)
    assert "housekeeping reminder" in msg
    assert "fast_dead" in msg
    assert "orig_cap" in msg
    assert "stale round-history" in msg
    assert HK.HOUSEKEEPING_TAG in msg
    assert str(HK.COOLDOWN_ROUNDS) in msg


def test_format_reminder_line_threshold_branch(tmp_path):
    # No dead helpers; only line threshold -> the ">{N} lines" branch.
    src = "\n".join(f"x{i}=1" for i in range(HK.LINE_THRESHOLD + 2)) + "\n"
    p = tmp_path / "run.py"
    p.write_text(src)
    f = HK._scan(p)
    msg = f.format_reminder(p)
    assert f"> {HK.LINE_THRESHOLD} lines" in msg


def test_format_reminder_truncates_many_stale(tmp_path):
    lines = [f"x{i}=1  # removed in round {i}" for i in range(6)]
    p = tmp_path / "run.py"
    p.write_text("\n".join(lines) + "\n")
    f = HK._scan(p)
    msg = f.format_reminder(p)
    assert "and 3 more" in msg  # 6 stale, first 3 shown


# ---------------------------------------------------------------------------
# maybe_print_reminder — cooldown gate behavior
# ---------------------------------------------------------------------------

def _make_task_with_dead_pipeline(tmp_path):
    pipe = tmp_path / "pipeline"
    pipe.mkdir()
    (pipe / "run.py").write_text("def fast_dead(x):\n    return x\n")
    return tmp_path


def test_maybe_print_no_pipeline_silent(tmp_path, capsys):
    HK.maybe_print_reminder(tmp_path, current_round=1)
    assert capsys.readouterr().out == ""
    assert HK._read_state(tmp_path) is None


def test_maybe_print_fires_and_records(tmp_path, capsys):
    _make_task_with_dead_pipeline(tmp_path)
    HK.maybe_print_reminder(tmp_path, current_round=10)
    out = capsys.readouterr().out
    assert "housekeeping reminder" in out
    assert HK._read_state(tmp_path)["last_warned_round"] == 10


def test_maybe_print_cooldown_suppresses(tmp_path, capsys):
    _make_task_with_dead_pipeline(tmp_path)
    HK._write_state(tmp_path, {"last_warned_round": 10})
    # round 12 is within COOLDOWN_ROUNDS(15) of 10 -> suppressed.
    HK.maybe_print_reminder(tmp_path, current_round=12)
    assert capsys.readouterr().out == ""
    # state unchanged
    assert HK._read_state(tmp_path)["last_warned_round"] == 10


def test_maybe_print_after_cooldown_fires_again(tmp_path, capsys):
    _make_task_with_dead_pipeline(tmp_path)
    HK._write_state(tmp_path, {"last_warned_round": 10})
    # round 26 is >= 15 rounds later -> fires.
    HK.maybe_print_reminder(tmp_path, current_round=26)
    assert "housekeeping reminder" in capsys.readouterr().out
    assert HK._read_state(tmp_path)["last_warned_round"] == 26


def test_maybe_print_housekeeping_hypothesis_resets_silently(tmp_path, capsys):
    _make_task_with_dead_pipeline(tmp_path)
    HK.maybe_print_reminder(tmp_path, current_round=10,
                            hypothesis="[housekeeping] cleaned dead helpers")
    assert capsys.readouterr().out == ""
    # cooldown reset to current_round even though no reminder printed.
    assert HK._read_state(tmp_path)["last_warned_round"] == 10


def test_maybe_print_housekeeping_hypothesis_leading_ws(tmp_path, capsys):
    _make_task_with_dead_pipeline(tmp_path)
    HK.maybe_print_reminder(tmp_path, current_round=5,
                            hypothesis="   [housekeeping] trimmed")
    assert capsys.readouterr().out == ""
    assert HK._read_state(tmp_path)["last_warned_round"] == 5


def test_maybe_print_no_findings_no_print(tmp_path, capsys):
    pipe = tmp_path / "pipeline"
    pipe.mkdir()
    (pipe / "run.py").write_text("def main():\n    return 1\n")
    HK.maybe_print_reminder(tmp_path, current_round=50)
    assert capsys.readouterr().out == ""
    # clean scan -> no state written
    assert HK._read_state(tmp_path) is None


# ---------------------------------------------------------------------------
# mark_dismissed
# ---------------------------------------------------------------------------

def test_mark_dismissed_resets_cooldown(tmp_path):
    HK.mark_dismissed(tmp_path, current_round=42)
    assert HK._read_state(tmp_path)["last_warned_round"] == 42
