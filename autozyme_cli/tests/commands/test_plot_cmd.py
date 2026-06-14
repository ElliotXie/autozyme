"""Unit tests for zyme.commands.plot — convergence-curve renderer.

matplotlib is installed on this machine, so cmd_plot is driven end-to-end with
the Agg backend: a real results.tsv -> real PNG/PDF/SVG written into tmp_path.
We also exercise the data-prep branches (phase filter, dataset filter, log
auto/on, no-decision-rows skip) and the error gates (no results.tsv, empty,
missing dataset).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from zyme.commands import plot as plotmod


# A results.tsv with baseline + keep + discard + crash rows so every render
# branch (accepted line, rejected X, crash marker, speedup annotation) fires.
RESULTS = (
    "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\t"
    "status\tmetrics_json\thypothesis\tdescription\tphase\n"
    "0\tabc\tds1\t100.0\t0.0\t500\tbaseline\t{}\tup\t\toptimize\n"
    "1\tdef\tds1\t60.0\t40.0\t500\tkeep\t{}\th\t\toptimize\n"
    "2\tghi\tds1\t80.0\t20.0\t500\tdiscard\t{}\th\t\toptimize\n"
    "3\tjkl\tds1\t30.0\t70.0\t500\tkeep\t{}\th\t\toptimize\n"
    "4\tmno\tds1\t100.0\t0.0\t500\tcrash\t{}\th\t\toptimize\n"
)


def _task(tmp_path, results=RESULTS):
    (tmp_path / "task.yaml").write_text("target_function: foo\n")
    if results is not None:
        (tmp_path / "results.tsv").write_text(results)
    return tmp_path


def _args(td, **kw):
    base = dict(task_dir=str(td), phase="optimize", dataset=None,
                output_dir=None, title=None, log="auto")
    base.update(kw)
    return SimpleNamespace(**base)


class TestCmdPlotGates:
    def test_no_results_dies(self, tmp_path):
        td = _task(tmp_path, results=None)
        with pytest.raises(SystemExit):
            plotmod.cmd_plot(_args(td))

    def test_empty_results_dies(self, tmp_path):
        td = _task(tmp_path, results="round\tdataset\tstatus\n")
        with pytest.raises(SystemExit):
            plotmod.cmd_plot(_args(td))

    def test_missing_dataset_dies(self, tmp_path):
        td = _task(tmp_path)
        with pytest.raises(SystemExit):
            plotmod.cmd_plot(_args(td, dataset="no_such_ds"))


class TestCmdPlotRender:
    def test_writes_three_formats(self, tmp_path, capsys):
        td = _task(tmp_path)
        plotmod.cmd_plot(_args(td))
        fig_dir = td / "figure"
        pngs = list(fig_dir.glob("*.png"))
        assert len(pngs) == 1
        base = pngs[0].with_suffix("")
        assert base.with_suffix(".pdf").is_file()
        assert base.with_suffix(".svg").is_file()
        assert "convergence_" in pngs[0].name
        assert "ds1" in pngs[0].name

    def test_custom_output_dir_and_title(self, tmp_path):
        td = _task(tmp_path)
        out = tmp_path / "myfigs"
        plotmod.cmd_plot(_args(td, output_dir=str(out), title="My Title"))
        assert list(out.glob("*.png"))

    def test_dataset_filter(self, tmp_path):
        # Two datasets; filter to one.
        results = RESULTS + (
            "0\tabc\tds2\t50.0\t0.0\t300\tbaseline\t{}\tup\t\toptimize\n"
            "1\tdef\tds2\t40.0\t20.0\t300\tkeep\t{}\th\t\toptimize\n")
        td = _task(tmp_path, results=results)
        plotmod.cmd_plot(_args(td, dataset="ds2"))
        pngs = list((td / "figure").glob("*.png"))
        assert len(pngs) == 1
        assert "ds2" in pngs[0].name

    def test_log_on(self, tmp_path):
        td = _task(tmp_path)
        plotmod.cmd_plot(_args(td, log="on"))
        assert list((td / "figure").glob("*.png"))

    def test_log_off(self, tmp_path):
        td = _task(tmp_path)
        plotmod.cmd_plot(_args(td, log="off"))
        assert list((td / "figure").glob("*.png"))

    def test_phase_all(self, tmp_path):
        td = _task(tmp_path)
        plotmod.cmd_plot(_args(td, phase="all"))
        assert list((td / "figure").glob("*.png"))

    def test_no_decision_rows_skipped_then_dies(self, tmp_path, capsys):
        # Only rerun/pending rows -> no decision rows -> nothing written -> die.
        results = (
            "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\t"
            "status\tmetrics_json\thypothesis\tdescription\tphase\n"
            "1\tdef\tds1\t60.0\t40.0\t500\trerun\t{}\th\t\toptimize\n"
            "2\tghi\tds1\t80.0\t20.0\t500\tpending\t{}\th\t\toptimize\n")
        td = _task(tmp_path, results=results)
        with pytest.raises(SystemExit):
            plotmod.cmd_plot(_args(td))


class TestRenderConvergenceDirect:
    """Exercise _render_convergence directly to hit its sub-branches."""

    def _setup_plt(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        return plt, ticker

    def test_single_keep_no_line(self, tmp_path):
        # Only one keep row -> the multi-point line branch (>=2) is skipped.
        plt, ticker = self._setup_plt()
        rows = [
            {"status": "baseline", "speed_sec": "100"},
        ]
        out_base = tmp_path / "fig"
        plotmod._render_convergence(rows, "t", out_base, "off", plt, ticker)
        assert out_base.with_suffix(".png").is_file()

    def test_log_auto_triggers_on_wide_range(self, tmp_path):
        plt, ticker = self._setup_plt()
        rows = [
            {"status": "baseline", "speed_sec": "1000"},
            {"status": "keep", "speed_sec": "10"},
            {"status": "keep", "speed_sec": "5"},
        ]
        out_base = tmp_path / "fig"
        # range 1000/5 = 200 > 20 -> auto log
        plotmod._render_convergence(rows, "t", out_base, "auto", plt, ticker)
        assert out_base.with_suffix(".svg").is_file()

    def test_bad_speed_value_coerced(self, tmp_path):
        plt, ticker = self._setup_plt()
        rows = [
            {"status": "baseline", "speed_sec": "notnum"},
            {"status": "keep", "speed_sec": "10"},
        ]
        out_base = tmp_path / "fig"
        plotmod._render_convergence(rows, "t", out_base, "off", plt, ticker)
        assert out_base.with_suffix(".png").is_file()
