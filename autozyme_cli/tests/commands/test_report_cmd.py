"""Unit tests for zyme.commands.report — the `zyme report` dispatch command.

`cmd_report` is a thin wrapper: resolve the task dir, refuse if there's no
results.tsv, otherwise call `zyme.report.render_report` and print where the
HTML landed (noting whether a narrative file was consumed). It can also open
a browser when `--open` is passed.

We exercise:
  - the no-results.tsv guard (die / SystemExit),
  - the full render path end-to-end (results.tsv on this machine renders fine),
  - output-path / narrative-path argument plumbing (monkeypatching
    render_report to capture what it was handed),
  - the narrative-used vs mechanical-only print branch,
  - the --open branch (monkeypatching webbrowser so no browser launches),
  - the --open failure branch (webbrowser raising is swallowed with a note).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import zyme.commands.report as report_cmd
from zyme.commands.report import cmd_report


_RESULTS = (
    "round\tcommit\tdataset\tspeed_sec\tspeedup_pct\tpeak_mb\tstatus\t"
    "metrics_json\thypothesis\tdescription\tphase\tthread\n"
    "0\tabc1234\ttiny_a\t10.0\t0.0\t512.0\tbaseline\t{}\tupstream\t\toptimize\t1\n"
    "1\tdef5678\ttiny_a\t8.0\t20.0\t500.0\tkeep\t{}\t[c] X\tnotes\toptimize\t1\n"
)


def _args(task_dir: Path, *, output=None, narrative=None, open=False):
    return SimpleNamespace(
        task_dir=str(task_dir), output=output, narrative=narrative, open=open,
    )


def _with_results(task_dir: Path) -> Path:
    (task_dir / "results.tsv").write_text(_RESULTS)
    return task_dir


# --------------------------------------------------------------------------
# Guard: no results.tsv
# --------------------------------------------------------------------------

class TestNoResultsGuard:
    def test_missing_results_tsv_dies(self, task_dir):
        # task_dir fixture has task.yaml but no results.tsv.
        with pytest.raises(SystemExit) as ei:
            cmd_report(_args(task_dir))
        assert ei.value.code == 1

    def test_missing_results_tsv_message(self, task_dir, capsys):
        with pytest.raises(SystemExit):
            cmd_report(_args(task_dir))
        err = capsys.readouterr().err
        assert "no results.tsv" in err


# --------------------------------------------------------------------------
# render_report dispatch — capture what cmd_report hands the renderer
# --------------------------------------------------------------------------

class TestRenderDispatch:
    def test_passes_none_paths_by_default(self, task_dir, monkeypatch):
        _with_results(task_dir)
        seen = {}

        def fake_render(td, *, output_path=None, narrative_path=None):
            seen["task_dir"] = td
            seen["output_path"] = output_path
            seen["narrative_path"] = narrative_path
            return td / "report.html"

        monkeypatch.setattr(report_cmd, "render_report", fake_render)
        cmd_report(_args(task_dir))
        assert seen["task_dir"] == task_dir
        assert seen["output_path"] is None
        assert seen["narrative_path"] is None

    def test_output_and_narrative_paths_threaded(self, task_dir, monkeypatch,
                                                 tmp_path):
        _with_results(task_dir)
        out = tmp_path / "custom.html"
        narr = tmp_path / "narr.md"
        narr.write_text("# narrative\n")
        seen = {}

        def fake_render(td, *, output_path=None, narrative_path=None):
            seen["output_path"] = output_path
            seen["narrative_path"] = narrative_path
            return output_path or (td / "report.html")

        monkeypatch.setattr(report_cmd, "render_report", fake_render)
        cmd_report(_args(task_dir, output=str(out), narrative=str(narr)))
        assert seen["output_path"] == out
        assert seen["narrative_path"] == narr


# --------------------------------------------------------------------------
# Narrative-used vs mechanical-only print branch
# --------------------------------------------------------------------------

class TestNarrativeNote:
    def _patch_render(self, monkeypatch, out_path):
        def fake_render(td, *, output_path=None, narrative_path=None):
            return out_path
        monkeypatch.setattr(report_cmd, "render_report", fake_render)

    def test_mechanical_only_when_no_narrative(self, task_dir, monkeypatch,
                                               capsys):
        _with_results(task_dir)
        out_path = task_dir / "report.html"
        self._patch_render(monkeypatch, out_path)
        cmd_report(_args(task_dir))
        msg = capsys.readouterr().out
        assert f"wrote {out_path}" in msg
        assert "mechanical only" in msg

    def test_default_narrative_path_detected(self, task_dir, monkeypatch,
                                             capsys):
        _with_results(task_dir)
        # Drop a narrative at the default location memory/report_narrative.md.
        narr = task_dir / "memory" / "report_narrative.md"
        narr.parent.mkdir(parents=True, exist_ok=True)
        narr.write_text("# n\n")
        self._patch_render(monkeypatch, task_dir / "report.html")
        cmd_report(_args(task_dir))
        assert "with narrative" in capsys.readouterr().out

    def test_explicit_narrative_path_detected(self, task_dir, monkeypatch,
                                              capsys, tmp_path):
        _with_results(task_dir)
        narr = tmp_path / "n.md"
        narr.write_text("# n\n")
        self._patch_render(monkeypatch, task_dir / "report.html")
        cmd_report(_args(task_dir, narrative=str(narr)))
        assert "with narrative" in capsys.readouterr().out


# --------------------------------------------------------------------------
# --open branch
# --------------------------------------------------------------------------

class TestOpenBranch:
    def test_open_invokes_webbrowser(self, task_dir, monkeypatch):
        _with_results(task_dir)
        out_path = task_dir / "report.html"
        monkeypatch.setattr(
            report_cmd, "render_report",
            lambda td, **kw: out_path,
        )
        opened = {}
        monkeypatch.setattr(
            report_cmd.webbrowser, "open",
            lambda uri: opened.setdefault("uri", uri),
        )
        cmd_report(_args(task_dir, open=True))
        assert opened["uri"] == out_path.as_uri()

    def test_open_failure_is_swallowed(self, task_dir, monkeypatch, capsys):
        _with_results(task_dir)
        out_path = task_dir / "report.html"
        monkeypatch.setattr(
            report_cmd, "render_report",
            lambda td, **kw: out_path,
        )

        def boom(uri):
            raise RuntimeError("no display")

        monkeypatch.setattr(report_cmd.webbrowser, "open", boom)
        # Must not raise — the failure prints a note and returns.
        cmd_report(_args(task_dir, open=True))
        assert "could not open browser" in capsys.readouterr().out


# --------------------------------------------------------------------------
# End-to-end against the real renderer (all upstreams installed on this box)
# --------------------------------------------------------------------------

class TestRealRender:
    def test_real_render_writes_html(self, task_dir, capsys):
        _with_results(task_dir)
        cmd_report(_args(task_dir))
        out = capsys.readouterr().out
        report_html = task_dir / "report.html"
        assert report_html.exists()
        assert f"wrote {report_html}" in out
        # Self-contained HTML.
        text = report_html.read_text()
        assert "<html" in text.lower() or "<!doctype html" in text.lower()
