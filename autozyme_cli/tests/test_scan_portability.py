"""Tests for zyme.scan_portability — hazard scan + .zyme/portability_scan.json."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from zyme.scan_portability import (
    PORTABILITY_SCAN_REL,
    load_portability_scan,
    save_portability_scan,
    scan_task,
)


def _write_pipeline(task: Path, content: str, lang: str = "R") -> None:
    (task / "pipeline").mkdir(parents=True, exist_ok=True)
    name = "run.R" if lang == "R" else "run.py"
    (task / "pipeline" / name).write_text(content, encoding="utf-8")


def _scaffold_task(tmp_path: Path, name: str = "test_foo") -> Path:
    task = tmp_path / name
    task.mkdir()
    (task / "task.yaml").write_text(f"task: {name}\n")
    return task


class TestPortabilityScan:
    def test_clean_no_parallel(self, tmp_path: Path):
        task = _scaffold_task(tmp_path)
        _write_pipeline(task, 'result <- 1 + 1\n')
        res = scan_task(task)
        assert res.verdict == "CLEAN"
        assert res.run_3_5 is False
        assert res.action == "skip_3_5"

    def test_clean_rcpp_parallel(self, tmp_path: Path):
        task = _scaffold_task(tmp_path)
        _write_pipeline(task, "RcppParallel::parallelFor(0, n, worker);\n", lang="R")
        res = scan_task(task)
        assert res.verdict == "CLEAN"
        assert res.run_3_5 is False

    def test_crash_raw_mclapply(self, tmp_path: Path):
        task = _scaffold_task(tmp_path)
        _write_pipeline(task, """
out <- parallel::mclapply(1:10, function(x) x, mc.cores = 4)
""")
        res = scan_task(task)
        assert res.verdict == "CRASH-ON-WIN"
        assert res.run_3_5 is True
        assert any(h.pattern == "raw_parallel_mclapply" for h in res.hits)

    def test_mac_bonus_zyme_mclapply(self, tmp_path: Path):
        task = _scaffold_task(tmp_path)
        _write_pipeline(task, """
out <- .zyme_mclapply(1:10, function(x) x, mc.cores = 4)
""")
        res = scan_task(task)
        assert res.verdict == "MAC_BONUS_ONLY"
        assert res.run_3_5 is False

    def test_mac_bonus_guarded_mclapply(self, tmp_path: Path):
        task = _scaffold_task(tmp_path)
        _write_pipeline(task, """
if (.Platform$OS.type != "windows") {
  out <- parallel::mclapply(1:10, function(x) x, mc.cores = 4)
} else {
  out <- lapply(1:10, function(x) x)
}
""")
        res = scan_task(task)
        assert res.verdict == "MAC_BONUS_ONLY"
        assert res.run_3_5 is False

    def test_harmful_psock(self, tmp_path: Path):
        task = _scaffold_task(tmp_path)
        _write_pipeline(task, """
cl <- parallel::makePSOCKcluster(4)
on.exit(parallel::stopCluster(cl))
out <- parallel::parLapply(cl, 1:10, function(x) x)
""")
        res = scan_task(task)
        assert res.verdict == "HARMFUL-DEFAULT"
        assert res.run_3_5 is True

    def test_needs_kernel_bplapply(self, tmp_path: Path):
        task = _scaffold_task(tmp_path)
        _write_pipeline(task, """
BPPARAM <- BiocParallel::MulticoreParam(workers = 4)
out <- BiocParallel::bplapply(1:10, function(x) x, BPPARAM = BPPARAM)
""")
        res = scan_task(task)
        assert res.verdict in {"NEEDS-KERNEL", "CRASH-ON-WIN"}
        assert res.run_3_5 is True

    def test_writes_zyme_json(self, tmp_path: Path):
        task = _scaffold_task(tmp_path)
        _write_pipeline(task, "x <- 1\n")
        res = scan_task(task)
        path = save_portability_scan(task, res)
        assert path == task / PORTABILITY_SCAN_REL
        loaded = load_portability_scan(task)
        assert loaded is not None
        assert loaded["verdict"] == "CLEAN"
        assert loaded["schema_version"] == 1
        assert "scanned_at" in loaded

    def test_json_roundtrip_fields(self, tmp_path: Path):
        task = _scaffold_task(tmp_path)
        _write_pipeline(task, "parallel::mclapply(1:2, identity, mc.cores=2)\n")
        res = scan_task(task)
        data = res.to_json_dict()
        assert data["run_3_5"] is True
        assert isinstance(data["hits"], list)
        assert data["hits"][0]["line"] >= 1
