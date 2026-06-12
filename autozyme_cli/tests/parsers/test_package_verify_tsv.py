"""Tests for package_verify.tsv filtering used by publish-speedups."""
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pytest

from zyme.parsers.package_verify_tsv import (
    LONG_HEADER,
    PublishFilter,
    PublishFilterError,
    filter_package_verify_rows,
    merge_published_tsvs,
    prepare_publish_content,
    read_package_verify,
    row_is_oom_sentinel,
    summarize_batches,
)


_HEADER = "\t".join(LONG_HEADER) + "\n"


def _write_tsv(path: Path, body: str) -> None:
    path.write_text(_HEADER + body, encoding="utf-8")


def _row(
    timestamp: str,
    tier: str,
    variant: str,
    sec: str,
    *,
    rep_idx: int = 1,
    pass_cell: str = "true",
    system_os: str = "macOS",
) -> str:
    row = {
        "timestamp": timestamp,
        "patch_name": "p",
        "tier": tier,
        "dataset": "",
        "rep_idx": str(rep_idx),
        "variant": variant,
        "sec": sec,
        "speedup_pct": "50.0" if variant == "patched" else "",
        "speedup_x": "2.0" if variant == "patched" else "",
        "peak_mb": "100",
        "peak_mb_change_pct": "",
        "peak_mb_fold": "",
        "pass": pass_cell if variant == "patched" else "",
        "metrics_json": "{}",
        "framework_version": "0.3.0",
        "package_version": "0.3.0",
        "note": "",
        "system_os": system_os,
        "system_cpu": "CPU",
        "system_ram_gb": "32",
        "system_threads": "1",
    }
    return "\t".join(row[col] for col in LONG_HEADER) + "\n"


def _batch(
    timestamp: str,
    tier: str,
    baseline_sec: str,
    patched_sec: str,
    *,
    pass_cell: str = "true",
    system_os: str = "macOS",
) -> str:
    return (
        _row(timestamp, tier, "baseline", baseline_sec, system_os=system_os)
        + _row(
            timestamp,
            tier,
            "patched",
            patched_sec,
            pass_cell=pass_cell,
            system_os=system_os,
        )
    )


def _sentinel(
    timestamp: str,
    tier: str,
    note: str,
    *,
    system_os: str = "macOS",
) -> str:
    row = {
        "timestamp": timestamp,
        "patch_name": "p",
        "tier": tier,
        "dataset": "",
        "rep_idx": "",
        "variant": "",
        "sec": "",
        "speedup_pct": "",
        "speedup_x": "",
        "peak_mb": "",
        "peak_mb_change_pct": "",
        "peak_mb_fold": "",
        "pass": "",
        "metrics_json": "",
        "framework_version": "0.3.0",
        "package_version": "0.3.0",
        "note": note,
        "system_os": system_os,
        "system_cpu": "CPU",
        "system_ram_gb": "32",
        "system_threads": "1",
    }
    return "\t".join(row[col] for col in LONG_HEADER) + "\n"


def test_read_and_latest_per_tier(tmp_path: Path) -> None:
    _write_tsv(
        tmp_path / "package_verify.tsv",
        _batch("2026-01-01T10:00:00", "tiny", "10", "5")
        + _batch("2026-01-02T10:00:00", "tiny", "8", "4")
        + _batch("2026-01-02T10:00:00", "medium", "40", "10")
        + _batch("2026-01-01T10:00:00", "medium", "50", "20"),
    )
    _, rows = read_package_verify(tmp_path / "package_verify.tsv")
    flt = PublishFilter(select="latest-per-tier")
    out = filter_package_verify_rows(rows, flt)
    assert len(out) == 4
    by_tier = {
        r["tier"]: r["timestamp"]
        for r in out
        if r["variant"] == "patched"
    }
    assert by_tier["tiny"] == "2026-01-02T10:00:00"
    assert by_tier["medium"] == "2026-01-02T10:00:00"


def test_latest_run_same_timestamp_batch(tmp_path: Path) -> None:
    _write_tsv(
        tmp_path / "package_verify.tsv",
        _batch("2026-01-01T10:00:00", "tiny", "10", "5")
        + _batch("2026-01-01T10:00:00", "medium", "40", "10")
        + _batch("2026-01-02T10:00:00", "tiny", "8", "4"),
    )
    _, rows = read_package_verify(tmp_path / "package_verify.tsv")
    out = filter_package_verify_rows(rows, PublishFilter(select="latest-run"))
    assert len(out) == 2
    assert {r["tier"] for r in out} == {"tiny"}
    assert {r["variant"] for r in out} == {"baseline", "patched"}
    assert out[0]["timestamp"] == "2026-01-02T10:00:00"


def test_tail_n_rows(tmp_path: Path) -> None:
    lines = [
        _row(f"2026-01-0{i}T10:00:00", "tiny", "patched", str(i))
        for i in range(1, 8)
    ]
    _write_tsv(tmp_path / "package_verify.tsv", "".join(lines))
    _, rows = read_package_verify(tmp_path / "package_verify.tsv")
    out = filter_package_verify_rows(rows, PublishFilter(select="tail", tail=5))
    assert len(out) == 5
    assert out[0]["sec"] == "3"
    assert out[-1]["sec"] == "7"


def test_require_all_tiers_skips_incomplete(tmp_path: Path) -> None:
    _write_tsv(
        tmp_path / "package_verify.tsv",
        _batch("2026-01-02T10:00:00", "tiny", "8", "4"),
    )
    _, rows = read_package_verify(tmp_path / "package_verify.tsv")
    flt = PublishFilter(select="latest-per-tier", require_all_tiers=True)
    with pytest.raises(PublishFilterError, match="missing tier"):
        filter_package_verify_rows(rows, flt)


def test_require_all_pass(tmp_path: Path) -> None:
    _write_tsv(
        tmp_path / "package_verify.tsv",
        _batch(
            "2026-01-02T10:00:00",
            "tiny",
            "8",
            "4",
            pass_cell="false",
        ),
    )
    _, rows = read_package_verify(tmp_path / "package_verify.tsv")
    flt = PublishFilter(select="latest-per-tier", require_all_pass=True)
    with pytest.raises(PublishFilterError, match="require-all-pass"):
        filter_package_verify_rows(rows, flt)


def test_platform_filter(tmp_path: Path) -> None:
    _write_tsv(
        tmp_path / "package_verify.tsv",
        _batch(
            "2026-01-01T10:00:00",
            "tiny",
            "10",
            "5",
            system_os="Windows 10",
        )
        + _batch(
            "2026-01-02T10:00:00",
            "tiny",
            "8",
            "4",
            system_os="macOS 14",
        ),
    )
    _, rows = read_package_verify(tmp_path / "package_verify.tsv")
    out = filter_package_verify_rows(
        rows, PublishFilter(select="latest-per-tier", platform="win"),
    )
    assert len(out) == 2
    assert "Windows" in out[0]["system_os"]


def test_prepare_publish_content_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "package_verify.tsv"
    _write_tsv(
        src,
        _batch("2026-01-01T10:00:00", "tiny", "10", "5")
        + _batch("2026-01-02T10:00:00", "tiny", "8", "4"),
    )
    text, n, summary = prepare_publish_content(
        src, PublishFilter(select="latest-per-tier"),
    )
    assert n == 2
    assert "select=latest-per-tier" in summary
    assert "2026-01-02T10:00:00" in text
    assert "2026-01-01T10:00:00" not in text


def test_sentinel_rows_are_publishable_under_all_pass_only(tmp_path: Path) -> None:
    src = tmp_path / "package_verify.tsv"
    _write_tsv(
        src,
        _batch("2026-01-01T10:00:00", "large", "10", "5")
        + _sentinel(
            "2026-01-02T10:00:00",
            "large",
            "OOM: memory limit reached",
        ),
    )

    text, n, _summary = prepare_publish_content(
        src, PublishFilter(select="latest-per-tier", all_pass_only=True),
    )

    assert n == 1
    assert "OOM: memory limit reached" in text
    assert "patched" not in text


def test_sentinel_rows_merge_and_summarize_as_crashed(tmp_path: Path) -> None:
    body = _sentinel(
        "2026-01-02T10:00:00",
        "large",
        "Error: vector memory limit reached",
    )
    text = _HEADER + body

    merged, stats = merge_published_tsvs(_HEADER, text)
    assert stats.added == 1
    assert "vector memory limit" in merged

    rows = list(csv.DictReader(StringIO(text), delimiter="\t"))
    summaries = summarize_batches(rows)
    assert len(summaries) == 1
    assert summaries[0].n_reps == 0
    assert not summaries[0].all_pass
    assert row_is_oom_sentinel(rows[0])
