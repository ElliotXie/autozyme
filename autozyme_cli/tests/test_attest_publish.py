from pathlib import Path

from zyme.commands.attest import (
    _cached_oom_tiers,
    _new_attest_rows_all_pass,
    _publish_after_attest,
    _read_verify_rows,
    _resolve_manifest_publish_target,
)


PACKAGE_VERIFY = """\
timestamp\tpatch_name\ttier\trep_idx\tvariant\tsec\tspeedup_pct\tspeedup_x\tpeak_mb\tpeak_mb_change_pct\tpeak_mb_fold\tpass\tmetrics_json\tframework_version\tnote\tsystem_os\tsystem_cpu\tsystem_ram_gb\tsystem_threads
2026-05-24T00:00:00\tseurat\tsmall\t1\tbaseline\t3.0\t\t\t1000\t\t\t\t{}\t0.3.0\t\tWindows\tCPU\t127.2\t1
2026-05-24T00:00:00\tseurat\tsmall\t1\tpatched\t0.1\t96.7\t30.0\t900\t10.0\t1.1\ttrue\t{"max_rel_err": 0}\t0.3.0\t\tWindows\tCPU\t127.2\t1
"""

OOM_SENTINEL = """\
timestamp\tpatch_name\ttier\trep_idx\tvariant\tsec\tspeedup_pct\tspeedup_x\tpeak_mb\tpeak_mb_change_pct\tpeak_mb_fold\tpass\tmetrics_json\tframework_version\tnote\tsystem_os\tsystem_cpu\tsystem_ram_gb\tsystem_threads
2026-05-24T00:00:00\tseurat\tlarge\t\t\t\t\t\t\t\t\t\t\t0.3.0\tOOM: memory limit reached\tmacOS\tCPU\t32.0\t1
"""


def _make_framework(tmp_path: Path) -> tuple[Path, Path]:
    framework = tmp_path / "autozyme-framework"
    # find_framework_root() requires an autozyme_cli child to disambiguate the
    # real framework root from arbitrary directories with the same name.
    (framework / "autozyme_cli").mkdir(parents=True)
    task_dir = (
        framework
        / "optimized_task"
        / "test_seurat_scanpy"
        / "normalize_data"
        / "v1"
    )
    task_dir.mkdir(parents=True)
    (task_dir / "task.yaml").write_text(
        "target_function: Seurat::NormalizeData\n",
        encoding="utf-8",
    )
    scripts = framework / "scripts"
    scripts.mkdir()
    (scripts / "seurat_attest_manifest.yaml").write_text(
        """\
patch: seurat
lang: R
tasks:
  - id: normalize
    path: optimized_task/test_seurat_scanpy/normalize_data/v1
    legacy_key: normalize
""",
        encoding="utf-8",
    )
    return framework, task_dir


def test_manifest_publish_target_resolves_seurat_method_file(tmp_path):
    framework, task_dir = _make_framework(tmp_path)

    resolved = _resolve_manifest_publish_target(task_dir)

    assert resolved == (
        "normalize",
        framework
        / "autozyme_r"
        / "inst"
        / "patches"
        / "seurat_normalize"
        / "speedups.tsv",
    )


def test_publish_after_attest_writes_package_snapshot_only(tmp_path):
    framework, task_dir = _make_framework(tmp_path)
    (task_dir / "package_verify.tsv").write_text(PACKAGE_VERIFY, encoding="utf-8")

    _publish_after_attest(task_dir)

    package_speedups = (
        framework
        / "autozyme_r"
        / "inst"
        / "patches"
        / "seurat_normalize"
        / "speedups.tsv"
    )
    assert package_speedups.is_file()
    assert "patched" in package_speedups.read_text(encoding="utf-8")
    assert not (task_dir / "speedups.tsv").exists()


def test_nonzero_worker_can_be_tolerated_only_for_new_passing_rows(tmp_path):
    _, task_dir = _make_framework(tmp_path)
    before_rows = _read_verify_rows(task_dir / "package_verify.tsv")
    (task_dir / "package_verify.tsv").write_text(PACKAGE_VERIFY, encoding="utf-8")
    after_rows = _read_verify_rows(task_dir / "package_verify.tsv")

    assert _new_attest_rows_all_pass(before_rows, after_rows)


def test_nonzero_worker_is_not_tolerated_for_failed_rows(tmp_path):
    _, task_dir = _make_framework(tmp_path)
    failed = PACKAGE_VERIFY.replace("\ttrue\t", "\tfalse\t")
    (task_dir / "package_verify.tsv").write_text(failed, encoding="utf-8")

    assert not _new_attest_rows_all_pass(
        [],
        _read_verify_rows(task_dir / "package_verify.tsv"),
    )


def test_cached_oom_tiers_match_same_host_latest_attempt(tmp_path):
    _, task_dir = _make_framework(tmp_path)
    (task_dir / "package_verify.tsv").write_text(OOM_SENTINEL, encoding="utf-8")

    assert _cached_oom_tiers(
        task_dir,
        "seurat",
        ("large",),
        {"platform": "mac", "cpu": "CPU", "ram_gb": 32.0, "threads": 1},
    ) == ["large"]


def test_cached_oom_tiers_ignored_after_later_success(tmp_path):
    _, task_dir = _make_framework(tmp_path)
    later_success = PACKAGE_VERIFY.replace("small", "large").replace(
        "Windows\tCPU\t127.2\t1", "macOS\tCPU\t32.0\t1"
    ).replace("2026-05-24T00:00:00", "2026-05-25T00:00:00")
    (task_dir / "package_verify.tsv").write_text(
        OOM_SENTINEL + "\n".join(later_success.splitlines()[1:]) + "\n",
        encoding="utf-8",
    )

    assert _cached_oom_tiers(
        task_dir,
        "seurat",
        ("large",),
        {"platform": "mac", "cpu": "CPU", "ram_gb": 32.0, "threads": 1},
    ) == []
