from pathlib import Path

from zyme.commands.bench import (
    _append_experiment_dispatch_doc,
    _reference_text_as_round0_pipeline,
    _reset_pipeline_to_reference,
    _template_check_lines,
    _write_experiment_doc,
)


R_REFERENCE = """\
#!/usr/bin/env Rscript
# reference.R
SCRIPT_DIR <- getwd()
TASK_DIR <- SCRIPT_DIR
TIER <- Sys.getenv("ZYME_TIER", unset = "tiny")
OUTPUT_DIR <- Sys.getenv(
  "ZYME_REFERENCE_DIR",
  unset = file.path(TASK_DIR, "reference_outputs", TIER)
)
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
cat("[reference] Running\\n")
saveRDS(list(ok = TRUE), file = file.path(OUTPUT_DIR, "result.rds"))
"""


def test_reference_text_as_round0_pipeline_r_retargets_paths():
    text = _reference_text_as_round0_pipeline(R_REFERENCE, "R")

    assert "TASK_DIR <- dirname(SCRIPT_DIR)" in text
    assert '"ZYME_TEST_DIR"' in text
    assert '"ZYME_REFERENCE_DIR"' not in text
    assert 'file.path(SCRIPT_DIR, sprintf("output_%s", TIER))' in text
    assert "[pipeline]" in text


def test_reset_pipeline_to_reference_writes_round0_pipeline(tmp_path: Path):
    template = tmp_path
    (template / "pipeline").mkdir()
    (template / "reference.R").write_text(R_REFERENCE, encoding="utf-8")
    (template / "pipeline" / "run.R").write_text(
        'install_override("FindAllMarkers", "Seurat", fast)\n',
        encoding="utf-8",
    )

    reset = _reset_pipeline_to_reference(template)

    assert reset == [("pipeline/run.R", "reference.R")]
    run_text = (template / "pipeline" / "run.R").read_text(encoding="utf-8")
    assert run_text != R_REFERENCE
    assert "TASK_DIR <- dirname(SCRIPT_DIR)" in run_text
    assert '"ZYME_TEST_DIR"' in run_text


def test_reset_pipeline_to_reference_preserves_round0_pipeline(tmp_path: Path):
    template = tmp_path
    (template / "pipeline").mkdir()
    (template / "reference.R").write_text(R_REFERENCE, encoding="utf-8")
    round0 = _reference_text_as_round0_pipeline(R_REFERENCE, "R")
    round0 = round0.replace(
        "saveRDS(list(ok = TRUE), file = file.path(OUTPUT_DIR, \"result.rds\"))",
        "saveRDS(with_profile({ list(ok = TRUE) }), file = file.path(OUTPUT_DIR, \"result.rds\"))",
    )
    (template / "pipeline" / "run.R").write_text(round0, encoding="utf-8")

    reset = _reset_pipeline_to_reference(template)

    assert reset == []
    assert (template / "pipeline" / "run.R").read_text(encoding="utf-8") == round0


def test_iterate_template_doctor_rejects_literal_reference_copy(tmp_path: Path):
    template = tmp_path
    (template / "pipeline").mkdir()
    (template / "bench_template.yaml").write_text("stage: iterate\n", encoding="utf-8")
    (template / "task.yaml").write_text("datasets: []\n", encoding="utf-8")
    (template / "reference.R").write_text(R_REFERENCE, encoding="utf-8")
    (template / "pipeline" / "run.R").write_text(R_REFERENCE, encoding="utf-8")

    errors, _ = _template_check_lines(
        {"stage": "iterate"},
        {"id": "x", "tiers": []},
        template,
    )

    assert any("literal reference copy" in e for e in errors)


def test_iterate_template_doctor_rejects_active_install_override(tmp_path: Path):
    template = tmp_path
    (template / "pipeline").mkdir()
    (template / "bench_template.yaml").write_text("stage: iterate\n", encoding="utf-8")
    (template / "task.yaml").write_text("datasets: []\n", encoding="utf-8")
    (template / "reference.R").write_text(R_REFERENCE, encoding="utf-8")
    run_text = _reference_text_as_round0_pipeline(R_REFERENCE, "R")
    run_text += '\ninstall_override("FindAllMarkers", "Seurat", fast)\n'
    (template / "pipeline" / "run.R").write_text(run_text, encoding="utf-8")

    errors, _ = _template_check_lines(
        {"stage": "iterate"},
        {"id": "x", "tiers": []},
        template,
    )

    assert any("active install_override" in e for e in errors)


def test_experiment_doc_records_purpose_and_dispatch(tmp_path: Path):
    _write_experiment_doc(
        tmp_path,
        suite={
            "id": "iterate_core",
            "stage": "iterate",
            "field": "Bio",
            "prompt_slot": "iterate",
        },
        selected_tasks=[{"id": "findallmarker"}],
        prompt_id="zyme_iterate_v1",
        prompt_card={"name": "bio_iterate_v1", "hypothesis": "baseline"},
        reps=1,
        created_at="2026-05-11T03:16:19",
        purpose="Compare Cursor and Codex resume-loop behavior on findallmarker.",
    )
    _append_experiment_dispatch_doc(
        tmp_path,
        agent="cursor",
        model="composer-2",
        effort="max",
        prompt_rel="prompts/2_iterate.md",
        tasks=[{"name": "findallmarker_r1"}],
        max_rounds=30,
        force_mode=True,
        reflect=True,
        reflect_prompt="prompts/5_reflect.md",
        reflection_root=tmp_path / "reflections",
        reflect_category="iteration",
        detach=True,
    )

    text = (tmp_path / "EXPERIMENT.md").read_text(encoding="utf-8")
    assert "Purpose: Compare Cursor and Codex resume-loop behavior" in text
    assert "- Tasks: `findallmarker`" in text
    assert "## Dispatch" in text
    assert "- Agent: `cursor`" in text
    assert "- Max rounds: 30" in text
    assert "- Reflect: True" in text
    assert "- Reflect prompt: `prompts/5_reflect.md`" in text
