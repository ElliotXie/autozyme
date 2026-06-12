from pathlib import Path

from zyme.prompts import sync_prompts_to_tasks as sync


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_find_tasks_accepts_root_and_categorized_layouts(tmp_path):
    workspace = tmp_path / "workspace"

    write(
        workspace / "test_bio" / "prompts" / "1_init.md",
        "You are a genius computational biologist.\n",
    )
    write(
        workspace / "test_general_bio" / "test_other" / "prompts" / "1_init.md",
        "# autozyme - init prompt (OtherField)\n\n"
        "You are an expert performance engineer.\n",
    )
    write(
        workspace / "test_not_a_task" / "README.md",
        "starts with test_ but has no prompts dir\n",
    )

    found = {task.name: flavor for task, flavor in sync.find_tasks(workspace)}

    assert found == {
        "test_bio": "Bio",
        "test_other": "OtherField",
    }


def test_find_workspace_root_accepts_release_checkout_name(tmp_path):
    prompts = tmp_path / "autozyme_release" / "autozyme_cli" / "zyme" / "prompts"
    prompts.mkdir(parents=True)

    assert sync._find_workspace_root(prompts) == tmp_path


def test_sync_preserves_legacy_init_inputs_and_skips_framework_readme(tmp_path, monkeypatch):
    framework = tmp_path / "framework_prompts"
    task = tmp_path / "workspace" / "test_bio"

    write(
        framework / "Bio" / "1_init.md",
        "# current init\n\n## Role\n\nCurrent role text.\n\n## Scope\n\nCurrent scope.\n",
    )
    write(framework / "Bio" / "2_iterate.md", "current iterate\n")
    write(framework / "Bio" / "README.md", "framework-only documentation\n")

    write(
        task / "prompts" / "1_init.md",
        "# old init\n\n## Role\n\nOld role text.\n\n"
        "## Your inputs for this run\n\n"
        "- **task_dir:** `D:\\autosearch\\test_bio`\n"
        "- **target_function:** `demo`\n\n"
        "## Scope\n\nOld scope.\n",
    )
    write(task / "prompts" / "2_iterate.md", "old iterate\n")

    monkeypatch.setattr(sync, "FRAMEWORK_PROMPTS", framework)

    report = sync.sync_task(task, "Bio", dry_run=False)

    assert report["updated"] == ["1_init.md", "2_iterate.md"]
    assert report["added"] == []
    assert not (task / "prompts" / "README.md").exists()

    init_text = (task / "prompts" / "1_init.md").read_text(encoding="utf-8")
    assert "Current role text." in init_text
    assert "Current scope." in init_text
    assert "## Your inputs for this run" in init_text
    assert "D:\\autosearch\\test_bio" in init_text
    assert "Old scope." not in init_text
