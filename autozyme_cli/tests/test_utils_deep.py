"""Deep coverage tests for zyme.utils.

Targets gaps not covered by tests/test_utils.py:
  - die_internal / die_resource / info exit codes
  - get_prompt_id_for_task
  - git() wrapper (success + failure)
  - task_dir_from_args
  - resolve_reference_script explicit-mode + active-mode + fallback paths
  - resolve_reference_output_dir tier-positional path
  - _grep_dcf_field no-match
  - read_upstream_repo_metadata pyproject/setup.cfg variants
  - read_upstream_repo_sha (real git clone)
  - read_installed_version (real python interpreter, present + absent pkg)
  - check_upstream_version_drift end-to-end + cache hit
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from zyme import utils
from zyme.utils import (
    EXIT_INTERNAL,
    EXIT_RESOURCE,
    _grep_dcf_field,
    check_upstream_version_drift,
    die_internal,
    die_resource,
    get_prompt_id_for_task,
    git,
    info,
    read_installed_version,
    read_upstream_repo_metadata,
    read_upstream_repo_sha,
    resolve_reference_output_dir,
    resolve_reference_script,
    task_dir_from_args,
)


# --------------------------------------------------------------------------
# die variants + info
# --------------------------------------------------------------------------

class TestDieVariants:
    def test_die_internal_exit_code(self):
        with pytest.raises(SystemExit) as exc:
            die_internal("boom")
        assert exc.value.code == EXIT_INTERNAL

    def test_die_resource_exit_code(self):
        with pytest.raises(SystemExit) as exc:
            die_resource("out of ram")
        assert exc.value.code == EXIT_RESOURCE

    def test_info_prints(self, capsys):
        info("hello")
        assert "[zyme] hello" in capsys.readouterr().out


# --------------------------------------------------------------------------
# get_prompt_id_for_task
# --------------------------------------------------------------------------

class TestGetPromptId:
    def test_empty_when_absent(self, tmp_path: Path):
        assert get_prompt_id_for_task(tmp_path) == ""

    def test_reads_from_meta(self, tmp_path: Path):
        (tmp_path / ".zyme_meta.yaml").write_text("prompt_id: p_iterate_x\n")
        assert get_prompt_id_for_task(tmp_path) == "p_iterate_x"


# --------------------------------------------------------------------------
# git wrapper
# --------------------------------------------------------------------------

class TestGit:
    def test_success_returns_stdout(self, tmp_path: Path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        out = git("rev-parse", "--is-inside-work-tree", cwd=tmp_path)
        assert out == "true"

    def test_failure_dies(self, tmp_path: Path):
        # Not a git repo (and no parent repo) -> git command fails -> die().
        with pytest.raises(SystemExit):
            git("rev-parse", "HEAD", cwd=tmp_path, check=True)

    def test_no_capture_returns_empty_string(self, tmp_path: Path):
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        out = git("status", cwd=tmp_path, capture=False)
        assert out == ""


# --------------------------------------------------------------------------
# task_dir_from_args
# --------------------------------------------------------------------------

class _Args:
    def __init__(self, task_dir):
        self.task_dir = task_dir


class TestTaskDirFromArgs:
    def test_resolves_with_task_yaml(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("x: 1\n")
        out = task_dir_from_args(_Args(str(tmp_path)))
        assert out == tmp_path.resolve()

    def test_dies_without_task_yaml(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            task_dir_from_args(_Args(str(tmp_path)))


# --------------------------------------------------------------------------
# resolve_reference_script
# --------------------------------------------------------------------------

class TestResolveReferenceScript:
    def test_explicit_mode_resolves(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "modes:\n"
            "  parallel_t8: {description: p, reference_script: reference.parallel_t8.R}\n"
        )
        out = resolve_reference_script(tmp_path, mode="parallel_t8")
        assert out.name == "reference.parallel_t8.R"

    def test_explicit_mode_default_script_when_missing(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "modes:\n  m: {description: d}\n"  # no reference_script field
        )
        out = resolve_reference_script(tmp_path, mode="m")
        assert out.name == "reference.R"

    def test_falls_back_to_reference_py(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        (tmp_path / "reference.py").write_text("")
        out = resolve_reference_script(tmp_path)
        assert out.name == "reference.py"

    def test_placeholder_when_nothing_exists(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text("target_repo: foo\n")
        # No reference.R/py present -> returns reference.R placeholder.
        out = resolve_reference_script(tmp_path)
        assert out.name == "reference.R"


# --------------------------------------------------------------------------
# resolve_reference_output_dir (tier-positional path)
# --------------------------------------------------------------------------

class TestResolveReferenceOutputDir:
    def test_tier_only_default_layout(self, tmp_path: Path):
        out = resolve_reference_output_dir(tmp_path, "medium")
        assert out.parts[-2:] == ("reference_outputs", "medium")

    def test_tier_required_when_empty(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            resolve_reference_output_dir(tmp_path, "")

    def test_tier_kwarg_uses_active_mode(self, tmp_path: Path):
        (tmp_path / "task.yaml").write_text(
            "active_mode: m\n"
            "modes:\n  m: {description: d, reference_script: r.R, "
            "reference_output_dir: \"refs/<mode>/<tier>\"}\n"
        )
        out = resolve_reference_output_dir(tmp_path, tier="large")
        assert out.parts[-3:] == ("refs", "m", "large")


# --------------------------------------------------------------------------
# _grep_dcf_field
# --------------------------------------------------------------------------

class TestGrepDcfField:
    def test_finds_field(self):
        text = "Package: spacexr\nVersion: 2.2.1\n"
        assert _grep_dcf_field(text, "Version") == "2.2.1"

    def test_no_match_returns_empty(self):
        assert _grep_dcf_field("Package: x\n", "Version") == ""


# --------------------------------------------------------------------------
# read_upstream_repo_metadata variants
# --------------------------------------------------------------------------

class TestReadUpstreamRepoMetadata:
    def test_pyproject_project_block(self, tmp_path: Path):
        up = tmp_path / "upstream_repo"
        up.mkdir()
        (up / "pyproject.toml").write_text(
            "[build-system]\nrequires = []\n"
            "[project]\nname = \"foo\"\nversion = \"1.2.3\"\n"
            "[tool.x]\nname = \"ignored\"\n"
        )
        out = read_upstream_repo_metadata(tmp_path)
        assert out == {"package": "foo", "version": "1.2.3",
                       "source": "pyproject.toml"}

    def test_setup_cfg_metadata_block(self, tmp_path: Path):
        up = tmp_path / "upstream_repo"
        up.mkdir()
        (up / "setup.cfg").write_text(
            "[metadata]\nname = barpkg\nversion = 0.9.1\n"
            "[options]\nname = ignored\n"
        )
        out = read_upstream_repo_metadata(tmp_path)
        assert out["package"] == "barpkg"
        assert out["version"] == "0.9.1"
        assert out["source"] == "setup.cfg"

    def test_file_dynamic_version_skipped(self, tmp_path: Path):
        up = tmp_path / "upstream_repo"
        up.mkdir()
        (up / "setup.cfg").write_text(
            "[metadata]\nname = foo\nversion = file: VERSION\n"
        )
        assert read_upstream_repo_metadata(tmp_path) is None

    def test_upstream_is_file_not_dir(self, tmp_path: Path):
        (tmp_path / "upstream_repo").write_text("not a dir")
        assert read_upstream_repo_metadata(tmp_path) is None


# --------------------------------------------------------------------------
# read_upstream_repo_sha (real git clone)
# --------------------------------------------------------------------------

class TestReadUpstreamRepoSha:
    def test_reads_head_from_git_clone(self, tmp_path: Path):
        up = tmp_path / "upstream_repo"
        up.mkdir()
        env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        import os
        run_env = {**os.environ, **env}
        subprocess.run(["git", "init"], cwd=up, capture_output=True)
        (up / "f.txt").write_text("x")
        subprocess.run(["git", "add", "."], cwd=up, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=up,
                       capture_output=True, env=run_env)
        sha = read_upstream_repo_sha(tmp_path)
        assert sha is not None and len(sha) == 40


# --------------------------------------------------------------------------
# read_installed_version (real python interpreter)
# --------------------------------------------------------------------------

def _py_task(tmp_path: Path) -> Path:
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "run.py").write_text("# stub\n")
    (tmp_path / "task.yaml").write_text("target_repo: foo\n")
    return tmp_path


class TestReadInstalledVersion:
    def test_present_package(self, tmp_path: Path):
        _py_task(tmp_path)
        # `json` has no __version__, but importlib.metadata won't find it
        # either; use a package guaranteed installed with a version: pytest.
        v = read_installed_version(tmp_path, "pytest")
        assert v is not None and v[0].isdigit()

    def test_absent_package_returns_none(self, tmp_path: Path):
        _py_task(tmp_path)
        v = read_installed_version(tmp_path, "definitely_not_a_real_pkg_xyz")
        assert v is None


# --------------------------------------------------------------------------
# check_upstream_version_drift end-to-end + cache
# --------------------------------------------------------------------------

class TestCheckUpstreamVersionDrift:
    def test_none_when_no_upstream(self, tmp_path: Path):
        _py_task(tmp_path)
        assert check_upstream_version_drift(tmp_path) is None

    def test_none_when_install_missing(self, tmp_path: Path):
        _py_task(tmp_path)
        up = tmp_path / "upstream_repo"
        up.mkdir()
        (up / "pyproject.toml").write_text(
            "[project]\nname = \"definitely_not_a_real_pkg_xyz\"\nversion = \"9.9.9\"\n"
        )
        # Package not installed -> installed_version None -> overall None.
        assert check_upstream_version_drift(tmp_path) is None

    def test_agrees_when_versions_match(self, tmp_path: Path, monkeypatch):
        _py_task(tmp_path)
        up = tmp_path / "upstream_repo"
        up.mkdir()
        (up / "pyproject.toml").write_text(
            "[project]\nname = \"pytest\"\nversion = \"REPLACED\"\n"
        )
        # Pin both sides so the comparison is deterministic regardless of host.
        import importlib.metadata as im
        installed = im.version("pytest")
        (up / "pyproject.toml").write_text(
            f"[project]\nname = \"pytest\"\nversion = \"{installed}\"\n"
        )
        out = check_upstream_version_drift(tmp_path)
        assert out is not None
        assert out["package"] == "pytest"
        assert out["agrees"] is True
        assert out["installed_version"] == installed

    def test_cache_hit_returns_cached(self, tmp_path: Path, monkeypatch):
        _py_task(tmp_path)
        up = tmp_path / "upstream_repo"
        up.mkdir()
        # Make it a git repo so a SHA is available for the cache key.
        subprocess.run(["git", "init"], cwd=up, capture_output=True)
        (up / "pyproject.toml").write_text(
            "[project]\nname = \"foo\"\nversion = \"1.0.0\"\n"
        )
        import os
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run(["git", "add", "."], cwd=up, capture_output=True)
        subprocess.run(["git", "commit", "-m", "i"], cwd=up,
                       capture_output=True, env=env)
        sha = read_upstream_repo_sha(tmp_path)

        # Pre-seed the cache; check_upstream_version_drift must return it
        # without re-spawning the interpreter.
        cache = tmp_path / ".zyme" / "version_check.json"
        cache.parent.mkdir(exist_ok=True)
        cached = {
            "package": "foo", "upstream_version": "1.0.0",
            "installed_version": "1.0.0", "agrees": True,
            "source": "pyproject.toml", "upstream_sha": sha,
        }
        cache.write_text(json.dumps(cached))

        # If the cache is consulted, read_installed_version must NOT run.
        def _boom(*a, **k):
            raise AssertionError("interpreter should not be spawned on cache hit")
        monkeypatch.setattr(utils, "read_installed_version", _boom)

        out = check_upstream_version_drift(tmp_path)
        assert out is not None
        assert out["agrees"] is True
        assert out["package"] == "foo"
