"""`zyme inspect-parallelism` — workspace-side wrapper around zyme.scan_parallelism.scan()."""

from pathlib import Path

from zyme.utils import die


def _auto_target(repo: Path) -> str | None:
    """Try to read target_function from task.yaml next to or above repo."""
    import yaml
    for candidate in [repo.parent / "task.yaml", repo.parent.parent / "task.yaml"]:
        if candidate.exists():
            try:
                cfg = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                tf = cfg.get("target_function", "")
                if tf:
                    return tf.split("::")[-1].split(".")[-1]
            except Exception:
                pass
    return None


def cmd_inspect_parallelism(args):
    """Scan an upstream repo and report every parallelism backend in use.

    Output: structured human-readable inventory + draft `parallelism_profile`
    YAML block for the init agent to review and paste into `task.yaml`.
    Read-only.
    """
    from zyme.scan_parallelism import scan, format_report, filter_knobs_by_target

    repo = Path(args.repo).resolve()
    if not repo.exists():
        die(f"upstream_repo not found: {repo}")
    if not repo.is_dir():
        die(f"path is not a directory: {repo}")

    inv = scan(repo)

    target = getattr(args, "target", None) or _auto_target(repo)
    if target:
        target_bare = target.split("::")[-1].split(".")[-1]
        filter_knobs_by_target(inv, repo, target_bare)

    print(format_report(inv, max_hits_per_backend=args.max_hits))
