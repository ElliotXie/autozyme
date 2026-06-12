"""`zyme report` — render a self-contained HTML optimization report.

Reads `results.tsv` + `memory/*.md` from the task directory and writes
`report.html`. Optionally consumes `memory/report_narrative.md` for
agent-curated narrative slots (hero tagline, tier descriptions, target
signature, etc.). Falls back to mechanical defaults when narrative absent.

The rendering itself lives in `zyme.report` so it can be invoked
programmatically too.
"""
import webbrowser
from pathlib import Path

from zyme.utils import task_dir_from_args, die
from zyme.report import render_report


def cmd_report(args):
    task_dir = task_dir_from_args(args)

    results_tsv = task_dir / "results.tsv"
    if not results_tsv.exists():
        die(f"no results.tsv in {task_dir} — run `zyme run` at least once first")

    output_path = Path(args.output) if args.output else None
    narrative_path = Path(args.narrative) if args.narrative else None

    out = render_report(
        task_dir,
        output_path=output_path,
        narrative_path=narrative_path,
    )

    # Tell the user where it landed + a one-line summary of the inputs consumed.
    narrative_used = (narrative_path or (task_dir / "memory" / "report_narrative.md")).exists()
    narrative_note = "with narrative" if narrative_used else "mechanical only (no narrative file)"
    print(f"wrote {out}  ({narrative_note})")

    if args.open:
        try:
            webbrowser.open(out.as_uri())
        except Exception as e:
            print(f"  (could not open browser: {e})")
