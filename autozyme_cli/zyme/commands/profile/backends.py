"""Backend resolution: availability probing, fallback rules.

Four backends:
  cpu     — cProfile (Python) / Rprof (R). Stdlib, always available.
  full    — Scalene (Python) / profvis (R). Optional installs.
  mem     — memray (Python) / Rprof+memory (R). memray optional install
            on Python; R uses stdlib Rprof.
  native  — macOS sample(1) wrapping the pipeline process tree (cross-
            process via pgrep watchdog). Sees BLAS / Cython / Rcpp
            internals that all in-process profilers miss. macOS only;
            same backend choice for Python and R tasks.

Fallback rule: `full` falls back to `cpu` when the optional tool is
missing (with a one-line install hint). `mem` and `native` do NOT fall
back — substituting CPU profile when the user asked for memory or for
native attribution would be wrong-axis data; we refuse loudly instead.
"""
import subprocess
import sys
from pathlib import Path


def resolve(requested: str, lang: str, executor: dict | None = None) -> tuple[str, str | None]:
    """Resolve a requested backend → (effective_backend, warning_or_None).

    Args:
        requested: "cpu", "full", or "mem".
        lang: "py" or "R".
        executor: parsed task.yaml `executor` block. Used to find the
            interpreter to probe.

    Returns:
        (effective_backend, warning_message_or_None). When effective !=
        requested, warning explains the substitution + install hint.

    Raises:
        RuntimeError: when `mem` is requested but memray is unavailable
            (no fallback for mem — see module docstring).
    """
    if requested == "cpu":
        return "cpu", None

    if requested == "mem":
        if lang == "py" and not _probe_python_pkg("memray", executor):
            raise RuntimeError(
                "backend=mem requires memray. Install with `pip install memray` "
                "in the task's python env. Refusing to fall back to cpu — `mem` "
                "is a different signal axis (allocation tracking) and silent "
                "substitution would mislead."
            )
        return "mem", None

    if requested == "full":
        if lang == "py":
            if _probe_python_pkg("scalene", executor):
                return "full", None
            return "cpu", (
                "[profile] backend=full requested but `scalene` not available "
                "in the task's python env. Falling back to backend=cpu (cProfile). "
                "For the full Scalene profile (line-level CPU+mem+native split), "
                "install: pip install scalene"
            )
        else:  # R
            if _probe_r_pkg("profvis", executor):
                return "full", None
            return "cpu", (
                "[profile] backend=full requested but `profvis` not installed in R. "
                "Falling back to backend=cpu (Rprof). For the HTML profvis viewer, "
                "install: install.packages('profvis')  (the CLI parses Rprof.out "
                "either way; profvis only adds a browser-renderable view.)"
            )

    if requested == "native":
        from zyme.commands.profile import native
        ok, reason = native.is_supported()
        if not ok:
            raise RuntimeError(
                f"backend=native unavailable: {reason}. Refusing to fall back "
                "to cpu — native is a different signal axis (sees BLAS/Rcpp/"
                "Cython internals that cpu cannot)."
            )
        return "native", None

    raise ValueError(f"unknown backend: {requested!r} "
                     f"(expected cpu, full, mem, or native)")


def _probe_python_pkg(pkg: str, executor: dict | None) -> bool:
    """Return True if `pkg` is importable in the task's python interpreter."""
    py_bin = _resolve_python_bin(executor)
    if not py_bin:
        return False
    try:
        result = subprocess.run(
            [py_bin, "-c", f"import {pkg}"],
            capture_output=True, timeout=15,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _probe_r_pkg(pkg: str, executor: dict | None) -> bool:
    """Return True if R package `pkg` is installed."""
    rscript = (executor or {}).get("rscript") or "Rscript"
    try:
        result = subprocess.run(
            [rscript, "-e",
             f'if (!requireNamespace("{pkg}", quietly=TRUE)) quit(status=1)'],
            capture_output=True, timeout=20,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _resolve_python_bin(executor: dict | None) -> str | None:
    """Resolve task's python interpreter — uses runner._resolve_python.

    Returns the absolute path, or None if resolution fails (caller treats
    that as "package unavailable").
    """
    if executor and executor.get("python"):
        try:
            from zyme.runner import _resolve_python
            return _resolve_python(executor["python"])
        except (RuntimeError, ImportError):
            return None
    # No executor block: use the interpreter running zyme. This matches
    # runner.py and avoids probing a different PATH python from a venv.
    return sys.executable


def memray_pipeline_args(memray_outfile: Path) -> list[str]:
    """Build the args inserted between py_bin and `run.py` for memray mode.

    memray's CLI:
        <py_bin> -m memray run -o <out> -f --native run.py

    Notes:
      - `--native` is essential for autozyme: it captures C/C++ stack frames,
        which is what makes memray see NumPy / torch / h5py native buffers
        (the actual memory pain in scientific computing). Without it,
        memray sees only Python-level allocations, no better than tracemalloc.
      - `-f` overwrites a stale memray.bin from a prior run.
      - No separator before the target script — memray run takes the script
        path directly after its flags (unlike scalene's `---` convention).
    """
    return [
        "-m", "memray", "run",
        "-o", str(memray_outfile),
        "-f", "--native",
    ]


def scalene_pipeline_args(scalene_outfile: Path) -> list[str]:
    """Build the args inserted between py_bin and `run.py` for Scalene mode.

    Scalene 2.x CLI:
        <py_bin> -m scalene run -o <outfile> --memory \
                --profile-system-libraries --- run.py

    Notes:
      - `--profile-system-libraries` is essential for autozyme: hot paths
        often live inside numpy/scipy/etc. internals; skipping them (the
        default) loses the signal we most want to see.
      - The `---` separator marks "everything after is the target program".
      - Output is always JSON in 2.x; the `-o` flag selects the path.
    """
    return [
        "-m", "scalene", "run",
        "-o", str(scalene_outfile),
        "--memory",
        "--profile-system-libraries",
        "---",
    ]
