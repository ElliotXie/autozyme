"""Synthetic multiprocessing fixture for profile override aggregation."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_HELPERS_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "zyme"))
if _HELPERS_DIR not in sys.path:
    sys.path.insert(0, _HELPERS_DIR)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from helpers import (  # noqa: E402
    emit_summary,
    install_override,
    peak_memory_mb,
    with_profile,
    with_subprofile,
)
import helpers  # noqa: E402
import worker_target  # noqa: E402


_orig_worker_payload = worker_target.worker_payload


def timed_worker_payload(n: int) -> int:
    return _orig_worker_payload(n)


def _pool_worker(n: int) -> int:
    value = worker_target.worker_payload(n)
    # multiprocessing workers do not always run normal interpreter atexit
    # hooks on every platform/start-method. Emit a cumulative snapshot from
    # the worker explicitly; the parser dedupes by (pid, name).
    helpers._emit_override_summaries()
    return value


def _load_config() -> dict:
    data_path = os.environ.get("ZYME_DATA_PATH") or os.path.join(
        os.path.dirname(_HERE), "data", "tiny.json",
    )
    with open(data_path) as fh:
        return json.load(fh)


def main() -> None:
    cfg = _load_config()
    n_workers = int(cfg.get("n_workers", 4))
    iters = int(cfg.get("iters", 8_000_000))

    install_override("worker_payload", "worker_target", timed_worker_payload)

    t0 = time.perf_counter()
    with with_profile():
        with with_subprofile("multiprocessing_pool"):
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
                results = pool.map(_pool_worker, [iters] * n_workers)
    elapsed = time.perf_counter() - t0

    print(f"[multiprocess_override] workers={n_workers} checksum={sum(results)}")
    emit_summary(speed_sec=elapsed, peak_mb=peak_memory_mb())


if __name__ == "__main__":
    main()
