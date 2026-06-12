"""Synthetic idle-only workload for native idle-frame filtering."""
from __future__ import annotations

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_HELPERS_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "zyme"))
if _HELPERS_DIR not in sys.path:
    sys.path.insert(0, _HELPERS_DIR)

from helpers import emit_summary, peak_memory_mb, with_profile  # noqa: E402


def _load_config() -> dict:
    data_path = os.environ.get("ZYME_DATA_PATH") or os.path.join(
        os.path.dirname(_HERE), "data", "tiny.json",
    )
    with open(data_path) as fh:
        return json.load(fh)


def main() -> None:
    sleep_s = float(_load_config().get("sleep_s", 0.8))
    t0 = time.perf_counter()
    with with_profile():
        time.sleep(sleep_s)
    elapsed = time.perf_counter() - t0

    print(f"[idle_sleep] slept={sleep_s:.3f}s")
    emit_summary(speed_sec=elapsed, peak_mb=peak_memory_mb())


if __name__ == "__main__":
    main()
