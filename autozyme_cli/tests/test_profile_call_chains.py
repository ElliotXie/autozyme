"""Unit tests for lightweight dynamic call-chain extraction."""
from __future__ import annotations

from zyme.commands.profile import parsers


def test_rprof_call_chain_uses_representative_root_to_leaf_stack(tmp_path):
    prof = tmp_path / "Rprof.out"
    prof.write_text(
        "sample.interval=5000\n"
        '"cpp_kernel" "process_bead_doublet" "fitPixels" "run.RCTD" "with_profile"\n'
        '"cpp_kernel" "process_bead_doublet" "fitPixels" "run.RCTD" "with_profile"\n'
        '"other_leaf" "helper" "run.RCTD" "with_profile"\n'
    )
    hotspots = [
        {
            "rank": 1,
            "label": '"cpp_kernel"',
            "self_time_s": 0.2,
            "total_time_s": 0.2,
            "self_pct": 80.0,
            "calls": None,
            "raw": {},
        }
    ]

    chains = parsers._rprof_call_chains(prof, hotspots)

    assert chains[0]["leaf"] == "cpp_kernel"
    assert chains[0]["chain"] == [
        "with_profile",
        "run.RCTD",
        "fitPixels",
        "process_bead_doublet",
        "cpp_kernel",
    ]
    assert chains[0]["samples"] == 2

