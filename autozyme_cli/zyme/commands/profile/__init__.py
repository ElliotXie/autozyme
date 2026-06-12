"""zyme.commands.profile — `zyme profile` subcommand.

Sub-package layout:
    command.py   — cmd_profile orchestration (entry point)
    backends.py  — backend resolution + availability probing
    parsers.py   — backend-native artifact → normalized profile.json
    render.py    — evidence-card stderr formatting
    archive.py   — profile_history/ writer
"""
from zyme.commands.profile.command import cmd_profile

__all__ = ["cmd_profile"]
