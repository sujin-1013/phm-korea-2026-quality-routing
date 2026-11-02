"""Compatibility wrapper for the final capacity-ladder experiment.

The maintained entrypoint is now ``experiments/quality_routing/run.py``.
This file remains only because older notes reference the original path.
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "quality_routing" / "run.py"),
        run_name="__main__",
    )
