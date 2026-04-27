from __future__ import annotations

"""Compatibility entry point for the downstream detector workflow.

The implementation stays in `scripts/run_detection_baseline.py` so existing
commands keep working, while this namespaced wrapper gives new users an obvious
place to look for downstream-evaluation commands.
"""

import runpy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_detection_baseline.py"


if __name__ == "__main__":
    runpy.run_path(SCRIPT.as_posix(), run_name="__main__")
