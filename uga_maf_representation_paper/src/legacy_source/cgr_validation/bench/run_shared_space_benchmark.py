#!/usr/bin/env python3
"""Compatibility wrapper for EXP-024 shared-space benchmark.

The experiment implementation lives under the research-script layout:
`cgr_validation_results/research/scripts/EXP024_shared_space/`.
"""

from __future__ import annotations

import runpy
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "cgr_validation_results" / "research" / "scripts" / "EXP024_shared_space" / "run_shared_space_benchmark.py"


if __name__ == "__main__":
    runpy.run_path(str(RUNNER), run_name="__main__")
