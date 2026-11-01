#!/usr/bin/env python
"""--full mode: rerun all 49 grid cells through the existing runner.

Trial counts are read from the saved trial tables, so each cell reruns with
the same tuning budget the grid used. The runner skips cells whose
predictions already exist; delete a cell's predictions and partials first
to force a from-scratch refit of that cell. On the 2-core VM this grid took
about a week of wall time; treat this mode as the large-machine path.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repro_common as rc

NS = rc.NS


def main():
    cells = rc.list_cells()
    print(f"full rerun: {len(cells)} cells (runner skips cells with predictions)")
    for task, rep, learner in cells:
        cell = rc.cell_name(task, rep, learner)
        t = pd.read_csv(NS / "metrics" / "trials" / f"{cell}.csv", dtype={"trial": str})
        n_trials = int(t[t["trial"] != "FINAL"]["trial"].astype(int).max()) + 1
        cmd = [sys.executable, str(HERE / "runner.py"), "--task", task,
               "--representation", rep, "--learner", learner,
               "--trials", str(n_trials)]
        print("+", " ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"FAILED {cell} (exit {r.returncode})")
            return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
