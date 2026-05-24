#!/usr/bin/env python3
"""Run every retained analysis in the locked unified UGA benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent

SCRIPTS = [
    "run_locked_hrd_cross_validation.py",
    "run_locked_kucab_low_burden_benchmark.py",
    "run_locked_mc3_luad_kmt2c_validation.py",
    "run_locked_mc3_clinical_endpoints.py",
    "run_locked_pcawg_signature_attribution.py",
    "run_locked_tcga_signature_endpoint_prediction.py",
]


def main() -> None:
    start = time.time()
    rows = []
    for script in SCRIPTS:
        script_path = SCRIPT_DIR / script
        t0 = time.time()
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(script_path)], cwd=str(EXPERIMENT_ROOT), check=True)
        rows.append({"script": script, "elapsed_seconds": round(time.time() - t0, 3)})
    manifest = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - start, 3),
        "scripts": rows,
        "locked_sbsdbs_model": "master_spec_sbs_dbs_d10_dp5",
        "locked_id_model": "id83_payload_only_d10_dp5",
        "modality_policy": (
            "Use only mutation modalities directly supported by retained source data. "
            "Kucab uses explicit SBS96+DBS78+ID83. TCGA-BRCA HRD and MC3 use SBS96+ID83; "
            "adjacent-SNV DBS reconstruction is not used."
        ),
    }
    (EXPERIMENT_ROOT / "data" / "run_all_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
