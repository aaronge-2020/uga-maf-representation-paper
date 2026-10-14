#!/usr/bin/env python3
"""Post-fix for the Task 6 clean-checkout rerun (run 2, 2026-10-13).

Background: scripts/reproduce_table1.py stage 4 (stage_prerun_archive) extracts
the "manuscript reference" rows from whatever
main_manuscript_complete_panel_endpoint_results.csv exists at archive time.
On the FIRST run that file held the true manuscript outputs (8 Table-1 rows).
On the RERUN it held run 1's PARTIAL regenerated outputs (only the 2
burden_only slots) -- so the new archive's table1_manuscript_reference.csv
contains 2 rows of REGENERATED values mislabeled as manuscript values, and the
entry point's own task6_table1_comparison.csv compares regenerated-vs-
regenerated for those 2 slots and manuscript=NaN for the other 6.

This script corrects that, transparently:
  1. Loads the TRUE manuscript reference from the first run's archive
     (results/tables/task6_prerun_archive/20261013T120157Z/table1_manuscript_reference.csv,
     8 rows, archived before any regeneration).
  2. Copies it over the second run's archive reference (documented below).
  3. Rebuilds results/tables/task6_table1_comparison.csv with the EXACT same
     schema and tolerance logic as scripts/reproduce_table1.py::compare_table1.
  4. Patches the "comparison" section of results/logs/task6_table1_run_report.json.

It changes no regenerated numbers; it only fixes which column is the
manuscript baseline.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE1_SLOTS = [
    ("cancer_type_top20", "burden_only", "linear"),
    ("cancer_type_top20", "burden_only", "xgboost"),
    ("cancer_type_top20", "standard_sbs96_id83", "linear"),
    ("cancer_type_top20", "standard_sbs96_id83", "xgboost"),
    ("cancer_type_top20", "MAF_stack_only", "linear"),
    ("cancer_type_top20", "MAF_stack_only", "xgboost"),
    ("cancer_type_top20", "signatures_plus_MAF_stack", "linear"),
    ("cancer_type_top20", "signatures_plus_MAF_stack", "xgboost"),
]
TABLE1_METRIC = "balanced_accuracy"
ENV_TOLERANCE = 0.001167

FIRST_ARCHIVE = REPO_ROOT / "results/tables/task6_prerun_archive/20261013T120157Z"
SECOND_ARCHIVE = REPO_ROOT / "results/tables/task6_prerun_archive/20261013T133518Z"


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    true_ref_path = FIRST_ARCHIVE / "table1_manuscript_reference.csv"
    bad_ref_path = SECOND_ARCHIVE / "table1_manuscript_reference.csv"
    true_ref = load_rows(true_ref_path)
    bad_ref = load_rows(bad_ref_path)
    assert len(true_ref) == 8, f"expected 8 true reference rows, got {len(true_ref)}"
    print(f"true manuscript reference: {len(true_ref)} rows from {true_ref_path}")
    print(f"second-run (mislabelled) reference: {len(bad_ref)} rows from {bad_ref_path}")
    for r in bad_ref:
        print(f"  mislabelled row: {r['representation']}/{r['learner']} "
              f"balanced_accuracy={r['balanced_accuracy']} (regenerated value)")

    # 1-2. Replace the mislabelled reference with the true one.
    backup = bad_ref_path.with_suffix(".csv.rerun_mislabelled_backup")
    shutil.copy2(bad_ref_path, backup)
    shutil.copy2(true_ref_path, bad_ref_path)
    print(f"replaced {bad_ref_path.name}; original mislabelled copy kept at {backup.name}")

    # 3. Rebuild the comparison with the true reference (same logic as
    #    scripts/reproduce_table1.py::compare_table1).
    tables_dir = REPO_ROOT / "results" / "tables"
    regen_rows = load_rows(tables_dir / "task6_table1_regenerated.csv")
    order = {slot: i for i, slot in enumerate(TABLE1_SLOTS)}
    regen_rows.sort(key=lambda r: order[(r.get("endpoint"), r.get("representation"), r.get("learner"))])
    assert len(regen_rows) == 8, f"expected 8 regenerated rows, got {len(regen_rows)}"
    ref = {(r.get("endpoint"), r.get("representation"), r.get("learner")): r for r in true_ref}
    comparison = []
    for row in regen_rows:
        slot = (row.get("endpoint"), row.get("representation"), row.get("learner"))
        man = ref.get(slot, {})
        try:
            man_score = float(man.get(TABLE1_METRIC, "nan"))
        except (TypeError, ValueError):
            man_score = float("nan")
        try:
            new_score = float(row.get(TABLE1_METRIC, "nan"))
        except (TypeError, ValueError):
            new_score = float("nan")
        delta = new_score - man_score
        comparison.append({
            "endpoint": slot[0], "representation": slot[1], "learner": slot[2],
            "metric": TABLE1_METRIC,
            "manuscript_balanced_accuracy": man_score,
            "regenerated_balanced_accuracy": new_score,
            "delta": delta,
            "abs_delta": abs(delta),
            "within_tolerance": abs(delta) <= ENV_TOLERANCE,
            "manuscript_runtime_seconds": man.get("runtime_seconds", ""),
            "regenerated_runtime_seconds": row.get("runtime_seconds", ""),
        })
    out = tables_dir / "task6_table1_comparison.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)
    print(f"rebuilt {out}")
    for c in comparison:
        print(f"  {c['representation']}/{c['learner']}: man={c['manuscript_balanced_accuracy']:.6f} "
              f"regen={c['regenerated_balanced_accuracy']:.6f} delta={c['delta']:+.6f} "
              f"within_tol={c['within_tolerance']}")

    # 4. Patch the run report's comparison section.
    report_path = REPO_ROOT / "results" / "logs" / "task6_table1_run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    max_abs = max((c["abs_delta"] for c in comparison if c["abs_delta"] == c["abs_delta"]), default=float("nan"))
    n_within = sum(1 for c in comparison if c["within_tolerance"])
    report["comparison"] = {
        "max_abs_delta": max_abs,
        "tolerance": ENV_TOLERANCE,
        "slots_within_tolerance": n_within,
        "slots_total": len(comparison),
        "match": bool(n_within == len(comparison) == len(TABLE1_SLOTS)),
        "rows": comparison,
        "postfix_note": (
            "The entry point's own comparison (written before this fix) used a "
            "mislabelled manuscript reference on the rerun (2 rows of regenerated "
            "values). Rebuilt here against the true manuscript reference from "
            "task6_prerun_archive/20261013T120157Z/table1_manuscript_reference.csv. "
            "No regenerated values were altered."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"patched {report_path} comparison section: "
          f"{n_within}/{len(comparison)} within tolerance, max|delta|={max_abs:.6f}")


if __name__ == "__main__":
    main()
