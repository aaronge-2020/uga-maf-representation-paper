#!/usr/bin/env python3
"""Generate Task 6 Table-1 comparison artifacts for the PARTIAL clean-checkout run.

Status 2026-10-13 ~22:40 UTC: 3 of 8 slots genuinely regenerated; 1 blocked
(standard_sbs96_id83/linear, skipped_blocked_slot); 4 pending (MAF slots).

Produces:
  results/tables/task6_table1_regenerated.csv  (completed + skipped rows)
  results/tables/task6_table1_comparison.csv   (vs true manuscript reference)

Pending slots are listed with empty regenerated values and status=pending.
The true manuscript reference comes from the first run's prerun archive
(task6_prerun_archive/20261013T120157Z/table1_manuscript_reference.csv),
NOT from any rerun archive (see task6_notes.md "Reference-extraction caveat").
"""

from __future__ import annotations

import csv
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
TRUE_REF = REPO_ROOT / "results/tables/task6_prerun_archive/20261013T120157Z/table1_manuscript_reference.csv"


def main() -> None:
    tables_dir = REPO_ROOT / "results" / "tables"
    # Current regenerated rows (whatever the pipeline has genuinely produced).
    regen: dict[tuple[str, str, str], dict] = {}
    ep_path = tables_dir / "main_manuscript_complete_panel_endpoint_results.csv"
    with ep_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            slot = (row.get("endpoint"), row.get("representation"), row.get("learner"))
            if slot in TABLE1_SLOTS:
                regen[slot] = row
    ref: dict[tuple[str, str, str], dict] = {}
    with TRUE_REF.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ref[(row.get("endpoint"), row.get("representation"), row.get("learner"))] = row
    assert len(ref) == 8, f"true reference must have 8 rows, got {len(ref)}"

    # task6_table1_regenerated.csv: completed rows as-is; skipped/pending marked.
    regen_rows = []
    for slot in TABLE1_SLOTS:
        if slot in regen:
            regen_rows.append(regen[slot])
        else:
            regen_rows.append({
                "endpoint": slot[0], "representation": slot[1], "learner": slot[2],
                "status": "pending_not_regenerated",
                "balanced_accuracy": "", "accuracy": "", "runtime_seconds": "",
            })
    regen_fieldnames = ["endpoint", "representation", "learner", "status",
                        "balanced_accuracy", "accuracy", "runtime_seconds"]
    out_regen = tables_dir / "task6_table1_regenerated.csv"
    with out_regen.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=regen_fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(regen_rows)
    print(f"wrote {out_regen} ({len(regen_rows)} rows)")

    # task6_table1_comparison.csv
    comparison = []
    for slot in TABLE1_SLOTS:
        man = ref[slot]
        man_score = float(man[TABLE1_METRIC])
        row = regen.get(slot)
        raw_status = (row or {}).get("status", "pending_not_regenerated")
        has_score = bool(row) and str(row.get(TABLE1_METRIC) or "").strip() != ""
        # Empty score with a "measured" row can only come from the documented
        # --skip wrapper (the runner stamps status="measured" after the call).
        if has_score:
            status = "measured"
        elif raw_status == "measured":
            status = "skipped_blocked_slot"
        else:
            status = raw_status
        if has_score:
            new_score = float(row[TABLE1_METRIC])
            delta = new_score - man_score
            comparison.append({
                "endpoint": slot[0], "representation": slot[1], "learner": slot[2],
                "metric": TABLE1_METRIC, "status": "measured",
                "manuscript_balanced_accuracy": man_score,
                "regenerated_balanced_accuracy": new_score,
                "delta": delta, "abs_delta": abs(delta),
                "within_tolerance": abs(delta) <= ENV_TOLERANCE,
                "manuscript_runtime_seconds": man.get("runtime_seconds", ""),
                "regenerated_runtime_seconds": row.get("runtime_seconds", ""),
            })
        else:
            comparison.append({
                "endpoint": slot[0], "representation": slot[1], "learner": slot[2],
                "metric": TABLE1_METRIC, "status": status,
                "manuscript_balanced_accuracy": man_score,
                "regenerated_balanced_accuracy": "",
                "delta": "", "abs_delta": "",
                "within_tolerance": "",
                "manuscript_runtime_seconds": man.get("runtime_seconds", ""),
                "regenerated_runtime_seconds": "",
            })
    out_comp = tables_dir / "task6_table1_comparison.csv"
    with out_comp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)
    print(f"wrote {out_comp}")
    for c in comparison:
        if c["status"] == "measured" and c["regenerated_balanced_accuracy"] != "":
            print(f"  {c['representation']}/{c['learner']}: man={c['manuscript_balanced_accuracy']:.6f} "
                  f"regen={c['regenerated_balanced_accuracy']:.6f} delta={c['delta']:+.6f} "
                  f"within_tol={c['within_tolerance']}")
        else:
            print(f"  {c['representation']}/{c['learner']}: {c['status']} "
                  f"(manuscript={c['manuscript_balanced_accuracy']:.6f})")


if __name__ == "__main__":
    main()
