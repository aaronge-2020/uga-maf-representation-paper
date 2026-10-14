# Task 4 inputs

Copied from `~/workspace/task4_assets/` (outside the repo) on 2026-10-14.

| File | Contents |
|---|---|
| `burden.csv` | Burden features (defines the matched-feature patient set) |
| `cdr.xlsx` | TCGA clinical data resource (centre/plate annotations) |

Note: `task4_centre_heldout.py` also reads the 182-col SBS96+ID83 bundle (`feats_std.csv.gz`)
and the 948-col MAF stack (`results/tables/quick_bio_v4_maf_features.csv.gz`); those come
from the HF dataset bundle via `scripts/fetch_datasets.py`.

Used by: `scripts/task4_centre_heldout.py`.
