
import sys
import pandas as pd
from pathlib import Path

# Add the script directory to sys.path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from exp23_workflow import build_config
from exp23_modeling import load_exposure_frames, intersect_cohort_with_exposures

try:
    cfg = build_config()
    print("Loading base cohort...")
    base_cohort = pd.read_csv(cfg.cohort_dir / "base_analysis_cohort.tsv", sep="\t")
    print(f"Base cohort size: {len(base_cohort)}")
    
    print("Loading exposures...")
    exposures = load_exposure_frames(cfg)
    for name, df in exposures.items():
        print(f"Exposure {name} size: {len(df)}")
        
    print("Intersecting...")
    final_cohort = intersect_cohort_with_exposures(base_cohort, exposures)
    print(f"Final cohort size: {len(final_cohort)}")
    
    if len(final_cohort) == 0:
        print("CRITICAL: Final cohort is empty. Check patient ID overlap.")
        
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
