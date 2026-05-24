
import sys
from pathlib import Path

# Add the script directory to sys.path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from exp23_workflow import build_config, run_analysis

try:
    cfg = build_config()
    print(f"Config built. Assets: {cfg.assets_dir}")
    run_analysis(cfg)
    print("Analysis finished successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
