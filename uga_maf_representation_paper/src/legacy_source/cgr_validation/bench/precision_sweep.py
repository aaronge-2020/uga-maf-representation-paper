import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_REPORT = BASE_DIR / "cgr_validation_results/research/reports/cross_cutting/PRECISION_FLOOR.md"

# --- BI-USM ENGINE ---
BASE_MAP = {"A": (0.0, 1.0), "C": (1.0, 1.0), "G": (0.0, 0.0), "T": (1.0, 0.0)}
CORNER = {(0, 1): "A", (1, 1): "C", (0, 0): "G", (1, 0): "T"}

def cgr_encode(seq, dtype=np.float64, weight=0.5):
    x, y = dtype(0.5), dtype(0.5)
    for b in seq:
        cx, cy = BASE_MAP[b]
        x = (dtype(1.0) - dtype(weight)) * x + dtype(weight) * dtype(cx)
        y = (dtype(1.0) - dtype(weight)) * y + dtype(weight) * dtype(cy)
    return x, y

def cgr_decode(x, y, k, dtype=np.float64):
    out = []
    xx, yy = dtype(x), dtype(y)
    eps = dtype(1e-15) if dtype == np.float64 else dtype(1e-7)
    for _ in range(int(k)):
        ix = 1 if (xx + eps) >= 0.5 else 0
        iy = 1 if (yy + eps) >= 0.5 else 0
        b = CORNER.get((ix, iy), "N")
        out.append(b)
        xx = dtype(2.0) * xx - dtype(ix)
        yy = dtype(2.0) * yy - dtype(iy)
    return "".join(reversed(out))

# --- BENCHMARK SWEEP ---
def run_precision_sweep():
    print("Initializing Precision Floor Sweep (k=1 to k=100)...")
    
    results = []
    
    for k in range(1, 101):
        print(f"Testing k={k} context length...", end="\r")
        
        # 1. Generate 1,000 random k-mers
        bases = "ACGT"
        n_samples = 1000
        
        acc_32 = 0
        acc_64 = 0
        
        for _ in range(n_samples):
            kmer = "".join(np.random.choice(list(bases), k))
            
            # TEST A: 32-bit Float
            x32, y32 = cgr_encode(kmer, dtype=np.float32)
            recon32 = cgr_decode(x32, y32, k, dtype=np.float32)
            if recon32 == kmer: acc_32 += 1
            
            # TEST B: 64-bit Double
            x64, y64 = cgr_encode(kmer, dtype=np.float64)
            recon64 = cgr_decode(x64, y64, k, dtype=np.float64)
            if recon64 == kmer: acc_64 += 1
            
        results.append({
            "k": k,
            "acc_32": acc_32 / n_samples,
            "acc_64": acc_64 / n_samples
        })

    df = pd.DataFrame(results)
    
    # 2. IDENTIFY CLIFFS
    cliff_32 = df[df['acc_32'] < 1.0]['k'].min()
    cliff_64 = df[df['acc_64'] < 1.0]['k'].min()
    
    # 3. REPORTING
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# The Precision Floor: The Hard Boundary of DNA Resolution\n\n")
        f.write("## Objective\n")
        f.write("Quantify the 'Resolution Cliff' of the Bidirectional USM coordinate representation across different floating-point precisions.\n\n")
        
        f.write("## The Resolution Cliff (Hard Technical Boundaries)\n\n")
        k_lossless = int(cliff_32) - 1 if pd.notna(cliff_32) else 23
        f.write(f"- **32-bit Float Cliff (FP32)**: Accuracy drops below 100% at **k = {cliff_32}**. ")
        f.write(f"Largest **single-walk** lossless length is **{k_lossless}** bases per (x,y). ")
        f.write(f"`vdkm/render.py` defaults **`--k` = 45** (22+ref+22): two **{k_lossless}-base** bidirectional walks meeting at the variant.\n")
        f.write(f"- **64-bit Double Cliff (FP64)**: Accuracy drops below 100% at **k = {cliff_64}**. This is the mathematical limit of standard scientific CPU computing.\n\n")
        
        f.write("## Accuracy Sweep: Table of Loss\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Conclusion: FP32 lossless limit and pipeline default\n")
        f.write("> [!IMPORTANT]\n")
        f.write("This sweep is for **one** CGR walk of length k in float32. ")
        f.write(f"The tokenizer uses a **45-bp** window by default so **each** Bi-USM leg has **{k_lossless}** bases (within this per-walk limit). ")
        if pd.notna(cliff_32):
            f.write(f"One float32 walk of length ≥ {int(cliff_32)} loses exact identity.\n\n")
        else:
            f.write("See the table above for where FP32 accuracy falls below 100%.\n\n")

    print("\nPrecision Sweep Complete.")
    print(f"32-bit Cliff at k={cliff_32}")
    print(f"64-bit Cliff at k={cliff_64}")

if __name__ == "__main__":
    run_precision_sweep()
