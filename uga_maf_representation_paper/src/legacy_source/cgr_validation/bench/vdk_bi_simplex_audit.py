import os
import time
import numpy as np
import pysam
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parents[1]
FASTA_PATH = BASE_DIR / "data/GRCH37/GCF_000001405.13/GCF_000001405.13_GRCh37_genomic.fna"
OUTPUT_PLOT = BASE_DIR / "cgr_validation_results/research/reports/cross_cutting/SNR_WALL_AUDIT.png"

# --- PAPER MATH ---
def get_paper_moments(seq, j_order=3):
    N = len(seq)
    moments_all = []
    for char in "ACGT":
        u = np.array([1.0 if b == char.upper() else 0.0 for b in seq])
        Nc = np.sum(u)
        if Nc == 0 or Nc == N: 
            moments_all.extend([0.0]*j_order)
            continue
        PS = np.abs(np.fft.rfft(u))**2
        PS_t = PS[1:] # Discard DC
        base_norm = Nc * (N - Nc)
        for j in range(1, j_order + 1):
            mj = np.sum(PS_t**j) / (base_norm**(j-1))
            moments_all.append(mj)
    return np.array(moments_all)

# --- SWEEP RUN ---
def run_snr_sweep():
    print("Starting SNR Scale-Up Audit (Window size N vs Accuracy)...")
    fasta = pysam.FastaFile(FASTA_PATH)
    
    # Anchors for consistent biological classes
    anchors = {
        "exon": 13112,       # OR4F5 region
        "repeat": 10001,     # Telomeric repeat
        "junk": 20000000     # Intergenic noise
    }
    
    n_values = [200, 500, 1000, 3000, 5000] # Scanning towards the paper's N values
    results = []
    
    for n in n_values:
        print(f"Testing N={n}...")
        dataset, labels = [], []
        # Sample 200 sites per class centered on the anchors
        for i, (cls, anchor) in enumerate(anchors.items()):
            for offset in range(0, 200 * 100, 100):
                try:
                    seq = fasta.fetch("NC_000001.11", anchor + offset, anchor + offset + n)
                    if len(seq) == n and "N" not in seq.upper():
                        dataset.append(seq)
                        labels.append(i)
                except: pass
        
        # Train & Evaluate (Random Forest 100nd estimators)
        feats = [get_paper_moments(s) for s in dataset]
        acc = np.mean(cross_val_score(RandomForestClassifier(n_estimators=50), feats, labels, cv=3))
        results.append(acc)
        print(f"  Accuracy at N={n}: {acc:.4f}")

    # REPORT
    print("\nSNR SWEEP SUMMARY:")
    for n, acc in zip(n_values, results):
        print(f"Window N={n} -> Accuracy: {acc:.4f}")
    
    # PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, results, marker='o', linewidth=2, color='darkgreen')
    plt.xlabel("Window Size (N) - Number of Bases")
    plt.ylabel("Classification Accuracy")
    plt.title("The 'SNR Wall': Why the Paper needs thousands of bases to see the heartbeat.")
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PLOT)
    print(f"Scale-up report saved to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    run_snr_sweep()
