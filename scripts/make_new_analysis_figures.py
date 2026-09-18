#!/usr/bin/env python3
"""Manuscript figures for the analyses added after the 1 August 2026 meeting.

`make_all_figures.py` builds the main panel from fixed table names and has no knowledge of the
survival baseline, the individual-feature selection, or the SHAP attribution. This script fills
that gap so those three analyses are presentable rather than CSV-only.

    python scripts/make_new_analysis_figures.py                 # every figure it can build
    python scripts/make_new_analysis_figures.py --only survival

Figures written to results/figures/:

    figure_S_survival_cancer_type_baseline.png
        Overall-survival C-index by arm, with the paired bootstrap difference against the
        cancer-type-only baseline. Answers whether the representation adds anything beyond
        knowing the tumour type.

    figure_S_feature_selection_sparsity.png
        Inner-CV score against the number of features retained, across the sparsity grid, with
        the selected setting marked. Shows that the chosen operating point is parsimonious
        without a meaningful loss of performance.

    figure_S_shap_by_block.png
        SHAP attribution aggregated to annotation blocks rather than individual features. This
        is the block-level view requested in the meeting checklist: it shows whether the model
        spreads its weight across interpretable annotation families or leans on one unstable
        feature.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "results/tables"
FIGURES = REPO_ROOT / "results/figures"

NAVY = "#1F3864"
BLUE = "#2E5496"
GREY = "#9CA3AF"
RED = "#9C0006"
GREEN = "#2E7D32"

ARM_LABELS = {
    "cancer_type": "Cancer type only",
    "bio_maf": "Bio MAF v4 only",
    "cancer_type_plus_bio_maf": "Cancer type + Bio MAF v4",
}

# Mutually exclusive annotation blocks, tested in order: the first pattern that matches a column
# assigns it. Order matters because driver per-gene columns also start with "driver_".
BLOCK_PATTERNS = [
    ("Per-gene driver counts", re.compile(r"^driver_gene_.*__[A-Z0-9\-]+$")),
    ("Driver evidence / tiers", re.compile(r"^driver_")),
    ("OncoKB role", re.compile(r"^oncokb_|^residual_non_oncokb")),
    ("Hotspots", re.compile(r"hotspot")),
    ("VEP impact", re.compile(r"^vep_impact")),
    ("VEP consequence", re.compile(r"^vep_consequence")),
    ("VEP other (SIFT, PolyPhen, biotype)", re.compile(r"^vep_")),
    ("Allele fraction (VAF)", re.compile(r"vaf")),
    ("Mutation type / burden", re.compile(r"^(id_|snv_|dnp_|mnp_|total_|burden|n_mut)")),
]


def assign_block(feature: str) -> str:
    for name, pattern in BLOCK_PATTERNS:
        if pattern.search(feature):
            return name
    return "Other"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)


def figure_survival(plt) -> Path | None:
    summary_path = TABLES / "survival_cancer_type_baseline_summary.csv"
    if not summary_path.exists():
        print("[survival] no summary table; skipping")
        return None
    summary = pd.read_csv(summary_path)
    comp_path = TABLES / "survival_cancer_type_baseline_comparisons.csv"
    comparisons = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()

    order = [arm for arm in ARM_LABELS if arm in set(summary["representation"].astype(str))]
    if not order:
        order = summary["representation"].astype(str).tolist()
    scores = [float(summary.loc[summary["representation"] == arm, "score"].iloc[0]) for arm in order]
    labels = [ARM_LABELS.get(arm, arm) for arm in order]

    has_comparisons = not comparisons.empty
    fig, axes = plt.subplots(1, 2 if has_comparisons else 1, figsize=(12 if has_comparisons else 7, 4.2))
    axes = np.atleast_1d(axes)
    ax = axes[0]

    colors = [RED if arm == "cancer_type" else BLUE for arm in order]
    bars = ax.barh(range(len(order)), scores, color=colors, height=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Harrell C-index (pooled out-of-fold)", fontsize=10)
    ax.set_xlim(0.5, max(0.8, max(scores) + 0.05))
    ax.axvline(0.5, color=GREY, linestyle=":", linewidth=1)
    for bar, score in zip(bars, scores):
        ax.text(score + 0.004, bar.get_y() + bar.get_height() / 2, f"{score:.3f}",
                va="center", fontsize=10, fontweight="bold")
    ax.set_title("Overall survival: does the representation\nbeat cancer type alone?", fontsize=11, loc="left")
    _style(ax)

    if has_comparisons:
        ax2 = axes[1]
        sub = comparisons[comparisons["reference"] == "cancer_type"]
        if sub.empty:
            sub = comparisons
        names = [f"{ARM_LABELS.get(r['candidate'], r['candidate'])}\nvs {ARM_LABELS.get(r['reference'], r['reference'])}"
                 for _, r in sub.iterrows()]
        deltas = sub["delta"].to_numpy(dtype=float)
        lows = deltas - sub["ci_low"].to_numpy(dtype=float)
        highs = sub["ci_high"].to_numpy(dtype=float) - deltas
        ypos = np.arange(len(sub))
        point_colors = [GREEN if row["ci_low"] > 0 else (RED if row["ci_high"] < 0 else GREY)
                        for _, row in sub.iterrows()]
        ax2.errorbar(deltas, ypos, xerr=[lows, highs], fmt="none", ecolor=GREY, capsize=4, linewidth=1.4)
        ax2.scatter(deltas, ypos, s=70, c=point_colors, zorder=3)
        ax2.axvline(0.0, color=NAVY, linestyle="--", linewidth=1.2)
        ax2.set_yticks(ypos)
        ax2.set_yticklabels(names, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel("Δ C-index (95% paired bootstrap CI)", fontsize=10)
        ax2.set_title("Incremental value over the baseline", fontsize=11, loc="left")
        _style(ax2)

    fig.tight_layout()
    out = FIGURES / "figure_S_survival_cancer_type_baseline.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[survival] -> {out.relative_to(REPO_ROOT)}")
    return out


def figure_sparsity(plt, endpoints: list[str]) -> Path | None:
    frames = []
    for endpoint in endpoints:
        grid_path = TABLES / f"feature_selection_{endpoint}_grid.csv"
        summary_path = TABLES / f"feature_selection_{endpoint}_summary.json"
        if not grid_path.exists() or not summary_path.exists():
            continue
        grid = pd.read_csv(grid_path)
        grid["endpoint"] = endpoint
        grid["winner"] = grid["setting"] == json.loads(summary_path.read_text(encoding="utf-8"))["winner_setting"]
        frames.append(grid)
    if not frames:
        print("[sparsity] no feature-selection grids; skipping")
        return None

    fig, axes = plt.subplots(1, len(frames), figsize=(6.0 * len(frames), 4.2), squeeze=False)
    for ax, grid in zip(axes[0], frames):
        endpoint = grid["endpoint"].iloc[0]
        grid = grid.sort_values("n_features_mean")
        ax.errorbar(grid["n_features_mean"], grid["inner_score_mean"],
                    yerr=grid["inner_score_sd"], fmt="o-", color=BLUE, ecolor=GREY,
                    capsize=3, linewidth=1.6, markersize=6, label="sparsity setting")
        win = grid[grid["winner"]]
        if not win.empty:
            ax.scatter(win["n_features_mean"], win["inner_score_mean"], s=200, facecolors="none",
                       edgecolors=RED, linewidths=2.2, zorder=4, label="selected")
        best = grid["inner_score_mean"].max()
        ax.axhline(best, color=GREY, linestyle=":", linewidth=1.2)
        ax.text(grid["n_features_mean"].max(), best, "  best inner-CV score",
                va="bottom", ha="right", fontsize=8, color="#595959")
        ax.set_xlabel("Features retained (mean across inner folds)", fontsize=10)
        ax.set_ylabel("Inner-CV score", fontsize=10)
        ax.set_title(f"Sparsity–performance trade-off: {endpoint}", fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        _style(ax)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)

    fig.tight_layout()
    out = FIGURES / "figure_S_feature_selection_sparsity.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[sparsity] -> {out.relative_to(REPO_ROOT)}")
    return out


def figure_shap_blocks(plt, endpoints: list[str]) -> Path | None:
    blocks_by_endpoint: dict[str, pd.Series] = {}
    for endpoint in endpoints:
        path = TABLES / f"feature_selection_{endpoint}_shap_values.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["block"] = frame["feature"].map(assign_block)
        grouped = frame.groupby("block")["mean_abs_shap"].sum()
        total = grouped.sum()
        if total > 0:
            blocks_by_endpoint[endpoint] = (grouped / total * 100).sort_values(ascending=True)
    if not blocks_by_endpoint:
        print("[shap-blocks] no SHAP value tables; skipping")
        return None

    fig, axes = plt.subplots(1, len(blocks_by_endpoint), figsize=(7.2 * len(blocks_by_endpoint), 4.6), squeeze=False)
    for ax, (endpoint, shares) in zip(axes[0], blocks_by_endpoint.items()):
        ax.barh(range(len(shares)), shares.to_numpy(), color=BLUE, height=0.65)
        ax.set_yticks(range(len(shares)))
        ax.set_yticklabels(shares.index, fontsize=9)
        ax.set_xlabel("Share of total |SHAP| attribution (%)", fontsize=10)
        ax.set_title(f"Attribution by annotation block: {endpoint}", fontsize=11, loc="left")
        for i, value in enumerate(shares.to_numpy()):
            ax.text(value + 0.5, i, f"{value:.1f}%", va="center", fontsize=8.5)
        ax.set_xlim(0, float(shares.max()) * 1.18)
        _style(ax)

    fig.tight_layout()
    out = FIGURES / "figure_S_shap_by_block.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[shap-blocks] -> {out.relative_to(REPO_ROOT)}")
    return out


ABLATION_LABELS = {
    "full": "Full model",
    "no_hrd_genes": "minus HRD/HRR gene features",
    "no_vaf": "minus VAF features",
}


def figure_hrd_ablation(plt) -> Path | None:
    """Paired deltas from removing each candidate shortcut, per HRD endpoint."""
    comp_path = TABLES / "hrd_shortcut_ablation_comparisons.csv"
    summary_path = TABLES / "hrd_shortcut_ablation_summary.csv"
    if not comp_path.exists() or not summary_path.exists():
        print("[hrd-ablation] no ablation tables; skipping")
        return None
    comparisons = pd.read_csv(comp_path)
    summary = pd.read_csv(summary_path)

    endpoints = list(dict.fromkeys(summary["endpoint"].astype(str)))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

    # Left: absolute score per arm, grouped by endpoint.
    ax = axes[0]
    arms = [a for a in ABLATION_LABELS if a in set(summary["arm"].astype(str))]
    width = 0.8 / max(1, len(arms))
    palette = {"full": NAVY, "no_hrd_genes": BLUE, "no_vaf": "#7FA3D1"}
    for j, arm in enumerate(arms):
        vals, xs = [], []
        for i, endpoint in enumerate(endpoints):
            row = summary[(summary["endpoint"] == endpoint) & (summary["arm"] == arm)]
            if row.empty:
                continue
            vals.append(float(row["score"].iloc[0]))
            xs.append(i + j * width - 0.4 + width / 2)
        ax.bar(xs, vals, width=width * 0.92, label=ABLATION_LABELS[arm], color=palette.get(arm, BLUE))
    ax.set_xticks(range(len(endpoints)))
    ax.set_xticklabels(endpoints, fontsize=9)
    ax.set_ylabel("Score (Spearman r / AUROC)", fontsize=10)
    ax.set_ylim(0.6, 1.0)
    ax.set_title("HRD performance with shortcuts removed", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)

    # Right: paired deltas with bootstrap CIs. A CI crossing zero means the shortcut is not
    # carrying the result, which is the defensible outcome.
    ax2 = axes[1]
    rows = comparisons.sort_values(["candidate", "endpoint"]).reset_index(drop=True)
    ypos = np.arange(len(rows))
    deltas = rows["delta"].to_numpy(dtype=float)
    lows = deltas - rows["ci_low"].to_numpy(dtype=float)
    highs = rows["ci_high"].to_numpy(dtype=float) - deltas
    colors = [RED if row["ci_high"] < 0 else (GREEN if row["ci_low"] > 0 else GREY) for _, row in rows.iterrows()]
    ax2.errorbar(deltas, ypos, xerr=[lows, highs], fmt="none", ecolor=GREY, capsize=4, linewidth=1.4)
    ax2.scatter(deltas, ypos, s=70, c=colors, zorder=3)
    ax2.axvline(0.0, color=NAVY, linestyle="--", linewidth=1.2)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([f"{r['endpoint']}\n{ABLATION_LABELS.get(r['candidate'], r['candidate'])}" for _, r in rows.iterrows()], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Change vs full model (95% paired bootstrap CI)", fontsize=10)
    ax2.set_title("Cost of removing each shortcut", fontsize=11, loc="left")
    _style(ax2)

    fig.tight_layout()
    out = FIGURES / "figure_S_hrd_shortcut_ablation.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[hrd-ablation] -> {out.relative_to(REPO_ROOT)}")
    return out


# What each representation can encode. The point is not that one is larger, but that each can
# express things the other cannot -- and that the information MuAt cannot express is where most
# of the tabular feature space lives.
INFORMATION_TYPES = [
    # (label, MuAt, ours, note)
    ("Sequence context\n(trinucleotide)",        True,  True,  "motif token / SBS96 channels"),
    ("Genomic position\n(1-Mb bin)",             True,  False, "MuAt only"),
    ("Genic / exonic / strand",                  True,  True,  "annotation token / VEP consequence"),
    ("Gene identity",                            False, True,  "794 columns, 397 genes"),
    ("Functional prediction\n(SIFT, PolyPhen)",  False, True,  "38 columns"),
    ("Curated driver knowledge\n(OncoKB, IntOGen)", False, True, "40 columns"),
    ("Hotspot annotation",                       False, True,  "4 columns"),
    ("Allele fraction",                          False, True,  "5 columns"),
]

# Composition of the 1,130-feature signatures + Bio MAF v4 representation, split by whether
# MuAt's three input modalities could express the same information at all.
FEATURE_BLOCKS = [
    ("Per-gene driver counts",              794, False),
    ("Signatures (SBS96 / ID83 / burden)",  182, True),
    ("VEP consequence",                      52, True),
    ("VEP other (SIFT, PolyPhen, biotype)",  38, False),
    ("Driver evidence / tiers",              24, False),
    ("OncoKB role",                          16, False),
    ("VEP impact",                           10, False),
    ("Allele fraction (VAF)",                 5, False),
    ("Hotspots",                              4, False),
    ("Other",                                 5, False),
]


def figure_feature_comparison(plt) -> Path | None:
    """MuAt's input modalities against the tabular representation's feature space."""
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={"width_ratios": [1, 1.18]})

    # --- Panel A: what each representation can encode -------------------------------------
    ax = axes[0]
    n = len(INFORMATION_TYPES)
    # Columns are spaced 1.6 apart rather than 1.0: the two axis labels are wide, and at unit
    # spacing they overlap each other.
    col_x = (0.0, 1.6)
    for i, (label, muat, ours, note) in enumerate(INFORMATION_TYPES):
        y = n - 1 - i
        for x, has in zip(col_x, (muat, ours)):
            if has:
                ax.scatter(x, y, s=340, marker="o", color=NAVY if x == col_x[0] else BLUE, zorder=3)
                ax.text(x, y, "✓", ha="center", va="center", color="white", fontsize=13, fontweight="bold", zorder=4)
            else:
                ax.scatter(x, y, s=340, marker="o", facecolors="white", edgecolors=GREY, linewidths=1.6, zorder=3)
        ax.text(2.6, y, note, va="center", fontsize=8, color="#595959")
    ax.set_yticks(range(n))
    ax.set_yticklabels([t[0] for t in reversed(INFORMATION_TYPES)], fontsize=9)
    ax.set_xticks(list(col_x))
    ax.set_xticklabels(["MuAt\n(3 modalities)", "Signatures +\nBio MAF v4"], fontsize=10, fontweight="bold")
    ax.set_xlim(-0.8, 5.6)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_title("A. What each representation can encode", fontsize=11, loc="left", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # --- Panel B: feature composition, split by MuAt expressibility ------------------------
    ax2 = axes[1]
    blocks = sorted(FEATURE_BLOCKS, key=lambda b: b[1])
    ypos = np.arange(len(blocks))
    colors = [BLUE if b[2] else RED for b in blocks]
    ax2.barh(ypos, [b[1] for b in blocks], color=colors, height=0.66)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([b[0] for b in blocks], fontsize=9)
    for i, b in enumerate(blocks):
        ax2.text(b[1] + 12, i, str(b[1]), va="center", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Number of features (of 1,130 total)", fontsize=10)
    ax2.set_xlim(0, 900)
    ax2.set_title("B. Feature composition of the tabular representation", fontsize=11, loc="left", fontweight="bold")
    ax2.legend(
        handles=[
            Patch(facecolor=BLUE, label="MuAt's inputs can express this"),
            Patch(facecolor=RED, label="MuAt's inputs cannot express this"),
        ],
        frameon=False, fontsize=9, loc="lower right",
    )
    _style(ax2)

    expressible = sum(b[1] for b in FEATURE_BLOCKS if b[2])
    total = sum(b[1] for b in FEATURE_BLOCKS)
    fig.text(
        0.5, -0.03,
        f"{total - expressible} of {total} features ({100 * (total - expressible) / total:.0f}%) encode information "
        f"outside MuAt's three input modalities. MuAt's finest spatial resolution is one megabase, "
        f"spanning 5-15 genes,\nso it cannot distinguish a mutation in TP53 from one in a neighbouring gene. "
        f"Conversely, MuAt encodes genomic position, which the tabular representation does not.",
        ha="center", fontsize=8.5, color="#404040",
    )

    fig.tight_layout()
    out = FIGURES / "figure_S_representation_comparison.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[feature-comparison] -> {out.relative_to(REPO_ROOT)}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+",
                        choices=["survival", "sparsity", "shap", "hrd_ablation", "features"], default=None)
    parser.add_argument("--endpoints", nargs="+", default=["cancer_type_top20", "OS"])
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required: pip install matplotlib")

    FIGURES.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only or ["survival", "sparsity", "shap", "hrd_ablation", "features"])
    made = []
    if "survival" in wanted:
        made.append(figure_survival(plt))
    if "sparsity" in wanted:
        made.append(figure_sparsity(plt, args.endpoints))
    if "shap" in wanted:
        made.append(figure_shap_blocks(plt, args.endpoints))
    if "hrd_ablation" in wanted:
        made.append(figure_hrd_ablation(plt))
    if "features" in wanted:
        made.append(figure_feature_comparison(plt))

    made = [path for path in made if path]
    print(f"\n{len(made)} figure(s) written to results/figures/")
    for path in made:
        print(f"  {path.name}")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
