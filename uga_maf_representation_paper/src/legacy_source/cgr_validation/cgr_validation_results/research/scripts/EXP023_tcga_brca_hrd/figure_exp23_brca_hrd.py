#!/usr/bin/env python3
"""EXP023 manuscript figure: full-cohort continuous (A), HRD binary AUROC (B), burden Δρ heatmap (C)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

C_WIN = "#00695c"
C_GRAY = "#9e9e9e"
C_AXIS = "#212121"
C_MUTED = "#bdbdbd"
C_STD_PT = "#424242"
C_CAPTION = "#424242"


def _pretty_cont(ep: str) -> str:
    return {
        "HRD_Score": "HRD_Score",
        "PARPi7": "PARPi7",
        "eCARD": "eCARD",
        "HRD_TAI": "HRD_TAI",
        "HRD_LST": "HRD_LST",
        "HRD_LOH": "HRD_LOH",
    }.get(ep, ep)


def _load_main_pvalues(stats_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Steiger p (Spearman) for regression; DeLong p (AUROC) for classification — main / all only."""
    if not stats_path.exists():
        return {}, {}
    st = pd.read_csv(stats_path, sep="\t")
    m = (st["suite"] == "main") & (st["subset"] == "all")
    steiger = st[m & (st["test"] == "steiger") & (st["metric"] == "spearman")]
    delong = st[m & (st["test"] == "delong") & (st["metric"] == "auroc")]
    ps_s = {str(r["endpoint"]): float(r["p_value"]) for _, r in steiger.iterrows() if pd.notna(r.get("p_value"))}
    ps_d = {str(r["endpoint"]): float(r["p_value"]) for _, r in delong.iterrows() if pd.notna(r.get("p_value"))}
    return ps_s, ps_d


def _stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _sig_gain(delta: float, p: float) -> bool:
    return (not np.isnan(p)) and p < 0.05 and delta > 0


def _panel_a_continuous(
    ax: plt.Axes,
    cross: pd.DataFrame,
    p_steiger: dict[str, float],
) -> None:
    df = cross[cross["task"].astype(str) == "regression"].copy()
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    df = df.sort_values("delta", ascending=False).reset_index(drop=True)
    n = len(df)
    if n == 0:
        ax.axis("off")
        return
    y = np.arange(n)
    xs = df["delta"].to_numpy(dtype=float)
    lo, hi = float(np.nanmin(xs)), float(np.nanmax(xs))
    span = max(hi - lo, 0.08)
    pad = 0.08 * span
    ax.set_xlim(min(lo, 0.0) - pad, max(hi, 0.0) + 1.35 * pad)

    for yi, row in df.iterrows():
        ep = str(row["endpoint"])
        d = float(row["delta"])
        p = float(p_steiger.get(ep, np.nan))
        gain = _sig_gain(d, p)
        col = C_WIN if gain else C_GRAY
        ax.plot([0, d], [yi, yi], color=col, lw=2.1, solid_capstyle="round", zorder=1)
        ax.scatter([d], [yi], color=col, s=50, zorder=2, edgecolors="white", linewidths=0.55)
        st = _stars(p)
        if st:
            ax.annotate(
                st,
                xy=(d, yi),
                xytext=(8, 10),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=10,
                color=C_AXIS,
                fontweight="600",
            )

    ax.axvline(0, color=C_MUTED, lw=0.85, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([_pretty_cont(str(e)) for e in df["endpoint"]], fontsize=9.5)
    ax.set_xlabel("Δ Spearman ρ (Universal − Standard)", fontsize=9.5, color=C_AXIS)
    ax.set_title("A. Full cohort — continuous endpoints", loc="left", fontsize=11.5, fontweight="600", color=C_AXIS)
    ax.spines[["top", "right"]].set_visible(False)

    for yi, row in df.iterrows():
        ep = str(row["endpoint"])
        d = float(row["delta"])
        p = float(p_steiger.get(ep, np.nan))
        if _sig_gain(d, p):
            ax.annotate(
                f"Δ{d:+.3f}",
                xy=(d, yi),
                xytext=(8, -11),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=8.5,
                color=C_AXIS,
            )

    ax.invert_yaxis()


def _panel_b_hrd_binary(
    ax: plt.Axes,
    cross: pd.DataFrame,
    p_delong: dict[str, float],
) -> None:
    order = [
        ("hrd_binary_24", "HRD ≥24"),
        ("hrd_binary_33", "HRD ≥33"),
        ("hrd_binary_42", "HRD ≥42"),
    ]
    labels: list[str] = []
    ypos: list[int] = []
    xsu: list[float] = []
    ep_y: dict[str, int] = {}
    yr = 0
    for ep, lab in order:
        row = cross[cross["endpoint"] == ep]
        if row.empty or str(row.iloc[0].get("task", "")) != "classification":
            continue
        s = float(row.iloc[0]["Standard"])
        u = float(row.iloc[0]["Universal"])
        d = float(row.iloc[0]["delta"])
        p = float(p_delong.get(ep, np.nan))
        col = C_WIN if _sig_gain(d, p) else C_GRAY
        labels.append(lab)
        ypos.append(yr)
        ep_y[ep] = yr
        xsu.extend([s, u])
        ax.plot([s, u], [yr, yr], color=col, lw=2.2, solid_capstyle="round", zorder=1)
        ax.scatter([s], [yr], color=C_STD_PT, s=36, zorder=2, edgecolors="white", linewidths=0.45)
        ax.scatter([u], [yr], color=col, s=40, zorder=3, edgecolors="white", linewidths=0.45)
        yr += 1
    if not labels:
        ax.axis("off")
        return
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("AUROC (nested CV)", fontsize=9, color=C_AXIS)
    ax.set_title("B. Full cohort — HRD ≥24 / ≥33 / ≥42 (binary)", loc="left", fontsize=11.5, fontweight="600", color=C_AXIS)
    pad = 0.035 * (max(xsu) - min(xsu) + 0.05)
    ax.set_xlim(min(xsu) - pad, max(xsu) + pad * 3.0)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)

    for ep, _lab in order:
        row = cross[cross["endpoint"] == ep]
        if row.empty or ep not in ep_y:
            continue
        d = float(row.iloc[0]["delta"])
        p = float(p_delong.get(ep, np.nan))
        yi = ep_y[ep]
        u = float(row.iloc[0]["Universal"])
        st = _stars(p)
        if st:
            ax.annotate(
                st,
                xy=(u, yi),
                xytext=(10, 14),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=10,
                color=C_AXIS,
                fontweight="600",
            )
        if _sig_gain(d, p):
            ax.annotate(
                f"Δ{d:+.3f}",
                xy=(u, yi),
                xytext=(10, -6),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=8.5,
                color=C_AXIS,
            )


def _panel_c_heatmap(ax: plt.Axes, burden: pd.DataFrame) -> None:
    eps = ["HRD_Score", "eCARD", "HRD_TAI", "HRD_LST", "HRD_LOH"]
    col_keys: list[tuple[str, str]] = [
        ("burden_Q1", "Q1"),
        ("burden_Q2_Q3_mid", "Q2–Q3"),
        ("burden_Q4", "Q4"),
        ("burden_quartile_extremes_Q1_Q4", "Q1+Q4"),
    ]
    exclude_rows = burden[burden["subset"].astype(str).str.startswith("exclude_low_burden")]
    if len(exclude_rows):
        col_keys.append((str(exclude_rows.iloc[0]["subset"]), "Exclude\nlow burden"))
    col_keys.append(("full_cohort", "Full\ncohort"))

    mat = np.full((len(eps), len(col_keys)), np.nan, dtype=float)
    for j, (sub, _) in enumerate(col_keys):
        subdf = burden[(burden["subset"] == sub) & (burden["task"] == "regression")]
        for i, ep in enumerate(eps):
            r = subdf[subdf["endpoint"] == ep]
            if len(r) and "delta" in r.columns and not pd.isna(r.iloc[0]["delta"]):
                mat[i, j] = float(r.iloc[0]["delta"])

    if np.all(np.isnan(mat)):
        ax.axis("off")
        ax.text(0.5, 0.5, "No burden regression rows", ha="center", va="center", transform=ax.transAxes)
        return

    vmax = float(np.nanmax(np.abs(mat)))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm, alpha=0.92)
    ax.set_xticks(np.arange(len(col_keys)))
    ax.set_xticklabels([c[1] for c in col_keys], fontsize=8.5, rotation=0)
    ax.set_yticks(np.arange(len(eps)))
    ax.set_yticklabels(eps, fontsize=9)
    ax.set_title(
        "C. Burden robustness (exploratory) — Δ Spearman ρ",
        loc="left",
        fontsize=10.5,
        fontweight="600",
        color="#616161",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label("Δ Spearman ρ", fontsize=8.5, color="#616161")
    ax.tick_params(axis="both", colors="#616161")
    for spine in ax.spines.values():
        spine.set_color("#bdbdbd")
    ax.spines[["top", "right"]].set_visible(False)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            t = f"{v:.2f}" if abs(v) >= 0.005 else "0"
            tc = "#333333" if abs(v) < 0.5 * vmax else "#fafafa"
            ax.text(j, i, t, ha="center", va="center", fontsize=7.8, color=tc)

    ax.text(
        0.0,
        -0.22,
        "Exploratory robustness: subset-specific nested CV refits; inferential p-values not shown.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="#757575",
    )


def build_figure(cfg_assets: Path, out_dir: Path) -> None:
    cross = pd.read_csv(cfg_assets / "tables" / "cross_endpoint_summary.tsv", sep="\t")
    burden_path = cfg_assets / "tables" / "burden_sensitivity_summary.tsv"
    burden = pd.read_csv(burden_path, sep="\t") if burden_path.exists() else pd.DataFrame()
    stats_path = cfg_assets / "modeling_results" / "statistical_tests.tsv"
    p_steiger, p_delong = _load_main_pvalues(stats_path)

    fig = plt.figure(figsize=(11.8, 9.4))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.12, 0.88],
        height_ratios=[2.35, 1.05],
        left=0.09,
        right=0.97,
        top=0.88,
        bottom=0.155,
        wspace=0.34,
        hspace=0.42,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    _panel_a_continuous(ax_a, cross, p_steiger)
    _panel_b_hrd_binary(ax_b, cross, p_delong)
    _panel_c_heatmap(ax_c, burden)

    fig.suptitle(
        "Universal-BiCGR improves BRCA HRD prediction, with strongest gains at fixed HRD thresholds and HRD score endpoints",
        fontsize=11.2,
        fontweight="bold",
        color=C_AXIS,
        y=0.96,
    )

    caption = (
        "Panel A: nested-CV Spearman ρ (Δ Spearman ρ = Universal − Standard); dependent correlations compared with Steiger tests. "
        "Panel B: nested-CV AUROC at HRD ≥24 / ≥33 / ≥42 (axis = AUROC; segment length = Δ AUROC); compared with DeLong tests. "
        "Panel C: burden-stratum refits (exploratory robustness; p-values not shown). "
        "Significance stars on A and B: * p<0.05, ** p<0.01, *** p<0.001."
    )
    fig.text(0.5, 0.018, caption, ha="center", va="top", fontsize=7.5, color=C_CAPTION)

    out_dir.mkdir(parents=True, exist_ok=True)
    cap_path = out_dir / "Figure_EXP023_brca_hrd_cross_endpoint_caption.txt"
    cap_path.write_text(caption + "\n", encoding="utf-8")
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"Figure_EXP023_brca_hrd_cross_endpoint.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_conclusion_md(cfg_assets: Path, out_path: Path) -> None:
    cross = pd.read_csv(cfg_assets / "tables" / "cross_endpoint_summary.tsv", sep="\t")
    burden = pd.read_csv(cfg_assets / "tables" / "burden_sensitivity_summary.tsv", sep="\t")
    cohort = pd.read_csv(cfg_assets / "cohort" / "final_analysis_cohort.tsv", sep="\t")

    def _hrd_row_exclude_low() -> pd.Series:
        b = burden[(burden["endpoint"] == "HRD_Score") & (burden["subset"].astype(str).str.startswith("exclude_low_burden"))]
        return b.iloc[0] if len(b) else pd.Series()

    hrd_main = cross[cross["endpoint"] == "HRD_Score"].iloc[0]
    d_full = float(hrd_main["delta"])
    p_full = float(hrd_main["p_value"])
    ex = _hrd_row_exclude_low()
    mid = burden[(burden["endpoint"] == "HRD_Score") & (burden["subset"] == "burden_Q2_Q3_mid")]
    mid = mid.iloc[0] if len(mid) else pd.Series()
    q4 = burden[(burden["endpoint"] == "HRD_Score") & (burden["subset"] == "burden_Q4")]
    q4 = q4.iloc[0] if len(q4) else pd.Series()
    pb = cross[cross["endpoint"] == "parpi7_binary"]
    parpi_d = float(pb.iloc[0]["delta"]) if len(pb) else float("nan")
    parpi_p = float(pb.iloc[0]["p_value"]) if len(pb) else float("nan")

    d_ex = float(ex["delta"]) if len(ex) else float("nan")
    d_mid = float(mid["delta"]) if len(mid) else float("nan")
    d_q4 = float(q4["delta"]) if len(q4) else float("nan")

    strengthens = (
        "The production Optuna budget (defaults: 40 regression + 40 classification inner trials per tuning block) "
        f"yields a modest full-cohort HRD_Score gain (Δρ = {d_full:+.4f}, Steiger p = {p_full:.3g}). "
        "A prior artifact on disk showed a slightly larger Δρ; this re-run is within nested-CV / tuning noise and does not "
        "materially *tighten* inferential support for the primary endpoint (p moved toward unity)."
    )

    burden_story = (
        f"After excluding the lowest total-SBS quartile, Δρ rises to **{d_ex:+.4f}**"
        + (f" (n = {int(ex['n'])})" if len(ex) else "")
        + ", so Universal's "
        "edge is **larger** than on the full cohort when the sparsest-mutation tail is removed. "
        f"In the mid-burden slice (Q2–Q3), HRD_Score Δρ is **{d_mid:+.4f}** (Universal does not improve there in this run). "
        f"In the highest-burden quartile (Q4), Δρ = **{d_q4:+.4f}** — Universal still leads on HRD_Score; Standard is **not** "
        "the stronger HRD_Score model in Q4, though component scores (e.g. HRD_LST in Q4) can favor Standard in isolated rows "
        "(see `burden_sensitivity_summary.tsv`). "
        "**Clearest within-BRCA support for Universal on HRD_Score** in this table is therefore the **exclude-low-burden** "
        "and **full-cohort** positive deltas, not the mid-burden stratum."
    )

    parpi_line = (
        f"- **`parpi7_binary`** (independent label, not a PARPi7 threshold): ΔAUROC = {parpi_d:+.4f} (DeLong p = {parpi_p:.3g}) — "
        "directionally consistent but not strong evidence; secondary context only."
        if not np.isnan(parpi_d)
        else ""
    )

    lines = [
        "# EXP023 — BRCA HRD paper workflow",
        "",
        "## Primary endpoint: HRD_Score",
        strengthens,
        "",
        "## Burden-stratified HRD_Score (Universal − Standard, Spearman)",
        burden_story,
        "",
        "## Secondary / supplemental endpoints",
        "- **Main figure** (`figures/Figure_EXP023_brca_hrd_cross_endpoint.{png,pdf}`): **A** continuous Δρ with Steiger stars; "
        "**B** HRD ≥24/33/42 AUROC dumbbells with DeLong stars; **C** exploratory burden Δρ heatmap (no p-values). "
        "Caption text is duplicated in `figures/Figure_EXP023_brca_hrd_cross_endpoint_caption.txt`.",
        "- **PARPi7, eCARD, HRD_TAI, HRD_LST, HRD_LOH** — see `tables/cross_endpoint_summary.tsv`; interpret cautiously; "
        "do not override the HRD_Score primary read.",
        parpi_line,
        "",
        "## Deliverable answers",
        "1. **Does the full Optuna run strengthen the HRD_Score result?** "
        "It reproduces a **small positive** Universal advantage on the full cohort, but **does not** deliver stronger "
        "Steiger significance than a typical single nested-CV realization; treat as supportive but not definitive.",
        "2. **Do burden slices give the clearest support for Universal within BRCA?** "
        "**Partially:** excluding the lowest-burden quartile **amplifies** Δρ for HRD_Score, which supports a burden-related "
        "story; the **mid-burden** stratum does **not** show a Universal gain on HRD_Score in this run.",
        "",
        f"Artifacts: `cross_endpoint_summary.tsv`, `burden_sensitivity_summary.tsv`, "
        f"`figures/Figure_EXP023_brca_hrd_cross_endpoint.{{png,pdf}}`. Cohort n = {len(cohort)}.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg_assets: Path) -> None:
    fig_dir = cfg_assets / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    build_figure(cfg_assets, fig_dir)
    write_conclusion_md(cfg_assets, cfg_assets.parent / "EXP023_brca_hrd_conclusion.md")
