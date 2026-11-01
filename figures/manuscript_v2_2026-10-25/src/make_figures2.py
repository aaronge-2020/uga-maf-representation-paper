#!/usr/bin/env python3
"""Publication redraw of manuscript Figures 2-8 (nature-figure skill, Python backend).

Values are frozen (frozen_data.json, extracted and verified in the prior audit
round). This script changes craft only: one palette, one type system, panel
labels, alignment-gated layouts, 300-dpi PNG + vector PDF at 170 mm wide.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

sys.path.insert(0, "/home/hatch/workspace/custom-skills/nature-skills/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment

BASE = "/home/hatch/workspace/scratch/uga-maf-audit/figregen2"
D = json.load(open(f"{BASE}/frozen_data.json"))
MM = 1 / 25.4
W170 = 170 * MM

# ---------------------------------------------------------------- style system
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
plt.rcParams.update({
    "font.size": 7.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#4D4D4D",
    "xtick.color": "#4D4D4D", "ytick.color": "#4D4D4D",
    "text.color": "#272727", "axes.labelcolor": "#272727",
    "legend.frameon": False,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.titlesize": 8.5, "axes.labelsize": 8,
})

C_BURDEN = "#767676"   # baseline neutral
C_SPECTRA = "#3775BA"  # blue secondary
C_BIOMAF = "#42949E"   # teal
C_SIGMAF = "#0F4D92"   # hero dark blue
C_MUAT = "#E28E2C"     # warm accent for the comparator
C_PUB = "#C9C9C9"      # external reference
C_UP = "#2E9E44"       # delta up
C_DOWN = "#B64342"     # delta down
C_INK = "#272727"
C_GRID = "#E3E3E3"


def panel_label(ax, label, xoff=-6, yoff=4):
    off = ScaledTranslation(xoff / 72, yoff / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, label, transform=ax.transAxes + off, fontsize=9,
            fontweight="bold", color=C_INK, ha="left", va="bottom")


def export(fig, name, alignment=None):
    """alignment: dict of kwargs for the gate, or None for single-panel."""
    if alignment is not None:
        require_matplotlib_panel_alignment(
            fig, json_out=f"{BASE}/out/{name}.alignment.json",
            overlay_svg=f"{BASE}/out/{name}.alignment.svg",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5,
            require_panel_labels=True, strict=True, **alignment)
    else:
        fig.canvas.draw()
    fig.savefig(f"{BASE}/out/{name}.pdf")
    fig.savefig(f"{BASE}/out/{name}.png", dpi=300)
    plt.close(fig)
    print("exported", name)


# ---------------------------------------------------------------- Figure 2
def fig2():
    f = D["fig2"]; cis = D["fig2_cis"]
    order = [("burden_only", "Burden", C_BURDEN), ("standard_sbs96_id83", "Spectra", C_SPECTRA),
             ("MAF_stack_only", "Bio\nMAF", C_BIOMAF), ("signatures_plus_MAF_stack", "Sig+\nMAF", C_SIGMAF),
             ("muat_faithful", "MuAt-\ncompatible", C_MUAT)]
    published = [0.641, 0.612, 0.906]
    panels = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
              ("top5_accuracy", "Top-5 accuracy")]
    fig, axes = plt.subplots(1, 3, figsize=(W170, 3.0), sharey=True)
    fig.subplots_adjust(left=0.072, right=0.995, top=0.78, bottom=0.14, wspace=0.10)
    for pi, (ax, (key, title)) in enumerate(zip(axes, panels)):
        vals = [f[r][key] for r, _, _ in order] + [published[pi]]
        cols = [c for _, _, c in order] + [C_PUB]
        bars = ax.bar(range(6), vals, color=cols, edgecolor="white", linewidth=0.4, width=0.68)
        pub = bars[5]; pub.set_edgecolor("#8a8a8a"); pub.set_linewidth(0.6)
        bx, bw, bh = pub.get_x(), pub.get_width(), pub.get_height()
        m = 0.171  # data slope giving ~45 degrees at this panel's aspect
        seg = []
        for y0 in np.arange(-0.05, bh + 0.05, 0.034):
            xa, ya, xb, yb = bx, y0, bx + bw, y0 + m * bw
            if yb > bh: xb, yb = bx + (bh - y0) / m, bh
            if ya < 0: xa, ya = bx - y0 / m, 0.0
            if 0 <= ya <= bh and 0 <= yb <= bh and xb > xa:
                seg.append([(xa, ya), (xb, yb)])
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(seg, colors=["#8a8a8a"], linewidths=0.55))
        for i, (r, _, _) in enumerate(order):
            m, lo, hi = cis[r][pi]
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=C_INK,
                        elinewidth=0.8, capsize=2.2, capthick=0.8)
        for i, v in enumerate(vals):
            top = v if i == 5 else cis[order[i][0]][pi][2]
            ax.text(i, top + 0.065, f"{v:.3f}", ha="center", fontsize=6.8, color=C_INK)
        ax.set_xticks([])
        ax.set_title(title, fontweight="bold", pad=7)
        ax.set_ylim(0, 1.20)
        ax.tick_params(length=3)
        panel_label(ax, "abc"[pi], yoff=13)
    axes[0].set_ylabel("Score")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in order] +               [plt.Rectangle((0, 0), 1, 1, color=C_PUB, ec="#8a8a8a")]
    fig.legend(handles, [l.replace("\n", "-") if False else l for _, l, _ in
                         [(r, lab, c) for r, lab, c in
                          [("burden_only", "Burden", 0), ("standard_sbs96_id83", "Spectra", 0),
                           ("MAF_stack_only", "Bio MAF", 0), ("signatures_plus_MAF_stack", "Sig+MAF", 0),
                           ("muat_faithful", "MuAt-compatible", 0)]]] + ["Published MuAt*"],
               loc="upper center", ncol=6, fontsize=7.2, bbox_to_anchor=(0.53, 1.0),
               handlelength=1.0, handleheight=0.95, columnspacing=0.9, handletextpad=0.4)
    fig.text(0.008, 0.012,
             "* Published MuAt: contextual (n = 7,352, official framework, ten-model ensemble); balanced accuracy 0.612 derived\n"
             "from the paper's supplementary per-class recall (Table S2).",
             fontsize=6.3, style="italic", color="#4D4D4D")
    export(fig, "figure2", alignment=dict())


# ---------------------------------------------------------------- Figure 3
def fig3():
    rows = D["fig3"]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(W170, 6.05))
    fig.subplots_adjust(left=0.148, right=0.965, top=0.925, bottom=0.07)
    for i, r in enumerate(rows):
        y = (n - 1 - i) * 1.0
        ax.barh(y + 0.21, r["spectra"], height=0.36, color=C_SPECTRA, edgecolor="white", linewidth=0.3)
        ax.barh(y - 0.21, r["sigmaf"], height=0.36, color=C_SIGMAF, edgecolor="white", linewidth=0.3)
        gx = max(r["spectra"], r["sigmaf"]) + 0.018
        ax.text(gx, y, f"+{r['gain']:.2f}", va="center", fontsize=7.3,
                fontweight="bold", color=C_SIGMAF)
    ax.set_yticks([ (n - 1 - i) * 1.0 for i in range(n)])
    ax.set_yticklabels([f"{r['label']} (n={r['n']:,})" for r in rows], fontsize=7.4)
    ax.set_xlim(0, 1.10); ax.set_ylim(-0.62, n - 0.28)
    ax.set_xlabel("Per-type recall (pooled out-of-fold, XGBoost)")
    ax.tick_params(length=3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_SPECTRA),
               plt.Rectangle((0, 0), 1, 1, color=C_SIGMAF)]
    ax.legend(handles, ["Spectra", "Sig+MAF"], loc="lower right", fontsize=7.5,
              handlelength=1.1, handleheight=0.9, borderaxespad=0.0,
              bbox_to_anchor=(1.0, 1.01))
    export(fig, "figure3")


# ------------------------------------------------------------- Figures 4 / 5
EP_ROWS = [("damage_class", "Kucab damage class", "Bal acc", "n=259"),
           ("HRD_Score", "HRD score", "rho", "n=772"),
           ("hrd_binary_24", "HRD24 high/low", "AUROC", "n=772"),
           ("hrd_binary_33", "HRD33 high/low", "AUROC", "n=772"),
           ("hrd_binary_42", "HRD42 high/low", "AUROC", "n=772"),
           ("cancer_type_top20", "Cancer type (top 20)", "Bal acc", "n=8,800")]
SUB = "Bars: pooled out-of-fold scores. Differences are descriptive; no confidence intervals or significance tests are shown."


XLIM = 0.40


def grouped(fig_key, series, title, fname, diff_defs, labels_abcd):
    fig = plt.figure(figsize=(W170, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.7, 1.25], height_ratios=[1, 1],
                          left=0.185, right=0.985, top=0.85, bottom=0.16,
                          wspace=0.16, hspace=0.62)
    axs = {}
    for ri, learner in enumerate(["elastic_net", "xgboost"]):
        ax = fig.add_subplot(gs[ri, 0]); axd = fig.add_subplot(gs[ri, 1])
        axs[learner] = (ax, axd)
        for row, (ep, disp, metric, nn) in enumerate(EP_ROWS):
            y = len(EP_ROWS) - row
            d = D[fig_key][ep][learner]
            for j, (skey, lab, col) in enumerate(series):
                v = d[skey]
                yy = y + (len(series) - 1 - j) * 0.34 - (len(series) - 1) * 0.17
                ax.barh(yy, v, height=0.29, color=col, edgecolor="white", linewidth=0.3)
                ax.text(v + 0.014, yy, f"{v:.3f}", va="center", fontsize=5.8, color=C_INK)
            for j, (a, b, lab) in enumerate(diff_defs):
                dv = d[a] - d[b]
                yy = y + (len(diff_defs) - 1 - j) * 0.34 - (len(diff_defs) - 1) * 0.17
                col = C_UP if dv >= 0 else C_DOWN
                axd.plot([dv], [yy], "o", ms=3.6, color=col, clip_on=False)
                axd.text(dv + (0.02 if dv >= 0 else -0.02), yy, f"{dv:+.3f}", va="center",
                         ha="left" if dv >= 0 else "right", fontsize=5.8, color=col)
        ax.set_yticks([len(EP_ROWS) - r for r in range(len(EP_ROWS))])
        ax.set_yticklabels([f"{disp}\n{metric}  {nn}" for _, disp, metric, nn in EP_ROWS], fontsize=7.2)
        ax.set_xlim(0, 1.07); ax.set_ylim(0.35, 6.75)
        ax.set_title("Elastic net" if learner == "elastic_net" else "XGBoost",
                     fontsize=8.5, fontweight="bold", pad=5)
        ax.set_xlabel("Performance score (pooled out-of-fold)")
        ax.tick_params(length=3)
        axd.axvline(0, color="#9a9a9a", lw=0.8)
        axd.set_yticks([]); axd.set_xlim(-XLIM, XLIM); axd.set_ylim(0.35, 6.75)
        axd.set_title("Differences", fontsize=8.5, pad=5)
        axd.tick_params(length=3)
        panel_label(ax, labels_abcd[ri * 2], yoff=13); panel_label(axd, labels_abcd[ri * 2 + 1], xoff=-4, yoff=13)
    fig.text(0.012, 0.972, title, fontsize=11.5, fontweight="bold", color=C_INK, va="top")
    fig.text(0.012, 0.918, SUB, fontsize=7.2, color="#4D4D4D", va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in series]
    fig.legend(handles, [lab for _, lab, _ in series], loc="lower center", ncol=2,
               fontsize=7.3, bbox_to_anchor=(0.5, 0.042), handlelength=1.1, handleheight=0.9,
               columnspacing=1.2, labelspacing=0.35)
    fig.text(0.5, 0.006, "Difference dots, top to bottom per endpoint: " +
             "    ".join(lab for _, _, lab in diff_defs),
             ha="center", fontsize=6.8, color="#4D4D4D")
    flat = [axs["elastic_net"][0], axs["elastic_net"][1], axs["xgboost"][0], axs["xgboost"][1]]
    export(fig, fname, alignment=dict(
        axes=flat, panel_ids=list(labels_abcd),
        row_groups=[[labels_abcd[0], labels_abcd[1]], [labels_abcd[2], labels_abcd[3]]],
        column_groups=[[labels_abcd[0], labels_abcd[2]], [labels_abcd[1], labels_abcd[3]]],
        exemptions=[{"panels": list(labels_abcd), "checks": ["panel-width"],
                     "reason": "bars:differences width ratio 3.2:1.05 is the design; value labels need the wide column, differences are a narrow readout"}]))


def fig4():
    globals().__setitem__("XLIM", 0.46)
    grouped("fig4", [("burden", "Mutation burden (TCGA: 3; Kucab: 1)", C_BURDEN),
                     ("spectra", "Mutational spectra alone (TCGA: 182; Kucab: 261)", C_SPECTRA)],
            "Figure 4. Count and pattern baselines", "figure4",
            [("spectra", "burden", "Spectra - burden")], "abcd")


def fig5():
    globals().__setitem__("XLIM", 0.60)
    grouped("fig5", [("spectra", "Mutational spectra alone (TCGA: 182; Kucab: 261)", C_SPECTRA),
                     ("biomaf", "Bio MAF (948); Kucab: event tokens (2,980)", C_BIOMAF),
                     ("sigmaf", "Mutational spectra + Bio MAF (1,130); Kucab: combination (3,241)", C_SIGMAF)],
            "Figure 5. Gene features versus patterns", "figure5",
            [("biomaf", "spectra", "Bio MAF - spectra"),
             ("sigmaf", "spectra", "Spectra + Bio MAF - spectra"),
             ("sigmaf", "biomaf", "Spectra + Bio MAF - Bio MAF")], "abcd")


# ---------------------------------------------------------------- Figure 6
def fig6():
    fig, axes = plt.subplots(2, 1, figsize=(W170, 5.6), sharex=True)
    fig.subplots_adjust(left=0.285, right=0.975, top=0.93, bottom=0.075, hspace=0.62)
    for ax, key, title, lab in [(axes[0], "cancer", "Attribution by annotation block: cancer type", "a"),
                                (axes[1], "os", "Attribution by annotation block: overall survival", "b")]:
        items = sorted(D["fig6"][key].items(), key=lambda kv: kv[1])  # ascending for barh
        names = [k for k, _ in items]; vals = [v for _, v in items]
        ax.barh(range(len(vals)), vals, color=C_SIGMAF, edgecolor="white", linewidth=0.3, height=0.68)
        for i, v in enumerate(vals):
            ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=7.2, color=C_INK)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.2)
        ax.set_xlim(0, 27.5)
        ax.set_xlabel("Share of total |SHAP| attribution (%)")
        ax.set_title(title, fontsize=8.5, pad=6)
        ax.tick_params(length=3)
        panel_label(ax, lab, yoff=13)
    export(fig, "figure6", alignment=dict())


# ---------------------------------------------------------------- Figure 7
def fig7():
    f = D["fig7"]
    fig = plt.figure(figsize=(W170, 3.35))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], left=0.165, right=0.985,
                          top=0.86, bottom=0.13, wspace=0.18, hspace=0.75)
    rows = [("HRD score (Spearman rho, n=772)", [0.776503, *f["sigmaf_hrd"][1:]],
             (f["muat_hrd_mean"], *f["muat_hrd_ci"]), (-f["diff_hrd"], -f["diff_hrd_ci"][1], -f["diff_hrd_ci"][0])),
            ("HRD33 high/low (AUROC, n=772)", [0.911064, *f["sigmaf_hrd33"][1:]],
             (f["muat_hrd33_mean"], *f["muat_hrd33_ci"]), (-f["diff_hrd33"], -f["diff_hrd33_ci"][1], -f["diff_hrd33_ci"][0]))]
    flat = []
    for ri, (label, sig, muat, diff) in enumerate(rows):
        ax = fig.add_subplot(gs[ri, 0]); axd = fig.add_subplot(gs[ri, 1]); flat += [ax, axd]
        vals = [sig[0], sig[0], muat[0]]
        cols = ["#4D4D4D", C_SIGMAF, C_MUAT]
        labels = ["Best tabular", "Sig + MAF default", "MuAt-compatible"]
        ax.barh([2, 1, 0], vals, color=cols, edgecolor="white", linewidth=0.3, height=0.58)
        for yy, (m, lo, hi) in zip([2, 1], [(sig[0], sig[1], sig[2])] * 2):
            ax.errorbar(m, yy, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=C_INK,
                        elinewidth=0.8, capsize=2.2, capthick=0.8)
        ax.errorbar(muat[0], 0, xerr=[[muat[0] - muat[1]], [muat[2] - muat[0]]], fmt="none",
                    ecolor=C_INK, elinewidth=0.8, capsize=2.2, capthick=0.8)
        for i, v in enumerate(vals):
            hi = [sig[2], sig[2], muat[2]][i]
            ax.text(hi + 0.018, 2 - i, f"{v:.3f}", va="center", fontsize=7.3, color=C_INK)
        ax.set_yticks([2, 1, 0]); ax.set_yticklabels(labels, fontsize=7.4)
        ax.set_xlim(0, 1.06); ax.set_ylim(-0.55, 2.55)
        ax.set_title(f"{label} — OOF primary metric", fontsize=8.5, pad=5)
        ax.tick_params(length=3)
        dv, lo, hi = diff
        axd.errorbar([dv], [0.5], xerr=[[dv - lo], [hi - dv]], fmt="o", ms=5, color=C_DOWN,
                     ecolor=C_DOWN, elinewidth=1.0, capsize=3, capthick=1.0)
        axd.axvline(0, color="#9a9a9a", lw=0.8)
        axd.set_yticks([]); axd.set_ylim(0, 1)
        axd.text(dv, 0.13, f"{dv:+.3f}", ha="center", fontsize=7.6, color=C_DOWN)
        axd.set_xlim(min(lo * 1.3, -0.06), 0.07)
        axd.set_title("MuAt-compatible minus best tabular\n(95% CI)", fontsize=8.5, pad=5)
        axd.tick_params(length=3)
        panel_label(ax, "ac"[ri], yoff=22); panel_label(axd, "bd"[ri], xoff=-4, yoff=22)
    export(fig, "figure7", alignment=dict(
        axes=flat, panel_ids=["a", "b", "c", "d"],
        row_groups=[["a", "b"], ["c", "d"]], column_groups=[["a", "c"], ["b", "d"]],
        exemptions=[{"panels": ["a", "b", "c", "d"], "checks": ["panel-width"],
                     "reason": "bars:forest width ratio 1.5:1.0 is the design; the difference readout is intentionally narrower"}]))


# ---------------------------------------------------------------- Figure 8
def fig8():
    f = D["fig8"]
    fig = plt.figure(figsize=(W170, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], left=0.20, right=0.985,
                          top=0.88, bottom=0.20, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0]); axd = fig.add_subplot(gs[0, 1])
    vals = f["bars"]; cols = [C_SIGMAF, C_BIOMAF, "#B64342"]
    order = [0, 1, 2]  # combined, Bio MAF only, cancer type only (top to bottom)
    ax.barh([2, 1, 0], [vals[i] for i in order], color=cols, edgecolor="white", linewidth=0.3, height=0.55)
    for i, v in zip(order, [vals[i] for i in order]):
        ax.text(v + 0.007, 2 - order.index(i), f"{v:.3f}", va="center", fontsize=8,
                fontweight="bold", color=C_INK)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([f["labels"][i] for i in order], fontsize=7.8)
    ax.set_xlim(0.5, 0.815)
    ax.set_xlabel("Harrell C-index (pooled out-of-fold)")
    ax.tick_params(length=3)
    panel_label(ax, "a")
    for i, dd in enumerate(f["diffs"]):
        y = 1 - i
        col = "#B64342" if i == 0 else "#8a8a8a"
        axd.errorbar([dd["d"]], [y], xerr=[[dd["d"] - dd["lo"]], [dd["hi"] - dd["d"]]],
                     fmt="o", ms=6, color=col, ecolor=col, elinewidth=1.0, capsize=3, capthick=1.0)
    axd.axvline(0, color=C_SIGMAF, ls="--", lw=0.9)
    axd.set_yticks([1, 0]); axd.set_yticklabels([dd["label"] for dd in f["diffs"]], fontsize=7.6)
    axd.set_xlim(-0.128, 0.022); axd.set_ylim(-0.55, 1.55)
    axd.set_xlabel("Delta C-index (95% paired bootstrap CI)")
    axd.tick_params(length=3)
    panel_label(axd, "b", xoff=-4)
    export(fig, "figure8", alignment=dict(
        axes=[ax, axd], panel_ids=["a", "b"], row_groups=[["a", "b"]], column_groups=[],
        exemptions=[{"panels": ["a", "b"], "checks": ["panel-width"],
                     "reason": "bars:forest width ratio 1.25:1.0 is the design"}]))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "7": fig7, "8": fig8}
    for k, fn in fns.items():
        if which in ("all", k):
            fn()
