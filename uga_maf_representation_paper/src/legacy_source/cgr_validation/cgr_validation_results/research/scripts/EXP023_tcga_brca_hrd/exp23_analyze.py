from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from exp23_config import WorkflowConfig
from exp23_modeling import get_main_models, get_minimal_sensitivity_models, intersect_cohort_with_exposures, load_exposure_frames, run_nested_cv_suite
from exp23_stats import paired_wilcoxon_tests, prediction_tests
from exp23_utils import df_to_md_table, ensure_stage_dirs, write_json


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _save_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="	", index=False)


def _run_suite(
    cfg: WorkflowConfig,
    cohort: pd.DataFrame,
    exposures: dict[str, pd.DataFrame],
    *,
    suite: str,
    subset: str,
    subset_df: pd.DataFrame,
    regression_endpoints: list[str],
    classification_endpoints: list[str],
    models,
) -> dict[str, pd.DataFrame]:
    metrics = []
    fold_metrics = []
    predictions = []
    coefficients = []
    studies = []

    for endpoint in regression_endpoints:
        result = run_nested_cv_suite(
            cfg,
            subset_df,
            exposures,
            suite=suite,
            subset=subset,
            endpoint=endpoint,
            task="regression",
            model_specs=models,
        )
        metrics.append(result["metrics"])
        fold_metrics.append(result["fold_metrics"])
        predictions.append(result["predictions"])
        coefficients.append(result["coefficients"])
        studies.append(result["studies"])

    for endpoint in classification_endpoints:
        result = run_nested_cv_suite(
            cfg,
            subset_df,
            exposures,
            suite=suite,
            subset=subset,
            endpoint=endpoint,
            task="classification",
            model_specs=models,
        )
        metrics.append(result["metrics"])
        fold_metrics.append(result["fold_metrics"])
        predictions.append(result["predictions"])
        coefficients.append(result["coefficients"])
        studies.append(result["studies"])

    return {
        "metrics": _concat_frames(metrics),
        "fold_metrics": _concat_frames(fold_metrics),
        "predictions": _concat_frames(predictions),
        "coefficients": _concat_frames(coefficients),
        "studies": _concat_frames(studies),
    }


def _assign_burden_quartile(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        out["burden_quartile"] = pd.qcut(out["total_sbs"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    except (ValueError, TypeError):
        out["burden_quartile"] = np.nan
    return out


def _paper_sensitivity_specs(final_cohort: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    specs: list[tuple[str, pd.DataFrame]] = []
    q25 = final_cohort["total_sbs"].quantile(0.25)
    subset_df = final_cohort[final_cohort["total_sbs"] >= q25].copy()
    if len(subset_df) >= 30:
        specs.append((f"exclude_low_burden_q25_{q25:.0f}", subset_df))
    if "burden_quartile" not in final_cohort.columns:
        return specs
    ext = final_cohort[final_cohort["burden_quartile"].isin(["Q1", "Q4"])].copy()
    if len(ext) >= 30:
        specs.append(("burden_quartile_extremes_Q1_Q4", ext))
    q1 = final_cohort[final_cohort["burden_quartile"] == "Q1"].copy()
    if len(q1) >= 25:
        specs.append(("burden_Q1", q1))
    mid = final_cohort[final_cohort["burden_quartile"].isin(["Q2", "Q3"])].copy()
    if len(mid) >= 30:
        specs.append(("burden_Q2_Q3_mid", mid))
    q4 = final_cohort[final_cohort["burden_quartile"] == "Q4"].copy()
    if len(q4) >= 25:
        specs.append(("burden_Q4", q4))
    return specs


def _paper_filter_endpoints(cohort: pd.DataFrame, regression: list[str], classification: list[str]) -> tuple[list[str], list[str]]:
    min_n = 40
    reg_out = [e for e in regression if e in cohort.columns and cohort[e].notna().sum() >= min_n]
    clf_out = []
    for e in classification:
        if e not in cohort.columns:
            continue
        sub = cohort[cohort[e].isin(["PARPi-high", "PARPi-low", "HRD-high", "HRD-low"])].copy()
        if len(sub) < min_n:
            continue
        vc = sub[e].value_counts()
        if len(vc) < 2 or vc.min() < 12:
            continue
        clf_out.append(e)
    return reg_out, clf_out


def _filter_binary_classification_columns(
    cohort: pd.DataFrame,
    columns: list[str],
    *,
    min_n: int = 40,
    min_class: int = 12,
) -> list[str]:
    """Keep binary label columns with enough HRD-high / HRD-low (or PARPi) balance."""
    out: list[str] = []
    for e in columns:
        if e not in cohort.columns:
            continue
        sub = cohort[cohort[e].isin(["PARPi-high", "PARPi-low", "HRD-high", "HRD-low"])].copy()
        if len(sub) < min_n:
            continue
        vc = sub[e].value_counts()
        if len(vc) < 2 or vc.min() < min_class:
            continue
        out.append(e)
    return out


def _export_paper_per_endpoint(cfg: WorkflowConfig, main: dict[str, pd.DataFrame]) -> None:
    md = cfg.modeling_dir
    met = main["metrics"]
    fold = main["fold_metrics"]
    pred = main["predictions"]
    stud = main["studies"]
    for ep in met.loc[met["suite"] == "main", "endpoint"].unique():
        sub_m = met[(met["suite"] == "main") & (met["endpoint"] == ep)].copy()
        task = sub_m["task"].iloc[0] if len(sub_m) else ""
        if task == "regression":
            _save_tsv(sub_m, md / f"continuous_results_{ep}.tsv")
            _save_tsv(fold[(fold["suite"] == "main") & (fold["endpoint"] == ep)], md / f"per_fold_results_{ep}.tsv")
            _save_tsv(pred[(pred["suite"] == "main") & (pred["endpoint"] == ep)], md / f"cv_predictions_{ep}.tsv")
            _save_tsv(stud[(stud["suite"] == "main") & (stud["endpoint"] == ep)], md / f"optuna_tuning_summary_{ep}.tsv")
        elif task == "classification":
            _save_tsv(sub_m, md / f"binary_results_{ep}.tsv")
            _save_tsv(fold[(fold["suite"] == "main") & (fold["endpoint"] == ep)], md / f"per_fold_results_{ep}.tsv")
            _save_tsv(pred[(pred["suite"] == "main") & (pred["endpoint"] == ep)], md / f"cv_predictions_{ep}.tsv")
            _save_tsv(stud[(stud["suite"] == "main") & (stud["endpoint"] == ep)], md / f"optuna_tuning_summary_{ep}.tsv")


def _paper_endpoint_markdown(cfg: WorkflowConfig, main: dict[str, pd.DataFrame], all_stats: pd.DataFrame) -> None:
    std, univ = "Baseline+Standard", f"Baseline+Universal-BiCGR d={cfg.universal_depth}"
    rep = cfg.reports_dir
    rep.mkdir(parents=True, exist_ok=True)
    met = main["metrics"]
    for ep in met.loc[met["suite"] == "main", "endpoint"].unique():
        sub = met[(met["suite"] == "main") & (met["endpoint"] == ep)]
        task = sub["task"].iloc[0]
        lines = [f"# {ep} (main cohort, nested CV)", ""]
        if task == "regression":
            cols = [c for c in ("model", "n", "spearman", "pearson", "mae", "r2") if c in sub.columns]
            tab = sub[sub["model"].isin([std, univ])][cols].copy().reset_index(drop=True)
            lines.extend(["## Models", "", df_to_md_table(tab), ""])
            st = all_stats[(all_stats["test"] == "steiger") & (all_stats["endpoint"] == ep) & (all_stats["subset"] == "all")]
            if not st.empty:
                st_cols = [c for c in ("test", "metric", "estimate_a", "estimate_b", "delta", "p_value", "ci_low", "ci_high") if c in st.columns]
                lines.extend(["## Standard vs Universal (Steiger)", "", df_to_md_table(st[st_cols].reset_index(drop=True)), ""])
        else:
            cols = [c for c in ("model", "n", "auroc", "auprc", "balanced_acc", "brier") if c in sub.columns]
            tab = sub[sub["model"].isin([std, univ])][cols].copy().reset_index(drop=True)
            lines.extend(["## Models", "", df_to_md_table(tab), ""])
            if not all_stats.empty and "test" in all_stats.columns:
                dl = all_stats[(all_stats["test"] == "delong") & (all_stats["endpoint"] == ep) & (all_stats["subset"] == "all")]
            else:
                dl = pd.DataFrame()
            if not dl.empty:
                dl_cols = [c for c in ("test", "metric", "estimate_a", "estimate_b", "delta", "p_value", "ci_low", "ci_high") if c in dl.columns]
                lines.extend(["## Standard vs Universal (DeLong)", "", df_to_md_table(dl[dl_cols].reset_index(drop=True)), ""])
        (rep / f"summary_{ep}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_paper_markdown_tables(
    cfg: WorkflowConfig,
    main: dict[str, pd.DataFrame],
    all_metrics: pd.DataFrame,
    all_stats: pd.DataFrame,
) -> None:
    rep = cfg.reports_dir
    rep.mkdir(parents=True, exist_ok=True)
    cross = _paper_cross_endpoint_table(cfg, main, all_stats)
    burden = _paper_burden_sensitivity_table(cfg, all_metrics, all_stats)
    (rep / "cross_endpoint_summary.md").write_text(
        "# Cross-endpoint summary\n\n" + df_to_md_table(cross),
        encoding="utf-8",
    )
    std, univ = "Baseline+Standard", f"Baseline+Universal-BiCGR d={cfg.universal_depth}"
    thr_met = all_metrics[
        (all_metrics["suite"] == "main")
        & (all_metrics["endpoint"].astype(str).str.startswith("hrd_binary"))
        & (all_metrics["model"].isin([std, univ]))
    ].copy()
    if not thr_met.empty:
        cols = [c for c in ("endpoint", "task", "model", "n", "auroc", "auprc") if c in thr_met.columns]
        (rep / "hrd_binary_threshold_summary.md").write_text(
            "# HRD binary thresholds (full cohort)\n\n"
            "Clinical **HRD-high / HRD-low** labels at fixed **HRD_Score** cutoffs (42, 33, 24).\n\n"
            + df_to_md_table(thr_met[cols].sort_values(["endpoint", "model"]).reset_index(drop=True)),
            encoding="utf-8",
        )
    (rep / "burden_sensitivity_summary.md").write_text(
        "# Burden sensitivity (Standard vs Universal)\n\n" + df_to_md_table(burden),
        encoding="utf-8",
    )
    stat_cols = [
        c
        for c in (
            "suite",
            "subset",
            "endpoint",
            "task",
            "test",
            "metric",
            "model_a",
            "model_b",
            "estimate_a",
            "estimate_b",
            "delta",
            "p_value",
        )
        if c in all_stats.columns
    ]
    main_tests = all_stats[
        (all_stats["subset"] == "all") & (all_stats["test"].isin(["steiger", "delong"]))
    ][stat_cols]
    sens_tests = all_stats[
        (all_stats["suite"] == "sensitivity")
        & (all_stats["test"] == "wilcoxon")
        & (all_stats["metric"].isin(["spearman", "auroc"]))
    ][stat_cols]
    parts = ["# Statistical tests", "", "## Main cohort (Steiger / DeLong)", "", df_to_md_table(main_tests.reset_index(drop=True))]
    if not sens_tests.empty:
        parts.extend(["", "## Burden sensitivity (Wilcoxon on fold metrics)", "", df_to_md_table(sens_tests.reset_index(drop=True))])
    (rep / "statistical_tests_summary.md").write_text("\n".join(parts), encoding="utf-8")


def _paper_cross_endpoint_table(cfg: WorkflowConfig, main: dict[str, pd.DataFrame], all_stats: pd.DataFrame) -> pd.DataFrame:
    std, univ = "Baseline+Standard", f"Baseline+Universal-BiCGR d={cfg.universal_depth}"
    met = main["metrics"]
    rows = []
    for ep in met.loc[met["suite"] == "main", "endpoint"].unique():
        sub = met[(met["suite"] == "main") & (met["endpoint"] == ep)]
        task = sub["task"].iloc[0]
        s = sub[sub["model"] == std].iloc[0]
        u = sub[sub["model"] == univ].iloc[0]
        n = int(s["n"])
        if task == "regression":
            key = "spearman"
            s_val, u_val = float(s[key]), float(u[key])
            st = all_stats[(all_stats["test"] == "steiger") & (all_stats["endpoint"] == ep) & (all_stats["subset"] == "all")]
            p = float(st.iloc[0]["p_value"]) if not st.empty else np.nan
        else:
            key = "auroc"
            s_val, u_val = float(s[key]), float(u[key])
            st = all_stats[(all_stats["test"] == "delong") & (all_stats["endpoint"] == ep) & (all_stats["subset"] == "all")]
            p = float(st.iloc[0]["p_value"]) if not st.empty else np.nan
        rows.append(
            {
                "endpoint": ep,
                "task": task,
                "n": n,
                "metric": key,
                "Standard": s_val,
                "Universal": u_val,
                "delta": u_val - s_val,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)


def _paper_burden_sensitivity_table(cfg: WorkflowConfig, all_metrics: pd.DataFrame, all_stats: pd.DataFrame) -> pd.DataFrame:
    std, univ = "Baseline+Standard", f"Baseline+Universal-BiCGR d={cfg.universal_depth}"
    sens = all_metrics[all_metrics["suite"] == "sensitivity"].copy()
    rows = []
    for (subset, endpoint, task), grp in sens.groupby(["subset", "endpoint", "task"]):
        a = grp[grp["model"] == std]
        b = grp[grp["model"] == univ]
        if a.empty or b.empty:
            continue
        srow, urow = a.iloc[0], b.iloc[0]
        n = int(srow["n"])
        if task == "regression":
            key = "spearman"
            s_val, u_val = float(srow[key]), float(urow[key])
        else:
            key = "auroc"
            s_val, u_val = float(srow[key]), float(urow[key])
        w = all_stats[
            (all_stats["test"] == "wilcoxon")
            & (all_stats["suite"] == "sensitivity")
            & (all_stats["subset"] == subset)
            & (all_stats["endpoint"] == endpoint)
            & (all_stats["metric"] == key)
            & (all_stats["model_a"] == std)
            & (all_stats["model_b"] == univ)
        ]
        p = float(w.iloc[0]["p_value"]) if not w.empty else np.nan
        rows.append(
            {
                "subset": subset,
                "endpoint": endpoint,
                "task": task,
                "n": n,
                f"Standard_{key}": s_val,
                f"Universal_{key}": u_val,
                "delta": u_val - s_val,
                "p_value": p,
            }
        )
    main = all_metrics[(all_metrics["suite"] == "main") & (all_metrics["subset"] == "all")]
    for endpoint, task in main[["endpoint", "task"]].drop_duplicates().values:
        grp = main[(main["endpoint"] == endpoint) & (main["task"] == task)]
        a = grp[grp["model"] == std]
        b = grp[grp["model"] == univ]
        if a.empty or b.empty:
            continue
        srow, urow = a.iloc[0], b.iloc[0]
        n = int(srow["n"])
        if task == "regression":
            key = "spearman"
            s_val, u_val = float(srow[key]), float(urow[key])
            st = all_stats[
                (all_stats["test"] == "steiger")
                & (all_stats["endpoint"] == endpoint)
                & (all_stats["subset"] == "all")
            ]
        else:
            key = "auroc"
            s_val, u_val = float(srow[key]), float(urow[key])
            st = all_stats[
                (all_stats["test"] == "delong")
                & (all_stats["endpoint"] == endpoint)
                & (all_stats["subset"] == "all")
            ]
        p = float(st.iloc[0]["p_value"]) if not st.empty else np.nan
        rows.append(
            {
                "subset": "full_cohort",
                "endpoint": endpoint,
                "task": task,
                "n": n,
                f"Standard_{key}": s_val,
                f"Universal_{key}": u_val,
                "delta": u_val - s_val,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)


def run_analysis(cfg: WorkflowConfig) -> dict:
    ensure_stage_dirs(cfg)
    base_cohort = pd.read_csv(cfg.cohort_dir / "base_analysis_cohort.tsv", sep="\t")
    exposures = load_exposure_frames(cfg)
    final_cohort = intersect_cohort_with_exposures(base_cohort, exposures)
    final_cohort = _assign_burden_quartile(final_cohort)
    _save_tsv(final_cohort, cfg.cohort_dir / "final_analysis_cohort.tsv")

    base_reg = ["HRD_Score", "PARPi7", "eCARD", "HRD_TAI", "HRD_LST", "HRD_LOH"]
    base_clf = ["parpi7_binary"]
    main_reg, main_clf = _paper_filter_endpoints(final_cohort, base_reg, base_clf)
    hrd_binary_cols = [f"hrd_binary_{t}" for t in cfg.hrd_thresholds]
    hrd_binary_ok = _filter_binary_classification_columns(final_cohort, hrd_binary_cols)
    extra_clf = [c for c in hrd_binary_ok if c not in main_clf]
    main_clf_full = [*main_clf, *extra_clf]
    # Burden strata: refit paper endpoints only (same as prior EXP029); HRD binary cutoffs run on full cohort in `main`.
    sens_reg, sens_clf = list(main_reg), list(main_clf)
    sens_models = get_minimal_sensitivity_models(cfg)
    sensitivity_specs = _paper_sensitivity_specs(final_cohort)

    main = _run_suite(
        cfg,
        final_cohort,
        exposures,
        suite="main",
        subset="all",
        subset_df=final_cohort,
        regression_endpoints=main_reg,
        classification_endpoints=main_clf_full,
        models=get_main_models(cfg),
    )

    sensitivity_runs = []
    for subset_name, subset_df in sensitivity_specs:
        sensitivity_runs.append(
            _run_suite(
                cfg,
                final_cohort,
                exposures,
                suite="sensitivity",
                subset=subset_name,
                subset_df=subset_df,
                regression_endpoints=sens_reg,
                classification_endpoints=sens_clf,
                models=sens_models,
            )
        )

    all_metrics = _concat_frames([main["metrics"], *[r["metrics"] for r in sensitivity_runs]])
    all_fold_metrics = _concat_frames([main["fold_metrics"], *[r["fold_metrics"] for r in sensitivity_runs]])
    all_predictions = _concat_frames([main["predictions"], *[r["predictions"] for r in sensitivity_runs]])
    all_coefficients = _concat_frames([main["coefficients"], *[r["coefficients"] for r in sensitivity_runs]])
    all_studies = _concat_frames([main["studies"], *[r["studies"] for r in sensitivity_runs]])

    comparisons = [("Baseline+Standard", f"Baseline+Universal-BiCGR d={cfg.universal_depth}")]
    wilcoxon_df = paired_wilcoxon_tests(all_fold_metrics, comparisons)
    prediction_df = prediction_tests(all_predictions, cfg, comparisons)
    all_stats = _concat_frames([wilcoxon_df, prediction_df])

    _save_tsv(all_metrics, cfg.modeling_dir / "all_metrics.tsv")
    _save_tsv(all_fold_metrics, cfg.modeling_dir / "all_fold_metrics.tsv")
    _save_tsv(all_predictions, cfg.modeling_dir / "all_cv_predictions.tsv")
    _save_tsv(all_coefficients, cfg.modeling_dir / "all_model_coefficients.tsv")
    _save_tsv(all_studies, cfg.modeling_dir / "all_optuna_studies.tsv")
    _save_tsv(all_stats, cfg.modeling_dir / "statistical_tests.tsv")

    main_cont = main["metrics"][(main["metrics"]["suite"] == "main") & (main["metrics"]["task"] == "regression")].copy()
    main_bin = main["metrics"][(main["metrics"]["suite"] == "main") & (main["metrics"]["task"] == "classification")].copy()
    _save_tsv(main_cont, cfg.modeling_dir / "continuous_results.tsv")
    _save_tsv(main_bin, cfg.modeling_dir / "binary_results.tsv")
    _save_tsv(main["fold_metrics"], cfg.modeling_dir / "per_fold_results.tsv")
    _save_tsv(main["predictions"], cfg.modeling_dir / "cv_predictions.tsv")
    _save_tsv(main["coefficients"], cfg.modeling_dir / "model_coefficients.tsv")
    _save_tsv(main["studies"], cfg.modeling_dir / "optuna_tuning_summary.tsv")

    sensitivity_metrics = all_metrics[all_metrics["suite"] == "sensitivity"].copy()
    _save_tsv(sensitivity_metrics, cfg.modeling_dir / "sensitivity_results.tsv")
    subtype_metrics = all_metrics[all_metrics["suite"] == "subtype"].copy()
    _save_tsv(subtype_metrics, cfg.modeling_dir / "stratified_results.tsv")

    manifest = {
        "final_analysis_cohort": int(len(final_cohort)),
        "main_metric_rows": int(len(main["metrics"])),
        "main_prediction_rows": int(len(main["predictions"])),
        "sensitivity_metric_rows": int(len(sensitivity_metrics)),
        "statistical_test_rows": int(len(all_stats)),
        "regression_endpoints": main_reg,
        "classification_endpoints": main_clf_full,
        "hrd_binary_threshold_columns": hrd_binary_ok,
    }
    write_json(cfg.metadata_dir / "analysis_manifest.json", manifest)

    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    _export_paper_per_endpoint(cfg, main)
    _paper_endpoint_markdown(cfg, main, all_stats)
    cross_df = _paper_cross_endpoint_table(cfg, main, all_stats)
    burden_df = _paper_burden_sensitivity_table(cfg, all_metrics, all_stats)
    _save_tsv(cross_df, cfg.tables_dir / "cross_endpoint_summary.tsv")
    _save_tsv(burden_df, cfg.tables_dir / "burden_sensitivity_summary.tsv")
    _write_paper_markdown_tables(cfg, main, all_metrics, all_stats)

    return manifest
