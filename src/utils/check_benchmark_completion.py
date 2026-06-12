"""Completion gate for the optimized manuscript benchmark run."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import BUNDLE_ROOT, load_yaml, resolve_paths_map
from utils.endpoint_registry import load_endpoint_registry, main_kucab_endpoints, main_non_survival_endpoints, main_survival_endpoints, muat_comparator_endpoints


FRESH_SCOPE = "pooled_global_oof"
FRESH_TUNING = "outer_train_inner_validation_randomized_search"
MC3_REPRESENTATIONS = ["burden_only", "standard_sbs96_id83", "MAF_stack_only", "signatures_plus_MAF_stack"]
KUCAB_REPRESENTATIONS = ["burden_only", "standard_sbs96_dbs78_id83", "MAF_stack_only", "signatures_plus_MAF_stack"]
NON_SURVIVAL_LEARNERS = ["linear", "xgboost"]
SURVIVAL_LEARNERS = ["cox_ph"]
CIRCULAR_GENE_ENDPOINTS = {"brca_gene_mutated", "mmr_gene_mutated", "pole_pold1_mutated", "luad_kmt2c_mutated"}


def _tables_dir(paths: dict[str, Any]) -> Path:
    return Path((paths.get("workspace") or {}).get("results_tables_dir") or (BUNDLE_ROOT / "results" / "tables"))


def _figures_dir(paths: dict[str, Any]) -> Path:
    return Path((paths.get("workspace") or {}).get("results_figures_dir") or (BUNDLE_ROOT / "results" / "figures"))


def _manuscript_dir(paths: dict[str, Any]) -> Path:
    return _figures_dir(paths).parent / "manuscript"


def _settings_for(settings: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    return dict(((settings.get("experiments") or {}).get(experiment_id) or {}))


def _registry(settings: dict[str, Any]) -> dict[str, Any]:
    return load_endpoint_registry(settings.get("endpoint_registry"))


def _configured_main_endpoints(settings: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    registry = _registry(settings)
    local = _settings_for(settings, "main_manuscript_complete_panel")
    non_survival = [str(x) for x in local.get("main_mc3_endpoints", main_non_survival_endpoints(registry))]
    survival = [str(x) for x in local.get("survival_endpoints", main_survival_endpoints(registry))]
    kucab = [str(x) for x in local.get("kucab_endpoints", main_kucab_endpoints(registry))]
    return non_survival, survival, kucab


def _configured_muat_endpoints(settings: dict[str, Any]) -> list[str]:
    registry = _registry(settings)
    local = _settings_for(settings, "muat_style_tcga_comparator")
    if local.get("endpoints"):
        return [str(x) for x in local.get("endpoints", [])]
    primary = str(local.get("primary_endpoint") or local.get("primary_task") or "").strip()
    secondary = [str(x) for x in local.get("secondary_endpoints", [])]
    if primary and primary != "tcga_20_type":
        return [primary, *[endpoint for endpoint in secondary if endpoint != primary]]
    return [str(x) for x in muat_comparator_endpoints(registry)]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _muat_output_prefix(settings: dict[str, Any]) -> str:
    local = _settings_for(settings, "muat_style_tcga_comparator")
    tag = str(local.get("output_tag") or "").strip()
    return "muat_style_tcga_comparator" if not tag else f"muat_style_tcga_comparator_{_safe_name(tag)}"


def _strict_no_leakage(settings: dict[str, Any]) -> bool:
    outputs = dict(settings.get("outputs") or {})
    return bool(outputs.get("strict_no_leakage", outputs.get("strict_manuscript", False)))


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _check(statuses: list[dict[str, Any]], name: str, passed: bool, detail: str = "") -> None:
    statuses.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})


def _summary_is_completed(path: Path) -> bool:
    frame = _read_csv(path)
    return bool(not frame.empty and "status" in frame.columns and (frame["status"].astype(str) == "completed").any())


def _included_from_audit(configured: list[str], audit: pd.DataFrame) -> list[str]:
    if audit.empty or "endpoint" not in audit.columns or "status" not in audit.columns:
        return configured
    included = set(audit.loc[audit["status"].astype(str) == "included", "endpoint"].astype(str))
    audited = set(audit["endpoint"].astype(str))
    return [endpoint for endpoint in configured if endpoint not in audited or endpoint in included]


def _fresh_main_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column, fallback in [("primary_metric_scope", ""), ("tuning", ""), ("score", math.nan)]:
        if column not in out.columns:
            out[column] = fallback
    score = pd.to_numeric(out["score"], errors="coerce")
    status_ok = True
    if "status" in out.columns:
        status_ok = out["status"].astype(str).isin(["measured", "completed", "ok", "nan", ""])
    return out[
        (out["primary_metric_scope"].astype(str) == FRESH_SCOPE)
        & (out["tuning"].astype(str) == FRESH_TUNING)
        & score.notna()
        & score.map(math.isfinite)
        & status_ok
    ].copy()


def _missing_slots(frame: pd.DataFrame, slots: set[tuple[str, str, str]]) -> set[tuple[str, str, str]]:
    if frame.empty:
        return set(slots)
    observed = {
        (str(row["endpoint"]), str(row["representation"]), str(row["learner"]))
        for _, row in frame.iterrows()
        if {"endpoint", "representation", "learner"}.issubset(frame.columns)
    }
    return set(slots) - observed


def _expected_main_slots(settings: dict[str, Any], audit: pd.DataFrame) -> set[tuple[str, str, str]]:
    non_survival, survival, kucab = _configured_main_endpoints(settings)
    non_survival = _included_from_audit(non_survival, audit)
    survival = _included_from_audit(survival, audit)
    slots: set[tuple[str, str, str]] = set()
    for endpoint in non_survival:
        for representation in MC3_REPRESENTATIONS:
            for learner in NON_SURVIVAL_LEARNERS:
                slots.add((endpoint, representation, learner))
    for endpoint in survival:
        for representation in MC3_REPRESENTATIONS:
            for learner in SURVIVAL_LEARNERS:
                slots.add((endpoint, representation, learner))
    for endpoint in kucab:
        for representation in KUCAB_REPRESENTATIONS:
            for learner in NON_SURVIVAL_LEARNERS:
                slots.add((endpoint, representation, learner))
    return slots


def _file_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def _active_os_event_hits(tables_dir: Path, figures_dir: Path, manuscript_dir: Path, active_table_prefixes: tuple[str, ...]) -> list[str]:
    roots = [tables_dir, figures_dir, manuscript_dir]
    suffixes = {".csv", ".tsv", ".json", ".md", ".html", ".txt", ".svg"}
    hits: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes and _file_contains(path, "os_event"):
                if "improvement_log" in path.name.lower():
                    continue
                if root == tables_dir:
                    lower_name = path.name.lower()
                    if "checkpoint" in lower_name or "checkpoint" in {part.lower() for part in path.parts}:
                        continue
                    if not any(path.name.startswith(prefix) for prefix in active_table_prefixes):
                        continue
                hits.append(str(path))
    return sorted(hits)


def _leakage_safe_maf_rows(frame: pd.DataFrame, *, strict: bool, settings: dict[str, Any]) -> tuple[bool, str]:
    if not strict or frame.empty:
        return True, "not required"
    if "representation" not in frame.columns:
        return False, "missing representation column"
    maf = frame[frame["representation"].astype(str).isin(["MAF_stack_only", "signatures_plus_MAF_stack"])].copy()
    if maf.empty:
        return False, "no active MAF-stack rows"
    text_cols = [col for col in ["atlas_status_note", "construction", "feature_builder", "leakage_status", "cache_key"] if col in maf.columns]
    text = maf[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower() if text_cols else pd.Series("", index=maf.index)
    forbidden_metadata_pattern = "leg" + "acy|transductive|full-cohort unsupervised|svd|idf|top-variance"
    bad_tokens = text.str.contains(forbidden_metadata_pattern, regex=True, na=False)
    safe_note_pattern = (
        "bio maf v4|"
        "frozen_biological_v4_nested_fs|"
        "nested inner-loop|"
        "feature-block selection|"
        "fixed capped categorical event-token|"
        "kucab event-stack analogue"
    )
    missing_safe_note = ~text.str.contains(safe_note_pattern, regex=True, na=False)
    bad = maf[bad_tokens | missing_safe_note].copy()
    if bad.empty:
        return True, f"{len(maf)} MAF-stack rows have leakage-safe metadata"
    detail_cols = [col for col in ["endpoint", "representation", "learner", "atlas_status_note", "cache_key"] if col in bad.columns]
    return False, json.dumps(bad.loc[:, detail_cols].head(12).to_dict("records"), default=str)


def _muat_strict_metadata_ok(results: pd.DataFrame, expected_endpoints: set[str], *, strict: bool, require_canonical_split: bool = False) -> tuple[bool, str]:
    if not strict:
        return True, "not required"
    if results.empty:
        return False, "missing MuAt-compatible results"
    work = results[results.get("endpoint", pd.Series(dtype=str)).astype(str).isin(expected_endpoints)].copy()
    if work.empty:
        return False, "no expected MuAt-compatible endpoint rows"
    failures: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        endpoint = str(row.get("endpoint", ""))
        n_samples = pd.to_numeric(pd.Series([row.get("n_samples")]), errors="coerce").iloc[0]
        folds = pd.to_numeric(pd.Series([row.get("n_folds", row.get("folds"))]), errors="coerce").iloc[0]
        dictionary_mode = str(row.get("dictionary_mode", "")).strip().lower()
        split_source = str(row.get("canonical_split_source", "")).strip()
        split_signature = str(row.get("canonical_split_signature", "")).strip()
        endpoint_failures = []
        if endpoint == "cancer_type_top20" and (not pd.notna(n_samples) or int(n_samples) != 8800):
            endpoint_failures.append(f"n_samples={n_samples}")
        if endpoint == "cancer_type_top20":
            n_classes = pd.to_numeric(pd.Series([row.get("n_classes")]), errors="coerce").iloc[0]
            balanced_accuracy = pd.to_numeric(pd.Series([row.get("balanced_accuracy")]), errors="coerce").iloc[0]
            top5 = pd.to_numeric(pd.Series([row.get("top5_accuracy")]), errors="coerce").iloc[0]
            paper_accuracy = pd.to_numeric(pd.Series([row.get("paper_reference_accuracy")]), errors="coerce").iloc[0]
            if not pd.notna(n_classes) or int(n_classes) != 20:
                endpoint_failures.append(f"n_classes={n_classes}")
            if not pd.notna(balanced_accuracy):
                endpoint_failures.append("missing balanced_accuracy")
            if not pd.notna(top5):
                endpoint_failures.append("missing top5_accuracy")
            if not pd.notna(paper_accuracy) or abs(float(paper_accuracy) - 0.641) > 1e-9:
                endpoint_failures.append(f"paper_reference_accuracy={paper_accuracy}")
        if not pd.notna(folds) or int(folds) != 5:
            endpoint_failures.append(f"n_folds={folds}")
        if dictionary_mode != "fixed_hash":
            endpoint_failures.append(f"dictionary_mode={dictionary_mode or 'missing'}")
        if (require_canonical_split or endpoint == "cancer_type_top20") and (not split_source or not split_signature):
            endpoint_failures.append("missing canonical split source/signature")
        if endpoint_failures:
            failures.append({"endpoint": endpoint, "failures": endpoint_failures})
    return (not failures), json.dumps(failures[:10], default=str) if failures else "fixed-hash canonical MuAt-compatible metadata present"


def check_completion(settings: dict[str, Any], paths: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    tables_dir = _tables_dir(paths)
    figures_dir = _figures_dir(paths)
    manuscript_dir = _manuscript_dir(paths)
    statuses: list[dict[str, Any]] = []

    main_results_path = tables_dir / "main_manuscript_complete_panel_endpoint_results.csv"
    main_summary_path = tables_dir / "main_manuscript_complete_panel_summary.csv"
    audit_path = tables_dir / "main_manuscript_complete_panel_endpoint_audit.csv"
    main_results = _read_csv(main_results_path)
    audit = _read_csv(audit_path)
    fresh_main = _fresh_main_rows(main_results)
    missing_main = _missing_slots(fresh_main, _expected_main_slots(settings, audit))
    _check(statuses, "main_panel_endpoint_results_present", main_results_path.exists(), str(main_results_path))
    _check(statuses, "main_panel_summary_completed", _summary_is_completed(main_summary_path), str(main_summary_path))
    _check(statuses, "main_panel_fresh_slots_complete", not missing_main, f"{len(missing_main)} missing: {sorted(missing_main)[:12]}")
    circular_hits = sorted(set(fresh_main.get("endpoint", pd.Series(dtype=str)).astype(str)).intersection(CIRCULAR_GENE_ENDPOINTS))
    _check(statuses, "no_circular_gene_endpoints_in_main_panel", not circular_hits, f"circular endpoints present: {circular_hits}")

    survival = set(_configured_main_endpoints(settings)[1])
    audit_survival = set(audit.loc[(audit.get("status", pd.Series(dtype=str)).astype(str) == "included"), "endpoint"].astype(str)) if not audit.empty and "endpoint" in audit.columns and "status" in audit.columns else set()
    _check(statuses, "cox_survival_audit_included", survival.issubset(audit_survival), f"missing included audit rows: {sorted(survival - audit_survival)}")

    muat_prefix = _muat_output_prefix(settings)
    muat_results_path = tables_dir / f"{muat_prefix}_endpoint_results.csv"
    muat_summary_path = tables_dir / f"{muat_prefix}_summary.csv"
    muat_results = _read_csv(muat_results_path)
    muat_expected = set(_configured_muat_endpoints(settings))
    muat_observed = set()
    if not muat_results.empty and {"endpoint", "primary_metric_scope"}.issubset(muat_results.columns):
        muat_observed = set(muat_results.loc[muat_results["primary_metric_scope"].astype(str) == FRESH_SCOPE, "endpoint"].astype(str))
    _check(statuses, "muat_style_results_present", muat_results_path.exists(), str(muat_results_path))
    _check(statuses, "muat_style_summary_completed", _summary_is_completed(muat_summary_path), str(muat_summary_path))
    _check(statuses, "muat_style_endpoints_complete", muat_expected.issubset(muat_observed), f"missing: {sorted(muat_expected - muat_observed)}")
    muat_settings = _settings_for(settings, "muat_style_tcga_comparator")
    muat_meta_ok, muat_meta_detail = _muat_strict_metadata_ok(
        muat_results,
        muat_expected,
        strict=_strict_no_leakage(settings),
        require_canonical_split=bool(muat_settings.get("require_canonical_split", False)),
    )
    _check(statuses, "muat_style_strict_metadata", muat_meta_ok, muat_meta_detail)

    maf_ok, maf_detail = _leakage_safe_maf_rows(fresh_main, strict=_strict_no_leakage(settings), settings=settings)
    _check(statuses, "main_panel_maf_stack_leakage_safe", maf_ok, maf_detail)
    figure_files = []
    if figures_dir.exists():
        figure_files.extend(figures_dir.glob("figure_*"))
    if (manuscript_dir / "figures").exists():
        figure_files.extend((manuscript_dir / "figures").glob("figure_*"))
    _check(statuses, "manuscript_figures_present", bool(figure_files), f"{len(figure_files)} figure files under {figures_dir} and {manuscript_dir / 'figures'}")

    active_prefixes = ["main_manuscript_complete_panel", _muat_output_prefix(settings)]
    os_event_hits = _active_os_event_hits(tables_dir, figures_dir, manuscript_dir, tuple(active_prefixes))
    _check(statuses, "no_active_os_event_outputs", not os_event_hits, f"{len(os_event_hits)} hits: {os_event_hits[:10]}")

    return all(row["status"] == "pass" for row in statuses), statuses


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment_settings.strict_no_leakage.yaml")
    parser.add_argument("--paths", default="config/paths.yaml")
    parser.add_argument("--json", action="store_true", help="Write machine-readable check results.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = load_yaml(args.config)
    paths = resolve_paths_map(load_yaml(args.paths))
    ok, statuses = check_completion(settings, paths)
    if args.json:
        print(json.dumps({"ok": ok, "checks": statuses}, indent=2, sort_keys=True))
    else:
        for row in statuses:
            print(f"[{row['status']}] {row['check']}: {row['detail']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
