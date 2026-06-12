"""Endpoint registry loading and label construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from utils.config import BUNDLE_ROOT, load_yaml


DEFAULT_REGISTRY = BUNDLE_ROOT / "config" / "endpoint_registry.yaml"
HRD_CONTINUOUS = {"HRD_Score", "eCARD", "HRD_TAI", "HRD_LST", "HRD_LOH", "PARPi7"}
HRD_BINARY = {"hrd_binary_24", "hrd_binary_33", "hrd_binary_42", "parpi7_binary"}
TCGA_CANCER_TYPE_TOP20 = [
    "BRCA",
    "UCEC",
    "LUAD",
    "LGG",
    "HNSC",
    "PRAD",
    "THCA",
    "LUSC",
    "SKCM",
    "STAD",
    "BLCA",
    "OV",
    "COAD",
    "GBM",
    "KIRC",
    "LIHC",
    "CESC",
    "KIRP",
    "SARC",
    "ESCA",
]
CANCER_TYPE_ENDPOINT_CLASSES = {
    "cancer_type_top20": TCGA_CANCER_TYPE_TOP20,
}


def load_endpoint_registry(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or DEFAULT_REGISTRY)


def main_non_survival_endpoints(registry: dict[str, Any] | None = None) -> list[str]:
    registry = registry or load_endpoint_registry()
    panel = dict(registry.get("main_manuscript_complete_panel") or {})
    names: list[str] = []
    for key in ("hrd_continuous", "hrd_binary", "mc3_multiclass", "mc3_binary"):
        names.extend(str(value) for value in (panel.get(key) or []))
    return names


def main_survival_endpoints(registry: dict[str, Any] | None = None) -> list[str]:
    registry = registry or load_endpoint_registry()
    return [str(value) for value in ((registry.get("main_manuscript_complete_panel") or {}).get("survival") or [])]


def main_kucab_endpoints(registry: dict[str, Any] | None = None) -> list[str]:
    registry = registry or load_endpoint_registry()
    return [str(value) for value in ((registry.get("main_manuscript_complete_panel") or {}).get("kucab") or ["damage_class"])]


def muat_comparator_endpoints(registry: dict[str, Any] | None = None) -> list[str]:
    registry = registry or load_endpoint_registry()
    return [str(value) for value in ((registry.get("muat_style_tcga_comparator") or {}).get("endpoints") or [])]


def kucab_sparsity_budgets(registry: dict[str, Any] | None = None) -> list[int]:
    registry = registry or load_endpoint_registry()
    return [int(value) for value in ((registry.get("kucab_sparsity_control") or {}).get("budgets") or [20, 41, 100, 200])]


def registry_minimums(registry: dict[str, Any] | None = None) -> dict[str, int]:
    registry = registry or load_endpoint_registry()
    defaults = {
        "binary_min_per_class": 25,
        "multiclass_min_per_class": 50,
        "survival_min_samples": 200,
        "survival_min_events": 25,
    }
    configured = dict(registry.get("minimums") or {})
    return {key: int(configured.get(key, value)) for key, value in defaults.items()}


def build_luad_kmt2c_labels(mc3_source_dir: str | Path, patients: Iterable[str]) -> pd.Series:
    source_dir = Path(mc3_source_dir)
    raw_dir = source_dir / "raw"
    cdr = pd.read_excel(raw_dir / "TCGA-CDR-SupplementalTableS1.xlsx", usecols=["bcr_patient_barcode", "type"])
    cdr["patient"] = cdr["bcr_patient_barcode"].astype(str).str[:12]
    cancer_type = cdr.drop_duplicates("patient").set_index("patient")["type"].astype(str)
    driver = pd.read_csv(source_dir / "driver_gene_labels_functional.csv", index_col=0)
    driver.index = driver.index.astype(str)
    patients = pd.Index([str(value) for value in patients])
    luad = cancer_type.reindex(patients)
    labels = driver.reindex(patients).fillna(0)
    y = labels.loc[luad[luad == "LUAD"].index, "kmt2c_mutated"].astype(int)
    y.name = "luad_kmt2c_mutated"
    return y


def build_cdr_cancer_type_labels(
    mc3_source_dir: str | Path,
    patients: Iterable[str] | None = None,
    classes: Iterable[str] | None = None,
    *,
    endpoint_name: str = "cancer_type",
) -> pd.Series:
    """Build fixed TCGA cancer-type labels from the CDR clinical table."""

    raw_dir = Path(mc3_source_dir) / "raw"
    cdr = pd.read_excel(raw_dir / "TCGA-CDR-SupplementalTableS1.xlsx", usecols=["bcr_patient_barcode", "type"])
    cdr["patient"] = cdr["bcr_patient_barcode"].astype(str).str[:12]
    labels = cdr.drop_duplicates("patient").set_index("patient")["type"].astype(str)
    if patients is not None:
        labels = labels.reindex(pd.Index([str(value) for value in patients]))
    labels = labels.dropna()
    if classes is not None:
        fixed = [str(value) for value in classes]
        labels = labels[labels.isin(fixed)]
    labels.name = endpoint_name
    return labels


def build_cdr_survival_labels(
    mc3_source_dir: str | Path,
    endpoints: Iterable[str],
    *,
    min_samples: int = 200,
    min_events: int = 25,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    raw_dir = Path(mc3_source_dir) / "raw"
    cdr = pd.read_excel(raw_dir / "TCGA-CDR-SupplementalTableS1.xlsx")
    cdr["patient"] = cdr["bcr_patient_barcode"].astype(str).str[:12]
    labels: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, object]] = []
    for endpoint in endpoints:
        event_col = str(endpoint)
        time_col = f"{event_col}.time"
        if event_col not in cdr.columns or time_col not in cdr.columns:
            audit_rows.append({"endpoint": endpoint, "task": "survival", "status": "excluded", "reason": "missing_event_or_time_column"})
            continue
        frame = cdr.loc[:, ["patient", event_col, time_col]].copy()
        frame[event_col] = pd.to_numeric(frame[event_col], errors="coerce")
        frame[time_col] = pd.to_numeric(frame[time_col], errors="coerce")
        frame = frame.dropna(subset=[event_col, time_col])
        frame = frame[frame[time_col] > 0].copy()
        frame["event"] = frame[event_col].astype(int)
        frame["time"] = frame[time_col].astype(float)
        frame = frame.drop_duplicates("patient").set_index("patient")[["time", "event"]]
        n = int(len(frame))
        events = int(frame["event"].sum())
        if n < int(min_samples) or events < int(min_events):
            audit_rows.append({"endpoint": endpoint, "task": "survival", "status": "excluded", "reason": "below_minimum_support", "n_samples": n, "n_events": events})
            continue
        labels[str(endpoint)] = frame
        audit_rows.append({"endpoint": endpoint, "task": "survival", "status": "included", "reason": "", "n_samples": n, "n_events": events})
    return labels, pd.DataFrame(audit_rows)
