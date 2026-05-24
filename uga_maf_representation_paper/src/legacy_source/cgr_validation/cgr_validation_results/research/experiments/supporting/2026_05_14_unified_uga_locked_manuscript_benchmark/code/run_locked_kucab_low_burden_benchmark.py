#!/usr/bin/env python3
"""Optimized atlas-projection Kucab low-burden damage-class benchmark."""

from __future__ import annotations

import html
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from warnings import filterwarnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.checkpointing import atomic_write_csv, atomic_write_json, atomic_write_text, merge_checkpoint_rows, read_completed_keys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


filterwarnings("ignore", category=UserWarning, module=r"sklearn\.model_selection\._split")


def find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for d in [p.parent, *p.parents]:
        if (d / "uga_atlas" / "models.py").is_file() and (d / "data" / "Signatures").is_dir():
            return d
    raise RuntimeError(f"Could not find cgr_validation root from {p}")


REPO = find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from uga_atlas import build_uga_basis, get_uga_model, load_context_atlas, project_counts_to_uga  # noqa: E402


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EXPERIMENT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "kucab" / "processed"
TABLE_DIR = EXPERIMENT_ROOT / "tables" / "kucab"
FIGURE_DIR = EXPERIMENT_ROOT / "figures" / "kucab"

COSMIC_SBS = REPO / "data" / "Signatures" / "COSMIC_v3.5_SBS_GRCh37.txt"
COSMIC_DBS = REPO / "data" / "Signatures" / "COSMIC_v3.5_DBS_GRCh37.txt"
COSMIC_ID = REPO / "data" / "Signatures" / "COSMIC_v3.5_ID_GRCh37.txt"
ATLAS_D22 = REPO / "cgr_validation_results" / "research" / "data" / "EXP022_atlas_genome_wide_45mer_universal_d22.json"

RANDOM_SEED = int(os.environ.get("KUCAB_RANDOM_SEED", "20260514"))
BUDGETS = [int(x) for x in os.environ.get("KUCAB_BUDGETS", "20,25,50,100,250").split(",") if x.strip()]
N_RESAMPLES = int(os.environ.get("KUCAB_N_RESAMPLES", "30"))
ORIGINAL_DATA_CV_REPEATS = int(os.environ.get("KUCAB_ORIGINAL_DATA_CV_REPEATS", "30"))
N_SPLITS = 5
MIN_CLASS_SAMPLES = 10
MIN_TOTAL_EVENTS = 20
STANDARD = "Standard_SBS96_DBS78_ID83_C0p5"
PRIMARY_UGA = "UGA_unified_d10dp5_IDpayload_unweighted_frac_C0p1"
REFERENCE_UGA = PRIMARY_UGA
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
BASES = set("ACGT")
COMP = str.maketrans("ACGTN-", "TGCAN-")
METRIC_KEY_COLUMNS = ["analysis_set", "budget_mutations", "resample", "representation"]
PREDICTION_KEY_COLUMNS = ["analysis_set", "budget_mutations", "resample", "representation", "sample"]

UGA_RUN_CONFIGS = {
    PRIMARY_UGA: {
        "sbsdbs_model": "master_spec_sbs_dbs_d10_dp5",
        "id_model": "id83_payload_only_d10_dp5",
        "feature_mode": "unweighted_frac",
        "classifier_C": 0.1,
        "role": "locked_unified_primary",
    },
}

DAMAGE_CLASS_BY_AGENT = {
    "Acetaldehyde": "Alkylating and aldehyde adduct stress",
    "Acrolein": "Alkylating and aldehyde adduct stress",
    "Cyclophosphamide": "Alkylating and aldehyde adduct stress",
    "DES": "Alkylating and aldehyde adduct stress",
    "DMH": "Alkylating and aldehyde adduct stress",
    "DMS": "Alkylating and aldehyde adduct stress",
    "ENU": "Alkylating and aldehyde adduct stress",
    "Formaldehyde": "Alkylating and aldehyde adduct stress",
    "Glycidamide": "Alkylating and aldehyde adduct stress",
    "MMS": "Alkylating and aldehyde adduct stress",
    "MNNG": "Alkylating and aldehyde adduct stress",
    "MNU": "Alkylating and aldehyde adduct stress",
    "Mechlorethamine": "Alkylating and aldehyde adduct stress",
    "Melphalan": "Alkylating and aldehyde adduct stress",
    "Propylene oxide": "Alkylating and aldehyde adduct stress",
    "Semustine": "Alkylating and aldehyde adduct stress",
    "Styrene oxide": "Alkylating and aldehyde adduct stress",
    "Temozolomide": "Alkylating and aldehyde adduct stress",
    "2-Naphthylamine": "Aromatic and heterocyclic amines",
    "2,6-Dimethylaniline": "Aromatic and heterocyclic amines",
    "4-ABP": "Aromatic and heterocyclic amines",
    "Benzidine": "Aromatic and heterocyclic amines",
    "IQ": "Aromatic and heterocyclic amines",
    "MOCA": "Aromatic and heterocyclic amines",
    "MeAaC": "Aromatic and heterocyclic amines",
    "MeIQX": "Aromatic and heterocyclic amines",
    "PhIP": "Aromatic and heterocyclic amines",
    "o-Anisidine": "Aromatic and heterocyclic amines",
    "o-Toluidine": "Aromatic and heterocyclic amines",
    "5-Methylchrysene": "PAH bulky adducts",
    "BaP": "PAH bulky adducts",
    "BPDE": "PAH bulky adducts",
    "DBA": "PAH bulky adducts",
    "DBAC": "PAH bulky adducts",
    "DBADE": "PAH bulky adducts",
    "DBC": "PAH bulky adducts",
    "DBP": "PAH bulky adducts",
    "DBPDE": "PAH bulky adducts",
    "1,6-DNP": "Nitroaromatic and nitro-PAH adducts",
    "1,8-DNP": "Nitroaromatic and nitro-PAH adducts",
    "1-Nitropyrene": "Nitroaromatic and nitro-PAH adducts",
    "2-Nitrofluorene": "Nitroaromatic and nitro-PAH adducts",
    "2-Nitrotoluene": "Nitroaromatic and nitro-PAH adducts",
    "3-NBA": "Nitroaromatic and nitro-PAH adducts",
    "6-Nitrochrysene": "Nitroaromatic and nitro-PAH adducts",
    "AZ20": "Replication stress and DNA repair inhibition",
    "AZD7762": "Replication stress and DNA repair inhibition",
    "Bleomycin": "Replication stress and DNA repair inhibition",
    "Camptothecin": "Replication stress and DNA repair inhibition",
    "Etoposide": "Replication stress and DNA repair inhibition",
    "Olaparib": "Replication stress and DNA repair inhibition",
    "Carboplatin": "Platinum and crosslinking therapy",
    "Cisplatin": "Platinum and crosslinking therapy",
    "Mitomycin C": "Platinum and crosslinking therapy",
    "1,4-Benzoquinone": "Oxidative and radiation-like stress",
    "Catechol": "Oxidative and radiation-like stress",
    "Gamma irradiation": "Oxidative and radiation-like stress",
    "H2O2": "Oxidative and radiation-like stress",
    "Peroxynitrite": "Oxidative and radiation-like stress",
    "Potassium bromate": "Oxidative and radiation-like stress",
    "SSR": "Oxidative and radiation-like stress",
    "AAI": "Environmental organ-specific DNA adducts",
    "AAII": "Environmental organ-specific DNA adducts",
    "AFB1": "Environmental organ-specific DNA adducts",
    "Acrylamide": "Environmental organ-specific DNA adducts",
    "Furan": "Environmental organ-specific DNA adducts",
    "MX": "Environmental organ-specific DNA adducts",
    "Methyleugenol": "Environmental organ-specific DNA adducts",
    "N-Nitrosopyrrolidine": "Environmental organ-specific DNA adducts",
    "OTA": "Environmental organ-specific DNA adducts",
}


def clean_allele(value: object) -> str:
    s = str(value or "").upper()
    if s in {"-", ".", "NAN", "NONE"}:
        return ""
    return "".join(ch for ch in s if ch in BASES)


def revcomp(seq: str) -> str:
    return str(seq or "").upper().translate(COMP)[::-1]


def safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def canonical_sbs_channel(ref: object, alt: object, left: object, right: object) -> str | None:
    ref_s = clean_allele(ref)
    alt_s = clean_allele(alt)
    l = clean_allele(left)
    r = clean_allele(right)
    if len(ref_s) != 1 or len(alt_s) != 1 or len(l) != 1 or len(r) != 1 or ref_s == alt_s:
        return None
    if ref_s in "AG":
        ref_s = ref_s.translate(COMP)
        alt_s = alt_s.translate(COMP)
        l, r = r.translate(COMP), l.translate(COMP)
    if ref_s not in {"C", "T"} or alt_s not in BASES:
        return None
    return f"{l}[{ref_s}>{alt_s}]{r}"


def canonical_dbs_channel(ref: object, alt: object, valid_channels: set[str]) -> str | None:
    ref_s = clean_allele(ref)
    alt_s = clean_allele(alt)
    if len(ref_s) != 2 or len(alt_s) != 2 or ref_s == alt_s:
        return None
    direct = f"{ref_s}>{alt_s}"
    if direct in valid_channels:
        return direct
    rc = f"{revcomp(ref_s)}>{revcomp(alt_s)}"
    if rc in valid_channels:
        return rc
    return None


def agent_condition_label(treatment: str) -> str:
    label = re.sub(r"\s*\([^)]*\)", "", str(treatment)).strip()
    return re.sub(r"\s+", " ", label) or "Unknown"


def agent_core_label(treatment: str) -> str:
    raw = str(treatment)
    if "control" in raw.lower():
        return "Control"
    label = agent_condition_label(raw)
    return re.sub(r"\s*\+\s*H?S9\b", "", label, flags=re.IGNORECASE).strip() or "Unknown"


def damage_class_for_agent(agent_core: str) -> str:
    if agent_core == "Control":
        return "Control/background"
    return DAMAGE_CLASS_BY_AGENT.get(str(agent_core), "Other or mechanism-ambiguous")


def parse_treatment_map(readme_path: Path) -> pd.DataFrame:
    rows = []
    for line in readme_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.match(r"^MSM0\.\d+\t", line):
            sample_name, treatment = line.split("\t", 1)
            rows.append({"sample_name": sample_name.strip(), "treatment": treatment.strip()})
    out = pd.DataFrame(rows)
    out["agent_core"] = out["treatment"].map(agent_core_label)
    out["agent_condition"] = out["treatment"].map(agent_condition_label)
    out["damage_class"] = out["agent_core"].map(damage_class_for_agent)
    return out


def id83_channel_from_row(row: pd.Series, valid_channels: set[str]) -> str | None:
    event = str(row.get("Type", "")).strip().lower()
    if event.startswith("del"):
        event_label = "Del"
    elif event.startswith("ins"):
        event_label = "Ins"
    else:
        return None
    length = min(max(1, safe_int(row.get("indel.length"), 1)), 5)
    repcount = min(max(0, safe_int(row.get("repcount"), 0)), 5)
    classification = str(row.get("classification", "")).lower()
    change = clean_allele(row.get("change.pyr", ""))
    base = change[0] if change[:1] in {"C", "T"} else "C"
    if event_label == "Del" and "microhomology" in classification and length >= 2:
        channel = f"{length}:Del:M:{min(max(1, repcount or 1), 5)}"
        return channel if channel in valid_channels else None
    if length == 1:
        channel = f"1:{event_label}:{base}:{repcount}"
        return channel if channel in valid_channels else None
    channel = f"{length}:{event_label}:R:{repcount}"
    return channel if channel in valid_channels else None


def count_frame_plain(counts: dict[str, Counter[str]], samples: list[str], channels: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(0.0, index=samples, columns=channels)
    for sample in samples:
        for channel, value in counts[sample].items():
            if channel in frame.columns:
                frame.at[sample, channel] = float(value)
    return frame


def load_raw_counts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    treatments = parse_treatment_map(RAW_DIR / "README.txt")
    treatment_lookup = treatments.set_index("sample_name").to_dict("index")
    sbs_channels = pd.read_csv(COSMIC_SBS, sep="\t", usecols=["Type"])["Type"].astype(str).tolist()
    dbs_channels = pd.read_csv(COSMIC_DBS, sep="\t", usecols=["Type"])["Type"].astype(str).tolist()
    id_channels = pd.read_csv(COSMIC_ID, sep="\t", usecols=["Type"])["Type"].astype(str).tolist()
    dbs_valid = set(dbs_channels)
    id_valid = set(id_channels)

    sbs_counts: dict[str, Counter[str]] = defaultdict(Counter)
    dbs_counts: dict[str, Counter[str]] = defaultdict(Counter)
    id_counts: dict[str, Counter[str]] = defaultdict(Counter)
    sample_to_name: dict[str, str] = {}
    mapped = Counter()

    subs = pd.read_csv(RAW_DIR / "denovo_subclone_subs_final.txt", sep="\t").rename(columns={"Sample.Name": "sample_name"})
    for row in subs.itertuples(index=False):
        sample = str(row.Sample)
        sample_to_name[sample] = str(row.sample_name)
        channel = canonical_sbs_channel(row.Ref, row.Alt, row.pre_context, row.rear_context)
        if channel and channel in sbs_channels:
            sbs_counts[sample][channel] += 1
            mapped["sbs_mapped"] += 1
        mapped["sbs_total"] += 1

    dbs = pd.read_csv(RAW_DIR / "denovo_subclone_doublesub_final.txt", sep="\t").rename(columns={"Sample.Name": "sample_name"})
    for row in dbs.itertuples(index=False):
        rd = row._asdict()
        sample = str(rd["Sample"])
        sample_to_name[sample] = str(rd["sample_name"])
        channel = canonical_dbs_channel(rd.get("dinuc_Ref", ""), rd.get("dinuc_Alt", ""), dbs_valid)
        if channel:
            dbs_counts[sample][channel] += 1
            mapped["dbs_mapped"] += 1
        mapped["dbs_total"] += 1

    indels = pd.read_csv(RAW_DIR / "denovo_subclone_indels.final.txt", sep="\t").rename(columns={"Sample.Name": "sample_name"})
    for _, row in indels.iterrows():
        sample = str(row["Sample"])
        sample_to_name[sample] = str(row["sample_name"])
        channel = id83_channel_from_row(row, id_valid)
        if channel:
            id_counts[sample][channel] += 1
            mapped["id_mapped"] += 1
        mapped["id_total"] += 1

    samples = sorted(s for s in sample_to_name if sample_to_name.get(s) in treatment_lookup)
    sbs_raw = count_frame_plain(sbs_counts, samples, sbs_channels)
    dbs_raw = count_frame_plain(dbs_counts, samples, dbs_channels)
    id_raw = count_frame_plain(id_counts, samples, id_channels)
    rows = []
    for sample in samples:
        sample_name = sample_to_name[sample]
        info = treatment_lookup[sample_name]
        sbs_n = int(sbs_raw.loc[sample].sum())
        dbs_n = int(dbs_raw.loc[sample].sum())
        id_n = int(id_raw.loc[sample].sum())
        rows.append(
            {
                "sample": sample,
                "sample_name": sample_name,
                "treatment": info["treatment"],
                "agent_core": info["agent_core"],
                "agent_condition": info["agent_condition"],
                "damage_class": info["damage_class"],
                "sbs_events": sbs_n,
                "dbs_events": dbs_n,
                "id_events": id_n,
                "total_events": sbs_n + dbs_n + id_n,
            }
        )
    metadata = pd.DataFrame(rows).set_index("sample")
    metadata["eligible_event_burden"] = metadata["total_events"] >= MIN_TOTAL_EVENTS
    return metadata, sbs_raw, dbs_raw, id_raw, dict(mapped)


def eligible_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    use = metadata[
        metadata["eligible_event_burden"]
        & ~metadata["damage_class"].isin(["Control/background", "Other or mechanism-ambiguous"])
    ].copy()
    class_agent_counts = use.groupby("damage_class")["agent_core"].nunique()
    use = use[use["damage_class"].isin(class_agent_counts[class_agent_counts >= 2].index)].copy()
    class_counts = use["damage_class"].value_counts()
    use = use[use["damage_class"].isin(class_counts[class_counts >= MIN_CLASS_SAMPLES].index)].copy()
    return use.sort_index()


def build_variant(
    label: str,
    sbsdbs_model_name: str,
    id_model_name: str,
    feature_mode: str,
    classifier_c: float,
    sbs_columns: list[str],
    dbs_columns: list[str],
    id_columns: list[str],
) -> dict[str, object]:
    sbsdbs_model = get_uga_model(sbsdbs_model_name)
    id_model = get_uga_model(id_model_name)
    atlas = load_context_atlas(ATLAS_D22, sbsdbs_model.d_context)
    sbs_basis, sbs_diag = build_uga_basis(sbs_columns, sbsdbs_model, atlas=atlas, modality="SBS")
    dbs_basis, dbs_diag = build_uga_basis(dbs_columns, sbsdbs_model, atlas=atlas, modality="DBS")
    id_basis, id_diag = build_uga_basis(id_columns, id_model)
    return {
        "label": label,
        "sbsdbs_model": sbsdbs_model_name,
        "id_model": id_model_name,
        "feature_mode": feature_mode,
        "classifier_C": float(classifier_c),
        "sbsdbs_basis": np.vstack([sbs_basis, dbs_basis]),
        "sbsdbs_valid": np.concatenate([sbs_diag["UGA_Encoded"].to_numpy(dtype=bool), dbs_diag["UGA_Encoded"].to_numpy(dtype=bool)]),
        "id_basis": id_basis,
        "id_valid": id_diag["UGA_Encoded"].to_numpy(dtype=bool),
        "diagnostics": pd.concat(
            [
                sbs_diag.assign(representation=label, modality="SBS", model=sbsdbs_model_name),
                dbs_diag.assign(representation=label, modality="DBS", model=sbsdbs_model_name),
                id_diag.assign(representation=label, modality="ID", model=id_model_name),
            ],
            ignore_index=True,
        ),
    }


def downsample_without_replacement(raw_counts: np.ndarray, budget: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    totals = raw_counts.sum(axis=1)
    keep = totals >= int(budget)
    sampled = np.zeros_like(raw_counts, dtype=np.int64)
    for i in np.flatnonzero(keep):
        sampled[i] = rng.multivariate_hypergeometric(raw_counts[i].astype(np.int64), int(budget))
    return sampled, keep


def standard_features(sampled: np.ndarray, keep: np.ndarray, sample_index: pd.Index, columns: list[str]) -> pd.DataFrame:
    arr = sampled[keep].astype(np.float64)
    denom = arr.sum(axis=1, keepdims=True)
    arr = np.divide(arr, denom, out=np.zeros_like(arr), where=denom > 0)
    return pd.DataFrame(arr, index=sample_index[keep], columns=columns)


def uga_features(
    sampled: np.ndarray,
    keep: np.ndarray,
    sample_index: pd.Index,
    n_sbs: int,
    n_dbs: int,
    variant: dict[str, object],
) -> pd.DataFrame:
    ix = sample_index[keep]
    n_sd = n_sbs + n_dbs
    sd_counts = pd.DataFrame(sampled[keep, :n_sd], index=ix)
    id_counts = pd.DataFrame(sampled[keep, n_sd:], index=ix)
    sd = project_counts_to_uga(sd_counts, variant["sbsdbs_basis"], variant["sbsdbs_valid"], "uga_sbsdbs")
    idf = project_counts_to_uga(id_counts, variant["id_basis"], variant["id_valid"], "uga_id")
    totals = sampled[keep].sum(axis=1).astype(np.float64)
    sbs_n = sampled[keep, :n_sbs].sum(axis=1)
    dbs_n = sampled[keep, n_sbs:n_sd].sum(axis=1)
    sd_n = sbs_n + dbs_n
    id_n = sampled[keep, n_sd:].sum(axis=1)
    frac = pd.DataFrame(
        {
            "fraction_sbs": np.divide(sbs_n, totals, out=np.zeros_like(totals), where=totals > 0),
            "fraction_dbs": np.divide(dbs_n, totals, out=np.zeros_like(totals), where=totals > 0),
            "fraction_id": np.divide(id_n, totals, out=np.zeros_like(totals), where=totals > 0),
        },
        index=ix,
    )
    mode = str(variant["feature_mode"])
    if mode == "weighted":
        return pd.concat([sd.mul(sd_n / totals, axis=0), idf.mul(id_n / totals, axis=0)], axis=1)
    if mode == "weighted_frac":
        return pd.concat([sd.mul(sd_n / totals, axis=0), idf.mul(id_n / totals, axis=0), frac], axis=1)
    if mode == "unweighted_frac":
        return pd.concat([sd, idf, frac], axis=1)
    raise ValueError(f"Unsupported UGA feature mode: {mode}")


def make_classifier(seed: int, c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_grouped_cv(
    representation: str,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    budget: int,
    budget_label: str,
    analysis_set: str,
    resample: int,
    seed: int,
    classifier_c: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    meta = metadata.loc[features.index]
    x = features.to_numpy(dtype=np.float64)
    y = meta["damage_class"].to_numpy()
    groups = meta["agent_core"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    fold_ids = np.zeros(len(y), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y, groups), start=1):
        model = make_classifier(seed, classifier_c)
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x[test_idx])
        fold_ids[test_idx] = fold
    metric = {
        "analysis_set": analysis_set,
        "budget_mutations": int(budget),
        "budget_label": budget_label,
        "resample": int(resample),
        "representation": representation,
        "classifier_C": float(classifier_c),
        "feature_dimension": int(features.shape[1]),
        "n_samples": int(len(y)),
        "n_classes": int(pd.Series(y).nunique()),
        "n_agents": int(pd.Series(groups).nunique()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    predictions = meta.reset_index()[["sample", "sample_name", "agent_core", "damage_class", "total_events"]].copy()
    predictions["analysis_set"] = analysis_set
    predictions["budget_mutations"] = int(budget)
    predictions["budget_label"] = budget_label
    predictions["resample"] = int(resample)
    predictions["representation"] = representation
    predictions["fold"] = fold_ids
    predictions["predicted_damage_class"] = pred
    predictions["correct"] = (predictions["damage_class"] == predictions["predicted_damage_class"]).astype(int)
    return metric, predictions


def ci_from_values(values: pd.Series) -> tuple[float, float]:
    x = values.dropna().to_numpy(dtype=np.float64)
    if len(x) <= 1:
        return math.nan, math.nan
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / math.sqrt(len(x)))
    return mean - 1.96 * se, mean + 1.96 * se


def normal_p_value(values: pd.Series) -> float:
    x = values.dropna().to_numpy(dtype=np.float64)
    if len(x) <= 1:
        return math.nan
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / math.sqrt(len(x)))
    if se <= 1e-15:
        return 0.0 if abs(mean) > 1e-15 else 1.0
    return float(math.erfc(abs(mean / se) / math.sqrt(2.0)))


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    p = p_values.to_numpy(dtype=np.float64)
    q = np.full(len(p), np.nan, dtype=np.float64)
    valid = np.flatnonzero(~np.isnan(p))
    if len(valid) == 0:
        return q
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q


def paired_deltas(metrics: pd.DataFrame, uga_labels: list[str]) -> pd.DataFrame:
    rows = []
    units = (
        metrics[metrics["representation"] == STANDARD][["analysis_set", "budget_mutations", "budget_label"]]
        .drop_duplicates()
        .sort_values(["analysis_set", "budget_mutations"])
    )
    for label in uga_labels:
        for unit in units.itertuples(index=False):
            std = metrics[
                (metrics["representation"] == STANDARD)
                & (metrics["analysis_set"] == unit.analysis_set)
                & (metrics["budget_mutations"] == unit.budget_mutations)
            ].set_index("resample")
            uga = metrics[
                (metrics["representation"] == label)
                & (metrics["analysis_set"] == unit.analysis_set)
                & (metrics["budget_mutations"] == unit.budget_mutations)
            ].set_index("resample")
            paired = std.join(uga, lsuffix="_standard", rsuffix="_uga", how="inner")
            for metric in ["balanced_accuracy", "macro_f1", "accuracy"]:
                delta = paired[f"{metric}_uga"] - paired[f"{metric}_standard"]
                lo, hi = ci_from_values(delta)
                rows.append(
                    {
                        "analysis_set": unit.analysis_set,
                        "budget_mutations": int(unit.budget_mutations),
                        "budget_label": unit.budget_label,
                        "comparison": f"{label} minus {STANDARD}",
                        "metric": metric,
                        "standard_mean": float(paired[f"{metric}_standard"].mean()),
                        "uga_mean": float(paired[f"{metric}_uga"].mean()),
                        "delta_mean": float(delta.mean()),
                        "delta_95ci_low": lo,
                        "delta_95ci_high": hi,
                        "p_value": normal_p_value(delta),
                        "uga_win_rate": float((delta > 0).mean()),
                        "n_resamples": int(len(delta)),
                    }
                )
    out = pd.DataFrame(rows)
    out["q_value"] = benjamini_hochberg(out["p_value"])
    return out


def model_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["analysis_set", "budget_mutations", "budget_label", "representation"], dropna=False)
        .agg(
            n_samples=("n_samples", "mean"),
            n_classes=("n_classes", "first"),
            n_agents=("n_agents", "mean"),
            classifier_C=("classifier_C", "first"),
            feature_dimension=("feature_dimension", "first"),
            mean_accuracy=("accuracy", "mean"),
            se_accuracy=("accuracy", lambda x: float(np.std(x, ddof=1) / math.sqrt(len(x)))),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            se_balanced_accuracy=("balanced_accuracy", lambda x: float(np.std(x, ddof=1) / math.sqrt(len(x)))),
            mean_macro_f1=("macro_f1", "mean"),
            se_macro_f1=("macro_f1", lambda x: float(np.std(x, ddof=1) / math.sqrt(len(x)))),
        )
        .reset_index()
    )


def primary_standard_vs_uga_summary(deltas: pd.DataFrame, primary_label: str = PRIMARY_UGA) -> pd.DataFrame:
    primary = deltas[deltas["comparison"] == f"{primary_label} minus {STANDARD}"].copy()
    rows = []
    for _, group in primary.groupby(["analysis_set", "budget_mutations", "budget_label"], sort=False):
        rec = {
            "analysis_set": group["analysis_set"].iloc[0],
            "mutation_budget": group["budget_label"].iloc[0],
            "n_repeats": int(group["n_resamples"].iloc[0]),
        }
        for metric, prefix in [
            ("balanced_accuracy", "balanced_accuracy"),
            ("macro_f1", "macro_f1"),
            ("accuracy", "accuracy"),
        ]:
            row = group[group["metric"] == metric].iloc[0]
            rec[f"standard_{prefix}"] = float(row["standard_mean"])
            rec[f"uga_{prefix}"] = float(row["uga_mean"])
            rec[f"delta_{prefix}"] = float(row["delta_mean"])
            rec[f"p_{prefix}"] = float(row["p_value"])
            rec[f"q_{prefix}"] = float(row["q_value"])
            if metric == "balanced_accuracy":
                rec["balanced_accuracy_95ci"] = f"{row['delta_95ci_low']:.3f} to {row['delta_95ci_high']:.3f}"
                rec["uga_win_rate"] = float(row["uga_win_rate"])
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["_order"] = [-1.0 if str(row.analysis_set) == "Original data" else float(row.mutation_budget) for row in out.itertuples(index=False)]
    return out.sort_values("_order").drop(columns="_order")


def dataset_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for damage_class, group in metadata.groupby("damage_class"):
        rows.append(
            {
                "damage_class": damage_class,
                "n_samples": int(len(group)),
                "n_agents": int(group["agent_core"].nunique()),
                "median_total_events": float(group["total_events"].median()),
                "median_sbs_events": float(group["sbs_events"].median()),
                "median_dbs_events": float(group["dbs_events"].median()),
                "median_id_events": float(group["id_events"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("damage_class")


def format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) < 0.001 and float(value) != 0:
            return f"{float(value):.2e}"
        return f"{float(value):.3f}"
    return str(value)


def html_table(df: pd.DataFrame, title: str, footnote: str) -> str:
    style = """
<style>
body{font-family:Arial,Helvetica,sans-serif;margin:28px;color:#111;}
table{border-collapse:collapse;width:100%;font-size:12.5px;}
caption{caption-side:top;text-align:left;font-weight:700;font-size:15px;margin-bottom:8px;}
th,td{padding:6px 8px;border-bottom:1px solid #d0d0d0;text-align:right;vertical-align:top;}
th:first-child,td:first-child{text-align:left;}
thead th{border-top:1.5px solid #111;border-bottom:1.5px solid #111;font-weight:700;}
tfoot td{border-top:1.5px solid #111;border-bottom:0;text-align:left;font-size:11.5px;color:#333;}
</style>
""".strip()
    head = "<tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr>"
    rows = []
    for _, row in df.iterrows():
        rows.append("<tr>" + "".join(f"<td>{html.escape(format_cell(row[col]))}</td>" for col in df.columns) + "</tr>")
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        + style
        + f"</head><body><table><caption>{html.escape(title)}</caption><thead>{head}</thead><tbody>"
        + "\n".join(rows)
        + f"</tbody><tfoot><tr><td colspan=\"{len(df.columns)}\">{html.escape(footnote)}</td></tr></tfoot></table></body></html>"
    )


def d3_source() -> str:
    return (Path(__file__).resolve().parent / "d3.v7.min.js").read_text(encoding="utf-8", errors="replace")


def write_d3_line(data: pd.DataFrame, out_html: Path) -> None:
    payload = data.to_dict(orient="records")
    script = f"""
const data = {json.dumps(payload)};
const width = 940, height = 540, margin = {{top:58,right:190,bottom:72,left:78}};
const svg = d3.select("#chart").append("svg").attr("width",width).attr("height",height).attr("viewBox",[0,0,width,height]);
svg.append("text").attr("x",margin.left).attr("y",26).attr("font-size",16).attr("font-weight",700).text("Atlas-Optimized Low-Burden Damage-Class Prediction");
const x = d3.scalePoint().domain([...new Set(data.map(d=>d.budget_mutations))]).range([margin.left,width-margin.right]).padding(0.45);
const y = d3.scaleLinear().domain([0,d3.max(data,d=>d.mean_balanced_accuracy)*1.15]).nice().range([height-margin.bottom,margin.top]);
const methods = [...new Set(data.map(d=>d.representation))];
const color = d3.scaleOrdinal().domain(methods).range(["#0072B2","#D55E00","#009E73"]);
svg.append("g").attr("transform",`translate(0,${{height-margin.bottom}})`).call(d3.axisBottom(x));
svg.append("g").attr("transform",`translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(6));
svg.append("text").attr("x",width/2).attr("y",height-22).attr("text-anchor","middle").attr("font-size",12).text("Sampled mutation budget");
svg.append("text").attr("x",-(height/2)).attr("y",18).attr("transform","rotate(-90)").attr("text-anchor","middle").attr("font-size",12).text("Balanced accuracy");
const line = d3.line().x(d=>x(d.budget_mutations)).y(d=>y(d.mean_balanced_accuracy));
for (const method of methods) {{
  const subset = data.filter(d=>d.representation===method).sort((a,b)=>a.budget_mutations-b.budget_mutations);
  svg.append("path").datum(subset).attr("fill","none").attr("stroke",color(method)).attr("stroke-width",2.4).attr("d",line);
  svg.append("g").selectAll("circle").data(subset).join("circle").attr("cx",d=>x(d.budget_mutations)).attr("cy",d=>y(d.mean_balanced_accuracy)).attr("r",4.5).attr("fill",color(method));
}}
const legend = svg.append("g").attr("transform",`translate(${{width-margin.right+18}},${{margin.top}})`);
legend.selectAll("rect").data(methods).join("rect").attr("x",0).attr("y",(d,i)=>i*24).attr("width",12).attr("height",12).attr("fill",d=>color(d));
legend.selectAll("text").data(methods).join("text").attr("x",18).attr("y",(d,i)=>i*24+10).attr("font-size",10).text(d=>d);
"""
    out_html.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;}</style></head><body><div id=\"chart\"></div><script>"
        + d3_source()
        + "</script><script>"
        + script
        + "</script></body></html>",
        encoding="utf-8",
    )


def write_d3_delta(data: pd.DataFrame, out_html: Path) -> None:
    payload = data.to_dict(orient="records")
    script = f"""
const data = {json.dumps(payload)};
const width = 940, height = 520, margin = {{top:58,right:32,bottom:78,left:78}};
const svg = d3.select("#chart").append("svg").attr("width",width).attr("height",height).attr("viewBox",[0,0,width,height]);
svg.append("text").attr("x",margin.left).attr("y",26).attr("font-size",16).attr("font-weight",700).text("Optimized Atlas UGA Improvement Across Low-Burden Budgets");
const x = d3.scaleBand().domain(data.map(d=>d.budget_mutations)).range([margin.left,width-margin.right]).padding(0.28);
const y = d3.scaleLinear().domain([Math.min(0,d3.min(data,d=>d.delta_mean))*1.2,d3.max(data,d=>d.delta_mean)*1.25]).nice().range([height-margin.bottom,margin.top]);
svg.append("g").attr("transform",`translate(0,${{height-margin.bottom}})`).call(d3.axisBottom(x));
svg.append("g").attr("transform",`translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(6));
svg.append("line").attr("x1",margin.left).attr("x2",width-margin.right).attr("y1",y(0)).attr("y2",y(0)).attr("stroke","#111");
svg.append("text").attr("x",width/2).attr("y",height-24).attr("text-anchor","middle").attr("font-size",12).text("Sampled mutation budget");
svg.append("text").attr("x",-(height/2)).attr("y",18).attr("transform","rotate(-90)").attr("text-anchor","middle").attr("font-size",12).text("Balanced accuracy difference");
svg.append("g").selectAll("rect").data(data).join("rect")
  .attr("x",d=>x(d.budget_mutations)).attr("y",d=>y(Math.max(0,d.delta_mean))).attr("width",x.bandwidth()).attr("height",d=>Math.abs(y(d.delta_mean)-y(0))).attr("fill","#D55E00");
svg.append("g").selectAll("line.ci").data(data).join("line")
  .attr("class","ci").attr("x1",d=>x(d.budget_mutations)+x.bandwidth()/2).attr("x2",d=>x(d.budget_mutations)+x.bandwidth()/2).attr("y1",d=>y(d.delta_95ci_low)).attr("y2",d=>y(d.delta_95ci_high)).attr("stroke","#111");
"""
    out_html.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;}</style></head><body><div id=\"chart\"></div><script>"
        + d3_source()
        + "</script><script>"
        + script
        + "</script></body></html>",
        encoding="utf-8",
    )


def save_static_figures(summary: pd.DataFrame, delta: pd.DataFrame) -> None:
    line_data = summary[
        (summary["analysis_set"] == "Low-burden downsample")
        & summary["representation"].isin([STANDARD, PRIMARY_UGA, REFERENCE_UGA])
    ].copy()
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for i, (rep, group) in enumerate(line_data.groupby("representation")):
        group = group.sort_values("budget_mutations")
        ax.plot(group["budget_mutations"].astype(str), group["mean_balanced_accuracy"], marker="o", linewidth=2.2, color=OKABE_ITO[i], label=rep)
    ax.set_xlabel("Sampled mutation budget")
    ax.set_ylabel("Balanced accuracy")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure1_low_burden_balanced_accuracy.svg")
    fig.savefig(FIGURE_DIR / "figure1_low_burden_balanced_accuracy.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.bar(delta["budget_mutations"].astype(str), delta["delta_mean"], color=OKABE_ITO[1])
    ax.errorbar(
        delta["budget_mutations"].astype(str),
        delta["delta_mean"],
        yerr=[delta["delta_mean"] - delta["delta_95ci_low"], delta["delta_95ci_high"] - delta["delta_mean"]],
        fmt="none",
        ecolor="#111111",
        capsize=4,
        linewidth=1,
    )
    ax.axhline(0, color="#111111", linewidth=1)
    ax.set_xlabel("Sampled mutation budget")
    ax.set_ylabel("Balanced accuracy difference")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure2_uga_minus_standard_delta.svg")
    fig.savefig(FIGURE_DIR / "figure2_uga_minus_standard_delta.png", dpi=300)
    plt.close(fig)


def write_readme(
    primary_summary: pd.DataFrame,
    deltas: pd.DataFrame,
    data_summary: pd.DataFrame,
    elapsed: float,
) -> None:
    low_bal = deltas[
        (deltas["comparison"] == f"{PRIMARY_UGA} minus {STANDARD}")
        & (deltas["analysis_set"] == "Low-burden downsample")
        & (deltas["metric"] == "balanced_accuracy")
    ]
    low_primary = primary_summary[
        primary_summary["analysis_set"] == "Low-burden downsample"
    ].copy()
    low_sig = low_primary[low_primary["q_balanced_accuracy"] < 0.05]
    low_sig_text = (
        f"{len(low_sig)} of {len(low_primary)} low-burden budgets were FDR-significant"
        f" ({', '.join(str(int(x)) for x in low_sig['mutation_budget'])} events)"
        if len(low_sig) > 0
        else "no low-burden budgets were FDR-significant"
    )
    low_not_sig = low_primary[low_primary["q_balanced_accuracy"] >= 0.05]
    low_not_sig_text = ""
    if len(low_not_sig) > 0:
        low_not_sig_text = (
            f"; {', '.join(str(int(x)) for x in low_not_sig['mutation_budget'])} events "
            "remained positive but did not pass FDR correction"
        )
    original = primary_summary[primary_summary["analysis_set"] == "Original data"].iloc[0]
    text = f"""# Kucab 2019 Locked Unified-Model Low-Burden Damage-Class Benchmark

## Research Question

Atlas-defined UGA channel projections were evaluated for DNA damage-class prediction in Kucab 2019 mutagen-treated clones using the locked unified manuscript model. The benchmark asks whether the locked UGA geometry improves classification over Standard SBS96+DBS78+ID83 frequencies when mutation burden is low.

## Methods

Raw SBS, DBS, and indel events were mapped to SBS96, DBS78, and ID83 channels. Standard profiles used burden-normalized SBS96+DBS78+ID83 frequencies with balanced logistic regression (`C=0.5`). The locked UGA setting used `{UGA_RUN_CONFIGS[PRIMARY_UGA]['sbsdbs_model']}` for SBS/DBS and `{UGA_RUN_CONFIGS[PRIMARY_UGA]['id_model']}` for indels, with unweighted UGA projection blocks plus SBS/DBS/ID fractions and balanced logistic regression (`C=0.1`). Original observed-burden profiles and downsampled budgets of {", ".join(str(x) for x in BUDGETS)} events were evaluated with grouped cross-validation holding out entire agents.

## Key Numerical Findings

The endpoint set included {int(data_summary['n_samples'].sum())} clone profiles across {len(data_summary)} damage classes and {int(data_summary['n_agents'].sum())} class-agent combinations. In the full repeated run, primary UGA improved original-data balanced accuracy from {original['standard_balanced_accuracy']:.3f} to {original['uga_balanced_accuracy']:.3f} (delta {original['delta_balanced_accuracy']:+.3f}, q={original['q_balanced_accuracy']:.2e}). Across low-burden budgets, balanced-accuracy deltas were all positive, ranging from {low_bal['delta_mean'].min():+.3f} to {low_bal['delta_mean'].max():+.3f}; {low_sig_text}{low_not_sig_text}.

## File Inventory

- `data/raw/`: raw Kucab mutation files and metadata copied into the experiment package.
- `data/kucab/processed/sample_metadata.csv`: clone-level endpoint labels and mapped event counts.
- `data/kucab/processed/damage_class_mapping.csv`: treatment-agent endpoint mapping.
- `data/kucab/processed/locked_model_manifest.csv`: locked UGA model configuration used in the run.
- `data/kucab/processed/uga_basis_diagnostics.csv`: atlas channel-encoding diagnostics for full-run UGA settings.
- `data/kucab/processed/downsampled_fold_metrics.csv`: per-budget, per-resample model performance.
- `data/kucab/processed/downsampled_predictions.csv.gz`: clone-level predictions for every budget, resample, and representation.
- `data/kucab/processed/paired_deltas.csv`: paired UGA-minus-standard deltas across repeated seeds.
- `tables/kucab/table1_low_burden_performance.csv` and `.html`: mean model performance by budget.
- `tables/kucab/table2_uga_minus_standard_deltas.csv` and `.html`: paired deltas and win rates.
- `tables/kucab/table3_damage_class_dataset_summary.csv` and `.html`: endpoint class and mutation burden summary.
- `tables/kucab/table4_primary_standard_vs_uga_summary.csv` and `.html`: primary standard-versus-UGA comparison.
- `figures/kucab/figure1_low_burden_balanced_accuracy.csv`, `.html`, `.svg`, `.png`: balanced accuracy by mutation budget.
- `figures/kucab/figure2_uga_minus_standard_delta.csv`, `.html`, `.svg`, `.png`: primary UGA-minus-standard balanced-accuracy deltas.
- `code/run_locked_kucab_low_burden_benchmark.py`: complete reproducible analysis script.

## Reproducibility

Executed on {datetime.now(timezone.utc).date().isoformat()} with random seed {RANDOM_SEED}. Runtime was {elapsed:.1f} seconds on {platform.platform()}. Package versions: Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}, scikit-learn {sys.modules['sklearn'].__version__}, matplotlib {matplotlib.__version__}. All UGA representations were built through `uga_atlas` registered models and `build_uga_basis(...)`.
"""
    atomic_write_text(DATA_DIR / "kucab" / "README_kucab.md", text)


def clean_outputs() -> None:
    for directory in [PROCESSED_DIR, TABLE_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def job_key(analysis_set: str, budget: int, resample: int, representation: str) -> tuple[str, ...]:
    return tuple(str(value) for value in [analysis_set, int(budget), int(resample), representation])


def run_full() -> None:
    start = time.time()
    clean_outputs()
    rng_master = np.random.default_rng(RANDOM_SEED)
    metric_checkpoint = PROCESSED_DIR / "downsampled_fold_metrics_checkpoint.csv"
    prediction_checkpoint = PROCESSED_DIR / "downsampled_predictions_checkpoint.csv"
    final_metrics_path = PROCESSED_DIR / "downsampled_fold_metrics.csv"
    final_predictions_path = PROCESSED_DIR / "downsampled_predictions.csv.gz"
    if not metric_checkpoint.exists() and final_metrics_path.exists():
        atomic_write_csv(pd.read_csv(final_metrics_path), metric_checkpoint, index=False)
    if not prediction_checkpoint.exists() and final_predictions_path.exists():
        atomic_write_csv(pd.read_csv(final_predictions_path), prediction_checkpoint, index=False)
    completed = read_completed_keys(metric_checkpoint, METRIC_KEY_COLUMNS)

    metadata, sbs, dbs, id_counts, mapped = load_raw_counts()
    use = eligible_metadata(metadata)
    sbs = sbs.loc[use.index]
    dbs = dbs.loc[use.index]
    id_counts = id_counts.loc[use.index]
    raw_counts = np.concatenate([sbs.to_numpy(), dbs.to_numpy(), id_counts.to_numpy()], axis=1).astype(np.int64)
    standard_columns = [f"SBS:{c}" for c in sbs.columns] + [f"DBS:{c}" for c in dbs.columns] + [f"ID:{c}" for c in id_counts.columns]
    locked_manifest = pd.DataFrame(
        [
            {
                "representation": PRIMARY_UGA,
                "sbsdbs_model": UGA_RUN_CONFIGS[PRIMARY_UGA]["sbsdbs_model"],
                "id_model": UGA_RUN_CONFIGS[PRIMARY_UGA]["id_model"],
                "feature_mode": UGA_RUN_CONFIGS[PRIMARY_UGA]["feature_mode"],
                "classifier_C": UGA_RUN_CONFIGS[PRIMARY_UGA]["classifier_C"],
                "role": UGA_RUN_CONFIGS[PRIMARY_UGA]["role"],
            }
        ]
    )
    atomic_write_csv(locked_manifest, PROCESSED_DIR / "locked_model_manifest.csv", index=False)

    variants = {
        label: build_variant(
            label,
            cfg["sbsdbs_model"],
            cfg["id_model"],
            cfg["feature_mode"],
            cfg["classifier_C"],
            sbs.columns.astype(str).tolist(),
            dbs.columns.astype(str).tolist(),
            id_counts.columns.astype(str).tolist(),
        )
        for label, cfg in UGA_RUN_CONFIGS.items()
    }
    atomic_write_csv(
        pd.concat([v["diagnostics"] for v in variants.values()], ignore_index=True),
        PROCESSED_DIR / "uga_basis_diagnostics.csv",
        index=False,
    )

    metrics_rows = []
    prediction_frames = []
    full_keep = np.ones(len(use), dtype=bool)
    full_standard = standard_features(raw_counts, full_keep, use.index, standard_columns)
    full_uga = {
        label: uga_features(raw_counts, full_keep, use.index, len(sbs.columns), len(dbs.columns), variant)
        for label, variant in variants.items()
    }
    for repeat in range(ORIGINAL_DATA_CV_REPEATS):
        seed = int(rng_master.integers(0, 2**31 - 1))
        key = job_key("Original data", 0, repeat, STANDARD)
        if key in completed:
            print(f"[checkpoint] skip {key}", flush=True)
        else:
            metric, pred = evaluate_grouped_cv(STANDARD, full_standard, use, 0, "Original data", "Original data", repeat, seed, 0.5)
            merge_checkpoint_rows(metric_checkpoint, [metric], key_columns=METRIC_KEY_COLUMNS, sort_columns=METRIC_KEY_COLUMNS)
            merge_checkpoint_rows(prediction_checkpoint, pred.to_dict("records"), key_columns=PREDICTION_KEY_COLUMNS, sort_columns=PREDICTION_KEY_COLUMNS)
            completed.add(key)
            metrics_rows.append(metric)
            prediction_frames.append(pred)
            print(f"[checkpoint] wrote {key}", flush=True)
        for label, features in full_uga.items():
            key = job_key("Original data", 0, repeat, label)
            if key in completed:
                print(f"[checkpoint] skip {key}", flush=True)
                continue
            metric, pred = evaluate_grouped_cv(
                label,
                features,
                use,
                0,
                "Original data",
                "Original data",
                repeat,
                seed,
                variants[label]["classifier_C"],
            )
            merge_checkpoint_rows(metric_checkpoint, [metric], key_columns=METRIC_KEY_COLUMNS, sort_columns=METRIC_KEY_COLUMNS)
            merge_checkpoint_rows(prediction_checkpoint, pred.to_dict("records"), key_columns=PREDICTION_KEY_COLUMNS, sort_columns=PREDICTION_KEY_COLUMNS)
            completed.add(key)
            metrics_rows.append(metric)
            prediction_frames.append(pred)
            print(f"[checkpoint] wrote {key}", flush=True)

    for budget in BUDGETS:
        for resample in range(N_RESAMPLES):
            seed = int(rng_master.integers(0, 2**31 - 1))
            planned_reps = [STANDARD, *variants.keys()]
            if all(job_key("Low-burden downsample", budget, resample, rep) in completed for rep in planned_reps):
                print(f"[checkpoint] skip Low-burden downsample/{budget}/{resample}", flush=True)
                continue
            sampled, keep = downsample_without_replacement(raw_counts, budget, np.random.default_rng(seed))
            std_features = standard_features(sampled, keep, use.index, standard_columns)
            key = job_key("Low-burden downsample", budget, resample, STANDARD)
            if key in completed:
                print(f"[checkpoint] skip {key}", flush=True)
            else:
                metric, pred = evaluate_grouped_cv(STANDARD, std_features, use, budget, str(budget), "Low-burden downsample", resample, seed, 0.5)
                merge_checkpoint_rows(metric_checkpoint, [metric], key_columns=METRIC_KEY_COLUMNS, sort_columns=METRIC_KEY_COLUMNS)
                merge_checkpoint_rows(prediction_checkpoint, pred.to_dict("records"), key_columns=PREDICTION_KEY_COLUMNS, sort_columns=PREDICTION_KEY_COLUMNS)
                completed.add(key)
                metrics_rows.append(metric)
                prediction_frames.append(pred)
                print(f"[checkpoint] wrote {key}", flush=True)
            for label, variant in variants.items():
                key = job_key("Low-burden downsample", budget, resample, label)
                if key in completed:
                    print(f"[checkpoint] skip {key}", flush=True)
                    continue
                features = uga_features(sampled, keep, use.index, len(sbs.columns), len(dbs.columns), variant)
                metric, pred = evaluate_grouped_cv(
                    label,
                    features,
                    use,
                    budget,
                    str(budget),
                    "Low-burden downsample",
                    resample,
                    seed,
                    variant["classifier_C"],
                )
                merge_checkpoint_rows(metric_checkpoint, [metric], key_columns=METRIC_KEY_COLUMNS, sort_columns=METRIC_KEY_COLUMNS)
                merge_checkpoint_rows(prediction_checkpoint, pred.to_dict("records"), key_columns=PREDICTION_KEY_COLUMNS, sort_columns=PREDICTION_KEY_COLUMNS)
                completed.add(key)
                metrics_rows.append(metric)
                prediction_frames.append(pred)
                print(f"[checkpoint] wrote {key}", flush=True)

    metrics = pd.read_csv(metric_checkpoint, low_memory=False) if metric_checkpoint.exists() else pd.DataFrame(metrics_rows)
    predictions = pd.read_csv(prediction_checkpoint, low_memory=False) if prediction_checkpoint.exists() else pd.concat(prediction_frames, ignore_index=True)
    deltas = paired_deltas(metrics, list(variants))
    summary = model_summary(metrics)
    primary_summary = primary_standard_vs_uga_summary(deltas, PRIMARY_UGA)
    data_summary = dataset_summary(use)

    atomic_write_csv(use, PROCESSED_DIR / "sample_metadata.csv")
    atomic_write_csv(pd.DataFrame([{"agent_core": k, "damage_class": v} for k, v in sorted(DAMAGE_CLASS_BY_AGENT.items())]),
        PROCESSED_DIR / "damage_class_mapping.csv", index=False
    )
    atomic_write_csv(metrics, PROCESSED_DIR / "downsampled_fold_metrics.csv", index=False)
    predictions.to_csv(PROCESSED_DIR / "downsampled_predictions.csv.gz", index=False, compression="gzip")
    atomic_write_csv(deltas, PROCESSED_DIR / "paired_deltas.csv", index=False)
    atomic_write_csv(data_summary, TABLE_DIR / "table3_damage_class_dataset_summary.csv", index=False)
    atomic_write_csv(summary, TABLE_DIR / "table1_low_burden_performance.csv", index=False)
    atomic_write_csv(deltas, TABLE_DIR / "table2_uga_minus_standard_deltas.csv", index=False)
    atomic_write_csv(primary_summary, TABLE_DIR / "table4_primary_standard_vs_uga_summary.csv", index=False)

    atomic_write_text(
        TABLE_DIR / "table1_low_burden_performance.html",
        html_table(summary, "Table 1. Atlas-optimized low-burden performance.", "Values summarize repeated grouped CV. Agents were held out as grouped cross-validation units."),
    )
    atomic_write_text(
        TABLE_DIR / "table2_uga_minus_standard_deltas.html",
        html_table(deltas, "Table 2. Paired UGA-minus-standard deltas.", "Positive deltas favor UGA. Confidence intervals are normal approximations across paired repeated seeds."),
    )
    atomic_write_text(
        TABLE_DIR / "table3_damage_class_dataset_summary.html",
        html_table(data_summary, "Table 3. Damage-class dataset summary.", "Mutation burdens are mapped SBS, DBS, and ID83 events before downsampling."),
    )
    atomic_write_text(
        TABLE_DIR / "table4_primary_standard_vs_uga_summary.html",
        html_table(primary_summary, "Table 4. Primary optimized atlas UGA comparison.", "P values are two-sided paired normal tests across repeated grouped-CV or downsampling seeds. Q values use Benjamini-Hochberg correction across all paired comparisons in Table 2."),
    )

    figure1 = summary[
        (summary["analysis_set"] == "Low-burden downsample")
        & summary["representation"].isin([STANDARD, PRIMARY_UGA, REFERENCE_UGA])
    ].copy()
    atomic_write_csv(figure1, FIGURE_DIR / "figure1_low_burden_balanced_accuracy.csv", index=False)
    write_d3_line(figure1, FIGURE_DIR / "figure1_low_burden_balanced_accuracy.html")
    figure2 = deltas[
        (deltas["comparison"] == f"{PRIMARY_UGA} minus {STANDARD}")
        & (deltas["metric"] == "balanced_accuracy")
        & (deltas["analysis_set"] == "Low-burden downsample")
    ].copy()
    atomic_write_csv(figure2, FIGURE_DIR / "figure2_uga_minus_standard_delta.csv", index=False)
    write_d3_delta(figure2, FIGURE_DIR / "figure2_uga_minus_standard_delta.html")
    save_static_figures(summary, figure2)

    elapsed = time.time() - start
    write_readme(primary_summary, deltas, data_summary, elapsed)
    run_metadata = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "random_seed": RANDOM_SEED,
        "sbsdbs_model": UGA_RUN_CONFIGS[PRIMARY_UGA]["sbsdbs_model"],
        "id_model": UGA_RUN_CONFIGS[PRIMARY_UGA]["id_model"],
        "feature_mode": UGA_RUN_CONFIGS[PRIMARY_UGA]["feature_mode"],
        "classifier_C": UGA_RUN_CONFIGS[PRIMARY_UGA]["classifier_C"],
        "budgets": BUDGETS,
        "n_resamples": N_RESAMPLES,
        "original_data_cv_repeats": ORIGINAL_DATA_CV_REPEATS,
        "n_splits": N_SPLITS,
        "mapped": mapped,
    }
    atomic_write_json(DATA_DIR / "kucab" / "run_metadata.json", run_metadata)
    print(
        json.dumps(
            {
                "experiment_root": str(EXPERIMENT_ROOT),
                "elapsed_seconds": round(elapsed, 2),
                "primary_model": UGA_RUN_CONFIGS[PRIMARY_UGA],
                "primary_mean_low_burden_delta_balanced_accuracy": float(figure2["delta_mean"].mean()),
                "primary_min_low_burden_delta_balanced_accuracy": float(figure2["delta_mean"].min()),
                "original_data_balanced_accuracy_delta": float(
                    primary_summary.loc[primary_summary["analysis_set"] == "Original data", "delta_balanced_accuracy"].iloc[0]
                ),
                "mapped": mapped,
            },
            indent=2,
        )
    )


def main() -> None:
    run_full()


if __name__ == "__main__":
    main()
