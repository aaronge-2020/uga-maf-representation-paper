from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from uga_atlas import get_uga_model

DEFAULT_UNIVERSAL_UGA_MODEL = get_uga_model("compact_sbs_dbs_d10")
DEFAULT_BICGR52_UGA_MODEL = get_uga_model("bicgr52_context_d10")


@dataclass(frozen=True)
class WorkflowConfig:
    repo_root: Path
    data_dir: Path
    assets_dir: Path
    reports_dir: Path
    ddr_path: Path
    clinical_path: Path | None
    maf_path: Path
    cosmic_sbs_path: Path
    cohort_acronym: str = "BRCA"
    """TCGA tumor type acronym in DDRscores.txt (e.g. BRCA, OV)."""
    random_state: int = 42
    outer_splits: int = 5
    inner_splits: int = 3
    regression_trials: int = 40
    classification_trials: int = 40
    optuna_storage: str | None = None
    optuna_trial_jobs: int = 1
    universal_uga_model: str = DEFAULT_UNIVERSAL_UGA_MODEL.name
    universal_depth: int = DEFAULT_UNIVERSAL_UGA_MODEL.d_context
    payload_depth: int = DEFAULT_UNIVERSAL_UGA_MODEL.d_payload
    payload_schema: str = DEFAULT_UNIVERSAL_UGA_MODEL.payload_schema
    bicgr52_uga_model: str = DEFAULT_BICGR52_UGA_MODEL.name
    bicgr52_depth: int = DEFAULT_BICGR52_UGA_MODEL.d_context
    min_burden: int = 1
    hrd_thresholds: tuple[int, ...] = (42, 33, 24)
    bootstrap_iterations: int = 1000
    fasta_walk: bool = True

    @property
    def metadata_dir(self) -> Path:
        return self.assets_dir / "metadata"

    @property
    def catalogs_dir(self) -> Path:
        return self.assets_dir / "catalogs"

    @property
    def labels_dir(self) -> Path:
        return self.assets_dir / "labels"

    @property
    def cohort_dir(self) -> Path:
        return self.assets_dir / "cohort"

    @property
    def exposures_dir(self) -> Path:
        return self.assets_dir / "exposures"

    @property
    def modeling_dir(self) -> Path:
        return self.assets_dir / "modeling_results"

    @property
    def figures_dir(self) -> Path:
        return self.assets_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.assets_dir / "tables"


def _search_repo_root(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if (candidate / "cgr_validation_results").exists():
            return candidate
    return None


def find_repo_root(anchor: Path | None = None) -> Path:
    for env_key in ("EXP023_REPO_ROOT", "EXP027_REPO_ROOT"):
        env_root = os.environ.get(env_key)
        if env_root:
            path = Path(env_root).expanduser().resolve()
            if path.exists():
                return path

    anchor = (anchor or Path(__file__).resolve()).resolve()
    found = _search_repo_root(anchor)
    if found is not None:
        return found

    cwd_found = _search_repo_root(Path.cwd().resolve())
    if cwd_found is not None:
        return cwd_found

    return Path.cwd().resolve()


def _resolve_maf_path(data_dir: Path, cohort_acronym: str, maf_filename: str | None) -> Path:
    if maf_filename:
        p = data_dir / maf_filename
        if not p.exists():
            raise FileNotFoundError(f"MAF not found: {p}")
        return p
    if cohort_acronym != "BRCA":
        for name in (
            "allCohortMAF.2026-04-27.maf",
            "allCohortMAF.2026-04-27.maf.gz",
            "cohortMAF.2026-04-27.maf",
            "cohortMAF.2026-04-27.maf.gz",
        ):
            p = data_dir / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No MAF found under {data_dir} for cohort {cohort_acronym}. "
            "Add allCohortMAF.2026-04-27.maf (or .maf.gz) or pass maf_filename=..."
        )
    for name in (
        "cohortMAF.2026-04-27.maf.gz",
        "cohortMAF.2026-04-27.maf",
        "allCohortMAF.2026-04-27.maf",
        "allCohortMAF.2026-04-27.maf.gz",
    ):
        p = data_dir / name
        if p.exists():
            return p
    return data_dir / "cohortMAF.2026-04-27.maf.gz"


def build_config(
    repo_root: str | Path | None = None,
    *,
    cohort_acronym: str = "BRCA",
    assets_slug: str | None = None,
    reports_slug: str | None = None,
    maf_filename: str | None = None,
    optuna_storage: str | None = None,
    regression_trials: int | None = None,
    classification_trials: int | None = None,
    outer_splits: int | None = None,
    inner_splits: int | None = None,
    fasta_walk: bool = True,
) -> WorkflowConfig:
    root = Path(repo_root).expanduser().resolve() if repo_root else find_repo_root()
    data_dir = root / "cgr_validation_results/research/data/TCGA-BRCA"
    acronym = cohort_acronym.strip().upper()
    slug = f"TCGA-{acronym}"
    slug_a = assets_slug or "EXP023_tcga_brca_hrd"
    slug_r = reports_slug or "EXP023_tcga_brca_hrd"
    default_assets = root / f"cgr_validation_results/research/assets/{slug_a}/{slug}"
    default_reports = root / f"cgr_validation_results/research/reports/{slug_r}/{slug}"
    maf_path = _resolve_maf_path(data_dir, acronym, maf_filename)
    clin = data_dir / "clinical_and_hrd.txt"
    clinical_path = clin if clin.exists() and acronym == "BRCA" else None
    cfg = WorkflowConfig(
        repo_root=root,
        data_dir=data_dir,
        assets_dir=default_assets,
        reports_dir=default_reports,
        ddr_path=data_dir / "DDRscores.txt",
        clinical_path=clinical_path,
        maf_path=maf_path,
        cosmic_sbs_path=root / "data/Signatures/COSMIC_v3.5_SBS_GRCh37.txt",
        cohort_acronym=acronym,
        optuna_storage=optuna_storage,
        fasta_walk=fasta_walk,
    )
    if regression_trials is not None:
        cfg = replace(cfg, regression_trials=regression_trials)
    if classification_trials is not None:
        cfg = replace(cfg, classification_trials=classification_trials)
    if outer_splits is not None:
        cfg = replace(cfg, outer_splits=outer_splits)
    if inner_splits is not None:
        cfg = replace(cfg, inner_splits=inner_splits)
    return cfg
