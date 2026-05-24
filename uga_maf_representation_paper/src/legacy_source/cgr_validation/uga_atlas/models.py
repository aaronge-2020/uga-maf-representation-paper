"""Registry of UGA operating points used by benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class UGAModelSpec:
    name: str
    kinds: tuple[str, ...]
    d_context: int
    d_payload: int
    payload_schema: str = "masked"
    context_source: str = "genome_atlas"
    note: str = ""

    @property
    def is_context_only(self) -> bool:
        return self.context_source == "context_only"

    def with_values(self, **kwargs) -> "UGAModelSpec":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class CompositeUGAModelSpec:
    name: str
    version: str
    sbs_model: str
    dbs_model: str
    id_model: str
    feature_mode: str
    learner_family: str
    feature_transform: str = "mean_coordinate_projection"
    transform_parameters: tuple[tuple[str, str], ...] = ()
    note: str = ""

    def component_names(self) -> dict[str, str]:
        return {"SBS": self.sbs_model, "DBS": self.dbs_model, "ID": self.id_model}

    def components(self) -> dict[str, UGAModelSpec]:
        return {
            modality: get_uga_model(model_name)
            for modality, model_name in self.component_names().items()
        }


MODEL_REGISTRY: dict[str, UGAModelSpec] = {
    "master_spec_sbs_dbs_d22": UGAModelSpec(
        name="master_spec_sbs_dbs_d22",
        kinds=("SBS96", "DBS78"),
        d_context=22,
        d_payload=2,
        payload_schema="masked",
        context_source="genome_atlas",
        note="Master-spec SBS96/DBS78 channel projection using genome-wide context centroids.",
    ),
    "compact_sbs_dbs_d10": UGAModelSpec(
        name="compact_sbs_dbs_d10",
        kinds=("SBS96", "DBS78"),
        d_context=10,
        d_payload=2,
        payload_schema="masked",
        context_source="genome_atlas",
        note="Compact masked SBS96/DBS78 projection for rapid benchmarks.",
    ),
    "master_spec_sbs_dbs_d13": UGAModelSpec(
        name="master_spec_sbs_dbs_d13",
        kinds=("SBS96", "DBS78"),
        d_context=13,
        d_payload=2,
        payload_schema="masked",
        context_source="genome_atlas",
        note="Intermediate-depth masked SBS96/DBS78 channel projection using genome-wide context centroids.",
    ),
    "master_spec_sbs_dbs_d16": UGAModelSpec(
        name="master_spec_sbs_dbs_d16",
        kinds=("SBS96", "DBS78"),
        d_context=16,
        d_payload=2,
        payload_schema="masked",
        context_source="genome_atlas",
        note="Moderate-depth masked SBS96/DBS78 channel projection using genome-wide context centroids.",
    ),
    "master_spec_sbs_dbs_d10_dp5": UGAModelSpec(
        name="master_spec_sbs_dbs_d10_dp5",
        kinds=("SBS96", "DBS78"),
        d_context=10,
        d_payload=5,
        payload_schema="masked",
        context_source="genome_atlas",
        note="Compact context with expanded masked payload depth for SBS96/DBS78 channel projection.",
    ),
    "master_spec_sbs_dbs_d10_dp10": UGAModelSpec(
        name="master_spec_sbs_dbs_d10_dp10",
        kinds=("SBS96", "DBS78"),
        d_context=10,
        d_payload=10,
        payload_schema="masked",
        context_source="genome_atlas",
        note="Compact context with ten-slot masked payload depth for SBS96/DBS78 channel projection.",
    ),
    "master_spec_sbs_dbs_d16_dp5": UGAModelSpec(
        name="master_spec_sbs_dbs_d16_dp5",
        kinds=("SBS96", "DBS78"),
        d_context=16,
        d_payload=5,
        payload_schema="masked",
        context_source="genome_atlas",
        note="Moderate context with expanded masked payload depth for SBS96/DBS78 channel projection.",
    ),
    "master_spec_sbs_dbs_d22_dp5": UGAModelSpec(
        name="master_spec_sbs_dbs_d22_dp5",
        kinds=("SBS96", "DBS78"),
        d_context=22,
        d_payload=5,
        payload_schema="masked",
        context_source="genome_atlas",
        note="Master-spec context with expanded masked payload depth for SBS96/DBS78 channel projection.",
    ),
    "gdsc_sbs_d10_dp10": UGAModelSpec(
        name="gdsc_sbs_d10_dp10",
        kinds=("SBS96",),
        d_context=10,
        d_payload=10,
        payload_schema="masked",
        context_source="genome_atlas",
        note="Legacy EXP026 GDSC SBS96 operating point with payload depth matched to context depth.",
    ),
    "bicgr52_context_d13": UGAModelSpec(
        name="bicgr52_context_d13",
        kinds=("SBS96", "DBS78"),
        d_context=13,
        d_payload=2,
        payload_schema="masked",
        context_source="context_only",
        note="Context-only BiCGR-52 ablation at d_context=13.",
    ),
    "bicgr52_context_d10": UGAModelSpec(
        name="bicgr52_context_d10",
        kinds=("SBS96", "DBS78"),
        d_context=10,
        d_payload=2,
        payload_schema="masked",
        context_source="context_only",
        note="Context-only BiCGR-52 ablation at d_context=10.",
    ),
    "compact_event_legacy_d10": UGAModelSpec(
        name="compact_event_legacy_d10",
        kinds=("SBS96", "SBS1536", "DBS78", "ID83"),
        d_context=10,
        d_payload=2,
        payload_schema="legacy",
        context_source="event",
        note="Legacy 48-feature event-level coordinate used by EXP-024 shared-space simulations.",
    ),
    "islam2022_sbs1536_d10": UGAModelSpec(
        name="islam2022_sbs1536_d10",
        kinds=("SBS1536",),
        d_context=10,
        d_payload=2,
        payload_schema="masked",
        context_source="label",
        note="SBS1536 label projection with observed two-base flanks and masked payload slots.",
    ),
    "label_sbs1536_d22": UGAModelSpec(
        name="label_sbs1536_d22",
        kinds=("SBS1536",),
        d_context=22,
        d_payload=2,
        payload_schema="masked",
        context_source="label",
        note="High-resolution SBS1536 label projection with observed two-base flanks and masked payload slots.",
    ),
    "label_sbs96_d10": UGAModelSpec(
        name="label_sbs96_d10",
        kinds=("SBS96",),
        d_context=10,
        d_payload=2,
        payload_schema="masked",
        context_source="label",
        note="SBS96 label projection using observed trinucleotide labels and masked payload slots.",
    ),
    "label_sbs96_d16": UGAModelSpec(
        name="label_sbs96_d16",
        kinds=("SBS96",),
        d_context=16,
        d_payload=2,
        payload_schema="masked",
        context_source="label",
        note="Moderate-depth SBS96 label projection using observed trinucleotide labels and masked payload slots.",
    ),
    "label_sbs96_d22": UGAModelSpec(
        name="label_sbs96_d22",
        kinds=("SBS96",),
        d_context=22,
        d_payload=2,
        payload_schema="masked",
        context_source="label",
        note="High-resolution SBS96 label projection using observed trinucleotide labels and masked payload slots.",
    ),
    "islam2022_dbs78_d10": UGAModelSpec(
        name="islam2022_dbs78_d10",
        kinds=("DBS78",),
        d_context=10,
        d_payload=2,
        payload_schema="masked",
        context_source="label",
        note="DBS78 compact label projection with masked REF/ALT payload slots.",
    ),
    "label_dbs78_d22": UGAModelSpec(
        name="label_dbs78_d22",
        kinds=("DBS78",),
        d_context=22,
        d_payload=2,
        payload_schema="masked",
        context_source="label",
        note="High-resolution DBS78 label projection with masked REF/ALT payload slots.",
    ),
    "id83_proxy_d10_dp5": UGAModelSpec(
        name="id83_proxy_d10_dp5",
        kinds=("ID83",),
        d_context=10,
        d_payload=5,
        payload_schema="masked",
        context_source="id83_proxy",
        note="ID83 categorical proxy with repeat and microhomology tokens encoded as synthetic local context.",
    ),
    "id83_token_pair_d10_dp5": UGAModelSpec(
        name="id83_token_pair_d10_dp5",
        kinds=("ID83",),
        d_context=10,
        d_payload=5,
        payload_schema="masked",
        context_source="id83_token_pair",
        note="ID83 proxy with deterministic category tokens on both context sides and masked REF/ALT payloads.",
    ),
    "id83_token_pair_d10_dp10": UGAModelSpec(
        name="id83_token_pair_d10_dp10",
        kinds=("ID83",),
        d_context=10,
        d_payload=10,
        payload_schema="masked",
        context_source="id83_token_pair",
        note="ID83 proxy with deterministic category tokens on both context sides and ten-slot masked REF/ALT payloads.",
    ),
    "id83_repeat_context_d10_dp5": UGAModelSpec(
        name="id83_repeat_context_d10_dp5",
        kinds=("ID83",),
        d_context=10,
        d_payload=5,
        payload_schema="masked",
        context_source="id83_repeat_context",
        note="ID83 proxy using repeat or microhomology-like synthetic context on both sides.",
    ),
    "id83_payload_only_d10_dp5": UGAModelSpec(
        name="id83_payload_only_d10_dp5",
        kinds=("ID83",),
        d_context=10,
        d_payload=5,
        payload_schema="masked",
        context_source="id83_payload_only",
        note="ID83 proxy ablation using masked REF/ALT payloads without synthetic context.",
    ),
    "id83_proxy_d22_dp5": UGAModelSpec(
        name="id83_proxy_d22_dp5",
        kinds=("ID83",),
        d_context=22,
        d_payload=5,
        payload_schema="masked",
        context_source="id83_proxy",
        note="D22 ID83 categorical proxy for MC3 direct-feature and biology-prediction benchmarks.",
    ),
    "id83_token_pair_d22_dp5": UGAModelSpec(
        name="id83_token_pair_d22_dp5",
        kinds=("ID83",),
        d_context=22,
        d_payload=5,
        payload_schema="masked",
        context_source="id83_token_pair",
        note="D22 ID83 proxy with deterministic category tokens on both context sides.",
    ),
    "id83_repeat_context_d22_dp5": UGAModelSpec(
        name="id83_repeat_context_d22_dp5",
        kinds=("ID83",),
        d_context=22,
        d_payload=5,
        payload_schema="masked",
        context_source="id83_repeat_context",
        note="D22 ID83 proxy using repeat or microhomology-like synthetic context on both sides.",
    ),
    "id83_payload_only_d22_dp5": UGAModelSpec(
        name="id83_payload_only_d22_dp5",
        kinds=("ID83",),
        d_context=22,
        d_payload=5,
        payload_schema="masked",
        context_source="id83_payload_only",
        note="D22 ID83 proxy ablation using masked REF/ALT payloads without synthetic context.",
    ),
    "observed_context_events_d10_dp10": UGAModelSpec(
        name="observed_context_events_d10_dp10",
        kinds=("SBS", "DBS", "ID"),
        d_context=10,
        d_payload=10,
        payload_schema="masked",
        context_source="observed_context",
        note="Event-level SBS, DBS, and indel encoding from observed GRCh37 FASTA flanks and raw REF/ALT payloads.",
    ),
}


COMPOSITE_MODEL_REGISTRY: dict[str, CompositeUGAModelSpec] = {
    "locked_payload_sbs_dbs_id_d10_v1": CompositeUGAModelSpec(
        name="locked_payload_sbs_dbs_id_d10_v1",
        version="v1",
        sbs_model="master_spec_sbs_dbs_d10_dp5",
        dbs_model="master_spec_sbs_dbs_d10_dp5",
        id_model="id83_payload_only_d10_dp5",
        feature_mode="pooled",
        learner_family="matched per benchmark",
        note=(
            "Locked manuscript composite used by the unified 2026-05-14 benchmark; "
            "ID83 uses blank context with masked REF/ALT payload fields."
        ),
    ),
    "compact_proxy_sbs_dbs_id_d10_v1": CompositeUGAModelSpec(
        name="compact_proxy_sbs_dbs_id_d10_v1",
        version="v1",
        sbs_model="compact_sbs_dbs_d10",
        dbs_model="compact_sbs_dbs_d10",
        id_model="id83_proxy_d10_dp5",
        feature_mode="separate",
        learner_family="elastic-net regression/logistic regression for TCGA; balanced logistic regression for Kucab",
        note=(
            "Best scout composite from the 2026-05-14 encoding search; SBS/DBS use compact genome-atlas "
            "centroids and ID83 uses the deterministic repeat/microhomology proxy context."
        ),
    ),
    "kernel_density_compact_payload_sbs_dbs_id_d10_v1": CompositeUGAModelSpec(
        name="kernel_density_compact_payload_sbs_dbs_id_d10_v1",
        version="v1",
        sbs_model="compact_sbs_dbs_d10",
        dbs_model="compact_sbs_dbs_d10",
        id_model="id83_payload_only_d10_dp5",
        feature_mode="separate",
        learner_family="L2 logistic regression/elastic-net regression for TCGA; balanced logistic regression for Kucab",
        feature_transform="row_normalized_rbf_kernel_density",
        transform_parameters=(
            ("sigma", "2.0"),
            ("distance", "Euclidean distance between channel UGA vectors"),
            ("row_normalization", "each source-channel kernel row sums to 1"),
            ("patient_features", "normalized channel counts multiplied by the channel kernel"),
            ("signature_features", "COSMIC reference signature columns projected by the same channel kernel before NNLS"),
        ),
        note=(
            "Best 2026-05-15 candidate after ultrafast screening and medium validation; preserves a smoothed "
            "distribution over UGA channel addresses instead of collapsing spectra to a single mean coordinate."
        ),
    ),
}


def get_uga_model(model: str | UGAModelSpec) -> UGAModelSpec:
    if isinstance(model, UGAModelSpec):
        return model
    if all(hasattr(model, attr) for attr in ("name", "kinds", "d_context", "d_payload", "payload_schema")):
        return UGAModelSpec(
            name=str(getattr(model, "name")),
            kinds=tuple(getattr(model, "kinds")),
            d_context=int(getattr(model, "d_context")),
            d_payload=int(getattr(model, "d_payload")),
            payload_schema=str(getattr(model, "payload_schema")),
            context_source=str(getattr(model, "context_source", "genome_atlas")),
            note=str(getattr(model, "note", "")),
        )
    key = str(model)
    if key not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown UGA model '{key}'. Available models: {known}")
    return MODEL_REGISTRY[key]


def list_uga_models() -> list[UGAModelSpec]:
    return [MODEL_REGISTRY[key] for key in sorted(MODEL_REGISTRY)]


def get_composite_uga_model(model: str | CompositeUGAModelSpec) -> CompositeUGAModelSpec:
    if isinstance(model, CompositeUGAModelSpec):
        return model
    key = str(model)
    if key not in COMPOSITE_MODEL_REGISTRY:
        known = ", ".join(sorted(COMPOSITE_MODEL_REGISTRY))
        raise KeyError(f"Unknown composite UGA model '{key}'. Available composites: {known}")
    return COMPOSITE_MODEL_REGISTRY[key]


def list_composite_uga_models() -> list[CompositeUGAModelSpec]:
    return [COMPOSITE_MODEL_REGISTRY[key] for key in sorted(COMPOSITE_MODEL_REGISTRY)]
