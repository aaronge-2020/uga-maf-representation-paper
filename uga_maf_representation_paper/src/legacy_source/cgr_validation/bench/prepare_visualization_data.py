import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add repo to path
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from uga_atlas import (
    build_universal_ref,
    get_uga_model,
    load_context_atlas as load_atlas,
    payload_length_bit_count,
    sig_columns,
    universal_fdim,
)


def payload_labels(prefix: str, d_payload: int, payload_schema: str) -> list[str]:
    if payload_schema == "masked":
        return [
            *[f"{prefix}X_{i}" for i in range(d_payload)],
            *[f"{prefix}Y_{i}" for i in range(d_payload)],
            *[f"{prefix}Mask_{i}" for i in range(d_payload)],
        ]
    if payload_schema == "length":
        return [
            *[f"{prefix}X_{i}" for i in range(d_payload)],
            *[f"{prefix}Y_{i}" for i in range(d_payload)],
            *[f"{prefix}Len_{i}" for i in range(payload_length_bit_count(d_payload))],
        ]
    return [
        *[f"{prefix}X_{i}" for i in range(d_payload)],
        *[f"{prefix}Y_{i}" for i in range(d_payload)],
    ]

def prepare_data():
    # Paths
    COSMIC_SBS = REPO / "data/Signatures/COSMIC_v3.5_SBS_GRCh37.txt"
    COSMIC_DBS = REPO / "data/Signatures/COSMIC_v3.5_DBS_GRCh37.txt"
    ATLAS_PATH = REPO / "cgr_validation_results/research/data/EXP022_atlas_genome_wide_45mer_universal.json"
    
    model = get_uga_model("compact_sbs_dbs_d10")
    d_context = model.d_context
    d_payload = model.d_payload
    payload_schema = model.payload_schema
    fdim = universal_fdim(d_context, d_payload, payload_schema)
    
    print(f"Loading atlas from {ATLAS_PATH}...")
    atlas = load_atlas(ATLAS_PATH, d_context)
    
    print("Loading COSMIC signatures...")
    df_sbs = pd.read_csv(COSMIC_SBS, sep="\t")
    df_dbs = pd.read_csv(COSMIC_DBS, sep="\t")
    
    sbs_sigs = sig_columns(df_sbs, "SBS")
    dbs_sigs = sig_columns(df_dbs, "DBS")
    
    print(f"Encoding {len(sbs_sigs)} SBS signatures...")
    # A_univ is [fdim, n_sigs]
    A_sbs = build_universal_ref(df_sbs, atlas, d_context, sbs_sigs, "SBS", d_payload, payload_schema)
    
    print(f"Encoding {len(dbs_sigs)} DBS signatures...")
    A_dbs = build_universal_ref(df_dbs, atlas, d_context, dbs_sigs, "DBS", d_payload, payload_schema)
    
    from sklearn.decomposition import PCA
    
    def get_pca(embeddings):
        if embeddings is None or len(embeddings) == 0: return []
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        return coords.tolist()

    print("Computing PCA...")
    pca_sbs = get_pca(A_sbs.T)
    pca_dbs = get_pca(A_dbs.T)

    # Prepare JSON structure
    data = {
        "metadata": {
            "d_context": d_context,
            "d_payload": d_payload,
            "payload_schema": payload_schema,
            "uga_model": model.name,
            "fdim": fdim,
            "labels": [
                *[f"Lx_{i}" for i in range(d_context)],
                *[f"Ly_{i}" for i in range(d_context)],
                *payload_labels("Ref", d_payload, payload_schema),
                *[f"Rx_{i}" for i in range(d_context)],
                *[f"Ry_{i}" for i in range(d_context)],
                *payload_labels("Alt", d_payload, payload_schema),
            ]
        },
        "sbs": {
            "names": sbs_sigs,
            "embeddings": A_sbs.T.tolist(), # [n_sigs, fdim]
            "standard": df_sbs[sbs_sigs].fillna(0).T.values.tolist(), # [n_sigs, 96]
            "pca": pca_sbs
        },
        "dbs": {
            "names": dbs_sigs,
            "embeddings": A_dbs.T.tolist(),
            "standard": df_dbs[dbs_sigs].fillna(0).T.values.tolist(), # [n_sigs, 78]
            "pca": pca_dbs
        }
    }
    
    out_path = REPO / "bench/signature_visualizer_data.json"
    with open(out_path, "w") as f:
        json.dump(data, f)
    
    print(f"Data saved to {out_path}")

if __name__ == "__main__":
    prepare_data()
