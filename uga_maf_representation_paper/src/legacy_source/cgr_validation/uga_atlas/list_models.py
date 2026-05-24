#!/usr/bin/env python3
"""List registered UGA model definitions."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from uga_atlas.models import list_composite_uga_models, list_uga_models
from uga_atlas.encoding import universal_vector_dim


def main() -> None:
    print("[component_models]")
    for spec in list_uga_models():
        dim = universal_vector_dim(spec.d_context, spec.d_payload, spec.payload_schema)
        kinds = ",".join(spec.kinds)
        print(
            f"{spec.name}\t{kinds}\td_context={spec.d_context}\t"
            f"d_payload={spec.d_payload}\tpayload_schema={spec.payload_schema}\tdim={dim}"
        )
    print("[composite_models]")
    for spec in list_composite_uga_models():
        components = ",".join(f"{k}={v}" for k, v in spec.component_names().items())
        print(
            f"{spec.name}\tversion={spec.version}\tfeature_mode={spec.feature_mode}\t"
            f"learner_family={spec.learner_family}\t{components}"
        )


if __name__ == "__main__":
    main()
