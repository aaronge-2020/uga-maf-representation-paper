"""Bio MAF v4: fixed, interpretable per-tumor summaries of somatic mutations."""

from .features import feature_names, featurize_maf

__version__ = "4.0.0"
__all__ = ["featurize_maf", "feature_names", "__version__"]
