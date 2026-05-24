"""EXP023 TCGA-BRCA HRD paper workflow (nested CV, multi-endpoint, burden sensitivity)."""

from exp23_config import WorkflowConfig, build_config


def run_prepare(*args, **kwargs):
    from exp23_prepare import run_prepare as _run_prepare

    return _run_prepare(*args, **kwargs)


def run_fit_exposures(*args, **kwargs):
    from exp23_exposures import run_fit_exposures as _run_fit_exposures

    return _run_fit_exposures(*args, **kwargs)


def run_analysis(*args, **kwargs):
    from exp23_analyze import run_analysis as _run_analysis

    return _run_analysis(*args, **kwargs)


__all__ = [
    "WorkflowConfig",
    "build_config",
    "run_prepare",
    "run_fit_exposures",
    "run_analysis",
]
