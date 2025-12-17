"""
scripts/utils/__init__.py
Utility module initialization for economic crisis prediction project.
"""

from .data_helpers import (
    load_csv,
    align_dataframes,
    handle_missing_values,
    normalize_features,
)
from .metrics import (
    compute_evaluation_metrics,
    compute_cross_validation_scores,
    generate_confusion_matrix,
)
from .config import load_config

__all__ = [
    "load_csv",
    "align_dataframes",
    "handle_missing_values",
    "normalize_features",
    "compute_evaluation_metrics",
    "compute_cross_validation_scores",
    "generate_confusion_matrix",
    "load_config",
]

__version__ = "1.0.0"
