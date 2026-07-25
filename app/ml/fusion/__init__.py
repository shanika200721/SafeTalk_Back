"""Offline research-only multimodal fusion evaluation utilities."""

from .synthetic import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    generate_synthetic_fusion_dataset,
    load_fusion_config,
    validate_synthetic_dataset,
)

__all__ = [
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "generate_synthetic_fusion_dataset",
    "load_fusion_config",
    "validate_synthetic_dataset",
]
