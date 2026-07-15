"""Behavioral preprocessing foundation.

This package defines privacy-preserving schemas, validation, deterministic
session aggregation, and engineering-only synthetic fixtures. It does not train
models, fit scalers, create splits, write PostgreSQL data, or trigger alerts.
"""

from app.ml.preprocessing.behavioral.constants import (
    BEHAVIORAL_FEATURE_SCHEMA_VERSION,
    BEHAVIORAL_MAPPING_VERSION,
    BEHAVIORAL_PREPROCESSING_VERSION,
    BEHAVIORAL_SYNTHETIC_SCHEMA_VERSION,
)

__all__ = [
    "BEHAVIORAL_PREPROCESSING_VERSION",
    "BEHAVIORAL_FEATURE_SCHEMA_VERSION",
    "BEHAVIORAL_MAPPING_VERSION",
    "BEHAVIORAL_SYNTHETIC_SCHEMA_VERSION",
]

