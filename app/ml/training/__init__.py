"""Phase 3B common training framework.

This package provides generic, split-safe training infrastructure only. It does
not implement or activate Profile, Text, Speech, fusion, or production
inference models.
"""

from app.ml.training.constants import (
    ARTIFACT_MANIFEST_VERSION,
    EVALUATION_SCHEMA_VERSION,
    MODEL_CARD_VERSION,
    TRAINING_FRAMEWORK_VERSION,
)

__all__ = [
    "ARTIFACT_MANIFEST_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "MODEL_CARD_VERSION",
    "TRAINING_FRAMEWORK_VERSION",
]
