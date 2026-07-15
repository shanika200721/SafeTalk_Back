"""Authoritative DASS-21 preprocessing and scoring package."""

from app.ml.preprocessing.dass21.constants import (
    DASS21_FEATURE_SCHEMA_VERSION,
    DASS21_SCORING_VERSION,
    DASS21_ITEM_MAPPING_VERSION,
    get_threshold_metadata,
)
from app.ml.preprocessing.dass21.scoring import (
    apply_dass21_multiplier,
    calculate_subscale_raw_scores,
    classify_anxiety_severity,
    classify_depression_severity,
    classify_stress_severity,
    convert_database_record,
    convert_frontend_payload,
    explain_dass21_result,
    normalize_subscale_score,
    score_dass21,
    validate_dass21_responses,
)

__all__ = [
    "DASS21_FEATURE_SCHEMA_VERSION",
    "DASS21_SCORING_VERSION",
    "DASS21_ITEM_MAPPING_VERSION",
    "apply_dass21_multiplier",
    "calculate_subscale_raw_scores",
    "classify_anxiety_severity",
    "classify_depression_severity",
    "classify_stress_severity",
    "convert_database_record",
    "convert_frontend_payload",
    "explain_dass21_result",
    "get_threshold_metadata",
    "normalize_subscale_score",
    "score_dass21",
    "validate_dass21_responses",
]
