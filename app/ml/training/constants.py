"""Shared constants for Phase 3B baseline training infrastructure."""

from __future__ import annotations


TRAINING_FRAMEWORK_VERSION = "1.0.0"
MODEL_CARD_VERSION = "1.0.0"
EVALUATION_SCHEMA_VERSION = "1.0.0"
ARTIFACT_MANIFEST_VERSION = "1.0.0"

CLINICAL_DISCLAIMER = (
    "This model is a research prototype and is not a clinical diagnostic or "
    "autonomous suicide-prevention system."
)

DEFAULT_RECORD_ID_COLUMN = "record_id"

PRODUCTION_IDENTIFIER_COLUMNS = {
    "student_id",
    "user_id",
    "email",
    "username",
    "full_name",
    "phone",
    "source_record_id",
    "hashed_password",
}

IDENTIFIER_COLUMN_TOKENS = (
    "record",
    "student",
    "user",
    "email",
    "username",
    "phone",
    "password",
    "identifier",
    "source",
)

SAFE_ARTIFACT_EXTENSIONS = {".json", ".md", ".csv", ".joblib", ".txt"}
